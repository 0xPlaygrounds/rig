use super::*;
use crate::message::{AssistantContent, Message};
use crate::streaming::{Item, StreamEvent};
use crate::test_utils::history::{assert_every_variant, assert_restated_agrees, decode};
use crate::wire::Mode;
use serde_json::json;

use super::super::{Interactions, create_request_body};

fn wire() -> Interactions {
    crate::providers::gemini::GeminiConfig::new("test-key").interactions("gemini-2.5-pro")
}

fn frame(value: serde_json::Value) -> WireFrame {
    WireFrame::Text(value.to_string())
}

/// A whole interaction resource holding `steps`.
fn whole(steps: Vec<serde_json::Value>) -> WireFrame {
    frame(json!({"id": "int_1", "status": "completed", "steps": steps}))
}

fn start(index: usize, step: serde_json::Value) -> WireFrame {
    frame(json!({"event_type": "step.start", "index": index, "step": step}))
}

fn delta(index: usize, delta: serde_json::Value) -> WireFrame {
    frame(json!({"event_type": "step.delta", "index": index, "delta": delta}))
}

fn stop(index: usize) -> WireFrame {
    frame(json!({"event_type": "step.stop", "index": index}))
}

fn completed() -> WireFrame {
    frame(json!({
        "event_type": "interaction.completed",
        "interaction": {"id": "int_1", "status": "completed"},
    }))
}

/// The turn `frames` decode to as a stream.
fn streamed(frames: Vec<WireFrame>) -> crate::completion::CompletionResponse {
    decode(&wire(), Mode::Streaming, frames).expect("the stream decodes")
}

/// The steps the decoded turn of `response` encodes to for the same model:
/// what follows the user's prompt, without the results the request
/// boundary adds for its calls.
fn replayed(response: &crate::completion::CompletionResponse) -> Vec<serde_json::Value> {
    let history = vec![Message::user("hello"), response.message().expect("a turn")];
    let request = <Completion as crate::wire::Operation>::prepare(
        crate::completion::CompletionRequest::from(history),
        &crate::wire::Wire::describe(&wire()),
    )
    .expect("the request is valid");
    let mut steps = create_request_body("gemini-2.5-pro".to_owned(), request, None)
        .expect("the request builds")
        .input;
    steps.remove(0);
    steps.retain(|step| step["type"] != "function_result");
    steps
}

#[test]
fn test_streaming_completion_response_has_model_version() {
    let response = StreamingCompletionResponse {
        usage: None,
        interaction: None,
        model_version: Some("gemini-2.5-pro-preview-05-06".to_string()),
    };
    let json = serde_json::to_string(&response).expect("serializes");
    let deserialized: StreamingCompletionResponse =
        serde_json::from_str(&json).expect("deserializes");
    assert_eq!(
        deserialized.model_version.as_deref(),
        Some("gemini-2.5-pro-preview-05-06")
    );
}

/// The Interactions wire bound to a transport answering with `frames` as
/// one SSE body, driven through the full normalized path.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
async fn drive_frames(
    frames: &[&str],
) -> (
    Vec<Result<Item<StreamEvent>, String>>,
    Result<crate::completion::CompletionResponse, ProviderError>,
) {
    use futures::StreamExt;

    let sse_bytes = bytes::Bytes::from(
        frames
            .iter()
            .map(|event| format!("data: {event}\n\n"))
            .collect::<String>(),
    );
    let model =
        crate::driver::Model::new(wire(), crate::test_utils::MockStreamingClient { sse_bytes });
    let mut stream = model
        .stream(crate::completion::CompletionRequest::new("hello"))
        .expect("stream should open");
    let mut items = Vec::new();
    while let Some(item) = stream.next().await {
        items.push(item.map_err(|error| error.to_string()));
    }
    (items, stream.finish().await)
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
fn texts_of(items: &[Result<Item<StreamEvent>, String>]) -> Vec<&str> {
    items
        .iter()
        .filter_map(|item| match item {
            Ok(Item::Event(StreamEvent::Text { text, .. })) => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[tokio::test]
async fn truncated_stream_does_not_synthesize_an_end() {
    let (items, outcome) = drive_frames(&[
        r#"{"event_type":"step.delta","index":0,"delta":{"type":"text","text":"hi"}}"#,
    ])
    .await;
    assert_eq!(texts_of(&items), ["hi"]);
    assert!(
        matches!(outcome, Err(ProviderError::Truncated)),
        "EOF without interaction.completed is truncation: {outcome:?}"
    );
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[tokio::test]
async fn provider_error_event_ends_the_stream_without_draining_later_frames() {
    let (items, outcome) = drive_frames(&[
        r#"{"event_type":"step.delta","index":0,"delta":{"type":"text","text":"hi"}}"#,
        r#"{"event_type":"error","error":{"code":"internal","message":"boom"}}"#,
        r#"{"event_type":"step.delta","index":0,"delta":{"type":"text","text":"dead"}}"#,
        r#"{"event_type":"something.future","payload":{"x":1}}"#,
        r#"{"event_type":"interaction.completed","interaction":{"id":"int_1","status":"completed"}}"#,
    ])
    .await;
    let error_position = items
        .iter()
        .position(Result::is_err)
        .expect("the provider error must reach the consumer");
    assert_eq!(error_position, items.len() - 1, "got {items:?}");
    assert_eq!(texts_of(&items), ["hi"]);
    assert!(outcome.is_err());
}

/// Streamed fragments are the arguments: the arguments the start
/// announced are not prepended to them.
#[test]
fn announced_arguments_never_concatenate_with_fragments() {
    let call = json!({"arguments": {"x": 1}, "id": "fc_1", "name": "add", "type": "function_call"});
    let response = streamed(vec![
        start(1, call),
        delta(
            1,
            json!({"type": "arguments_delta", "arguments": "{\"x\":2}"}),
        ),
        stop(1),
        completed(),
    ]);
    let calls: Vec<_> = response
        .message()
        .into_iter()
        .flat_map(|message| match message {
            Message::Assistant(turn) => turn.tool_calls().cloned().collect::<Vec<_>>(),
            _ => Vec::new(),
        })
        .collect();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.arguments_value(), json!({"x": 2}));
    assert_eq!(calls[0].id.provider().map(|id| id.as_str()), Some("fc_1"));
    assert_eq!(
        replayed(&response),
        [json!({"arguments": {"x": 2}, "id": "fc_1", "name": "add", "type": "function_call"})],
        "the replayed step carries the streamed arguments"
    );
}

/// With no fragments, the announced arguments are the call's.
#[test]
fn announced_arguments_finalize_a_call_with_no_fragments() {
    let call = json!({"arguments": {"x": 7}, "id": "fc_1", "name": "add", "type": "function_call"});
    let response = streamed(vec![start(1, call.clone()), stop(1), completed()]);
    let Some(AssistantContent::ToolCall(tool_call)) = response.choice.first() else {
        panic!("one call: {:?}", response.choice);
    };
    assert_eq!(tool_call.function.arguments_value(), json!({"x": 7}));
    assert_eq!(replayed(&response), [call]);
}

/// A call whose `step.stop` never came is kept, with the arguments it
/// streamed, when the interaction completes.
#[test]
fn a_missing_step_stop_does_not_lose_the_call() {
    let response = streamed(vec![
        start(
            1,
            json!({"arguments": {}, "id": "fc_1", "name": "get_weather", "type": "function_call"}),
        ),
        delta(
            1,
            json!({"type": "arguments_delta", "arguments": "{\"city\":\"Paris\"}"}),
        ),
        completed(),
    ]);
    let Some(AssistantContent::ToolCall(tool_call)) = response.choice.first() else {
        panic!("one call: {:?}", response.choice);
    };
    assert_eq!(
        tool_call.function.arguments_value(),
        json!({"city": "Paris"})
    );
}

/// A resumed stream can join a step after its start: the deltas open the
/// step they imply, and the thought keeps its text and signature.
#[test]
fn deltas_without_a_start_open_the_step_they_imply() {
    let response = streamed(vec![
        delta(
            0,
            json!({"type": "thought_summary", "content": {"type": "text", "text": "think1 "}}),
        ),
        delta(
            0,
            json!({"type": "thought_summary", "content": {"type": "text", "text": "think2"}}),
        ),
        delta(
            0,
            json!({"type": "thought_signature", "signature": "sig-abc"}),
        ),
        stop(0),
        delta(1, json!({"type": "text", "text": "answer"})),
        completed(),
    ]);
    let [
        AssistantContent::Reasoning(reasoning),
        AssistantContent::Text(text),
    ] = response.choice.as_slice()
    else {
        panic!("a thought, then the answer: {:?}", response.choice);
    };
    assert_eq!(reasoning.text, "think1 think2");
    assert_eq!(text.text, "answer");
    assert_eq!(
        replayed(&response),
        [
            json!({
                "type": "thought",
                "summary": [
                    {"type": "text", "text": "think1 "},
                    {"type": "text", "text": "think2"},
                ],
                "signature": "sig-abc",
            }),
            json!({"type": "model_output", "content": [{"type": "text", "text": "answer"}]}),
        ]
    );
}

/// A thought with only a signature is kept: the signature is what the
/// model needs back.
#[test]
fn a_signature_only_thought_replays_its_signature() {
    let response = streamed(vec![
        start(0, json!({"type": "thought"})),
        delta(
            0,
            json!({"type": "thought_signature", "signature": "sig-only"}),
        ),
        stop(0),
        completed(),
    ]);
    assert!(matches!(
        response.choice.as_slice(),
        [AssistantContent::Reasoning(reasoning)] if reasoning.text.is_empty()
    ));
    assert_eq!(
        replayed(&response),
        [json!({"type": "thought", "signature": "sig-only"})]
    );
}

/// A whole thought, a model output with annotated text, and a hosted
/// search: the unary resource and its restatement as a stream fold into
/// the same turn.
#[test]
fn a_whole_interaction_and_its_stream_agree() {
    let thought = json!({"type": "thought", "signature": "sig", "summary": [{"type": "text", "text": "plan"}]});
    let search = json!({"type": "google_search_call", "id": "call_1", "signature": "s", "arguments": {"queries": ["euro 2024"]}, "search_type": "web_search"});
    let output = json!({"type": "model_output", "content": [{"type": "text", "text": "Spain won.", "annotations": [{"start_index": 0, "end_index": 5, "source": "https://example.com"}]}]});
    assert_restated_agrees(
        &wire(),
        [whole(vec![thought, search.clone(), output])],
        [
            start(0, json!({"type": "thought"})),
            delta(
                0,
                json!({"type": "thought_summary", "content": {"type": "text", "text": "plan"}}),
            ),
            delta(0, json!({"type": "thought_signature", "signature": "sig"})),
            stop(0),
            start(
                1,
                json!({"type": "google_search_call", "id": "call_1", "signature": ""}),
            ),
            delta(1, search),
            stop(1),
            start(2, json!({"type": "model_output"})),
            delta(
                2,
                json!({"type": "text", "text": "Spain", "annotations": [{"start_index": 0, "end_index": 5, "source": "https://example.com"}]}),
            ),
            delta(2, json!({"type": "text", "text": " won."})),
            stop(2),
            completed(),
        ],
    );
}

/// A thought `step.start` that already states its summary writes it, as a
/// whole reply does.
#[test]
fn a_thought_start_with_its_summary_agrees_with_the_whole() {
    let thought = json!({"type": "thought", "signature": "sig", "summary": [{"type": "text", "text": "plan"}]});
    assert_restated_agrees(
        &wire(),
        [whole(vec![thought.clone()])],
        [start(0, thought), stop(0), completed()],
    );
}

/// A model output that is one image is an image block, streamed or whole,
/// and replays verbatim.
#[test]
fn an_image_output_is_an_image_block_in_both_modes() {
    let image = json!({"type": "image", "data": "aW1n", "mime_type": "image/png"});
    let output = json!({"type": "model_output", "content": [image.clone()]});
    let response =
        decode(&wire(), Mode::Unary, [whole(vec![output.clone()])]).expect("the resource decodes");
    let [AssistantContent::Image(decoded)] = response.choice.as_slice() else {
        panic!("one image: {:?}", response.choice);
    };
    assert_eq!(
        decoded.data,
        crate::message::DocumentSourceKind::Base64("aW1n".to_owned())
    );
    assert_eq!(decoded.media_type, Some(ImageMediaType::PNG));
    assert_eq!(replayed(&response), [output]);
    assert_restated_agrees(
        &wire(),
        [whole(vec![
            json!({"type": "model_output", "content": [image.clone()]}),
        ])],
        [
            start(0, json!({"type": "model_output"})),
            delta(0, image),
            stop(0),
            completed(),
        ],
    );
}

/// A step type rig has never seen, and a field rig has never seen on a
/// known step, survive decoding in both modes and go back to the same
/// model verbatim.
#[test]
fn an_invented_step_and_field_survive_both_modes_and_replay() {
    let invented = json!({"type": "hologram", "id": "h_1", "frames": [1, 2]});
    let output =
        json!({"type": "model_output", "tone": "dry", "content": [{"type": "text", "text": "hi"}]});
    let whole_frames = [whole(vec![invented.clone(), output.clone()])];
    let stream_frames = vec![
        start(0, json!({"type": "hologram", "id": "h_1"})),
        delta(0, json!({"type": "hologram", "frames": [1, 2]})),
        stop(0),
        start(1, json!({"type": "model_output", "tone": "dry"})),
        delta(1, json!({"type": "text", "text": "hi"})),
        stop(1),
        completed(),
    ];
    assert_restated_agrees(&wire(), whole_frames.clone(), stream_frames.clone());
    for response in [
        decode(&wire(), Mode::Unary, whole_frames).expect("the resource decodes"),
        streamed(stream_frames),
    ] {
        assert!(matches!(
            response.choice.first(),
            Some(AssistantContent::Opaque(opaque)) if opaque.replay
        ));
        assert_eq!(replayed(&response), [invented.clone(), output.clone()]);
    }
}

/// A call whose name is not a string has no name: it is dropped, and the
/// reply still decodes.
#[test]
fn a_call_without_a_string_name_is_dropped() {
    let response = decode(
        &wire(),
        Mode::Unary,
        [whole(vec![
            json!({"type": "function_call", "name": 5}),
            json!({"type": "model_output", "content": [{"type": "text", "text": "hi"}]}),
        ])],
    )
    .expect("the resource decodes");
    assert!(
        matches!(response.choice.as_slice(), [AssistantContent::Text(text)] if text.text == "hi"),
        "{:?}",
        response.choice
    );
}

/// Every step kind decodes, whole and streamed alike: thoughts, calls and
/// output to their canonical blocks, input steps to opaque steps that are
/// never sent back, and hosted-tool steps and invented ones to opaque
/// steps that are.
#[test]
fn every_step_kind_decodes() {
    #[deny(clippy::wildcard_enum_match_arm)]
    fn variant_index(kind: StepKind) -> usize {
        match kind {
            StepKind::Thought => 0,
            StepKind::Output => 1,
            StepKind::Call => 2,
            StepKind::Input => 3,
            StepKind::Other => 4,
        }
    }
    let samples = [
        json!({"type": "thought", "signature": "sig"}),
        json!({"type": "model_output", "content": [{"type": "text", "text": "hello"}]}),
        json!({"type": "function_call", "id": "call_1", "name": "add", "arguments": {"x": 1}}),
        json!({"type": "user_input", "content": [{"type": "text", "text": "hi"}]}),
        json!({"type": "function_result", "call_id": "call_1", "name": "add", "result": 1}),
        json!({"type": "code_execution_call", "id": "c"}),
        json!({"type": "code_execution_result", "call_id": "c", "result": "2"}),
        json!({"type": "url_context_call", "id": "u"}),
        json!({"type": "url_context_result", "call_id": "u"}),
        json!({"type": "google_search_call", "id": "g"}),
        json!({"type": "google_search_result", "call_id": "g"}),
        json!({"type": "mcp_server_tool_call", "id": "m", "name": "lookup"}),
        json!({"type": "mcp_server_tool_result", "call_id": "m", "name": "lookup"}),
        json!({"type": "file_search_result"}),
        json!({"type": "x_rig_invented", "id": "x"}),
    ];
    assert_every_variant(&samples, |step| variant_index(step_kind(step)), 5);
    for step in &samples {
        assert_restated_agrees(
            &wire(),
            [whole(vec![step.clone()])],
            [start(0, step.clone()), stop(0), completed()],
        );
        let response =
            decode(&wire(), Mode::Unary, [whole(vec![step.clone()])]).expect("the sample decodes");
        let block = response.choice.first().cloned();
        let expected = match step_kind(step) {
            StepKind::Thought => matches!(&block, Some(AssistantContent::Reasoning(_))),
            StepKind::Output => {
                matches!(&block, Some(AssistantContent::Text(text)) if text.text == "hello")
            }
            StepKind::Call => matches!(&block, Some(AssistantContent::ToolCall(_))),
            StepKind::Input => {
                matches!(&block, Some(AssistantContent::Opaque(opaque)) if !opaque.replay)
            }
            StepKind::Other => {
                matches!(&block, Some(AssistantContent::Opaque(opaque)) if opaque.replay && opaque.item == *step)
            }
        };
        assert!(expected, "{step} decoded to {block:?}");
    }
}

/// Every model output content kind decodes to its block: text to text, an
/// image with a source to an image, and audio, video, documents, a
/// sourceless image and invented items to opaque blocks that replay.
#[test]
fn every_content_kind_decodes() {
    #[deny(clippy::wildcard_enum_match_arm)]
    fn variant_index(kind: ContentKind) -> usize {
        match kind {
            ContentKind::Text => 0,
            ContentKind::Image => 1,
            ContentKind::Other => 2,
        }
    }
    let samples = [
        json!({"type": "text", "text": "hello"}),
        json!({"type": "image", "uri": "https://example.com/a.png", "mime_type": "image/png"}),
        json!({"type": "image", "mime_type": "image/png"}),
        json!({"type": "audio", "data": "YQ==", "mime_type": "audio/wav"}),
        json!({"type": "video", "uri": "https://example.com/a.mp4"}),
        json!({"type": "document", "data": "YQ==", "mime_type": "application/pdf"}),
        json!({"type": "x_rig_invented"}),
    ];
    assert_every_variant(&samples, |item| variant_index(content_kind(item)), 3);
    for item in &samples {
        let step = json!({"type": "model_output", "content": [item]});
        let response =
            decode(&wire(), Mode::Unary, [whole(vec![step.clone()])]).expect("the sample decodes");
        let block = response.choice.first();
        let expected = match (content_kind(item), item.get("uri").or(item.get("data"))) {
            (ContentKind::Text, _) => matches!(block, Some(AssistantContent::Text(_))),
            (ContentKind::Image, Some(_)) => matches!(block, Some(AssistantContent::Image(_))),
            (ContentKind::Image, None) | (ContentKind::Other, _) => {
                matches!(block, Some(AssistantContent::Opaque(opaque)) if opaque.replay && opaque.item == step)
            }
        };
        assert!(expected, "{item} decoded to {block:?}");
        assert_eq!(replayed(&response), [step]);
    }
}

/// An unstored interaction (`store: false`) streams its status update
/// without an `interaction_id`; it is still that event, not a whole
/// interaction that ends the reply. Frame as recorded in
/// `gemini/interactions_api/code_execution_usage_streamed.yaml`.
#[test]
fn a_status_update_without_an_interaction_id_is_a_status_update() {
    let event = classify_interactions_frame(
        r#"{"event_type":"interaction.status_update","status":"in_progress"}"#,
    );
    assert!(matches!(
        event,
        WireEvent::Known(InteractionsEvent::Sse(event))
            if event.event_type == "interaction.status_update"
    ));
}

/// A tagged frame is an event, never a whole interaction, even with a
/// `status` key; a step event without its index fails the reply.
#[test]
fn a_tagged_frame_never_passes_for_a_whole_interaction() {
    let event = classify_interactions_frame(r#"{"event_type":"step.stop","status":"completed"}"#);
    assert!(matches!(event, WireEvent::Known(InteractionsEvent::Sse(_))));
    let outcome = decode(
        &wire(),
        Mode::Streaming,
        [frame(
            json!({"event_type": "step.stop", "status": "completed"}),
        )],
    );
    assert!(outcome.is_err(), "{outcome:?}");
}

/// The unary reply, an untagged interaction resource, decodes whole.
#[test]
fn an_untagged_interaction_resource_decodes_whole() {
    let event = classify_interactions_frame(
        r#"{"id":"v1_1","object":"interaction","status":"completed","steps":[]}"#,
    );
    assert!(matches!(
        event,
        WireEvent::Known(InteractionsEvent::Whole(_))
    ));
}
