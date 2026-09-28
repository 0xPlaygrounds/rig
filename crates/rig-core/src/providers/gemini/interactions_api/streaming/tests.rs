use super::*;
use crate::message::AssistantContent;
use crate::streaming::{Item, StreamEvent};
use serde_json::json;

/// The request every stream test below sends. The decoder is what they
/// exercise, so the request only has to be well-formed.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
fn interactions_request() -> crate::completion::CompletionRequest {
    crate::completion::CompletionRequest {
        model: None,
        chat_history: crate::NonEmpty::new(crate::message::Message::user("hello")),
        documents: Vec::new(),
        tools: Vec::new(),
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    }
}

/// The Interactions wire bound to a transport answering with `frames` as
/// one SSE body.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
fn interactions_stream(
    frames: &[&str],
) -> crate::driver::Model<
    crate::providers::gemini::interactions_api::Interactions,
    crate::test_utils::MockStreamingClient,
> {
    let sse_bytes = bytes::Bytes::from(
        frames
            .iter()
            .map(|event| format!("data: {event}\n\n"))
            .collect::<String>(),
    );
    crate::driver::Model::new(
        crate::providers::gemini::GeminiConfig::new("test-key").interactions("gemini-2.5-pro"),
        crate::test_utils::MockStreamingClient { sse_bytes },
    )
}

#[test]
fn test_streaming_completion_response_has_model_version() {
    let response = StreamingCompletionResponse {
        usage: None,
        interaction: None,
        model_version: Some("gemini-2.5-pro-preview-05-06".to_string()),
    };

    assert_eq!(
        response.model_version.as_deref(),
        Some("gemini-2.5-pro-preview-05-06")
    );

    let json = serde_json::to_string(&response).unwrap();
    let deserialized: StreamingCompletionResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(
        deserialized.model_version.as_deref(),
        Some("gemini-2.5-pro-preview-05-06")
    );
}

#[test]
fn test_content_delta_text_event() {
    let event_json = json!({
        "event_type": "step.delta",
        "index": 0,
        "delta": {
            "type": "text",
            "text": "Hello"
        }
    });

    let event: InteractionSseEvent = serde_json::from_value(event_json).unwrap();
    let InteractionSseEvent::StepDelta { delta, .. } = event else {
        panic!("expected step delta");
    };
    assert!(matches!(
        delta_content(delta),
        Some(Content::Text(TextContent { text, .. })) if text == "Hello"
    ));
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[tokio::test]
async fn truncated_stream_does_not_synthesize_an_end() {
    // Content deltas then EOF without `interaction.completed`: the
    // truncated stream delivers its content, then its truncation.
    let (items, outcome) = drive_frames(&[
        r#"{"event_type":"step.delta","index":0,"delta":{"type":"text","text":"hi"}}"#,
    ])
    .await;
    assert_eq!(texts_of(&items), ["hi"]);
    assert!(
        matches!(outcome, Err(crate::error::ProviderError::Truncated)),
        "EOF without interaction.completed is truncation: {outcome:?}"
    );
}

/// Drive Interactions SSE frames through the full normalized path and
/// collect what the consumer sees, in order, and what the reply finished
/// with.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
async fn drive_frames(
    frames: &[&str],
) -> (
    Vec<Result<Item<StreamEvent>, String>>,
    Result<crate::completion::CompletionResponse, crate::error::ProviderError>,
) {
    use futures::StreamExt;

    let model = interactions_stream(frames);
    let mut stream = model
        .stream(interactions_request())
        .expect("stream should open");

    let mut items = Vec::new();
    while let Some(item) = stream.next().await {
        items.push(item.map_err(|error| error.to_string()));
    }
    (items, stream.finish().await)
}

/// The text fragments among `items`.
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

/// What the parts among `items` ended with.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
fn ended(items: &[Result<Item<StreamEvent>, String>]) -> Vec<&AssistantContent> {
    items
        .iter()
        .filter_map(|item| match item {
            Ok(Item::Event(StreamEvent::End { content, .. })) => Some(content),
            _ => None,
        })
        .collect()
}

/// The calls among `items`.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
fn calls_of(items: &[Result<Item<StreamEvent>, String>]) -> Vec<&crate::message::ToolCall> {
    ended(items)
        .into_iter()
        .filter_map(|content| match content {
            AssistantContent::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect()
}

/// A `model_output` step interleaving text and a function call in one
/// step's `content`: every convertible item must surface, in wire
/// order. `find_map` kept only the first — a `function_call` following
/// text in the same step silently vanished.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[tokio::test]
async fn a_model_output_step_yields_every_convertible_item() {
    let (items, _) = drive_frames(&[
        r#"{"event_type":"step.start","index":0,"step":{"type":"model_output","content":[{"type":"text","text":"answer: "},{"type":"function_call","name":"add","arguments":{"x":1},"id":"fc_9"}]}}"#,
        r#"{"event_type":"interaction.completed","interaction":{"id":"int_1","status":"completed"}}"#,
    ])
    .await;

    let calls = calls_of(&items);
    assert_eq!(
        texts_of(&items),
        ["answer: "],
        "the text survives, got {items:?}"
    );
    assert_eq!(
        calls.len(),
        1,
        "the function_call after text must also survive, got {items:?}"
    );
    let call = calls.first().expect("one call");
    assert_eq!(call.function.name, "add");
    assert_eq!(call.function.arguments, serde_json::json!({"x": 1}));
}

/// A `step.start` that announces non-empty arguments AND fragments the
/// real payload across `arguments_delta` events: the deltas are the
/// arguments. Concatenating the announce payload with the fragments
/// yields `{..}{..}` — unparseable under the step's Error policy, so
/// the call was lost outright.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[tokio::test]
async fn announce_arguments_never_concatenate_with_fragments() {
    let (items, _) = drive_frames(&[
        r#"{"event_type":"step.start","index":1,"step":{"arguments":{"x":1},"id":"fc_1","name":"add","type":"function_call"}}"#,
        r#"{"delta":{"arguments":"{\"x\":1}","type":"arguments_delta"},"event_type":"step.delta","index":1}"#,
        r#"{"event_type":"step.stop","index":1}"#,
        r#"{"event_type":"interaction.completed","interaction":{"id":"int_1","status":"completed"}}"#,
    ])
    .await;

    let tool_calls = calls_of(&items);
    assert_eq!(
        tool_calls.len(),
        1,
        "the announced-then-fragmented call must survive, got {items:?}"
    );
    assert_eq!(
        tool_calls.first().expect("one call").function.arguments,
        serde_json::json!({"x": 1}),
        "streamed fragments are the arguments; the announce payload is not prepended"
    );
}

/// A partial announce with NO fragments: the announce payload is the
/// only arguments the wire sent, so it finalizes the call
/// (replace-if-no-deltas).
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[tokio::test]
async fn announce_arguments_finalize_a_call_with_no_fragments() {
    let (items, _) = drive_frames(&[
        r#"{"event_type":"step.start","index":1,"step":{"arguments":{"x":7},"id":"fc_1","name":"add","type":"function_call"}}"#,
        r#"{"event_type":"step.stop","index":1}"#,
        r#"{"event_type":"interaction.completed","interaction":{"id":"int_1","status":"completed"}}"#,
    ])
    .await;

    let tool_calls = calls_of(&items);
    assert_eq!(tool_calls.len(), 1, "got {items:?}");
    assert_eq!(
        tool_calls.first().expect("one call").function.arguments,
        serde_json::json!({"x": 7})
    );
}

/// Interactions is a single-identifier wire: its `fc_…` id must land
/// in `provider.call_id` with `item_id` empty. Filling both slots
/// fabricated a Responses-shaped dual identity whose fake item id
/// passed the foreign-id guard on cross-provider replay.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[tokio::test]
async fn a_streamed_call_carries_a_single_wire_identity() {
    let (items, _) = drive_frames(&[
        r#"{"event_type":"step.start","index":1,"step":{"arguments":{},"id":"fc_1","name":"add","type":"function_call"}}"#,
        r#"{"delta":{"arguments":"{\"x\":1}","type":"arguments_delta"},"event_type":"step.delta","index":1}"#,
        r#"{"event_type":"step.stop","index":1}"#,
        r#"{"event_type":"interaction.completed","interaction":{"id":"int_1","status":"completed"}}"#,
    ])
    .await;

    let tool_calls = calls_of(&items);
    let provider = tool_calls
        .first()
        .expect("one call")
        .id
        .provider()
        .expect("the wire issued an id");
    assert_eq!(provider.call_id, "fc_1");
    assert_eq!(
        provider.item_id, None,
        "a single-identifier wire must not fabricate a dual identity"
    );
}

/// A `step.stop` that never arrives must not lose the call: the wire
/// announced it (`step.start`), streamed its full arguments
/// (`arguments_delta`), and proved the turn finished
/// (`interaction.completed`). Before this fix the assembly stayed open,
/// `finish` never ran (terminal return), and the accumulator's
/// end-of-stream clear dropped the whole call — the agent then treated
/// a tool-calling turn as plain text.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[tokio::test]
async fn a_missing_step_stop_does_not_lose_the_announced_call() {
    let (items, outcome) = drive_frames(&[
        r#"{"event_type":"step.start","index":1,"step":{"arguments":{},"id":"fc_1","name":"get_weather","type":"function_call"}}"#,
        r#"{"delta":{"arguments":"{\"city\":\"Paris\"}","type":"arguments_delta"},"event_type":"step.delta","index":1}"#,
        r#"{"event_type":"interaction.completed","interaction":{"id":"int_1","status":"completed"}}"#,
    ])
    .await;

    let tool_calls = calls_of(&items);
    assert_eq!(
        tool_calls.len(),
        1,
        "the announced call must survive the missing step.stop, got {items:?}"
    );
    let tool_call = tool_calls.first().expect("one call");
    assert_eq!(tool_call.function.name, "get_weather");
    assert_eq!(
        tool_call.function.arguments,
        serde_json::json!({"city": "Paris"}),
        "the streamed argument fragments finalize the call"
    );
    assert_eq!(
        tool_call
            .id
            .provider()
            .map(|provider| provider.call_id.as_str()),
        Some("fc_1")
    );

    // The turn completed normally.
    let aggregated_calls = outcome
        .expect("the reply ended")
        .choice
        .iter()
        .filter(|content| matches!(content, crate::message::AssistantContent::ToolCall(_)))
        .count();
    assert_eq!(
        aggregated_calls, 1,
        "the call reaches the aggregated choice"
    );
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[tokio::test]
async fn provider_error_event_ends_the_stream_without_draining_later_frames() {
    // A provider `error` event, then more frames: well-formed content, an
    // unknown frame, and a terminal `interaction.completed`. The error
    // must be the LAST item — the driver stops reading (`is_finished`),
    // so nothing after it is interpreted or passed through as `Unknown`.
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
        .position(|item| item.is_err())
        .expect("the provider error must reach the consumer");
    assert_eq!(
        error_position,
        items.len() - 1,
        "the in-band error must end the stream: no later text, Unknown passthrough, or terminal; got {items:?}"
    );
    assert_eq!(
        texts_of(&items),
        ["hi"],
        "content before the error must survive"
    );
    assert!(outcome.is_err());
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[tokio::test]
async fn thought_signature_completes_the_accumulated_reasoning_block() {
    // Text-then-signature: the signed part carries the full accumulated
    // thought text and the signature, beside the later text.
    let (items, _) = drive_frames(&[
        r#"{"event_type":"step.delta","index":0,"delta":{"type":"thought_summary","content":{"type":"text","text":"think1 "}}}"#,
        r#"{"event_type":"step.delta","index":0,"delta":{"type":"thought_summary","content":{"type":"text","text":"think2"}}}"#,
        r#"{"event_type":"step.delta","index":0,"delta":{"type":"thought_signature","signature":"sig-abc"}}"#,
        r#"{"event_type":"step.delta","index":1,"delta":{"type":"text","text":"answer"}}"#,
    ])
    .await;

    let signed = ended(&items)
        .into_iter()
        .find_map(|content| match content {
            AssistantContent::Reasoning(reasoning) => Some(reasoning.clone()),
            _ => None,
        })
        .expect("the signature must yield a completed Reasoning block");
    assert_eq!(
        signed.value().content,
        vec![crate::completion::message::ReasoningContent::Text {
            text: "think1 think2".to_string(),
            signature: Some("sig-abc".to_string()),
        }],
        "the signed block must restate the accumulated text with the signature"
    );

    // Exactly one reasoning part ended: the signature closed the one the
    // fragments opened.
    let reasoning = ended(&items)
        .into_iter()
        .filter(|content| matches!(content, AssistantContent::Reasoning(_)))
        .count();
    assert_eq!(reasoning, 1, "got {items:?}");
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[tokio::test]
async fn signature_only_thought_still_carries_the_signature() {
    // Signature with no preceding thought-summary text: the signature is
    // the provider's replay-validated payload and must still survive as a
    // signed (empty-text) Reasoning block.
    let (items, _) = drive_frames(&[
        r#"{"event_type":"step.delta","index":0,"delta":{"type":"thought_signature","signature":"sig-only"}}"#,
        r#"{"event_type":"step.delta","index":1,"delta":{"type":"text","text":"answer"}}"#,
    ])
    .await;

    let signed = ended(&items)
        .into_iter()
        .find_map(|content| match content {
            AssistantContent::Reasoning(reasoning) => Some(reasoning.clone()),
            _ => None,
        })
        .expect("a signature-only block must still yield a signed Reasoning");
    assert_eq!(
        signed.value().content,
        vec![crate::completion::message::ReasoningContent::Text {
            text: String::new(),
            signature: Some("sig-only".to_string()),
        }]
    );
}

#[test]
fn test_content_delta_function_call_event() {
    let event_json = json!({
        "event_type": "step.delta",
        "index": 0,
        "delta": {
            "type": "function_call",
            "name": "get_weather",
            "arguments": {"location": "Paris"},
            "id": "call-1"
        }
    });

    let event: InteractionSseEvent = serde_json::from_value(event_json).unwrap();
    let InteractionSseEvent::StepDelta { delta, .. } = event else {
        panic!("expected step delta");
    };

    let Some(Content::FunctionCall(call)) = delta_content(delta) else {
        panic!("a function call delta is a whole call");
    };
    assert_eq!(call.name.as_deref(), Some("get_weather"));
    assert_eq!(call.id.as_deref(), Some("call-1"));
    assert_eq!(call.arguments, Some(json!({"location": "Paris"})));
}
