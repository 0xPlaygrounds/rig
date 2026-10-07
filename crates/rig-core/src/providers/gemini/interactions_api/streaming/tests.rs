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

/// The steps the decoded turn of `response` encodes to for the same model,
/// with the tools its calls name declared: what follows the user's prompt,
/// without the results the request boundary adds for its calls.
fn replayed(response: &crate::completion::CompletionResponse) -> Vec<serde_json::Value> {
    let turn = response.message().expect("a turn");
    let tools = match &turn {
        Message::Assistant(turn) => turn
            .tool_calls()
            .map(|call| crate::completion::ToolDefinition {
                name: call.function.name.clone(),
                description: "A tool.".to_owned(),
                parameters: json!({"type": "object", "properties": {}}),
            })
            .collect(),
        _ => Vec::new(),
    };
    let mut request =
        crate::completion::CompletionRequest::from(vec![Message::user("hello"), turn]);
    request.tools = tools;
    let request = <Completion as crate::wire::Operation>::prepare(
        request,
        &crate::wire::Wire::describe(&wire()),
    )
    .expect("the request is valid");
    let body = create_request_body(&wire(), &request, None).expect("the request builds");
    let mut steps = body
        .get("input")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    steps.remove(0);
    steps.retain(|step| step["type"] != "function_result");
    steps
}

/// How a model output content item decodes, by its `type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ContentKind {
    Text,
    /// An image block, or an opaque one when it states no data or URI.
    Image,
    /// Audio, video, a document, or a type rig does not know.
    Other,
}

fn content_kind(content: &serde_json::Value) -> ContentKind {
    match content.get("type").and_then(serde_json::Value::as_str) {
        Some("text") => ContentKind::Text,
        Some("image") => ContentKind::Image,
        _ => ContentKind::Other,
    }
}

/// How the decoder classifies the frame `data`.
fn classify_interactions_frame(data: &str) -> WireEvent<InteractionsEvent> {
    InteractionsDecoder::default().classify(WireFrame::Text(data.to_owned()))
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

/// A text item's annotations cite bytes of its text, past a multi-byte
/// dash: a URL, a file page and a place. Annotations streamed after the
/// text as `text_annotation_delta` join the text item, so the stream cites
/// and replays what the whole interaction does. Other annotations cite
/// nothing.
#[test]
fn annotations_cite_their_text_item_in_both_modes() {
    let text = "Spain won 2\u{2013}1 in Berlin.";
    let at = |part: &str| text.find(part).expect("the text holds it");
    let span = |part: &str| json!({"start_index": at(part), "end_index": at(part) + part.len()});
    let annotation = |part: &str, fields: serde_json::Value| {
        let mut annotation = span(part);
        if let (Some(annotation), serde_json::Value::Object(fields)) =
            (annotation.as_object_mut(), fields)
        {
            annotation.extend(fields);
        }
        annotation
    };
    let annotations = json!([
        annotation(
            "in Berlin.",
            json!({"type": "url_citation", "url": "https://example.com", "title": "example.com"})
        ),
        annotation(
            "Spain won",
            json!({"type": "file_citation", "document_uri": "files/report", "file_name": "report.pdf", "page_number": 3})
        ),
        annotation(
            "Berlin",
            json!({"type": "place_citation", "place_id": "places/berlin", "name": "Berlin"})
        ),
        annotation("Spain", json!({"type": "word_info"})),
    ]);
    let output = json!({"type": "model_output", "content": [{"type": "text", "text": text, "annotations": annotations}]});
    let streamed = [
        start(0, json!({"type": "model_output"})),
        delta(0, json!({"type": "text", "text": &text[..at("1 in")]})),
        delta(0, json!({"type": "text", "text": &text[at("1 in")..]})),
        delta(
            0,
            json!({"type": "text_annotation_delta", "annotations": annotations}),
        ),
        stop(0),
        completed(),
    ];
    assert_restated_agrees(&wire(), [whole(vec![output.clone()])], streamed.clone());
    let response = decode(&wire(), Mode::Streaming, streamed).expect("the stream decodes");
    let [AssistantContent::Text(block)] = response.choice.as_slice() else {
        panic!("one text block: {:?}", response.choice);
    };
    let cited: Vec<_> = block
        .citations()
        .iter()
        .map(|citation| (block.cited(citation), citation.sources.clone()))
        .collect();
    let url = Source::new(SourceLocation::Url {
        url: "https://example.com".to_owned(),
    })
    .title("example.com");
    let file = Source::new(SourceLocation::Document {
        index: None,
        id: Some("files/report".to_owned()),
        within: Some(DocumentRange::Pages(3..4)),
    })
    .title("report.pdf");
    let place = Source::new(SourceLocation::Document {
        index: None,
        id: Some("places/berlin".to_owned()),
        within: None,
    })
    .title("Berlin");
    assert_eq!(
        cited,
        [
            (Some("in Berlin."), vec![url]),
            (Some("Spain won"), vec![file]),
            (Some("Berlin"), vec![place]),
        ]
    );
    assert_eq!(replayed(&response), [output]);
}
