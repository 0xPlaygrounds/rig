use futures::StreamExt;
use rig::completion::{CompletionRequest, CompletionResponse};
use rig::message::{AdditionalParams, AssistantContent, Text};
use rig::providers::openai::OpenAIConfig;
use rig::streaming::{CompletionStream, Item, StreamEvent};
use rig::test_utils::{MockHttpResponse, SequencedHttpClient};
use serde_json::{Value, json};

pub fn search_request() -> CompletionRequest {
    CompletionRequest::new(
        "Use web search to find the Rust programming language's official website. Reply with one short sentence and a source citation.",
    )
    .max_tokens(512)
    .additional_params(json!({"store": false, "tools": [{"type": "web_search"}]}))
}

pub fn snapshot_texts(document: &Value) -> Vec<Text> {
    document["output"]
        .as_array()
        .expect("output items")
        .iter()
        .filter(|item| item["type"] == "message")
        .map(|item| {
            let mut text = Text::new("");
            for part in item["content"].as_array().expect("message content") {
                text.text
                    .push_str(part["text"].as_str().expect("output text"));
                let mut extras = part.as_object().expect("content part").clone();
                extras.remove("type");
                extras.remove("text");
                extras.retain(|_, value| {
                    !(value.is_null()
                        || value.as_array().is_some_and(Vec::is_empty)
                        || value.as_object().is_some_and(serde_json::Map::is_empty))
                });
                if let Some(phase) = item.get("phase").filter(|v| !v.is_null()) {
                    extras.insert("phase".into(), phase.clone());
                }
                if !extras.is_empty() {
                    let incoming = AdditionalParams::from_entries([(
                        "openai_responses",
                        Value::Object(extras),
                    )])
                    .expect("extras");
                    match &mut text.additional_params {
                        Some(params) => params.merge(incoming),
                        slot @ None => *slot = Some(incoming),
                    }
                }
            }
            text
        })
        .collect()
}

pub async fn assert_stream_snapshots(
    mut stream: CompletionStream,
) -> (CompletionResponse, Vec<Value>) {
    let mut ends = Vec::new();
    let mut fragments = String::new();
    let mut annotation_events = Vec::new();
    while let Some(item) = stream.next().await {
        match item.expect("stream item") {
            Item::Event(StreamEvent::End {
                content: AssistantContent::Text(text),
                ..
            }) => ends.push(text),
            Item::Event(StreamEvent::Text { text, .. }) => fragments.push_str(&text),
            Item::Unknown(payload)
                if payload.value()["type"] == "response.output_text.annotation.added" =>
            {
                annotation_events.push(payload.value().clone());
            }
            _ => {}
        }
    }
    let response = stream.finish().await.expect("terminal response");
    let expected = snapshot_texts(&response.raw);
    assert!(!expected.is_empty(), "search must return text");
    assert_eq!(
        ends, expected,
        "text End must retain each snapshot's extras exactly once"
    );
    assert_eq!(
        fragments,
        expected.iter().map(|t| t.text.as_str()).collect::<String>()
    );
    (response, annotation_events)
}

pub fn assert_unary_snapshots(response: &CompletionResponse) {
    assert_eq!(
        response
            .choice
            .iter()
            .filter_map(|part| match part {
                AssistantContent::Text(text) => Some(text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>(),
        snapshot_texts(&response.raw)
    );
}

pub fn assert_has_citations(document: &Value) {
    assert!(
        snapshot_texts(document).iter().any(|text| text
            .additional_params
            .as_ref()
            .and_then(|p| p.get("openai_responses"))
            .and_then(|p| p.get("annotations"))
            .and_then(Value::as_array)
            .is_some_and(|a| !a.is_empty())),
        "hosted search must produce annotations"
    );
}

/// Replay frozen bytes, then exercise terminal-only metadata with a filtered replay.
/// It omits all output_item.done and annotation.added frames, drops SSE event headers,
/// and reserializes remaining JSON values into canonical data lines. The unary parity
/// body is reserialized from response.completed.response, not a second live reply.
pub async fn assert_recorded_parity(provider: &str, scenario: &str) {
    let bodies = crate::cassettes::recorded_interaction_bodies(provider, scenario);
    let (_, streaming_body) = bodies
        .iter()
        .find(|(_, body)| body.contains("response.output_text.delta"))
        .expect("recorded streamed search");
    let frames: Vec<Value> = streaming_body
        .lines()
        .filter_map(|line| {
            line.strip_prefix("data:")
                .and_then(|data| serde_json::from_str(data.trim()).ok())
        })
        .collect();
    let terminal = &frames
        .iter()
        .find(|frame| frame["type"] == "response.completed")
        .expect("completed frame")["response"];
    assert_has_citations(terminal);
    let snapshots: Vec<_> = frames
        .iter()
        .filter(|f| f["type"] == "response.output_item.done" && f["item"]["type"] == "message")
        .collect();
    assert!(
        !snapshots.is_empty(),
        "real event sequence includes message done snapshots"
    );
    for snapshot in snapshots {
        let matching = terminal["output"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["id"] == snapshot["item"]["id"])
            .unwrap();
        assert_eq!(snapshot["item"]["content"], matching["content"]);
    }
    let unary_body = serde_json::to_vec(terminal).unwrap();
    let http = SequencedHttpClient::new([MockHttpResponse::Success(unary_body.into())]);
    let model =
        rig_test_support::cassette_models::OpenAiModels::new(OpenAIConfig::new("offline"), http)
            .responses("fixture-model");
    let unary = model
        .call(CompletionRequest::new("scripted parity"))
        .await
        .expect("identical-content unary");
    let expected = snapshot_texts(terminal);
    assert_unary_snapshots(&unary);
    for (label, bytes) in [
        ("unmodified recorded SSE", streaming_body.clone()),
        (
            "filtered JSON replay: omit output_item.done and annotation.added",
            frames
                .iter()
                .filter(|frame| {
                    frame["type"] != "response.output_item.done"
                        && frame["type"] != "response.output_text.annotation.added"
                })
                .map(|frame| format!("data: {frame}\n\n"))
                .collect::<String>(),
        ),
    ] {
        let mut headers = rig::http_client::HeaderMap::new();
        headers.insert("content-type", "text/event-stream".parse().unwrap());
        let http =
            SequencedHttpClient::new([MockHttpResponse::SuccessWithHeaders(bytes.into(), headers)]);
        let model = rig_test_support::cassette_models::OpenAiModels::new(
            OpenAIConfig::new("offline"),
            http,
        )
        .responses("fixture-model");
        let (response, annotations) =
            assert_stream_snapshots(model.stream(CompletionRequest::new(label)).unwrap()).await;
        assert_eq!(snapshot_texts(&response.raw), expected);
        let expected_events = if label == "unmodified recorded SSE" {
            frames
                .iter()
                .filter(|frame| frame["type"] == "response.output_text.annotation.added")
                .cloned()
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        assert_eq!(
            annotations, expected_events,
            "raw annotation events remain verbatim"
        );
    }
    let (_, unary_body) = bodies
        .iter()
        .find(|(_, body)| !body.contains("data:"))
        .expect("live unary counterpart");
    let unary_document = serde_json::from_str::<Value>(unary_body).unwrap();
    assert_has_citations(&unary_document);
    let live_unary = snapshot_texts(&unary_document);
    assert_eq!(
        live_unary, expected,
        "recorded live counterparts carry equal text and extras"
    );
}
