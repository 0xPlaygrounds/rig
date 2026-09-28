//! Server events for the websocket transport spikes: the same shapes the
//! session tests script.

#![allow(dead_code)]

use rig_core::providers::openai::responses_api::{
    CompletionResponse, IncompleteDetailsReason, Output, ResponseObject, ResponseStatus,
    ResponsesUsage,
};
use serde_json::json;

pub fn raw_response(response: &rig_core::completion::CompletionResponse) -> CompletionResponse {
    serde_json::from_value(response.raw.clone()).expect("`raw` is the Responses document")
}

pub fn sample_response(status: ResponseStatus) -> CompletionResponse {
    CompletionResponse {
        id: "resp_123".to_string(),
        object: ResponseObject::Response,
        provider_request_id: None,
        created_at: 0,
        status,
        error: None,
        incomplete_details: None,
        instructions: None,
        max_output_tokens: None,
        model: "gpt-5.4".to_string(),
        usage: Some(ResponsesUsage {
            input_tokens: 1,
            input_tokens_details: None,
            output_tokens: 2,
            output_tokens_details: Some(
                rig_core::providers::openai::responses_api::OutputTokensDetails {
                    reasoning_tokens: 0,
                },
            ),
            total_tokens: 3,
        }),
        output: Vec::new(),
        tools: Vec::new(),
        additional_parameters: Default::default(),
        provider_reasoning: None,
        reasoning_metadata: None,
        reasoning_context: None,
    }
}

pub fn with_id(id: &str, status: ResponseStatus) -> CompletionResponse {
    CompletionResponse {
        id: id.to_string(),
        ..sample_response(status)
    }
}

pub fn incomplete() -> CompletionResponse {
    let mut response = sample_response(ResponseStatus::Incomplete);
    response.incomplete_details = Some(IncompleteDetailsReason {
        reason: "max_output_tokens".to_string(),
    });
    response
}

pub fn response_event(kind: &str, response: CompletionResponse, sequence: u64) -> String {
    json!({
        "type": kind,
        "sequence_number": sequence,
        "response": serde_json::to_value(response).expect("response should serialize"),
    })
    .to_string()
}

pub fn completed(id: &str, sequence: u64) -> String {
    response_event(
        "response.completed",
        with_id(id, ResponseStatus::Completed),
        sequence,
    )
}

pub fn text_delta(item_id: &str, delta: &str, sequence: u64) -> String {
    json!({
        "type": "response.output_text.delta",
        "content_index": 0,
        "delta": delta,
        "item_id": item_id,
        "logprobs": [],
        "output_index": 0,
        "sequence_number": sequence,
    })
    .to_string()
}

pub fn reasoning_summary_delta(item_id: &str, delta: &str, sequence: u64) -> String {
    json!({
        "type": "response.reasoning_summary_text.delta",
        "delta": delta,
        "item_id": item_id,
        "output_index": 1,
        "summary_index": 0,
        "sequence_number": sequence
    })
    .to_string()
}

pub fn reasoning_text_delta(item_id: &str, delta: &str, sequence: u64) -> String {
    json!({
        "type": "response.reasoning_text.delta",
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "sequence_number": sequence,
        "delta": delta,
    })
    .to_string()
}

pub fn message_output(id: &str, status: &str, text: &str) -> Output {
    serde_json::from_value(json!({
        "type": "message",
        "id": id,
        "status": status,
        "role": "assistant",
        "content": [{ "type": "output_text", "annotations": [], "text": text }]
    }))
    .expect("output message should deserialize")
}

/// The trailing `response.done` OpenAI may emit after a terminal event.
pub fn late_done(id: &str, status: &str) -> String {
    json!({
        "type": "response.done",
        "response": { "id": id, "status": status },
    })
    .to_string()
}

pub fn completed_turn_with_late_done(id: &str, sequence: u64) -> Vec<String> {
    vec![completed(id, sequence), late_done(id, "completed")]
}

/// A `response.done` carrying the whole response: a done-first turn.
pub fn done_with_body(response: CompletionResponse) -> String {
    json!({
        "type": "response.done",
        "response": serde_json::to_value(response).expect("response should serialize"),
    })
    .to_string()
}

pub fn assert_response_create(payload: &str) {
    assert!(
        payload.contains("\"type\":\"response.create\""),
        "expected response.create payload, got {payload}"
    );
}

/// The SSE conformance fixture's frames, re-wrapped as websocket messages,
/// and the same frames as one SSE body.
pub fn fixture_frames() -> (Vec<String>, String) {
    let fixture =
        rig_core::test_utils::streaming_conformance::fixtures::openai_responses::fixture();
    let byte_frame = |frame: &rig_core::test_utils::streaming_conformance::WireInput| {
        frame
            .as_bytes()
            .cloned()
            .expect("the Responses fixture scripts byte frames")
    };
    let mut frames: Vec<bytes::Bytes> = Vec::new();
    frames.extend(fixture.text_frames.iter().map(byte_frame));
    frames.extend(fixture.tool_call_frames.iter().map(byte_frame));
    frames.extend(fixture.unknown_event_frame.iter().map(byte_frame));
    frames.extend(fixture.terminal_frames.iter().map(byte_frame));
    let messages: Vec<String> = frames
        .iter()
        .flat_map(|frame| {
            std::str::from_utf8(frame)
                .expect("SSE fixture frames should be UTF-8")
                .lines()
                .filter_map(|line| line.strip_prefix("data:").map(str::trim))
                .filter(|data| !data.is_empty() && *data != "[DONE]")
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>()
        })
        .collect();
    let body = messages
        .iter()
        .map(|message| format!("data: {message}\n\n"))
        .collect();
    (messages, body)
}

/// The same reply over HTTP SSE: every item and the response.
pub async fn over_sse(
    body: String,
) -> (
    Vec<rig_core::streaming::Item<rig_core::streaming::StreamEvent>>,
    rig_core::completion::CompletionResponse,
) {
    use futures::StreamExt;
    let model = rig_core::providers::openai::OpenAIConfig::new("test-key")
        .connect(rig_core::test_utils::SequencedStreamingHttpClient::new(
            vec![Ok(bytes::Bytes::from(body))],
        ))
        .responses("gpt-4o");
    let mut stream = model
        .stream(rig_core::completion::CompletionRequest::new("hello"))
        .expect("stream opens");
    let mut items = Vec::new();
    while let Some(item) = stream.next().await {
        items.push(item.expect("SSE item"));
    }
    (items, stream.finish().await.expect("SSE response"))
}

/// A three-message conversation, as an agent sends on its second turn.
pub fn history() -> rig_core::completion::CompletionRequest {
    use rig_core::completion::Message;
    let messages = rig_core::NonEmpty::from_vec(vec![
        Message::user("first question"),
        Message::assistant("first answer"),
        Message::user("second question"),
    ])
    .expect("non-empty");
    rig_core::completion::CompletionRequest::from(messages)
}
