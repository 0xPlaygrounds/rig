//! The OpenAI Responses websocket session, driven over a scripted in-memory
//! connection.
//!
//! These are protocol tests: turn lifecycle, `previous_response_id` chaining,
//! the late `response.done` filter, accumulator replay, and what a session does
//! after a fatal event. None of that involves a socket, so none of it is tested
//! through one — the session takes a
//! [`WebSocketConnection`](rig_core::ws_client::WebSocketConnection) and the
//! script supplies the frames. The real backend is exercised end-to-end in
//! `rig-tungstenite`'s own suite.

#![cfg(feature = "websocket")]
#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

#[path = "common/websocket_script.rs"]
mod websocket_script;

use rig_core::completion::{AssistantContent, CompletionModel as _, FinishReason};
use rig_core::providers::openai::responses_api::{
    CompletionResponse, IncompleteDetailsReason, Output, ResponseObject, ResponseStatus,
    ResponsesUsage,
};
use serde_json::json;
use std::time::Duration;
use websocket_script::{Script, session, session_with_timeout, test_client, test_model};

/// The terminal body every turn ends on unless a test needs another shape.
fn sample_response(status: ResponseStatus) -> CompletionResponse {
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

fn response_event(kind: &str, response: CompletionResponse, sequence: u64) -> String {
    json!({
        "type": kind,
        "sequence_number": sequence,
        "response": serde_json::to_value(response).expect("response should serialize"),
    })
    .to_string()
}

fn text_delta(item_id: &str, delta: &str, sequence: u64) -> String {
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

fn message_output(id: &str, status: &str, text: &str) -> Output {
    serde_json::from_value(json!({
        "type": "message",
        "id": id,
        "status": status,
        "role": "assistant",
        "content": [{ "type": "output_text", "annotations": [], "text": text }]
    }))
    .expect("output message should deserialize")
}

/// Every session entry point writes a `response.create`; assert the script saw
/// one rather than trusting the turn advanced by accident.
fn assert_response_create(payload: &str) {
    assert!(
        payload.contains("\"type\":\"response.create\""),
        "expected response.create payload, got {payload}"
    );
}

#[tokio::test]
async fn incomplete_turn_keeps_streamed_partial_output() {
    // The content exists ONLY in the delta events; the terminal
    // `response.incomplete` body has an empty `output`, which is a sequence the
    // wire protocol permits.
    let mut response = sample_response(ResponseStatus::Incomplete);
    response.incomplete_details = Some(IncompleteDetailsReason {
        reason: "max_output_tokens".to_string(),
    });
    let script = Script::turn([
        text_delta("msg_incomplete_1", "partial", 1),
        response_event("response.incomplete", response, 2),
    ]);

    let client = test_client();
    let model = test_model(&client);
    let mut session = session(&client, &script);

    let normalized = session
        .completion(model.completion_request("hello").build())
        .await
        .expect("incomplete turn should be a successful terminal");

    assert_response_create(&script.sent()[0]);
    // The streamed partial text survives, and normalization maps the incomplete
    // status to the same finish reason as the unary path.
    assert_eq!(normalized.finish_reason(), Some(FinishReason::Length));
    assert_eq!(normalized.usage.input_tokens, 1);
    assert_eq!(normalized.usage.output_tokens, 2);
    assert_eq!(normalized.usage.total_tokens, 3);
    assert!(matches!(
        normalized.choice.first(),
        Some(AssistantContent::Text(text)) if text.text == "partial"
    ));
}

/// #2258 P2: the websocket session shares `decode_item_chunk`, so text for one
/// message item interleaved with reasoning must aggregate as one text part here
/// too.
#[tokio::test]
async fn same_item_text_resumes_as_one_part_across_interleaved_reasoning() {
    let script = Script::turn([
        text_delta("msg_1", "hello ", 1),
        json!({
            "type": "response.reasoning_summary_text.delta",
            "delta": "because",
            "item_id": "rs_2",
            "output_index": 1,
            "summary_index": 0,
            "sequence_number": 2
        })
        .to_string(),
        text_delta("msg_1", "world", 3),
        response_event(
            "response.completed",
            sample_response(ResponseStatus::Completed),
            4,
        ),
    ]);

    let client = test_client();
    let model = test_model(&client);
    let mut session = session(&client, &script);

    let normalized = session
        .completion(model.completion_request("hello").build())
        .await
        .expect("interleaved turn should normalize");

    let texts: Vec<_> = normalized
        .choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(
        texts,
        ["hello world"],
        "same-item text must aggregate as one part around the reasoning"
    );
    assert!(
        normalized
            .choice
            .iter()
            .any(|content| matches!(content, AssistantContent::Reasoning(_))),
        "the interleaved reasoning must survive"
    );
}

#[tokio::test]
async fn completed_turn_without_deltas_falls_back_to_terminal_body() {
    // No delta events at all: the terminal body carries the full output, so
    // normalization must fall back to it.
    let mut response = sample_response(ResponseStatus::Completed);
    response.output = vec![message_output("msg_terminal_1", "completed", "hello there")];
    let script = Script::turn([response_event("response.completed", response, 1)]);

    let client = test_client();
    let model = test_model(&client);
    let mut session = session(&client, &script);

    let normalized = session
        .completion(model.completion_request("hello").build())
        .await
        .expect("completed turn should normalize");

    assert_response_create(&script.sent()[0]);
    assert!(matches!(
        normalized.choice.first(),
        Some(AssistantContent::Text(text)) if text.text == "hello there"
    ));
    assert_eq!(normalized.message_id.as_deref(), Some("msg_terminal_1"));
}

#[tokio::test]
async fn incomplete_turn_without_deltas_normalizes_terminal_body_output() {
    // No delta events at all AND an incomplete terminal whose body carries the
    // partial output: the body must be normalized rather than the turn reading
    // as empty.
    let mut response = sample_response(ResponseStatus::Incomplete);
    response.incomplete_details = Some(IncompleteDetailsReason {
        reason: "max_output_tokens".to_string(),
    });
    response.output = vec![message_output(
        "msg_body_only_1",
        "incomplete",
        "partial from body",
    )];
    let script = Script::turn([response_event("response.incomplete", response, 1)]);

    let client = test_client();
    let model = test_model(&client);
    let mut session = session(&client, &script);

    let normalized = session
        .completion(model.completion_request("hello").build())
        .await
        .expect("incomplete turn with body output should normalize");

    assert!(matches!(
        normalized.choice.first(),
        Some(AssistantContent::Text(text)) if text.text == "partial from body"
    ));
    assert_eq!(normalized.finish_reason(), Some(FinishReason::Length));
    assert_eq!(normalized.message_id.as_deref(), Some("msg_body_only_1"));
}

#[tokio::test]
async fn malformed_known_event_rejects_reuse_and_allows_close() {
    let script = Script::turn([json!({ "type": "response.completed" }).to_string()]);

    let client = test_client();
    let model = test_model(&client);
    let mut session = session(&client, &script);

    session
        .send(model.completion_request("hello").build())
        .await
        .expect("request should send");

    let error = session
        .next_event()
        .await
        .expect_err("malformed known event should fail");
    assert!(
        error.to_string().contains("StreamingCompletionChunk"),
        "expected strict decode failure, got {error}"
    );

    let closed = session
        .send(model.completion_request("retry").build())
        .await
        .expect_err("session should close after fatal parse error");
    assert!(
        closed.to_string().contains("session is closed"),
        "expected closed-session error, got {closed}"
    );

    session
        .close()
        .await
        .expect("explicit close after fatal parse error should succeed");
    assert!(script.closed(), "the close handshake should reach the peer");
}

#[tokio::test]
async fn event_timeout_rejects_reuse_and_allows_close() {
    // A turn that is accepted and then goes quiet: only the event timeout ends
    // the wait.
    let script = Script::turn([]).stalling();

    let client = test_client();
    let model = test_model(&client);
    let mut session = session_with_timeout(&client, &script, Some(Duration::from_millis(20)));

    session
        .send(model.completion_request("hello").build())
        .await
        .expect("request should send");

    let error = session
        .next_event()
        .await
        .expect_err("next_event should time out");
    assert!(
        error
            .to_string()
            .contains("Timed out waiting for the next OpenAI websocket event"),
        "expected timeout error, got {error}"
    );

    let closed = session
        .send(model.completion_request("retry").build())
        .await
        .expect_err("timed-out session should close");
    assert!(
        closed.to_string().contains("session is closed"),
        "expected closed-session error, got {closed}"
    );

    session
        .close()
        .await
        .expect("explicit close after timeout should succeed");
    assert!(script.closed(), "the close handshake should reach the peer");
}

/// One completed turn: the terminal `response.completed`, then the trailing
/// `response.done` OpenAI may emit after it.
fn completed_turn_with_late_done(response_id: &str, sequence: u64) -> Vec<String> {
    let response = CompletionResponse {
        id: response_id.to_string(),
        ..sample_response(ResponseStatus::Completed)
    };
    vec![
        response_event("response.completed", response, sequence),
        json!({
            "type": "response.done",
            "response": { "id": response_id, "status": "completed" },
        })
        .to_string(),
    ]
}

#[tokio::test]
async fn late_response_done_is_ignored_on_next_turn() {
    let script = Script::turns([
        completed_turn_with_late_done("resp_1", 1),
        completed_turn_with_late_done("resp_2", 3),
    ]);

    let client = test_client();
    let model = test_model(&client);
    let mut session = session(&client, &script);

    let first = session
        .raw_completion(model.completion_request("first").build())
        .await
        .expect("first response should complete");
    assert_eq!(first.id, "resp_1");
    assert_eq!(session.previous_response_id(), Some("resp_1"));

    let second = session
        .raw_completion(model.completion_request("second").build())
        .await
        .expect("second response should complete");
    assert_eq!(second.id, "resp_2");
    assert_eq!(session.previous_response_id(), Some("resp_2"));
}

#[tokio::test]
async fn clearing_previous_response_id_does_not_disable_late_done_filter() {
    let script = Script::turns([
        completed_turn_with_late_done("resp_1", 1),
        completed_turn_with_late_done("resp_2", 1),
    ]);

    let client = test_client();
    let model = test_model(&client);
    let mut session = session(&client, &script);

    let first = session
        .raw_completion(model.completion_request("first").build())
        .await
        .expect("first response should complete");
    assert_eq!(first.id, "resp_1");

    session.clear_previous_response_id();
    assert_eq!(session.previous_response_id(), None);

    let second = session
        .raw_completion(model.completion_request("second").build())
        .await
        .expect("second response should complete");
    assert_eq!(second.id, "resp_2");
}

#[tokio::test]
async fn failed_turn_keeps_late_done_out_of_next_request() {
    let failed = CompletionResponse {
        id: "resp_failed".to_string(),
        status: ResponseStatus::Failed,
        ..sample_response(ResponseStatus::Completed)
    };
    let script = Script::turns([
        vec![
            response_event("response.failed", failed, 1),
            json!({
                "type": "response.done",
                "response": { "id": "resp_failed", "status": "failed" },
            })
            .to_string(),
        ],
        completed_turn_with_late_done("resp_2", 2),
    ]);

    let client = test_client();
    let model = test_model(&client);
    let mut session = session(&client, &script);

    let error = session
        .raw_completion(model.completion_request("first").build())
        .await
        .expect_err("failed response should error");
    assert!(error.to_string().contains("failed response"));
    assert_eq!(session.previous_response_id(), None);

    let second = session
        .raw_completion(model.completion_request("second").build())
        .await
        .expect("second response should complete");
    assert_eq!(second.id, "resp_2");
}

/// A `response.done`-only turn (no `response.completed` before it) still ends
/// the turn and still chains the next request.
#[tokio::test]
async fn done_first_completed_turn_updates_previous_response_id() {
    let done_only = |response_id: &str| {
        vec![
            json!({
                "type": "response.done",
                "response": serde_json::to_value(CompletionResponse {
                    id: response_id.to_string(),
                    ..sample_response(ResponseStatus::Completed)
                })
                .expect("response should serialize"),
            })
            .to_string(),
        ]
    };
    let script = Script::turns([done_only("resp_1"), done_only("resp_2")]);

    let client = test_client();
    let model = test_model(&client);
    let mut session = session(&client, &script);

    let first = session
        .raw_completion(model.completion_request("first").build())
        .await
        .expect("first response should complete");
    assert_eq!(first.id, "resp_1");
    assert_eq!(session.previous_response_id(), Some("resp_1"));

    let second = session
        .raw_completion(model.completion_request("second").build())
        .await
        .expect("second response should complete");
    assert_eq!(second.id, "resp_2");
    assert_eq!(session.previous_response_id(), Some("resp_2"));

    // The chain is visible on the wire, not just in the session's state.
    let sent = script.sent();
    assert_response_create(&sent[1]);
    assert!(
        sent[1].contains("\"previous_response_id\":\"resp_1\""),
        "expected chained previous_response_id in payload, got {}",
        sent[1]
    );
}

#[tokio::test]
async fn done_first_failed_turn_does_not_chain_next_request() {
    let failed = CompletionResponse {
        id: "resp_failed".to_string(),
        status: ResponseStatus::Failed,
        ..sample_response(ResponseStatus::Completed)
    };
    let script = Script::turns([
        vec![
            json!({
                "type": "response.done",
                "response": serde_json::to_value(failed).expect("response should serialize"),
            })
            .to_string(),
        ],
        vec![
            json!({
                "type": "response.done",
                "response": serde_json::to_value(CompletionResponse {
                    id: "resp_2".to_string(),
                    ..sample_response(ResponseStatus::Completed)
                })
                .expect("response should serialize"),
            })
            .to_string(),
        ],
    ]);

    let client = test_client();
    let model = test_model(&client);
    let mut session = session(&client, &script);

    let error = session
        .raw_completion(model.completion_request("first").build())
        .await
        .expect_err("failed response should error");
    assert!(error.to_string().contains("failed response"));
    assert_eq!(session.previous_response_id(), None);

    let second = session
        .raw_completion(model.completion_request("second").build())
        .await
        .expect("second response should complete");
    assert_eq!(second.id, "resp_2");
    assert_eq!(session.previous_response_id(), Some("resp_2"));

    // A failed turn must not chain: the retry starts a fresh conversation.
    let sent = script.sent();
    assert!(
        !sent[1].contains("previous_response_id"),
        "a failed turn must not chain, got {}",
        sent[1]
    );
}

#[tokio::test]
async fn close_is_idempotent() {
    let script = Script::turn([]);
    let client = test_client();
    let mut session = session(&client, &script);

    session.close().await.expect("first close should succeed");
    session.close().await.expect("second close should succeed");
    assert!(script.closed());
}

#[tokio::test]
async fn send_while_in_flight_returns_error() {
    // The turn is accepted and stays open: the session must refuse a second
    // `response.create` rather than interleaving turns.
    let script = Script::turn([]).stalling();
    let client = test_client();
    let model = test_model(&client);
    let mut session = session(&client, &script);

    session
        .send(model.completion_request("first").build())
        .await
        .expect("first request should send");

    let error = session
        .send(model.completion_request("second").build())
        .await
        .expect_err("second send while in-flight should error");
    assert!(
        error.to_string().contains("already in flight"),
        "expected in-flight error, got {error}"
    );
}

#[tokio::test]
async fn send_after_close_returns_error() {
    let script = Script::turn([]);
    let client = test_client();
    let model = test_model(&client);
    let mut session = session(&client, &script);

    session.close().await.expect("close should succeed");

    let error = session
        .send(model.completion_request("after close").build())
        .await
        .expect_err("send after close should error");
    assert!(
        error.to_string().contains("session is closed"),
        "expected closed-session error, got {error}"
    );
}

#[tokio::test]
async fn next_event_without_send_returns_error() {
    let script = Script::turn([]);
    let client = test_client();
    let mut session = session(&client, &script);

    let error = session
        .next_event()
        .await
        .expect_err("next_event without send should error");
    assert!(
        error
            .to_string()
            .contains("No OpenAI websocket response is currently in flight"),
        "expected not-in-flight error, got {error}"
    );
}

#[tokio::test]
async fn unknown_event_is_skipped_and_reasoning_metadata_is_preserved() {
    let mut response = sample_response(ResponseStatus::Completed);
    response.id = "resp_after_unknown".to_string();
    let metadata = json!({
        "context": "all_turns",
        "effort": "ultra",
        "summary": null,
        "future_control": true
    });
    response.reasoning_metadata = Some(
        metadata
            .as_object()
            .expect("reasoning metadata should be an object")
            .clone(),
    );
    response.reasoning_context = Some("all_turns".to_string());

    let script = Script::turn([
        json!({ "type": "response.some_future_event", "data": "should be skipped" }).to_string(),
        response_event("response.completed", response, 1),
    ]);

    let client = test_client();
    let model = test_model(&client);
    let mut session = session(&client, &script);

    let response = session
        .raw_completion(model.completion_request("hello").build())
        .await
        .expect("response should complete despite unknown event");
    assert_eq!(response.id, "resp_after_unknown");
    assert_eq!(response.reasoning_context.as_deref(), Some("all_turns"));
    assert_eq!(response.reasoning_metadata.as_ref(), metadata.as_object());
}

/// Re-wraps SSE conformance fixture frames as websocket text payloads: the wire
/// events are identical across the two transports, only the framing (`data:`
/// lines vs. one JSON message per ws frame) differs.
fn ws_messages_from_sse_frames<'a>(
    frames: impl IntoIterator<Item = &'a bytes::Bytes>,
) -> Vec<String> {
    frames
        .into_iter()
        .flat_map(|frame| {
            std::str::from_utf8(frame)
                .expect("SSE fixture frames should be UTF-8")
                .lines()
                .filter_map(|line| line.strip_prefix("data:").map(str::trim))
                .filter(|data| !data.is_empty() && *data != "[DONE]")
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Websocket conformance over the shared Responses fixture: the SAME frames the
/// SSE conformance suite streams, re-wrapped as ws messages, must yield the same
/// content through the shared `classify_responses_frame` + accumulator
/// interpretation — text and tool-call deltas delivered, the unknown event
/// skipped, usage and finish reason taken from the terminal.
#[tokio::test]
async fn websocket_conformance_replays_sse_fixture_frames() {
    let fixture =
        rig_core::test_utils::streaming_conformance::fixtures::openai_responses::fixture();
    // The shared fixture scripts byte frames; re-wrap them as ws messages.
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

    let script = Script::turn(ws_messages_from_sse_frames(frames.iter()));
    let client = test_client();
    let model = test_model(&client);
    let mut session = session(&client, &script);

    let normalized = session
        .completion(model.completion_request("hello").build())
        .await
        .expect("fixture turn should normalize");

    assert_response_create(&script.sent()[0]);
    let texts: Vec<&str> = normalized
        .choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(texts, fixture.expected_texts);
    let tool_names: Vec<&str> = normalized
        .choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::ToolCall(call) => Some(call.function.name.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(tool_names, vec![fixture.expected_tool_name]);
    assert_eq!(normalized.usage.total_tokens, fixture.expected_usage_total);
    // The fixture's expected finish reason applies to its text-only sequences;
    // this combined replay carries a tool call, which the shared normalization
    // maps to `ToolCalls` on every transport.
    assert_eq!(normalized.finish_reason(), Some(FinishReason::ToolCalls));
}

/// Regression for the diverged websocket dispatch: `response.reasoning_text.delta`
/// was absent from the ws-private known-event list and silently dropped, while
/// the SSE path delivered it. Routed through the shared classifier, the
/// reasoning delta must survive to the normalized response.
#[tokio::test]
async fn reasoning_text_delta_arrives_over_websocket() {
    let script = Script::turn([
        json!({
            "type": "response.reasoning_text.delta",
            "item_id": "rs_1",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 1,
            "delta": "thinking hard",
        })
        .to_string(),
        text_delta("msg_1", "answer", 2),
        response_event(
            "response.completed",
            sample_response(ResponseStatus::Completed),
            3,
        ),
    ]);

    let client = test_client();
    let model = test_model(&client);
    let mut session = session(&client, &script);

    let normalized = session
        .completion(model.completion_request("hello").build())
        .await
        .expect("turn with reasoning deltas should normalize");

    assert!(
        normalized.choice.iter().any(|content| matches!(
            content,
            AssistantContent::Reasoning(reasoning)
                if reasoning.content.iter().any(|block| matches!(
                    block,
                    rig_core::message::ReasoningContent::Text { text, .. }
                        if text.contains("thinking hard")
                ))
        )),
        "reasoning delta should survive over websocket, got {:?}",
        normalized.choice
    );
    assert!(
        normalized.choice.iter().any(|content| matches!(
            content,
            AssistantContent::Text(text) if text.text == "answer"
        )),
        "text delta should survive alongside reasoning, got {:?}",
        normalized.choice
    );
}
