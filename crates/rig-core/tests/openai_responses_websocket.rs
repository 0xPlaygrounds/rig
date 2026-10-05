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

use rig_core::completion::CompletionRequest;
use rig_core::completion::{AssistantContent, FinishReason};
use serde_json::json;
use std::time::Duration;
use websocket_script::{Script, session, session_with_timeout, test_client};

/// The terminal body every turn ends on unless a test needs another shape.
/// The provider's own terminal response object, read back off `raw`: the
/// session's `completion` folds it, and the document survives verbatim.
fn raw_response(response: rig_core::completion::CompletionResponse) -> serde_json::Value {
    response.raw
}

fn sample_response(status: &str) -> serde_json::Value {
    json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 0,
        "status": status,
        "model": "gpt-5.4",
        "usage": {
            "input_tokens": 1,
            "output_tokens": 2,
            "output_tokens_details": { "reasoning_tokens": 0 },
            "total_tokens": 3
        },
        "output": [],
        "tools": []
    })
}

/// [`sample_response`] under `id`.
fn response_with(id: &str, status: &str) -> serde_json::Value {
    let mut response = sample_response(status);
    response["id"] = json!(id);
    response
}

fn response_event(kind: &str, response: serde_json::Value, sequence: u64) -> String {
    json!({
        "type": kind,
        "sequence_number": sequence,
        "response": response,
    })
    .to_string()
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
async fn event_timeout_rejects_reuse_and_allows_close() {
    // A turn that is accepted and then goes quiet: only the event timeout ends
    // the wait.
    let script = Script::turn([]).stalling();

    let client = test_client();
    let mut session = session_with_timeout(&client, &script, Some(Duration::from_millis(20)));

    session
        .send(CompletionRequest::new("hello"))
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
        .send(CompletionRequest::new("retry"))
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
    let response = response_with(response_id, "completed");
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
    let mut session = session(&client, &script);

    let first = raw_response(
        session
            .completion(CompletionRequest::new("first"))
            .await
            .expect("first response should complete"),
    );
    assert_eq!(first["id"], "resp_1");
    assert_eq!(session.previous_response_id(), Some("resp_1"));

    let second = raw_response(
        session
            .completion(CompletionRequest::new("second"))
            .await
            .expect("second response should complete"),
    );
    assert_eq!(second["id"], "resp_2");
    assert_eq!(session.previous_response_id(), Some("resp_2"));
}

#[tokio::test]
async fn clearing_previous_response_id_does_not_disable_late_done_filter() {
    let script = Script::turns([
        completed_turn_with_late_done("resp_1", 1),
        completed_turn_with_late_done("resp_2", 1),
    ]);

    let client = test_client();
    let mut session = session(&client, &script);

    let first = raw_response(
        session
            .completion(CompletionRequest::new("first"))
            .await
            .expect("first response should complete"),
    );
    assert_eq!(first["id"], "resp_1");

    session.clear_previous_response_id();
    assert_eq!(session.previous_response_id(), None);

    let second = raw_response(
        session
            .completion(CompletionRequest::new("second"))
            .await
            .expect("second response should complete"),
    );
    assert_eq!(second["id"], "resp_2");
}

#[tokio::test]
async fn failed_turn_keeps_late_done_out_of_next_request() {
    let failed = response_with("resp_failed", "failed");
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
    let mut session = session(&client, &script);

    let error = session
        .completion(CompletionRequest::new("first"))
        .await
        .expect_err("failed response should error");
    assert!(error.to_string().contains("failed response"));
    assert_eq!(session.previous_response_id(), None);

    let second = raw_response(
        session
            .completion(CompletionRequest::new("second"))
            .await
            .expect("second response should complete"),
    );
    assert_eq!(second["id"], "resp_2");
}

#[tokio::test]
async fn done_first_failed_turn_does_not_chain_next_request() {
    let failed = response_with("resp_failed", "failed");
    let script = Script::turns([
        vec![
            json!({
                "type": "response.done",
                "response": failed,
            })
            .to_string(),
        ],
        vec![
            json!({
                "type": "response.done",
                "response": response_with("resp_2", "completed"),
            })
            .to_string(),
        ],
    ]);

    let client = test_client();
    let mut session = session(&client, &script);

    let error = session
        .completion(CompletionRequest::new("first"))
        .await
        .expect_err("failed response should error");
    assert!(error.to_string().contains("failed response"));
    assert_eq!(session.previous_response_id(), None);

    let second = raw_response(
        session
            .completion(CompletionRequest::new("second"))
            .await
            .expect("second response should complete"),
    );
    assert_eq!(second["id"], "resp_2");
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
    let mut session = session(&client, &script);

    session
        .send(CompletionRequest::new("first"))
        .await
        .expect("first request should send");

    let error = session
        .send(CompletionRequest::new("second"))
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
    let mut session = session(&client, &script);

    session.close().await.expect("close should succeed");

    let error = session
        .send(CompletionRequest::new("after close"))
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
    let mut session = session(&client, &script);

    let normalized = session
        .completion(CompletionRequest::new("hello"))
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
    assert_eq!(
        normalized.usage.total_tokens,
        Some(fixture.expected_usage_total)
    );
    // The fixture's expected finish reason applies to its text-only sequences;
    // this combined replay carries a tool call, which the shared normalization
    // maps to `ToolCalls` on every transport.
    assert_eq!(normalized.finish_reason(), Some(FinishReason::ToolCalls));
}
