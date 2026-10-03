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

fn message_output(id: &str, status: &str, text: &str) -> serde_json::Value {
    json!({
        "type": "message",
        "id": id,
        "status": status,
        "role": "assistant",
        "content": [{ "type": "output_text", "annotations": [], "text": text }]
    })
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
    let mut response = sample_response("incomplete");
    response["incomplete_details"] = json!({ "reason": "max_output_tokens" });
    let script = Script::turn([
        text_delta("msg_incomplete_1", "partial", 1),
        response_event("response.incomplete", response, 2),
    ]);

    let client = test_client();
    let mut session = session(&client, &script);

    let normalized = session
        .completion(CompletionRequest::new("hello"))
        .await
        .expect("incomplete turn should be a successful terminal");

    assert_response_create(&script.sent()[0]);
    // The streamed partial text survives, and normalization maps the incomplete
    // status to the same finish reason as the unary path.
    assert_eq!(normalized.finish_reason(), Some(FinishReason::Length));
    assert_eq!(normalized.usage.input_tokens, Some(1));
    assert_eq!(normalized.usage.output_tokens, Some(2));
    assert_eq!(normalized.usage.total_tokens, Some(3));
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
        response_event("response.completed", sample_response("completed"), 4),
    ]);

    let client = test_client();
    let mut session = session(&client, &script);

    let normalized = session
        .completion(CompletionRequest::new("hello"))
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
    let mut response = sample_response("completed");
    response["output"] = json!([message_output("msg_terminal_1", "completed", "hello there")]);
    let script = Script::turn([response_event("response.completed", response, 1)]);

    let client = test_client();
    let mut session = session(&client, &script);

    let normalized = session
        .completion(CompletionRequest::new("hello"))
        .await
        .expect("completed turn should normalize");

    assert_response_create(&script.sent()[0]);
    assert!(matches!(
        normalized.choice.first(),
        Some(AssistantContent::Text(text)) if text.text == "hello there"
    ));
    assert_eq!(
        normalized.choice[0]
            .native_item()
            .and_then(|item| item["id"].as_str()),
        Some("msg_terminal_1")
    );
}

#[tokio::test]
async fn incomplete_turn_without_deltas_normalizes_terminal_body_output() {
    // No delta events at all AND an incomplete terminal whose body carries the
    // partial output: the body must be normalized rather than the turn reading
    // as empty.
    let mut response = sample_response("incomplete");
    response["incomplete_details"] = json!({ "reason": "max_output_tokens" });
    response["output"] = json!([message_output(
        "msg_body_only_1",
        "incomplete",
        "partial from body",
    )]);
    let script = Script::turn([response_event("response.incomplete", response, 1)]);

    let client = test_client();
    let mut session = session(&client, &script);

    let normalized = session
        .completion(CompletionRequest::new("hello"))
        .await
        .expect("incomplete turn with body output should normalize");

    assert!(matches!(
        normalized.choice.first(),
        Some(AssistantContent::Text(text)) if text.text == "partial from body"
    ));
    assert_eq!(normalized.finish_reason(), Some(FinishReason::Length));
    assert_eq!(
        normalized.choice[0]
            .native_item()
            .and_then(|item| item["id"].as_str()),
        Some("msg_body_only_1")
    );
}

#[tokio::test]
async fn malformed_frame_rejects_reuse_and_allows_close() {
    let script = Script::turn(["{not json".to_string()]);

    let client = test_client();
    let mut session = session(&client, &script);

    session
        .send(CompletionRequest::new("hello"))
        .await
        .expect("request should send");

    let error = session
        .next_event()
        .await
        .expect_err("a frame that is not JSON should fail");
    assert!(
        matches!(error, rig_core::error::ProviderError::Json(_)),
        "expected a JSON error, got {error}"
    );

    let closed = session
        .send(CompletionRequest::new("retry"))
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

/// A `response.done`-only turn (no `response.completed` before it) still ends
/// the turn and still chains the next request.
#[tokio::test]
async fn done_first_completed_turn_updates_previous_response_id() {
    let done_only = |response_id: &str| {
        vec![
            json!({
                "type": "response.done",
                "response": response_with(response_id, "completed"),
            })
            .to_string(),
        ]
    };
    let script = Script::turns([done_only("resp_1"), done_only("resp_2")]);

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

#[tokio::test]
async fn unknown_event_is_skipped_and_reasoning_metadata_is_preserved() {
    let mut response = sample_response("completed");
    response["id"] = json!("resp_after_unknown");
    let metadata = json!({
        "context": "all_turns",
        "effort": "ultra",
        "summary": null,
        "future_control": true
    });
    response["reasoning"] = metadata.clone();

    let script = Script::turn([
        json!({ "type": "response.some_future_event", "data": "should be skipped" }).to_string(),
        response_event("response.completed", response, 1),
    ]);

    let client = test_client();
    let mut session = session(&client, &script);

    let response = raw_response(
        session
            .completion(CompletionRequest::new("hello"))
            .await
            .expect("response should complete despite unknown event"),
    );
    assert_eq!(response["id"], "resp_after_unknown");
    assert_eq!(response["reasoning"], metadata);
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
        response_event("response.completed", sample_response("completed"), 3),
    ]);

    let client = test_client();
    let mut session = session(&client, &script);

    let normalized = session
        .completion(CompletionRequest::new("hello"))
        .await
        .expect("turn with reasoning deltas should normalize");

    assert!(
        normalized.choice.iter().any(|content| matches!(
            content,
            AssistantContent::Reasoning(reasoning) if reasoning.text.contains("thinking hard")
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

/// The session sends without the driver, so it runs the request-boundary
/// check itself: each empty piece is rejected before anything is written,
/// and the session stays usable.
#[tokio::test]
async fn an_empty_turn_is_rejected_before_anything_is_sent() {
    let with = |message: serde_json::Value| {
        CompletionRequest::new("hello").message(
            serde_json::from_value::<rig_core::message::Message>(message)
                .expect("an empty list parses"),
        )
    };
    let mut empty_history = CompletionRequest::new("hello");
    empty_history.chat_history.clear();
    let requests = [
        (empty_history, "request has an empty chat history"),
        (
            with(json!({"role": "user", "content": []})),
            "user message at index 0 has no content",
        ),
        (
            with(json!({"role": "assistant", "id": null, "content": []})),
            "assistant message at index 0 has no content",
        ),
    ];

    let client = test_client();
    let script = Script::turns(Vec::<Vec<String>>::new());
    let mut session = session(&client, &script);
    for (request, expected) in requests {
        let error = session
            .completion(request.clone())
            .await
            .expect_err("the turn is rejected");
        assert!(error.to_string().contains(expected), "{error}");
        let error = session
            .send(request)
            .await
            .expect_err("the send is rejected");
        assert!(error.to_string().contains(expected), "{error}");
    }
    assert!(script.sent().is_empty(), "nothing reached the socket");

    session
        .send(CompletionRequest::new("hello"))
        .await
        .expect("the session still sends");
    assert_eq!(script.sent().len(), 1);
}

/// The session shapes its history for the model the way the driver does: a
/// turn another model produced replays from its canonical fields, and a turn
/// that failed is not sent at all.
#[tokio::test]
async fn the_session_shapes_history_for_its_model() {
    use rig_core::message::{AssistantMessage, Message, Origin, Reasoning, StopReason, Text};
    let foreign = Message::Assistant(AssistantMessage {
        content: vec![
            AssistantContent::Reasoning(Reasoning::new("Thinking."))
                .with_native(json!({"type": "thinking", "signature": "sig"})),
            AssistantContent::Text(Text::new("Answer.")),
        ],
        origin: Some(Origin::new("anthropic.messages", "anthropic", "claude")),
        stop: Some(StopReason::Stop),
    });
    let failed = Message::Assistant(AssistantMessage {
        content: vec![AssistantContent::Text(Text::new("cut short"))],
        origin: Some(Origin::new("openai.responses", "openai", "gpt-5.4")),
        stop: Some(StopReason::Error("boom".to_owned())),
    });
    let request = CompletionRequest::new("next").messages([
        Message::user("first"),
        foreign,
        Message::user("again"),
        failed,
    ]);

    let client = test_client();
    let script = Script::turns(Vec::<Vec<String>>::new());
    let mut session = session(&client, &script);
    session.send(request).await.expect("the turn is sent");

    let sent: serde_json::Value =
        serde_json::from_str(&script.sent()[0]).expect("the payload is JSON");
    let assistant: Vec<&serde_json::Value> = sent["input"]
        .as_array()
        .expect("input is an array")
        .iter()
        .filter(|item| item["role"] == "assistant")
        .collect();
    let texts: Vec<&str> = assistant
        .iter()
        .filter_map(|item| item["content"][0]["text"].as_str())
        .collect();
    assert_eq!(texts, ["Thinking.", "Answer."], "{sent}");
}
