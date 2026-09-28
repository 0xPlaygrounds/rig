//! The OpenAI Responses websocket transport, driven by `Model` over a
//! scripted in-memory connection.
//!
//! These are protocol tests: the turn lifecycle, chaining, the late
//! `response.done` filter, the queue, dropped turns, and what a connection
//! does after a fatal event. None of that involves a socket, so the
//! transport takes a [`WebSocketConnection`](rig_core::ws_client::WebSocketConnection)
//! and the script supplies the frames. `streaming_conformance_websocket`
//! drives a real socket.

#![cfg(feature = "websocket")]
#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

#[path = "common/websocket_script.rs"]
mod websocket_script;
#[path = "common/ws_turns.rs"]
mod ws_turns;

use futures::StreamExt;
use rig_core::completion::{AssistantContent, CompletionRequest, FinishReason};
use rig_core::driver::{DynModel, Model};
use rig_core::operation::Completion;
use rig_core::providers::openai::responses_api::ResponseStatus;
use rig_core::providers::openai::responses_api::websocket::{ResponsesSocket, ResponsesWebSocket};
use rig_core::streaming::{Item, StreamEvent};
use std::time::Duration;
use websocket_script::{Script, test_client};
use ws_turns::*;

type SocketModel = Model<ResponsesSocket, ResponsesWebSocket>;

fn transport(script: &Script) -> ResponsesWebSocket {
    ResponsesWebSocket::from_connection(script.connection())
}

fn model(script: &Script) -> SocketModel {
    with_transport(transport(script))
}

fn chaining(script: &Script) -> SocketModel {
    with_transport(transport(script).chaining())
}

fn with_transport(transport: ResponsesWebSocket) -> SocketModel {
    Model::new(ResponsesSocket::new(test_client().wire), transport)
}

fn texts(response: &rig_core::completion::CompletionResponse) -> Vec<String> {
    response
        .choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.clone()),
            _ => None,
        })
        .collect()
}

fn sent_json(script: &Script, index: usize) -> serde_json::Value {
    serde_json::from_str(&script.sent()[index]).expect("the transport sends JSON")
}

// A turn is an ordinary reply.

#[tokio::test]
async fn stream_delivers_events_before_the_turn_ends() {
    // No terminal: a transport that buffered the turn would never yield.
    let script =
        Script::turn([text_delta("msg_1", "hel", 1), text_delta("msg_1", "lo", 2)]).stalling();
    let model = model(&script);
    let mut stream = model.stream("hello").expect("stream opens");
    let mut fragments = Vec::new();
    while fragments.len() < 2 {
        let item = tokio::time::timeout(Duration::from_secs(1), stream.next())
            .await
            .expect("an event arrives before the turn ends")
            .expect("the stream is open")
            .expect("the item is an event");
        if let Item::Event(StreamEvent::Text { text, .. }) = item {
            fragments.push(text);
        }
    }
    assert_eq!(fragments, ["hel", "lo"]);
    assert_response_create(&script.sent()[0]);
}

#[tokio::test]
async fn call_equals_stream_finish() {
    let turn = || {
        Script::turn([
            text_delta("msg_1", "hello ", 1),
            reasoning_summary_delta("rs_2", "because", 2),
            text_delta("msg_1", "world", 3),
            completed("resp_1", 4),
        ])
    };
    let called = model(&turn()).call("hello").await.expect("call");
    let streamed = model(&turn())
        .stream("hello")
        .expect("stream opens")
        .finish()
        .await
        .expect("finish");
    assert_eq!(called, streamed);
    // Same-item text aggregates as one part around the interleaved
    // reasoning, and the reasoning survives.
    assert_eq!(texts(&called), ["hello world"]);
    assert!(
        called
            .choice
            .iter()
            .any(|content| matches!(content, AssistantContent::Reasoning(_)))
    );
}

#[tokio::test]
async fn websocket_and_sse_yield_the_same_items_and_response() {
    let (messages, body) = fixture_frames();
    let (sse_items, sse_response) = over_sse(body).await;

    let script = Script::turn(messages);
    let model = model(&script);
    let mut stream = model.stream("hello").expect("stream opens");
    let mut items = Vec::new();
    while let Some(item) = stream.next().await {
        items.push(item.expect("websocket item"));
    }
    let response = stream.finish().await.expect("websocket response");

    assert_eq!(items, sse_items);
    let (mut websocket, mut sse) = (response.clone(), sse_response);
    websocket.provider_request_id = None;
    sse.provider_request_id = None;
    assert_eq!(websocket, sse);
    // The fixture carries a tool call, which every transport maps to
    // `ToolCalls`.
    assert_eq!(response.finish_reason(), Some(FinishReason::ToolCalls));
}

#[tokio::test]
async fn erases_to_dyn_model() {
    let script = Script::turn([text_delta("msg_1", "hi", 1), completed("resp_1", 2)]);
    let erased: DynModel<Completion> = model(&script).into();
    let response = erased
        .call(CompletionRequest::new("hello"))
        .await
        .expect("call");
    assert_eq!(texts(&response), ["hi"]);
}

// Terminal bodies.

#[tokio::test]
async fn incomplete_turn_keeps_streamed_partial_output() {
    let script = Script::turn([
        text_delta("msg_incomplete_1", "partial", 1),
        response_event("response.incomplete", incomplete(), 2),
    ]);
    let response = model(&script)
        .call("hello")
        .await
        .expect("incomplete is a terminal");
    assert_eq!(response.finish_reason(), Some(FinishReason::Length));
    assert_eq!(response.usage.input_tokens, Some(1));
    assert_eq!(response.usage.output_tokens, Some(2));
    assert_eq!(response.usage.total_tokens, Some(3));
    assert_eq!(texts(&response), ["partial"]);
}

#[tokio::test]
async fn completed_turn_without_deltas_falls_back_to_terminal_body() {
    let mut body = with_id("resp_1", ResponseStatus::Completed);
    body.output = vec![message_output("msg_terminal_1", "completed", "hello there")];
    let script = Script::turn([response_event("response.completed", body, 1)]);
    let response = model(&script).call("hello").await.expect("completed");
    assert_eq!(texts(&response), ["hello there"]);
    assert_eq!(response.message_id.as_deref(), Some("msg_terminal_1"));
}

#[tokio::test]
async fn incomplete_turn_without_deltas_normalizes_terminal_body_output() {
    let mut body = incomplete();
    body.output = vec![message_output(
        "msg_body_only_1",
        "incomplete",
        "partial from body",
    )];
    let script = Script::turn([response_event("response.incomplete", body, 1)]);
    let response = model(&script).call("hello").await.expect("incomplete");
    assert_eq!(texts(&response), ["partial from body"]);
    assert_eq!(response.finish_reason(), Some(FinishReason::Length));
    assert_eq!(response.message_id.as_deref(), Some("msg_body_only_1"));
}

/// `response.reasoning_text.delta` goes through the shared classifier, as
/// on SSE.
#[tokio::test]
async fn reasoning_text_delta_arrives_over_websocket() {
    let script = Script::turn([
        reasoning_text_delta("rs_1", "thinking hard", 1),
        text_delta("msg_1", "answer", 2),
        completed("resp_1", 3),
    ]);
    let response = model(&script).call("hello").await.expect("completed");
    assert!(
        response.choice.iter().any(|content| matches!(
            content,
            AssistantContent::Reasoning(reasoning)
                if reasoning.open(reasoning.issuer()).is_some_and(|reasoning| reasoning
                    .content
                    .iter()
                    .any(|block| matches!(
                        block,
                        rig_core::message::ReasoningContent::Text { text, .. }
                            if text.contains("thinking hard")
                    )))
        )),
        "{:?}",
        response.choice
    );
    assert_eq!(texts(&response), ["answer"]);
}

#[tokio::test]
async fn unknown_event_reaches_the_consumer_and_metadata_survives() {
    let mut body = with_id("resp_after_unknown", ResponseStatus::Completed);
    let metadata = serde_json::json!({
        "context": "all_turns",
        "effort": "ultra",
        "summary": null,
        "future_control": true
    });
    body.reasoning_metadata = metadata.as_object().cloned();
    body.reasoning_context = Some("all_turns".to_string());
    let script = Script::turn([
        serde_json::json!({ "type": "response.some_future_event", "data": "x" }).to_string(),
        response_event("response.completed", body, 1),
    ]);
    let model = model(&script);
    let mut stream = model.stream("hello").expect("stream opens");
    let mut unknown = 0;
    while let Some(item) = stream.next().await {
        if let Item::Unknown(_) = item.expect("item") {
            unknown += 1;
        }
    }
    let response = stream.finish().await.expect("completed");
    assert_eq!(unknown, 1);
    let raw = raw_response(&response);
    assert_eq!(raw.id, "resp_after_unknown");
    assert_eq!(raw.reasoning_context.as_deref(), Some("all_turns"));
    assert_eq!(raw.reasoning_metadata.as_ref(), metadata.as_object());
}

// Failures.

#[tokio::test]
async fn failed_turn_errors_with_the_failure_envelope() {
    let script = Script::turn([response_event(
        "response.failed",
        with_id("resp_failed", ResponseStatus::Failed),
        1,
    )]);
    let error = model(&script).call("hello").await.expect_err("failed");
    assert!(error.to_string().contains("failed response"), "{error}");
}

#[tokio::test]
async fn error_event_preserves_the_provider_payload() {
    let script = Script::turn([serde_json::json!({
        "type": "error",
        "error": { "code": "rate_limit_exceeded", "message": "slow down" },
    })
    .to_string()]);
    let error = model(&script).call("hello").await.expect_err("error event");
    let body = error
        .provider_response_json()
        .expect("the body is JSON")
        .expect("the body is kept");
    assert_eq!(body["error"]["code"], "rate_limit_exceeded");
}

#[tokio::test]
async fn done_without_a_body_is_an_error() {
    let script = Script::turn([late_done("resp_1", "completed")]);
    let error = model(&script).call("hello").await.expect_err("no body");
    assert!(
        error
            .to_string()
            .contains("before a terminal response body was available"),
        "{error}"
    );
}

#[tokio::test]
async fn malformed_terminal_fails_its_turn_and_the_next_turn_runs() {
    let script = Script::turns([
        vec![serde_json::json!({ "type": "response.completed" }).to_string()],
        vec![completed("resp_2", 1)],
    ]);
    let model = model(&script);
    let error = model.call("first").await.expect_err("malformed");
    assert!(
        error.to_string().contains("StreamingCompletionChunk"),
        "{error}"
    );
    let second = model.call("second").await.expect("second");
    assert_eq!(raw_response(&second).id, "resp_2");
}

#[tokio::test]
async fn malformed_event_mid_turn_leaves_its_rest_out_of_the_next_turn() {
    let script = Script::turns([
        vec![
            text_delta("msg_1", "one", 1),
            serde_json::json!({ "type": "response.output_text.delta" }).to_string(),
            text_delta("msg_1", "stale", 3),
            completed("resp_1", 4),
        ],
        vec![text_delta("msg_2", "two", 1), completed("resp_2", 2)],
    ]);
    let model = chaining(&script);
    model.call("first").await.expect_err("malformed delta");
    let second = model.call("second").await.expect("second");
    assert_eq!(texts(&second), ["two"]);
    assert_eq!(raw_response(&second).id, "resp_2");
    // The first turn failed, so its response is not continued.
    assert!(sent_json(&script, 1).get("previous_response_id").is_none());
}

#[tokio::test]
async fn event_timeout_fails_the_turn_and_closes_the_connection() {
    let script = Script::turn([]).stalling();
    let model = with_transport(transport(&script).event_timeout(Some(Duration::from_millis(20))));
    let error = model.call("hello").await.expect_err("timeout");
    assert!(
        error
            .to_string()
            .contains("Timed out waiting for the next OpenAI websocket event"),
        "{error}"
    );
    let closed = model.call("retry").await.expect_err("closed");
    assert!(closed.to_string().contains("session is closed"), "{closed}");
    model.transport.close().await.expect("close after timeout");
    assert!(script.closed(), "the close handshake reaches the peer");
}

#[tokio::test]
async fn a_peer_hanging_up_mid_turn_closes_the_connection() {
    let script = Script::turn([text_delta("msg_1", "one", 1)]);
    let model = model(&script);
    let error = model.call("hello").await.expect_err("hung up");
    assert!(
        error
            .to_string()
            .contains("closed before the turn finished"),
        "{error}"
    );
    let closed = model.call("retry").await.expect_err("closed");
    assert!(closed.to_string().contains("session is closed"), "{closed}");
}

#[tokio::test]
async fn close_is_idempotent_and_later_turns_fail() {
    let script = Script::turn([]);
    let model = model(&script);
    model.transport.close().await.expect("close");
    model.transport.close().await.expect("close again");
    assert!(script.closed());
    let error = model.call("after close").await.expect_err("closed");
    assert!(error.to_string().contains("session is closed"), "{error}");
}

// Chaining.

#[tokio::test]
async fn nothing_is_chained_by_default() {
    let script = Script::turns([vec![completed("resp_1", 1)], vec![completed("resp_2", 1)]]);
    let model = model(&script);
    model.call("first").await.expect("first");
    // The id is readable without chaining.
    assert_eq!(
        model.transport.last_response_id().await.as_deref(),
        Some("resp_1")
    );
    model.call("second").await.expect("second");
    assert!(
        sent_json(&script, 1).get("previous_response_id").is_none(),
        "{}",
        script.sent()[1]
    );
}

#[tokio::test]
async fn full_history_is_sent_without_a_chain_by_default() {
    let script = Script::turns([vec![completed("resp_1", 1)], vec![completed("resp_2", 1)]]);
    let model = model(&script);
    model.call("first question").await.expect("first");
    model.call(history()).await.expect("second");
    let sent = sent_json(&script, 1);
    assert!(sent.get("previous_response_id").is_none(), "{sent}");
    assert_eq!(sent["input"].as_array().map(Vec::len), Some(3));
}

#[tokio::test]
async fn chaining_continues_the_last_response_and_late_done_is_ignored() {
    let script = Script::turns([
        completed_turn_with_late_done("resp_1", 1),
        completed_turn_with_late_done("resp_2", 3),
    ]);
    let model = chaining(&script);
    let first = model.call("first").await.expect("first");
    assert_eq!(raw_response(&first).id, "resp_1");
    let mut stream = model.stream("second").expect("stream opens");
    while let Some(item) = stream.next().await {
        assert!(
            !matches!(item.expect("item"), Item::Unknown(_)),
            "the first turn's trailing response.done leaked into the second"
        );
    }
    let second = stream.finish().await.expect("second");
    assert_eq!(raw_response(&second).id, "resp_2");
    assert_eq!(sent_json(&script, 1)["previous_response_id"], "resp_1");
    assert_eq!(
        model.transport.last_response_id().await.as_deref(),
        Some("resp_2")
    );
}

#[tokio::test]
async fn a_request_that_names_its_own_previous_response_wins() {
    let script = Script::turns([vec![completed("resp_1", 1)], vec![completed("resp_2", 1)]]);
    let model = chaining(&script);
    model.call("first").await.expect("first");
    let mut request = CompletionRequest::new("second");
    request.additional_params = Some(serde_json::json!({ "previous_response_id": "resp_0" }));
    model.call(request).await.expect("second");
    assert_eq!(sent_json(&script, 1)["previous_response_id"], "resp_0");
}

#[tokio::test]
async fn clearing_the_chain_does_not_disable_the_late_done_filter() {
    let script = Script::turns([
        completed_turn_with_late_done("resp_1", 1),
        completed_turn_with_late_done("resp_2", 1),
    ]);
    let model = chaining(&script);
    model.call("first").await.expect("first");
    model.transport.clear_chain().await;
    assert_eq!(model.transport.last_response_id().await, None);
    let second = model.call("second").await.expect("second");
    assert_eq!(raw_response(&second).id, "resp_2");
    assert!(sent_json(&script, 1).get("previous_response_id").is_none());
}

#[tokio::test]
async fn failed_turn_resets_the_chain_and_its_late_done_is_ignored() {
    let script = Script::turns([
        vec![completed("resp_0", 1)],
        vec![
            response_event(
                "response.failed",
                with_id("resp_failed", ResponseStatus::Failed),
                1,
            ),
            late_done("resp_failed", "failed"),
        ],
        completed_turn_with_late_done("resp_2", 2),
    ]);
    let model = chaining(&script);
    model.call("zeroth").await.expect("zeroth");
    let error = model.call("first").await.expect_err("failed");
    assert!(error.to_string().contains("failed response"), "{error}");
    assert_eq!(model.transport.last_response_id().await, None);
    let second = model.call("second").await.expect("second");
    assert_eq!(raw_response(&second).id, "resp_2");
    assert_eq!(sent_json(&script, 1)["previous_response_id"], "resp_0");
    assert!(sent_json(&script, 2).get("previous_response_id").is_none());
}

/// A `response.done`-only turn (no `response.completed` before it) still
/// ends the turn and still chains the next one.
#[tokio::test]
async fn done_first_completed_turn_chains_the_next() {
    let script = Script::turns([
        vec![done_with_body(with_id("resp_1", ResponseStatus::Completed))],
        vec![done_with_body(with_id("resp_2", ResponseStatus::Completed))],
    ]);
    let model = chaining(&script);
    let first = model.call("first").await.expect("first");
    assert_eq!(raw_response(&first).id, "resp_1");
    let second = model.call("second").await.expect("second");
    assert_eq!(raw_response(&second).id, "resp_2");
    assert_response_create(&script.sent()[1]);
    assert_eq!(sent_json(&script, 1)["previous_response_id"], "resp_1");
}

#[tokio::test]
async fn done_first_failed_turn_does_not_chain_the_next() {
    let script = Script::turns([
        vec![done_with_body(with_id(
            "resp_failed",
            ResponseStatus::Failed,
        ))],
        vec![done_with_body(with_id("resp_2", ResponseStatus::Completed))],
    ]);
    let model = chaining(&script);
    let error = model.call("first").await.expect_err("failed");
    assert!(error.to_string().contains("failed response"), "{error}");
    let second = model.call("second").await.expect("second");
    assert_eq!(raw_response(&second).id, "resp_2");
    assert!(sent_json(&script, 1).get("previous_response_id").is_none());
}

// The queue.

#[tokio::test]
async fn concurrent_turns_queue_on_one_connection() {
    let script = Script::turns([
        vec![text_delta("msg_1", "one", 1), completed("resp_1", 2)],
        vec![text_delta("msg_2", "two", 1), completed("resp_2", 2)],
    ]);
    let model = chaining(&script);
    let mut first = model.stream("first").expect("stream opens");
    first
        .next()
        .await
        .expect("an item")
        .expect("the first turn is in flight");
    let second = model.call("second");
    tokio::pin!(second);
    // The first turn holds the connection: the second is not even sent.
    assert!(
        tokio::time::timeout(Duration::from_millis(50), &mut second)
            .await
            .is_err()
    );
    assert_eq!(script.sent().len(), 1);
    let first = first.finish().await.expect("first");
    let second = second.await.expect("second");
    assert_eq!(texts(&first), ["one"]);
    assert_eq!(texts(&second), ["two"]);
    // Sent after the first ended, so it chains it.
    assert_eq!(sent_json(&script, 1)["previous_response_id"], "resp_1");
}

#[tokio::test]
async fn a_turn_read_to_its_end_releases_the_connection_without_finish() {
    let script = Script::turns([
        vec![text_delta("msg_1", "one", 1), completed("resp_1", 2)],
        vec![completed("resp_2", 1)],
    ]);
    let model = model(&script);
    let mut stream = model.stream("first").expect("stream opens");
    // Read up to the text part's end, then hold the stream unfinished.
    loop {
        let item = stream.next().await.expect("an item").expect("ok");
        if let Item::Event(StreamEvent::End { .. }) = item {
            break;
        }
    }
    let second = tokio::time::timeout(Duration::from_secs(1), model.call("second"))
        .await
        .expect("the connection was released at the first turn's end")
        .expect("second");
    assert_eq!(raw_response(&second).id, "resp_2");
    drop(stream);
}

// Dropped turns.

#[tokio::test]
async fn dropped_stream_is_drained_before_the_next_turn() {
    let script = Script::turns([
        vec![
            text_delta("msg_1", "one ", 1),
            text_delta("msg_1", "more", 2),
            completed("resp_1", 3),
        ],
        vec![text_delta("msg_2", "two", 1), completed("resp_2", 2)],
    ]);
    let model = chaining(&script);
    let mut stream = model.stream("first").expect("stream opens");
    stream.next().await.expect("an item").expect("first item");
    drop(stream);

    let second = model.call("second").await.expect("second");
    assert_eq!(texts(&second), ["two"]);
    assert_eq!(raw_response(&second).id, "resp_2");
    // The caller never saw the dropped turn's response: nothing continues it.
    assert!(
        sent_json(&script, 1).get("previous_response_id").is_none(),
        "{}",
        script.sent()[1]
    );
}

#[tokio::test]
async fn a_stream_dropped_before_its_first_poll_sends_nothing() {
    let script = Script::turn([completed("resp_1", 1)]);
    let model = model(&script);
    drop(model.stream("never sent").expect("stream opens"));
    assert!(script.sent().is_empty());
    let reply = model.call("sent").await.expect("the next turn runs");
    assert_eq!(raw_response(&reply).id, "resp_1");
    assert_eq!(script.sent().len(), 1);
}

#[tokio::test]
async fn draining_a_dropped_turn_is_bounded() {
    // The dropped turn never ends and there is no event timeout: only the
    // drain timeout ends the wait.
    let script = Script::turn([text_delta("msg_1", "one", 1)]).stalling();
    let model = with_transport(transport(&script).drain_timeout(Duration::from_millis(50)));
    let mut stream = model.stream("first").expect("stream opens");
    stream.next().await.expect("an item").expect("first item");
    drop(stream);

    let error = tokio::time::timeout(Duration::from_secs(1), model.call("second"))
        .await
        .expect("the drain is bounded")
        .expect_err("the drain timed out");
    assert!(
        error.to_string().contains("dropped OpenAI websocket turn"),
        "{error}"
    );
    assert_eq!(script.sent().len(), 1, "the second turn was not sent");
    let closed = model.call("third").await.expect_err("closed");
    assert!(closed.to_string().contains("session is closed"), "{closed}");
}

// The request.

#[tokio::test]
async fn http_only_flags_are_not_sent() {
    let script = Script::turn([completed("resp_1", 1)]);
    let model = model(&script);
    let mut request = CompletionRequest::new("hello");
    request.additional_params = Some(serde_json::json!({ "background": true }));
    model
        .stream(request)
        .expect("stream opens")
        .finish()
        .await
        .expect("streamed");
    let sent = sent_json(&script, 0);
    assert_eq!(sent["type"], "response.create");
    assert!(sent.get("stream").is_none(), "{sent}");
    assert!(sent.get("background").is_none(), "{sent}");
    assert!(sent.get("generate").is_none(), "{sent}");
}

#[tokio::test]
async fn warmup_sends_generate_false() {
    let script = Script::turn([completed("resp_warm", 1)]);
    let model = Model::new(
        ResponsesSocket::new(test_client().wire).warmup(),
        transport(&script),
    );
    let response = model.call("prepare").await.expect("warmup");
    assert_eq!(response.response_id.as_deref(), Some("resp_warm"));
    assert_eq!(sent_json(&script, 0)["generate"], false);
}

/// A turn that states its tool call only in the terminal body (no item
/// events) still delivers the call.
#[tokio::test]
async fn completed_turn_without_item_events_delivers_its_tool_call() {
    let mut body = with_id("resp_1", ResponseStatus::Completed);
    body.output = vec![
        serde_json::from_value(serde_json::json!({
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "add",
            "arguments": "{\"x\":1,\"y\":2}",
            "status": "completed"
        }))
        .expect("function call output"),
    ];
    let script = Script::turn([response_event("response.completed", body, 1)]);
    let response = model(&script).call("hello").await.expect("completed");
    assert!(
        response
            .choice
            .iter()
            .any(|content| matches!(content, AssistantContent::ToolCall(call) if call.function.name.as_str() == "add")),
        "{:?}",
        response.choice
    );
}

/// A frame that is no event fails its turn, and the rest of that turn is
/// drained rather than read as the next turn.
#[tokio::test]
async fn a_frame_without_a_type_leaves_its_turn_to_be_drained() {
    let script = Script::turns([
        vec![
            text_delta("msg_1", "one", 1),
            "not an event".to_string(),
            text_delta("msg_1", "stale", 3),
            completed("resp_1", 4),
        ],
        vec![text_delta("msg_2", "two", 1), completed("resp_2", 2)],
    ]);
    let model = chaining(&script);
    model.call("first").await.expect_err("a frame with no type");
    let second = model.call("second").await.expect("second");
    assert_eq!(texts(&second), ["two"]);
    assert_eq!(raw_response(&second).id, "resp_2");
    assert!(sent_json(&script, 1).get("previous_response_id").is_none());
}

/// A streamed terminal the decoder rejects (no `sequence_number`) fails the
/// turn in the decoder, so its response is not continued.
#[tokio::test]
async fn a_streamed_terminal_the_decoder_rejects_is_not_chained() {
    let terminal = serde_json::json!({
        "type": "response.completed",
        "response": serde_json::to_value(with_id("resp_1", ResponseStatus::Completed))
            .expect("serializes"),
    })
    .to_string();
    let script = Script::turns([
        vec![text_delta("msg_1", "one", 1), terminal],
        vec![completed("resp_2", 1)],
    ]);
    let model = chaining(&script);
    model
        .call("first")
        .await
        .expect_err("the decoder rejects the terminal");
    model.call("second").await.expect("second");
    assert!(sent_json(&script, 1).get("previous_response_id").is_none());
}
