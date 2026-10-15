//! The OpenAI Responses websocket transport, driven over a scripted
//! in-memory connection.
//!
//! These are protocol tests: turn lifecycle, queued turns, the drain of a
//! turn the caller dropped, `previous_response_id` chaining, the trailing
//! `response.done` filter, events from another response, and what a
//! connection does after a fatal event. None of that involves a socket, so
//! the script supplies the frames. The real backend is exercised end to end
//! in `streaming_conformance_websocket` and `websocket_off_runtime`.

#![cfg(feature = "websocket")]
#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

#[path = "common/websocket_script.rs"]
mod websocket_script;

use futures::StreamExt;
use rig_core::completion::{AssistantContent, CompletionResponse};
use rig_core::driver::DynModel;
use rig_core::observe::{
    Action, AdapterContext, AdapterEnding, AdapterEvent, ObservationLog, Subject,
};
use rig_core::operation::Completion;
use rig_core::streaming::Item;
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;
use websocket_script::{Script, chained_socket, socket, socket_with, test_client};

fn response(id: &str, status: &str) -> serde_json::Value {
    json!({
        "id": id,
        "object": "response",
        "created_at": 0,
        "status": status,
        "model": "gpt-4o",
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

fn event(kind: &str, id: &str, status: &str) -> String {
    json!({ "type": kind, "response": response(id, status) }).to_string()
}

fn delta(text: &str) -> String {
    json!({
        "type": "response.output_text.delta",
        "output_index": 0,
        "content_index": 0,
        "item_id": "msg_1",
        "delta": text,
    })
    .to_string()
}

/// A turn that answers `text` as `id`, then the trailing `response.done`
/// OpenAI may send after it.
fn answered(id: &str, text: &str) -> Vec<String> {
    vec![
        event("response.created", id, "in_progress"),
        delta(text),
        event("response.completed", id, "completed"),
        json!({ "type": "response.done", "response": { "id": id, "status": "completed" } })
            .to_string(),
    ]
}

fn text(response: &CompletionResponse) -> String {
    response
        .choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}

fn previous_response_id(sent: &str) -> Option<String> {
    let sent: serde_json::Value = serde_json::from_str(sent).expect("sent JSON");
    sent["previous_response_id"].as_str().map(str::to_owned)
}

/// The shared Responses fixture's frames, as websocket messages: one per SSE
/// `data:` line. The events are the same over both transports.
fn fixture_messages() -> Vec<String> {
    let fixture =
        rig_core::test_utils::streaming_conformance::fixtures::openai_responses::fixture();
    fixture
        .text_frames
        .iter()
        .chain(&fixture.tool_call_frames)
        .chain(&fixture.unknown_event_frame)
        .chain(&fixture.terminal_frames)
        .flat_map(|frame| {
            let bytes = frame.as_bytes().cloned().expect("byte frames");
            std::str::from_utf8(&bytes)
                .expect("UTF-8 frames")
                .lines()
                .filter_map(|line| line.strip_prefix("data:").map(str::trim))
                .filter(|data| !data.is_empty() && *data != "[DONE]")
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>()
        })
        .collect()
}

/// A turn yields the same items and response as the same events over SSE,
/// `call` is `stream().finish()`, and the model erases to a `DynModel`.
#[tokio::test]
async fn websocket_and_sse_yield_the_same_items_and_response() {
    let messages = fixture_messages();
    let sse: String = messages
        .iter()
        .map(|message| format!("data: {message}\n\n"))
        .collect();
    let http = rig_core::providers::openai::OpenAIConfig::new("test-key")
        .connect(rig_core::test_utils::SequencedStreamingHttpClient::new(
            vec![Ok(bytes::Bytes::from(sse))],
        ))
        .responses("gpt-4o");
    let mut over_http = http.stream("hello").expect("opens");
    let mut http_items = Vec::new();
    while let Some(item) = over_http.next().await {
        http_items.push(format!("{:?}", item.expect("item")));
    }
    let http_response = over_http.finish().await.expect("folds");

    let script = Script::turns([messages.clone(), messages]);
    let client = test_client();
    let socket = socket(&client, &script);
    let mut over_socket = socket.stream("hello").expect("opens");
    let mut socket_items = Vec::new();
    while let Some(item) = over_socket.next().await {
        socket_items.push(format!("{:?}", item.expect("item")));
    }
    let streamed = over_socket.finish().await.expect("folds");
    assert_eq!(socket_items, http_items);
    assert_eq!(streamed.choice, http_response.choice);
    assert_eq!(streamed.usage, http_response.usage);
    assert_eq!(streamed.raw, http_response.raw);

    let erased: DynModel<Completion> = socket.into();
    let called = erased.call("hello").await.expect("calls");
    assert_eq!(called.choice, streamed.choice);
    let sent = script.sent();
    assert_eq!(sent.len(), 2);
    assert!(
        sent.iter()
            .all(|sent| sent.contains("\"type\":\"response.create\""))
    );
}

/// A turn sent while another holds the connection waits for it, where the
/// session used to refuse it.
#[tokio::test]
async fn concurrent_turns_queue_on_one_connection() {
    let script = Script::turns([answered("resp_1", "one"), answered("resp_2", "two")]);
    let client = test_client();
    let socket = socket(&client, &script);

    let (first, second) = tokio::join!(socket.call("first"), socket.call("second"));
    let mut answers = vec![
        text(&first.expect("first turn")),
        text(&second.expect("second turn")),
    ];
    answers.sort();
    assert_eq!(answers, ["one", "two"]);
    assert_eq!(script.sent().len(), 2);
}

/// A dropped turn is read to its end before the next is sent, so none of its
/// events reach the next turn, and the chain does not continue a response
/// the caller never saw.
#[tokio::test]
async fn a_dropped_turn_is_drained_before_the_next() {
    let script = Script::turns([answered("resp_1", "dropped"), answered("resp_2", "kept")]);
    let client = test_client();
    let socket = chained_socket(&client, &script);

    let mut dropped = socket.stream("first").expect("opens");
    let _ = dropped.next().await;
    drop(dropped);

    let kept = socket.call("second").await.expect("the next turn runs");
    assert_eq!(text(&kept), "kept");
    assert_eq!(kept.origin.response_id.as_deref(), Some("resp_2"));
    assert_eq!(previous_response_id(&script.sent()[1]), None);
}

/// A dropped turn that never ends costs the connection after the drain
/// timeout.
#[tokio::test]
async fn a_drain_past_its_timeout_fails_the_connection() {
    let script = Script::turn([
        event("response.created", "resp_1", "in_progress"),
        delta("never ends"),
    ])
    .stalling();
    let client = test_client();
    let socket = socket_with(&client, &script, |transport| {
        transport.drain_timeout(Duration::from_millis(20))
    });

    let mut dropped = socket.stream("first").expect("opens");
    let _ = dropped.next().await;
    drop(dropped);

    let error = socket
        .call("second")
        .await
        .expect_err("the drain times out");
    assert!(error.to_string().contains("Timed out reading"), "{error}");
    let error = socket
        .call("third")
        .await
        .expect_err("the connection failed");
    assert!(error.to_string().contains("session is closed"), "{error}");
}

/// Chaining continues the last response, its trailing `response.done` never
/// reaches the next turn, and clearing the chain keeps that filter.
#[tokio::test]
async fn chaining_continues_the_last_response_and_late_done_is_ignored() {
    let script = Script::turns([
        answered("resp_1", "one"),
        answered("resp_2", "two"),
        answered("resp_3", "three"),
    ]);
    let client = test_client();
    let socket = chained_socket(&client, &script);

    assert_eq!(text(&socket.call("first").await.expect("first")), "one");
    assert_eq!(
        socket.transport.last_response_id().as_deref(),
        Some("resp_1")
    );

    let mut second = socket.stream("second").expect("opens");
    while let Some(item) = second.next().await {
        assert!(
            !matches!(item.expect("item"), Item::Unknown(_)),
            "the previous turn's response.done leaked"
        );
    }
    assert_eq!(text(&second.finish().await.expect("second")), "two");

    socket.transport.clear_chain();
    assert_eq!(text(&socket.call("third").await.expect("third")), "three");

    let sent = script.sent();
    assert_eq!(previous_response_id(&sent[0]), None);
    assert_eq!(previous_response_id(&sent[1]).as_deref(), Some("resp_1"));
    assert_eq!(previous_response_id(&sent[2]), None);
}

/// A failed turn fails with the provider's envelope, ends the chain, and its
/// trailing `response.done` is skipped.
#[tokio::test]
async fn a_failed_turn_resets_the_chain() {
    let mut failed = response("resp_failed", "failed");
    failed["error"] = json!({ "code": "server_error", "message": "the model failed" });
    let script = Script::turns([
        vec![
            event("response.completed", "resp_1", "completed"),
        ],
        vec![
            json!({ "type": "response.failed", "response": failed }).to_string(),
            json!({ "type": "response.done", "response": { "id": "resp_failed", "status": "failed" } })
                .to_string(),
        ],
        answered("resp_2", "two"),
    ]);
    let client = test_client();
    let socket = chained_socket(&client, &script);

    socket.call("first").await.expect("first");
    let error = socket.call("second").await.expect_err("the turn failed");
    assert!(error.to_string().contains("the model failed"), "{error}");
    assert!(error.is_retryable(), "{error}");
    assert_eq!(socket.transport.last_response_id(), None);

    assert_eq!(text(&socket.call("third").await.expect("third")), "two");
    assert_eq!(previous_response_id(&script.sent()[2]), None);
}

/// A turn that ends at `response.done` alone ends there: completed, it
/// chains; failed, it fails.
#[tokio::test]
async fn a_response_done_alone_ends_its_turn() {
    let mut failed = response("resp_failed", "failed");
    failed["error"] = json!({ "code": "server_error", "message": "done and failed" });
    let mut completed = response("resp_1", "completed");
    completed["output"] = json!([{
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{ "type": "output_text", "text": "from the body", "annotations": [] }]
    }]);
    let script = Script::turns([
        vec![json!({ "type": "response.done", "response": failed }).to_string()],
        vec![json!({ "type": "response.done", "response": completed }).to_string()],
        answered("resp_2", "two"),
    ]);
    let client = test_client();
    let socket = chained_socket(&client, &script);

    let error = socket.call("first").await.expect_err("failed");
    assert!(error.to_string().contains("done and failed"), "{error}");
    let second = socket.call("second").await.expect("completed");
    assert_eq!(text(&second), "from the body");
    socket.call("third").await.expect("third");
    assert_eq!(
        previous_response_id(&script.sent()[2]).as_deref(),
        Some("resp_1")
    );
}

/// An `error` after the turn's response opened fails the turn; whatever that
/// response still sends is drained, never read as the next turn's answer.
#[tokio::test]
async fn an_error_mid_response_drains_the_rest_of_it() {
    let error = json!({ "type": "error", "error": { "code": "server_error", "message": "mid" } });
    let mut first = vec![
        event("response.created", "resp_1", "in_progress"),
        error.to_string(),
    ];
    first.extend(answered("resp_1", "stale"));
    let script = Script::turns([first, answered("resp_2", "fresh")]);
    let client = test_client();
    let socket = socket(&client, &script);

    let error = socket
        .call("first")
        .await
        .expect_err("the error fails the turn");
    assert!(error.to_string().contains("mid"), "{error}");
    let second = socket.call("second").await.expect("second");
    assert_eq!(text(&second), "fresh");
    assert_eq!(second.origin.response_id.as_deref(), Some("resp_2"));
}

/// A terminal for another response, or a frame with no type, fails the turn,
/// and the connection reads that turn's own end before the next.
#[tokio::test]
async fn frames_that_are_no_event_of_the_turn_fail_it() {
    let mut foreign = vec![
        event("response.created", "resp_1", "in_progress"),
        event("response.completed", "resp_0", "completed"),
    ];
    foreign.extend(answered("resp_1", "stale"));
    let mut typeless = vec![
        event("response.created", "resp_2", "in_progress"),
        response("resp_2", "completed").to_string(),
    ];
    typeless.extend(answered("resp_2", "stale"));
    let script = Script::turns([foreign, typeless, answered("resp_3", "fresh")]);
    let client = test_client();
    let socket = socket(&client, &script);

    let error = socket.call("first").await.expect_err("foreign terminal");
    assert!(error.to_string().contains("another response"), "{error}");
    socket.call("second").await.expect_err("typeless frame");
    let third = socket.call("third").await.expect("third");
    assert_eq!(text(&third), "fresh");
}

/// An event timeout fails the turn and the connection; close still works,
/// and closing twice does nothing.
#[tokio::test]
async fn event_timeout_fails_the_turn_and_the_connection() {
    let script = Script::turn([]).stalling();
    let client = test_client();
    let socket = socket_with(&client, &script, |transport| {
        transport.event_timeout(Some(Duration::from_millis(20)))
    });

    let error = socket.call("hello").await.expect_err("times out");
    assert!(
        error
            .to_string()
            .contains("Timed out waiting for the next OpenAI websocket event"),
        "{error}"
    );
    let error = socket
        .call("retry")
        .await
        .expect_err("the connection failed");
    assert!(error.to_string().contains("session is closed"), "{error}");

    socket.transport.close().await.expect("close");
    socket.transport.close().await.expect("close again");
    assert!(script.closed());
    assert_eq!(script.sent().len(), 1);
}

/// A turn is one observed attempt: started on the Responses route, answered
/// by the upgrade, with the provider's facts, and finished at its end.
#[tokio::test]
async fn a_turn_is_observed_as_an_attempt() {
    let script = Script::turn(answered("resp_1", "one"));
    let client = test_client();
    let socket = socket(&client, &script);
    let log = Arc::new(ObservationLog::default());
    let context = AdapterContext::new(log.clone(), Subject::default(), "turn");

    socket
        .call_observed("hello", context)
        .await
        .expect("the turn completes");

    let events: Vec<AdapterEvent> = log
        .trace()
        .observations
        .into_iter()
        .filter_map(|observation| match observation.action {
            Action::Adapter { observation } => Some(observation.event),
            _ => None,
        })
        .collect();
    assert!(matches!(
        events.first(),
        Some(AdapterEvent::Started { method, route }) if method == "GET" && route == "/responses"
    ));
    assert!(events.contains(&AdapterEvent::Response { status: 101 }));
    assert!(
        events
            .iter()
            .any(|event| matches!(event, AdapterEvent::Provider { .. }))
    );
    assert!(matches!(
        events.last(),
        Some(AdapterEvent::Finished {
            ending: AdapterEnding::Terminal
        })
    ));
}
