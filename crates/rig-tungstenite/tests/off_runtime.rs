//! The bundled backend must work when the caller has no tokio runtime.
//!
//! Bevy task pools, smol and `futures::executor` are the cases this exists for.
//! A websocket differs from a unary request in living long enough that the
//! socket cannot simply be driven per-call: it moves onto the fallback runtime
//! as an actor, and the caller polls only `futures` channels. That is invisible
//! in an ordinary tokio test, so this drives a whole session — connect, send,
//! receive, close — with `futures::executor::block_on` and no tokio runtime on
//! the calling thread.

#![cfg(not(target_family = "wasm"))]
#![allow(clippy::expect_used, clippy::panic)]

use rig_core::client::CompletionClient as _;
use rig_core::completion::CompletionModel as _;
use rig_core::providers::openai::responses_api::websocket::ResponsesWebSocketExt as _;
use rig_core::test_utils::RecordingHttpClient;
use rig_tungstenite::{DefaultWebSocketBuilder as _, DefaultWebSocketClient as _};
use std::sync::mpsc;
use std::time::Duration;

/// Serve one websocket turn on its own tokio runtime, on its own thread: the
/// server needs a reactor even though the client under test must not have one.
///
/// With `events` empty the server accepts the turn and then goes quiet, which
/// is what an event timeout has to survive.
fn serve_one_turn(events: Vec<String>) -> String {
    serve_one_turn_after(Duration::ZERO, events)
}

/// [`serve_one_turn`], holding the events back for `delay` after the request
/// arrives, so a test can cancel a read while the frame is provably not yet on
/// the wire.
fn serve_one_turn_after(delay: Duration, events: Vec<String>) -> String {
    use futures::{SinkExt, StreamExt};

    let (address_tx, address_rx) = mpsc::channel();
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("server runtime should build");
        runtime.block_on(async move {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .expect("bind");
            address_tx
                .send(listener.local_addr().expect("address"))
                .expect("address should send");

            let (stream, _) = listener.accept().await.expect("accept");
            let mut socket = tokio_tungstenite::accept_async(stream)
                .await
                .expect("upgrade");

            let request = socket
                .next()
                .await
                .expect("request should arrive")
                .expect("request should be valid");
            assert!(
                request
                    .into_text()
                    .expect("request should be text")
                    .contains("\"type\":\"response.create\""),
                "the session should open the turn with response.create"
            );

            if !delay.is_zero() {
                tokio::time::sleep(delay).await;
            }

            for event in events {
                socket
                    .send(tokio_tungstenite::tungstenite::Message::text(event))
                    .await
                    .expect("event should send");
            }

            // Wait for the client's close handshake so the assertion below is
            // about a completed round trip, not a race.
            while let Some(Ok(message)) = socket.next().await {
                if message.is_close() {
                    break;
                }
            }
        });
    });

    let address = address_rx.recv().expect("server should report its address");
    format!("http://{address}/v1")
}

#[test]
fn a_whole_session_runs_without_a_tokio_runtime() {
    let completed = serde_json::json!({
        "type": "response.completed",
        "sequence_number": 2,
        "response": {
            "id": "resp_off_runtime",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "error": null,
            "incomplete_details": null,
            "instructions": null,
            "max_output_tokens": null,
            "model": "gpt-5.4",
            "usage": null,
            "output": [],
            "tools": []
        }
    })
    .to_string();
    let delta = serde_json::json!({
        "type": "response.output_text.delta",
        "content_index": 0,
        "delta": "off runtime",
        "item_id": "msg_1",
        "logprobs": [],
        "output_index": 0,
        "sequence_number": 1
    })
    .to_string();

    let base_url = serve_one_turn(vec![delta, completed]);

    // No tokio runtime on this thread: everything below is driven by the
    // `futures` executor.
    assert!(
        tokio::runtime::Handle::try_current().is_err(),
        "this test is meaningless inside a tokio runtime"
    );

    futures::executor::block_on(async move {
        let client = rig_core::providers::openai::Client::builder()
            .api_key("test-key")
            .base_url(&base_url)
            .http_client(RecordingHttpClient::new("{}"))
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-5.4");

        let mut session = match client.responses_websocket("gpt-5.4").await {
            Ok(session) => session,
            Err(error) => panic!("session should connect off-runtime: {error}"),
        };

        let response = session
            .completion(model.completion_request("hello").build())
            .await
            .expect("the turn should complete off-runtime");

        assert!(
            matches!(
                response.choice.first(),
                Some(rig_core::completion::AssistantContent::Text(text))
                    if text.text == "off runtime"
            ),
            "the streamed delta should arrive off-runtime, got {:?}",
            response.choice
        );

        session.close().await.expect("close should succeed");
    });
}

/// The off-runtime path must stay usable after an event timeout.
///
/// A serial connection actor deadlocks here: the timed-out read leaves it
/// parked on the socket, and the `close()` that follows waits forever for a
/// frame that is never coming. The pre-split suite asserted exactly this
/// against a live server (`event_timeout_rejects_reuse_and_allows_close`); its
/// in-memory replacement in rig-core cannot, because a scripted connection's
/// `close()` always resolves. So it is asserted here, where the real actor is.
///
/// Every await is bounded: a regression must fail this test, not hang it.
#[test]
fn an_event_timeout_still_allows_close_without_a_tokio_runtime() {
    // The server accepts the `response.create` and then says nothing.
    let base_url = serve_one_turn(Vec::new());

    assert!(
        tokio::runtime::Handle::try_current().is_err(),
        "this test is meaningless inside a tokio runtime"
    );

    futures::executor::block_on(async move {
        let client = rig_core::providers::openai::Client::builder()
            .api_key("test-key")
            .base_url(&base_url)
            .http_client(RecordingHttpClient::new("{}"))
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-5.4");

        let mut session = match client
            .responses_websocket_builder("gpt-5.4")
            .event_timeout(Duration::from_millis(50))
            .connect()
            .await
        {
            Ok(session) => session,
            Err(error) => panic!("session should connect off-runtime: {error}"),
        };

        session
            .send(model.completion_request("hello").build())
            .await
            .expect("request should send");

        let error = session
            .next_event()
            .await
            .expect_err("a silent server should trip the event timeout");
        assert!(
            error
                .to_string()
                .contains("Timed out waiting for the next OpenAI websocket event"),
            "expected the event timeout, got {error}"
        );

        // The regression: this used to wait on an actor still parked in the
        // read the timeout abandoned.
        rig_core::wasm_compat::timeout(Duration::from_secs(5), session.close())
            .await
            .expect("close() must not hang after an event timeout")
            .expect("close should succeed");
    });
}

/// A cancelled read must not swallow the frame the actor already took off the
/// socket: the next read has to see it.
///
/// The session itself never cancels a read except by timing out, but a host
/// that races `next_event()` in its own `select!` does, and losing a
/// `response.completed` that way hangs the following turn.
#[test]
fn a_cancelled_read_does_not_lose_the_frame_off_runtime() {
    let delta = serde_json::json!({
        "type": "response.output_text.delta",
        "content_index": 0,
        "delta": "kept",
        "item_id": "msg_1",
        "logprobs": [],
        "output_index": 0,
        "sequence_number": 1
    })
    .to_string();
    // The server holds the delta back, so the read below is provably cancelled
    // before the frame exists — no timing race in either direction.
    let base_url = serve_one_turn_after(Duration::from_millis(200), vec![delta]);

    futures::executor::block_on(async move {
        let client = rig_core::providers::openai::Client::builder()
            .api_key("test-key")
            .base_url(&base_url)
            .http_client(RecordingHttpClient::new("{}"))
            .build()
            .expect("client should build");
        let model = client.completion_model("gpt-5.4");

        let mut session = match client.responses_websocket("gpt-5.4").await {
            Ok(session) => session,
            Err(error) => panic!("session should connect off-runtime: {error}"),
        };
        session
            .send(model.completion_request("hello").build())
            .await
            .expect("request should send");

        // The command reaches the actor and is then abandoned by the caller.
        let cancelled =
            rig_core::wasm_compat::timeout(Duration::from_millis(20), session.next_event()).await;
        assert!(cancelled.is_err(), "the read should have been cancelled");

        // The delta arrives with no reader waiting; the next read must get it
        // rather than hang for a frame the actor already consumed.
        let event = rig_core::wasm_compat::timeout(Duration::from_secs(5), session.next_event())
            .await
            .expect("the next read must not hang waiting for a frame that already arrived")
            .expect("the buffered delta should be delivered");
        assert!(
            matches!(
                event,
                rig_core::providers::openai::responses_api::websocket::ResponsesWebSocketEvent::Item(_)
            ),
            "expected the delta the cancelled read had taken off the socket"
        );
    });
}
