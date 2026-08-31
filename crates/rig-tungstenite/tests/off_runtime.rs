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
use rig_core::test_utils::RecordingHttpClient;
use rig_tungstenite::DefaultWebSocketClient as _;
use std::sync::mpsc;

/// Serve one websocket turn on its own tokio runtime, on its own thread: the
/// server needs a reactor even though the client under test must not have one.
fn serve_one_turn(events: Vec<String>) -> String {
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
