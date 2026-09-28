//! The bundled backend must work when the caller has no tokio runtime.
//!
//! Bevy task pools, smol and `futures::executor` are the cases this exists for.
//! A websocket differs from a unary request in living long enough that the
//! socket cannot simply be driven per-call: it moves onto the fallback runtime
//! as an actor, and the caller polls only `futures` channels. That is invisible
//! in an ordinary tokio test, so this drives whole turns (connect, send,
//! receive, close) with `futures::executor::block_on` and no tokio runtime on
//! the calling thread.

#![cfg(not(target_family = "wasm"))]
#![allow(clippy::expect_used, clippy::panic)]

use futures::StreamExt;
use rig_core::providers::openai::OpenAIConfig;
use rig_core::streaming::{Item, StreamEvent};

use rig_core::test_utils::RecordingHttpClient;

use std::sync::mpsc;
use std::time::Duration;

fn block_on<F: std::future::Future>(future: F) -> F::Output {
    futures::executor::block_on(rig_core::wasm_compat::timeout(
        std::time::Duration::from_secs(10),
        future,
    ))
    .expect("client operation deadline")
}

/// Serve websocket turns on their own tokio runtime, on their own thread:
/// the server needs a reactor even though the client under test must not
/// have one. Each turn's events answer one `response.create`.
///
/// When `release` is given, the first turn's events wait for it, so a
/// cancelled read cannot race a sleeping server waking up on a loaded
/// machine. A turn with no events accepts the request and goes quiet.
fn serve_turns(
    release: Option<futures::channel::oneshot::Receiver<()>>,
    turns: Vec<Vec<String>>,
) -> String {
    use futures::SinkExt;

    let (address_tx, address_rx) = mpsc::channel();
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("server runtime should build");
        runtime.block_on(async move {
            let _ = tokio::time::timeout(Duration::from_secs(30), async move {
                let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                    .await
                    .expect("bind");
                address_tx
                    .send(listener.local_addr().expect("address"))
                    .expect("address should send");

                let (stream, _) = listener.accept().await.expect("accept");
                let mut socket = rig_tungstenite::tokio_tungstenite::accept_async(stream)
                    .await
                    .expect("upgrade");

                let mut release = release;
                for events in turns {
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
                        "the transport should open the turn with response.create"
                    );
                    if let Some(release) = release.take() {
                        release.await.expect("release events");
                    }
                    for event in events {
                        socket
                            .send(
                                rig_tungstenite::tokio_tungstenite::tungstenite::Message::text(
                                    event,
                                ),
                            )
                            .await
                            .expect("event should send");
                    }
                }

                // Wait for the client's close handshake so the assertion below is
                // about a completed round trip, not a race.
                while let Some(Ok(message)) = socket.next().await {
                    if message.is_close() {
                        break;
                    }
                }
            })
            .await;
        });
    });

    let address = address_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("server should report its address");
    format!("http://{address}/v1")
}

fn completed(id: &str) -> String {
    serde_json::json!({
        "type": "response.completed",
        "sequence_number": 2,
        "response": {
            "id": id,
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
    .to_string()
}

fn delta(text: &str) -> String {
    serde_json::json!({
        "type": "response.output_text.delta",
        "content_index": 0,
        "delta": text,
        "item_id": "msg_1",
        "logprobs": [],
        "output_index": 0,
        "sequence_number": 1
    })
    .to_string()
}

fn bound(
    base_url: &str,
) -> rig_core::driver::Model<rig_core::providers::openai::responses_api::wire::Responses> {
    OpenAIConfig::new("test-key")
        .with_base_url(base_url)
        .connect(RecordingHttpClient::new("{}"))
        .responses("gpt-5.4")
}

#[test]
fn a_whole_turn_streams_without_a_tokio_runtime() {
    let base_url = serve_turns(
        None,
        vec![vec![delta("off runtime"), completed("resp_off_runtime")]],
    );

    // No tokio runtime on this thread: everything below is driven by the
    // `futures` executor.
    assert!(
        tokio::runtime::Handle::try_current().is_err(),
        "this test is meaningless inside a tokio runtime"
    );

    block_on(async move {
        let model = match bound(&base_url).responses_websocket().connect().await {
            Ok(model) => model,
            Err(error) => panic!("the transport should connect off-runtime: {error}"),
        };

        let mut stream = model.stream("hello").expect("stream opens");
        let mut fragments = Vec::new();
        while let Some(item) = stream.next().await {
            if let Item::Event(StreamEvent::Text { text, .. }) = item.expect("item") {
                fragments.push(text);
            }
        }
        let response = stream
            .finish()
            .await
            .expect("the turn should complete off-runtime");

        assert_eq!(fragments, ["off runtime"]);
        assert_eq!(response.response_id.as_deref(), Some("resp_off_runtime"));
        model.transport.close().await.expect("close should succeed");
    });
}

/// The off-runtime path must stay usable after an event timeout.
///
/// A serial connection actor deadlocks here: the timed-out read leaves it
/// parked on the socket, and the `close()` that follows waits forever for a
/// frame that is never coming. The in-memory transport tests cannot assert
/// this, because a scripted connection's `close()` always resolves. So it is
/// asserted here, where the real actor is.
///
/// Every await is bounded: a regression must fail this test, not hang it.
#[test]
fn an_event_timeout_still_allows_close_without_a_tokio_runtime() {
    // The server accepts the `response.create` and then says nothing.
    let base_url = serve_turns(None, vec![Vec::new()]);

    assert!(
        tokio::runtime::Handle::try_current().is_err(),
        "this test is meaningless inside a tokio runtime"
    );

    block_on(async move {
        let model = match bound(&base_url)
            .responses_websocket()
            .event_timeout(Duration::from_millis(50))
            .connect()
            .await
        {
            Ok(model) => model,
            Err(error) => panic!("the transport should connect off-runtime: {error}"),
        };

        let error = model
            .call("hello")
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
        rig_core::wasm_compat::timeout(Duration::from_secs(5), model.transport.close())
            .await
            .expect("close() must not hang after an event timeout")
            .expect("close should succeed");
    });
}

/// A cancelled read must not swallow the frame the actor already took off
/// the socket: the next read has to see it.
///
/// Dropping a stream mid-turn abandons its read. The dropped turn's
/// `response.completed` then arrives, and the next turn has to read it while
/// draining: if the actor lost it, the drain would wait for a terminal that
/// never comes.
#[test]
fn a_cancelled_read_does_not_lose_the_frame_off_runtime() {
    // The server holds the first turn's terminal back, so the read below is
    // provably abandoned before the frame exists: no timing race either way.
    let (release, released) = futures::channel::oneshot::channel();
    let base_url = serve_turns(
        Some(released),
        vec![vec![completed("resp_1")], vec![completed("resp_2")]],
    );

    block_on(async move {
        let model = match bound(&base_url)
            .responses_websocket()
            .drain_timeout(Duration::from_secs(5))
            .connect()
            .await
        {
            Ok(model) => model,
            Err(error) => panic!("the transport should connect off-runtime: {error}"),
        };

        // The turn is sent and its read reaches the actor, then the caller
        // gives up on the stream.
        let mut stream = model.stream("first").expect("stream opens");
        let cancelled =
            rig_core::wasm_compat::timeout(Duration::from_millis(20), stream.next()).await;
        assert!(cancelled.is_err(), "the read should still be waiting");
        drop(stream);
        release
            .send(())
            .expect("release the terminal after cancellation");

        let second = rig_core::wasm_compat::timeout(Duration::from_secs(8), model.call("second"))
            .await
            .expect("the drain must not wait for a frame that already arrived")
            .expect("the second turn completes");
        assert_eq!(second.response_id.as_deref(), Some("resp_2"));
        model.transport.close().await.expect("close should succeed");
    });
}
