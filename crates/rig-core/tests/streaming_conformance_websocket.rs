//! Wire-conformance suite for the `openai_responses_websocket` family.
//!
//! End-to-end over the REAL tungstenite backend and a local websocket server:
//! this is the suite that proves the bundled backend delivers the wire
//! faithfully. The protocol itself is tested against an in-memory connection in
//! rig-core.
//!
//! The frames are the shared OpenAI Responses fixture's, re-wrapped as one
//! JSON websocket message per SSE `data:` line — the wire events are identical
//! across the two transports, only the framing differs. The driver runs the
//! REAL session pipeline (`ResponsesWebSocketSession::next_event` over a local
//! ws server) and replays the observed events through the Responses wire's
//! decoder via `drain_openai_responses_websocket_events`.
//!
//! A corrupt frame fails the websocket session (`fail_session`) and ends the
//! turn, as a corrupt frame ends every reply.

#![cfg(not(target_family = "wasm"))]
use futures::{SinkExt, StreamExt};
use rig_core::completion::CompletionRequest;
use rig_core::error::ProviderError;
use rig_core::providers::openai::OpenAIConfig;
use rig_core::providers::openai::responses_api::websocket::ResponsesWebSocketEvent;
use rig_core::test_utils::{RecordingHttpClient, SequencedStreamingHttpClient};
use rig_core::test_utils::streaming_conformance::{
    self as conformance, fixtures::openai_responses,
};

use rig_tungstenite::tokio_tungstenite::{accept_async, tungstenite::Message};
use tokio::net::TcpListener;

/// Lower the fixture's byte frames onto ws text messages (one per `data:`
/// line); an `Err` chunk truncates the script and marks an abrupt abort.
fn ws_script(chunks: conformance::WireChunks) -> Result<(Vec<String>, bool), ProviderError> {
    let mut messages = Vec::new();
    for chunk in chunks {
        match chunk {
            Ok(frame) => {
                let bytes = frame.as_bytes().cloned().ok_or_else(|| {
                    ProviderError::Provider(
                        "typed-event frame fed to the websocket driver".to_string(),
                    )
                })?;
                let text = std::str::from_utf8(&bytes).map_err(|error| {
                    ProviderError::Provider(format!("non-UTF-8 fixture frame: {error}"))
                })?;
                messages.extend(
                    text.lines()
                        .filter_map(|line| line.strip_prefix("data:").map(str::trim))
                        .filter(|data| !data.is_empty() && *data != "[DONE]")
                        .map(ToOwned::to_owned),
                );
            }
            // A scripted transport failure: everything after it is undeliverable.
            Err(_) => return Ok((messages, true)),
        }
    }
    Ok((messages, false))
}

/// Serve one websocket turn: upgrade, read the `response.create` request,
/// send the scripted messages, then end the connection — abruptly (no close
/// handshake, the client observes a transport reset) when `abort`, cleanly
/// otherwise.
fn spawn_server(listener: TcpListener, messages: Vec<String>, abort: bool) {
    tokio::spawn(async move {
        let Ok((stream, _)) = listener.accept().await else {
            return;
        };
        let Ok(mut socket) = accept_async(stream).await else {
            return;
        };
        // The session always sends `response.create` before reading events.
        let _ = socket.next().await;
        for message in messages {
            if socket.send(Message::text(message)).await.is_err() {
                return;
            }
        }
        if abort {
            drop(socket);
        } else {
            let _ = socket.close(None).await;
        }
    });
}

/// Drain one OpenAI Responses *websocket* turn's server events into
/// everything a streaming consumer would observe: each event the session
/// read, re-serialized as the frame it arrived as, through the SAME
/// Responses decoder and fold the session's `completion` runs. A session
/// failure ends the frames, as a transport failure ends a stream.
async fn drain_openai_responses_websocket_events(
    events: Vec<Result<ResponsesWebSocketEvent, ProviderError>>,
) -> Result<conformance::DrainedStream, ProviderError> {
    let mut body = String::new();
    let mut failed = false;
    for event in events {
        let frame = match event {
            Ok(ResponsesWebSocketEvent::Item(chunk)) => serde_json::to_string(&chunk)?,
            Ok(ResponsesWebSocketEvent::Response(chunk)) => serde_json::to_string(&chunk)?,
            Ok(ResponsesWebSocketEvent::Unknown(value)) => value.value().to_string(),
            // `response.done` is a websocket-only trailer the fixtures never
            // script.
            Ok(ResponsesWebSocketEvent::Done(_)) => continue,
            Ok(ResponsesWebSocketEvent::Error(_)) | Err(_) => {
                failed = true;
                break;
            }
        };
        body.push_str(&format!("data: {frame}\n\n"));
    }
    let mut chunks = vec![Ok(bytes::Bytes::from(body))];
    if failed {
        chunks.push(Err(rig_core::http_client::Error::StreamEnded));
    }
    let model = OpenAIConfig::new("test-key")
        .connect(SequencedStreamingHttpClient::new(chunks))
        .responses("gpt-5.4");
    let stream = model.stream(CompletionRequest::new("hello"))?;
    Ok(conformance::fixtures::drain(stream).await)
}

fn driver() -> conformance::WireDriver {
    conformance::WireDriver::new("openai-responses-websocket", |chunks| {
        Box::pin(async move {
            let (messages, abort) = ws_script(chunks)?;
            let listener = TcpListener::bind("127.0.0.1:0").await.map_err(|error| {
                ProviderError::Provider(format!("listener bind failed: {error}"))
            })?;
            let address = listener.local_addr().map_err(|error| {
                ProviderError::Provider(format!("listener address failed: {error}"))
            })?;
            spawn_server(listener, messages, abort);

            // The HTTP transport is never used: a websocket session only
            // borrows the wire for its request mapping.
            let bound = OpenAIConfig::new("test-key")
                .with_base_url(format!("http://{address}/v1"))
                .connect(RecordingHttpClient::new("{}"))
                .responses("gpt-5.4");
            let mut session = bound.responses_websocket().connect().await?;
            session.send(CompletionRequest::new("hello")).await?;

            // Collect the turn exactly as the production session loop does:
            // stop at the first terminal event or session error.
            let mut events = Vec::new();
            loop {
                match session.next_event().await {
                    Ok(event) => {
                        let terminal = event.is_terminal();
                        events.push(Ok(event));
                        if terminal {
                            break;
                        }
                    }
                    Err(error) => {
                        events.push(Err(error));
                        break;
                    }
                }
            }

            drain_openai_responses_websocket_events(events).await
        })
    })
}

fn fixture() -> conformance::ProviderWireFixture {
    conformance::ProviderWireFixture {
        driver: driver(),
        ..openai_responses::fixture()
    }
}

pub mod openai_responses_websocket_suite {
    use super::*;

    rig_core::streaming_conformance_suite! {
        provider: "openai_responses_websocket",
        fixture: fixture(),
        manifest: [partial_tool_args, zero_usage_terminal, malformed_frame, unknown_event_frame, defective_known_frame, refusal],
    }
}

/// Compile-linked manifest of the wire families this binary covers.
///
/// This suite lives outside the `rig` facade's `core` test binary, so the
/// workspace registry cannot link it: it lists `openai_responses_websocket` in
/// `OUT_OF_BINARY_FAMILIES` and relies on the "Test out-of-facade streaming
/// conformance and structural guards" CI step to execute this binary. The test
/// below keeps the family name honest at the definition site, which is the
/// direction the registry loses for out-of-binary suites (#2258 F3).
const SUITE_FAMILIES: &[&str] = &[openai_responses_websocket_suite::WIRE_FAMILY];

#[test]
fn suite_families_are_registered_wire_families() {
    for family in SUITE_FAMILIES {
        assert!(
            rig_core::test_utils::streaming_conformance::WIRE_FAMILIES.contains(family),
            "suite names wire family {family:?}, absent from WIRE_FAMILIES"
        );
    }
}
