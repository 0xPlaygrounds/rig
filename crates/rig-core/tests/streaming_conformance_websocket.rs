//! Wire-conformance suite for the `openai_responses_websocket` family.
//!
//! End-to-end over the REAL tungstenite backend and a local websocket server:
//! this is the suite that proves the bundled backend and the websocket
//! transport deliver the wire faithfully. The protocol itself is tested
//! against an in-memory connection in `ws_transport`.
//!
//! The frames are the shared OpenAI Responses fixture's, re-wrapped as one
//! JSON websocket message per SSE `data:` line: the wire events are identical
//! across the two transports, only the framing differs. The driver streams
//! one observed turn of the websocket model and drains it as every other
//! wire's driver does.

#![cfg(not(target_family = "wasm"))]
use futures::{SinkExt, StreamExt};
use rig_core::completion::CompletionRequest;
use rig_core::error::ProviderError;
use rig_core::providers::openai::OpenAIConfig;
use rig_core::test_utils::RecordingHttpClient;
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
        // The transport always sends `response.create` before reading events.
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

            // The HTTP transport is never used: the websocket model only
            // wraps this model's wire.
            let bound = OpenAIConfig::new("test-key")
                .with_base_url(format!("http://{address}/v1"))
                .connect(RecordingHttpClient::new("{}"))
                .responses("gpt-5.4");
            let model = bound.responses_websocket().connect().await?;
            conformance::fixtures::drain_observed(&model, CompletionRequest::new("hello")).await
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
