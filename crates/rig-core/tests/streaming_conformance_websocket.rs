//! Wire-conformance suite for the `openai_responses_websocket` family.
//!
//! The frames are the shared OpenAI Responses fixture's, re-wrapped as one
//! JSON websocket message per SSE `data:` line — the wire events are identical
//! across the two transports, only the framing differs. The driver runs the
//! REAL session pipeline (`ResponsesWebSocketSession::next_event` over a local
//! ws server) and replays the observed events through the shared
//! `RawChoiceAccumulator` + normalization via
//! `drain_openai_responses_websocket_events`.
//!
//! The websocket turn is request/response: a corrupt frame fails the whole
//! session (`fail_session`) instead of surfacing an in-band `Err` item beside
//! a still-completing terminal, so the two defective-frame scenarios are
//! sanctioned `xfail`s rather than capability gaps — the wire CAN spell the
//! frames; the pipeline's policy differs by design (documented in
//! MIGRATING.md, #2258).

#![cfg(all(
    not(target_family = "wasm"),
    feature = "websocket",
    feature = "test-utils"
))]

use futures::{SinkExt, StreamExt};
use rig_core::client::CompletionClient as _;
use rig_core::completion::{CompletionError, CompletionModel as _};
use rig_core::test_utils::streaming_conformance::{
    self as conformance, fixtures::openai_responses,
};
use tokio::net::TcpListener;
use tokio_tungstenite::{accept_async, tungstenite::Message};

/// Lower the fixture's byte frames onto ws text messages (one per `data:`
/// line); an `Err` chunk truncates the script and marks an abrupt abort.
fn ws_script(chunks: conformance::WireChunks) -> Result<(Vec<String>, bool), CompletionError> {
    let mut messages = Vec::new();
    for chunk in chunks {
        match chunk {
            Ok(frame) => {
                let bytes = frame.as_bytes().cloned().ok_or_else(|| {
                    CompletionError::ProviderError(
                        "typed-event frame fed to the websocket driver".to_string(),
                    )
                })?;
                let text = std::str::from_utf8(&bytes).map_err(|error| {
                    CompletionError::ProviderError(format!("non-UTF-8 fixture frame: {error}"))
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

fn driver() -> conformance::WireDriver {
    conformance::WireDriver::new("openai-responses-websocket", |chunks| {
        Box::pin(async move {
            let (messages, abort) = ws_script(chunks)?;
            let listener = TcpListener::bind("127.0.0.1:0").await.map_err(|error| {
                CompletionError::ProviderError(format!("listener bind failed: {error}"))
            })?;
            let address = listener.local_addr().map_err(|error| {
                CompletionError::ProviderError(format!("listener address failed: {error}"))
            })?;
            spawn_server(listener, messages, abort);

            let client = rig_core::providers::openai::Client::builder()
                .api_key("test-key")
                .base_url(format!("http://{address}/v1"))
                .build()
                .map_err(|error| CompletionError::ProviderError(error.to_string()))?;
            let model = client.completion_model("gpt-5.4");
            let mut session = client.responses_websocket("gpt-5.4").await?;
            session
                .send(model.completion_request("hello").build())
                .await?;

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

            Ok(conformance::drain_openai_responses_websocket_events(
                "openai-responses-websocket",
                events,
            )
            .await)
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
        xfail: [
            "malformed_frame_surfaces_err_and_terminal_still_completes: the websocket turn is request/response — a corrupt frame fails the whole session, there is no in-band Err channel beside a completing terminal (#2258 unification review)",
            "defective_known_event_surfaces_err: the websocket turn is request/response — a schema-defective known frame fails the whole session instead of surfacing an in-band Err (#2258 unification review)",
        ],
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
