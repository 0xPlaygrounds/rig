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

#![cfg(not(target_family = "wasm"))]
#![allow(clippy::expect_used)]

use futures::{SinkExt, StreamExt};
use rig_core::client::CompletionClient as _;
use rig_core::completion::{CompletionError, CompletionModel as _};
use rig_core::providers::openai::responses_api::websocket::ResponsesWebSocketEvent;
use rig_core::test_utils::RecordingHttpClient;
use rig_core::test_utils::streaming_conformance::{
    self as conformance, fixtures::openai_responses,
};
use rig_tungstenite::DefaultWebSocketClient as _;
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

/// Drain one OpenAI Responses *websocket* turn's server events into
/// everything a streaming consumer would observe, through the SAME decode
/// state machine the production session drives
/// (`RawChoiceAccumulator` + `normalize_responses_stream`).
///
/// The websocket pipeline is request/response: `next_event` has no in-band
/// `Err` channel, so the caller collects events (stopping at the first
/// terminal or session error) and this helper replays them. One policy the
/// helper supplies that the buffered session cannot: tool calls the provider
/// fully delivered flush before a session error, mirroring the SSE loop's
/// flush-before-terminal-error contract (`RawChoiceAccumulator::flush_tool_calls`).
async fn drain_openai_responses_websocket_events(
    provider: &'static str,
    events: Vec<Result<ResponsesWebSocketEvent, CompletionError>>,
) -> conformance::DrainedStream {
    use ResponsesWebSocketEvent;
    use rig_core::providers::internal::adapter::AdapterOutput;
    use rig_core::providers::openai::responses_api::ResponsesUsage;
    use rig_core::providers::openai::responses_api::streaming::{
        RawChoiceAccumulator, ResponseChunkKind, ResponsesStreamOptions,
    };

    let mut accumulator = RawChoiceAccumulator::new(provider, ResponsesUsage::new());
    let mut out = AdapterOutput::new();
    let mut errored = false;
    for event in events {
        match event {
            Ok(ResponsesWebSocketEvent::Item(chunk)) => {
                accumulator.decode_item_chunk(chunk, ResponsesStreamOptions::strict(), &mut out);
            }
            Ok(ResponsesWebSocketEvent::Response(chunk)) => {
                let terminal = matches!(
                    chunk.kind,
                    ResponseChunkKind::ResponseCompleted
                        | ResponseChunkKind::ResponseFailed
                        | ResponseChunkKind::ResponseIncomplete
                );
                if let Err(error) =
                    accumulator.record_response_chunk(chunk.kind, chunk.response, "")
                {
                    accumulator.flush_tool_calls(&mut out);
                    out.error(error);
                    errored = true;
                    break;
                }
                if terminal {
                    break;
                }
            }
            // Semantic skip, raw passthrough: an unknown frame never reaches
            // the accumulator but is still yielded verbatim.
            Ok(ResponsesWebSocketEvent::Unknown(value)) => out.unknown(value),
            // `response.done` / `error` envelopes are websocket-only shapes the
            // fixtures never script; the production session maps them to a
            // terminal or a provider error before this replay runs.
            Ok(ResponsesWebSocketEvent::Done(_)) => {}
            Ok(ResponsesWebSocketEvent::Error(error)) => {
                accumulator.flush_tool_calls(&mut out);
                out.error(CompletionError::ProviderError(error.to_string()));
                errored = true;
                break;
            }
            Err(error) => {
                accumulator.flush_tool_calls(&mut out);
                out.error(error);
                errored = true;
                break;
            }
        }
    }
    if !errored {
        accumulator.finish(&mut out);
    }

    let stream = rig_core::streaming::StreamingCompletionResponse::stream(
        provider,
        Box::pin(futures::stream::iter(out.into_items())),
    );
    conformance::fixtures::drain(stream).await
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

            // The HTTP transport is never used: a websocket session only
            // borrows the model for its request mapping.
            let client = rig_core::providers::openai::Client::builder()
                .api_key("test-key")
                .base_url(format!("http://{address}/v1"))
                .http_client(RecordingHttpClient::new("{}"))
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

            Ok(drain_openai_responses_websocket_events("openai-responses-websocket", events).await)
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
