//! Matrix for opt-in raw terminal-record capture on Bedrock's ConverseStream
//! path (`CompletionRequest::capture_raw_response` → `StreamFinal::raw`).
//!
//! # The feature
//!
//! `StreamFinal::raw` is the value
//! [`CompletionModel::raw_stream`](rig::bedrock::completion::CompletionModel::raw_stream)
//! would have yielded as its `FinalResponse` — [`BedrockStreamingResponse`]:
//! the `metadata` event's usage, the `messageStop` event's `stopReason` in
//! Bedrock's own vocabulary, and the operation's AWS request id — serialized
//! with `serde_json::to_value`. It is the terminal record only, populated only
//! when the request opted in, and never on the wire.
//!
//! Bedrock streams the AWS event-stream binary framing (recorded base64), so
//! the premise checks below decode the fixture body and locate the JSON
//! payloads of the `messageStop` and `metadata` events inside it.
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `stream_capture_off_raw_is_none` | flag off (default) | terminal `raw == None` | unrecorded (no valid AWS credentials in this environment) |
//! | 2 | `stream_capture_on_terminal_round_trips_provider_type` | flag on | `BedrockStreamingResponse::deserialize(&*raw)` re-serializes equal | unrecorded (no valid AWS credentials in this environment) |
//! | 3 | `stream_capture_on_exposes_bedrock_stop_reason` | terminal-only field | `raw.stop_reason` is Bedrock's own spelling of the recorded `stopReason`; `raw.usage.total_tokens` equals the recorded `metadata` usage | unrecorded (no valid AWS credentials in this environment) |
//! | 4 | `stream_request_invariant_off_vs_on` | on-wire request | flag-off and flag-on request bodies byte-identical | unrecorded (no valid AWS credentials in this environment) |
//!
//! Every cell is unrecorded: the `AWS_*` variables present when this matrix
//! was written carried an expired session token (`aws sts get-caller-identity`
//! failed), and a fixture is never fabricated. To record once valid
//! credentials exist: remove the `#[ignore]` attributes, flip the table to
//! `recorded`, then run
//! `RIG_PROVIDER_TEST_MODE=record cargo test -p rig --all-features --test bedrock bedrock::cassette::raw_stream_capture_matrix -- --nocapture --test-threads=1`
//! and review `tests/cassettes/bedrock/raw_stream_capture_matrix/` (the
//! streaming bodies are base64 — decode before scanning).

use base64::{Engine, prelude::BASE64_STANDARD};
use futures::StreamExt;
use rig::bedrock;
use rig::bedrock::streaming::BedrockStreamingResponse;
use rig::bedrock::types::converse_output::StopReason;
use rig::completion::CompletionModel as _;
use rig::prelude::*;
use rig::streaming::{StreamFinal, StreamedAssistantContent};
use serde::Deserialize;
use serde_json::Value;

use super::super::support::with_bedrock_cassette;
use crate::cassettes::recorded_interaction_bodies;

const BEDROCK_PROVIDER: &str = "bedrock";
const MODEL: &str = bedrock::completion::AMAZON_NOVA_LITE;
const PROMPT: &str = "Reply with exactly the single word: pong";

fn request(
    model: &bedrock::completion::CompletionModel,
    capture_raw: bool,
) -> rig::completion::CompletionRequest {
    model
        .completion_request(PROMPT)
        .temperature(0.0)
        .max_tokens(16)
        .capture_raw_response(capture_raw)
        .build()
}

async fn terminal_of(mut stream: rig::streaming::StreamingCompletionResponse) -> StreamFinal {
    let mut finals = Vec::new();
    while let Some(item) = stream.next().await {
        if let StreamedAssistantContent::Final(record) = item.expect("stream item should be ok") {
            finals.push(record);
        }
    }
    assert_eq!(
        finals.len(),
        1,
        "stream should yield exactly one terminal record"
    );
    finals.remove(0)
}

/// The recorded event-stream bytes of one interaction, base64-decoded.
fn recorded_event_stream(scenario: &str, interaction: usize) -> Vec<u8> {
    let bodies = recorded_interaction_bodies(BEDROCK_PROVIDER, scenario);
    let (_, response) = bodies
        .get(interaction)
        .unwrap_or_else(|| panic!("{scenario}: interaction {interaction} should be recorded"));
    BASE64_STANDARD
        .decode(response.trim())
        .unwrap_or_else(|err| panic!("{scenario}: streaming body should be base64: {err}"))
}

/// Finds the first JSON object embedded in the event-stream bytes for which
/// `accept` holds. Event payloads are plain JSON between binary frame headers,
/// so scanning every `{` and parsing a prefix value is enough.
fn embedded_json_object(bytes: &[u8], accept: impl Fn(&Value) -> bool) -> Option<Value> {
    bytes
        .iter()
        .enumerate()
        .filter(|(_, byte)| **byte == b'{')
        .find_map(|(start, _)| {
            serde_json::Deserializer::from_slice(&bytes[start..])
                .into_iter::<Value>()
                .next()
                .and_then(Result::ok)
                .filter(|value| value.is_object() && accept(value))
        })
}

/// The premise every streaming cell rests on: the recorded stream carries a
/// `messageStop` event with a `stopReason` and a `metadata` event with usage.
/// Returns `(stopReason, metadata usage)`.
fn recorded_terminal_events(scenario: &str, interaction: usize) -> (String, Value) {
    let bytes = recorded_event_stream(scenario, interaction);
    let stop = embedded_json_object(&bytes, |value| value.get("stopReason").is_some())
        .unwrap_or_else(|| {
            panic!("{scenario}: the recorded stream must carry a messageStop event")
        });
    let stop_reason = stop["stopReason"]
        .as_str()
        .unwrap_or_else(|| panic!("{scenario}: stopReason should be a string"))
        .to_string();
    let metadata = embedded_json_object(&bytes, |value| {
        value.pointer("/usage/totalTokens").is_some()
    })
    .unwrap_or_else(|| {
        panic!("{scenario}: the recorded stream must carry a usage-bearing metadata event")
    });
    (stop_reason, metadata["usage"].clone())
}

fn recorded_request_body(scenario: &str, interaction: usize) -> String {
    recorded_interaction_bodies(BEDROCK_PROVIDER, scenario)
        .get(interaction)
        .map(|(request, _)| request.clone())
        .unwrap_or_else(|| panic!("{scenario}: interaction {interaction} should be recorded"))
}

// ---------------------------------------------------------------------------
// 1: off → None
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "unrecorded (no valid AWS credentials in this environment)"]
async fn stream_capture_off_raw_is_none() {
    let scenario = "raw_stream_capture_matrix/stream_capture_off_raw_is_none";
    with_bedrock_cassette(
        "raw_stream_capture_matrix/stream_capture_off_raw_is_none",
        |client| async move {
            let model = client.completion_model(MODEL);
            let request = request(&model, false);
            assert!(!request.capture_raw_response, "premise: default is off");
            let terminal =
                terminal_of(model.stream(request).await.expect("stream should start")).await;

            assert!(
                terminal.raw.is_none(),
                "terminal raw must stay None when capture was not requested, got {:?}",
                terminal.raw
            );
            assert!(terminal.usage.total_tokens > 0, "usage is unaffected");
        },
    )
    .await;

    recorded_terminal_events(scenario, 0);
}

// ---------------------------------------------------------------------------
// 2: on → raw is the raw_stream FinalResponse, serialized
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "unrecorded (no valid AWS credentials in this environment)"]
async fn stream_capture_on_terminal_round_trips_provider_type() {
    let scenario = "raw_stream_capture_matrix/stream_capture_on_terminal_round_trips_provider_type";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = std::sync::Arc::clone(&captured);
    with_bedrock_cassette(
        "raw_stream_capture_matrix/stream_capture_on_terminal_round_trips_provider_type",
        |client| async move {
            let model = client.completion_model(MODEL);
            let terminal = terminal_of(
                model
                    .stream(request(&model, true))
                    .await
                    .expect("stream should start"),
            )
            .await;

            let raw = terminal
                .raw
                .as_deref()
                .expect("terminal raw must be populated when capture was requested");
            let typed = BedrockStreamingResponse::deserialize(raw)
                .expect("raw must deserialize into BedrockStreamingResponse");
            assert_eq!(
                serde_json::to_value(&typed).expect("terminal type should serialize"),
                *raw,
                "BedrockStreamingResponse must round-trip through its own serde"
            );

            // The typed terminal agrees with the normalized one: raw is the
            // record normalize_stream mapped.
            let usage = typed.usage.expect("terminal carries usage");
            assert_eq!(usage.total_tokens as u64, terminal.usage.total_tokens);
            assert_eq!(usage.input_tokens as u64, terminal.usage.input_tokens);
            assert_eq!(usage.output_tokens as u64, terminal.usage.output_tokens);
            assert_eq!(typed.provider_request_id, terminal.provider_request_id);
            *sink.lock().expect("capture mutex") = Some(raw.clone());
        },
    )
    .await;

    let (_, usage) = recorded_terminal_events(scenario, 0);
    let raw = captured
        .lock()
        .expect("capture mutex")
        .take()
        .expect("the test body must have captured raw");
    assert_eq!(raw["usage"]["total_tokens"], usage["totalTokens"]);
}

// ---------------------------------------------------------------------------
// 3: terminal-only fields
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "unrecorded (no valid AWS credentials in this environment)"]
async fn stream_capture_on_exposes_bedrock_stop_reason() {
    let scenario = "raw_stream_capture_matrix/stream_capture_on_exposes_bedrock_stop_reason";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = std::sync::Arc::clone(&captured);
    with_bedrock_cassette(
        "raw_stream_capture_matrix/stream_capture_on_exposes_bedrock_stop_reason",
        |client| async move {
            let model = client.completion_model(MODEL);
            let terminal = terminal_of(
                model
                    .stream(request(&model, true))
                    .await
                    .expect("stream should start"),
            )
            .await;

            // The normalized terminal spells the finish reason in rig's
            // vocabulary; Bedrock's own spelling is only on raw.
            let mut without_raw = terminal.clone();
            without_raw.raw = None;
            let normalized =
                serde_json::to_value(&without_raw).expect("StreamFinal should serialize");
            assert!(normalized.get("stop_reason").is_none());
            assert_eq!(
                terminal.finish_reason,
                Some(rig::completion::FinishReason::Stop)
            );

            let raw = terminal
                .raw
                .as_deref()
                .expect("terminal raw must be populated when capture was requested")
                .clone();
            *sink.lock().expect("capture mutex") = Some(raw);
        },
    )
    .await;

    let raw = captured
        .lock()
        .expect("capture mutex")
        .take()
        .expect("the test body must have captured raw");
    let (stop_reason, usage) = recorded_terminal_events(scenario, 0);
    assert_eq!(
        stop_reason, "end_turn",
        "{scenario}: premise — the recorded turn ended on end_turn"
    );
    let typed = BedrockStreamingResponse::deserialize(&raw).expect("raw must deserialize");
    assert_eq!(
        typed.stop_reason,
        Some(StopReason::EndTurn),
        "raw carries Bedrock's own stopReason for the recorded end_turn"
    );
    assert_eq!(raw["stop_reason"], Value::String("EndTurn".to_string()));
    assert_eq!(raw["usage"]["total_tokens"], usage["totalTokens"]);
    assert_eq!(raw["usage"]["input_tokens"], usage["inputTokens"]);
    assert_eq!(raw["usage"]["output_tokens"], usage["outputTokens"]);
}

// ---------------------------------------------------------------------------
// 4: the flag never reaches the provider
// ---------------------------------------------------------------------------

/// One scenario, two interactions in wire order — off then on.
#[tokio::test]
#[ignore = "unrecorded (no valid AWS credentials in this environment)"]
async fn stream_request_invariant_off_vs_on() {
    let scenario = "raw_stream_capture_matrix/stream_request_invariant_off_vs_on";
    with_bedrock_cassette(
        "raw_stream_capture_matrix/stream_request_invariant_off_vs_on",
        |client| async move {
            let model = client.completion_model(MODEL);
            let off = terminal_of(
                model
                    .stream(request(&model, false))
                    .await
                    .expect("flag-off stream should start"),
            )
            .await;
            let on = terminal_of(
                model
                    .stream(request(&model, true))
                    .await
                    .expect("flag-on stream should start"),
            )
            .await;
            assert!(off.raw.is_none());
            assert!(on.raw.is_some());
            assert_eq!(off.provider, on.provider);
        },
    )
    .await;

    let off_request = recorded_request_body(scenario, 0);
    let on_request = recorded_request_body(scenario, 1);
    assert_eq!(
        off_request, on_request,
        "the flag-on streaming request body must be byte-identical to the \
         flag-off one — capture_raw_response must never reach Bedrock"
    );
    assert!(!off_request.contains("capture_raw"));
    recorded_terminal_events(scenario, 0);
    recorded_terminal_events(scenario, 1);
}
