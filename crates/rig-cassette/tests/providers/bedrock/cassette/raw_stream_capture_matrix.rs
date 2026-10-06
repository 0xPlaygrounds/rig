//! Matrix for raw capture on Bedrock's ConverseStream path
//! ([`CompletionResponse::raw`](rig::completion::CompletionResponse::raw)).
//!
//! # The feature
//!
//! Capture is always on. Every stream the seam yields carries `raw`: the
//! `ConverseOutput` a unary call returns, rebuilt from the events read off
//! the event-stream body the SDK decoded, so `messageStop` and `metadata`
//! fields sit at its top level. Nothing about it is sent to Bedrock.
//! `raw == Value::Null` means only that a `CompletionResponse` was built by
//! hand without a provider terminal behind it, which no cell here can
//! produce.
//!
//! Bedrock streams the AWS event-stream binary framing (recorded base64), so
//! the premise checks below decode the fixture body and locate the JSON
//! payloads of the `messageStop` and `metadata` events inside it.
//!
//! Both cells stream their one recorded turn through the shared execution
//! helper [`capture_terminal`](crate::raw_capture::capture_terminal) and
//! assert against the parked terminal after the wrapper returns. The shared
//! *format* contracts (`raw_capture::chat`, `raw_capture::responses`) do not
//! apply: ConverseStream is neither dialect, and its frames are binary rather
//! than SSE JSON.
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `stream_raw_is_the_recorded_terminal_events` | provider JSON | `raw.stopReason` and `raw.usage` equal the recorded events' | unrecorded (no valid AWS credentials in this environment) |
//! | 2 | `stream_raw_exposes_bedrock_stop_reason` | terminal-only field | `raw.stopReason` is Bedrock's own spelling; the normalized terminal lacks it | unrecorded (no valid AWS credentials in this environment) |
//!
//! Every cell is unrecorded: no valid AWS credentials were available when
//! this matrix was written, and a fixture is never fabricated. To record once
//! valid credentials exist: remove the `#[ignore]` attributes, flip the table
//! to `recorded`, then run
//! `RIG_PROVIDER_TEST_MODE=record cargo test -p rig --all-features --test bedrock bedrock::cassette::raw_stream_capture_matrix -- --nocapture --test-threads=1`
//! and review `crates/rig-cassette/fixtures/cassettes/bedrock/raw_stream_capture_matrix/` (the
//! streaming bodies are base64; decode before scanning).

use base64::{Engine, prelude::BASE64_STANDARD};
use rig::bedrock;
use serde_json::Value;

use super::super::support::with_bedrock_cassette;
use crate::cassettes::recorded_interaction_bodies;
use crate::raw_capture::{assert_normalized_lacks, capture_terminal};
use crate::support::Observed;
use crate::support::normalized_without_raw;
use rig::completion::CompletionRequest;

const BEDROCK_PROVIDER: &str = "bedrock";
const MODEL: &str = bedrock::completion::AMAZON_NOVA_LITE;
const PROMPT: &str = "Reply with exactly the single word: pong";

fn request() -> rig::completion::CompletionRequest {
    CompletionRequest::new(PROMPT)
        .temperature(0.0)
        .max_tokens(16)
}

/// The recorded event-stream bytes of the scenario's single interaction,
/// base64-decoded.
fn recorded_event_stream(scenario: &str) -> Vec<u8> {
    let bodies = recorded_interaction_bodies(BEDROCK_PROVIDER, scenario);
    assert_eq!(
        bodies.len(),
        1,
        "{scenario}: the scenario must record exactly one interaction"
    );
    let (_, response) = &bodies[0];
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
/// Returns the `messageStop` and `metadata` payloads.
fn recorded_terminal_events(scenario: &str) -> (Value, Value) {
    let bytes = recorded_event_stream(scenario);
    let stop = embedded_json_object(&bytes, |value| value.get("stopReason").is_some())
        .unwrap_or_else(|| {
            panic!("{scenario}: the recorded stream must carry a messageStop event")
        });
    let metadata = embedded_json_object(&bytes, |value| {
        value.pointer("/usage/totalTokens").is_some()
    })
    .unwrap_or_else(|| {
        panic!("{scenario}: the recorded stream must carry a usage-bearing metadata event")
    });
    (stop, metadata)
}

// ---------------------------------------------------------------------------
// 1: raw holds what Bedrock sent in the terminal events
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "unrecorded (no valid AWS credentials in this environment)"]
async fn stream_raw_is_the_recorded_terminal_events() {
    let scenario = "raw_stream_capture_matrix/stream_raw_is_the_recorded_terminal_events";
    let captured = Observed::default();
    let sink = captured.clone();
    with_bedrock_cassette(
        "raw_stream_capture_matrix/stream_raw_is_the_recorded_terminal_events",
        |client| async move {
            capture_terminal(client.completion(MODEL), request(), sink)
                .await
                .expect("stream should start");
        },
    )
    .await;

    let terminal = captured.take();
    let (stop, metadata) = recorded_terminal_events(scenario);
    assert_eq!(terminal.raw["stopReason"], stop["stopReason"]);
    assert_eq!(terminal.raw["usage"], metadata["usage"]);
    assert_eq!(
        terminal.usage.total_tokens,
        metadata
            .pointer("/usage/totalTokens")
            .and_then(Value::as_u64)
    );
}

// ---------------------------------------------------------------------------
// 2: terminal-only fields
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "unrecorded (no valid AWS credentials in this environment)"]
async fn stream_raw_exposes_bedrock_stop_reason() {
    let scenario = "raw_stream_capture_matrix/stream_raw_exposes_bedrock_stop_reason";
    let captured = Observed::default();
    let sink = captured.clone();
    with_bedrock_cassette(
        "raw_stream_capture_matrix/stream_raw_exposes_bedrock_stop_reason",
        |client| async move {
            capture_terminal(client.completion(MODEL), request(), sink)
                .await
                .expect("stream should start");
        },
    )
    .await;

    let terminal = captured.take();
    // The normalized terminal spells the finish reason in rig's vocabulary;
    // Bedrock's own spelling is only on raw.
    assert_normalized_lacks(&normalized_without_raw(terminal.clone()), &["stopReason"]);
    assert_eq!(
        terminal.finish_reason(),
        Some(rig::completion::FinishReason::Stop)
    );
    let (stop, _) = recorded_terminal_events(scenario);
    assert_eq!(
        stop["stopReason"], "end_turn",
        "{scenario}: premise: the recorded turn ended on end_turn"
    );
    assert_eq!(terminal.raw["stopReason"], "end_turn");
}
