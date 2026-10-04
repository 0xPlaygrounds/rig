//! Matrix for raw response capture on Bedrock's blocking Converse path
//! ([`CompletionResponse::raw`](rig::completion::CompletionResponse::raw)).
//!
//! # The feature
//!
//! Capture is always on. Every completion the seam returns carries `raw`: the
//! JSON body Bedrock sent, read off the HTTP response the SDK decoded. Nothing
//! about it is sent to Bedrock. `raw == Value::Null` means only that a
//! `CompletionResponse` was built by hand without a provider response behind
//! it, which no cell here can produce.
//!
//! Bedrock's response carries `metrics.latencyMs`, the server-side latency the
//! normalized [`rig::completion::CompletionResponse`] has no field for; cell 2
//! reads it back through `raw` and checks it against the fixture body.
//!
//! Every cell runs its one recorded turn through the shared execution helper
//! [`capture_completion`](crate::raw_capture::capture_completion) and asserts
//! against the parked response after the wrapper returns. The shared *format*
//! contracts (`raw_capture::chat`, `raw_capture::responses`) do not apply:
//! Converse is neither dialect.
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `raw_is_the_recorded_body` | provider JSON | `raw` equals the recorded body | unrecorded (no valid AWS credentials in this environment) |
//! | 2 | `raw_exposes_latency_metrics` | provider-only field | `raw.metrics.latencyMs` equals the fixture's | unrecorded (no valid AWS credentials in this environment) |
//! | 3 | `normalized_fields_match_the_recorded_body` | normalized view | choice text and usage equal the fixture body | unrecorded (no valid AWS credentials in this environment) |
//!
//! Every cell is unrecorded: no valid AWS credentials were available when
//! this matrix was written, and a fixture is never fabricated. The bodies are
//! complete and would pass once recorded; the `#[ignore]` attribute is the
//! only thing standing between them and the table's `recorded` status.
//!
//! To record once valid credentials exist (they are read by the AWS SDK's
//! default provider chain, `AWS_PROFILE` or `AWS_ACCESS_KEY_ID`/
//! `AWS_SECRET_ACCESS_KEY`[/`AWS_SESSION_TOKEN`], with region `us-east-1`):
//! remove the `#[ignore]` attributes, flip the table to `recorded`, then run
//! `RIG_PROVIDER_TEST_MODE=record cargo test -p rig --all-features --test bedrock bedrock::cassette::raw_capture_matrix -- --nocapture --test-threads=1`
//! and review the new fixtures under `crates/rig-cassette/fixtures/cassettes/bedrock/raw_capture_matrix/`
//! (the scrubber placeholders `x-amzn-requestid`; nothing else in a Converse
//! body is account state).

use rig::bedrock;
use serde_json::Value;

use super::super::support::with_bedrock_cassette;
use crate::cassettes::recorded_json_turn;
use crate::raw_capture::{assert_normalized_lacks, capture_completion};
use crate::support::{Observed, normalized_without_raw};
use rig::completion::CompletionRequest;

const BEDROCK_PROVIDER: &str = "bedrock";
const MODEL: &str = bedrock::completion::AMAZON_NOVA_LITE;
const PROMPT: &str = "Reply with exactly the single word: pong";

fn request() -> rig::completion::CompletionRequest {
    CompletionRequest::new(PROMPT)
        .temperature(0.0)
        .max_tokens(16)
}

/// The premise every cell rests on: the recorded body is a completed Converse
/// response reporting `metrics.latencyMs` and usage.
fn assert_recorded_converse_with_metrics(body: &Value, scenario: &str) {
    assert!(
        body.pointer("/metrics/latencyMs")
            .and_then(Value::as_i64)
            .is_some(),
        "{scenario}: the recorded body must report `metrics.latencyMs`; without \
         it this cell cannot prove raw exposes a provider-only field"
    );
    assert!(
        body.pointer("/usage/totalTokens").is_some(),
        "{scenario}: the recorded body must report usage"
    );
    assert!(
        body.get("stopReason").and_then(Value::as_str).is_some(),
        "{scenario}: the recorded body must carry a stopReason"
    );
}

// ---------------------------------------------------------------------------
// 1: raw is the body Bedrock sent
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "unrecorded (no valid AWS credentials in this environment)"]
async fn raw_is_the_recorded_body() {
    let scenario = "raw_capture_matrix/raw_is_the_recorded_body";
    let captured = Observed::default();
    let sink = captured.clone();
    with_bedrock_cassette(
        "raw_capture_matrix/raw_is_the_recorded_body",
        |client| async move {
            capture_completion(client.completion(MODEL), request(), sink)
                .await
                .expect("completion should succeed");
        },
    )
    .await;
    let response = captured.take();
    let (_, body) = recorded_json_turn(BEDROCK_PROVIDER, scenario);
    assert_recorded_converse_with_metrics(&body, scenario);
    assert_eq!(response.raw, body, "raw is the recorded body");
    assert!(!response.choice.is_empty());
}

// ---------------------------------------------------------------------------
// 2: a provider-only field rig does not normalize is readable from raw
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "unrecorded (no valid AWS credentials in this environment)"]
async fn raw_exposes_latency_metrics() {
    let scenario = "raw_capture_matrix/raw_exposes_latency_metrics";
    let captured = Observed::default();
    let sink = captured.clone();
    with_bedrock_cassette(
        "raw_capture_matrix/raw_exposes_latency_metrics",
        |client| async move {
            capture_completion(client.completion(MODEL), request(), sink)
                .await
                .expect("completion should succeed");
        },
    )
    .await;
    let response = captured.take();
    let raw = response.raw.clone();
    // The normalized `CompletionResponse` must not grow a `metrics` field:
    // the latency is reachable only through the capture.
    assert_normalized_lacks(&normalized_without_raw(response), &["metrics"]);

    let (_, body) = recorded_json_turn(BEDROCK_PROVIDER, scenario);
    assert_recorded_converse_with_metrics(&body, scenario);
    assert_eq!(
        raw.pointer("/metrics/latencyMs"),
        body.pointer("/metrics/latencyMs"),
        "raw.metrics.latencyMs must equal the recorded wire value"
    );
}

// ---------------------------------------------------------------------------
// 3: raw and the normalized view tell one story
// ---------------------------------------------------------------------------

/// The fields the wire body decides (choice text, usage) equal the
/// recorded body.
#[tokio::test]
#[ignore = "unrecorded (no valid AWS credentials in this environment)"]
async fn normalized_fields_match_the_recorded_body() {
    let scenario = "raw_capture_matrix/normalized_fields_match_the_recorded_body";
    let captured = Observed::default();
    let sink = captured.clone();
    with_bedrock_cassette(
        "raw_capture_matrix/normalized_fields_match_the_recorded_body",
        |client| async move {
            capture_completion(client.completion(MODEL), request(), sink)
                .await
                .expect("completion should succeed");
        },
    )
    .await;
    let response = captured.take();
    assert_eq!(response.provider(), BEDROCK_PROVIDER);
    assert!(
        response.provider_request_id.is_some(),
        "Bedrock always reports an x-amzn-requestid on success"
    );
    let (_, body) = recorded_json_turn(BEDROCK_PROVIDER, scenario);
    assert_recorded_converse_with_metrics(&body, scenario);
    let live = normalized_without_raw(response);
    let text = body
        .pointer("/output/message/content/0/text")
        .and_then(Value::as_str)
        .expect("recorded Converse body carries an assistant text block");
    assert_eq!(
        live.pointer("/choice/0/text").and_then(Value::as_str),
        Some(text),
        "the normalized choice must be the recorded body's text"
    );
    assert_eq!(
        live.pointer("/usage/total_tokens"),
        body.pointer("/usage/totalTokens"),
        "the normalized usage must be the recorded body's usage"
    );
}
