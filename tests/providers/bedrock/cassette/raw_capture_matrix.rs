//! Matrix for raw response capture on Bedrock's blocking Converse path
//! ([`CompletionResponse::raw`](rig::completion::CompletionResponse::raw)).
//!
//! # The feature
//!
//! Capture is always on. Every completion the seam returns carries `raw`: the
//! value
//! [`CompletionModel::raw_completion`](rig::bedrock::completion::CompletionModel::raw_completion)
//! would have returned — [`AwsConverseOutput`], rig's serializable mirror of
//! the SDK's `ConverseOutput` — serialized with `serde_json::to_value` before
//! normalization. Nothing about it is sent to Bedrock. `raw == None` means
//! only that a `CompletionResponse` was built by hand without a provider
//! response behind it, which no cell here can produce.
//!
//! Bedrock's response carries `metrics.latencyMs`, the server-side latency the
//! normalized [`rig::completion::CompletionResponse`] has no field for; cell 2
//! reads it back through `raw` and checks it against the fixture body. Two
//! Converse fields are deliberately *not* on `raw` even when the wire carried
//! them: the guardrail `trace`, `performance_config` and `service_tier` keep
//! the SDK's own (non-`Serialize`) types and are `#[serde(skip)]` on
//! [`InternalConverseOutput`](rig::bedrock::types::converse_output::InternalConverseOutput)
//! (#2311), so `raw` — being the serialized value — omits them; typed access
//! to the trace stays on the `raw_completion` route
//! (`raw_provider_data/guardrail_trace_survives_into_raw_completion`).
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `raw_round_trips_provider_type` | typed access | `AwsConverseOutput::deserialize(&*raw)` re-serializes equal | unrecorded (no valid AWS credentials in this environment) |
//! | 2 | `raw_exposes_latency_metrics` | provider-only field | `raw.metrics.latency_ms` equals the fixture's `metrics.latencyMs` | unrecorded (no valid AWS credentials in this environment) |
//! | 3 | `normalized_fields_equal_raw_renormalized` | normalized view | the normalized response equals `raw` re-normalized (`try_into`); choice text and usage equal the fixture body | unrecorded (no valid AWS credentials in this environment) |
//!
//! Every cell is unrecorded: the `AWS_*` variables present when this matrix
//! was written carried an expired session token (`aws sts get-caller-identity`
//! failed), and a fixture is never fabricated. The bodies are complete and
//! would pass once recorded; the `#[ignore]` attribute is the only thing
//! standing between them and the table's `recorded` status.
//!
//! To record once valid credentials exist (they are read by the AWS SDK's
//! default provider chain — `AWS_PROFILE` or `AWS_ACCESS_KEY_ID`/
//! `AWS_SECRET_ACCESS_KEY`[/`AWS_SESSION_TOKEN`] — with region `us-east-1`):
//! remove the `#[ignore]` attributes, flip the table to `recorded`, then run
//! `RIG_PROVIDER_TEST_MODE=record cargo test -p rig --all-features --test bedrock bedrock::cassette::raw_capture_matrix -- --nocapture --test-threads=1`
//! and review the new fixtures under `tests/cassettes/bedrock/raw_capture_matrix/`
//! (the scrubber placeholders `x-amzn-requestid`; nothing else in a Converse
//! body is account state).

use rig::bedrock;
use rig::bedrock::types::assistant_content::AwsConverseOutput;
use rig::completion::{CompletionModel as _, CompletionResponse as RigCompletionResponse};
use rig::prelude::*;
use serde::Deserialize;
use serde_json::Value;

use super::super::support::with_bedrock_cassette;
use crate::cassettes::recorded_interaction_bodies;

const BEDROCK_PROVIDER: &str = "bedrock";
const MODEL: &str = bedrock::completion::AMAZON_NOVA_LITE;
const PROMPT: &str = "Reply with exactly the single word: pong";

fn request(model: &bedrock::completion::CompletionModel) -> rig::completion::CompletionRequest {
    model
        .completion_request(PROMPT)
        .temperature(0.0)
        .max_tokens(16)
        .build()
}

/// The premise every cell rests on: the recorded body is a completed Converse
/// response reporting `metrics.latencyMs` and usage.
fn assert_recorded_converse_with_metrics(body: &Value, scenario: &str) {
    assert!(
        body.pointer("/metrics/latencyMs")
            .and_then(Value::as_i64)
            .is_some(),
        "{scenario}: the recorded body must report `metrics.latencyMs` — without \
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

/// The single recorded interaction of a scenario, request and response parsed
/// as JSON.
fn recorded_json_interaction(scenario: &str) -> (Value, Value) {
    let bodies = recorded_interaction_bodies(BEDROCK_PROVIDER, scenario);
    assert_eq!(
        bodies.len(),
        1,
        "{scenario}: the scenario must record exactly one interaction"
    );
    let (request, response) = &bodies[0];
    let request: Value = serde_json::from_str(request)
        .unwrap_or_else(|err| panic!("{scenario}: recorded request should be JSON: {err}"));
    let response: Value = serde_json::from_str(response)
        .unwrap_or_else(|err| panic!("{scenario}: recorded response should be JSON: {err}"));
    (request, response)
}

fn normalized_without_raw(mut response: RigCompletionResponse) -> Value {
    response.raw = None;
    serde_json::to_value(&response).expect("normalized response should serialize")
}

// ---------------------------------------------------------------------------
// 1: raw is the raw_completion value, serialized
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "unrecorded (no valid AWS credentials in this environment)"]
async fn raw_round_trips_provider_type() {
    let scenario = "raw_capture_matrix/raw_round_trips_provider_type";
    with_bedrock_cassette(
        "raw_capture_matrix/raw_round_trips_provider_type",
        |client| async move {
            let model = client.completion_model(MODEL);
            let response = model
                .completion(request(&model))
                .await
                .expect("completion should succeed");

            let raw = response
                .raw
                .as_deref()
                .expect("every provider-backed completion carries raw");
            let typed = AwsConverseOutput::deserialize(raw)
                .expect("raw must deserialize into AwsConverseOutput");
            assert_eq!(
                serde_json::to_value(&typed).expect("provider type should serialize"),
                *raw,
                "AwsConverseOutput must round-trip through its own serde"
            );
            assert!(!response.choice.is_empty());
        },
    )
    .await;

    let (_, body) = recorded_json_interaction(scenario);
    assert_recorded_converse_with_metrics(&body, scenario);
}

// ---------------------------------------------------------------------------
// 2: a provider-only field rig does not normalize is readable from raw
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "unrecorded (no valid AWS credentials in this environment)"]
async fn raw_exposes_latency_metrics() {
    let scenario = "raw_capture_matrix/raw_exposes_latency_metrics";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = std::sync::Arc::clone(&captured);
    with_bedrock_cassette(
        "raw_capture_matrix/raw_exposes_latency_metrics",
        |client| async move {
            let model = client.completion_model(MODEL);
            let response = model
                .completion(request(&model))
                .await
                .expect("completion should succeed");

            let normalized = normalized_without_raw(response.clone());
            assert!(
                normalized.get("metrics").is_none(),
                "normalized CompletionResponse must not grow a `metrics` field"
            );

            let raw = response
                .raw
                .as_deref()
                .expect("every provider-backed completion carries raw")
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
    let (_, body) = recorded_json_interaction(scenario);
    assert_recorded_converse_with_metrics(&body, scenario);

    // The mirror type spells the field `latency_ms`; the wire spells it
    // `latencyMs`. Same value either way.
    assert_eq!(
        raw.pointer("/metrics/latency_ms"),
        body.pointer("/metrics/latencyMs"),
        "raw.metrics.latency_ms must equal the recorded wire value"
    );
    let typed = AwsConverseOutput::deserialize(&raw).expect("raw must deserialize");
    assert_eq!(
        typed.0.metrics.map(|metrics| metrics.latency_ms),
        body.pointer("/metrics/latencyMs").and_then(Value::as_i64)
    );
    // The SDK-typed extras are `#[serde(skip)]`, so `raw` never carries them
    // — the documented boundary of the serialized escape hatch.
    for skipped in ["trace", "performance_config", "service_tier"] {
        assert!(
            raw.get(skipped).is_none(),
            "raw must not carry the serde-skipped `{skipped}` field"
        );
    }
}

// ---------------------------------------------------------------------------
// 3: raw and the typed route tell one story
// ---------------------------------------------------------------------------

/// The normalized response, with `raw` stripped, must equal the normalization
/// (`try_into`) of `raw` read back through the mirror type — and the fields
/// the wire body decides (choice text, usage) must equal the recorded body.
/// Capture is a pure serialization of the value normalization consumed.
#[tokio::test]
#[ignore = "unrecorded (no valid AWS credentials in this environment)"]
async fn normalized_fields_equal_raw_renormalized() {
    let scenario = "raw_capture_matrix/normalized_fields_equal_raw_renormalized";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = std::sync::Arc::clone(&captured);
    with_bedrock_cassette(
        "raw_capture_matrix/normalized_fields_equal_raw_renormalized",
        |client| async move {
            let model = client.completion_model(MODEL);
            let response = model
                .completion(request(&model))
                .await
                .expect("completion should succeed");

            let raw = response
                .raw
                .as_deref()
                .expect("every provider-backed completion carries raw");
            // The AWS request id is the `x-amzn-requestid` header, not part of
            // the Converse body, so the raw-derived normalization is given the
            // same one before the field-for-field comparison.
            let from_raw: RigCompletionResponse = AwsConverseOutput::deserialize(raw)
                .expect("raw must deserialize into AwsConverseOutput")
                .try_into()
                .expect("raw must normalize");
            let from_raw =
                from_raw.with_optional_provider_request_id(response.provider_request_id.clone());

            assert_eq!(response.provider, BEDROCK_PROVIDER);
            assert_eq!(from_raw.provider, response.provider);
            assert_eq!(from_raw.model, response.model);
            assert_eq!(from_raw.finish_reason(), response.finish_reason());
            assert_eq!(from_raw.identity(), response.identity());
            assert_eq!(from_raw.usage, response.usage);
            assert!(!response.choice.is_empty());
            assert_eq!(
                normalized_without_raw(from_raw),
                normalized_without_raw(response.clone()),
                "re-normalizing raw must reproduce the normalized response field-for-field"
            );

            *sink.lock().expect("capture mutex") = Some(response);
        },
    )
    .await;

    let response = captured
        .lock()
        .expect("capture mutex")
        .take()
        .expect("the test body must have captured the response");
    let (_, body) = recorded_json_interaction(scenario);
    assert_recorded_converse_with_metrics(&body, scenario);
    assert!(
        response.provider_request_id.is_some(),
        "Bedrock always reports an x-amzn-requestid on success"
    );
    // Only the fields the wire body decides are compared against it: the
    // request id lives in a header the scrubber placeholders on disk.
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
