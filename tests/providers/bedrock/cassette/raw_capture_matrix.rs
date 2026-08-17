//! Matrix for opt-in raw response capture on Bedrock's blocking Converse path
//! (`CompletionRequest::capture_raw_response` → `CompletionResponse::raw`).
//!
//! # The feature
//!
//! `raw` is the value
//! [`CompletionModel::raw_completion`](rig::bedrock::completion::CompletionModel::raw_completion)
//! would have returned — [`AwsConverseOutput`], rig's serializable mirror of
//! the SDK's `ConverseOutput` — serialized with `serde_json::to_value`. It is
//! populated only when the request opted in and never reaches the wire.
//!
//! Bedrock's response carries `metrics.latencyMs`, the server-side latency the
//! normalized [`rig::completion::CompletionResponse`] has no field for; cell 3
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
//! | 1 | `capture_off_raw_is_none` | flag off (default) | `raw == None` | unrecorded (no valid AWS credentials in this environment) |
//! | 2 | `capture_on_raw_round_trips_provider_type` | flag on | `AwsConverseOutput::deserialize(&*raw)` re-serializes equal | unrecorded (no valid AWS credentials in this environment) |
//! | 3 | `capture_on_exposes_latency_metrics` | provider-only field | `raw.metrics.latency_ms` equals the fixture's `metrics.latencyMs` | unrecorded (no valid AWS credentials in this environment) |
//! | 4 | `request_invariant_off_vs_on` | on-wire request | flag-off and flag-on request bodies byte-identical | unrecorded (no valid AWS credentials in this environment) |
//! | 5 | `normalized_fields_identical_off_vs_on` | normalized view | off/on responses normalize their own wire bytes identically; only `raw` differs | unrecorded (no valid AWS credentials in this environment) |
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
use crate::cassettes::{recorded_interaction_bodies, recorded_json_request};

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

fn recorded_json_interactions(scenario: &str) -> Vec<(Value, Value)> {
    recorded_interaction_bodies(BEDROCK_PROVIDER, scenario)
        .into_iter()
        .map(|(request, response)| {
            let request: Value = serde_json::from_str(&request)
                .unwrap_or_else(|err| panic!("{scenario}: recorded request should be JSON: {err}"));
            let response: Value = serde_json::from_str(&response).unwrap_or_else(|err| {
                panic!("{scenario}: recorded response should be JSON: {err}")
            });
            (request, response)
        })
        .collect()
}

fn normalized_without_raw(mut response: RigCompletionResponse) -> Value {
    response.raw = None;
    serde_json::to_value(&response).expect("normalized response should serialize")
}

// ---------------------------------------------------------------------------
// 1: off → None
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "unrecorded (no valid AWS credentials in this environment)"]
async fn capture_off_raw_is_none() {
    let scenario = "raw_capture_matrix/capture_off_raw_is_none";
    with_bedrock_cassette(
        "raw_capture_matrix/capture_off_raw_is_none",
        |client| async move {
            let model = client.completion_model(MODEL);
            let request = request(&model, false);
            assert!(
                !request.capture_raw_response,
                "premise: the builder default is off"
            );

            let response = model
                .completion(request)
                .await
                .expect("completion should succeed");

            assert!(
                response.raw.is_none(),
                "raw must stay None when capture was not requested, got {:?}",
                response.raw
            );
            assert!(!response.choice.is_empty());
        },
    )
    .await;

    let (_, body) = recorded_json_interactions(scenario)
        .into_iter()
        .next()
        .expect("scenario should record one interaction");
    assert_recorded_converse_with_metrics(&body, scenario);
}

// ---------------------------------------------------------------------------
// 2: on → raw is the raw_completion value, serialized
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "unrecorded (no valid AWS credentials in this environment)"]
async fn capture_on_raw_round_trips_provider_type() {
    let scenario = "raw_capture_matrix/capture_on_raw_round_trips_provider_type";
    with_bedrock_cassette(
        "raw_capture_matrix/capture_on_raw_round_trips_provider_type",
        |client| async move {
            let model = client.completion_model(MODEL);
            let response = model
                .completion(request(&model, true))
                .await
                .expect("completion should succeed");

            let raw = response
                .raw
                .as_deref()
                .expect("raw must be populated when capture was requested");
            let typed = AwsConverseOutput::deserialize(raw)
                .expect("raw must deserialize into AwsConverseOutput");
            assert_eq!(
                serde_json::to_value(&typed).expect("provider type should serialize"),
                *raw,
                "AwsConverseOutput must round-trip through its own serde"
            );

            // The typed view normalizes to the same surface `completion`
            // produced: raw is the value the seam normalized.
            let renormalized: RigCompletionResponse =
                typed.try_into().expect("typed raw must normalize");
            assert_eq!(
                normalized_without_raw(renormalized),
                normalized_without_raw(response),
                "normalizing the captured raw must reproduce the normalized response"
            );
        },
    )
    .await;

    let (_, body) = recorded_json_interactions(scenario)
        .into_iter()
        .next()
        .expect("scenario should record one interaction");
    assert_recorded_converse_with_metrics(&body, scenario);
}

// ---------------------------------------------------------------------------
// 3: a provider-only field rig does not normalize is readable from raw
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "unrecorded (no valid AWS credentials in this environment)"]
async fn capture_on_exposes_latency_metrics() {
    let scenario = "raw_capture_matrix/capture_on_exposes_latency_metrics";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = std::sync::Arc::clone(&captured);
    with_bedrock_cassette(
        "raw_capture_matrix/capture_on_exposes_latency_metrics",
        |client| async move {
            let model = client.completion_model(MODEL);
            let response = model
                .completion(request(&model, true))
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
                .expect("raw must be populated when capture was requested")
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
    let (_, body) = recorded_json_interactions(scenario)
        .into_iter()
        .next()
        .expect("scenario should record one interaction");
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
// 4: the flag never reaches the provider
// ---------------------------------------------------------------------------

/// One scenario, two interactions in wire order — off then on.
#[tokio::test]
#[ignore = "unrecorded (no valid AWS credentials in this environment)"]
async fn request_invariant_off_vs_on() {
    let scenario = "raw_capture_matrix/request_invariant_off_vs_on";
    with_bedrock_cassette(
        "raw_capture_matrix/request_invariant_off_vs_on",
        |client| async move {
            let model = client.completion_model(MODEL);
            let off = model
                .completion(request(&model, false))
                .await
                .expect("flag-off completion should succeed");
            let on = model
                .completion(request(&model, true))
                .await
                .expect("flag-on completion should succeed");
            assert!(off.raw.is_none());
            assert!(on.raw.is_some());
        },
    )
    .await;

    let bodies = recorded_interaction_bodies(BEDROCK_PROVIDER, scenario);
    assert_eq!(
        bodies.len(),
        2,
        "{scenario}: the scenario must record exactly the off and on requests"
    );
    let (off_request, _) = &bodies[0];
    let (on_request, _) = &bodies[1];
    assert_eq!(
        off_request, on_request,
        "the flag-on request body must be byte-identical to the flag-off one — \
         capture_raw_response is local policy and must never reach Bedrock"
    );
    assert!(!off_request.contains("capture_raw"));
    let first: Value = recorded_json_request(BEDROCK_PROVIDER, scenario);
    assert!(first.get("messages").is_some_and(Value::is_array));
}

// ---------------------------------------------------------------------------
// 5: normalization is a pure function of the wire bytes, flag or no flag
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "unrecorded (no valid AWS credentials in this environment)"]
async fn normalized_fields_identical_off_vs_on() {
    let scenario = "raw_capture_matrix/normalized_fields_identical_off_vs_on";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = std::sync::Arc::clone(&captured);
    with_bedrock_cassette(
        "raw_capture_matrix/normalized_fields_identical_off_vs_on",
        |client| async move {
            let model = client.completion_model(MODEL);
            let off = model
                .completion(request(&model, false))
                .await
                .expect("flag-off completion should succeed");
            let on = model
                .completion(request(&model, true))
                .await
                .expect("flag-on completion should succeed");

            assert!(off.raw.is_none());
            assert!(on.raw.is_some());
            assert_eq!(on.provider, off.provider);
            assert_eq!(on.model, off.model);
            assert_eq!(on.finish_reason(), off.finish_reason());
            assert!(!off.choice.is_empty());
            assert!(!on.choice.is_empty());
            *sink.lock().expect("capture mutex") = vec![off, on];
        },
    )
    .await;

    let responses = std::mem::take(&mut *captured.lock().expect("capture mutex"));
    let interactions = recorded_json_interactions(scenario);
    assert_eq!(
        interactions.len(),
        2,
        "{scenario}: expected off and on turns"
    );

    // The wire body has no room for the AWS request id (it is the
    // `x-amzn-requestid` header, scrubbed to a placeholder on disk), so the
    // body-derived comparison masks it and checks it separately.
    for ((_, body), response) in interactions.into_iter().zip(responses) {
        assert_recorded_converse_with_metrics(&body, scenario);
        assert!(
            response.provider_request_id.is_some(),
            "Bedrock always reports an x-amzn-requestid on success"
        );
        let mut live = normalized_without_raw(response);
        live["provider_request_id"] = Value::Null;
        // Only the fields the wire body decides are compared: choice, usage,
        // finish reason, model and provider all come from the JSON body.
        for field in ["choice", "usage", "finish_reason", "provider", "model"] {
            assert!(
                live.get(field).is_some(),
                "normalized response should carry `{field}`"
            );
        }
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
}
