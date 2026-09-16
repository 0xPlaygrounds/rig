//! Matrix for raw response capture on mistral.rs's `/v1/chat/completions`
//! route ([`CompletionResponse::raw`](rig::completion::CompletionResponse::raw)).
//!
//! # The feature
//!
//! Capture is always on. mistral.rs is a local server rather than a hosted
//! provider, so it has no dialect of its own: it is driven through the plain
//! `OPENAI` chat-completions wire pointed at its base URL, and every
//! completion carries `raw` — the reply *document*, set by the driver from
//! the response bytes. Nothing about it is sent to the server.
//! `raw == Value::Null` means only that a `CompletionResponse` was built by
//! hand without a provider response behind it, which no cell here can
//! produce.
//!
//! mistral.rs stamps `system_fingerprint: "local"` and the `object`/`created`
//! envelope on every response; none has a home on the normalized
//! [`rig::completion::CompletionResponse`], and cell 2 reads them back
//! through `raw`. Its per-second throughput fields inside `usage`
//! (`avg_compl_tok_per_sec` and friends) are not modelled by the shared
//! [`openai::CompletionResponse`] either — and they still reach a caller,
//! because `raw` is the document rather than the parse: cell 1 pins that by
//! requiring `raw` to reproduce the recorded reply key for key while the
//! typed view is only as wide as the shared shape.
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `raw_is_the_reply_document` | body fidelity | `raw` reproduces the recorded reply, and reads back as `openai::CompletionResponse` | unrecorded (no mistral.rs server in this environment) |
//! | 2 | `raw_exposes_envelope_fields` | provider-only fields | `system_fingerprint`/`object`/`created` in `raw` equal the fixture body | unrecorded (no mistral.rs server in this environment) |
//! | 3 | `normalized_fields_equal_raw_renormalized` | normalized view | the normalized fields are the fields the reply document carries, live and in the fixture | unrecorded (no mistral.rs server in this environment) |
//!
//! The scenario literals — and therefore the fixture filenames — keep the
//! names they were recorded under; the cell names describe what the cells now
//! assert.
//!
//! Every cell is unrecorded: no mistral.rs server was listening on
//! `127.0.0.1:1234` when this matrix was written, and a fixture is never
//! fabricated. To record: start `mistralrs-server` on that port serving
//! `Qwen/Qwen3-4B` (or export `MISTRALRS_BASE_URL`/`MISTRALRS_MODEL`), remove
//! the `#[ignore]` attributes, flip the table to `recorded`, then run
//! `RIG_PROVIDER_TEST_MODE=record cargo test -p rig --all-features --test mistralrs mistralrs::cassette::raw_capture_matrix -- --nocapture --test-threads=1`
//! and review `tests/cassettes/mistralrs/raw_capture_matrix/`.

use rig::completion::{
    CompletionModel, CompletionRequest, CompletionResponse as RigCompletionResponse,
};
use rig::providers::openai;
use serde::Deserialize;
use serde_json::Value;

use super::super::support::{model_name, with_mistralrs_completions_cassette};
use crate::cassettes::{CassetteMode, recorded_interaction_bodies};

const MISTRALRS_PROVIDER: &str = "mistralrs";
/// The plain OpenAI dialect names itself `openai`, and a normalized response
/// is attributed to the dialect that produced it.
const NORMALIZED_PROVIDER: &str = "openai";
/// `/no_think` keeps Qwen3's reasoning trace out of the recording, exactly as
/// the neighbouring mistral.rs cassettes do.
const PROMPT: &str = "/no_think Reply with exactly the single word: pong";

fn request(model: &(impl CompletionModel + Clone)) -> CompletionRequest {
    model.completion_request(PROMPT).max_tokens(64).build()
}

fn assert_recorded_envelope(body: &Value, scenario: &str) {
    assert_eq!(
        body.get("object").and_then(Value::as_str),
        Some("chat.completion"),
        "{scenario}: the recorded body must be a chat.completion envelope"
    );
    assert!(
        body.get("created").and_then(Value::as_u64).is_some(),
        "{scenario}: the recorded body must carry `created`"
    );
    assert!(
        body.get("system_fingerprint")
            .and_then(Value::as_str)
            .is_some(),
        "{scenario}: the recorded body must carry `system_fingerprint` — without \
         it this cell cannot prove raw exposes a provider-only field"
    );
    assert!(
        body.get("usage").is_some_and(Value::is_object),
        "{scenario}: the recorded body must report usage"
    );
}

/// The single recorded interaction of a scenario, request and response parsed
/// as JSON.
fn recorded_json_interaction(scenario: &str) -> (Value, Value) {
    let bodies = recorded_interaction_bodies(MISTRALRS_PROVIDER, scenario);
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

/// Replay reads the scrubbed fixture back, so volatile fields (`created` → 0,
/// `chatcmpl-…` → placeholders) compare exactly; a live recording proves only
/// that both sides carry the field with the same JSON type.
fn assert_wire_value_matches(live: &Value, recorded: &Value, field: &str) {
    let (live_value, recorded_value) = (live.get(field), recorded.get(field));
    match CassetteMode::current() {
        CassetteMode::Replay => assert_eq!(
            live_value, recorded_value,
            "{field}: replayed value must equal the recorded wire value"
        ),
        CassetteMode::Record => {
            let (Some(live_value), Some(recorded_value)) = (live_value, recorded_value) else {
                panic!("{field}: both the live value and the recording must carry it");
            };
            assert_eq!(
                std::mem::discriminant(live_value),
                std::mem::discriminant(recorded_value),
                "{field}: live and recorded values must share a JSON type"
            );
        }
    }
}

fn normalized_without_raw(mut response: RigCompletionResponse) -> Value {
    response.raw = Value::Null;
    serde_json::to_value(&response).expect("normalized response should serialize")
}

// ---------------------------------------------------------------------------
// 1: raw is the reply document, and the shared type reads it back
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "unrecorded (no mistral.rs server in this environment)"]
async fn raw_is_the_reply_document() {
    let scenario = "raw_capture_matrix/raw_round_trips_provider_type";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = std::sync::Arc::clone(&captured);
    with_mistralrs_completions_cassette(
        "raw_capture_matrix/raw_round_trips_provider_type",
        |client| async move {
            let model = client.chat(model_name());
            let response = model
                .completion(request(&model))
                .await
                .expect("completion should succeed");
            let typed = openai::CompletionResponse::deserialize(&response.raw)
                .expect("raw must deserialize into openai::CompletionResponse");
            // The typed view agrees with the normalized one on what the model
            // said, so raw is a superset, not a divergent copy.
            assert_eq!(Some(typed.model.as_str()), response.model.as_deref());
            assert_eq!(Some(typed.id.as_str()), response.response_id.as_deref());
            assert_eq!(response.provider, NORMALIZED_PROVIDER);
            assert!(!response.choice.is_empty());
            *sink.lock().expect("capture mutex") = Some(response.raw);
        },
    )
    .await;

    let raw = captured
        .lock()
        .expect("capture mutex")
        .take()
        .expect("the test body must have captured raw");
    let (_, body) = recorded_json_interaction(scenario);
    assert_recorded_envelope(&body, scenario);
    openai::CompletionResponse::deserialize(&body)
        .expect("recorded body must be a chat-completions response");
    // The document, key for key: whatever mistral.rs sent — including the
    // `usage` throughput fields no shared type models — is what a caller
    // reads off `raw`.
    for key in body
        .as_object()
        .expect("the recorded reply is a JSON object")
        .keys()
        .filter(|key| key.as_str() != "id" && key.as_str() != "created")
    {
        assert_eq!(
            raw.get(key),
            body.get(key),
            "raw should carry the provider's `{key}` unchanged"
        );
    }
    for field in ["created", "id"] {
        assert_wire_value_matches(&raw, &body, field);
    }
}

// ---------------------------------------------------------------------------
// 2: envelope fields rig does not normalize are readable from raw
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "unrecorded (no mistral.rs server in this environment)"]
async fn raw_exposes_envelope_fields() {
    let scenario = "raw_capture_matrix/raw_exposes_envelope_fields";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = std::sync::Arc::clone(&captured);
    with_mistralrs_completions_cassette(
        "raw_capture_matrix/raw_exposes_envelope_fields",
        |client| async move {
            let model = client.chat(model_name());
            let response = model
                .completion(request(&model))
                .await
                .expect("completion should succeed");
            let normalized = normalized_without_raw(response.clone());
            for field in ["object", "created", "system_fingerprint"] {
                assert!(
                    normalized.get(field).is_none(),
                    "normalized CompletionResponse must not grow a `{field}` field"
                );
            }
            let raw = response.raw;
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
    assert_recorded_envelope(&body, scenario);
    for field in ["object", "system_fingerprint", "model"] {
        assert_eq!(
            raw.get(field),
            body.get(field),
            "raw.{field} must equal the recorded wire value"
        );
    }
    for field in ["created", "id"] {
        assert_wire_value_matches(&raw, &body, field);
    }
    let typed = openai::CompletionResponse::deserialize(&raw).expect("raw must deserialize");
    assert_eq!(
        typed.system_fingerprint.as_deref(),
        body["system_fingerprint"].as_str()
    );
    assert_eq!(Some(typed.object.as_str()), body["object"].as_str());
}

// ---------------------------------------------------------------------------
// 3: raw and the normalized view tell one story
// ---------------------------------------------------------------------------

/// The normalized fields are the fields the reply document carries: there is
/// one decoder and one mapping, so the provider-native view read back out of
/// `raw` — and the recorded wire body it came from — must agree with the
/// normalized response on identity, model, finish reason and usage.
#[tokio::test]
#[ignore = "unrecorded (no mistral.rs server in this environment)"]
async fn normalized_fields_equal_raw_renormalized() {
    let scenario = "raw_capture_matrix/normalized_fields_equal_raw_renormalized";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = std::sync::Arc::clone(&captured);
    with_mistralrs_completions_cassette(
        "raw_capture_matrix/normalized_fields_equal_raw_renormalized",
        |client| async move {
            let model = client.chat(model_name());
            let response = model
                .completion(request(&model))
                .await
                .expect("completion should succeed");

            let typed = openai::CompletionResponse::deserialize(&response.raw)
                .expect("raw must deserialize into openai::CompletionResponse");
            assert_eq!(response.provider, NORMALIZED_PROVIDER);
            assert_eq!(Some(typed.model.as_str()), response.model.as_deref());
            assert_eq!(Some(typed.id.as_str()), response.response_id.as_deref());
            let choice = typed
                .choices
                .first()
                .expect("the reply must carry a choice");
            assert!(
                !choice.finish_reason.is_empty() && response.finish_reason().is_some(),
                "the native finish reason `{}` must reach the normalized response",
                choice.finish_reason
            );
            let usage = typed.usage.expect("mistral.rs reports usage");
            assert_eq!(
                Some(usage.prompt_tokens as u64),
                response.usage.input_tokens
            );
            assert_eq!(Some(usage.total_tokens as u64), response.usage.total_tokens);
            assert!(!response.choice.is_empty());

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
    assert_recorded_envelope(&body, scenario);
    // And the same fields against the recorded bytes, so a recording that
    // stopped carrying them fails loudly instead of covering nothing.
    assert_eq!(response.model.as_deref(), body["model"].as_str(), "model");
    assert_eq!(
        response.usage.input_tokens,
        body["usage"]["prompt_tokens"].as_u64(),
        "input tokens"
    );
    assert_eq!(
        response.usage.total_tokens,
        body["usage"]["total_tokens"].as_u64(),
        "total tokens"
    );
}
