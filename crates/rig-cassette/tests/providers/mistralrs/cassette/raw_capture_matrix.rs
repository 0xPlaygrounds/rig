//! Matrix for raw response capture on mistral.rs's `/v1/chat/completions` route
//! ([`CompletionResponse::raw`](rig::completion::CompletionResponse::raw)).
//!
//! Capture is always on. mistral.rs is a local server with no dialect of its own: it is driven
//! through the plain `OPENAI` chat-completions wire pointed at its base URL, and every
//! completion carries `raw`, the reply document the driver set from the response bytes.
//! `raw == Value::Null` means only that a `CompletionResponse` was built by hand, which no cell
//! here can produce.
//!
//! mistral.rs stamps `system_fingerprint: "local"`, the `object`/`created` envelope and
//! per-second throughput fields in `usage`, none of which has a normalized home.
//! `raw_is_the_reply_document` requires `raw` to reproduce the recorded reply key for key,
//! `raw_exposes_envelope_fields` reads the envelope back through `raw`, and
//! `normalized_fields_equal_raw_renormalized` checks the normalized fields are the ones the
//! document carries. The cells are ignored until recorded against a local `mistralrs-server`
//! (`MISTRALRS_BASE_URL`, `MISTRALRS_MODEL`); a fixture is never fabricated.

use rig::completion::CompletionRequest;
use serde_json::Value;

use super::super::support::{model_name, with_mistralrs_completions_cassette};
use crate::cassettes::recorded_json_turn;
use crate::raw_capture::{assert_normalized_lacks, capture_completion};
use crate::support::{
    Observed, assert_matches_recorded_document, assert_wire_value_matches, normalized_without_raw,
};

const MISTRALRS_PROVIDER: &str = "mistralrs";
/// The plain OpenAI dialect names itself `openai`, and a normalized response
/// is attributed to the dialect that produced it.
const NORMALIZED_PROVIDER: &str = "openai";
/// `/no_think` keeps Qwen3's reasoning trace out of the recording, exactly as
/// the neighbouring mistral.rs cassettes do.
const PROMPT: &str = "/no_think Reply with exactly the single word: pong";

fn request() -> CompletionRequest {
    CompletionRequest::new(PROMPT).max_tokens(64)
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

// ---------------------------------------------------------------------------
// 1: raw is the reply document, key for key
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "unrecorded (no mistral.rs server in this environment)"]
async fn raw_is_the_reply_document() {
    let scenario = "raw_capture_matrix/raw_round_trips_provider_type";
    let captured = Observed::default();
    let sink = captured.clone();
    with_mistralrs_completions_cassette(
        "raw_capture_matrix/raw_round_trips_provider_type",
        |client| async move {
            capture_completion(client.chat(model_name()), request(), sink)
                .await
                .expect("completion should succeed");
        },
    )
    .await;

    let response = captured.take();
    let typed = response.raw.clone();
    // `raw` agrees with the normalized view on what the model said,
    // so raw is a superset, not a divergent copy.
    assert_eq!(typed["model"].as_str(), response.model());
    assert_eq!(typed["id"].as_str(), response.response_id());
    assert_eq!(response.provider(), NORMALIZED_PROVIDER);
    assert!(!response.choice.is_empty());

    let raw = &response.raw;
    let (_, body) = recorded_json_turn(MISTRALRS_PROVIDER, scenario);
    assert_recorded_envelope(&body, scenario);
    // The document, key for key: whatever mistral.rs sent, including the
    // `usage` throughput fields the normalized response has no home for, is
    // what a caller reads off `raw`.
    assert_matches_recorded_document(raw, &body, &["id"], "raw is the provider's document");
    assert_wire_value_matches(raw, &body, "id");
}

// ---------------------------------------------------------------------------
// 2: envelope fields rig does not normalize are readable from raw
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "unrecorded (no mistral.rs server in this environment)"]
async fn raw_exposes_envelope_fields() {
    let scenario = "raw_capture_matrix/raw_exposes_envelope_fields";
    let captured = Observed::default();
    let sink = captured.clone();
    with_mistralrs_completions_cassette(
        "raw_capture_matrix/raw_exposes_envelope_fields",
        |client| async move {
            capture_completion(client.chat(model_name()), request(), sink)
                .await
                .expect("completion should succeed");
        },
    )
    .await;

    let response = captured.take();
    let normalized = normalized_without_raw(response.clone());
    assert_normalized_lacks(&normalized, &["object", "created", "system_fingerprint"]);

    let raw = &response.raw;
    let (_, body) = recorded_json_turn(MISTRALRS_PROVIDER, scenario);
    assert_recorded_envelope(&body, scenario);
    for field in ["object", "system_fingerprint", "model"] {
        assert_eq!(
            raw.get(field),
            body.get(field),
            "raw.{field} must equal the recorded wire value"
        );
    }
    for field in ["created", "id"] {
        assert_wire_value_matches(raw, &body, field);
    }
    assert_eq!(
        raw["system_fingerprint"].as_str(),
        body["system_fingerprint"].as_str()
    );
    assert_eq!(raw["object"].as_str(), body["object"].as_str());
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
    let captured = Observed::default();
    let sink = captured.clone();
    with_mistralrs_completions_cassette(
        "raw_capture_matrix/normalized_fields_equal_raw_renormalized",
        |client| async move {
            capture_completion(client.chat(model_name()), request(), sink)
                .await
                .expect("completion should succeed");
        },
    )
    .await;

    let response = captured.take();
    let typed = response.raw.clone();
    assert_eq!(response.provider(), NORMALIZED_PROVIDER);
    assert_eq!(typed["model"].as_str(), response.model());
    assert_eq!(typed["id"].as_str(), response.response_id());
    let finish = typed["choices"][0]["finish_reason"]
        .as_str()
        .unwrap_or_default();
    // Deliberately weaker than the shared body contract: mistral.rs's finish
    // vocabulary is unrecorded here, so this claims only that a native reason
    // was reported and that one reached the normalized response.
    assert!(
        !finish.is_empty() && response.finish_reason().is_some(),
        "the native finish reason `{finish}` must reach the normalized response"
    );
    let usage = &typed["usage"];
    assert_eq!(usage["prompt_tokens"].as_u64(), response.usage.input_tokens);
    assert_eq!(usage["total_tokens"].as_u64(), response.usage.total_tokens);
    assert!(!response.choice.is_empty());

    let (_, body) = recorded_json_turn(MISTRALRS_PROVIDER, scenario);
    assert_recorded_envelope(&body, scenario);
    // And the same fields against the recorded bytes, so a recording that
    // stopped carrying them fails loudly instead of covering nothing.
    assert_eq!(response.model(), body["model"].as_str(), "model");
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
