//! Raw provider response capture on Groq's blocking chat-completions path.
//!
//! **The feature.** Every blocking completion attaches the provider's own
//! reply to the normalized [`rig::completion::CompletionResponse::raw`].
//! Capture is always on: there is no flag to request it, nothing about it
//! reaches the wire, and a `Value::Null` only ever means a response built by
//! hand with no provider payload behind it.
//!
//! **What `raw` is.** The driver sets it from the reply's bytes
//! (`driver::call`), so it is the provider's response *document* rather than a
//! round-trip through whatever type the decoder parsed. Two things follow for
//! Groq specifically. Its timing accounting (`usage.queue_time`,
//! `prompt_time`, `completion_time`, `total_time`) is reachable through the
//! typed escape hatch — [`openai::CompletionResponse`] models those fields and
//! the normalized [`rig::completion::Usage`] has no slot for any of them. And
//! its `x_groq` envelope, which *no* shared chat-completions type models,
//! reaches a caller anyway, precisely because `raw` is the body. `raw` is a
//! second view of the same response, never a substitute for a normalized
//! field.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `raw_is_the_verbatim_response_body` | body fidelity | `raw` reproduces the recorded reply field for field, reads back as `openai::CompletionResponse`, and still carries the unmodelled `x_groq` | recorded |
//! | 2 | `raw_exposes_queue_time` | provider-only fields | `raw`'s `usage.queue_time`/`prompt_time` and `system_fingerprint` equal the fixture body, and none of them has a normalized slot | recorded |
//! | 3 | `normalized_fields_match_raw_renormalized` | normalized view | the response reproduces its fixture bytes (body and `x-request-id` header), and the same checks hold against its own `raw` | recorded |
//!
//! The scenario literals — and therefore the fixture filenames — keep the
//! names they were recorded under; the cell names describe what the cells now
//! assert.
//!
//! Every cell is recorded. Each re-derives its premise from its own fixture
//! after the wrapper returns: cell 2 reads the queue time out of the recorded
//! body rather than trusting the number the typed view reports, and cell 3
//! checks the normalized fields against the recorded body and header before
//! checking them against `raw`, so a recording that stopped carrying usage, a
//! finish reason, or the request-id header fails loudly instead of covering
//! nothing.

use rig::completion::{CompletionModel, CompletionRequest};
use rig::providers::openai;
use serde::Deserialize;
use serde_json::json;

use super::RAW_CAPTURE_MATRIX_MODEL;
use super::support::with_groq_cassette_result;
use crate::cassettes::{recorded_json_turn, recorded_response_header};
use crate::raw_capture::{assert_contracted_request_id, capture_completion, chat};
use crate::support::{Observed, assert_matches_recorded_document, assert_matches_recorded_token};

const PROVIDER: &str = "groq";
const PROMPT: &str = "Reply with the single word: pong";
const REQUEST_ID_HEADER: &str = "x-request-id";

fn request(model: &(impl CompletionModel + Clone)) -> CompletionRequest {
    model.completion_request(PROMPT).max_tokens(16).build()
}

/// The `x-request-id` the single recorded interaction carried.
fn recorded_request_id(scenario: &str) -> Option<String> {
    recorded_response_header(PROVIDER, scenario, 0, REQUEST_ID_HEADER)
}

// ================================================================
// 1. raw is the reply document
// ================================================================

#[tokio::test]
async fn raw_is_the_verbatim_response_body() {
    const SCENARIO: &str = "raw_capture_matrix/raw_round_trips_openai_type";
    let sink = Observed::default();
    with_groq_cassette_result("raw_capture_matrix/raw_round_trips_openai_type", |client| {
        capture_completion(
            client.completion(RAW_CAPTURE_MATRIX_MODEL),
            request,
            sink.clone(),
        )
    })
    .await
    .expect("raw_round_trips_openai_type should replay from its cassette");
    let response = sink.take();

    let (_, body) = recorded_json_turn(PROVIDER, SCENARIO);
    assert!(
        body["choices"][0]["message"]["content"].is_string(),
        "the recorded turn should be a plain text answer"
    );

    // The reply document, not a projection of it: every top-level key the
    // provider sent is reachable and holds the recorded value. The generated
    // id goes through the token helper because a recording pass mints a live
    // one while replay serves the scrubbed fixture back.
    let raw = &response.raw;
    assert_matches_recorded_token(
        raw["id"].as_str(),
        body["id"].as_str(),
        "raw's own response id",
    );
    assert_matches_recorded_document(raw, &body, &["id"], "raw is the provider's document");

    // And it is still the shared chat-completions shape: the typed escape
    // hatch reads it back, and agrees with the normalized identity.
    let typed = openai::CompletionResponse::deserialize(raw)
        .expect("raw is the shared OpenAI chat-completions reply Groq sends");
    assert_matches_recorded_token(
        Some(typed.id.as_str()),
        response.response_id.as_deref(),
        "typed response id",
    );
    assert_eq!(Some(typed.model.as_str()), response.model.as_deref());
    // `x_groq` is Groq's own envelope and no shared type models it — which is
    // exactly why `raw` being the document rather than the parse is the
    // difference between a caller reaching it and losing it.
    assert!(
        serde_json::to_value(&typed)
            .expect("typed serializes")
            .get("x_groq")
            .is_none(),
        "no shared chat-completions type models Groq's envelope"
    );
    assert_eq!(
        raw.get("x_groq"),
        body.get("x_groq"),
        "x_groq reaches the caller through raw: {raw}"
    );
}

// ================================================================
// 2. Fields the normalized response provably lacks
// ================================================================

#[tokio::test]
async fn raw_exposes_queue_time() {
    const SCENARIO: &str = "raw_capture_matrix/raw_exposes_queue_time";
    let sink = Observed::default();
    with_groq_cassette_result("raw_capture_matrix/raw_exposes_queue_time", |client| {
        capture_completion(
            client.completion(RAW_CAPTURE_MATRIX_MODEL),
            request,
            sink.clone(),
        )
    })
    .await
    .expect("raw_exposes_queue_time should replay from its cassette");
    let response = sink.take();

    let (_, body) = recorded_json_turn(PROVIDER, SCENARIO);
    let recorded_queue_time = body["usage"]["queue_time"]
        .as_f64()
        .expect("Groq reports usage.queue_time on every response");
    let recorded_prompt_time = body["usage"]["prompt_time"]
        .as_f64()
        .expect("Groq reports usage.prompt_time on every response");
    let recorded_fingerprint = body["system_fingerprint"]
        .as_str()
        .expect("Groq reports a system_fingerprint");

    let raw = &response.raw;
    assert_eq!(raw["usage"]["queue_time"], json!(recorded_queue_time));
    assert_eq!(raw["usage"]["prompt_time"], json!(recorded_prompt_time));
    assert_matches_recorded_token(
        raw["system_fingerprint"].as_str(),
        Some(recorded_fingerprint),
        "system fingerprint",
    );
    // The typed view models Groq's timings, so the escape hatch is typed
    // rather than a hand-indexed JSON walk.
    let typed = openai::CompletionResponse::deserialize(raw).expect("raw reads back typed");
    let usage = typed.usage.expect("the recorded turn reports usage");
    assert_eq!(usage.queue_time, Some(recorded_queue_time));
    assert_eq!(usage.prompt_time, Some(recorded_prompt_time));

    // And the normalized view has no slot for any of them.
    let normalized_usage = serde_json::to_value(response.usage).expect("usage serializes");
    assert!(
        normalized_usage.get("queue_time").is_none()
            && normalized_usage.get("prompt_time").is_none(),
        "the normalized usage has no timing slots: {normalized_usage}"
    );
    let normalized = serde_json::to_value(&response).expect("response serializes");
    assert!(normalized.get("system_fingerprint").is_none());
}

// ================================================================
// 3. The normalized view and raw tell one story
// ================================================================

#[tokio::test]
async fn normalized_fields_match_raw_renormalized() {
    const SCENARIO: &str = "raw_capture_matrix/normalized_fields_match_raw_renormalized";
    let sink = Observed::default();
    with_groq_cassette_result(
        "raw_capture_matrix/normalized_fields_match_raw_renormalized",
        |client| {
            capture_completion(
                client.completion(RAW_CAPTURE_MATRIX_MODEL),
                request,
                sink.clone(),
            )
        },
    )
    .await
    .expect("normalized_fields_match_raw_renormalized should replay from its cassette");
    let response = sink.take();

    let (_, body) = recorded_json_turn(PROVIDER, SCENARIO);
    chat::assert_reproduces_body(&response, PROVIDER, &body, "the recorded body");
    // Groq contracts `x-request-id`; the recorded header is the premise.
    assert_contracted_request_id(
        response.provider_request_id.as_deref(),
        recorded_request_id(SCENARIO).as_deref(),
        REQUEST_ID_HEADER,
    );

    // One seam, two views: the normalized fields hold against the response's
    // own `raw` exactly as they hold against the fixture bytes, because `raw`
    // *is* those bytes. Capture adds a view; it never changes the mapping.
    let raw = response.raw.clone();
    chat::assert_reproduces_body(&response, PROVIDER, &raw, "the response's own raw");

    // The provider-native view of the same reply, through the typed escape
    // hatch: its own fields are what the decoder mapped from.
    let typed = openai::CompletionResponse::deserialize(&raw).expect("raw reads back typed");
    chat::assert_native_matches_normalized(&response, &typed, "the typed view of raw");
    assert_eq!(
        typed
            .choices
            .first()
            .expect("the reply carries a choice")
            .finish_reason
            .as_str(),
        body["choices"][0]["finish_reason"]
            .as_str()
            .expect("recorded finish reason"),
        "the native finish reason is the recorded one"
    );
}
