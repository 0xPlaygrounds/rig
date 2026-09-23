//! Raw provider response capture on Perplexity's blocking chat-completions
//! path.
//!
//! **The feature.** Every blocking completion attaches the provider's own
//! reply to the normalized [`rig::completion::CompletionResponse::raw`].
//! Capture is always on: there is no flag to request it, nothing about it
//! reaches the wire, and a `Value::Null` only ever means a response built by
//! hand with no provider payload behind it.
//!
//! **What `raw` is.** The driver sets it from the reply's bytes
//! (`driver::call`), so it is the provider's response *document*, not a
//! round-trip through whatever type the decoder happened to parse. That
//! matters here more than for any other provider in this family: Perplexity's
//! wire carries `citations` and `search_results`, which no shared
//! chat-completions type models, and they reach a caller through `raw`
//! precisely because `raw` is the body. Cell 2 pins those two fields against
//! the fixture that has them, so the capability stays stated rather than
//! rediscovered.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `raw_is_the_verbatim_response_body` | body fidelity | `raw` reproduces the recorded reply, field for field | recorded |
//! | 2 | `raw_exposes_object_and_citations` | provider-only fields | `raw.object`, `raw.citations` and `raw.search_results` equal the fixture's, and none of them has a normalized slot | recorded |
//! | 3 | `normalized_fields_match_raw_renormalized` | normalized view | the response reproduces its fixture bytes, and the same checks hold against its own `raw` | recorded |
//!
//! The scenario literals — and therefore the fixture filenames — keep the
//! names they were recorded under; the cell names describe what the cells now
//! assert.
//!
//! Every cell is recorded. Each re-derives its premise from its own fixture
//! after the wrapper returns, so a recording that stopped carrying a usage
//! block, a finish reason or a citations array fails loudly instead of
//! covering nothing. Perplexity contracts no request-id header, so
//! `provider_request_id` is `None` on every turn here — a documented outcome,
//! pinned as such with [`assert_no_request_id`], never folded into the shared
//! body contract. Perplexity's models search the web on every turn, so the
//! prompt is deliberately trivial.

use rig::completion::{CompletionModel, CompletionRequest};
use rig::providers::perplexity;
use serde_json::json;

use super::super::support::with_perplexity_cassette;
use crate::cassettes::recorded_json_turn;
use crate::raw_capture::{assert_no_request_id, capture_completion, chat};
use crate::support::{Observed, assert_matches_recorded_document, assert_matches_recorded_token};

const PROVIDER: &str = "perplexity";
const MODEL: &str = perplexity::SONAR;
const PROMPT: &str = "Reply with the single word: pong";
/// Names the dialect in the "no id header" outcome the cells pin.
const DIALECT: &str = "Perplexity";

fn request(model: &(impl CompletionModel + Clone)) -> CompletionRequest {
    model.completion_request(PROMPT).max_tokens(16).build()
}

// ================================================================
// 1. raw is the reply document
// ================================================================

#[tokio::test]
async fn raw_is_the_verbatim_response_body() {
    const SCENARIO: &str = "raw_capture_matrix/raw_round_trips_openai_type";
    let observed = Observed::default();
    let sink = observed.clone();
    with_perplexity_cassette(
        "raw_capture_matrix/raw_round_trips_openai_type",
        |client| async move {
            capture_completion(client.completion(MODEL), request, sink)
                .await
                .expect("the turn should succeed");
        },
    )
    .await;
    let response = observed.take();

    let (_, body) = recorded_json_turn(PROVIDER, SCENARIO);
    assert!(
        body["choices"][0]["message"]["content"].is_string(),
        "the recorded turn should be a plain text answer"
    );

    // The reply document, not a projection of it: every top-level key the
    // provider sent is reachable, and the values are the recorded ones. The
    // generated id is compared through the token helper because a recording
    // pass mints a live one while replay serves the scrubbed fixture back.
    let raw = &response.raw;
    assert_matches_recorded_token(
        raw["id"].as_str(),
        body["id"].as_str(),
        "raw's own response id",
    );
    assert_matches_recorded_document(raw, &body, &["id"], "raw is the provider's document");
    assert_eq!(
        Some(raw["id"].as_str()),
        Some(response.response_id.as_deref())
    );
}

// ================================================================
// 2. The fields with no normalized slot reach the caller through raw
// ================================================================

#[tokio::test]
async fn raw_exposes_object_and_citations() {
    const SCENARIO: &str = "raw_capture_matrix/raw_exposes_object_not_citations";
    let observed = Observed::default();
    let sink = observed.clone();
    with_perplexity_cassette(
        "raw_capture_matrix/raw_exposes_object_not_citations",
        |client| async move {
            capture_completion(client.completion(MODEL), request, sink)
                .await
                .expect("the turn should succeed");
        },
    )
    .await;
    let response = observed.take();

    let (_, body) = recorded_json_turn(PROVIDER, SCENARIO);
    let recorded_object = body["object"]
        .as_str()
        .expect("Perplexity tags every completion with an object");
    assert!(
        body["citations"].is_array(),
        "the recorded Perplexity turn carries citations: {body}"
    );

    let raw = &response.raw;
    assert_eq!(raw["object"], json!(recorded_object));
    // The normalized view has no slot for the tag.
    let normalized = serde_json::to_value(&response).expect("response serializes");
    assert!(normalized.get("object").is_none());
    // Nor for Perplexity's search evidence — which is exactly why `raw` being
    // the reply document rather than a parsed type's re-serialization is the
    // difference between a caller reaching its citations and losing them.
    assert!(normalized.get("citations").is_none());
    assert_eq!(
        raw.get("citations"),
        body.get("citations"),
        "citations reach the caller through raw: {raw}"
    );
    assert_eq!(
        raw.get("search_results"),
        body.get("search_results"),
        "and so do search results: {raw}"
    );
}

// ================================================================
// 3. The normalized view and raw tell one story
// ================================================================

#[tokio::test]
async fn normalized_fields_match_raw_renormalized() {
    const SCENARIO: &str = "raw_capture_matrix/normalized_fields_match_raw_renormalized";
    let observed = Observed::default();
    let sink = observed.clone();
    with_perplexity_cassette(
        "raw_capture_matrix/normalized_fields_match_raw_renormalized",
        |client| async move {
            capture_completion(client.completion(MODEL), request, sink)
                .await
                .expect("the turn should succeed");
        },
    )
    .await;
    let response = observed.take();

    let (_, body) = recorded_json_turn(PROVIDER, SCENARIO);
    chat::assert_reproduces_body(&response, PROVIDER, &body, "the recorded body");
    // Perplexity contracts no request-id header, so `None` is the documented
    // outcome — its own contract, stated once for the turn rather than per
    // view, because it is a property of the transport and not of the bytes.
    assert_no_request_id(response.provider_request_id.as_deref(), DIALECT);

    // One seam, two views: the normalized fields hold against the response's
    // own `raw` exactly as they hold against the fixture bytes, because `raw`
    // *is* those bytes. Capture adds a view; it never changes the mapping.
    let raw = response.raw.clone();
    chat::assert_reproduces_body(&response, PROVIDER, &raw, "the response's own raw");
}
