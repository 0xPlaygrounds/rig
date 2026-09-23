//! Raw provider response capture on OpenRouter's blocking chat-completions
//! path.
//!
//! **The feature.** Every blocking completion attaches the gateway's own reply
//! to the normalized [`rig::completion::CompletionResponse::raw`]. Capture is
//! always on: there is no flag to request it, nothing about it reaches the
//! wire, and a `Value::Null` only ever means a response built by hand with no
//! provider payload behind it.
//!
//! **What `raw` is.** The driver sets it from the reply's bytes
//! (`driver::call`), so it is the provider's response *document*, not a
//! round-trip through whatever type the decoder parsed. That matters most for
//! a gateway: OpenRouter says which upstream served the turn (`provider`) and
//! what it cost (`usage.cost`), and neither has a slot on the normalized
//! response, so they reach a caller through `raw` precisely because `raw` is
//! the body. The document still reads back as OpenRouter's own
//! [`openrouter::CompletionResponse`], which is the typed escape hatch.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `raw_reads_back_as_openrouter_type` | typed read-back | `raw` deserializes into `openrouter::CompletionResponse` and its identity agrees with the normalized response | recorded |
//! | 2 | `raw_exposes_routed_provider` | provider-only fields | `raw.provider` and `raw.usage.cost` equal the fixture body, and neither has a normalized slot | recorded |
//! | 3 | `normalized_fields_match_raw_renormalized` | normalized view | the response reproduces its fixture bytes, and the same checks hold against its own `raw` | recorded |
//!
//! The scenario literals — and therefore the fixture filenames — keep the
//! names they were recorded under; the cell names describe what the cells now
//! assert.
//!
//! Every cell is recorded. Each re-derives its premise from its own fixture
//! after the wrapper returns: cell 2 reads the routed provider out of the
//! recorded body rather than trusting the string the typed view reports, and
//! cell 3 checks the normalized fields against the recorded body before
//! checking them against its own `raw`, so a recording that stopped carrying a
//! usage block or a finish reason fails loudly instead of covering nothing.
//! OpenRouter contracts no request-id header, so `provider_request_id` is
//! `None` on every turn here — a documented outcome, pinned as such.

use rig::completion::{CompletionModel, CompletionRequest};
use rig::providers::openrouter;
use serde::Deserialize as _;
use serde_json::json;

use super::super::DEFAULT_MODEL;
use super::super::support::with_openrouter_cassette_result;
use crate::cassettes::recorded_json_turn;
use crate::raw_capture::{assert_no_request_id, capture_completion, chat};
use crate::support::{Observed, assert_matches_recorded_document, assert_matches_recorded_token};

const PROVIDER: &str = "openrouter";
const PROMPT: &str = "Reply with the single word: pong";

fn request(model: &(impl CompletionModel + Clone)) -> CompletionRequest {
    model.completion_request(PROMPT).max_tokens(16).build()
}

// ================================================================
// 1. raw is the reply document, and reads back as OpenRouter's type
// ================================================================

#[tokio::test]
async fn raw_reads_back_as_openrouter_type() {
    const SCENARIO: &str = "raw_capture_matrix/raw_round_trips_openrouter_type";
    let sink = Observed::default();
    with_openrouter_cassette_result(
        "raw_capture_matrix/raw_round_trips_openrouter_type",
        |client| capture_completion(client.completion(DEFAULT_MODEL), request, sink.clone()),
    )
    .await
    .expect("raw_round_trips_openrouter_type should replay from its cassette");
    let response = sink.take();

    let (_, body) = recorded_json_turn(PROVIDER, SCENARIO);
    assert!(
        body["choices"][0]["message"]["content"].is_string(),
        "the recorded turn should be a plain text answer"
    );

    // The document, not a projection of it: every top-level key the gateway
    // sent is reachable with the recorded value. The generated id goes through
    // the token helper because a recording pass mints a live one while replay
    // serves the scrubbed fixture back.
    let raw = &response.raw;
    assert_matches_recorded_token(
        raw["id"].as_str(),
        body["id"].as_str(),
        "raw's own response id",
    );
    assert_matches_recorded_document(raw, &body, &["id"], "raw is the gateway's document");

    // And it reads back as OpenRouter's own response type — the typed escape
    // hatch — whose identity is the identity the decoder reported.
    let typed = openrouter::CompletionResponse::deserialize(raw)
        .expect("raw is OpenRouter's own CompletionResponse");
    assert_eq!(Some(typed.id.as_str()), response.response_id.as_deref());
    assert_eq!(Some(typed.model.as_str()), response.model.as_deref());
    assert_eq!(
        typed.choices.len(),
        1,
        "the recorded turn carries one candidate"
    );
}

// ================================================================
// 2. Fields with no normalized slot reach the caller through raw
// ================================================================

#[tokio::test]
async fn raw_exposes_routed_provider() {
    const SCENARIO: &str = "raw_capture_matrix/raw_exposes_routed_provider";
    let sink = Observed::default();
    with_openrouter_cassette_result("raw_capture_matrix/raw_exposes_routed_provider", |client| {
        capture_completion(client.completion(DEFAULT_MODEL), request, sink.clone())
    })
    .await
    .expect("raw_exposes_routed_provider should replay from its cassette");
    let response = sink.take();

    let (_, body) = recorded_json_turn(PROVIDER, SCENARIO);
    let recorded_provider = body["provider"]
        .as_str()
        .expect("OpenRouter names the upstream that served the turn");
    let recorded_cost = body["usage"]["cost"]
        .as_f64()
        .expect("OpenRouter reports usage.cost on every response");

    let raw = &response.raw;
    assert_eq!(raw["provider"], json!(recorded_provider));
    assert_eq!(raw["usage"]["cost"], json!(recorded_cost));
    // And the normalized view has no slot for either: `provider` on the
    // normalized response is rig's descriptor name, not the routed upstream.
    assert_eq!(response.provider, PROVIDER);
    let normalized_usage = serde_json::to_value(response.usage).expect("usage serializes");
    assert!(
        normalized_usage.get("cost").is_none(),
        "the normalized usage has no cost slot: {normalized_usage}"
    );
    // The routed upstream is also what OpenRouter's own type calls it.
    let typed = openrouter::CompletionResponse::deserialize(raw)
        .expect("raw is OpenRouter's own CompletionResponse");
    assert_eq!(typed.provider.as_deref(), Some(recorded_provider));
}

// ================================================================
// 3. The normalized view and raw tell one story
// ================================================================

#[tokio::test]
async fn normalized_fields_match_raw_renormalized() {
    const SCENARIO: &str = "raw_capture_matrix/normalized_fields_match_raw_renormalized";
    let sink = Observed::default();
    with_openrouter_cassette_result(
        "raw_capture_matrix/normalized_fields_match_raw_renormalized",
        |client| capture_completion(client.completion(DEFAULT_MODEL), request, sink.clone()),
    )
    .await
    .expect("normalized_fields_match_raw_renormalized should replay from its cassette");
    let response = sink.take();

    let (_, body) = recorded_json_turn(PROVIDER, SCENARIO);
    chat::assert_reproduces_body(&response, PROVIDER, &body, "the recorded body");
    // OpenRouter contracts no request-id header, so `None` is the documented
    // outcome. That is a claim about the dialect rather than about these
    // bytes, so it is stated once rather than once per view.
    assert_no_request_id(response.provider_request_id.as_deref(), "OpenRouter");

    // One seam, two views: the normalized fields hold against the response's
    // own `raw` exactly as they hold against the fixture bytes, because `raw`
    // *is* those bytes. Capture adds a view; it never changes the mapping.
    let raw = response.raw.clone();
    chat::assert_reproduces_body(&response, PROVIDER, &raw, "the response's own raw");
}
