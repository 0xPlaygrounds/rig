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
//! the body. A caller reads it as JSON under OpenRouter's own field names.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `raw_reads_back_as_openrouter_type` | JSON read-back | `raw` is the reply document and its identity agrees with the normalized response | recorded |
//!
//! The cell re-derives its premise from its fixture after the wrapper returns,
//! so a recording that stopped carrying what it reads fails loudly instead of
//! covering nothing.
//! OpenRouter contracts no request-id header, so `provider_request_id` is
//! `None` on every turn here — a documented outcome, pinned as such.

use rig::completion::CompletionRequest;

use super::super::DEFAULT_MODEL;
use super::super::support::with_openrouter_cassette_result;
use crate::cassettes::recorded_json_turn;
use crate::raw_capture::capture_completion;
use crate::support::{Observed, assert_matches_recorded_document, assert_matches_recorded_token};

const PROVIDER: &str = "openrouter";
const PROMPT: &str = "Reply with the single word: pong";

fn request() -> CompletionRequest {
    CompletionRequest::new(PROMPT).max_tokens(16)
}

// ================================================================
// 1. raw is the reply document, read under OpenRouter's field names
// ================================================================

#[tokio::test]
async fn raw_reads_back_as_openrouter_type() {
    const SCENARIO: &str = "raw_capture_matrix/raw_round_trips_openrouter_type";
    let sink = Observed::default();
    with_openrouter_cassette_result(
        "raw_capture_matrix/raw_round_trips_openrouter_type",
        |client| capture_completion(client.completion(DEFAULT_MODEL), request(), sink.clone()),
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

    // And its identity, read as JSON, is the identity the decoder reported.
    let typed = raw.clone();
    assert_eq!(typed["id"].as_str(), response.response_id());
    assert_eq!(typed["model"].as_str(), response.model());
    assert_eq!(
        typed["choices"].as_array().map(Vec::len),
        Some(1),
        "the recorded turn carries one candidate"
    );
}

// ================================================================
// 2. Fields with no normalized slot reach the caller through raw
// ================================================================

// ================================================================
// 3. The normalized view and raw tell one story
// ================================================================
