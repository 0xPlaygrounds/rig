//! Raw provider response capture on OpenRouter's blocking chat-completions
//! path.
//!
//! Every blocking completion attaches the gateway's own reply to the
//! normalized [`rig::completion::CompletionResponse::raw`]. Capture is always
//! on, and a `Value::Null` only means a response built by hand. The driver
//! sets `raw` from the reply's bytes (`driver::call`), so it is the provider's
//! response document. The upstream that served the turn (`provider`) and its
//! cost (`usage.cost`) have no normalized slot and reach a caller through
//! `raw` under OpenRouter's own field names.
//!
//! The cell re-derives its premise from its fixture, so a recording that
//! stopped carrying what it reads fails loudly. OpenRouter contracts no
//! request-id header, so `provider_request_id` is `None` on every turn.

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
