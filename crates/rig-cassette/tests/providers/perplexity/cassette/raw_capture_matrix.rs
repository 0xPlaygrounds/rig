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
//! precisely because `raw` is the body.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
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

use rig::completion::CompletionRequest;
use rig::providers::perplexity;

use super::super::support::with_perplexity_cassette;
use crate::cassettes::recorded_json_turn;
use crate::raw_capture::{assert_no_request_id, capture_completion, chat};
use crate::support::Observed;

const PROVIDER: &str = "perplexity";
const MODEL: &str = perplexity::SONAR;
const PROMPT: &str = "Reply with the single word: pong";
/// Names the dialect in the "no id header" outcome the cells pin.
const DIALECT: &str = "Perplexity";

fn request() -> CompletionRequest {
    CompletionRequest::new(PROMPT).max_tokens(16)
}

// ================================================================
// 1. raw is the reply document
// ================================================================

// ================================================================
// 2. The fields with no normalized slot reach the caller through raw
// ================================================================

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
            capture_completion(client.completion(MODEL), request(), sink)
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
