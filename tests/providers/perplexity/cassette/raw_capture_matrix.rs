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
//! pinned as such. Perplexity's models search the web on every turn, so the
//! prompt is deliberately trivial.

use rig::completion::{CompletionModel, CompletionRequest, CompletionResponse};
use rig::providers::perplexity;
use serde_json::{Value, json};

use super::super::support::{
    BoundPerplexity, assert_matches_recorded_token, with_perplexity_cassette,
};
use crate::cassettes::recorded_json_turn;
use crate::support::{Observed, assistant_text, recorded_chat_finish_reason};

const PROVIDER: &str = "perplexity";
const MODEL: &str = perplexity::SONAR;
const PROMPT: &str = "Reply with the single word: pong";

fn request(model: &(impl CompletionModel + Clone)) -> CompletionRequest {
    model.completion_request(PROMPT).max_tokens(16).build()
}

/// The normalized fields, checked against the wire bytes that produced them.
fn assert_reproduces_fixture(response: &CompletionResponse, body: &Value) {
    assert_eq!(response.provider, PROVIDER, "provider");
    assert_matches_recorded_token(
        response.response_id.as_deref(),
        body["id"].as_str(),
        "response id",
    );
    assert_eq!(response.model.as_deref(), body["model"].as_str(), "model");
    assert_eq!(
        response.finish_reason(),
        Some(recorded_chat_finish_reason(body)),
        "finish reason"
    );
    assert_eq!(
        response.usage.input_tokens,
        body["usage"]["prompt_tokens"].as_u64(),
        "input tokens"
    );
    assert_eq!(
        response.usage.output_tokens,
        body["usage"]["completion_tokens"].as_u64(),
        "output tokens"
    );
    assert_eq!(
        response.usage.total_tokens,
        body["usage"]["total_tokens"].as_u64(),
        "total tokens"
    );
    assert_eq!(
        assistant_text(&response.choice),
        body["choices"][0]["message"]["content"]
            .as_str()
            .expect("recorded content"),
        "choice text"
    );
    // Perplexity contracts no request-id header, so `None` is the documented
    // outcome.
    assert_eq!(response.provider_request_id, None, "request id");
}

/// Where a cell parks the response its recorded turn produced.
///
/// The cassette-safety scan reads each wrapper call's scenario out of the AST
/// and accepts only a string literal, so the `with_perplexity_cassette(..)`
/// call has to stay inline in every `#[tokio::test]` with its own literal —
/// folding the three near-identical cells into one `observe(scenario)` helper
/// hands the scan a variable and orphans the fixtures. These two helpers share
/// everything except that call.
///
/// The body every cell runs: one recorded turn, parked in `sink`.
async fn run(client: BoundPerplexity, sink: Observed<CompletionResponse>) {
    let model = client.completion(MODEL);
    let response = model
        .completion(request(&model))
        .await
        .expect("the turn should succeed");
    sink.put(response);
}

// ================================================================
// 1. raw is the reply document
// ================================================================

#[tokio::test]
async fn raw_is_the_verbatim_response_body() {
    const SCENARIO: &str = "raw_capture_matrix/raw_round_trips_openai_type";
    let sink = Observed::default();
    with_perplexity_cassette("raw_capture_matrix/raw_round_trips_openai_type", |client| {
        run(client, sink.clone())
    })
    .await;
    let response = sink.take();

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
    for key in body
        .as_object()
        .expect("the recorded reply is a JSON object")
        .keys()
        .filter(|key| key.as_str() != "id")
    {
        assert_eq!(
            raw.get(key),
            body.get(key),
            "raw should carry the provider's `{key}` unchanged"
        );
    }
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
    let sink = Observed::default();
    with_perplexity_cassette(
        "raw_capture_matrix/raw_exposes_object_not_citations",
        |client| run(client, sink.clone()),
    )
    .await;
    let response = sink.take();

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
    let sink = Observed::default();
    with_perplexity_cassette(
        "raw_capture_matrix/normalized_fields_match_raw_renormalized",
        |client| run(client, sink.clone()),
    )
    .await;
    let response = sink.take();

    let (_, body) = recorded_json_turn(PROVIDER, SCENARIO);
    assert_reproduces_fixture(&response, &body);

    // One seam, two views: the normalized fields hold against the response's
    // own `raw` exactly as they hold against the fixture bytes, because `raw`
    // *is* those bytes. Capture adds a view; it never changes the mapping.
    let raw = response.raw.clone();
    assert_reproduces_fixture(&response, &raw);
}
