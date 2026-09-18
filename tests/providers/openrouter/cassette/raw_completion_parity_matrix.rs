//! Parity between OpenRouter's captured reply document and the normalized
//! response, on a dialect that contracts no transport request-id header.
//!
//! **The contract.** OpenRouter has exactly one unary seam: the chat wire
//! encodes the request, the driver carries it, and the chat decoder folds the
//! reply into a [`rig::completion::CompletionResponse`] whose `raw` holds the
//! gateway's own reply document. Two things follow, and these cells pin both:
//!
//! 1. **`encode` is deterministic.** The same built request produces the same
//!    request bytes every time, so a caller can replay it — and both recorded
//!    turns of each scenario must therefore carry byte-identical bodies.
//! 2. **`raw` is a faithful second view, not a summary.** Read back as
//!    OpenRouter's own [`openrouter::CompletionResponse`], its provider-native
//!    fields are the fields the normalized response reports.
//!
//! **Why there is no third thing to compare.** `OPENROUTER.request_id_header`
//! is `None`, so [`rig::completion::CompletionResponse::provider_request_id`]
//! is `None` on every turn here, exactly as its doc allows. That is still
//! worth pinning: a `None`-contract dialect must report `None` rather than, say,
//! inventing an id from the body, and a future decision to contract a header
//! would move these cells rather than silently changing behaviour.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `raw_reproduces_the_completion_it_rode_on` | captured `raw` vs the response it rode on | identity / finish reason / model / usage agree; both turns send the same bytes; the id is `None` on both sides | recorded |
//! | 2 | `no_request_id_contract_holds_on_both_turns` | the `None` header contract | `provider_request_id == None` on two independent turns, because the dialect names no header | recorded |
//!
//! Both cells are recorded, each as one scenario with **two** interactions,
//! because each needs two independent replies to separate "the same bytes went
//! out twice" from "one reply agreed with itself"; the harness replays
//! interactions in order. Since two live turns carry two different response
//! ids, each side is compared with *its own* interaction's recorded body, and
//! the two are then compared field for field where the wire makes them equal
//! (provider, model, finish reason, absence of any transport id). The scenario
//! literals keep the names they were recorded under; the cell names describe
//! what the cells now assert.

use rig::completion::{CompletionModel, CompletionRequest, CompletionResponse};
use rig::providers::openai;
use rig::providers::openai::wire::OPENROUTER;
use rig::providers::openrouter;
use serde::Deserialize as _;
use serde_json::Value;

use super::super::DEFAULT_MODEL;
use super::super::support::with_openrouter_cassette_result;
use crate::cassettes::recorded_json_turns;
use crate::raw_capture::{assert_no_request_id, capture_completion_pair, chat};
use crate::support::Observed;

const PROVIDER: &str = "openrouter";
const PROMPT: &str = "Reply with the single word: pong";

fn request(model: &(impl CompletionModel + Clone)) -> CompletionRequest {
    model.completion_request(PROMPT).max_tokens(16).build()
}

/// One turn's normalized view against its own interaction's recorded bytes:
/// the shared chat-completions contract, plus the two identity claims that
/// are this dialect's own — a chat reply names no assistant message, and a
/// dialect naming no id header reports no transport id.
fn assert_reproduces_fixture(response: &CompletionResponse, body: &Value, context: &str) {
    chat::assert_reproduces_body(response, PROVIDER, body, context);
    let identity = response.identity();
    assert_eq!(
        identity.message_id, None,
        "{context}: chat has no message id"
    );
    assert_no_request_id(identity.provider_request_id.as_deref(), "OpenRouter");
}

// ================================================================
// 1. raw reproduces the response it rode on
// ================================================================

#[tokio::test]
async fn raw_reproduces_the_completion_it_rode_on() {
    const SCENARIO: &str = "raw_completion_parity_matrix/raw_with_request_id_reproduces_completion";
    let sink = Observed::default();
    with_openrouter_cassette_result(
        "raw_completion_parity_matrix/raw_with_request_id_reproduces_completion",
        |client| capture_completion_pair(client.completion(DEFAULT_MODEL), request, sink.clone()),
    )
    .await
    .expect("raw_with_request_id_reproduces_completion should replay from its cassette");
    let (first, second) = sink.take();

    let interactions = recorded_json_turns(PROVIDER, SCENARIO);
    assert_eq!(interactions.len(), 2, "one turn, then its twin");
    assert_eq!(
        interactions[0].0, interactions[1].0,
        "`encode` is deterministic, so both turns must send the same request bytes"
    );

    assert_eq!(
        OPENROUTER.request_id_header, None,
        "the OpenRouter dialect contracts no request-id header"
    );

    assert_reproduces_fixture(&first, &interactions[0].1, "first turn");
    assert_reproduces_fixture(&second, &interactions[1].1, "second turn");

    // Same response, two views: the document `raw` carries, read back as the
    // chat-completions reply the decoder mapped from, reports the fields the
    // decoder reported.
    let native = openai::CompletionResponse::deserialize(&second.raw)
        .expect("raw is the chat-completions reply OpenRouter serves");
    chat::assert_native_matches_normalized(&second, &native, "the typed view of raw");

    // OpenRouter's own type is the gateway-aware escape hatch over the same
    // document, and it is where the gateway's own finish-reason spelling
    // survives — the shared type requires the word, the normalized response
    // maps it away.
    let typed = openrouter::CompletionResponse::deserialize(&second.raw)
        .expect("raw is OpenRouter's own completion response");
    assert_eq!(
        typed.choices[0].finish_reason.as_deref(),
        interactions[1].1["choices"][0]["finish_reason"].as_str(),
        "the document keeps the gateway's own finish-reason spelling"
    );

    // Where the wire makes the two turns equal, the two turns agree.
    assert_eq!(second.provider, first.provider);
    assert_eq!(second.model, first.model);
    assert_eq!(second.finish_reason(), first.finish_reason());
    assert_eq!(
        second.identity().provider_request_id,
        first.identity().provider_request_id
    );
    assert_eq!(second.identity().message_id, first.identity().message_id);
}

// ================================================================
// 2. The None-contract statement itself
// ================================================================

#[tokio::test]
async fn no_request_id_contract_holds_on_both_turns() {
    const SCENARIO: &str =
        "raw_completion_parity_matrix/plain_raw_completion_matches_completion_without_id";
    let sink = Observed::default();
    with_openrouter_cassette_result(
        "raw_completion_parity_matrix/plain_raw_completion_matches_completion_without_id",
        |client| capture_completion_pair(client.completion(DEFAULT_MODEL), request, sink.clone()),
    )
    .await
    .expect("plain_raw_completion_matches_completion_without_id should replay from its cassette");
    let (first, second) = sink.take();

    let interactions = recorded_json_turns(PROVIDER, SCENARIO);
    assert_eq!(interactions.len(), 2);
    assert_reproduces_fixture(&first, &interactions[0].1, "first turn");
    assert_reproduces_fixture(&second, &interactions[1].1, "second turn");
    // There is no header to read, so neither turn has a transport id and the
    // dialect is where that is stated.
    assert_eq!(OPENROUTER.request_id_header, None);
    assert_no_request_id(first.provider_request_id.as_deref(), "OpenRouter");
    assert_no_request_id(second.provider_request_id.as_deref(), "OpenRouter");
    assert_eq!(first.finish_reason(), second.finish_reason());
    assert_eq!(first.model, second.model);
}
