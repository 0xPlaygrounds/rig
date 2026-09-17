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
use rig::providers::openai::wire::OPENROUTER;
use rig::providers::openrouter;
use serde::Deserialize as _;
use serde_json::Value;

use super::super::DEFAULT_MODEL;
use super::super::support::{
    BoundOpenRouter, assert_matches_recorded_token, with_openrouter_cassette_result,
};
use crate::support::{Observed, recorded_chat_finish_reason};

const PROVIDER: &str = "openrouter";
const PROMPT: &str = "Reply with the single word: pong";

fn request(model: &(impl CompletionModel + Clone)) -> CompletionRequest {
    model.completion_request(PROMPT).max_tokens(16).build()
}

fn recorded_json(scenario: &str) -> Vec<(Value, Value)> {
    crate::cassettes::recorded_interaction_bodies(PROVIDER, scenario)
        .into_iter()
        .map(|(request, response)| {
            (
                serde_json::from_str(&request).expect("recorded request should be JSON"),
                serde_json::from_str(&response).expect("recorded response should be JSON"),
            )
        })
        .collect()
}

fn assert_reproduces_fixture(response: &CompletionResponse, body: &Value, context: &str) {
    assert_eq!(response.provider, PROVIDER, "{context}: provider");
    let identity = response.identity();
    assert_eq!(
        identity.message_id, None,
        "{context}: chat has no message id"
    );
    assert_matches_recorded_token(
        identity.response_id.as_deref(),
        body["id"].as_str(),
        &format!("{context}: response id"),
    );
    assert_eq!(
        identity.provider_request_id, None,
        "{context}: OpenRouter contracts no request-id header"
    );
    assert_eq!(
        response.finish_reason(),
        Some(recorded_chat_finish_reason(body)),
        "{context}: finish reason"
    );
    assert_eq!(
        response.model.as_deref(),
        body["model"].as_str(),
        "{context}: model"
    );
    assert_eq!(
        (
            response.usage.input_tokens,
            response.usage.output_tokens,
            response.usage.total_tokens
        ),
        (
            body["usage"]["prompt_tokens"].as_u64(),
            body["usage"]["completion_tokens"].as_u64(),
            body["usage"]["total_tokens"].as_u64(),
        ),
        "{context}: usage"
    );
}

/// Where a cell parks the two responses its recorded turns produced.
///
/// The wrapper call itself stays inline in every cell with its own scenario
/// literal: `cassette_safety` reads the registered scenarios out of the AST
/// and accepts only a literal, so a shared helper that took the scenario as a
/// parameter would register nothing and orphan the fixture.
///
/// The body both cells share: the same built request, sent twice.
async fn run_two_turns(
    client: BoundOpenRouter,
    sink: Observed<(CompletionResponse, CompletionResponse)>,
) -> Result<(), anyhow::Error> {
    let model = client.completion(DEFAULT_MODEL);
    let first = model.completion(request(&model)).await?;
    let second = model.completion(request(&model)).await?;
    sink.put((first, second));
    Ok(())
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
        |client| run_two_turns(client, sink.clone()),
    )
    .await
    .expect("raw_with_request_id_reproduces_completion should replay from its cassette");
    let (first, second) = sink.take();

    let interactions = recorded_json(SCENARIO);
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

    // Same response, two views: the document `raw` carries, read back with
    // OpenRouter's own type, reports the fields the decoder reported.
    let typed = openrouter::CompletionResponse::deserialize(&second.raw)
        .expect("raw is OpenRouter's own completion response");
    assert_eq!(Some(typed.id.as_str()), second.response_id.as_deref());
    assert_eq!(Some(typed.model.as_str()), second.model.as_deref());
    assert_eq!(
        typed
            .usage
            .as_ref()
            .and_then(|usage| u64::try_from(usage.prompt_tokens).ok()),
        second.usage.input_tokens,
        "the document's prompt tokens are the normalized input tokens"
    );
    assert_eq!(
        typed
            .usage
            .as_ref()
            .and_then(|usage| u64::try_from(usage.completion_tokens).ok()),
        second.usage.output_tokens,
        "and its completion tokens are the normalized output tokens"
    );
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
        |client| run_two_turns(client, sink.clone()),
    )
    .await
    .expect("plain_raw_completion_matches_completion_without_id should replay from its cassette");
    let (first, second) = sink.take();

    let interactions = recorded_json(SCENARIO);
    assert_eq!(interactions.len(), 2);
    assert_reproduces_fixture(&first, &interactions[0].1, "first turn");
    assert_reproduces_fixture(&second, &interactions[1].1, "second turn");
    // There is no header to read, so neither turn has a transport id and the
    // dialect is where that is stated.
    assert_eq!(OPENROUTER.request_id_header, None);
    assert_eq!(first.provider_request_id, None);
    assert_eq!(second.provider_request_id, None);
    assert_eq!(first.finish_reason(), second.finish_reason());
    assert_eq!(first.model, second.model);
}
