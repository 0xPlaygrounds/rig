//! Typed-route parity on OpenRouter: `raw_completion_with_request_id` →
//! `normalize` → `with_optional_provider_request_id` reproduces everything
//! `CompletionModel::completion` reports.
//!
//! **The contract.** OpenRouter declares its own `Response` type
//! ([`openrouter::CompletionResponse`]) on the shared OpenAI-compatible
//! model, and — unlike Groq or Mistral — contracts *no* request-id header:
//! `OpenRouterExt::REQUEST_ID_HEADER` is `None`. The typed route therefore
//! has nothing to reassemble; `raw_completion_with_request_id` returns
//! `(raw, None)` and `raw_completion(..).normalize(..)` already reproduces
//! `completion` exactly. That is still worth pinning: the parity contract
//! must hold trivially for a `None`-contract provider rather than, say,
//! inventing an id from the body, and a future decision to contract a header
//! would move these cells rather than silently changing behaviour.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `raw_with_request_id_reproduces_completion` | typed route vs `completion` | both reproduce their own fixture's identity, finish reason, model and usage; the returned id is `None` on both sides | recorded |
//! | 2 | `plain_raw_completion_matches_completion_without_id` | `raw_completion` + `normalize` | `provider_request_id == None`, exactly as `completion` reports it (`REQUEST_ID_HEADER` is `None`) | recorded |
//!
//! Both cells are recorded. Cell 1 records the `completion` turn and the
//! `raw_completion_with_request_id` turn as two interactions of one scenario;
//! since two live turns carry two different response ids, each side is
//! compared with *its own* interaction's recorded body, and the two sides are
//! then compared field for field where the wire makes them equal (provider,
//! model, finish reason, absence of any transport id). Cell 2 is the
//! `None`-contract twin of Groq's `plain_raw_completion_lacks_request_id`:
//! there is no header to lack, so plain `raw_completion` and `completion`
//! agree — and the cell says so instead of pretending a difference.

use rig::completion::{
    CompletionModel, CompletionRequest, FinishReason, NormalizeCompletionResponse,
};
use rig::prelude::*;
use rig::providers::openai::completion::OpenAICompatibleProvider;
use rig::providers::openrouter;
use serde_json::Value;

use super::super::DEFAULT_MODEL;
use super::super::support::{assert_matches_recorded_token, with_openrouter_cassette_result};

const PROVIDER: &str = "openrouter";
const PROMPT: &str = "Reply with the single word: pong";

fn request(model: &openrouter::CompletionModel) -> CompletionRequest {
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

fn recorded_finish_reason(body: &Value) -> FinishReason {
    match body["choices"][0]["finish_reason"].as_str() {
        Some("stop") => FinishReason::Stop,
        Some("length") => FinishReason::Length,
        other => panic!("recorded turn should finish on stop or length, got {other:?}"),
    }
}

fn assert_reproduces_fixture(
    response: &rig::completion::CompletionResponse,
    body: &Value,
    context: &str,
) {
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
        Some(recorded_finish_reason(body)),
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
            body["usage"]["prompt_tokens"].as_u64().expect("prompt"),
            body["usage"]["completion_tokens"]
                .as_u64()
                .expect("completion"),
            body["usage"]["total_tokens"].as_u64().expect("total"),
        ),
        "{context}: usage"
    );
}

// ================================================================
// 1. The typed route reproduces the normalized one
// ================================================================

#[tokio::test]
async fn raw_with_request_id_reproduces_completion() {
    const SCENARIO: &str = "raw_completion_parity_matrix/raw_with_request_id_reproduces_completion";
    let observed = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = observed.clone();
    with_openrouter_cassette_result(
        "raw_completion_parity_matrix/raw_with_request_id_reproduces_completion",
        |client| async move {
            let model = client.completion_model(DEFAULT_MODEL);
            let normalized = model.completion(request(&model)).await?;
            let (raw, request_id) = model
                .raw_completion_with_request_id(request(&model))
                .await?;
            let reassembled = raw
                .normalize(PROVIDER)?
                .with_optional_provider_request_id(request_id.clone());
            *sink.lock().expect("observation lock") = Some((normalized, reassembled, request_id));
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("raw_with_request_id_reproduces_completion should replay from its cassette");

    let (normalized, reassembled, request_id) = observed
        .lock()
        .expect("observation lock")
        .take()
        .expect("the cell should observe both routes");
    let interactions = recorded_json(SCENARIO);
    assert_eq!(interactions.len(), 2, "one completion turn, one typed turn");
    assert_eq!(
        interactions[0].0, interactions[1].0,
        "both routes build the same request body"
    );

    assert_eq!(
        <openrouter::OpenRouterExt as OpenAICompatibleProvider>::REQUEST_ID_HEADER,
        None,
        "OpenRouter contracts no request-id header"
    );
    assert_eq!(
        request_id, None,
        "with no contracted header the typed route reports no id"
    );

    assert_reproduces_fixture(&normalized, &interactions[0].1, "completion");
    assert_reproduces_fixture(&reassembled, &interactions[1].1, "typed route");
    // Where the wire makes the two turns equal, the two routes agree.
    assert_eq!(reassembled.provider, normalized.provider);
    assert_eq!(reassembled.model, normalized.model);
    assert_eq!(reassembled.finish_reason(), normalized.finish_reason());
    assert_eq!(
        reassembled.identity().provider_request_id,
        normalized.identity().provider_request_id
    );
    assert_eq!(
        reassembled.identity().message_id,
        normalized.identity().message_id
    );
}

// ================================================================
// 2. Plain raw_completion already matches: there is no id to lack
// ================================================================

#[tokio::test]
async fn plain_raw_completion_matches_completion_without_id() {
    const SCENARIO: &str =
        "raw_completion_parity_matrix/plain_raw_completion_matches_completion_without_id";
    let observed = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = observed.clone();
    with_openrouter_cassette_result(
        "raw_completion_parity_matrix/plain_raw_completion_matches_completion_without_id",
        |client| async move {
            let model = client.completion_model(DEFAULT_MODEL);
            let normalized = model.completion(request(&model)).await?;
            let plain = model
                .raw_completion(request(&model))
                .await?
                .normalize(PROVIDER)?;
            *sink.lock().expect("observation lock") = Some((normalized, plain));
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("plain_raw_completion_matches_completion_without_id should replay from its cassette");

    let (normalized, plain) = observed
        .lock()
        .expect("observation lock")
        .take()
        .expect("the cell should observe both routes");
    let interactions = recorded_json(SCENARIO);
    assert_eq!(interactions.len(), 2);
    assert_reproduces_fixture(&normalized, &interactions[0].1, "completion");
    assert_reproduces_fixture(&plain, &interactions[1].1, "plain raw route");
    // The `None`-contract statement itself: neither route has a transport
    // id, so plain `raw_completion` lacks nothing `completion` has.
    assert_eq!(plain.provider_request_id, None);
    assert_eq!(normalized.provider_request_id, None);
    assert_eq!(plain.finish_reason(), normalized.finish_reason());
    assert_eq!(plain.model, normalized.model);
}
