//! Typed-route parity on Groq: `raw_completion_with_request_id` →
//! `normalize` → `with_optional_provider_request_id` reproduces everything
//! `CompletionModel::completion` reports.
//!
//! **The contract.** Groq is a compatible provider that reuses the shared
//! [`openai::CompletionResponse`] wire type, so the transport request id —
//! Groq's `x-request-id`, contracted by `Groq::REQUEST_ID_HEADER` — cannot
//! live on the raw type. `raw_completion` therefore drops it, and a caller
//! reassembling a normalized response from the typed escape hatch would
//! silently lack the `provider_request_id` that `completion` reports.
//! `raw_completion_with_request_id` returns the pair so the typed route can
//! reproduce the normalized one exactly.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `raw_with_request_id_reproduces_completion` | typed route vs `completion` | both reproduce their own fixture's identity, finish reason, model and usage; the typed route's id is the recorded `x-request-id` | recorded |
//! | 2 | `plain_raw_completion_lacks_request_id` | `raw_completion` + `normalize` | `provider_request_id == None` although the recorded response carries `x-request-id` | recorded |
//!
//! Both cells are recorded. Cell 1 records the `completion` turn and the
//! `raw_completion_with_request_id` turn as two interactions of one scenario;
//! since two live turns carry two different ids, each side is compared with
//! *its own* interaction's recorded body and `x-request-id` header, and the
//! two sides are then compared field for field where the wire makes them
//! equal (provider, model, finish reason). The premise both cells re-derive
//! from their fixture is that Groq's recorded responses carry the
//! `x-request-id` header at all — otherwise the reassembly step would be
//! vacuous.

use rig::completion::{
    CompletionModel, CompletionRequest, FinishReason, NormalizeCompletionResponse,
};
use rig::prelude::*;
use rig::providers::groq;
use rig::providers::openai::completion::OpenAICompatibleProvider;
use serde_json::Value;

use super::RAW_CAPTURE_MODEL;
use super::support::{
    assert_matches_recorded_token, recorded_response_headers, with_groq_cassette_result,
};

const PROVIDER: &str = "groq";
const PROMPT: &str = "Reply with the single word: pong";

fn request(model: &groq::CompletionModel) -> CompletionRequest {
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

/// The `x-request-id` the recorded interaction at `index` carried — the
/// premise of every cell here.
fn recorded_request_id(scenario: &str, index: usize) -> String {
    recorded_response_headers(scenario)[index]
        .iter()
        .find(|(name, _)| name == "x-request-id").map_or_else(|| {
            panic!("interaction {index} of {scenario} must carry the x-request-id header Groq contracts")
        }, |(_, value)| value.clone())
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
    request_id: &str,
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
    assert_matches_recorded_token(
        identity.provider_request_id.as_deref(),
        Some(request_id),
        &format!("{context}: request id"),
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
    with_groq_cassette_result(
        "raw_completion_parity_matrix/raw_with_request_id_reproduces_completion",
        |client| async move {
            let model = client.completion_model(RAW_CAPTURE_MODEL);
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
        <groq::Groq as OpenAICompatibleProvider>::REQUEST_ID_HEADER,
        Some("x-request-id"),
        "Groq contracts x-request-id"
    );
    let completion_id = recorded_request_id(SCENARIO, 0);
    let typed_id = recorded_request_id(SCENARIO, 1);
    assert_matches_recorded_token(
        request_id.as_deref(),
        Some(typed_id.as_str()),
        "the id raw_completion_with_request_id returned is the recorded header",
    );

    assert_reproduces_fixture(
        &normalized,
        &interactions[0].1,
        &completion_id,
        "completion",
    );
    assert_reproduces_fixture(&reassembled, &interactions[1].1, &typed_id, "typed route");
    // Where the wire makes the two turns equal, the two routes agree.
    assert_eq!(reassembled.provider, normalized.provider);
    assert_eq!(reassembled.model, normalized.model);
    assert_eq!(reassembled.finish_reason(), normalized.finish_reason());
    assert_eq!(
        reassembled.identity().message_id,
        normalized.identity().message_id
    );
    assert!(reassembled.identity().provider_request_id.is_some());
    assert!(normalized.identity().provider_request_id.is_some());
}

// ================================================================
// 2. Plain raw_completion drops the id; completion keeps it
// ================================================================

#[tokio::test]
async fn plain_raw_completion_lacks_request_id() {
    const SCENARIO: &str = "raw_completion_parity_matrix/plain_raw_completion_lacks_request_id";
    with_groq_cassette_result(
        "raw_completion_parity_matrix/plain_raw_completion_lacks_request_id",
        |client| async move {
            let model = client.completion_model(RAW_CAPTURE_MODEL);
            let raw = model.raw_completion(request(&model)).await?;
            let normalized = raw.normalize(PROVIDER)?;
            assert_eq!(
                normalized.provider_request_id, None,
                "the wire type has no slot for the transport id, so plain raw_completion drops it"
            );
            assert!(normalized.response_id.is_some());
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("plain_raw_completion_lacks_request_id should replay from its cassette");

    // The premise: the header was there to drop.
    let request_id = recorded_request_id(SCENARIO, 0);
    assert!(!request_id.trim().is_empty());
}
