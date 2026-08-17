//! Typed-route parity on xAI: `raw_completion` → `normalize` reproduces
//! everything `CompletionModel::completion` reports.
//!
//! **The contract.** xAI runs on the shared Responses model, whose
//! `raw_completion` returns the Responses [`CompletionResponse`] — a type
//! that, unlike the chat-completions wire types, has a `provider_request_id`
//! slot of its own: the request driver stamps the `x-request-id` header
//! (`ResponsesProviderExt::REQUEST_ID_HEADER`, xAI keeps the default) onto
//! the typed value before returning it. So the Responses family has no
//! `raw_completion_with_request_id`: `raw_completion(..).normalize(..)`
//! already carries the id, and `with_optional_provider_request_id` is a no-op
//! re-attachment. Both are pinned here so the chat-completions and Responses
//! escape hatches are documented as agreeing on the outcome while differing
//! in where the id rides.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `raw_normalize_reproduces_completion` | typed route vs `completion` | both reproduce their own fixture's identity, finish reason, model and usage; both ids are the recorded `x-request-id` | recorded |
//! | 2 | `raw_completion_carries_request_id_on_the_type` | where the id rides | the typed value's `provider_request_id` is the recorded header, and its mirrored serialization omits it | recorded |
//!
//! Both cells are recorded. Cell 1 records the `completion` turn and the
//! `raw_completion` turn as two interactions of one scenario; since two live
//! turns carry two different ids, each side is compared with *its own*
//! interaction's recorded body and `x-request-id` header, and the two sides
//! are then compared field for field where the wire makes them equal
//! (provider, model, finish reason). The premise both cells re-derive from
//! their fixture is that xAI's recorded responses carry the `x-request-id`
//! header at all — otherwise the id-carrying claim would be vacuous.

use rig::completion::{
    CompletionModel, CompletionRequest, FinishReason, NormalizeCompletionResponse,
};
use rig::prelude::*;
use rig::providers::xai;
use serde_json::Value;

use super::support::{
    assert_matches_recorded_token, recorded_response_headers, with_xai_cassette_result,
};

const PROVIDER: &str = "xai";
const MODEL: &str = xai::GROK_3_MINI;
const PROMPT: &str = "Reply with the single word: pong";

fn request(model: &xai::CompletionModel) -> CompletionRequest {
    model.completion_request(PROMPT).build()
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
        .find(|(name, _)| name == "x-request-id")
        .map(|(_, value)| value.clone())
        .unwrap_or_else(|| {
            panic!(
                "interaction {index} of {scenario} must carry the x-request-id header xAI contracts"
            )
        })
}

fn recorded_message_id(body: &Value) -> &str {
    body["output"]
        .as_array()
        .expect("output items")
        .iter()
        .find(|item| item["type"] == "message")
        .and_then(|item| item["id"].as_str())
        .expect("the recorded turn carries a message item with an id")
}

fn recorded_finish_reason(body: &Value) -> FinishReason {
    match body["status"].as_str() {
        Some("completed") => FinishReason::Stop,
        other => panic!("recorded turn should have completed, got {other:?}"),
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
    assert_matches_recorded_token(
        identity.message_id.as_deref(),
        Some(recorded_message_id(body)),
        &format!("{context}: message id"),
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
            body["usage"]["input_tokens"].as_u64().expect("input"),
            body["usage"]["output_tokens"].as_u64().expect("output"),
            body["usage"]["total_tokens"].as_u64().expect("total"),
        ),
        "{context}: usage"
    );
}

// ================================================================
// 1. The typed route reproduces the normalized one
// ================================================================

#[tokio::test]
async fn raw_normalize_reproduces_completion() {
    const SCENARIO: &str = "raw_completion_parity_matrix/raw_normalize_reproduces_completion";
    let observed = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = observed.clone();
    with_xai_cassette_result(
        "raw_completion_parity_matrix/raw_normalize_reproduces_completion",
        |client| async move {
            let model = client.completion_model(MODEL);
            let normalized = model.completion(request(&model)).await?;
            let raw = model.raw_completion(request(&model)).await?;
            let request_id = raw.provider_request_id.clone();
            // The re-attachment is a no-op for the Responses family — the id
            // already rode in on the typed value — and pinning it proves the
            // reassembly recipe is safe to apply uniformly across families.
            let reassembled = raw
                .normalize(PROVIDER)?
                .with_optional_provider_request_id(request_id.clone());
            *sink.lock().expect("observation lock") = Some((normalized, reassembled, request_id));
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("raw_normalize_reproduces_completion should replay from its cassette");

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

    let completion_id = recorded_request_id(SCENARIO, 0);
    let typed_id = recorded_request_id(SCENARIO, 1);
    assert_matches_recorded_token(
        request_id.as_deref(),
        Some(typed_id.as_str()),
        "the id the typed value carried is the recorded header",
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
    assert!(reassembled.identity().provider_request_id.is_some());
    assert!(normalized.identity().provider_request_id.is_some());
}

// ================================================================
// 2. The id rides on the typed value, not in its mirrored body
// ================================================================

#[tokio::test]
async fn raw_completion_carries_request_id_on_the_type() {
    const SCENARIO: &str =
        "raw_completion_parity_matrix/raw_completion_carries_request_id_on_the_type";
    let observed = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = observed.clone();
    with_xai_cassette_result(
        "raw_completion_parity_matrix/raw_completion_carries_request_id_on_the_type",
        |client| async move {
            let model = client.completion_model(MODEL);
            let raw = model.raw_completion(request(&model)).await?;
            let typed_id = raw.provider_request_id.clone();
            let mirrored = serde_json::to_value(&raw)?;
            let normalized = raw.normalize(PROVIDER)?;
            *sink.lock().expect("observation lock") = Some((typed_id, mirrored, normalized));
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("raw_completion_carries_request_id_on_the_type should replay from its cassette");

    let (typed_id, mirrored, normalized) = observed
        .lock()
        .expect("observation lock")
        .take()
        .expect("the cell should observe the typed value");
    let recorded_id = recorded_request_id(SCENARIO, 0);
    assert_matches_recorded_token(
        typed_id.as_deref(),
        Some(recorded_id.as_str()),
        "the typed value carries the recorded x-request-id",
    );
    // Plain `raw_completion` + `normalize` therefore does *not* lack the id
    // on this family — the mirror image of Groq's chat-completions cell.
    assert_matches_recorded_token(
        normalized.provider_request_id.as_deref(),
        Some(recorded_id.as_str()),
        "normalize carries the id forward",
    );
    // The mirrored serialization is the wire body and never invents a field
    // for a header, which is exactly why the normalized response keeps the id
    // beside `raw` rather than inside it.
    assert!(
        mirrored.get("provider_request_id").is_none(),
        "the transport id is not part of the wire body: {mirrored}"
    );
    let (_, body) = &recorded_json(SCENARIO)[0];
    assert!(body.get("provider_request_id").is_none());
}
