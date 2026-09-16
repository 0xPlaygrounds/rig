//! View parity on xAI: the provider-native fields in
//! [`rig::completion::CompletionResponse::raw`] are what
//! `CompletionModel::completion` reports.
//!
//! **The contract.** xAI runs on the shared Responses model, so the provider
//! reply captured in `raw` is the Responses [`CompletionResponse`] — a type
//! that, unlike the chat-completions wire types, has a `provider_request_id`
//! slot of its own. That slot is nevertheless empty in `raw`: `raw` is the
//! reply *document*, and `x-request-id`
//! (`ResponsesProviderExt::REQUEST_ID_HEADER`, xAI keeps the default) is a
//! header, which the driver stamps onto the normalized response instead. So
//! the two views of one reply differ by exactly that one field. One decoder
//! reads the document once, so what is pinned here is the mapping it
//! performed — the typed fields beside the normalized ones — rather than a
//! second mapping to compare it against.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `raw_normalize_reproduces_completion` | provider-native fields vs `completion` | each turn reproduces its own fixture's identity, finish reason, model and usage, and its normalized fields are the mapping of the typed fields in its own `raw` | recorded |
//! | 2 | `raw_completion_carries_request_id_on_the_type` | where the id rides | the normalized response's `provider_request_id` is the recorded header and the reply document omits it | recorded |
//!
//! Both cells are recorded. Cell 1 records two completion turns as two
//! interactions of one scenario; since two live turns carry two different
//! ids, each turn is compared with *its own* interaction's recorded body and
//! `x-request-id` header, and the two turns are then compared field for field
//! where the wire makes them equal (provider, model, finish reason). The
//! premise both cells re-derive from their fixture is that xAI's recorded
//! responses carry the `x-request-id` header at all — otherwise the
//! id-carrying claim would be vacuous.

use rig::completion::{CompletionModel, CompletionRequest, FinishReason};
use rig::driver::Bound;
use rig::providers::openai::responses_api;
use rig::providers::openai::responses_api::wire::Responses;
use rig::providers::xai;
use serde::Deserialize;
use serde_json::Value;

use super::support::{
    assert_matches_recorded_token, recorded_response_headers, with_xai_cassette_result,
};

const PROVIDER: &str = "xai";
const MODEL: &str = xai::GROK_3_MINI;
const PROMPT: &str = "Reply with the single word: pong";

fn request(model: &Bound<Responses>) -> CompletionRequest {
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
        .find(|(name, _)| name == "x-request-id").map_or_else(|| {
            panic!(
                "interaction {index} of {scenario} must carry the x-request-id header xAI contracts"
            )
        }, |(_, value)| value.clone())
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
            body["usage"]["input_tokens"].as_u64(),
            body["usage"]["output_tokens"].as_u64(),
            body["usage"]["total_tokens"].as_u64(),
        ),
        "{context}: usage"
    );
}

/// The provider-native fields the reply document carries, beside the
/// normalized fields the decoder produced from them.
fn assert_maps_provider_fields(response: &rig::completion::CompletionResponse, context: &str) {
    let reply = responses_api::CompletionResponse::deserialize(&response.raw)
        .expect("`raw` is the serialized Responses CompletionResponse");
    assert_eq!(
        Some(reply.id.as_str()),
        response.response_id.as_deref(),
        "{context}: response id"
    );
    assert_eq!(
        Some(reply.model.as_str()),
        response.model.as_deref(),
        "{context}: model"
    );
    assert_eq!(
        reply.status,
        responses_api::ResponseStatus::Completed,
        "{context}: status"
    );
    assert_eq!(
        response.finish_reason(),
        Some(FinishReason::Stop),
        "{context}: finish reason"
    );
    let usage = reply
        .usage
        .as_ref()
        .expect("the recorded reply carries usage");
    assert_eq!(
        (
            Some(usage.input_tokens),
            Some(usage.output_tokens),
            Some(usage.total_tokens)
        ),
        (
            response.usage.input_tokens,
            response.usage.output_tokens,
            response.usage.total_tokens
        ),
        "{context}: usage"
    );
    assert_eq!(
        reply.provider_request_id, None,
        "{context}: a reply document has no slot for a response header"
    );
    assert!(
        response.provider_request_id.is_some(),
        "{context}: the transport id rides on the normalized view"
    );
}

// ================================================================
// 1. The normalized view is the mapping of the provider's own fields
// ================================================================

#[tokio::test]
async fn raw_normalize_reproduces_completion() {
    const SCENARIO: &str = "raw_completion_parity_matrix/raw_normalize_reproduces_completion";
    let observed = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = observed.clone();
    with_xai_cassette_result(
        "raw_completion_parity_matrix/raw_normalize_reproduces_completion",
        |client| async move {
            let model = client.completion(MODEL);
            let first = model.completion(request(&model)).await?;
            let second = model.completion(request(&model)).await?;
            *sink.lock().expect("observation lock") = Some((first, second));
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("raw_normalize_reproduces_completion should replay from its cassette");

    let (first, second) = observed
        .lock()
        .expect("observation lock")
        .take()
        .expect("the cell should observe both turns");
    let interactions = recorded_json(SCENARIO);
    assert_eq!(interactions.len(), 2, "two completion turns");
    assert_eq!(
        interactions[0].0, interactions[1].0,
        "both turns send the same request body"
    );

    let first_id = recorded_request_id(SCENARIO, 0);
    let second_id = recorded_request_id(SCENARIO, 1);
    assert_reproduces_fixture(&first, &interactions[0].1, &first_id, "first turn");
    assert_reproduces_fixture(&second, &interactions[1].1, &second_id, "second turn");
    assert_maps_provider_fields(&first, "first turn");
    assert_maps_provider_fields(&second, "second turn");
    // Where the wire makes the two turns equal, the two agree.
    assert_eq!(second.provider, first.provider);
    assert_eq!(second.model, first.model);
    assert_eq!(second.finish_reason(), first.finish_reason());
    assert!(first.identity().provider_request_id.is_some());
    assert!(second.identity().provider_request_id.is_some());
}

// ================================================================
// 2. The id rides on the normalized response, not in the reply document
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
            let model = client.completion(MODEL);
            let response = model.completion(request(&model)).await?;
            let reply = responses_api::CompletionResponse::deserialize(&response.raw)
                .expect("`raw` is the serialized Responses CompletionResponse");
            *sink.lock().expect("observation lock") = Some((
                response.provider_request_id,
                response.raw,
                reply.provider_request_id,
            ));
            Ok::<(), anyhow::Error>(())
        },
    )
    .await
    .expect("raw_completion_carries_request_id_on_the_type should replay from its cassette");

    let (normalized_id, mirrored, document_id) = observed
        .lock()
        .expect("observation lock")
        .take()
        .expect("the cell should observe the captured view");
    let recorded_id = recorded_request_id(SCENARIO, 0);
    assert_matches_recorded_token(
        normalized_id.as_deref(),
        Some(recorded_id.as_str()),
        "the normalized response carries the recorded x-request-id",
    );
    // The reply document therefore *does* lack the id on this family — it is
    // the one field the capture cannot carry.
    assert_eq!(
        document_id, None,
        "a reply document has no header to report: {mirrored}"
    );
    // The captured document is the reply body and never invents a field for a
    // header, which is exactly why the normalized response keeps the id
    // beside `raw` rather than inside it.
    assert!(
        mirrored.get("provider_request_id").is_none(),
        "the transport id is not part of the wire body: {mirrored}"
    );
    let (_, body) = &recorded_json(SCENARIO)[0];
    assert!(body.get("provider_request_id").is_none());
}
