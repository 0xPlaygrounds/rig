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
//! id-carrying claim would be vacuous. Both the recorded-body comparison and
//! the native-beside-normalized one are the shared Responses format
//! contract, [`crate::raw_capture::responses`]; what stays here is the
//! per-interaction bookkeeping and the header premise.

use rig::completion::{CompletionModel, CompletionRequest};
use rig::driver::Bound;
use rig::providers::openai::responses_api;
use rig::providers::openai::wire::OpenAiWire;
use rig::providers::xai;
use serde::Deserialize;

use super::support::with_xai_cassette_result;
use crate::cassettes::{recorded_json_turns, recorded_response_header};
use crate::raw_capture::{
    assert_contracted_request_id, capture_completion, capture_completion_pair, responses,
};
use crate::support::Observed;

const PROVIDER: &str = "xai";
const MODEL: &str = xai::GROK_3_MINI;
const PROMPT: &str = "Reply with the single word: pong";
const REQUEST_ID_HEADER: &str = "x-request-id";

fn request(model: &Bound<OpenAiWire>) -> CompletionRequest {
    model.completion_request(PROMPT).build()
}

/// The `x-request-id` the recorded interaction at `index` carried — the
/// premise of every cell here.
fn recorded_request_id(scenario: &str, index: usize) -> String {
    recorded_response_header(PROVIDER, scenario, index, REQUEST_ID_HEADER).unwrap_or_else(|| {
        panic!("interaction {index} of {scenario} must carry the x-request-id header xAI contracts")
    })
}

/// The provider-native fields the reply document carries, beside the
/// normalized fields the decoder produced from them.
fn assert_maps_provider_fields(response: &rig::completion::CompletionResponse, context: &str) {
    let reply = responses_api::CompletionResponse::deserialize(&response.raw)
        .expect("`raw` is the serialized Responses CompletionResponse");
    responses::assert_native_matches_normalized(response, &reply, context);
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
    let sink = Observed::default();
    with_xai_cassette_result(
        "raw_completion_parity_matrix/raw_normalize_reproduces_completion",
        |client| capture_completion_pair(client.completion(MODEL), request, sink.clone()),
    )
    .await
    .expect("raw_normalize_reproduces_completion should replay from its cassette");

    let (first, second) = sink.take();
    let interactions = recorded_json_turns(PROVIDER, SCENARIO);
    assert_eq!(interactions.len(), 2, "two completion turns");
    assert_eq!(
        interactions[0].0, interactions[1].0,
        "both turns send the same request body"
    );

    let first_id = recorded_request_id(SCENARIO, 0);
    let second_id = recorded_request_id(SCENARIO, 1);
    // Each reply is checked against its own interaction: two live turns carry
    // two different ids, so a cross-comparison would prove nothing.
    responses::assert_reproduces_body(&first, PROVIDER, &interactions[0].1, "first turn");
    assert_contracted_request_id(
        first.identity().provider_request_id.as_deref(),
        Some(first_id.as_str()),
        REQUEST_ID_HEADER,
    );
    responses::assert_reproduces_body(&second, PROVIDER, &interactions[1].1, "second turn");
    assert_contracted_request_id(
        second.identity().provider_request_id.as_deref(),
        Some(second_id.as_str()),
        REQUEST_ID_HEADER,
    );
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
    let sink = Observed::default();
    with_xai_cassette_result(
        "raw_completion_parity_matrix/raw_completion_carries_request_id_on_the_type",
        |client| capture_completion(client.completion(MODEL), request, sink.clone()),
    )
    .await
    .expect("raw_completion_carries_request_id_on_the_type should replay from its cassette");

    let response = sink.take();
    let recorded_id = recorded_request_id(SCENARIO, 0);
    assert_contracted_request_id(
        response.provider_request_id.as_deref(),
        Some(recorded_id.as_str()),
        REQUEST_ID_HEADER,
    );

    let mirrored = &response.raw;
    let reply = responses_api::CompletionResponse::deserialize(mirrored)
        .expect("`raw` is the serialized Responses CompletionResponse");
    // The reply document therefore *does* lack the id on this family — it is
    // the one field the capture cannot carry.
    assert_eq!(
        reply.provider_request_id, None,
        "a reply document has no header to report: {mirrored}"
    );
    // The captured document is the reply body and never invents a field for a
    // header, which is exactly why the normalized response keeps the id
    // beside `raw` rather than inside it.
    assert!(
        mirrored.get("provider_request_id").is_none(),
        "the transport id is not part of the wire body: {mirrored}"
    );
    let (_, body) = &recorded_json_turns(PROVIDER, SCENARIO)[0];
    assert!(body.get("provider_request_id").is_none());
}
