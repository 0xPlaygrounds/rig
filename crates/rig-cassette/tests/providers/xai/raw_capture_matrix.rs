//! Raw provider response capture on xAI's blocking path.
//!
//! **The feature.** Every blocking completion attaches the provider's own
//! reply document — the response body, verbatim — onto the normalized
//! [`rig::completion::CompletionResponse::raw`]. xAI speaks the OpenAI
//! Responses wire, so that document is what the Responses
//! [`CompletionResponse`] reads back; the transport `provider_request_id` is
//! a header rather than a body field, so it is *not* in `raw` (it is on the
//! normalized response instead). Capture is always on: there is
//! no flag to request it, nothing about it reaches the wire, and a
//! `Value::Null` only ever means a response built by hand with no provider
//! payload behind it. `raw` is a second view of the same response, never a
//! substitute for a normalized field. The Responses envelope carries a `status`
//! and, on xAI, a `service_tier` and a `metadata.system_fingerprint` the
//! normalized response has no slot for; those are the fields pinned here as
//! reachable only through `raw`.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `raw_round_trips_responses_type` | typed round trip | `raw` reads back as the Responses `CompletionResponse`, whose fields are the document's | recorded |
//! | 2 | `raw_exposes_status_and_service_tier` | provider-only field | `raw.status`, `raw.service_tier` and `raw.metadata.system_fingerprint` equal the fixture body | recorded |
//! | 3 | `normalized_fields_match_raw_renormalized` | normalized view | the response reproduces its fixture bytes (including the `x-request-id` header) and its fields are the mapping of the provider-native fields in its own `raw` | recorded |
//!
//! Every cell is recorded. Each re-derives its premise from its own fixture
//! after the wrapper returns: cell 2 reads the status and tier out of the
//! recorded body rather than trusting what the typed view reports, and cell 3
//! checks the normalized fields against the recorded body and the recorded
//! `x-request-id` header before reading the provider-native fields back out
//! of `raw`, so a recording that stopped carrying usage, a status, or the id
//! header fails loudly instead of covering nothing. The Responses format
//! contract those checks are written in — the recorded body's identity,
//! finish reason and accounting, and the provider-native view of the same
//! reply — is [`crate::raw_capture::responses`].

use rig::completion::{CompletionModel, CompletionRequest};
use rig::driver::Bound;
use rig::providers::openai::responses_api;
use rig::providers::openai::wire::OpenAiWire;
use rig::providers::xai;
use serde::Deserialize;
use serde_json::json;

use super::support::with_xai_cassette_result;
use crate::cassettes::{recorded_json_turn, recorded_response_header};
use crate::raw_capture::{
    assert_contracted_request_id, assert_normalized_lacks, capture_completion, responses,
};
use crate::support::{Observed, assert_matches_recorded_token, normalized_without_raw};

const PROVIDER: &str = "xai";
const MODEL: &str = xai::GROK_3_MINI;
const PROMPT: &str = "Reply with the single word: pong";
const REQUEST_ID_HEADER: &str = "x-request-id";

fn request(model: &Bound<OpenAiWire>) -> CompletionRequest {
    model.completion_request(PROMPT).build()
}

/// The `x-request-id` the single recorded interaction carried.
fn recorded_request_id(scenario: &str) -> Option<String> {
    recorded_response_header(PROVIDER, scenario, 0, REQUEST_ID_HEADER)
}

// ================================================================
// 1. raw round-trips the Responses type
// ================================================================

#[tokio::test]
async fn raw_round_trips_responses_type() {
    const SCENARIO: &str = "raw_capture_matrix/raw_round_trips_responses_type";
    let sink = Observed::default();
    with_xai_cassette_result(
        "raw_capture_matrix/raw_round_trips_responses_type",
        |client| capture_completion(client.completion(MODEL), request, sink.clone()),
    )
    .await
    .expect("raw_round_trips_responses_type should replay from its cassette");
    let response = sink.take();

    let raw = &response.raw;
    let typed = responses_api::CompletionResponse::deserialize(raw)
        .expect("raw is the Responses CompletionResponse xAI parses into");
    // `raw` is the reply document, so the typed view's fields are the
    // document's fields.
    assert_eq!(Some(typed.id.as_str()), raw["id"].as_str());
    assert_eq!(Some(typed.model.as_str()), raw["model"].as_str());
    assert_eq!(typed.status, responses_api::ResponseStatus::Completed);
    assert_eq!(raw["status"], json!("completed"));
    assert_eq!(Some(typed.id.as_str()), response.response_id.as_deref());
    // The transport id is not part of the reply document, so the capture
    // never carries it — it lives on the normalized response only.
    assert!(raw.get("provider_request_id").is_none());
    assert_eq!(typed.provider_request_id, None);
    assert!(response.provider_request_id.is_some());

    let (_, response_body) = recorded_json_turn(PROVIDER, SCENARIO);
    assert_eq!(response_body["object"], json!("response"));
}

// ================================================================
// 2. Fields the normalized response provably lacks
// ================================================================

#[tokio::test]
async fn raw_exposes_status_and_service_tier() {
    const SCENARIO: &str = "raw_capture_matrix/raw_exposes_status_and_service_tier";
    let sink = Observed::default();
    with_xai_cassette_result(
        "raw_capture_matrix/raw_exposes_status_and_service_tier",
        |client| capture_completion(client.completion(MODEL), request, sink.clone()),
    )
    .await
    .expect("raw_exposes_status_and_service_tier should replay from its cassette");
    let response = sink.take();

    let (_, body) = recorded_json_turn(PROVIDER, SCENARIO);
    let recorded_status = body["status"]
        .as_str()
        .expect("a Responses body carries a status");
    let recorded_tier = body["service_tier"]
        .as_str()
        .expect("xAI reports service_tier on its Responses body");
    let recorded_fingerprint = body["metadata"]["system_fingerprint"]
        .as_str()
        .expect("xAI reports metadata.system_fingerprint");

    let raw = &response.raw;
    assert_eq!(raw["status"], json!(recorded_status));
    assert_eq!(raw["service_tier"], json!(recorded_tier));
    assert_matches_recorded_token(
        raw["metadata"]["system_fingerprint"].as_str(),
        Some(recorded_fingerprint),
        "system fingerprint",
    );
    // And the normalized view has no slot for any of them: the status is
    // folded into a finish reason, the rest has nowhere to go. The capture is
    // cleared out of the serialized view first, so the fields it carries
    // cannot satisfy the check it is the counterexample to.
    let normalized = normalized_without_raw(response);
    assert_normalized_lacks(&normalized, &["status", "service_tier", "metadata"]);
}

// ================================================================
// 3. The normalized view and raw tell one story
// ================================================================

#[tokio::test]
async fn normalized_fields_match_raw_renormalized() {
    const SCENARIO: &str = "raw_capture_matrix/normalized_fields_match_raw_renormalized";
    let sink = Observed::default();
    with_xai_cassette_result(
        "raw_capture_matrix/normalized_fields_match_raw_renormalized",
        |client| capture_completion(client.completion(MODEL), request, sink.clone()),
    )
    .await
    .expect("normalized_fields_match_raw_renormalized should replay from its cassette");
    let response = sink.take();

    let (_, body) = recorded_json_turn(PROVIDER, SCENARIO);
    responses::assert_reproduces_body(&response, PROVIDER, &body, "the recorded body");
    // xAI contracts `x-request-id`; the recorded header is the premise.
    assert_contracted_request_id(
        response.provider_request_id.as_deref(),
        recorded_request_id(SCENARIO).as_deref(),
        REQUEST_ID_HEADER,
    );

    // The normalized fields are the mapping of the provider-native fields in
    // the response's own `raw`: one decoder reads that document once, and
    // this pins what it read rather than a second copy of the mapping.
    let typed = responses_api::CompletionResponse::deserialize(&response.raw)
        .expect("raw is the Responses type");
    responses::assert_native_matches_normalized(&response, &typed, "the typed view of raw");
}
