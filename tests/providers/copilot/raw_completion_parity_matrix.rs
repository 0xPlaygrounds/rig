//! Matrix for the typed escape hatch's parity with the normalized path on
//! both Copilot routes.
//!
//! # The contract
//!
//! `raw_completion_with_request_id(req) → (raw, id)`, then
//! `raw.normalize("copilot")?.with_optional_provider_request_id(id)`, must
//! reproduce what `completion(req)` reports for `identity()`,
//! `finish_reason()`, `model` and `usage`. The pair exists because the two
//! routes differ in where the transport id lives:
//!
//! - **chat route** — the wire type is the shared
//!   [`openai::CompletionResponse`], which has no slot for the `x-request-id`
//!   response header; plain `raw_completion(..).normalize(..)` therefore
//!   *lacks* `provider_request_id`, and only the `_with_request_id` pair plus
//!   `with_optional_provider_request_id` reproduces `completion`.
//! - **responses route** — the wire type
//!   ([`responses_api::CompletionResponse`]) carries `provider_request_id`
//!   itself (stamped by the request driver), so `raw_completion(..)` already
//!   holds the id and the pair's second element is that same value.
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `chat_raw_with_request_id_reproduces_completion` | chat route, `_with_request_id` + reassembly | identity/finish_reason/model/usage reproduce `completion`; the id equals the recorded `x-request-id` | unrecorded (no COPILOT credentials in this environment) |
//! | 2 | `chat_plain_raw_completion_lacks_request_id` | chat route, plain `raw_completion` | `normalize` yields `provider_request_id == None` although the recorded response carried `x-request-id` | unrecorded (no COPILOT credentials in this environment) |
//! | 3 | `responses_raw_completion_carries_request_id` | responses route | `raw.provider_request_id` is the recorded `x-request-id`; `normalize` alone reproduces `completion` | unrecorded (no COPILOT credentials in this environment) |
//!
//! Every cell is unrecorded: none of `GITHUB_COPILOT_API_KEY`,
//! `COPILOT_API_KEY`, `COPILOT_GITHUB_ACCESS_TOKEN`/`GITHUB_TOKEN` nor a Copilot
//! OAuth cache was present when this matrix was written, and a fixture is
//! never fabricated. Each cell's premise is that the recorded response
//! headers carry `x-request-id` (allowlisted and placeholdered by the
//! scrubber); a recording without it proves nothing and fails loudly. To
//! record: export `GITHUB_COPILOT_API_KEY`, remove the `#[ignore]` attributes,
//! flip the table to `recorded`, then run
//! `RIG_PROVIDER_TEST_MODE=record cargo test -p rig --all-features --test copilot copilot::raw_completion_parity_matrix -- --nocapture --test-threads=1`
//! and review `tests/cassettes/copilot/raw_completion_parity_matrix/`.

use rig::completion::NormalizeCompletionResponse as _;
use rig::completion::{
    CompletionModel as _, CompletionResponse as RigCompletionResponse, FinishReason,
};
use rig::prelude::*;
use rig::providers::copilot::{self, CopilotCompletionResponse};
use rig::providers::openai;
use rig::providers::openai::responses_api;
use serde::Deserialize;
use serde_json::Value;

use crate::cassettes::{CassetteMode, cassette_path, recorded_interaction_bodies};
use crate::copilot::with_copilot_cassette;

const COPILOT_PROVIDER: &str = "copilot";
const CHAT_MODEL: &str = copilot::GPT_4O;
const RESPONSES_MODEL: &str = copilot::GPT_5_3_CODEX;
const PROMPT: &str = "Reply with exactly the single word: pong";
const REQUEST_ID_HEADER: &str = "x-request-id";

fn request(model: &copilot::CompletionModel) -> rig::completion::CompletionRequest {
    model.completion_request(PROMPT).max_tokens(64).build()
}

/// The recorded response headers of every interaction of a scenario, in wire
/// order. Read straight from the YAML: the shared body readers deliberately
/// expose bodies only, and the transport id is a header.
#[derive(Deserialize)]
struct RecordedInteraction {
    then: RecordedResponse,
}

#[derive(Deserialize)]
struct RecordedResponse {
    #[serde(default)]
    header: Vec<RecordedHeader>,
}

#[derive(Deserialize)]
struct RecordedHeader {
    name: String,
    value: String,
}

fn recorded_response_headers(scenario: &str) -> Vec<Vec<(String, String)>> {
    let path = cassette_path(COPILOT_PROVIDER, scenario);
    let contents = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("cassette {} should be readable: {err}", path.display()));
    serde_yaml::Deserializer::from_str(&contents)
        .map(|document| {
            RecordedInteraction::deserialize(document)
                .unwrap_or_else(|err| panic!("cassette {} should parse: {err}", path.display()))
                .then
                .header
                .into_iter()
                .map(|header| (header.name.to_ascii_lowercase(), header.value))
                .collect()
        })
        .collect()
}

/// The premise: interaction `index` of the scenario recorded an
/// `x-request-id` response header. Returns its (scrubbed, on replay) value.
fn recorded_request_id(scenario: &str, index: usize) -> String {
    let headers = recorded_response_headers(scenario);
    let interaction = headers
        .get(index)
        .unwrap_or_else(|| panic!("{scenario}: interaction {index} should be recorded"));
    interaction
        .iter()
        .find(|(name, _)| name == REQUEST_ID_HEADER)
        .map(|(_, value)| value.clone())
        .unwrap_or_else(|| {
            panic!(
                "{scenario}: interaction {index} must have recorded an `{REQUEST_ID_HEADER}` \
                 response header — without it this cell proves nothing about the transport id"
            )
        })
}

/// Replay reads the placeholdered header back, so the id compares exactly; a
/// live recording sees Copilot's real id while the fixture holds the
/// placeholder, so the claim there is presence and non-emptiness.
fn assert_request_id_matches_recording(live: Option<&str>, recorded: &str, context: &str) {
    match CassetteMode::current() {
        CassetteMode::Replay => assert_eq!(
            live,
            Some(recorded),
            "{context}: provider_request_id must be the recorded x-request-id"
        ),
        CassetteMode::Record => assert!(
            live.is_some_and(|id| !id.trim().is_empty()),
            "{context}: provider_request_id must be populated from x-request-id"
        ),
    }
}

fn recorded_json_bodies(scenario: &str) -> Vec<Value> {
    recorded_interaction_bodies(COPILOT_PROVIDER, scenario)
        .into_iter()
        .map(|(_, response)| {
            serde_json::from_str(&response)
                .unwrap_or_else(|err| panic!("{scenario}: recorded response should be JSON: {err}"))
        })
        .collect()
}

/// Cross-route parity on everything the response body decides identically for
/// two independent turns of the same request.
fn assert_route_parity(via_raw: &RigCompletionResponse, via_completion: &RigCompletionResponse) {
    assert_eq!(via_raw.finish_reason(), via_completion.finish_reason());
    assert_eq!(via_raw.finish_reason(), Some(FinishReason::Stop));
    assert_eq!(via_raw.model, via_completion.model);
    assert_eq!(via_raw.provider, via_completion.provider);
    assert_eq!(via_raw.provider, COPILOT_PROVIDER);
    assert_eq!(
        via_raw.usage.input_tokens,
        via_completion.usage.input_tokens
    );
    let (raw_identity, completion_identity) = (via_raw.identity(), via_completion.identity());
    assert_eq!(
        raw_identity.response_id.is_some(),
        completion_identity.response_id.is_some(),
        "both routes populate the same identity axes"
    );
    assert_eq!(
        raw_identity.message_id.is_some(),
        completion_identity.message_id.is_some()
    );
    assert_eq!(
        raw_identity.provider_request_id.is_some(),
        completion_identity.provider_request_id.is_some()
    );
}

// ---------------------------------------------------------------------------
// 1: chat route — the pair + reassembly reproduces completion
// ---------------------------------------------------------------------------

/// One scenario, two interactions in wire order: the typed route, then
/// `completion`.
#[tokio::test]
#[ignore = "unrecorded (no COPILOT credentials in this environment)"]
async fn chat_raw_with_request_id_reproduces_completion() {
    let scenario = "raw_completion_parity_matrix/chat_raw_with_request_id_reproduces_completion";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = std::sync::Arc::clone(&captured);
    with_copilot_cassette(
        "raw_completion_parity_matrix/chat_raw_with_request_id_reproduces_completion",
        |client| async move {
            let model = client.completion_model(CHAT_MODEL);

            let (raw, id) = model
                .raw_completion_with_request_id(request(&model))
                .await
                .expect("raw completion should succeed");
            assert!(
                matches!(raw, CopilotCompletionResponse::Chat(_)),
                "premise: gpt-4o routes through chat completions"
            );
            assert!(id.is_some(), "the chat route reports x-request-id");
            let via_raw = raw
                .normalize(COPILOT_PROVIDER)
                .expect("raw route must normalize")
                .with_optional_provider_request_id(id.clone());
            assert_eq!(via_raw.identity().provider_request_id, id);

            let via_completion = model
                .completion(request(&model))
                .await
                .expect("completion should succeed");
            assert!(via_completion.identity().provider_request_id.is_some());

            assert_route_parity(&via_raw, &via_completion);
            *sink.lock().expect("capture mutex") = vec![via_raw, via_completion];
        },
    )
    .await;

    let responses = std::mem::take(&mut *captured.lock().expect("capture mutex"));
    let bodies = recorded_json_bodies(scenario);
    assert_eq!(
        bodies.len(),
        2,
        "{scenario}: expected the raw and the completion turns"
    );
    for (index, (response, body)) in responses.iter().zip(&bodies).enumerate() {
        let context = ["raw_completion_with_request_id + normalize", "completion"][index];
        // Each route's id is the header its own interaction recorded.
        assert_request_id_matches_recording(
            response.identity().provider_request_id.as_deref(),
            &recorded_request_id(scenario, index),
            context,
        );
        // And the rest of the normalized surface is that interaction's body.
        let from_wire = openai::CompletionResponse::deserialize(body)
            .expect("recorded body must be a chat-completions response")
            .normalize(COPILOT_PROVIDER)
            .expect("recorded body must normalize");
        assert_eq!(
            response.finish_reason(),
            from_wire.finish_reason(),
            "{context}"
        );
        assert_eq!(response.model, from_wire.model, "{context}");
        assert_eq!(response.usage, from_wire.usage, "{context}");
    }
}

// ---------------------------------------------------------------------------
// 2: chat route — plain raw_completion drops the transport id
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "unrecorded (no COPILOT credentials in this environment)"]
async fn chat_plain_raw_completion_lacks_request_id() {
    let scenario = "raw_completion_parity_matrix/chat_plain_raw_completion_lacks_request_id";
    with_copilot_cassette(
        "raw_completion_parity_matrix/chat_plain_raw_completion_lacks_request_id",
        |client| async move {
            let model = client.completion_model(CHAT_MODEL);
            let raw = model
                .raw_completion(request(&model))
                .await
                .expect("raw completion should succeed");
            let CopilotCompletionResponse::Chat(chat) = &raw else {
                panic!("premise: gpt-4o routes through chat completions");
            };
            assert!(
                serde_json::to_value(chat.as_ref())
                    .expect("wire type should serialize")
                    .get("provider_request_id")
                    .is_none(),
                "the shared chat-completions wire type has no slot for the transport id"
            );
            let normalized = raw
                .normalize(COPILOT_PROVIDER)
                .expect("raw route must normalize");
            assert_eq!(
                normalized.identity().provider_request_id,
                None,
                "plain raw_completion + normalize cannot know the x-request-id — that is \
                 exactly why raw_completion_with_request_id exists"
            );
            assert!(normalized.identity().response_id.is_some());
        },
    )
    .await;

    // The premise that makes the None meaningful: the wire *did* carry one.
    let recorded = recorded_request_id(scenario, 0);
    assert!(!recorded.trim().is_empty());
}

// ---------------------------------------------------------------------------
// 3: responses route — the wire type carries the id itself
// ---------------------------------------------------------------------------

/// One scenario, two interactions in wire order: the typed route, then
/// `completion`.
#[tokio::test]
#[ignore = "unrecorded (no COPILOT credentials in this environment)"]
async fn responses_raw_completion_carries_request_id() {
    let scenario = "raw_completion_parity_matrix/responses_raw_completion_carries_request_id";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = std::sync::Arc::clone(&captured);
    with_copilot_cassette(
        "raw_completion_parity_matrix/responses_raw_completion_carries_request_id",
        |client| async move {
            let model = client.completion_model(RESPONSES_MODEL);

            let (raw, id) = model
                .raw_completion_with_request_id(request(&model))
                .await
                .expect("raw completion should succeed");
            let CopilotCompletionResponse::Responses(responses) = &raw else {
                panic!("premise: a codex model routes through /responses");
            };
            assert!(
                responses.provider_request_id.is_some(),
                "the Responses wire type carries the transport id itself"
            );
            assert_eq!(
                responses.provider_request_id, id,
                "the pair's second element is the id already on the wire type"
            );
            // No reassembly needed on this route: normalize alone keeps it.
            let via_raw = raw
                .normalize(COPILOT_PROVIDER)
                .expect("raw route must normalize");
            assert_eq!(via_raw.identity().provider_request_id, id);

            let via_completion = model
                .completion(request(&model))
                .await
                .expect("completion should succeed");
            assert!(via_completion.identity().provider_request_id.is_some());

            assert_route_parity(&via_raw, &via_completion);
            *sink.lock().expect("capture mutex") = vec![via_raw, via_completion];
        },
    )
    .await;

    let responses = std::mem::take(&mut *captured.lock().expect("capture mutex"));
    let bodies = recorded_json_bodies(scenario);
    assert_eq!(
        bodies.len(),
        2,
        "{scenario}: expected the raw and the completion turns"
    );
    for (index, (response, body)) in responses.iter().zip(&bodies).enumerate() {
        let context = ["raw_completion + normalize", "completion"][index];
        assert_request_id_matches_recording(
            response.identity().provider_request_id.as_deref(),
            &recorded_request_id(scenario, index),
            context,
        );
        let from_wire = responses_api::CompletionResponse::deserialize(body)
            .expect("recorded body must be a Responses envelope")
            .normalize(COPILOT_PROVIDER)
            .expect("recorded body must normalize");
        assert_eq!(
            response.finish_reason(),
            from_wire.finish_reason(),
            "{context}"
        );
        assert_eq!(response.model, from_wire.model, "{context}");
        assert_eq!(response.usage, from_wire.usage, "{context}");
    }
}
