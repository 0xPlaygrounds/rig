//! Matrix for the typed escape hatch's parity with the normalized path on
//! both Copilot routes.
//!
//! # The contract
//!
//! One `completion(req)` yields both views of one reply: the normalized
//! [`CompletionResponse`](rig::completion::CompletionResponse), and
//! [`raw`](rig::completion::CompletionResponse::raw) — the route's own reply
//! body, verbatim, which reads back into that route's own response type.
//! The two must agree on `identity()`, `finish_reason()`, `model` and
//! `usage`, because the normalized view is a projection of that exact body,
//! produced by the one decoder — there is no second mapping to compare it
//! against any more.
//!
//! Copilot relays two request shapes, so `raw` has two shapes — the shared
//! [`openai::CompletionResponse`] on the chat-completions route, the
//! [`responses_api::CompletionResponse`] on the Responses route — and the
//! route is decided by the model id alone
//! ([`wire::routes_through_responses`](rig::providers::copilot::wire::routes_through_responses)).
//! `raw` carries no routing tag: the body is the provider's, and the tag was
//! rig's.
//!
//! The transport id is *not* route-dependent any more. The driver captures
//! `x-request-id` from the response headers for both routes and stamps it
//! onto the normalized response, so a cell asserts that the two routes
//! *agree* rather than that one of them loses the id.
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `chat_raw_with_request_id_reproduces_completion` | chat route | `raw` reads back as [`openai::CompletionResponse`], re-normalizes to the same identity/finish_reason/model/usage, and `provider_request_id` is the recorded `x-request-id` | unrecorded (no COPILOT credentials in this environment) |
//! | 2 | `responses_raw_completion_carries_request_id` | responses route | the same, with `raw` reading back as [`responses_api::CompletionResponse`] | unrecorded (no COPILOT credentials in this environment) |
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

use rig::completion::{
    CompletionModel as _, CompletionResponse as RigCompletionResponse, FinishReason,
};
use rig::driver::Bound;
use rig::providers::copilot;
use rig::providers::copilot::wire::CopilotWire;
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

fn request(model: &Bound<CopilotWire>) -> rig::completion::CompletionRequest {
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
        .map_or_else(
            || {
                panic!(
                    "{scenario}: interaction {index} must have recorded an `{REQUEST_ID_HEADER}` \
                 response header — without it this cell proves nothing about the transport id"
                )
            },
            |(_, value)| value.clone(),
        )
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

/// Parity between the two views of one reply: what the normalized response
/// reports, and what `raw` reports once it is read back into the route's own
/// type and converted forward.
/// The chat route's own body agrees with the folded response on every field
/// the body models. The old cell re-normalized `raw` and compared the two
/// results, which only ever proved that two copies of one mapping agreed;
/// with one mapping left, the honest assertion is that the document the
/// provider sent says what the fold reported.
fn assert_chat_parity(typed: &openai::CompletionResponse, folded: &RigCompletionResponse) {
    assert_eq!(Some(typed.model.as_str()), folded.model.as_deref());
    assert_eq!(
        Some(typed.id.as_str()),
        folded.identity().response_id.as_deref()
    );
    assert_eq!(folded.provider, COPILOT_PROVIDER);
    assert_eq!(folded.finish_reason(), Some(FinishReason::Stop));
    let usage = typed.usage.as_ref().expect("the reply reports usage");
    assert_eq!(
        usage.prompt_tokens as u64,
        folded.usage.input_tokens.expect("input tokens")
    );
    assert_eq!(
        usage.total_tokens as u64,
        folded.usage.total_tokens.expect("total tokens")
    );
}

/// The Responses route's own body, same property.
fn assert_responses_parity(
    typed: &responses_api::CompletionResponse,
    folded: &RigCompletionResponse,
) {
    assert_eq!(Some(typed.model.as_str()), folded.model.as_deref());
    assert_eq!(
        Some(typed.id.as_str()),
        folded.identity().response_id.as_deref()
    );
    assert_eq!(folded.provider, COPILOT_PROVIDER);
    assert_eq!(folded.finish_reason(), Some(FinishReason::Stop));
}

#[allow(dead_code)]
fn assert_route_parity(via_raw: &RigCompletionResponse, via_completion: &RigCompletionResponse) {
    assert_eq!(via_raw.finish_reason(), via_completion.finish_reason());
    assert_eq!(via_raw.finish_reason(), Some(FinishReason::Stop));
    assert_eq!(via_raw.model, via_completion.model);
    assert_eq!(via_raw.provider, via_completion.provider);
    assert_eq!(via_raw.usage, via_completion.usage);
    let (raw_identity, completion_identity) = (via_raw.identity(), via_completion.identity());
    assert_eq!(
        raw_identity.response_id, completion_identity.response_id,
        "the normalized view is a projection of this exact body"
    );
    assert_eq!(raw_identity.message_id, completion_identity.message_id);
}

// ---------------------------------------------------------------------------
// 1: chat route — raw and normalized are two views of one reply
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "unrecorded (no COPILOT credentials in this environment)"]
async fn chat_raw_with_request_id_reproduces_completion() {
    let scenario = "raw_completion_parity_matrix/chat_raw_with_request_id_reproduces_completion";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = std::sync::Arc::clone(&captured);
    with_copilot_cassette(
        "raw_completion_parity_matrix/chat_raw_with_request_id_reproduces_completion",
        |client| async move {
            let model = client.completion(CHAT_MODEL);
            assert!(
                matches!(model.wire, CopilotWire::Chat { .. }),
                "premise: gpt-4o routes through chat completions"
            );

            let via_completion = model
                .completion(request(&model))
                .await
                .expect("completion should succeed");
            assert!(
                via_completion.identity().provider_request_id.is_some(),
                "the driver stamps x-request-id on the chat route too"
            );

            // The other view of the same reply: the route's own body. There
            // is one mapping now, so the typed parse is compared field by
            // field against the folded response rather than re-derived
            // through a second implementation of it.
            let typed = openai::CompletionResponse::deserialize(&via_completion.raw)
                .expect("`raw` is the chat route's own reply body");
            assert_chat_parity(&typed, &via_completion);

            *sink.lock().expect("capture mutex") = Some(via_completion);
        },
    )
    .await;

    let response = captured
        .lock()
        .expect("capture mutex")
        .take()
        .expect("the cell ran");
    let bodies = recorded_json_bodies(scenario);
    assert_eq!(bodies.len(), 1, "{scenario}: expected one recorded turn");
    assert_request_id_matches_recording(
        response.identity().provider_request_id.as_deref(),
        &recorded_request_id(scenario, 0),
        "completion",
    );
    assert_eq!(
        response.raw, bodies[0],
        "`raw` is the recorded reply body, verbatim"
    );
}

// ---------------------------------------------------------------------------
// 2: responses route — the same, on the other body shape
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "unrecorded (no COPILOT credentials in this environment)"]
async fn responses_raw_completion_carries_request_id() {
    let scenario = "raw_completion_parity_matrix/responses_raw_completion_carries_request_id";
    let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = std::sync::Arc::clone(&captured);
    with_copilot_cassette(
        "raw_completion_parity_matrix/responses_raw_completion_carries_request_id",
        |client| async move {
            let model = client.completion(RESPONSES_MODEL);
            assert!(
                matches!(model.wire, CopilotWire::Responses { .. }),
                "premise: a codex model routes through /responses"
            );

            let via_completion = model
                .completion(request(&model))
                .await
                .expect("completion should succeed");
            assert!(via_completion.identity().provider_request_id.is_some());

            let typed = responses_api::CompletionResponse::deserialize(&via_completion.raw)
                .expect("`raw` is the Responses route's own reply body");
            assert_responses_parity(&typed, &via_completion);

            *sink.lock().expect("capture mutex") = Some(via_completion);
        },
    )
    .await;

    let response = captured
        .lock()
        .expect("capture mutex")
        .take()
        .expect("the cell ran");
    let bodies = recorded_json_bodies(scenario);
    assert_eq!(bodies.len(), 1, "{scenario}: expected one recorded turn");
    assert_request_id_matches_recording(
        response.identity().provider_request_id.as_deref(),
        &recorded_request_id(scenario, 0),
        "completion",
    );
    assert_eq!(
        response.raw, bodies[0],
        "`raw` is the recorded reply body, verbatim"
    );
}
