//! Matrix for the reply document's parity with the normalized path on both
//! Copilot routes.
//!
//! One `completion(req)` yields both views of one reply: the normalized
//! [`CompletionResponse`](rig::completion::CompletionResponse), and
//! [`raw`](rig::completion::CompletionResponse::raw), the route's own reply
//! body, verbatim, read as JSON. The two must agree on `identity()`,
//! `finish_reason()`, `model` and `usage`, because the normalized view is a
//! projection of that exact body, produced by the one decoder. Copilot relays
//! two request shapes, so `raw` is a chat-completions reply body on the
//! chat-completions route and the Responses response object on the Responses
//! route. The route is decided by the model id alone
//! ([`wire::routes_through_responses`](rig::providers::copilot::wire::routes_through_responses)).
//! The driver stamps `x-request-id` from the response headers onto the
//! normalized response on both routes, so the cells assert the routes agree.
//!
//! The cells are ignored until recorded, because no Copilot credentials were
//! available and a fixture is never fabricated. To record, export
//! `GITHUB_COPILOT_API_KEY`, remove the `#[ignore]` attributes, and run
//! `RIG_PROVIDER_TEST_MODE=record cargo test -p rig-cassette --all-features --test integration copilot::raw_completion_parity_matrix -- --nocapture --test-threads=1`.
//! A recording without `x-request-id` fails loudly.

use rig::completion::FinishReason;
use rig::providers::copilot;
use rig::providers::openai::wire::OpenAiWire;
use serde_json::Value;

use crate::cassettes::{recorded_interaction_bodies, recorded_response_header};
use crate::copilot::with_copilot_cassette_result;
use crate::raw_capture::{assert_contracted_request_id, capture_completion, chat, responses};
use crate::support::Observed;
use rig::completion::CompletionRequest;

const COPILOT_PROVIDER: &str = "copilot";
const CHAT_MODEL: &str = copilot::GPT_4O;
const RESPONSES_MODEL: &str = copilot::GPT_5_3_CODEX;
const PROMPT: &str = "Reply with exactly the single word: pong";
const REQUEST_ID_HEADER: &str = "x-request-id";

fn request() -> rig::completion::CompletionRequest {
    CompletionRequest::new(PROMPT).max_tokens(64)
}

/// The `x-request-id` interaction `index` of the scenario recorded. The
/// transport id is a header, so this reads the fixture's header side; that a
/// recording carries one at all is the premise
/// [`assert_contracted_request_id`] states.
fn recorded_request_id(scenario: &str, index: usize) -> Option<String> {
    recorded_response_header(COPILOT_PROVIDER, scenario, index, REQUEST_ID_HEADER)
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

// ---------------------------------------------------------------------------
// 1: chat route — raw and normalized are two views of one reply
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "unrecorded (no COPILOT credentials in this environment)"]
async fn chat_raw_with_request_id_reproduces_completion() {
    let scenario = "raw_completion_parity_matrix/chat_raw_with_request_id_reproduces_completion";
    let captured = Observed::default();
    let sink = captured.clone();
    with_copilot_cassette_result(
        "raw_completion_parity_matrix/chat_raw_with_request_id_reproduces_completion",
        |client| async move {
            let model = client.completion(CHAT_MODEL);
            assert!(
                matches!(model.wire, OpenAiWire::Chat(_)),
                "premise: gpt-4o routes through chat completions"
            );
            capture_completion(model, request(), sink).await
        },
    )
    .await
    .expect("chat_raw_with_request_id_reproduces_completion should replay from its cassette");

    let response = captured.take();
    // The other view of the same reply: the route's own body. There is one
    // mapping now, so the body, read as JSON, is compared field by field
    // against the folded response rather than re-derived through a second
    // implementation of it.
    let typed = response.raw.clone();
    chat::assert_native_matches_normalized(&response, &typed, "the chat route's own body");
    assert_eq!(response.provider(), COPILOT_PROVIDER);
    // The native comparison pins the reason to the body's word; this cell
    // also pins which word a plain answer carries.
    assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
    // Both ids come from the same live reply, so this compares exactly in
    // either cassette mode.
    assert_eq!(
        typed["id"].as_str(),
        response.identity().response_id.as_deref()
    );

    let bodies = recorded_json_bodies(scenario);
    assert_eq!(bodies.len(), 1, "{scenario}: expected one recorded turn");
    assert_contracted_request_id(
        response.identity().provider_request_id.as_deref(),
        recorded_request_id(scenario, 0).as_deref(),
        REQUEST_ID_HEADER,
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
    let captured = Observed::default();
    let sink = captured.clone();
    with_copilot_cassette_result(
        "raw_completion_parity_matrix/responses_raw_completion_carries_request_id",
        |client| async move {
            let model = client.completion(RESPONSES_MODEL);
            assert!(
                matches!(model.wire, OpenAiWire::Responses(_)),
                "premise: a codex model routes through /responses"
            );
            capture_completion(model, request(), sink).await
        },
    )
    .await
    .expect("responses_raw_completion_carries_request_id should replay from its cassette");

    let response = captured.take();
    let typed = &response.raw;
    responses::assert_native_matches_normalized(&response, typed, "the Responses route's own body");
    assert_eq!(response.provider(), COPILOT_PROVIDER);
    assert_eq!(
        typed["id"].as_str(),
        response.identity().response_id.as_deref()
    );

    let bodies = recorded_json_bodies(scenario);
    assert_eq!(bodies.len(), 1, "{scenario}: expected one recorded turn");
    assert_contracted_request_id(
        response.identity().provider_request_id.as_deref(),
        recorded_request_id(scenario, 0).as_deref(),
        REQUEST_ID_HEADER,
    );
    assert_eq!(
        response.raw, bodies[0],
        "`raw` is the recorded reply body, verbatim"
    );
}
