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
//! and review `crates/rig-cassette/fixtures/cassettes/copilot/raw_completion_parity_matrix/`.

use rig::completion::{CompletionModel as _, FinishReason};
use rig::driver::Bound;
use rig::providers::copilot;
use rig::providers::copilot::wire::CopilotWire;
use rig::providers::openai;
use rig::providers::openai::responses_api;
use rig::providers::openai::wire::OpenAiWire;
use serde::Deserialize;
use serde_json::Value;

use crate::cassettes::{recorded_interaction_bodies, recorded_response_header};
use crate::copilot::with_copilot_cassette_result;
use crate::raw_capture::{assert_contracted_request_id, capture_completion, chat, responses};
use crate::support::Observed;

const COPILOT_PROVIDER: &str = "copilot";
const CHAT_MODEL: &str = copilot::GPT_4O;
const RESPONSES_MODEL: &str = copilot::GPT_5_3_CODEX;
const PROMPT: &str = "Reply with exactly the single word: pong";
const REQUEST_ID_HEADER: &str = "x-request-id";

fn request(model: &Bound<CopilotWire>) -> rig::completion::CompletionRequest {
    model.completion_request(PROMPT).max_tokens(64).build()
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
                matches!(model.wire.wire, OpenAiWire::Chat(_)),
                "premise: gpt-4o routes through chat completions"
            );
            capture_completion(model, request, sink).await
        },
    )
    .await
    .expect("chat_raw_with_request_id_reproduces_completion should replay from its cassette");

    let response = captured.take();
    // The other view of the same reply: the route's own body. There is one
    // mapping now, so the typed parse is compared field by field against the
    // folded response rather than re-derived through a second implementation
    // of it.
    let typed = openai::CompletionResponse::deserialize(&response.raw)
        .expect("`raw` is the chat route's own reply body");
    chat::assert_native_matches_normalized(&response, &typed, "the chat route's own body");
    assert_eq!(response.provider, COPILOT_PROVIDER);
    // The native comparison pins the reason to the body's word; this cell
    // also pins which word a plain answer carries.
    assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
    // Both ids come from the same live reply, so this compares exactly in
    // either cassette mode.
    assert_eq!(
        Some(typed.id.as_str()),
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
                matches!(model.wire.wire, OpenAiWire::Responses(_)),
                "premise: a codex model routes through /responses"
            );
            capture_completion(model, request, sink).await
        },
    )
    .await
    .expect("responses_raw_completion_carries_request_id should replay from its cassette");

    let response = captured.take();
    let typed = responses_api::CompletionResponse::deserialize(&response.raw)
        .expect("`raw` is the Responses route's own reply body");
    responses::assert_native_matches_normalized(
        &response,
        &typed,
        "the Responses route's own body",
    );
    assert_eq!(response.provider, COPILOT_PROVIDER);
    assert_eq!(
        Some(typed.id.as_str()),
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
