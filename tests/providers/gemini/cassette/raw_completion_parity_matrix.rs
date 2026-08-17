//! Parity contract between Gemini's raw and normalized unary seams.
//!
//! # The contract
//!
//! Gemini is a `TryFrom`-shaped provider: `CompletionModel::completion` is
//! `raw_completion(req)?.try_into()` (`rig::completion::CompletionResponse:
//! TryFrom<GenerateContentResponse>` for the REST route,
//! `TryFrom<Interaction>` for the Interactions API) plus the always-on raw
//! capture seam. So a caller holding the concrete model who prefers the
//! escape hatch and normalizes by hand must land on the same `identity()`,
//! `finish_reason()`, `model` and `usage` that `completion()` reports — and
//! `try_into` over the typed `raw` every `completion()` carries must
//! reproduce that very response.
//!
//! Neither Gemini route reports a transport request-id header (verified
//! against the live API; see the `send_completion` call in
//! `gemini/completion.rs` and `gemini/interactions_api/mod.rs`), so
//! [`rig::completion::CompletionResponse::provider_request_id`] is `None` on
//! both sides by design — its doc names Gemini as the documented `None` case.
//! Two live requests get two `responseId`s, so identity parity is asserted
//! exactly on the captured-raw side (same response) and shape-wise across the
//! two requests.
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `rest_raw_try_into_matches_completion` | `generateContent`: `raw_completion` + `try_into` vs `completion` | identity / finish reason / model / usage agree | recorded |
//! | 2 | `interactions_raw_try_into_matches_completion` | Interactions API: `raw_completion` + `try_into` vs `completion` | identity / finish reason / model / usage agree | recorded |
//!
//! Both cells are recorded (`GEMINI_API_KEY` was available). Each records one
//! scenario with **two** interactions — the raw request first, then the
//! `completion` twin — because the contract is between the two seams; the
//! harness replays interactions in order.

use rig::completion::{CompletionModel as _, CompletionResponse as RigCompletionResponse};
use rig::prelude::*;
use rig::providers::gemini::completion::gemini_api_types::GenerateContentResponse;
use rig::providers::gemini::interactions_api::{Interaction, InteractionsCompletionModel};
use serde::Deserialize;
use serde_json::Value;

use super::super::support::{with_gemini_cassette, with_gemini_interactions_cassette};

const PROVIDER: &str = "gemini";
const REST_MODEL: &str = "gemini-2.5-flash-lite";
const INTERACTIONS_MODEL: &str = "gemini-3-flash-preview";
const PROMPT: &str = "Reply with exactly this one word and nothing else: parity";

/// The parity a caller can rely on across two live requests: everything the
/// contract names, except that each request gets its own response id.
fn assert_cross_request_parity(
    via_raw: &RigCompletionResponse,
    via_completion: &RigCompletionResponse,
) {
    assert_eq!(via_raw.finish_reason(), via_completion.finish_reason());
    assert_eq!(via_raw.model, via_completion.model);
    assert_eq!(via_raw.provider, via_completion.provider);
    // Identical request bytes tokenize identically; the output side is the
    // model's to vary.
    assert_eq!(
        via_raw.usage.input_tokens,
        via_completion.usage.input_tokens
    );

    let raw_identity = via_raw.identity();
    let completion_identity = via_completion.identity();
    assert_eq!(raw_identity.message_id, completion_identity.message_id);
    assert_eq!(
        raw_identity.provider_request_id, None,
        "Gemini sends no request-id header, so the raw path reports None by design"
    );
    assert_eq!(
        completion_identity.provider_request_id, None,
        "and so does `completion()` — the same seam, one request"
    );
    assert!(
        raw_identity
            .response_id
            .as_deref()
            .is_some_and(|id| !id.is_empty())
    );
    assert!(
        completion_identity
            .response_id
            .as_deref()
            .is_some_and(|id| !id.is_empty())
    );
}

/// The exact half of the contract: normalizing the captured raw by hand lands
/// on the very response `completion()` returned.
fn assert_same_response_parity(
    reproduced: &RigCompletionResponse,
    via_completion: &RigCompletionResponse,
) {
    assert_eq!(reproduced.identity(), via_completion.identity());
    assert_eq!(reproduced.finish_reason(), via_completion.finish_reason());
    assert_eq!(reproduced.model, via_completion.model);
    assert_eq!(reproduced.usage, via_completion.usage);
    assert_eq!(reproduced.choice, via_completion.choice);
    assert_eq!(reproduced.provider, via_completion.provider);
}

fn assert_two_recorded_turns(scenario: &str, status_pointer: &str, status: &str) {
    let bodies = crate::cassettes::recorded_interaction_bodies(PROVIDER, scenario);
    assert_eq!(
        bodies.len(),
        2,
        "{scenario}: the cell records the raw request and then its `completion` twin"
    );
    let (raw_request, _) = &bodies[0];
    let (completion_request, _) = &bodies[1];
    assert_eq!(
        raw_request, completion_request,
        "{scenario}: both seams must send the same request bytes"
    );
    for (_, response_body) in &bodies {
        let response: Value =
            serde_json::from_str(response_body).expect("recorded response should be JSON");
        assert_eq!(
            response.pointer(status_pointer),
            Some(&Value::String(status.to_string())),
            "{scenario}: both recorded turns should have finished naturally"
        );
    }
}

#[tokio::test]
async fn rest_raw_try_into_matches_completion() {
    const SCENARIO: &str = "raw_completion_parity_matrix/rest_raw_try_into_matches_completion";
    with_gemini_cassette(
        "raw_completion_parity_matrix/rest_raw_try_into_matches_completion",
        |client| async move {
            let model = client.completion_model(REST_MODEL);
            let request = || model.completion_request(PROMPT).temperature(0.0).build();

            let raw = model
                .raw_completion(request())
                .await
                .expect("raw completion should succeed");
            let via_raw: RigCompletionResponse = raw.try_into().expect("raw should normalize");

            let via_completion = model
                .completion(request())
                .await
                .expect("completion should succeed");

            assert_cross_request_parity(&via_raw, &via_completion);

            // Same response: the captured raw, typed and normalized by hand,
            // reproduces `completion()` exactly.
            let captured = via_completion
                .raw
                .as_deref()
                .expect("a provider-backed response always carries raw");
            let reproduced: RigCompletionResponse = GenerateContentResponse::deserialize(captured)
                .expect("captured raw is Gemini's own type")
                .try_into()
                .expect("captured raw should normalize");
            assert_same_response_parity(&reproduced, &via_completion);
        },
    )
    .await;

    assert_two_recorded_turns(SCENARIO, "/candidates/0/finishReason", "STOP");
}

#[tokio::test]
async fn interactions_raw_try_into_matches_completion() {
    const SCENARIO: &str =
        "raw_completion_parity_matrix/interactions_raw_try_into_matches_completion";
    with_gemini_interactions_cassette(
        "raw_completion_parity_matrix/interactions_raw_try_into_matches_completion",
        |client| async move {
            let model: InteractionsCompletionModel<reqwest::Client> =
                client.completion_model(INTERACTIONS_MODEL);
            let request = || model.completion_request(PROMPT).temperature(0.0).build();

            let raw = model
                .raw_completion(request())
                .await
                .expect("raw completion should succeed");
            let via_raw: RigCompletionResponse = raw.try_into().expect("raw should normalize");

            let via_completion = model
                .completion(request())
                .await
                .expect("completion should succeed");

            assert_cross_request_parity(&via_raw, &via_completion);

            let captured = via_completion
                .raw
                .as_deref()
                .expect("a provider-backed response always carries raw");
            let reproduced: RigCompletionResponse = Interaction::deserialize(captured)
                .expect("captured raw is the Interactions API's own type")
                .try_into()
                .expect("captured raw should normalize");
            assert_same_response_parity(&reproduced, &via_completion);
        },
    )
    .await;

    assert_two_recorded_turns(SCENARIO, "/status", "completed");
}
