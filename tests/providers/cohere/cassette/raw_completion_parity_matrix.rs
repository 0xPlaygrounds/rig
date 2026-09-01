//! Parity contract between Cohere's raw and normalized unary seams.
//!
//! # The contract
//!
//! Cohere is a `TryFrom`-shaped provider: `CompletionModel::completion` is
//! `raw_completion(req)?.try_into()` (`rig::completion::CompletionResponse:
//! TryFrom<cohere::completion::CompletionResponse>`) plus the always-on raw
//! capture seam. So a caller holding the concrete model who prefers the
//! escape hatch and normalizes by hand must land on the same `identity()`,
//! `finish_reason()`, `model` and `usage` that `completion()` reports — and
//! `try_into` over the typed `raw` every `completion()` carries must
//! reproduce that very response.
//!
//! Cohere reports no documented request-id response header (its
//! `x-debug-trace-id` is a debug trace handle, deliberately not adopted — see
//! the `send_completion` call in `cohere/completion.rs`), so
//! [`rig::completion::CompletionResponse::provider_request_id`] is `None` on
//! both sides by design, exactly as its doc allows. Two live requests get two
//! generation ids, so identity parity is asserted exactly on the captured-raw
//! side (same response) and shape-wise across the two requests.
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `raw_try_into_matches_completion` | `raw_completion` + `try_into` vs `completion` | identity / finish reason / model / usage agree | recorded |
//!
//! Recorded (`COHERE_API_KEY` was available) as one scenario with **two**
//! interactions — the raw request first, then the `completion` twin — because
//! the contract is between the two seams; the harness replays interactions in
//! order.

use rig::completion::{CompletionModel as _, CompletionResponse as RigCompletionResponse};
use rig::prelude::*;
use rig::providers::cohere::completion::CompletionResponse;
use serde::Deserialize;
use serde_json::Value;

use super::super::{CASSETTE_MODEL, support::with_cohere_cassette};

const PROVIDER: &str = "cohere";
const PROMPT: &str = "Reply with exactly this one word and nothing else: parity";

#[tokio::test]
async fn raw_try_into_matches_completion() {
    const SCENARIO: &str = "raw_completion_parity_matrix/raw_try_into_matches_completion";
    with_cohere_cassette(
        "raw_completion_parity_matrix/raw_try_into_matches_completion",
        |client| async move {
            let model = client.completion_model(CASSETTE_MODEL);
            let request = || {
                model
                    .completion_request(PROMPT)
                    .temperature(0.0)
                    .max_tokens(16)
                    .build()
            };

            let raw = model
                .raw_completion(request())
                .await
                .expect("raw completion should succeed");
            let via_raw: RigCompletionResponse = raw.try_into().expect("raw should normalize");

            let via_completion = model
                .completion(request())
                .await
                .expect("completion should succeed");

            // Across two live requests: everything the contract names, except
            // that each request gets its own generation id.
            assert_eq!(via_raw.finish_reason(), via_completion.finish_reason());
            assert_eq!(via_raw.model, via_completion.model);
            assert_eq!(
                via_raw.model, None,
                "Cohere's /v2/chat payload names no model on either seam"
            );
            assert_eq!(via_raw.provider, via_completion.provider);
            // Identical request bytes tokenize identically; the output side is
            // the model's to vary.
            assert_eq!(
                via_raw.usage.input_tokens,
                via_completion.usage.input_tokens
            );
            let raw_identity = via_raw.identity();
            let completion_identity = via_completion.identity();
            assert_eq!(raw_identity.message_id, completion_identity.message_id);
            assert_eq!(
                raw_identity.provider_request_id, None,
                "Cohere has no adopted request-id header, so the raw path reports None by design"
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

            // Same response: the captured raw, typed and normalized by hand,
            // reproduces `completion()` exactly.
            let captured = &via_completion.raw;
            let reproduced: RigCompletionResponse = CompletionResponse::deserialize(captured)
                .expect("captured raw is Cohere's own type")
                .try_into()
                .expect("captured raw should normalize");
            assert_eq!(reproduced.identity(), via_completion.identity());
            assert_eq!(reproduced.finish_reason(), via_completion.finish_reason());
            assert_eq!(reproduced.model, via_completion.model);
            assert_eq!(reproduced.usage, via_completion.usage);
            assert_eq!(reproduced.choice, via_completion.choice);
            assert_eq!(reproduced.provider, via_completion.provider);
        },
    )
    .await;

    let bodies = crate::cassettes::recorded_interaction_bodies(PROVIDER, SCENARIO);
    assert_eq!(
        bodies.len(),
        2,
        "{SCENARIO}: the cell records the raw request and then its `completion` twin"
    );
    let (raw_request, _) = &bodies[0];
    let (completion_request, _) = &bodies[1];
    assert_eq!(
        raw_request, completion_request,
        "{SCENARIO}: both seams must send the same request bytes"
    );
    for (_, response_body) in &bodies {
        let response: Value =
            serde_json::from_str(response_body).expect("recorded response should be JSON");
        assert_eq!(
            response.get("finish_reason"),
            Some(&Value::String("COMPLETE".to_string())),
            "{SCENARIO}: both recorded turns should have finished COMPLETE"
        );
    }
}
