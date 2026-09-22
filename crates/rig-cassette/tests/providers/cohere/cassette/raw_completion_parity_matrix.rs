//! Parity contract between Cohere's captured raw payload and the normalized
//! unary response.
//!
//! # The contract
//!
//! Cohere has exactly one unary seam: the chat wire encodes the request, the
//! driver carries it, and `ChatDecoder` folds the reply into a
//! [`rig::completion::CompletionResponse`] whose `raw` holds Cohere's own
//! [`cohere::completion::CompletionResponse`] verbatim. Two things follow, and
//! this cell pins both:
//!
//! 1. **`encode` is deterministic.** The same built request produces the same
//!    request bytes every time, so a caller can replay it.
//! 2. **`raw` is a faithful second view, not a summary.** Deserializing it
//!    into Cohere's own type and normalizing it by hand must reproduce the
//!    very response it came attached to — identity, finish reason, model,
//!    usage, choice and provider.
//!
//! Cohere reports no documented request-id response header (its
//! `x-debug-trace-id` is a debug trace handle, deliberately not adopted), so
//! [`rig::completion::CompletionResponse::provider_request_id`] is `None`,
//! exactly as its doc allows. Two live requests get two generation ids, so
//! identity equality is asserted on the captured-raw side (same response) and
//! shape-wise across the two requests.
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `raw_try_into_matches_completion` | captured `raw` + `try_into` vs the response it rode on | identity / finish reason / model / usage agree | recorded |
//!
//! Recorded (`COHERE_API_KEY` was available) as one scenario with **two**
//! interactions, because the cell needs two independent replies to separate
//! "the same bytes went out twice" from "one reply agreed with itself"; the
//! harness replays interactions in order.

use rig::completion::{CompletionModel, CompletionResponse as RigCompletionResponse, FinishReason};
use rig::providers::cohere::completion::{CompletionResponse, FinishReason as CohereFinishReason};
use serde::Deserialize;
use serde_json::Value;

use super::super::{CASSETTE_MODEL, support::with_cohere_cassette};
use crate::raw_capture::capture_completion_pair;
use crate::support::Observed;

const PROVIDER: &str = "cohere";
const PROMPT: &str = "Reply with exactly this one word and nothing else: parity";

fn request(model: &(impl CompletionModel + Clone)) -> rig::completion::CompletionRequest {
    model
        .completion_request(PROMPT)
        .temperature(0.0)
        .max_tokens(16)
        .build()
}

#[tokio::test]
async fn raw_try_into_matches_completion() {
    const SCENARIO: &str = "raw_completion_parity_matrix/raw_try_into_matches_completion";
    let observed: Observed<(RigCompletionResponse, RigCompletionResponse)> = Observed::default();
    let sink = observed.clone();
    with_cohere_cassette(
        "raw_completion_parity_matrix/raw_try_into_matches_completion",
        |client| async move {
            capture_completion_pair(client.completion(CASSETTE_MODEL), request, sink)
                .await
                .expect("the same request should succeed twice");
        },
    )
    .await;

    let (first, second) = observed.take();

    // Across two live requests: everything the contract names, except that
    // each request gets its own generation id.
    assert_eq!(first.finish_reason(), second.finish_reason());
    assert_eq!(first.model, second.model);
    assert_eq!(
        first.model, None,
        "Cohere's /v2/chat payload names no model"
    );
    assert_eq!(first.provider, second.provider);
    // Identical request bytes tokenize identically; the output side is the
    // model's to vary.
    assert_eq!(first.usage.input_tokens, second.usage.input_tokens);
    let first_identity = first.identity();
    let second_identity = second.identity();
    assert_eq!(first_identity.message_id, second_identity.message_id);
    assert_eq!(
        first_identity.provider_request_id, None,
        "Cohere has no adopted request-id header, so the driver reports None by design"
    );
    assert_eq!(
        second_identity.provider_request_id, None,
        "and so does the second reply — the same seam, the same header set"
    );
    assert!(
        first_identity
            .response_id
            .as_deref()
            .is_some_and(|id| !id.is_empty())
    );
    assert!(
        second_identity
            .response_id
            .as_deref()
            .is_some_and(|id| !id.is_empty())
    );

    // Same response, two views: the captured payload typed as Cohere's own
    // says what the normalized response says. There is one mapping now, so
    // the honest assertion is provider-native field against normalized
    // field, not a second normalization compared with the first.
    let typed =
        CompletionResponse::deserialize(&second.raw).expect("captured raw is Cohere's own type");
    assert_eq!(
        Some(typed.id.as_str()),
        second.identity().response_id.as_deref(),
        "the payload's generation id is the response id"
    );
    assert_eq!(
        typed.finish_reason,
        CohereFinishReason::Complete,
        "the recorded turn completed naturally in Cohere's vocabulary"
    );
    assert_eq!(
        second.finish_reason(),
        Some(FinishReason::Stop),
        "and the decoder maps COMPLETE onto a natural stop"
    );
    assert_eq!(
        typed
            .usage
            .as_ref()
            .and_then(|usage| usage.tokens.as_ref())
            .and_then(|tokens| tokens.input_tokens)
            .map(|tokens| tokens as u64),
        second.usage.input_tokens,
        "and reports the same input tokens"
    );

    let bodies = crate::cassettes::recorded_interaction_bodies(PROVIDER, SCENARIO);
    assert_eq!(
        bodies.len(),
        2,
        "{SCENARIO}: the cell records the request and then its twin"
    );
    let (first_request, _) = &bodies[0];
    let (second_request, _) = &bodies[1];
    assert_eq!(
        first_request, second_request,
        "{SCENARIO}: `encode` is deterministic, so both turns must send the same request bytes"
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
