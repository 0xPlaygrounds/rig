//! Determinism and faithfulness contract for Gemini's unary seam, on both of
//! its surfaces.
//!
//! # The contract
//!
//! Each Gemini surface has exactly one unary seam: the wire encodes the
//! request, the driver carries the bytes, and one decoder folds the reply into
//! a [`rig::completion::CompletionResponse`] whose `raw` holds the provider's
//! own reply document verbatim. Two things follow, and each cell pins both:
//!
//! 1. **`encode` is deterministic.** The same built request produces the same
//!    request bytes every time, so a caller can replay it.
//! 2. **`raw` is a faithful second view, not a summary.** Read back into the
//!    surface's own response type, it reproduces the very response it rode on
//!    — identity, finish reason, model, usage and text.
//!
//! Neither Gemini route reports a transport request-id response header
//! (verified against the live API), so
//! [`rig::completion::CompletionResponse::provider_request_id`] is `None` on
//! both surfaces by design — its doc names Gemini as the documented `None`
//! case. Two live requests get two response ids, so identity equality is
//! asserted on the captured-raw side (same response) and shape-wise across the
//! two requests.
//!
//! # Matrix
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `rest_raw_try_into_matches_completion` | `generateContent`: two turns through the one seam | the two request bodies are byte-identical; the captured `raw` reproduces the response it rode on | recorded |
//! | 2 | `interactions_raw_try_into_matches_completion` | Interactions API: the same, over its own document shape | the two request bodies are byte-identical; the captured `raw` reproduces the response it rode on | recorded |
//!
//! Both cells are recorded (`GEMINI_API_KEY` was available). Each records one
//! scenario with **two** interactions, because the cell needs two independent
//! replies to separate "the same bytes went out twice" from "one reply agreed
//! with itself"; the harness replays interactions in order and fails on an
//! interaction nothing consumed, so both turns are still issued.

use rig::completion::{CompletionModel, CompletionResponse as RigCompletionResponse, FinishReason};
use rig::providers::gemini::completion::gemini_api_types::{
    ContentCandidate, GenerateContentResponse, PartKind,
};
use rig::providers::gemini::interactions_api::{Interaction, InteractionStatus};
use serde::Deserialize;
use serde_json::Value;

use super::super::support::{with_gemini_cassette, with_gemini_interactions_cassette};
use crate::raw_capture::{assert_no_request_id, capture_completion_pair};
use crate::support::{Observed, assistant_text};

const PROVIDER: &str = "gemini";
const REST_MODEL: &str = "gemini-2.5-flash-lite";
const INTERACTIONS_MODEL: &str = "gemini-3-flash-preview";
const PROMPT: &str = "Reply with exactly this one word and nothing else: parity";

/// The one request both cells send, twice: the same built request through one
/// seam is what makes "the same bytes went out twice" a claim about `encode`
/// rather than about the cell.
fn request(model: &(impl CompletionModel + Clone)) -> rig::completion::CompletionRequest {
    model.completion_request(PROMPT).temperature(0.0).build()
}

/// The parity a caller can rely on across two turns of identical bytes:
/// everything the contract names, except that each request gets its own
/// response id.
fn assert_cross_request_parity(first: &RigCompletionResponse, second: &RigCompletionResponse) {
    assert_eq!(first.finish_reason(), second.finish_reason());
    assert_eq!(first.model, second.model);
    assert_eq!(first.provider, second.provider);
    // Identical request bytes tokenize identically; the output side is the
    // model's to vary.
    assert_eq!(first.usage.input_tokens, second.usage.input_tokens);

    let first_identity = first.identity();
    let second_identity = second.identity();
    assert_eq!(first_identity.message_id, second_identity.message_id);
    // Gemini sends no request-id header, so the driver reports None by
    // design — and so does the second reply: the same seam, the same header
    // set.
    assert_no_request_id(first_identity.provider_request_id.as_deref(), "Gemini");
    assert_no_request_id(second_identity.provider_request_id.as_deref(), "Gemini");
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
}

/// The visible (non-`thought`) text of a `generateContent` candidate, folded
/// the way the decoder folds text blocks.
fn visible_text(candidate: &ContentCandidate) -> String {
    candidate
        .content
        .as_ref()
        .into_iter()
        .flat_map(|content| content.parts.iter())
        .filter(|part| !part.thought.unwrap_or(false))
        .filter_map(|part| match &part.part {
            PartKind::Text(text) => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

/// The determinism half, read off the fixture: two recorded turns whose
/// request bodies are byte-identical, each answered by a naturally finished
/// reply.
///
/// The request side can only be proved here — the response says nothing about
/// what went out, and the cassette replays regardless.
fn assert_two_recorded_turns(scenario: &str, status_pointer: &str, status: &str) {
    let bodies = crate::cassettes::recorded_interaction_bodies(PROVIDER, scenario);
    assert_eq!(
        bodies.len(),
        2,
        "{scenario}: the cell records the request and then its twin"
    );
    let (first_request, _) = &bodies[0];
    let (second_request, _) = &bodies[1];
    assert_eq!(
        first_request, second_request,
        "{scenario}: `encode` is deterministic, so both turns must send the same request bytes"
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
    let observed: Observed<(RigCompletionResponse, RigCompletionResponse)> = Observed::default();
    let sink = observed.clone();
    with_gemini_cassette(
        "raw_completion_parity_matrix/rest_raw_try_into_matches_completion",
        |client| async move {
            capture_completion_pair(client.completion(REST_MODEL), request, sink)
                .await
                .expect("both turns of the same request should succeed");
        },
    )
    .await;

    let (first, second) = observed.take();
    assert_cross_request_parity(&first, &second);

    // Same response: the captured raw, read back as Gemini's own
    // `generateContent` document, reproduces the response it rode on.
    let typed = GenerateContentResponse::deserialize(&second.raw)
        .expect("captured raw is Gemini's own generateContent document");
    assert_eq!(typed.model_version.as_deref(), second.model.as_deref());
    assert_eq!(
        Some(typed.response_id.as_str()),
        second.response_id.as_deref()
    );
    assert_eq!(
        typed
            .usage_metadata
            .as_ref()
            .map(|usage| usage.prompt_token_count as u64),
        second.usage.input_tokens
    );
    assert_eq!(
        typed
            .usage_metadata
            .as_ref()
            .map(|usage| usage.total_token_count as u64),
        second.usage.total_tokens
    );
    let candidate = typed
        .candidates
        .first()
        .expect("the recorded turn carries a candidate");
    assert_eq!(
        visible_text(candidate),
        assistant_text(&second.choice),
        "the normalized text is exactly the document's visible text parts"
    );
    assert_eq!(
        second.finish_reason(),
        Some(FinishReason::Stop),
        "the document's STOP reaches the caller as rig's Stop"
    );

    assert_two_recorded_turns(SCENARIO, "/candidates/0/finishReason", "STOP");
}

#[tokio::test]
async fn interactions_raw_try_into_matches_completion() {
    const SCENARIO: &str =
        "raw_completion_parity_matrix/interactions_raw_try_into_matches_completion";
    let observed: Observed<(RigCompletionResponse, RigCompletionResponse)> = Observed::default();
    let sink = observed.clone();
    with_gemini_interactions_cassette(
        "raw_completion_parity_matrix/interactions_raw_try_into_matches_completion",
        |client| async move {
            capture_completion_pair(
                client.map_wire(|config| config.interactions(INTERACTIONS_MODEL)),
                request,
                sink,
            )
            .await
            .expect("both turns of the same request should succeed");
        },
    )
    .await;

    let (first, second) = observed.take();
    assert_cross_request_parity(&first, &second);

    // Same response: the captured raw, read back as the Interactions API's
    // own document, reproduces the response it rode on.
    let typed = Interaction::deserialize(&second.raw)
        .expect("captured raw is the Interactions API's own document");
    assert_eq!(typed.model, second.model);
    assert_eq!(Some(typed.id.as_str()), second.response_id.as_deref());
    assert_eq!(
        typed
            .usage
            .as_ref()
            .and_then(|usage| usage.total_input_tokens),
        second.usage.input_tokens
    );
    assert_eq!(
        typed.usage.as_ref().and_then(|usage| usage.total_tokens),
        second.usage.total_tokens
    );
    assert!(
        matches!(typed.status, Some(InteractionStatus::Completed)),
        "the document keeps the API's own lifecycle spelling, got {:?}",
        typed.status
    );
    assert_eq!(
        second.finish_reason(),
        Some(FinishReason::Stop),
        "and `completed` reaches the caller as rig's Stop"
    );

    assert_two_recorded_turns(SCENARIO, "/status", "completed");
}
