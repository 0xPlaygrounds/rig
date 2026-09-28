//! Execution and format contracts shared by the raw-capture matrices.
//!
//! The matrices under `tests/providers/*/raw_capture_matrix.rs`,
//! `raw_stream_capture_matrix.rs` and `raw_completion_parity_matrix.rs` all
//! do the same three things: run one recorded turn, park what it produced,
//! and compare the normalized view against the recorded bytes in the
//! vocabulary of the provider's wire format. The first is the same operation
//! for every provider and lives here; the third is the same operation for
//! every provider *of one format* and lives in [`chat`] and [`responses`].
//! Everything else — the model, the request, the fixture premises, and what a
//! particular provider puts in `raw` that nothing normalizes — stays in the
//! cell, where a reader can see it.
//!
//! Two things deliberately do **not** move here.
//!
//! The cassette wrapper call stays at each `#[tokio::test]` site with its
//! scenario as a string literal: `crates/rig-cassette/tests/common/cassette_safety.rs` discovers
//! fixtures by parsing that literal out of the wrapper call's first argument,
//! so a hoisted or variable scenario would orphan the cassette. The execution
//! helpers below are therefore written to be the *body* passed to that
//! wrapper, never its caller.
//!
//! Expected values stay independent of the code that produced the actual
//! result: every assertion here compares a normalized field against recorded
//! bytes or against the provider-native record the cell hands it. Nothing
//! here re-derives an expectation by running the decoder under test.

pub mod chat;
pub mod responses;

use crate::support::{Observed, collect_required_terminal, collect_text_and_terminal};
use rig_core::completion::{CompletionRequest, CompletionResponse};
use rig_core::error::ProviderError;

/// Run one recorded blocking turn and park the response it produced.
///
/// `request` is built at the call site, so the prompt and every parameter
/// stay visible there.
pub async fn capture_completion(
    model: impl Into<rig_core::DynModel<rig_core::operation::Completion>>,
    request: CompletionRequest,
    sink: Observed<CompletionResponse>,
) -> Result<(), ProviderError> {
    let model: rig_core::DynModel<rig_core::operation::Completion> = model.into();
    sink.put(model.call(request).await?);
    Ok(())
}

/// Run the same built request twice against one model and park both
/// responses, in order.
///
/// The parity matrices need two independent replies to separate "the same
/// request bytes went out twice" from "one reply agreed with itself"; the
/// harness replays a scenario's interactions in order, so the pair is
/// compared with interaction 0 and interaction 1 respectively.
pub async fn capture_completion_pair(
    model: impl Into<rig_core::DynModel<rig_core::operation::Completion>>,
    request: CompletionRequest,
    sink: Observed<(CompletionResponse, CompletionResponse)>,
) -> Result<(), ProviderError> {
    let model: rig_core::DynModel<rig_core::operation::Completion> = model.into();
    let first = model.call(request.clone()).await?;
    let second = model.call(request).await?;
    sink.put((first, second));
    Ok(())
}

/// Stream one recorded turn and park the visible text beside the response
/// the stream must have finished with.
pub async fn capture_text_and_terminal(
    model: impl Into<rig_core::DynModel<rig_core::operation::Completion>>,
    request: CompletionRequest,
    sink: Observed<(String, CompletionResponse)>,
) -> Result<(), ProviderError> {
    let model: rig_core::DynModel<rig_core::operation::Completion> = model.into();
    let (text, terminal) = collect_text_and_terminal(model.stream(request)?).await;
    sink.put((
        text,
        terminal.expect("stream should finish with a response"),
    ));
    Ok(())
}

/// Stream one recorded turn and park the response it must have finished
/// with, discarding the visible text.
pub async fn capture_terminal(
    model: impl Into<rig_core::DynModel<rig_core::operation::Completion>>,
    request: CompletionRequest,
    sink: Observed<CompletionResponse>,
) -> Result<(), ProviderError> {
    let model: rig_core::DynModel<rig_core::operation::Completion> = model.into();
    sink.put(collect_required_terminal(model.stream(request)?).await);
    Ok(())
}

/// Assert the transport request id the driver stamped is the one the
/// recording's response header carried.
///
/// For the dialects that *contract* an id header: the recorded header is the
/// premise, so a recording that stopped carrying it fails here instead of
/// making the claim vacuous. `header` names it for the failure message.
pub fn assert_contracted_request_id(observed: Option<&str>, recorded: Option<&str>, header: &str) {
    assert!(
        recorded.is_some(),
        "the recorded response must carry the {header} header this dialect contracts"
    );
    crate::support::assert_matches_recorded_token(observed, recorded, "request id");
}

/// Assert no transport request id was reported, the documented outcome for a
/// dialect that sends no id header.
pub fn assert_no_request_id(observed: Option<&str>, dialect: &str) {
    assert_eq!(
        observed, None,
        "{dialect} contracts no request-id header, so `None` is the outcome"
    );
}

/// Assert the normalized surface has no slot for any of the named fields.
///
/// The values are reachable only through the capture, which is the claim the
/// "provider-only field" cells make; `normalized` comes from
/// [`crate::support::normalized_without_raw`] so the capture cannot satisfy
/// the check it is the counterexample to.
pub fn assert_normalized_lacks(normalized: &serde_json::Value, fields: &[&str]) {
    for field in fields {
        assert!(
            normalized.get(field).is_none(),
            "the normalized view has no `{field}` slot: {normalized}"
        );
    }
}
