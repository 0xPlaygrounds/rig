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

use rig_core::completion::{CompletionModel, CompletionRequest, CompletionResponse};
use rig_core::error::ProviderError;
use rig_core::streaming::StreamFinal;

use crate::support::{
    Observed, collect_required_terminal, collect_sole_terminal, collect_text_and_sole_terminal,
    collect_text_and_terminal,
};

/// Run one recorded blocking turn and park the response it produced.
///
/// `build` is the cell's own request builder, so the prompt, the model and
/// every parameter stay visible at the call site.
pub async fn capture_completion<M>(
    model: M,
    build: impl FnOnce(&M) -> CompletionRequest,
    sink: Observed<CompletionResponse>,
) -> Result<(), ProviderError>
where
    M: CompletionModel,
{
    let request = build(&model);
    sink.put(model.completion(request).await?);
    Ok(())
}

/// Run the same built request twice against one model and park both
/// responses, in order.
///
/// The parity matrices need two independent replies to separate "the same
/// request bytes went out twice" from "one reply agreed with itself"; the
/// harness replays a scenario's interactions in order, so the pair is
/// compared with interaction 0 and interaction 1 respectively.
pub async fn capture_completion_pair<M>(
    model: M,
    build: impl Fn(&M) -> CompletionRequest,
    sink: Observed<(CompletionResponse, CompletionResponse)>,
) -> Result<(), ProviderError>
where
    M: CompletionModel,
{
    let first = model.completion(build(&model)).await?;
    let second = model.completion(build(&model)).await?;
    sink.put((first, second));
    Ok(())
}

/// Stream one recorded turn and park the visible text beside the terminal
/// record the stream must have ended with.
pub async fn capture_text_and_terminal<M>(
    model: M,
    build: impl FnOnce(&M) -> CompletionRequest,
    sink: Observed<(String, StreamFinal)>,
) -> Result<(), ProviderError>
where
    M: CompletionModel,
{
    let request = build(&model);
    let (text, terminal) = collect_text_and_terminal(model.stream(request).await?).await;
    sink.put((
        text,
        terminal.expect("stream should end with a terminal record"),
    ));
    Ok(())
}

/// Stream one recorded turn and park the terminal record it must have ended
/// with, discarding the visible text.
///
/// Keeps the last terminal record, so a dialect that repeats its accounting
/// across closing frames is fine here; a dialect whose contract is *one*
/// terminal record uses [`capture_sole_terminal`] instead.
pub async fn capture_terminal<M>(
    model: M,
    build: impl FnOnce(&M) -> CompletionRequest,
    sink: Observed<StreamFinal>,
) -> Result<(), ProviderError>
where
    M: CompletionModel,
{
    let request = build(&model);
    sink.put(collect_required_terminal(model.stream(request).await?).await);
    Ok(())
}

/// Stream one recorded turn and park its one terminal record, failing when
/// the stream emitted none or more than one.
pub async fn capture_sole_terminal<M>(
    model: M,
    build: impl FnOnce(&M) -> CompletionRequest,
    sink: Observed<StreamFinal>,
) -> Result<(), ProviderError>
where
    M: CompletionModel,
{
    let request = build(&model);
    sink.put(collect_sole_terminal(model.stream(request).await?).await);
    Ok(())
}

/// Stream one recorded turn and park the visible text beside its one
/// terminal record, failing when the stream emitted none or more than one.
///
/// For the dialects whose contract is a single terminal record *and* whose
/// cells are about what the stream said — [`capture_text_and_terminal`]
/// keeps the last of several records, [`capture_sole_terminal`] drops the
/// text.
pub async fn capture_text_and_sole_terminal<M>(
    model: M,
    build: impl FnOnce(&M) -> CompletionRequest,
    sink: Observed<(String, StreamFinal)>,
) -> Result<(), ProviderError>
where
    M: CompletionModel,
{
    let request = build(&model);
    sink.put(collect_text_and_sole_terminal(model.stream(request).await?).await);
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

/// A streamed terminal record serialized with its raw capture cleared, so an
/// assertion about the normalized surface cannot be satisfied by something
/// `raw` happens to carry.
///
/// The blocking counterpart is [`crate::support::normalized_without_raw`].
pub fn stream_normalized_without_raw(terminal: &StreamFinal) -> serde_json::Value {
    let mut terminal = terminal.clone();
    terminal.raw = serde_json::Value::Null;
    serde_json::to_value(&terminal).expect("terminal record should serialize")
}

/// Assert the normalized surface has no slot for any of the named fields.
///
/// The values are reachable only through the capture, which is the claim the
/// "provider-only field" cells make; `normalized` comes from
/// [`stream_normalized_without_raw`] or
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
