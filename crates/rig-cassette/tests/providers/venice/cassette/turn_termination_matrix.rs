//! Live-recorded matrix for the model-turn termination metadata a hook sees
//! (rig#2184 / PR #2341), against the real Venice wire — the OpenAI
//! Chat Completions dialect; the cells are the OpenAI matrix's, on this
//! wire's model.
//!
//! `ModelTurnFinished` carries `finish_reason: Option<&FinishReason>` and
//! `max_tokens: Option<u64>` — the normalized reason the provider stopped, and
//! the effective output-token cap *that attempt* ran under, after agent
//! configuration, the runner override, and any completion-call `RequestPatch`.
//!
//! **Why cassettes and not mocks.** The unit cells in `rig-agent` drive a
//! `MockCompletionModel`, so they prove the agent plumbs whatever the model
//! layer hands it — they cannot prove that OpenAI Chat Completions's wire
//! `length` becomes `FinishReason::Length` by the time a hook sees it.
//! These cells replay real recorded bytes through the provider mapper, the
//! agent, and the hook stack, so they pin the whole chain the word
//! "normalized" is a claim about. The probe hook and the escalation hook are
//! the *same* types every provider suite uses (`crate::support`) — if a provider
//! needed its own, the metadata would not be provider-neutral.
//!
//! This provider's vocabulary, and what it normalizes to:
//!
//! | normalized | OpenAI Chat Completions wire value | how |
//! |---|---|---|
//! | `Stop` | `stop` | direct |
//! | `Length` | `length` | direct |
//! | `ToolCalls` | `tool_calls` | direct — OpenAI reports a distinct value for tool turns |
//!
//! | # | cell | surface | asserts |
//! |---|------|---------|---------|
//! | 1 | `blocking_truncated_turn_reports_length_and_cap` | blocking | `Length` + the cap this attempt ran under |
//! | 2 | `streaming_truncated_turn_reports_length_and_cap` | streaming | the same, on the other surface |
//! | 3 | `blocking_completed_turn_reports_stop_and_cap` | blocking | `Stop`, and that it fails `truncated_output()` |
//! | 4 | `streaming_completed_turn_reports_stop_and_cap` | streaming | the same |
//! | 5 | `blocking_tool_turn_reports_tool_calls` | blocking | `ToolCalls` |
//! | 6 | `streaming_tool_turn_reports_tool_calls` | streaming | `ToolCalls` |
//! | 7 | `blocking_escalating_retry_reports_each_attempts_own_cap` | blocking | two attempts, two caps, two reasons |
//! | 8 | `streaming_escalating_retry_reports_each_attempts_own_cap` | streaming | the same |
//!
//! Cells 7 and 8 are #2184's acceptance criterion against a live provider: the
//! first attempt truncates under a deliberately tiny cap, a provider-neutral
//! hook reads `FinishReason::Length` off the event and asks for a repeat with
//! a larger cap, and the second attempt reports *its own* cap rather than the
//! agent's baseline. Both attempts live in one cassette, so the escalation is
//! replayed rather than re-derived.
//!
//! Every cell re-reads its own fixture and fails if the recorded turn stopped
//! carrying the wire reason the cell is about — otherwise a provider changing
//! behavior would leave the cell green while covering nothing.
//!
//! **Deliberately not covered here.** `ContentFilter` has no benign trigger:
//! eliciting it means asking a provider to produce content it must refuse.
//! `Other(_)` has no benign reachable wire value on this endpoint. A provider
//! reporting *no* reason is not reachable here either — OpenAI Chat Completions always
//! reports one. All three are pinned by unit cells beside the fix
//! (`crates/rig-agent/src/agent/runner.rs`, `model_turn_finished_*`) and in
//! `crates/rig-core/src/completion/request.rs` (`truncated_output_*`), where
//! the whole vocabulary can be enumerated without a live call.

use rig::completion::FinishReason;
use rig::prelude::*;
use serde::Deserialize;
use serde_json::Value;

use super::super::support::with_venice_cassette;
use crate::cassettes;
use crate::support::{
    Adder, EscalateCapOnTruncation, TurnTerminationProbe, collect_stream_final_response,
};

pub(super) const MODEL: &str = rig::providers::venice::MISTRAL_SMALL_3_2_24B;
pub(super) const TINY_CAP: u64 = 16;
/// Roomy enough for every prompt below to finish naturally.
pub(super) const ROOMY_CAP: u64 = 512;
/// Truncates at `TINY_CAP` and completes at `ROOMY_CAP`.
pub(super) const TRUNCATING_PROMPT: &str = "Write two sentences about maple trees.";
pub(super) const RETRY_PROMPT: &str = "Write two sentences about maple trees.";
pub(super) const SHORT_PROMPT: &str = "Reply with exactly the word: cedar.";
pub(super) const TOOL_PROMPT: &str = "Calculate 2 + 3.";
pub(super) const CONCISE_PREAMBLE: &str =
    "You are a concise assistant. Answer directly in plain text.";
pub(super) const TOOL_PREAMBLE: &str = "Use the provided tool to answer arithmetic questions.";

// ---------------------------------------------------------------------------
// Length — the provider cut the turn short at the cap we set.
// ---------------------------------------------------------------------------

crate::matrix::case_matrix! {
    wrapper: with_venice_cassette, family: turn_termination_matrix_case;
    # [tokio :: test]
    blocking_truncated_turn_reports_length_and_cap: ("turn_termination_matrix/blocking_truncated_turn_reports_length_and_cap", blocking_truncated_turn_reports_length_and_cap_15);
    # [tokio :: test]
    streaming_truncated_turn_reports_length_and_cap: ("turn_termination_matrix/streaming_truncated_turn_reports_length_and_cap", streaming_truncated_turn_reports_length_and_cap_16);
    # [tokio :: test]
    blocking_completed_turn_reports_stop_and_cap: ("turn_termination_matrix/blocking_completed_turn_reports_stop_and_cap", blocking_completed_turn_reports_stop_and_cap_17);
    # [tokio :: test]
    streaming_completed_turn_reports_stop_and_cap: ("turn_termination_matrix/streaming_completed_turn_reports_stop_and_cap", streaming_completed_turn_reports_stop_and_cap_18);
    # [ignore = "Venice mistral-small-3-2-24b-instruct answered without calling add in attempts 1, 2 and 3 (2026-09-13, record-venice-termination-blocking-attempt-{1,2,3}.log); exhausted the reasoning-matrix prompt's three-attempt limit"]
    # [tokio :: test]
    blocking_tool_turn_reports_tool_calls: ("turn_termination_matrix/blocking_tool_turn_reports_tool_calls", blocking_tool_turn_reports_tool_calls_19);
    # [ignore = "Venice mistral-small-3-2-24b-instruct answered without calling add in attempts 1, 2 and 3 (2026-09-13, record-venice-termination-streaming-attempt-{1,2,3}.log); exhausted the reasoning-matrix prompt's three-attempt limit"]
    # [tokio :: test]
    streaming_tool_turn_reports_tool_calls: ("turn_termination_matrix/streaming_tool_turn_reports_tool_calls", streaming_tool_turn_reports_tool_calls_20);
    # [tokio :: test]
    blocking_escalating_retry_reports_each_attempts_own_cap: ("turn_termination_matrix/blocking_escalating_retry_reports_each_attempts_own_cap", blocking_escalating_retry_reports_each_attempts_own_cap_21);
    # [tokio :: test]
    streaming_escalating_retry_reports_each_attempts_own_cap: ("turn_termination_matrix/streaming_escalating_retry_reports_each_attempts_own_cap", streaming_escalating_retry_reports_each_attempts_own_cap_22);
}

// ---------------------------------------------------------------------------
// Stop — the control. A completed turn must not read as truncated.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// ToolCalls — the reason a portable hook must never mistake for retryable.
// OpenAI reports a distinct `tool_calls` wire value, so this maps directly
// rather than through `reconcile_with_output`.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// The acceptance criterion: escalate the cap on truncation, against the real
// provider, and report each attempt's own cap.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Fixture-premise checks: the recorded bytes must still say what the cell
// claims, or the cell passes while covering nothing.
// ---------------------------------------------------------------------------

fn recorded_interactions(scenario: &str) -> Vec<serde_yaml::Value> {
    {
        let path = cassettes::cassette_path("venice", scenario);
        let contents = std::fs::read_to_string(&path).unwrap_or_else(|error| {
            {
                panic!(
                    "provider cassette {} should be readable after recording: {error}",
                    path.display()
                )
            }
        });
        serde_yaml::Deserializer::from_str(&contents)
            .map(|document| serde_yaml::Value::deserialize(document).expect("cassette interaction"))
            .collect()
    }
}

fn interaction_bodies(scenario: &str, side: &str) -> Vec<String> {
    {
        recorded_interactions(scenario)
            .iter()
            .filter_map(|interaction| {
                {
                    interaction
                        .get(side)
                        .and_then(|side| side.get("body"))
                        .and_then(serde_yaml::Value::as_str)
                        .map(ToOwned::to_owned)
                }
            })
            .collect()
    }
}

/// Every JSON object in a recorded body. A blocking response is one object; a
/// streamed one is a sequence of `data:` frames, so both are handled by
/// scanning line by line and keeping whatever parses.
fn body_json_objects(body: &str) -> Vec<Value> {
    {
        body.lines()
            .map(|line| line.strip_prefix("data: ").unwrap_or(line).trim())
            .filter(|line| !line.is_empty() && *line != "[DONE]")
            .filter_map(|line| serde_json::from_str::<Value>(line).ok())
            .collect()
    }
}

/// One wire finish reason per recorded interaction, in order — the first
/// non-null one the body carries. A stream repeats `null` on every chunk until
/// the terminal one, so taking the first non-null entry yields exactly one
/// reason per call on both surfaces.
pub(super) fn recorded_wire_reasons(scenario: &str) -> Vec<String> {
    {
        interaction_bodies(scenario, "then")
            .iter()
            .filter_map(|body| {
                {
                    body_json_objects(body).iter().find_map(|json| {
                        json.get("choices")?.as_array()?.iter().find_map(|choice| {
                            choice
                                .get("finish_reason")
                                .and_then(Value::as_str)
                                .map(ToOwned::to_owned)
                        })
                    })
                }
            })
            .collect()
    }
}

pub(super) fn assert_recorded_wire_reason(scenario: &str, expected: &str) {
    {
        let reasons = recorded_wire_reasons(scenario);
        assert!(
            reasons.contains(&expected.to_owned()),
            "cassette {scenario} no longer records a `{expected}` finish reason (recorded: \
         {reasons:?}); this cell would pass while covering nothing"
        );
    }
}

/// The output-token cap of every recorded *request*, in order — proof that the
/// cap the hook reported is the cap that actually went on the wire.
pub(super) fn recorded_request_caps(scenario: &str) -> Vec<u64> {
    {
        interaction_bodies(scenario, "when")
            .iter()
            .filter_map(|body| serde_json::from_str::<Value>(body).ok())
            .filter_map(|body| {
                body.get("max_tokens")
                    .or_else(|| body.get("max_completion_tokens"))
                    .and_then(Value::as_u64)
            })
            .collect()
    }
}

pub(super) fn assert_recorded_request_cap(scenario: &str, expected: u64) {
    {
        let caps = recorded_request_caps(scenario);
        assert!(
            caps.contains(&expected),
            "cassette {scenario} no longer records a request capped at {expected} (recorded: {caps:?})"
        );
    }
}
