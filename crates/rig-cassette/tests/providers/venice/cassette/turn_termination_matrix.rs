//! Live-recorded matrix for the model-turn termination metadata a hook sees,
//! against the real Venice wire. Venice speaks the OpenAI Chat Completions
//! dialect, so the cells are the OpenAI matrix's, on this wire's model.
//!
//! `ModelTurnFinished` carries `finish_reason: Option<&FinishReason>` and
//! `max_tokens: Option<u64>`: the normalized reason the provider stopped, and
//! the effective output-token cap that attempt ran under, after agent
//! configuration, the runner override, and any completion-call `RequestPatch`.
//! These cells replay real recorded bytes through the provider mapper, the
//! agent, and the hook stack, so they pin that the wire's `stop`, `length` and
//! `tool_calls` become `Stop`, `Length` and `ToolCalls` by the time a hook sees
//! them. The probe and escalation hooks are the shared `crate::support` types,
//! so the metadata stays provider-neutral.
//!
//! Every cell re-reads its own fixture and fails if the recorded turn stopped
//! carrying the wire reason the cell is about. `ContentFilter`, `Other(_)` and
//! a missing reason have no benign trigger on this endpoint and are pinned by
//! unit tests in `crates/rig-agent/src/agent/runner.rs` and
//! `crates/rig-core/src/completion/request.rs`.

use rig::completion::FinishReason;
use serde_json::Value;

use super::super::support::with_venice_cassette;
use crate::cassettes;
use crate::support::{Adder, TurnTerminationProbe, collect_stream_final_response};

pub(super) const MODEL: &str = rig::providers::venice::MISTRAL_SMALL_3_2_24B;
pub(super) const TINY_CAP: u64 = 16;
/// Roomy enough for every prompt below to finish naturally.
pub(super) const ROOMY_CAP: u64 = 512;
/// Truncates at `TINY_CAP` and completes at `ROOMY_CAP`.
pub(super) const TRUNCATING_PROMPT: &str = "Write two sentences about maple trees.";
pub(super) const TOOL_PROMPT: &str = "Calculate 2 + 3.";
pub(super) const CONCISE_PREAMBLE: &str =
    "You are a concise assistant. Answer directly in plain text.";
pub(super) const TOOL_PREAMBLE: &str = "Use the provided tool to answer arithmetic questions.";

// ---------------------------------------------------------------------------
// Length, and ToolCalls: the reason a portable hook must never mistake for
// retryable.
// ---------------------------------------------------------------------------

crate::matrix::case_matrix! {
    wrapper: with_venice_cassette, family: turn_termination_matrix_case;
    # [tokio :: test]
    blocking_truncated_turn_reports_length_and_cap: ("turn_termination_matrix/blocking_truncated_turn_reports_length_and_cap", blocking_truncated_turn_reports_length_and_cap_15);
    # [tokio :: test]
    streaming_truncated_turn_reports_length_and_cap: ("turn_termination_matrix/streaming_truncated_turn_reports_length_and_cap", streaming_truncated_turn_reports_length_and_cap_16);
    # [ignore = "Venice mistral-small-3-2-24b-instruct answered without calling add in attempts 1, 2 and 3 (2026-09-13, record-venice-termination-blocking-attempt-{1,2,3}.log); exhausted the reasoning-matrix prompt's three-attempt limit"]
    # [tokio :: test]
    blocking_tool_turn_reports_tool_calls: ("turn_termination_matrix/blocking_tool_turn_reports_tool_calls", blocking_tool_turn_reports_tool_calls_19);
    # [ignore = "Venice mistral-small-3-2-24b-instruct answered without calling add in attempts 1, 2 and 3 (2026-09-13, record-venice-termination-streaming-attempt-{1,2,3}.log); exhausted the reasoning-matrix prompt's three-attempt limit"]
    # [tokio :: test]
    streaming_tool_turn_reports_tool_calls: ("turn_termination_matrix/streaming_tool_turn_reports_tool_calls", streaming_tool_turn_reports_tool_calls_20);
}

// ---------------------------------------------------------------------------
// Fixture-premise checks: the recorded bytes must still say what the cell
// claims, or the cell passes while covering nothing.
// ---------------------------------------------------------------------------

fn interaction_bodies(scenario: &str, side: &str) -> Vec<String> {
    cassettes::recorded_interaction_bodies("venice", scenario)
        .into_iter()
        .map(|(request, response)| if side == "when" { request } else { response })
        .filter(|body| !body.is_empty())
        .collect()
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
