//! Live-recorded matrix for the model-turn termination metadata a hook sees
//! (rig#2184 / PR #2341), against real Gemini.
//!
//! `ModelTurnFinished` carries `finish_reason: Option<&FinishReason>` and
//! `max_tokens: Option<u64>` — the normalized reason the provider stopped, and
//! the effective output-token cap *that attempt* ran under, after agent
//! configuration, the runner override, and any completion-call `RequestPatch`.
//!
//! **Why cassettes and not mocks.** The unit cells in `rig-agent` drive a
//! `MockCompletionModel`, so they prove the agent plumbs whatever the model
//! layer hands it — they cannot prove that Gemini's wire
//! `MAX_TOKENS` becomes `FinishReason::Length` by the time a hook sees it.
//! These cells replay real recorded bytes through the provider mapper, the
//! agent, and the hook stack, so they pin the whole chain the word
//! "normalized" is a claim about. The probe hook and the escalation hook are
//! the *same* types every provider suite uses (`crate::support`) — if a provider
//! needed its own, the metadata would not be provider-neutral.
//!
//! This provider's vocabulary, and what it normalizes to:
//!
//! | normalized | Gemini wire value | how |
//! |---|---|---|
//! | `Stop` | `STOP` | direct |
//! | `Length` | `MAX_TOKENS` | direct |
//! | `ToolCalls` | `STOP` | **reconciled** — Gemini has no distinct tool-turn value, so rig upgrades a bare `STOP` via `FinishReason::reconcile_with_output` |
//!
//! | # | cell | surface | asserts |
//! |---|------|---------|---------|
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
//! reporting *no* reason is not reachable here either — Gemini always
//! reports one. All three are pinned by unit cells beside the fix
//! (`crates/rig-agent/src/agent/runner.rs`, `model_turn_finished_*`) and in
//! `crates/rig-core/src/completion/request.rs` (`truncated_output_*`), where
//! the whole vocabulary can be enumerated without a live call.

use serde::Deserialize;
use serde_json::Value;

use crate::cassettes;

/// Gemini counts hidden thinking tokens against `maxOutputTokens`, so every
/// cell here pins `thinkingBudget: 0`. Without it the whole cap can go to
/// thinking, the candidate comes back content-less, and the unary mapper
/// errors before any hook runs.
pub(super) const TINY_CAP: u64 = 24;
/// Truncates at `TINY_CAP` and completes at `ROOMY_CAP`.
pub(super) const TRUNCATING_PROMPT: &str = "Write a 200-word story about a lighthouse keeper.";
pub(super) const CONCISE_PREAMBLE: &str =
    "You are a concise assistant. Answer directly in plain text.";

/// Thinking off, so the output-token cap governs visible text alone.
pub(super) fn no_thinking() -> serde_json::Value {
    serde_json::json!({ "generationConfig": { "thinkingConfig": { "thinkingBudget": 0 } } })
}

// ---------------------------------------------------------------------------
// Length — the provider cut the turn short at the cap we set.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Stop — the control. A completed turn must not read as truncated.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// ToolCalls — the reason a portable hook must never mistake for retryable.
// Gemini is the interesting one: it has NO distinct tool-turn wire value and
// reports a bare `STOP`. `FinishReason::reconcile_with_output` upgrades it to
// `ToolCalls` because the turn carried a call. This cell is the live proof of
// that reconciliation — without it, a portable retry hook would have to
// special-case Gemini.
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
        let path = cassettes::cassette_path("gemini", scenario);
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
                        json.get("candidates")?
                            .as_array()?
                            .iter()
                            .find_map(|candidate| {
                                candidate
                                    .get("finishReason")
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
                body.get("generationConfig")
                    .and_then(|config| config.get("maxOutputTokens"))
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
