//! Live-recorded matrix for the model-turn termination metadata a hook sees
//! (rig#2184 / PR #2341), against real Anthropic Messages.
//!
//! `ModelTurnFinished` carries `finish_reason: Option<&FinishReason>` and
//! `max_tokens: Option<u64>` — the normalized reason the provider stopped, and
//! the effective output-token cap *that attempt* ran under, after agent
//! configuration, the runner override, and any completion-call `RequestPatch`.
//!
//! **Why cassettes and not mocks.** The unit cells in `rig-agent` drive a
//! `MockCompletionModel`, so they prove the agent plumbs whatever the model
//! layer hands it — they cannot prove that Anthropic Messages's wire
//! `max_tokens` becomes `FinishReason::Length` by the time a hook sees it.
//! These cells replay real recorded bytes through the provider mapper, the
//! agent, and the hook stack, so they pin the whole chain the word
//! "normalized" is a claim about. The probe hook and the escalation hook are
//! the *same* types every provider suite uses (`crate::support`) — if a provider
//! needed its own, the metadata would not be provider-neutral.
//!
//! This provider's vocabulary, and what it normalizes to:
//!
//! | normalized | Anthropic Messages wire value | how |
//! |---|---|---|
//! | `Stop` | `end_turn` | direct |
//! | `Length` | `max_tokens` | direct |
//! | `ToolCalls` | `tool_use` | direct — Anthropic reports a distinct value for tool turns |
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
//! reporting *no* reason is not reachable here either — Anthropic Messages always
//! reports one. All three are pinned by unit cells beside the fix
//! (`crates/rig-agent/src/agent/runner.rs`, `model_turn_finished_*`) and in
//! `crates/rig-core/src/completion/request.rs` (`truncated_output_*`), where
//! the whole vocabulary can be enumerated without a live call.

use rig::completion::FinishReason;
use rig::prelude::*;
use rig::providers::anthropic;
use rig::streaming::StreamingPrompt;
use serde::Deserialize;
use serde_json::Value;

use super::super::support::with_anthropic_turn_metadata_cassette;
use crate::cassettes;
use crate::support::{
    Adder, EscalateCapOnTruncation, TurnTerminationProbe, collect_stream_final_response,
};

/// Anthropic's floor is 1; 3 is the proven cap that truncates mid-word while
/// still returning text, so the turn succeeds and a hook can observe it.
const TINY_CAP: u64 = 3;
/// Roomy enough for every prompt below to finish naturally.
const ROOMY_CAP: u64 = 512;
/// Truncates at `TINY_CAP` and completes at `ROOMY_CAP`.
const TRUNCATING_PROMPT: &str =
    "Repeat exactly these four words, one per line, and nothing else: alpha bravo charlie delta";
const RETRY_PROMPT: &str =
    "Repeat exactly these four words, one per line, and nothing else: alpha bravo charlie delta";
const SHORT_PROMPT: &str = "Reply with exactly the word: cedar.";
const TOOL_PROMPT: &str = "Calculate 2 + 3.";
const CONCISE_PREAMBLE: &str = "You are a concise assistant. Answer directly in plain text.";
const TOOL_PREAMBLE: &str = "Use the provided tool to answer arithmetic questions.";

// ---------------------------------------------------------------------------
// Length — the provider cut the turn short at the cap we set.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn blocking_truncated_turn_reports_length_and_cap() {
    {
        const SCENARIO: &str =
            "turn_termination_matrix/blocking_truncated_turn_reports_length_and_cap";
        let probe = TurnTerminationProbe::default();
        let observed = probe.clone();

        with_anthropic_turn_metadata_cassette(
            "turn_termination_matrix/blocking_truncated_turn_reports_length_and_cap",
            |client| async move {
                {
                    client
                        .agent(anthropic::completion::CLAUDE_HAIKU_4_5)
                        .preamble(CONCISE_PREAMBLE)
                        .temperature(0.0)
                        .max_tokens(TINY_CAP)
                        .add_hook(probe)
                        .build()
                        .runner(TRUNCATING_PROMPT)
                        .run()
                        .await
                        .expect("a partially truncated turn still carries an answer");
                }
            },
        )
        .await;

        assert_eq!(
            observed.first_reason(),
            Some(FinishReason::Length),
            "the wire `max_tokens` must reach the hook as FinishReason::Length"
        );
        assert_eq!(
            observed.first_max_tokens(),
            Some(TINY_CAP),
            "the hook must report the cap this attempt actually ran under"
        );
        assert!(
            observed
                .first_reason()
                .is_some_and(|reason| reason.truncated_output()),
            "a truncated turn must satisfy the portable retry predicate"
        );
        assert_recorded_wire_reason(SCENARIO, "max_tokens");
        assert_recorded_request_cap(SCENARIO, TINY_CAP);
    }
}

#[tokio::test]
async fn streaming_truncated_turn_reports_length_and_cap() {
    {
        const SCENARIO: &str =
            "turn_termination_matrix/streaming_truncated_turn_reports_length_and_cap";
        let probe = TurnTerminationProbe::default();
        let observed = probe.clone();

        with_anthropic_turn_metadata_cassette(
            "turn_termination_matrix/streaming_truncated_turn_reports_length_and_cap",
            |client| async move {
                {
                    let agent = client
                        .agent(anthropic::completion::CLAUDE_HAIKU_4_5)
                        .preamble(CONCISE_PREAMBLE)
                        .temperature(0.0)
                        .max_tokens(TINY_CAP)
                        .build();

                    let mut stream = agent.stream_prompt(TRUNCATING_PROMPT).add_hook(probe).await;
                    let _ = collect_stream_final_response(&mut stream).await;
                }
            },
        )
        .await;

        assert_eq!(
            observed.first_reason(),
            Some(FinishReason::Length),
            "the streaming surface must report the same reason as the blocking one"
        );
        assert_eq!(observed.first_max_tokens(), Some(TINY_CAP));
        assert_recorded_wire_reason(SCENARIO, "max_tokens");
        assert_recorded_request_cap(SCENARIO, TINY_CAP);
    }
}

// ---------------------------------------------------------------------------
// Stop — the control. A completed turn must not read as truncated.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn blocking_completed_turn_reports_stop_and_cap() {
    {
        const SCENARIO: &str =
            "turn_termination_matrix/blocking_completed_turn_reports_stop_and_cap";
        let probe = TurnTerminationProbe::default();
        let observed = probe.clone();

        with_anthropic_turn_metadata_cassette(
            "turn_termination_matrix/blocking_completed_turn_reports_stop_and_cap",
            |client| async move {
                {
                    client
                        .agent(anthropic::completion::CLAUDE_HAIKU_4_5)
                        .preamble(CONCISE_PREAMBLE)
                        .temperature(0.0)
                        .max_tokens(ROOMY_CAP)
                        .add_hook(probe)
                        .build()
                        .runner(SHORT_PROMPT)
                        .run()
                        .await
                        .expect("a short answer under a roomy cap");
                }
            },
        )
        .await;

        assert_eq!(observed.first_reason(), Some(FinishReason::Stop));
        assert_eq!(observed.first_max_tokens(), Some(ROOMY_CAP));
        assert!(
            !observed
                .first_reason()
                .is_some_and(|reason| reason.truncated_output()),
            "a completed turn must not satisfy the retry predicate"
        );
        assert_recorded_wire_reason(SCENARIO, "end_turn");
    }
}

#[tokio::test]
async fn streaming_completed_turn_reports_stop_and_cap() {
    {
        const SCENARIO: &str =
            "turn_termination_matrix/streaming_completed_turn_reports_stop_and_cap";
        let probe = TurnTerminationProbe::default();
        let observed = probe.clone();

        with_anthropic_turn_metadata_cassette(
            "turn_termination_matrix/streaming_completed_turn_reports_stop_and_cap",
            |client| async move {
                {
                    let agent = client
                        .agent(anthropic::completion::CLAUDE_HAIKU_4_5)
                        .preamble(CONCISE_PREAMBLE)
                        .temperature(0.0)
                        .max_tokens(ROOMY_CAP)
                        .build();

                    let mut stream = agent.stream_prompt(SHORT_PROMPT).add_hook(probe).await;
                    let _ = collect_stream_final_response(&mut stream).await;
                }
            },
        )
        .await;

        assert_eq!(observed.first_reason(), Some(FinishReason::Stop));
        assert_eq!(observed.first_max_tokens(), Some(ROOMY_CAP));
        assert_recorded_wire_reason(SCENARIO, "end_turn");
    }
}

// ---------------------------------------------------------------------------
// ToolCalls — the reason a portable hook must never mistake for retryable.
// Anthropic reports a distinct `tool_use` wire value, so this maps directly
// rather than through `reconcile_with_output`.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn blocking_tool_turn_reports_tool_calls() {
    {
        const SCENARIO: &str = "turn_termination_matrix/blocking_tool_turn_reports_tool_calls";
        let probe = TurnTerminationProbe::default();
        let observed = probe.clone();

        with_anthropic_turn_metadata_cassette(
            "turn_termination_matrix/blocking_tool_turn_reports_tool_calls",
            |client| async move {
                {
                    client
                        .agent(anthropic::completion::CLAUDE_HAIKU_4_5)
                        .preamble(TOOL_PREAMBLE)
                        .temperature(0.0)
                        .max_tokens(ROOMY_CAP)
                        .tool(Adder)
                        .add_hook(probe)
                        .build()
                        .runner(TOOL_PROMPT)
                        .max_turns(3)
                        .run()
                        .await
                        .expect("the tool turn should complete the run");
                }
            },
        )
        .await;

        assert_eq!(
            observed.first_reason(),
            Some(FinishReason::ToolCalls),
            "the turn that issued the tool call must read as ToolCalls"
        );
        assert_eq!(observed.first_max_tokens(), Some(ROOMY_CAP));
        assert!(
            !observed
                .first_reason()
                .is_some_and(|reason| reason.truncated_output()),
            "a tool turn must not satisfy the retry predicate"
        );
        assert_recorded_wire_reason(SCENARIO, "tool_use");
    }
}

#[tokio::test]
async fn streaming_tool_turn_reports_tool_calls() {
    {
        const SCENARIO: &str = "turn_termination_matrix/streaming_tool_turn_reports_tool_calls";
        let probe = TurnTerminationProbe::default();
        let observed = probe.clone();

        with_anthropic_turn_metadata_cassette(
            "turn_termination_matrix/streaming_tool_turn_reports_tool_calls",
            |client| async move {
                {
                    let agent = client
                        .agent(anthropic::completion::CLAUDE_HAIKU_4_5)
                        .preamble(TOOL_PREAMBLE)
                        .temperature(0.0)
                        .max_tokens(ROOMY_CAP)
                        .tool(Adder)
                        .build();

                    let mut stream = agent
                        .stream_prompt(TOOL_PROMPT)
                        .add_hook(probe)
                        .max_turns(3)
                        .await;
                    let _ = collect_stream_final_response(&mut stream).await;
                }
            },
        )
        .await;

        assert_eq!(
            observed.first_reason(),
            Some(FinishReason::ToolCalls),
            "streaming must resolve the tool turn exactly as blocking does"
        );
        assert_eq!(observed.first_max_tokens(), Some(ROOMY_CAP));
        assert_recorded_wire_reason(SCENARIO, "tool_use");
    }
}

// ---------------------------------------------------------------------------
// The acceptance criterion: escalate the cap on truncation, against the real
// provider, and report each attempt's own cap.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn blocking_escalating_retry_reports_each_attempts_own_cap() {
    {
        const SCENARIO: &str =
            "turn_termination_matrix/blocking_escalating_retry_reports_each_attempts_own_cap";
        let probe = TurnTerminationProbe::default();
        let escalate = EscalateCapOnTruncation::new(TINY_CAP, ROOMY_CAP);
        let observed = probe.clone();
        let escalations = escalate.clone();

        with_anthropic_turn_metadata_cassette(
            "turn_termination_matrix/blocking_escalating_retry_reports_each_attempts_own_cap",
            |client| async move {
                {
                    client
                        .agent(anthropic::completion::CLAUDE_HAIKU_4_5)
                        .preamble(CONCISE_PREAMBLE)
                        .temperature(0.0)
                        // The agent baseline. Neither attempt should report it: the
                        // hook's patch replaces it on every prepared request.
                        .max_tokens(64)
                        // Observers first: a hook returning a non-continue action
                        // short-circuits every hook registered behind it.
                        .add_hook(probe)
                        .add_hook(escalate)
                        .build()
                        .runner(RETRY_PROMPT)
                        .max_turns(2)
                        .run()
                        .await
                        .expect("the retried attempt should answer");
                }
            },
        )
        .await;

        assert_eq!(
            observed.observations(),
            vec![
                (Some(FinishReason::Length), Some(TINY_CAP)),
                (Some(FinishReason::Stop), Some(ROOMY_CAP)),
            ],
            "each attempt must report its own post-patch cap, never the agent's baseline of 64"
        );
        assert_eq!(escalations.escalations(), vec![ROOMY_CAP]);
        assert_eq!(escalations.retries(), 1);

        // ...and the recorded traffic corroborates it: two calls, the caps the
        // hook chose, and the two reasons in order.
        assert_eq!(recorded_request_caps(SCENARIO), vec![TINY_CAP, ROOMY_CAP]);
        assert_eq!(
            recorded_wire_reasons(SCENARIO),
            vec!["max_tokens".to_owned(), "end_turn".to_owned()]
        );
    }
}

#[tokio::test]
async fn streaming_escalating_retry_reports_each_attempts_own_cap() {
    {
        const SCENARIO: &str =
            "turn_termination_matrix/streaming_escalating_retry_reports_each_attempts_own_cap";
        let probe = TurnTerminationProbe::default();
        let escalate = EscalateCapOnTruncation::new(TINY_CAP, ROOMY_CAP);
        let observed = probe.clone();
        let escalations = escalate.clone();

        with_anthropic_turn_metadata_cassette(
            "turn_termination_matrix/streaming_escalating_retry_reports_each_attempts_own_cap",
            |client| async move {
                {
                    let agent = client
                        .agent(anthropic::completion::CLAUDE_HAIKU_4_5)
                        .preamble(CONCISE_PREAMBLE)
                        .temperature(0.0)
                        // The agent baseline. Neither attempt should report it: the
                        // hook's patch replaces it on every prepared request.
                        .max_tokens(64)
                        .build();

                    let mut stream = agent
                        .stream_prompt(RETRY_PROMPT)
                        .add_hook(probe)
                        .add_hook(escalate)
                        .max_turns(2)
                        .await;
                    let _ = collect_stream_final_response(&mut stream).await;
                }
            },
        )
        .await;

        assert_eq!(
            observed.observations(),
            vec![
                (Some(FinishReason::Length), Some(TINY_CAP)),
                (Some(FinishReason::Stop), Some(ROOMY_CAP)),
            ],
            "the streaming surface must escalate and report identically to blocking"
        );
        assert_eq!(escalations.escalations(), vec![ROOMY_CAP]);
        assert_eq!(recorded_request_caps(SCENARIO), vec![TINY_CAP, ROOMY_CAP]);
    }
}

// ---------------------------------------------------------------------------
// Fixture-premise checks: the recorded bytes must still say what the cell
// claims, or the cell passes while covering nothing.
// ---------------------------------------------------------------------------

fn recorded_interactions(scenario: &str) -> Vec<serde_yaml::Value> {
    {
        let path = cassettes::cassette_path("anthropic", scenario);
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
fn recorded_wire_reasons(scenario: &str) -> Vec<String> {
    {
        interaction_bodies(scenario, "then")
            .iter()
            .filter_map(|body| {
                {
                    body_json_objects(body).iter().find_map(|json| {
                        json.get("stop_reason")
                            .or_else(|| {
                                json.get("delta").and_then(|delta| delta.get("stop_reason"))
                            })
                            .and_then(Value::as_str)
                            .map(ToOwned::to_owned)
                    })
                }
            })
            .collect()
    }
}

fn assert_recorded_wire_reason(scenario: &str, expected: &str) {
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
fn recorded_request_caps(scenario: &str) -> Vec<u64> {
    {
        interaction_bodies(scenario, "when")
            .iter()
            .filter_map(|body| serde_json::from_str::<Value>(body).ok())
            .filter_map(|body| body.get("max_tokens").and_then(Value::as_u64))
            .collect()
    }
}

fn assert_recorded_request_cap(scenario: &str, expected: u64) {
    {
        let caps = recorded_request_caps(scenario);
        assert!(
            caps.contains(&expected),
            "cassette {scenario} no longer records a request capped at {expected} (recorded: {caps:?})"
        );
    }
}
