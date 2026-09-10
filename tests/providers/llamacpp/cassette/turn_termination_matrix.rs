//! Model-turn termination metadata (rig#2184 / #2341) against llama.cpp.
//!
//! `ModelTurnFinished` carries `finish_reason: Option<&FinishReason>` and
//! `max_tokens: Option<u64>` — the normalized reason the provider stopped, and
//! the effective output-token cap *that attempt* ran under. Anthropic, Gemini
//! and OpenAI each have a recorded matrix for it; llama.cpp had none, and its
//! whole vocabulary is reachable without asking a model to misbehave.
//!
//! **Server**: the competent tier — `unsloth/Qwen3-8B-GGUF` Q4_K_M,
//! `--jinja --seed 42 --temp 0 -c 8192`, `llama-server` b10499-6d05498. The
//! tool cells need a model that reliably calls; the rest would run on the
//! smoke tier but stay here so one server records the whole matrix.
//!
//! This provider's vocabulary, and what it normalizes to:
//!
//! | normalized | llama.cpp wire value | how |
//! | --- | --- | --- |
//! | `Stop` | `stop` | direct |
//! | `Length` | `length` | direct |
//! | `ToolCalls` | `tool_calls` | direct |
//!
//! `server-task.cpp` builds the field from exactly those three, so the
//! vocabulary is *closed* — there is no `Other(_)`, no `ContentFilter`, and no
//! reachable "no reason at all" on this wire. `response_shape_matrix`'s
//! finish-reason cell sweeps the corpus and fails if a fourth value ever
//! appears.
//!
//! | # | Cell | Surface | Asserts |
//! | --- | --- | --- | --- |
//! | 1 | [`blocking_truncated_turn_reports_length_and_cap`] | blocking | `Length` + the cap this attempt ran under |
//! | 2 | [`streaming_truncated_turn_reports_length_and_cap`] | streaming | the same, on the other surface |
//! | 3 | [`blocking_completed_turn_reports_stop`] | blocking | `Stop`, and that it fails `truncated_output()` |
//! | 4 | [`blocking_tool_turn_reports_tool_calls`] | blocking | `ToolCalls` |
//! | 5 | [`blocking_escalating_retry_reports_each_attempts_own_cap`] | blocking | two attempts, two caps, two reasons |
//!
//! Cell 5 is #2184's acceptance criterion against this provider: the first
//! attempt truncates under a deliberately tiny cap, a provider-neutral hook
//! reads `FinishReason::Length` off the event and asks for a repeat with a
//! larger cap, and the second attempt reports *its own* cap rather than the
//! agent's baseline. Both attempts live in one cassette.
//!
//! Every cell re-reads its own fixture and fails if the recorded turn stopped
//! carrying the wire reason the cell is about, so a provider changing
//! behaviour cannot leave a cell green while covering nothing.

use rig::completion::FinishReason;
use rig::prelude::*;
use serde_json::Value;

use crate::cassettes::{recorded_interaction_bodies, recorded_json_request};
use crate::support::{
    Adder, EscalateCapOnTruncation, TurnTerminationProbe, collect_stream_final_response,
};

use super::super::cassette_support::*;

/// Far below what the prompt below needs, so the turn is cut short with
/// partial text kept. Large enough that Qwen3 gets past its `<think>` block —
/// a turn that spends the whole cap on hidden tokens produces no answer at
/// all, which is a different subject.
const TINY_CAP: u64 = 24;
/// Roomy enough for every prompt below to finish naturally.
const ROOMY_CAP: u64 = 512;
const TRUNCATING_PROMPT: &str = "/no_think Write two sentences about maple trees.";
const SHORT_PROMPT: &str = "/no_think Reply with exactly the word: cedar.";
const TOOL_PROMPT: &str = "/no_think Calculate 2 + 3.";
const CONCISE_PREAMBLE: &str = "You are a concise assistant. Answer directly in plain text.";
const TOOL_PREAMBLE: &str = "Use the provided tool to answer arithmetic questions.";

/// Every recorded response's `finish_reason`, in order.
fn recorded_wire_reasons(scenario: &str) -> Vec<String> {
    recorded_interaction_bodies("llamacpp", scenario)
        .iter()
        .filter_map(|(_, response)| {
            response
                .lines()
                .map(|line| line.strip_prefix("data: ").unwrap_or(line).trim())
                .filter(|line| !line.is_empty() && *line != "[DONE]")
                .filter_map(|line| serde_json::from_str::<Value>(line).ok())
                .find_map(|json| {
                    json.get("choices")?.as_array()?.iter().find_map(|choice| {
                        choice
                            .get("finish_reason")
                            .and_then(Value::as_str)
                            .map(ToOwned::to_owned)
                    })
                })
        })
        .collect()
}

/// The output-token cap of every recorded *request*, in order — proof that the
/// cap the hook reported is the cap that went on the wire.
fn recorded_request_caps(scenario: &str) -> Vec<u64> {
    recorded_interaction_bodies("llamacpp", scenario)
        .iter()
        .filter_map(|(request, _)| serde_json::from_str::<Value>(request).ok())
        .filter_map(|body| body.get("max_tokens").and_then(Value::as_u64))
        .collect()
}

fn assert_recorded_wire_reason(scenario: &str, expected: &str) {
    let reasons = recorded_wire_reasons(scenario);
    assert!(
        reasons.contains(&expected.to_owned()),
        "cassette {scenario} no longer records a `{expected}` finish reason \
         (recorded: {reasons:?}); this cell would pass while covering nothing"
    );
}

// ---------------------------------------------------------------------------
// Length
// ---------------------------------------------------------------------------

#[tokio::test]
async fn blocking_truncated_turn_reports_length_and_cap() {
    let probe = TurnTerminationProbe::default();
    let observed = probe.clone();

    with_llamacpp_competent_cassette(
        "turn_termination_matrix/blocking_truncated_turn",
        |client| async move {
            client
                .agent(CASSETTE_MODEL)
                .preamble(CONCISE_PREAMBLE)
                .temperature(0.0)
                .max_tokens(TINY_CAP)
                .add_hook(probe)
                .build()
                .runner(TRUNCATING_PROMPT)
                .run()
                .await
                .expect("a partially truncated turn still carries an answer");
        },
    )
    .await;

    assert_eq!(
        observed.first_reason(),
        Some(FinishReason::Length),
        "llama.cpp's wire `length` must reach the hook as FinishReason::Length"
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
    assert_recorded_wire_reason("turn_termination_matrix/blocking_truncated_turn", "length");
    assert_eq!(
        recorded_request_caps("turn_termination_matrix/blocking_truncated_turn"),
        vec![TINY_CAP]
    );
}

#[tokio::test]
async fn streaming_truncated_turn_reports_length_and_cap() {
    let probe = TurnTerminationProbe::default();
    let observed = probe.clone();

    with_llamacpp_competent_cassette(
        "turn_termination_matrix/streaming_truncated_turn",
        |client| async move {
            let agent = client
                .agent(CASSETTE_MODEL)
                .preamble(CONCISE_PREAMBLE)
                .temperature(0.0)
                .max_tokens(TINY_CAP)
                .build();

            let mut stream = agent
                .stream_prompt(TRUNCATING_PROMPT)
                .add_hook(probe)
                .stream()
                .await;
            let _ = collect_stream_final_response(&mut stream).await;
        },
    )
    .await;

    assert_eq!(
        observed.first_reason(),
        Some(FinishReason::Length),
        "the streaming surface must report the same reason as the blocking one"
    );
    assert_eq!(observed.first_max_tokens(), Some(TINY_CAP));
    assert_recorded_wire_reason("turn_termination_matrix/streaming_truncated_turn", "length");
}

// ---------------------------------------------------------------------------
// Stop
// ---------------------------------------------------------------------------

#[tokio::test]
async fn blocking_completed_turn_reports_stop() {
    let probe = TurnTerminationProbe::default();
    let observed = probe.clone();

    with_llamacpp_competent_cassette(
        "turn_termination_matrix/blocking_completed_turn",
        |client| async move {
            client
                .agent(CASSETTE_MODEL)
                .preamble(CONCISE_PREAMBLE)
                .temperature(0.0)
                .max_tokens(ROOMY_CAP)
                .add_hook(probe)
                .build()
                .runner(SHORT_PROMPT)
                .run()
                .await
                .expect("a short prompt under a roomy cap should finish");
        },
    )
    .await;

    assert_eq!(observed.first_reason(), Some(FinishReason::Stop));
    assert_eq!(observed.first_max_tokens(), Some(ROOMY_CAP));
    assert!(
        observed
            .first_reason()
            .is_some_and(|reason| !reason.truncated_output()),
        "a completed turn must not satisfy the retry predicate, or every turn \
         would be retried forever"
    );
    assert_recorded_wire_reason("turn_termination_matrix/blocking_completed_turn", "stop");
}

// ---------------------------------------------------------------------------
// ToolCalls
// ---------------------------------------------------------------------------

#[tokio::test]
async fn blocking_tool_turn_reports_tool_calls() {
    let probe = TurnTerminationProbe::default();
    let observed = probe.clone();

    with_llamacpp_competent_cassette(
        "turn_termination_matrix/blocking_tool_turn",
        |client| async move {
            client
                .agent(CASSETTE_MODEL)
                .preamble(TOOL_PREAMBLE)
                .temperature(0.0)
                .max_tokens(ROOMY_CAP)
                .tool(Adder)
                .add_hook(probe)
                .build()
                .runner(TOOL_PROMPT)
                .max_turns(4)
                .run()
                .await
                .expect("a tool round trip should complete");
        },
    )
    .await;

    assert_eq!(
        observed.first_reason(),
        Some(FinishReason::ToolCalls),
        "llama.cpp reports a distinct value for a tool turn, so the hook must \
         see ToolCalls rather than Stop: {:?}",
        observed.observations()
    );
    assert_recorded_wire_reason("turn_termination_matrix/blocking_tool_turn", "tool_calls");
}

// ---------------------------------------------------------------------------
// The escalation loop
// ---------------------------------------------------------------------------

#[tokio::test]
async fn blocking_escalating_retry_reports_each_attempts_own_cap() {
    let probe = TurnTerminationProbe::default();
    let escalate = EscalateCapOnTruncation::new(TINY_CAP, ROOMY_CAP);
    let observed = probe.clone();
    let escalations = escalate.clone();

    with_llamacpp_competent_cassette(
        "turn_termination_matrix/escalating_retry",
        |client| async move {
            client
                .agent(CASSETTE_MODEL)
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
                .runner(TRUNCATING_PROMPT)
                .max_turns(2)
                .run()
                .await
                .expect("the retried attempt should answer");
        },
    )
    .await;

    assert_eq!(
        observed.observations(),
        vec![
            (Some(FinishReason::Length), Some(TINY_CAP)),
            (Some(FinishReason::Stop), Some(ROOMY_CAP)),
        ],
        "each attempt must report its own post-patch cap, never the agent's \
         baseline of 64"
    );
    assert_eq!(escalations.escalations(), vec![ROOMY_CAP]);
    assert_eq!(escalations.retries(), 1);

    // ...and the recorded traffic corroborates it.
    assert_eq!(
        recorded_request_caps("turn_termination_matrix/escalating_retry"),
        vec![TINY_CAP, ROOMY_CAP]
    );
    assert_eq!(
        recorded_wire_reasons("turn_termination_matrix/escalating_retry"),
        vec!["length".to_owned(), "stop".to_owned()]
    );
    // The prompt is identical on both attempts, so the only difference is the
    // cap — otherwise the second attempt's `stop` could be explained by asking
    // an easier question.
    let first = recorded_json_request("llamacpp", "turn_termination_matrix/escalating_retry");
    assert_eq!(first["max_tokens"], serde_json::json!(TINY_CAP));
}
