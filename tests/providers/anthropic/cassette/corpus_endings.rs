//! Matrix F of the effect corpus: hook-ended runs. Every `Stop` in the
//! hook surface that fires after at least one dispatch, on the Anthropic
//! wire (`CLAUDE_SONNET_4_6`, temperature 0), each ending the run in
//! `PromptCancelled` with the hook's reason. The cells that stop before
//! any dispatch are mock-scripted in `tests/core/golden_endings.rs` (no
//! wire, no cassette). Producers of the goldens
//! `crates/rig-verify/tests/corpus_endings.rs` replays by both
//! interpreters; the enumeration lives there.

use futures::StreamExt;
use rig::agent::{AgentHook, MultiTurnStreamItem, StreamingError};
use rig::completion::PromptError;
use rig::effect::EffectFamily;
use rig::error::ErrorKind;
use rig::prelude::*;
use rig::providers::anthropic::completion::CLAUDE_SONNET_4_6;

use super::super::support::{with_anthropic_cassette, with_anthropic_corpus_endings_cassette};
use crate::goldens::{
    CANCEL_ADD_DISPATCH, CANCEL_ADD_OUTCOME, CANCEL_ANSWER, CancelAddDispatch, CancelAddOutcome,
    CancelAnswer, RecordSettled, STOP_AFTER_TURN, STOP_AT_ANSWER, STOP_ON_TEXT_DELTA,
    STOP_ON_TOOL_CALL_DELTA, StopAfterTurn, StopAtAnswer, StopOnTextDelta, StopOnToolCallDelta,
    WriteNote, families,
};
use crate::support::{Adder, BASIC_PREAMBLE, BASIC_PROMPT, TOOLS_PREAMBLE};

const ADD_PROMPT: &str = "Use the add tool to add 17 and 25, then reply with just the number.";
/// The delta-stop cells stream long enough that the engine's drop lands
/// mid-stream on a live wire (a short answer arrives whole before the
/// first delta is observed): a long essay, a long tool argument, a long
/// think.
const ESSAY_PROMPT: &str =
    "Write a 600-word essay on the history of the Rust programming language.";
const NOTE_PREAMBLE: &str =
    "You are a note-taking assistant. Use the write_note tool to save notes.";
const NOTE_PROMPT: &str = "Save a note titled 'Rust' whose body is a 400-word essay on the history of the Rust programming language, then reply with just the word saved.";

fn cancelled_reason(error: &PromptError) -> &str {
    match error {
        PromptError::PromptCancelled { reason, .. } => reason,
        other => panic!("a cancelled run, not {other:?}"),
    }
}

/// The reason the stream's error item carries, after draining the stream.
async fn streamed_cancel(stream: &mut rig::agent::StreamingResult) -> String {
    let mut reason = None;
    while let Some(item) = stream.next().await {
        match item {
            Err(StreamingError::Prompt(error)) => {
                reason = Some(cancelled_reason(&error).to_owned());
            }
            Err(other) => panic!("a cancelled run, not {other:?}"),
            Ok(MultiTurnStreamItem::FinalResponse(_)) => panic!("the hook stops the run"),
            Ok(_) => {}
        }
    }
    reason.expect("the stream ends with the cancel")
}

fn assert_settled_error(settled: &RecordSettled) {
    let seen = settled.0.lock().expect("settled").clone();
    assert!(
        seen.as_deref()
            .is_some_and(|seen| seen.starts_with("error:")),
        "on_run_settled saw the error: {seen:?}"
    );
}

/// A unary tool program under `hook`, expected to end cancelled with
/// `reason`; the log's families are `shape`.
async fn unary_tool_run(
    client: rig::providers::anthropic::Client,
    hook: impl AgentHook + 'static,
    reason: &str,
    shape: &[EffectFamily],
    thinking: bool,
) -> rig::effect_log::EffectLog {
    let settled = RecordSettled::default();
    let mut builder = client
        .agent(CLAUDE_SONNET_4_6)
        .name("golden")
        .preamble(TOOLS_PREAMBLE)
        .tool(Adder)
        .add_hook(hook)
        .add_hook(settled.clone())
        .record_effects();
    builder = if thinking {
        builder.additional_params(
            serde_json::json!({ "thinking": { "type": "enabled", "budget_tokens": 1024 } }),
        )
    } else {
        builder.temperature(0.0)
    };
    let agent = builder.build();
    let error = agent
        .prompt(ADD_PROMPT)
        .max_turns(3)
        .await
        .expect_err("the hook stops the run");
    assert_eq!(cancelled_reason(&error), reason);
    assert_settled_error(&settled);
    let log = agent.take_effect_log().expect("recording");
    assert_eq!(families(&log), shape);
    log
}

/// A streamed program under `hook` (events kept), expected to end
/// cancelled with `reason`. (A reasoning-delta stop was recorded twice
/// under extended thinking and stopped: the recording proxy delivered
/// the thinking block whole, so the stop landed after it and the record
/// was a complete completion that a replay — which cancels at the delta —
/// cannot reproduce. Pruned, with the finding in the matrix doc.)
/// The program a streamed cell runs.
#[derive(Clone, Copy)]
enum Streamed {
    /// The tool program with `add`.
    Tools,
    /// The basic program asked for a long essay.
    Essay,
    /// `write_note` with a long body.
    Note,
}

async fn streamed_run(
    client: rig::providers::anthropic::Client,
    hook: impl AgentHook + 'static,
    reason: &str,
    program: Streamed,
) -> rig::effect_log::EffectLog {
    let settled = RecordSettled::default();
    let mut base = client.agent(CLAUDE_SONNET_4_6).name("golden");
    base = base.temperature(0.0);
    let agent = match program {
        Streamed::Tools => base
            .preamble(TOOLS_PREAMBLE)
            .tool(Adder)
            .add_hook(hook)
            .add_hook(settled.clone())
            .record_effects_with_events()
            .build(),
        Streamed::Note => base
            .preamble(NOTE_PREAMBLE)
            .tool(WriteNote)
            .add_hook(hook)
            .add_hook(settled.clone())
            .record_effects_with_events()
            .build(),
        Streamed::Essay => base
            .preamble(BASIC_PREAMBLE)
            .add_hook(hook)
            .add_hook(settled.clone())
            .record_effects_with_events()
            .build(),
    };
    let prompt = match program {
        Streamed::Tools => ADD_PROMPT,
        Streamed::Essay => ESSAY_PROMPT,
        Streamed::Note => NOTE_PROMPT,
    };
    {
        let mut stream = agent.stream_prompt(prompt).max_turns(3).stream().await;
        assert_eq!(streamed_cancel(&mut stream).await, reason);
    }
    for _ in 0..64 {
        tokio::task::yield_now().await;
    }
    assert_settled_error(&settled);
    agent.take_effect_log().expect("recording")
}

fn last_outcome_kind(log: &rig::effect_log::EffectLog) -> Option<ErrorKind> {
    log.records
        .last()
        .and_then(|record| record.outcome.as_ref().err().map(|report| report.kind))
}

// -- unary --------------------------------------------------------------------

/// `on_dispatch` → `Deny(Cancelled)`: the completion is recorded, the tool
/// never reaches the bus, the run stops.
#[tokio::test]
async fn tool_dispatch_cancelled_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_endings_cassette(
        "corpus_endings/tool_dispatch_cancelled",
        |client| async move {
            let log = unary_tool_run(
                client,
                CancelAddDispatch,
                CANCEL_ADD_DISPATCH,
                &[EffectFamily::Completion],
                false,
            )
            .await;
            crate::goldens::golden_effects("anthropic_endings_tool_dispatch_cancelled", &log);
        },
    )
    .await;
}

/// `on_outcome` → `Replace(Err(Cancelled))` on the tool's result: the tool
/// ran and its record holds the real result; the run stops after it.
#[tokio::test]
async fn tool_outcome_cancelled_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_endings_cassette(
        "corpus_endings/tool_outcome_cancelled",
        |client| async move {
            let log = unary_tool_run(
                client,
                CancelAddOutcome,
                CANCEL_ADD_OUTCOME,
                &[EffectFamily::Completion, EffectFamily::Tool],
                false,
            )
            .await;
            assert!(
                log.records[1].outcome.is_ok(),
                "the record holds the tool's answer"
            );
            crate::goldens::golden_effects("anthropic_endings_tool_outcome_cancelled", &log);
        },
    )
    .await;
}

/// `on_outcome` → `Replace(Err(Cancelled))` on a text answer.
#[tokio::test]
async fn answer_outcome_cancelled_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_endings_cassette(
        "corpus_endings/answer_outcome_cancelled",
        |client| async move {
            let settled = RecordSettled::default();
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .name("golden")
                .preamble(BASIC_PREAMBLE)
                .temperature(0.0)
                .add_hook(CancelAnswer)
                .add_hook(settled.clone())
                .record_effects()
                .build();
            let error = agent
                .prompt(BASIC_PROMPT)
                .await
                .expect_err("the hook stops the run");
            assert_eq!(cancelled_reason(&error), CANCEL_ANSWER);
            assert_settled_error(&settled);
            let log = agent.take_effect_log().expect("recording");
            assert_eq!(families(&log), [EffectFamily::Completion]);
            assert!(
                log.records[0].outcome.is_ok(),
                "the record holds the answer"
            );
            crate::goldens::golden_effects("anthropic_endings_answer_outcome_cancelled", &log);
        },
    )
    .await;
}

/// `on_model_turn_finished` → `Stop` on the first turn.
#[tokio::test]
async fn turn_finished_stop_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_endings_cassette(
        "corpus_endings/turn_finished_stop",
        |client| async move {
            let log = unary_tool_run(
                client,
                StopAfterTurn,
                STOP_AFTER_TURN,
                &[EffectFamily::Completion],
                false,
            )
            .await;
            crate::goldens::golden_effects("anthropic_endings_turn_finished_stop", &log);
        },
    )
    .await;
}

/// `on_model_turn_finished` → `Stop` at the answer turn of a tool program:
/// the tool turn's records precede the stop.
#[tokio::test]
async fn answer_turn_stop_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_endings_cassette(
        "corpus_endings/answer_turn_stop",
        |client| async move {
            let log = unary_tool_run(
                client,
                StopAtAnswer,
                STOP_AT_ANSWER,
                &[
                    EffectFamily::Completion,
                    EffectFamily::Tool,
                    EffectFamily::Completion,
                ],
                false,
            )
            .await;
            crate::goldens::golden_effects("anthropic_endings_answer_turn_stop", &log);
        },
    )
    .await;
}

// -- streamed -----------------------------------------------------------------

/// `on_text_delta` → `Stop`: the engine drops the model's stream at the
/// first delta, so the completion is recorded as the cancel it was, on
/// every transport.
#[tokio::test]
async fn text_delta_stop_effect_log_is_the_golden_fixture() {
    // The consumer-cancel cell's cassette (Matrix D): the same program,
    // asked for the same essay; a hook changes nothing on the wire.
    with_anthropic_cassette("effect_corpus/cancelled_stream", |client| async move {
        let log = streamed_run(client, StopOnTextDelta, STOP_ON_TEXT_DELTA, Streamed::Essay).await;
        assert_eq!(families(&log), [EffectFamily::Completion]);
        assert_eq!(
            last_outcome_kind(&log),
            Some(ErrorKind::Cancelled),
            "{:?}",
            log.records[0].outcome
        );
        crate::goldens::golden_effects("anthropic_endings_text_delta_stop", &log);
    })
    .await;
}

/// `on_tool_call_delta` → `Stop`.
#[tokio::test]
async fn tool_call_delta_stop_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_endings_cassette(
        "corpus_endings/tool_call_delta_stop",
        |client| async move {
            let log = streamed_run(
                client,
                StopOnToolCallDelta,
                STOP_ON_TOOL_CALL_DELTA,
                Streamed::Note,
            )
            .await;
            assert_eq!(families(&log), [EffectFamily::Completion]);
            assert_eq!(
                last_outcome_kind(&log),
                Some(ErrorKind::Cancelled),
                "{:?}",
                log.records[0].outcome
            );
            crate::goldens::golden_effects("anthropic_endings_tool_call_delta_stop", &log);
        },
    )
    .await;
}

/// `on_dispatch` → `Deny(Cancelled)`, streamed with events: the completion
/// completed and is recorded whole; the tool never reaches the bus.
#[tokio::test]
async fn tool_dispatch_cancelled_streamed_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_endings_cassette(
        "corpus_endings/tool_dispatch_cancelled_streamed",
        |client| async move {
            let log = streamed_run(
                client,
                CancelAddDispatch,
                CANCEL_ADD_DISPATCH,
                Streamed::Tools,
            )
            .await;
            assert_eq!(families(&log), [EffectFamily::Completion]);
            assert!(
                log.records[0].outcome.is_ok(),
                "the stream completed: {:?}",
                log.records[0].outcome
            );
            crate::goldens::golden_effects(
                "anthropic_endings_tool_dispatch_cancelled_streamed",
                &log,
            );
        },
    )
    .await;
}

/// `on_model_turn_finished` → `Stop`, streamed with events.
#[tokio::test]
async fn turn_finished_stop_streamed_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_endings_cassette(
        "corpus_endings/turn_finished_stop_streamed",
        |client| async move {
            let log = streamed_run(client, StopAfterTurn, STOP_AFTER_TURN, Streamed::Tools).await;
            assert_eq!(families(&log), [EffectFamily::Completion]);
            assert!(log.records[0].outcome.is_ok(), "the stream completed");
            crate::goldens::golden_effects("anthropic_endings_turn_finished_stop_streamed", &log);
        },
    )
    .await;
}

/// `on_outcome` → `Replace(Err(Cancelled))` on the tool, streamed.
#[tokio::test]
async fn tool_outcome_cancelled_streamed_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_endings_cassette(
        "corpus_endings/tool_outcome_cancelled_streamed",
        |client| async move {
            let log = streamed_run(
                client,
                CancelAddOutcome,
                CANCEL_ADD_OUTCOME,
                Streamed::Tools,
            )
            .await;
            assert_eq!(
                families(&log),
                [EffectFamily::Completion, EffectFamily::Tool]
            );
            assert!(
                log.records[1].outcome.is_ok(),
                "the record holds the tool's answer"
            );
            crate::goldens::golden_effects(
                "anthropic_endings_tool_outcome_cancelled_streamed",
                &log,
            );
        },
    )
    .await;
}
