//! Native application policies for the hook-ended provider corpus.
use super::{
    super::support::{with_anthropic_cassette, with_anthropic_corpus_endings_cassette},
    corpus_endings::{ADD_PROMPT, ESSAY_PROMPT, NOTE_PREAMBLE, NOTE_PROMPT, last_outcome_kind},
};
use crate::{
    ecs_agent::EcsAgent,
    goldens::{
        CANCEL_ADD_DISPATCH, CANCEL_ADD_OUTCOME, CANCEL_ANSWER, STOP_AFTER_TURN, STOP_AT_ANSWER,
        STOP_ON_TEXT_DELTA, STOP_ON_TOOL_CALL_DELTA, WriteNote, families,
    },
    support::{Adder, BASIC_PREAMBLE, BASIC_PROMPT, TOOLS_PREAMBLE},
};
use bevy_ecs::prelude::*;
use rig::{
    effect::EffectFamily, error::ErrorKind, prelude::*,
    providers::anthropic::completion::CLAUDE_SONNET_4_6,
};
use rig_ecs::{
    agent::{AdditionalParams, Failed, Failure, PolicyVersion, RunResult, Settled, Temperature},
    bus::{BusSet, EffectOutcome, PendingEffect, RigSchedule},
    systems::{RigSet, spawn_run},
};
#[path = "ecs_endings/policies.rs"]
mod policies;
use policies::*;
// Select concrete native systems at setup only. No hook engine is executed.
#[derive(Clone, Copy, Debug)]
enum Ending {
    CancelAddDispatch,
    CancelAddOutcome,
    CancelAnswer,
    StopAfterTurn,
    StopAtAnswer,
    StopOnTextDelta,
    StopOnToolCallDelta,
}
use Ending::*;
#[derive(Clone, Copy)]
enum Streamed {
    Tools,
    Essay,
    Note,
}
fn agent(
    client: &rig::providers::anthropic::Client,
    ending: Ending,
    preamble: &str,
    streamed: bool,
) -> EcsAgent {
    let model = client.completion_model(CLAUDE_SONNET_4_6);
    let mut ecs = if matches!(ending, StopOnToolCallDelta) {
        // Backpressure after the real first delta lets the native policy cancel
        // before transport scheduling can publish additional argument chunks.
        EcsAgent::for_golden(
            super::ecs_outcome::FirstToolDelta::new(model),
            preamble,
            streamed,
        )
    } else {
        EcsAgent::for_golden(model, preamble, streamed)
    };
    ecs.app.world_mut().entity_mut(ecs.agent).insert((
        Temperature(Some(0.0)),
        PolicyVersion(format!("ecs-endings/v1:{ending:?},RecordSettled")),
    ));
    ecs.declared_policies = vec![format!("{ending:?}"), "RecordSettled".into()];
    ecs.app
        .init_resource::<Terminal>()
        .add_observer(failed)
        .add_observer(settled);
    match ending {
        CancelAddDispatch => {
            ecs.app
                .add_systems(RigSchedule, cancel_dispatch.in_set(BusSet::Gate));
        }
        CancelAddOutcome => {
            ecs.app
                .add_systems(RigSchedule, cancel_tool_outcome.in_set(BusSet::Judge));
        }
        CancelAnswer => {
            ecs.app
                .add_systems(RigSchedule, cancel_answer.in_set(BusSet::Judge));
        }
        StopAfterTurn => {
            ecs.app.add_systems(
                RigSchedule,
                stop_after_turn.after(RigSet::Fold).before(RigSet::Judge),
            );
        }
        StopAtAnswer => {
            ecs.app.add_systems(
                RigSchedule,
                stop_at_answer.after(RigSet::Fold).before(RigSet::Judge),
            );
        }
        StopOnTextDelta => {
            ecs.app.add_systems(
                RigSchedule,
                stop_text_delta.after(BusSet::Collect).before(RigSet::Fold),
            );
        }
        StopOnToolCallDelta => {
            ecs.app.add_systems(
                RigSchedule,
                stop_tool_delta.after(BusSet::Collect).before(RigSet::Fold),
            );
        }
    };
    ecs
}
async fn cancelled_run(ecs: &mut EcsAgent, run: Entity, reason: &str) {
    let error = ecs
        .wait_for_outcome(run)
        .await
        .expect_err("the policy stops the run");
    match error {
        Failure::Cancelled(report) => assert_eq!(report.message, reason),
        other => panic!("a cancelled run, not {other:?}"),
    }
    assert!(ecs.app.world().get::<Failed>(run).is_some());
    assert!(
        ecs.app.world().get::<Settled>(run).is_none(),
        "no successful terminal"
    );
    assert!(
        ecs.app.world().get::<RunResult>(run).is_none(),
        "no final response"
    );
    let seen = ecs.app.world().resource::<Terminal>().0.get(&run);
    assert!(
        seen.is_some_and(|seen| seen.starts_with("error:")),
        "on_run_settled saw the error: {seen:?}"
    );
    assert!(
        ecs.effect_log().header.stream_errors.is_empty(),
        "no unexpected provider stream errors"
    );
    let world = ecs.app.world_mut();
    let mut pending = world.query_filtered::<Option<&EffectOutcome>, With<PendingEffect>>();
    assert!(
        pending.iter(world).all(|outcome| outcome.is_some()),
        "all retained effects have finished; dropped stream is gone"
    );
}
async fn unary_tool_run(
    client: rig::providers::anthropic::Client,
    ending: Ending,
    reason: &str,
    shape: &[EffectFamily],
    thinking: bool,
) -> rig::effect_log::EffectLog {
    let mut ecs = agent(&client, ending, TOOLS_PREAMBLE, false);
    ecs.tool(Adder);
    if thinking {
        ecs.app.world_mut().entity_mut(ecs.agent).insert((
            Temperature(None),
            AdditionalParams(Some(
                serde_json::json!({"thinking":{"type":"enabled","budget_tokens":1024}}),
            )),
        ));
    }
    let run = spawn_run(
        ecs.app.world_mut(),
        ecs.agent,
        &[],
        ADD_PROMPT,
        false,
        Some(3),
    );
    cancelled_run(&mut ecs, run, reason).await;
    let log = ecs.effect_log();
    assert_eq!(families(&log), shape);
    log
}
async fn streamed_run(
    client: rig::providers::anthropic::Client,
    ending: Ending,
    reason: &str,
    program: Streamed,
) -> rig::effect_log::EffectLog {
    let (preamble, prompt) = match program {
        Streamed::Tools => (TOOLS_PREAMBLE, ADD_PROMPT),
        Streamed::Essay => (BASIC_PREAMBLE, ESSAY_PROMPT),
        Streamed::Note => (NOTE_PREAMBLE, NOTE_PROMPT),
    };
    let mut ecs = agent(&client, ending, preamble, true);
    match program {
        Streamed::Tools => ecs.tool(Adder),
        Streamed::Note => ecs.tool(WriteNote),
        Streamed::Essay => {}
    }
    let run = spawn_run(ecs.app.world_mut(), ecs.agent, &[], prompt, true, Some(3));
    cancelled_run(&mut ecs, run, reason).await;
    for _ in 0..64 {
        tokio::task::yield_now().await;
    }
    ecs.effect_log()
}

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
            crate::ecs_goldens::golden_effects("anthropic_endings_tool_dispatch_cancelled", &log);
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
            crate::ecs_goldens::golden_effects("anthropic_endings_tool_outcome_cancelled", &log);
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
            let mut ecs = agent(&client, CancelAnswer, BASIC_PREAMBLE, false);
            let run = spawn_run(
                ecs.app.world_mut(),
                ecs.agent,
                &[],
                BASIC_PROMPT,
                false,
                None,
            );
            cancelled_run(&mut ecs, run, CANCEL_ANSWER).await;
            let log = ecs.effect_log();
            assert_eq!(families(&log), [EffectFamily::Completion]);
            assert!(
                log.records[0].outcome.is_ok(),
                "the record holds the answer"
            );
            crate::ecs_goldens::golden_effects("anthropic_endings_answer_outcome_cancelled", &log);
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
            crate::ecs_goldens::golden_effects("anthropic_endings_turn_finished_stop", &log);
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
            crate::ecs_goldens::golden_effects("anthropic_endings_answer_turn_stop", &log);
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
        crate::ecs_goldens::golden_effects("anthropic_endings_text_delta_stop", &log);
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
            crate::ecs_goldens::golden_effects("anthropic_endings_tool_call_delta_stop", &log);
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
            crate::ecs_goldens::golden_effects(
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
            crate::ecs_goldens::golden_effects(
                "anthropic_endings_turn_finished_stop_streamed",
                &log,
            );
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
            crate::ecs_goldens::golden_effects(
                "anthropic_endings_tool_outcome_cancelled_streamed",
                &log,
            );
        },
    )
    .await;
}
