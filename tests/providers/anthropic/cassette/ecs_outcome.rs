//! Native provider-executed failure and cancellation corpus.
#[path = "ecs_outcome/delivery.rs"]
mod delivery;
use super::super::support::{
    with_anthropic_cassette, with_anthropic_cassette_bogus_key,
    with_anthropic_corpus_outcome_cassette,
};
use super::corpus_outcome::{ADD_PROMPT, NOTE_PREAMBLE, NOTE_PROMPT, tool_outcome};
use crate::ecs_agent::EcsAgent;
use crate::goldens::{BROKEN_ADD, FailingAdd, WriteNote, families};
use crate::support::{Adder, BASIC_PREAMBLE, BASIC_PROMPT, TOOLS_PREAMBLE};
use bevy_ecs::prelude::*;
use rig::effect::EffectFamily;
use rig::error::ErrorKind;
use rig::prelude::*;
use rig::providers::anthropic::completion::CLAUDE_SONNET_4_6;
use rig::streaming::{Delta, StreamEvent};
use rig_ecs::{
    agent::{DefaultMaxTurns, Failure, MaxTurns, RunResult, Settled, Temperature},
    bus::{BusSet, EffectOutcome, RigSchedule, Streamed},
    systems::{RigSet, spawn_run},
};

// Application consumer at the public publication boundary. Despawning the
// issued entity drops its owned native task; the runtime records cancellation
// and ends its owning run. No expected event prefix drives this operation.
fn drop_at_tool_delta(
    mut commands: Commands,
    streams: Query<(Entity, &Streamed), Without<EffectOutcome>>,
) {
    for (entity, stream) in &streams {
        if stream.events.iter().any(|event| {
            matches!(
                event,
                StreamEvent::BlockDelta {
                    delta: Delta::ToolName { .. } | Delta::ToolArguments { .. },
                    ..
                }
            )
        }) {
            commands.entity(entity).despawn();
        }
    }
}

#[tokio::test]
async fn cancel_after_tool_call_delta_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_outcome_cassette(
        "corpus_outcome/cancel_after_tool_call_delta",
        |client| async move {
            let mut ecs = EcsAgent::for_golden(
                delivery::FirstToolDelta::new(client.completion_model(CLAUDE_SONNET_4_6)),
                NOTE_PREAMBLE,
                true,
            );
            ecs.app
                .world_mut()
                .entity_mut(ecs.agent)
                .insert(Temperature(Some(0.0)));
            ecs.tool(WriteNote);

            ecs.app.add_systems(
                RigSchedule,
                drop_at_tool_delta
                    .after(BusSet::Collect)
                    .before(RigSet::Fold),
            );
            let run = spawn_run(
                ecs.app.world_mut(),
                ecs.agent,
                &[],
                NOTE_PROMPT,
                true,
                Some(3),
            );
            let error = ecs
                .wait_for_outcome(run)
                .await
                .expect_err("dropping the native stream cancels the run");
            assert!(matches!(error, Failure::Cancelled(_)), "{error:?}");
            assert!(ecs.app.world().get::<Settled>(run).is_none());
            assert!(ecs.app.world().get::<RunResult>(run).is_none());
            for _ in 0..64 {
                tokio::task::yield_now().await;
            }
            let log = ecs.effect_log();
            assert_eq!(families(&log), [EffectFamily::Completion]);
            let report = log.records[0]
                .outcome
                .as_ref()
                .expect_err("a dropped stream is recorded as a cancel");
            assert_eq!(report.kind, ErrorKind::Cancelled, "{report:?}");
            crate::ecs_goldens::golden_effects(
                "anthropic_outcome_cancel_after_tool_call_delta",
                &log,
            );
        },
    )
    .await;
}

/// A tool that fails: the tool record's outcome is a failed result, the
/// model sees the failure and answers around it.
#[tokio::test]
async fn tool_error_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_outcome_cassette("corpus_outcome/tool_error", |client| async move {
        let mut ecs = EcsAgent::for_golden(
            client.completion_model(CLAUDE_SONNET_4_6),
            TOOLS_PREAMBLE,
            false,
        );
        ecs.app
            .world_mut()
            .entity_mut(ecs.agent)
            .insert(Temperature(Some(0.0)));
        ecs.tool(FailingAdd);

        let output = ecs.prompt_with_max_turns(ADD_PROMPT, false, Some(3)).await;
        assert!(!output.is_empty());
        let log = ecs.effect_log();
        assert_eq!(
            families(&log),
            [
                EffectFamily::Completion,
                EffectFamily::Tool,
                EffectFamily::Completion
            ]
        );
        let result = tool_outcome(&log);
        assert!(result.is_error(), "{result:?}");
        assert!(result.output().render().contains(BROKEN_ADD), "{result:?}");
        crate::ecs_goldens::golden_effects("anthropic_outcome_tool_error", &log);
    })
    .await;
}

/// The same, streamed with events kept.
#[tokio::test]
async fn tool_error_streamed_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_outcome_cassette(
        "corpus_outcome/tool_error_streamed",
        |client| async move {
            let mut ecs = EcsAgent::for_golden(
                client.completion_model(CLAUDE_SONNET_4_6),
                TOOLS_PREAMBLE,
                true,
            );
            ecs.app
                .world_mut()
                .entity_mut(ecs.agent)
                .insert(Temperature(Some(0.0)));
            ecs.tool(FailingAdd);

            let output = ecs.prompt_with_max_turns(ADD_PROMPT, true, Some(3)).await;
            assert!(!output.is_empty());
            let log = ecs.effect_log();
            assert_eq!(
                families(&log),
                [
                    EffectFamily::Completion,
                    EffectFamily::Tool,
                    EffectFamily::Completion
                ]
            );
            assert!(log.records[0].events.is_some(), "events are kept");
            assert!(tool_outcome(&log).is_error());
            crate::ecs_goldens::golden_effects("anthropic_outcome_tool_error_streamed", &log);
        },
    )
    .await;
}

/// The wire's own error: an invalid key, a 401 envelope. The completion
/// record's outcome is the provider's error and the run fails at it.
#[tokio::test]
async fn model_error_effect_log_is_the_golden_fixture() {
    with_anthropic_cassette_bogus_key("corpus_outcome/model_error", |client| async move {
        let mut ecs = EcsAgent::for_golden(
            client.completion_model(CLAUDE_SONNET_4_6),
            BASIC_PREAMBLE,
            false,
        );
        ecs.app
            .world_mut()
            .entity_mut(ecs.agent)
            .insert(Temperature(Some(0.0)));

        let run = spawn_run(
            ecs.app.world_mut(),
            ecs.agent,
            &[],
            BASIC_PROMPT,
            false,
            None,
        );
        let error = ecs
            .wait_for_outcome(run)
            .await
            .expect_err("an invalid key is refused");
        let kind = match &error {
            Failure::Provider(report) => report.kind,
            other => panic!("a provider report, not {other:?}"),
        };
        assert_eq!(kind, ErrorKind::ProviderResponse, "{error:?}");
        let log = ecs.effect_log();
        assert_eq!(families(&log), [EffectFamily::Completion]);
        let report = log.records[0]
            .outcome
            .as_ref()
            .expect_err("the record holds the provider's error");
        assert_eq!(report.kind, ErrorKind::ProviderResponse);
        assert_eq!(report.http_status, Some(401), "{report:?}");
        crate::ecs_goldens::golden_effects("anthropic_outcome_model_error", &log);
    })
    .await;
}

/// The same, streamed: the error arrives as the stream's first item.
#[tokio::test]
async fn model_error_streamed_effect_log_is_the_golden_fixture() {
    with_anthropic_cassette_bogus_key("corpus_outcome/model_error_streamed", |client| async move {
        let mut ecs = EcsAgent::for_golden(
            client.completion_model(CLAUDE_SONNET_4_6),
            BASIC_PREAMBLE,
            true,
        );
        ecs.app
            .world_mut()
            .entity_mut(ecs.agent)
            .insert(Temperature(Some(0.0)));

        let run = spawn_run(
            ecs.app.world_mut(),
            ecs.agent,
            &[],
            BASIC_PROMPT,
            true,
            None,
        );
        let error = ecs
            .wait_for_outcome(run)
            .await
            .expect_err("an invalid key is refused");
        assert!(matches!(error, Failure::Provider(_)), "{error:?}");
        let log = ecs.effect_log();
        let kinds: Vec<_> = log.header.stream_errors.values().flatten().collect();
        assert_eq!(kinds.len(), 1, "one actual error item: {kinds:?}");
        assert!(ecs.app.world().get::<Settled>(run).is_none());
        assert!(ecs.app.world().get::<RunResult>(run).is_none());
        assert_eq!(families(&log), [EffectFamily::Completion]);
        let report = log.records[0]
            .outcome
            .as_ref()
            .expect_err("the record holds the provider's error");
        assert_eq!(report.kind, ErrorKind::ProviderResponse, "{report:?}");
        crate::ecs_goldens::golden_effects("anthropic_outcome_model_error_streamed", &log);
    })
    .await;
}

/// The runner's budget exhausted with a tool call pending: one model call
/// allowed, the tool runs, the next call is refused by the budget. Two
/// records, then `MaxTurnsError`. Its own recording: the run makes one
/// request, and a cassette with a second interaction refuses to leave it
/// unused.
#[tokio::test]
async fn max_turns_exhausted_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_outcome_cassette(
        "corpus_outcome/max_turns_exhausted",
        |client| async move {
            let mut ecs = EcsAgent::for_golden(
                client.completion_model(CLAUDE_SONNET_4_6),
                TOOLS_PREAMBLE,
                false,
            );
            ecs.app
                .world_mut()
                .entity_mut(ecs.agent)
                .insert(Temperature(Some(0.0)));
            ecs.tool(Adder);

            let run = spawn_run(
                ecs.app.world_mut(),
                ecs.agent,
                &[],
                ADD_PROMPT,
                false,
                Some(1),
            );
            let error = ecs
                .wait_for_outcome(run)
                .await
                .expect_err("one call cannot finish a tool turn");
            assert!(matches!(error, Failure::MaxTurns { limit: 1 }), "{error:?}");
            let log = ecs.effect_log();
            assert_eq!(
                families(&log),
                [EffectFamily::Completion, EffectFamily::Tool]
            );
            crate::ecs_goldens::golden_effects("anthropic_outcome_max_turns_exhausted", &log);
        },
    )
    .await;
}

/// The builder's `default_max_turns` is in the spec the header hashes; the
/// runner's `max_turns` is not. This cell is the tool-call turn under a
/// default budget of three and no runner budget: its records are the
/// `anthropic_tool_call_turn` golden's, its header is another program's.
#[tokio::test]
async fn default_max_turns_effect_log_is_the_golden_fixture() {
    with_anthropic_cassette("effect_corpus/tool_call_turn", |client| async move {
        let mut ecs = EcsAgent::for_golden(
            client.completion_model(CLAUDE_SONNET_4_6),
            TOOLS_PREAMBLE,
            false,
        );
        ecs.app
            .world_mut()
            .entity_mut(ecs.agent)
            .insert(Temperature(Some(0.0)));
        ecs.tool(Adder);
        ecs.app
            .world_mut()
            .entity_mut(ecs.agent)
            .insert((DefaultMaxTurns(Some(3)), MaxTurns(3)));

        let output = ecs.prompt(ADD_PROMPT, false).await;
        assert!(output.contains("42"), "{}", output);
        let log = ecs.effect_log();
        assert_eq!(
            families(&log),
            [
                EffectFamily::Completion,
                EffectFamily::Tool,
                EffectFamily::Completion
            ]
        );
        crate::ecs_goldens::golden_effects("anthropic_outcome_default_max_turns", &log);
    })
    .await;
}
