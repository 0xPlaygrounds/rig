//! Independent native per-turn policies, exercised against the original provider cassettes.
use super::super::support::with_anthropic_corpus_shaping_cassette;
use super::corpus_shaping::{
    ADD_PROMPT, CONTEXT_PROMPT, NAME_PROMPT, SUM_EVENT_PROMPT, TOOL_TURN, request_at,
};
use crate::ecs_agent::{EcsAgent, RuntimeHandler};
use crate::goldens::{PIRATE_PREAMBLE, SHAPING_CONTEXT, event_schema, families};
use crate::support::{Adder, BASIC_PREAMBLE, TOOLS_PREAMBLE};
use bevy_ecs::prelude::*;
use rig::effect::{EffectFamily, HandlerKey};
use rig::message::ToolChoice;
use rig::prelude::*;
use rig::providers::anthropic::completion::{CLAUDE_HAIKU_4_5, CLAUDE_SONNET_4_6};
use rig_ecs::{
    agent::*,
    bus::{Handlers, RigSchedule},
    systems::{RigSet, spawn_run},
};
#[path = "ecs_shaping/policies.rs"]
mod policies;
use policies::*;

#[tokio::test]
async fn tool_choice_required_first_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_shaping_cassette(
        "corpus_shaping/tool_choice_required_first",
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
            ecs.app.add_systems(
                RigSchedule,
                required_first
                    .after(RigSet::Select)
                    .before(RigSet::Assemble),
            );
            ecs.declared_policies = vec!["PatchToolChoiceRequiredFirst".into()];
            ecs.app
                .world_mut()
                .entity_mut(ecs.agent)
                .insert(PolicyVersion("ecs-shaping/v1:required_first".into()));

            let output = ecs.prompt_with_max_turns(ADD_PROMPT, false, Some(3)).await;
            assert!(output.contains("42"), "{output}");
            let log = ecs.effect_log();
            assert_eq!(families(&log), TOOL_TURN);
            assert_eq!(request_at(&log, 0).tool_choice, Some(ToolChoice::Required));
            assert_eq!(request_at(&log, 2).tool_choice, None);
            crate::ecs_goldens::golden_effects(
                "anthropic_shaping_tool_choice_required_first",
                &log,
            );
        },
    )
    .await;
}

/// `tool_choice: None` on turn 2 of a committed `Tool`-mode run: the turn
/// cannot call the output tool (the engine warns and proceeds); what the
/// run then does is the record.

#[tokio::test]
async fn tool_choice_none_on_committed_output_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_shaping_cassette(
        "corpus_shaping/tool_choice_none_on_committed_output",
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
            ecs.app.world_mut().entity_mut(ecs.agent).insert(Output {
                mode: OutputKind::Tool,
                schema: Some(event_schema().into()),
            });
            ecs.app.add_systems(
                RigSchedule,
                none_second.after(RigSet::Select).before(RigSet::Assemble),
            );
            ecs.declared_policies = vec!["PatchToolChoiceNoneSecond".into()];
            ecs.app
                .world_mut()
                .entity_mut(ecs.agent)
                .insert(PolicyVersion("ecs-shaping/v1:none_second".into()));

            let run = spawn_run(
                ecs.app.world_mut(),
                ecs.agent,
                &[],
                SUM_EVENT_PROMPT,
                false,
                Some(3),
            );
            let outcome = ecs.wait_for_outcome(run).await;
            let log = ecs.effect_log();
            // The patched turn answers in text, the run's output validation
            // reprompts for the output tool, and the unpatched turn 3 calls it.
            assert!(outcome.is_ok(), "{outcome:?}");
            assert_eq!(
                families(&log),
                [
                    EffectFamily::Completion,
                    EffectFamily::Tool,
                    EffectFamily::Completion,
                    EffectFamily::Completion
                ]
            );
            assert_eq!(request_at(&log, 2).tool_choice, Some(ToolChoice::None));
            crate::ecs_goldens::golden_effects(
                "anthropic_shaping_tool_choice_none_on_committed_output",
                &log,
            );
        },
    )
    .await;
}

/// A context document patched into every turn's request.

#[tokio::test]
async fn extra_context_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_shaping_cassette("corpus_shaping/extra_context", |client| async move {
        let mut ecs = EcsAgent::for_golden(
            client.completion_model(CLAUDE_SONNET_4_6),
            BASIC_PREAMBLE,
            false,
        );
        ecs.app
            .world_mut()
            .entity_mut(ecs.agent)
            .insert(Temperature(Some(0.0)));
        ecs.app.add_systems(
            RigSchedule,
            extra_context.after(RigSet::Select).before(RigSet::Assemble),
        );
        ecs.declared_policies = vec!["PatchExtraContext".into()];
        ecs.app
            .world_mut()
            .entity_mut(ecs.agent)
            .insert(PolicyVersion("ecs-shaping/v1:extra_context".into()));

        let output = ecs
            .prompt_with_max_turns(CONTEXT_PROMPT, false, Some(3))
            .await;
        assert!(output.to_lowercase().contains("jiro"), "{output}");
        let log = ecs.effect_log();
        assert_eq!(families(&log), [EffectFamily::Completion]);
        assert!(
            request_at(&log, 0)
                .documents
                .iter()
                .any(|doc| doc.text == SHAPING_CONTEXT),
            "the patched document is in the request"
        );
        crate::ecs_goldens::golden_effects("anthropic_shaping_extra_context", &log);
    })
    .await;
}

/// The same, streamed with events.

#[tokio::test]
async fn extra_context_streamed_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_shaping_cassette(
        "corpus_shaping/extra_context_streamed",
        |client| async move {
            let mut ecs = EcsAgent::for_golden(
                client.completion_model(CLAUDE_SONNET_4_6),
                BASIC_PREAMBLE,
                true,
            );
            ecs.app
                .world_mut()
                .entity_mut(ecs.agent)
                .insert(Temperature(Some(0.0)));
            ecs.app.add_systems(
                RigSchedule,
                extra_context.after(RigSet::Select).before(RigSet::Assemble),
            );
            ecs.declared_policies = vec!["PatchExtraContext".into()];
            ecs.app
                .world_mut()
                .entity_mut(ecs.agent)
                .insert(PolicyVersion("ecs-shaping/v1:extra_context".into()));

            let output = ecs.prompt(CONTEXT_PROMPT, true).await;
            assert!(output.to_lowercase().contains("jiro"), "{output}");
            let log = ecs.effect_log();
            assert!(log.records[0].events.is_some(), "events are kept");
            crate::ecs_goldens::golden_effects("anthropic_shaping_extra_context_streamed", &log);
        },
    )
    .await;
}

/// Three hooks' patches merged in registration order: a preamble, a
/// document, a first-turn tool choice.

#[tokio::test]
async fn merged_three_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_shaping_cassette("corpus_shaping/merged_three", |client| async move {
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
        ecs.app.add_systems(
            RigSchedule,
            (preamble_always, extra_context, required_first)
                .chain()
                .after(RigSet::Select)
                .before(RigSet::Assemble),
        );
        ecs.declared_policies = vec![
            "PreambleOverride".into(),
            "PatchExtraContext".into(),
            "PatchToolChoiceRequiredFirst".into(),
        ];
        ecs.app
            .world_mut()
            .entity_mut(ecs.agent)
            .insert(PolicyVersion(
                "ecs-shaping/v1:preamble_always+extra_context+required_first".into(),
            ));

        let output = ecs.prompt_with_max_turns(ADD_PROMPT, false, Some(3)).await;
        assert!(output.contains("42"), "{output}");
        let log = ecs.effect_log();
        assert_eq!(families(&log), TOOL_TURN);
        let first = request_at(&log, 0);
        assert!(
            first
                .system_instructions()
                .is_some_and(|system| system.starts_with(PIRATE_PREAMBLE)),
            "the patched preamble"
        );
        assert!(
            first
                .documents
                .iter()
                .any(|doc| doc.text == SHAPING_CONTEXT)
        );
        assert_eq!(first.tool_choice, Some(ToolChoice::Required));
        assert_eq!(request_at(&log, 2).tool_choice, None);
        crate::ecs_goldens::golden_effects("anthropic_shaping_merged_three", &log);
    })
    .await;
}

/// The route selected on the first turn only: `fast` asks, the default
/// answers.

#[tokio::test]
async fn route_on_first_turn_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_shaping_cassette(
        "corpus_shaping/route_on_first_turn",
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
            let route = Handlers::with(ecs.app.world_mut(), |handlers| {
                handlers.register(
                    "golden/model:fast",
                    RuntimeHandler {
                        inner: std::sync::Arc::new(
                            rig_core::serve::adapters::CompletionAdapter::new(
                                "fast",
                                client.completion_model(CLAUDE_HAIKU_4_5),
                            ),
                        ),
                        runtime: tokio::runtime::Handle::current(),
                    },
                )
            })
            .expect("bus installed")
            .expect("fresh route");
            ecs.app.insert_resource(SelectedRoute(route));
            ecs.app
                .world_mut()
                .spawn((Route(route), ChildOf(ecs.agent)));
            ecs.tool(Adder);
            ecs.app.add_systems(
                RigSchedule,
                route_first.after(RigSet::Advance).before(RigSet::Select),
            );
            ecs.declared_policies = vec!["RouteOnFirstTurn".into()];
            ecs.app
                .world_mut()
                .entity_mut(ecs.agent)
                .insert(PolicyVersion("ecs-shaping/v1:route_first".into()));

            let output = ecs.prompt_with_max_turns(ADD_PROMPT, false, Some(3)).await;
            assert!(output.contains("42"), "{output}");
            let log = ecs.effect_log();
            assert_eq!(families(&log), TOOL_TURN);
            assert_eq!(log.records[0].key, HandlerKey::from("golden/model:fast"));
            assert_eq!(log.records[2].key, HandlerKey::from("golden/model:default"));
            crate::ecs_goldens::golden_effects("anthropic_shaping_route_on_first_turn", &log);
        },
    )
    .await;
}

/// A route registered after build (`register_model`) and selected on
/// every turn: served, recorded, in the signature and the handler table,
/// and not in the required row (the row is the builder's).

#[tokio::test]
async fn late_route_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_shaping_cassette("corpus_shaping/late_route", |client| async move {
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
        let route = Handlers::with(ecs.app.world_mut(), |handlers| {
            handlers.register(
                "golden/model:late",
                RuntimeHandler {
                    inner: std::sync::Arc::new(rig_core::serve::adapters::CompletionAdapter::new(
                        "late",
                        client.completion_model(CLAUDE_HAIKU_4_5),
                    )),
                    runtime: tokio::runtime::Handle::current(),
                },
            )
        })
        .expect("bus installed")
        .expect("fresh route");
        ecs.app.insert_resource(SelectedRoute(route));
        ecs.app.add_systems(
            RigSchedule,
            route_always.after(RigSet::Advance).before(RigSet::Select),
        );
        ecs.declared_policies = vec!["SelectLate".into()];
        ecs.app
            .world_mut()
            .entity_mut(ecs.agent)
            .insert(PolicyVersion("ecs-shaping/v1:route_always".into()));

        let output = ecs.prompt_with_max_turns(ADD_PROMPT, false, Some(3)).await;
        assert!(output.contains("42"), "{output}");
        let log = ecs.effect_log();
        assert_eq!(families(&log), TOOL_TURN);
        let late = HandlerKey::from("golden/model:late");
        assert_eq!(log.records[0].key, late);
        assert!(
            !log.header.required.contains_key(&late),
            "{:?}",
            log.header.required
        );
        assert!(log.header.signature.contains_key(&late));
        assert!(
            log.header
                .handlers
                .iter()
                .any(|handler| handler.key == late)
        );
        crate::ecs_goldens::golden_effects("anthropic_shaping_late_route", &log);
    })
    .await;
}

/// `max_tokens: 5` on turn 2: the answer is cut where the patch says.

#[tokio::test]
async fn max_tokens_second_turn_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_shaping_cassette(
        "corpus_shaping/max_tokens_second_turn",
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
            ecs.app.add_systems(
                RigSchedule,
                max_tokens_second
                    .after(RigSet::Select)
                    .before(RigSet::Assemble),
            );
            ecs.declared_policies = vec!["PatchMaxTokensSecond".into()];
            ecs.app
                .world_mut()
                .entity_mut(ecs.agent)
                .insert(PolicyVersion("ecs-shaping/v1:max_tokens_second".into()));

            let _ = ecs.prompt_with_max_turns(ADD_PROMPT, false, Some(3)).await;
            let log = ecs.effect_log();
            assert_eq!(families(&log), TOOL_TURN);
            assert_eq!(request_at(&log, 0).max_tokens, None);
            assert_eq!(request_at(&log, 2).max_tokens, Some(5));
            crate::ecs_goldens::golden_effects("anthropic_shaping_max_tokens_second_turn", &log);
        },
    )
    .await;
}

/// Extended thinking on turn 2 only (with the temperature it needs).

#[tokio::test]
async fn thinking_second_turn_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_shaping_cassette(
        "corpus_shaping/thinking_second_turn",
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
            ecs.app.add_systems(
                RigSchedule,
                thinking_second
                    .after(RigSet::Select)
                    .before(RigSet::Assemble),
            );
            ecs.declared_policies = vec!["PatchThinkingSecond".into()];
            ecs.app
                .world_mut()
                .entity_mut(ecs.agent)
                .insert(PolicyVersion("ecs-shaping/v1:thinking_second".into()));

            let output = ecs.prompt_with_max_turns(ADD_PROMPT, false, Some(3)).await;
            assert!(output.contains("42"), "{output}");
            let log = ecs.effect_log();
            assert_eq!(families(&log), TOOL_TURN);
            assert!(request_at(&log, 0).additional_params.is_none());
            assert!(request_at(&log, 2).additional_params.is_some());
            assert_eq!(request_at(&log, 2).temperature, Some(1.0));
            crate::ecs_goldens::golden_effects("anthropic_shaping_thinking_second_turn", &log);
        },
    )
    .await;
}

/// The pirate preamble on turn 2 only.

#[tokio::test]
async fn preamble_second_turn_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_shaping_cassette(
        "corpus_shaping/preamble_second_turn",
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
            ecs.app.add_systems(
                RigSchedule,
                preamble_second
                    .after(RigSet::Select)
                    .before(RigSet::Assemble),
            );
            ecs.declared_policies = vec!["PatchPreambleSecond".into()];
            ecs.app
                .world_mut()
                .entity_mut(ecs.agent)
                .insert(PolicyVersion("ecs-shaping/v1:preamble_second".into()));

            let _ = ecs.prompt_with_max_turns(ADD_PROMPT, false, Some(3)).await;
            let log = ecs.effect_log();
            assert_eq!(families(&log), TOOL_TURN);
            assert!(
                request_at(&log, 0)
                    .system_instructions()
                    .is_some_and(|system| system.starts_with(TOOLS_PREAMBLE))
            );
            assert!(
                request_at(&log, 2)
                    .system_instructions()
                    .is_some_and(|system| system.starts_with(PIRATE_PREAMBLE))
            );
            crate::ecs_goldens::golden_effects("anthropic_shaping_preamble_second_turn", &log);
        },
    )
    .await;
}

/// No tools advertised on turn 2 (`active_tools: []`): the answer turn
/// sees none.

#[tokio::test]
async fn active_tools_none_second_turn_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_shaping_cassette(
        "corpus_shaping/active_tools_none_second_turn",
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
            ecs.app.add_systems(
                RigSchedule,
                active_tools_second
                    .after(RigSet::Select)
                    .before(RigSet::Assemble),
            );
            ecs.declared_policies = vec!["PatchActiveToolsNoneSecond".into()];
            ecs.app
                .world_mut()
                .entity_mut(ecs.agent)
                .insert(PolicyVersion("ecs-shaping/v1:active_tools_second".into()));

            let output = ecs.prompt_with_max_turns(ADD_PROMPT, false, Some(3)).await;
            assert!(output.contains("42"), "{output}");
            let log = ecs.effect_log();
            assert_eq!(families(&log), TOOL_TURN);
            assert_eq!(request_at(&log, 0).tools.len(), 1);
            assert!(request_at(&log, 2).tools.is_empty());
            crate::ecs_goldens::golden_effects(
                "anthropic_shaping_active_tools_none_second_turn",
                &log,
            );
        },
    )
    .await;
}

/// A prior exchange patched in as turn 1's history.

#[tokio::test]
async fn history_first_turn_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_shaping_cassette(
        "corpus_shaping/history_first_turn",
        |client| async move {
            let mut ecs = EcsAgent::for_golden(
                client.completion_model(CLAUDE_SONNET_4_6),
                BASIC_PREAMBLE,
                false,
            );
            ecs.app
                .world_mut()
                .entity_mut(ecs.agent)
                .insert(Temperature(Some(0.0)));
            ecs.app.add_systems(
                RigSchedule,
                history_first.after(RigSet::Select).before(RigSet::Assemble),
            );
            ecs.declared_policies = vec!["PatchHistoryFirst".into()];
            ecs.app
                .world_mut()
                .entity_mut(ecs.agent)
                .insert(PolicyVersion("ecs-shaping/v1:history_first".into()));

            let output = ecs.prompt_with_max_turns(NAME_PROMPT, false, Some(3)).await;
            assert!(output.contains("Ada"), "{output}");
            let log = ecs.effect_log();
            assert_eq!(families(&log), [EffectFamily::Completion]);
            assert!(
                request_at(&log, 0).chat_history.len() >= 3,
                "the patched exchange precedes the prompt: {:?}",
                request_at(&log, 0).chat_history
            );
            crate::ecs_goldens::golden_effects("anthropic_shaping_history_first_turn", &log);
        },
    )
    .await;
}
