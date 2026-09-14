//! Provider-integrated native implementations of the complete hook family.
use super::super::support::with_anthropic_corpus_hooks_cassette;
use super::corpus_hooks::{
    ADD_PROMPT, request_at, tool_record_args, tool_record_outputs, tool_result_texts,
};
use crate::ecs_agent::{EcsAgent, RuntimeHandler};
use crate::goldens::{
    DENY_REASON, LOOKUP_ARGS, PIRATE_PREAMBLE, REPLACED_ANSWER, REPLACED_RESULT, families,
};
use crate::support::{Adder, BASIC_PREAMBLE, BASIC_PROMPT, TOOLS_PREAMBLE};
use bevy_ecs::prelude::*;
use rig::effect::{EffectFamily, Outcome};
use rig::message::{AssistantContent, Message};
use rig::prelude::*;
use rig::providers::anthropic::completion::CLAUDE_SONNET_4_6;
use rig_ecs::{
    agent::*,
    bus::{Bound, BusSet, Handlers, RigSchedule},
    systems::RigSet,
};
#[path = "ecs_hooks/policies.rs"]
mod policies;
use policies::*;

#[tokio::test]
async fn observe_everything_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_hooks_cassette("corpus_hooks/observe_everything", |client| async move {
        let mut ecs = EcsAgent::for_golden_with_setup(
            client.completion_model(CLAUDE_SONNET_4_6),
            TOOLS_PREAMBLE,
            false,
            |world| {
                Handlers::with(world, |handlers| {
                    handlers.register(
                        "golden/memory",
                        RuntimeHandler {
                            inner: std::sync::Arc::new(
                                rig_core::serve::adapters::MemoryAdapter::new(
                                    rig_core::memory::InMemoryConversationMemory::new(),
                                ),
                            ),
                            runtime: tokio::runtime::Handle::current(),
                        },
                    )
                })
                .expect("bus installed")
                .expect("memory key");
            },
        );
        let memory = ecs
            .app
            .world_mut()
            .query::<(Entity, &Bound)>()
            .iter(ecs.app.world())
            .find(|(_, b)| b.key.as_str() == "golden/memory")
            .expect("registered memory")
            .0;
        ecs.app.world_mut().entity_mut(ecs.agent).insert((
            Remembers(memory),
            Conversation("golden-conversation".into()),
        ));
        ecs.app
            .world_mut()
            .entity_mut(ecs.agent)
            .insert(Temperature(Some(0.0)));
        ecs.tool(Adder);
        ecs.app.init_resource::<Observed>().add_systems(
            RigSchedule,
            observe_all.after(BusSet::Dispatch).before(BusSet::Collect),
        );
        ecs.declared_policies = vec!["ObserveEverything".into()];
        ecs.app
            .world_mut()
            .entity_mut(ecs.agent)
            .insert(PolicyVersion("ecs-hooks/v1:ObserveEverything".into()));

        let output = ecs.prompt_with_max_turns(ADD_PROMPT, false, Some(3)).await;
        assert!(output.contains("42"), "{}", output);
        let log = ecs.effect_log();
        assert_eq!(
            families(&log),
            [
                EffectFamily::Memory,
                EffectFamily::Completion,
                EffectFamily::Tool,
                EffectFamily::Completion,
                EffectFamily::Memory,
            ]
        );
        assert_eq!(log.header.hooks, ["ObserveEverything"]);
        assert_eq!(ecs.app.world().resource::<Observed>().0, families(&log));
        crate::ecs_goldens::golden_effects("anthropic_hooks_observe_everything", &log);
    })
    .await;
}

/// `on_dispatch` → `Patch`: the tool record holds the patched arguments,
/// the model's history keeps the call it made.

#[tokio::test]
async fn patch_tool_args_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_hooks_cassette("corpus_hooks/patch_tool_args", |client| async move {
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
            .add_systems(RigSchedule, patch_args.in_set(BusSet::Gate));
        ecs.declared_policies = vec!["PatchAddArgs".into()];
        ecs.app
            .world_mut()
            .entity_mut(ecs.agent)
            .insert(PolicyVersion("ecs-hooks/v1:PatchAddArgs".into()));

        let output = ecs.prompt_with_max_turns(ADD_PROMPT, false, Some(3)).await;
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
        assert_eq!(tool_record_args(&log), [r#"{"x":40,"y":2}"#]);
        let history = request_at(&log, 2);
        let called = history
            .chat_history
            .iter()
            .find_map(|message| match message {
                Message::Assistant { content, .. } => content.iter().find_map(|c| match c {
                    AssistantContent::ToolCall(call) => Some(call.function.arguments.clone()),
                    _ => None,
                }),
                _ => None,
            })
            .expect("the model's call is in history");
        assert_eq!(called, serde_json::json!({"x": 17, "y": 25}));
        crate::ecs_goldens::golden_effects("anthropic_hooks_patch_tool_args", &log);
    })
    .await;
}

/// The same, streamed with events kept.

#[tokio::test]
async fn patch_tool_args_streamed_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_hooks_cassette(
        "corpus_hooks/patch_tool_args_streamed",
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
            ecs.tool(Adder);
            ecs.app
                .add_systems(RigSchedule, patch_args.in_set(BusSet::Gate));
            ecs.declared_policies = vec!["PatchAddArgs".into()];
            ecs.app
                .world_mut()
                .entity_mut(ecs.agent)
                .insert(PolicyVersion("ecs-hooks/v1:PatchAddArgs".into()));

            let output = ecs.prompt_with_max_turns(ADD_PROMPT, true, Some(3)).await;
            assert!(output.contains("42"), "{output}");
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
            assert_eq!(tool_record_args(&log), [r#"{"x":40,"y":2}"#]);
            crate::ecs_goldens::golden_effects("anthropic_hooks_patch_tool_args_streamed", &log);
        },
    )
    .await;
}

/// `on_dispatch` → `Deny`: no tool record; the model sees the reason as
/// the tool's result and answers without it.

#[tokio::test]
async fn deny_tool_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_hooks_cassette("corpus_hooks/deny_tool", |client| async move {
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
            .add_systems(RigSchedule, deny_tools.in_set(BusSet::Gate));
        ecs.declared_policies = vec!["DenyAdd".into()];
        ecs.app
            .world_mut()
            .entity_mut(ecs.agent)
            .insert(PolicyVersion("ecs-hooks/v1:DenyAdd".into()));

        let output = ecs.prompt_with_max_turns(ADD_PROMPT, false, Some(3)).await;
        assert!(!output.is_empty());
        let log = ecs.effect_log();
        assert_eq!(
            families(&log),
            [EffectFamily::Completion, EffectFamily::Completion]
        );
        assert_eq!(tool_result_texts(request_at(&log, 1)), [DENY_REASON]);
        crate::ecs_goldens::golden_effects("anthropic_hooks_deny_tool", &log);
    })
    .await;
}

/// The same, streamed with events kept.

#[tokio::test]
async fn deny_tool_streamed_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_hooks_cassette("corpus_hooks/deny_tool_streamed", |client| async move {
        let mut ecs = EcsAgent::for_golden(
            client.completion_model(CLAUDE_SONNET_4_6),
            TOOLS_PREAMBLE,
            true,
        );
        ecs.app
            .world_mut()
            .entity_mut(ecs.agent)
            .insert(Temperature(Some(0.0)));
        ecs.tool(Adder);
        ecs.app
            .add_systems(RigSchedule, deny_tools.in_set(BusSet::Gate));
        ecs.declared_policies = vec!["DenyAdd".into()];
        ecs.app
            .world_mut()
            .entity_mut(ecs.agent)
            .insert(PolicyVersion("ecs-hooks/v1:DenyAdd".into()));

        let output = ecs.prompt_with_max_turns(ADD_PROMPT, true, Some(3)).await;
        assert!(!output.is_empty());
        let log = ecs.effect_log();
        assert_eq!(
            families(&log),
            [EffectFamily::Completion, EffectFamily::Completion]
        );
        assert!(log.records[0].events.is_some(), "events are kept");
        assert_eq!(tool_result_texts(request_at(&log, 1)), [DENY_REASON]);
        crate::ecs_goldens::golden_effects("anthropic_hooks_deny_tool_streamed", &log);
    })
    .await;
}

/// `on_outcome` → `Replace` on a tool: the record holds the tool's answer,
/// the transcript the replacement, and the model answers from the latter.

#[tokio::test]
async fn replace_tool_result_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_hooks_cassette("corpus_hooks/replace_tool_result", |client| async move {
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
            .add_systems(RigSchedule, replace_results.in_set(BusSet::Judge));
        ecs.declared_policies = vec!["ReplaceAddResult".into()];
        ecs.app
            .world_mut()
            .entity_mut(ecs.agent)
            .insert(PolicyVersion("ecs-hooks/v1:ReplaceAddResult".into()));

        let output = ecs.prompt_with_max_turns(ADD_PROMPT, false, Some(3)).await;
        assert!(output.contains(REPLACED_RESULT), "{}", output);
        let log = ecs.effect_log();
        assert_eq!(
            families(&log),
            [
                EffectFamily::Completion,
                EffectFamily::Tool,
                EffectFamily::Completion
            ]
        );
        assert_eq!(tool_record_outputs(&log), ["42"]);
        assert_eq!(tool_result_texts(request_at(&log, 2)), [REPLACED_RESULT]);
        crate::ecs_goldens::golden_effects("anthropic_hooks_replace_tool_result", &log);
    })
    .await;
}

/// `on_outcome` → `Replace` on a completion: the run's output is the
/// replacement, the record the model's text.

#[tokio::test]
async fn replace_answer_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_hooks_cassette("corpus_hooks/replace_answer", |client| async move {
        let mut ecs = EcsAgent::for_golden(
            client.completion_model(CLAUDE_SONNET_4_6),
            BASIC_PREAMBLE,
            false,
        );
        ecs.app
            .world_mut()
            .entity_mut(ecs.agent)
            .insert(Temperature(Some(0.0)));
        ecs.app
            .add_systems(RigSchedule, replace_answer.in_set(BusSet::Judge));
        ecs.declared_policies = vec!["ReplaceAnswer".into()];
        ecs.app
            .world_mut()
            .entity_mut(ecs.agent)
            .insert(PolicyVersion("ecs-hooks/v1:ReplaceAnswer".into()));

        let output = ecs.prompt_with_max_turns(BASIC_PROMPT, false, None).await;
        assert_eq!(output, REPLACED_ANSWER);
        let log = ecs.effect_log();
        assert_eq!(families(&log), [EffectFamily::Completion]);
        let recorded = match &log.records[0].outcome {
            Ok(Outcome::Completion(response)) => response,
            other => panic!("a completion, not {other:?}"),
        };
        assert!(
            !recorded
                .choice
                .iter()
                .any(|c| matches!(c, AssistantContent::Text(t) if t.text == REPLACED_ANSWER)),
            "the record holds the model's answer, not the replacement"
        );
        crate::ecs_goldens::golden_effects("anthropic_hooks_replace_answer", &log);
    })
    .await;
}

/// `on_completion_call` → a request patch: the request's system prompt is
/// the hook's, the spec's preamble is the builder's.

#[tokio::test]
async fn preamble_override_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_hooks_cassette("corpus_hooks/preamble_override", |client| async move {
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
            preamble.after(RigSet::Select).before(RigSet::Assemble),
        );
        ecs.declared_policies = vec!["PreambleOverride".into()];
        ecs.app
            .world_mut()
            .entity_mut(ecs.agent)
            .insert(PolicyVersion("ecs-hooks/v1:PreambleOverride".into()));

        let output = ecs.prompt_with_max_turns(BASIC_PROMPT, false, None).await;
        assert!(!output.is_empty());
        let log = ecs.effect_log();
        assert_eq!(families(&log), [EffectFamily::Completion]);
        assert_eq!(
            request_at(&log, 0).system_instructions(),
            Some(PIRATE_PREAMBLE)
        );
        assert_eq!(
            ecs.app
                .world()
                .get::<Preamble>(ecs.agent)
                .expect("agent preamble")
                .0
                .as_deref(),
            Some(BASIC_PREAMBLE)
        );
        crate::ecs_goldens::golden_effects("anthropic_hooks_preamble_override", &log);
    })
    .await;
}

/// `on_model_turn_finished` → `Retry` with feedback: the first answer lacks
/// `DONE`, the hook asks again, the second has it. Two completions.

#[tokio::test]
async fn demand_done_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_hooks_cassette("corpus_hooks/demand_done", |client| async move {
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
            demand_done.after(RigSet::Fold).before(RigSet::Judge),
        );
        ecs.declared_policies = vec!["DemandDone".into()];
        ecs.app
            .world_mut()
            .entity_mut(ecs.agent)
            .insert(PolicyVersion("ecs-hooks/v1:DemandDone".into()));

        let output = ecs
            .prompt_with_max_turns(BASIC_PROMPT, false, Some(3))
            .await;
        assert!(output.contains("DONE"), "{}", output);
        let log = ecs.effect_log();
        assert_eq!(
            families(&log),
            [EffectFamily::Completion, EffectFamily::Completion]
        );
        crate::ecs_goldens::golden_effects("anthropic_hooks_demand_done", &log);
    })
    .await;
}

/// A hook that dispatches through the run's bus in `on_run_start`: the
/// hook's own tool call is the first record, under the tool's key.

#[tokio::test]
async fn lookup_before_run_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_hooks_cassette("corpus_hooks/lookup_before_run", |client| async move {
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
            .add_observer(lookup_at_start)
            .configure_sets(RigSchedule, RigSet::Advance.run_if(lookup_finished));
        ecs.declared_policies = vec!["LookupBeforeRun".into()];
        ecs.app
            .world_mut()
            .entity_mut(ecs.agent)
            .insert(PolicyVersion("ecs-hooks/v1:LookupBeforeRun".into()));

        let output = ecs.prompt_with_max_turns(ADD_PROMPT, false, Some(3)).await;
        assert!(output.contains("42"), "{}", output);
        let log = ecs.effect_log();
        assert_eq!(
            families(&log),
            [
                EffectFamily::Tool,
                EffectFamily::Completion,
                EffectFamily::Tool,
                EffectFamily::Completion
            ]
        );
        assert_eq!(tool_record_args(&log)[0], LOOKUP_ARGS);
        assert_eq!(log.records[0].key.as_str(), crate::goldens::LOOKUP_KEY);
        crate::ecs_goldens::golden_effects("anthropic_hooks_lookup_before_run", &log);
    })
    .await;
}

/// Two hooks in a stack: the header names both in registration order, and
/// both decisions land (the patched call in the record, the replaced
/// result in the transcript).

#[tokio::test]
async fn two_hooks_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_hooks_cassette("corpus_hooks/two_hooks", |client| async move {
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
            .add_systems(RigSchedule, patch_args.in_set(BusSet::Gate));
        ecs.app
            .add_systems(RigSchedule, replace_results.in_set(BusSet::Judge));
        ecs.declared_policies = vec!["PatchAddArgs".into(), "ReplaceAddResult".into()];
        ecs.app
            .world_mut()
            .entity_mut(ecs.agent)
            .insert(PolicyVersion(
                "ecs-hooks/v1:PatchAddArgs+ReplaceAddResult".into(),
            ));

        let output = ecs.prompt_with_max_turns(ADD_PROMPT, false, Some(3)).await;
        assert!(output.contains(REPLACED_RESULT), "{}", output);
        let log = ecs.effect_log();
        assert_eq!(log.header.hooks, ["PatchAddArgs", "ReplaceAddResult"]);
        assert_eq!(tool_record_args(&log), [r#"{"x":40,"y":2}"#]);
        assert_eq!(tool_record_outputs(&log), ["42"]);
        assert_eq!(tool_result_texts(request_at(&log, 2)), [REPLACED_RESULT]);
        crate::ecs_goldens::golden_effects("anthropic_hooks_two_hooks", &log);
    })
    .await;
}
