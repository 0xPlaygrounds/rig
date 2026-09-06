//! Native handler-layer corpus: real middleware around independently dispatched ECS effects.
use super::super::support::{
    with_anthropic_corpus_hooks_cassette, with_anthropic_corpus_layers_cassette,
};
use super::corpus_layers::{ADD_PROMPT, NAME_PROMPT, tool_record_args, tool_record_outputs};
use crate::{
    ecs_agent::{EcsAgent, RuntimeHandler},
    goldens::{
        Adder, CONVERSATION, DENY_REASON, DenyAddLayer, MEMORY_KEY, PATCHED_AGAIN_ARGS,
        PATCHED_ARGS, PatchAddArgsLayer, PatchAgainLayer, REPLACED_RESULT, ReplaceAddResultLayer,
        ReplaceLoadLayer, families, replaced_history,
    },
    support::{BASIC_PREAMBLE, TOOLS_PREAMBLE},
};
use bevy_ecs::prelude::*;
use rig::{
    effect::{EffectFamily, EffectKind},
    prelude::*,
    providers::anthropic::completion::CLAUDE_SONNET_4_6,
    serve::{
        ErasedHandler,
        adapters::{MemoryAdapter, ToolAdapter},
    },
};
use rig_ecs::{
    agent::{
        Conversation, Grant, Order, Parts, PolicyVersion, Remembered, Remembers, Temperature,
        ToolCallSlot,
    },
    bus::{BusSet, EffectOutcome, Handlers, Issued, PendingEffect, RigSchedule},
    systems::{Fresh, RigSet},
};
use std::sync::Arc;

fn layered_agent(
    client: rig::providers::anthropic::Client,
    layers: impl FnOnce(ErasedHandler) -> ErasedHandler,
) -> EcsAgent {
    let mut ecs = EcsAgent::for_golden(
        client.completion_model(CLAUDE_SONNET_4_6),
        TOOLS_PREAMBLE,
        false,
    );
    ecs.app
        .world_mut()
        .entity_mut(ecs.agent)
        .insert(Temperature(Some(0.0)));
    // Keep middleware outside the executor adapter so the recorder observes the
    // inner exchange, before the outer after-verdict changes the delivered result.
    let handler = layers(ErasedHandler::new(RuntimeHandler {
        inner: Arc::new(ToolAdapter::new(Adder)),
        runtime: tokio::runtime::Handle::current(),
    }));
    ecs.declared_policies = handler.descriptor().layers.clone();
    let tool = Handlers::with(ecs.app.world_mut(), |h| {
        h.register_erased("golden/tool:add#0", handler)
    })
    .expect("bus")
    .expect("fresh layered tool");
    ecs.app
        .world_mut()
        .spawn((Grant(tool), Order(0), ChildOf(ecs.agent)));
    ecs
}
async fn run_tool(ecs: &mut EcsAgent) -> rig::effect_log::EffectLog {
    let response = ecs.prompt_with_max_turns(ADD_PROMPT, false, Some(3)).await;
    assert!(!response.is_empty());
    ecs.effect_log()
}
async fn own_bus(
    client: rig::providers::anthropic::Client,
    layers: impl FnOnce(ErasedHandler) -> ErasedHandler,
    configure: impl FnOnce(&mut EcsAgent),
) -> rig::effect_log::EffectLog {
    let mut ecs = layered_agent(client, layers);
    configure(&mut ecs);
    run_tool(&mut ecs).await
}
type UnissuedTools<'w, 's> = Query<
    'w,
    's,
    &'static mut PendingEffect,
    (With<ToolCallSlot>, Without<Issued>, Without<EffectOutcome>),
>;
fn patch_args(mut tools: UnissuedTools) {
    for mut effect in &mut tools {
        if let EffectKind::ToolCall { name, args } = &mut effect.kind
            && name == "add"
        {
            *args = PATCHED_ARGS.into();
        }
    }
}
#[derive(Resource, Default)]
struct HistoryChecks(usize);
fn check_history(
    fresh: Query<&ChildOf, Added<Fresh>>,
    remembered: Query<(&ChildOf, &Order, &Parts), With<Remembered>>,
    mut checks: ResMut<HistoryChecks>,
) {
    for turn in &fresh {
        let mut history: Vec<_> = remembered
            .iter()
            .filter(|(parent, _, _)| parent.parent() == turn.parent())
            .collect();
        history.sort_by_key(|(_, order, _)| order.0);
        assert_eq!(
            history
                .iter()
                .map(|(_, _, parts)| parts.0.to_message())
                .collect::<Vec<_>>(),
            replaced_history()
        );
        checks.0 += 1;
    }
}
fn memory_agent(client: rig::providers::anthropic::Client) -> EcsAgent {
    let mut memory_entity = None;
    let mut ecs = EcsAgent::for_golden_with_setup(
        client.completion_model(CLAUDE_SONNET_4_6),
        BASIC_PREAMBLE,
        false,
        |world| {
            let memory = ErasedHandler::new(RuntimeHandler {
                inner: Arc::new(MemoryAdapter::new(
                    rig::memory::InMemoryConversationMemory::new(),
                )),
                runtime: tokio::runtime::Handle::current(),
            })
            .layered(ReplaceLoadLayer);
            memory_entity = Some(
                Handlers::with(world, |h| h.register_erased(MEMORY_KEY, memory))
                    .expect("bus")
                    .expect("fresh memory"),
            );
        },
    );
    ecs.app.world_mut().entity_mut(ecs.agent).insert((
        Remembers(memory_entity.expect("registered memory")),
        Conversation(CONVERSATION.into()),
        Temperature(Some(0.0)),
        PolicyVersion("ecs-layers/v1:history_is_replaced".into()),
    ));
    ecs.declared_policies = vec!["HistoryIsReplaced".into(), "ReplaceLoadLayer".into()];
    ecs.app.init_resource::<HistoryChecks>().add_systems(
        RigSchedule,
        check_history.after(RigSet::Select).before(RigSet::Assemble),
    );
    ecs
}

#[tokio::test]
async fn deny_tool_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_hooks_cassette("corpus_hooks/deny_tool", |client| async move {
        let log = own_bus(client, |adder| adder.layered(DenyAddLayer), |_| {}).await;
        assert_eq!(
            families(&log),
            [EffectFamily::Completion, EffectFamily::Completion]
        );
        assert_eq!(log.header.hooks, ["DenyAddLayer"]);
        crate::ecs_goldens::golden_effects("anthropic_layers_deny_tool", &log);
    })
    .await;
}

#[tokio::test]
async fn patch_tool_args_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_hooks_cassette("corpus_hooks/patch_tool_args", |client| async move {
        let log = own_bus(client, |adder| adder.layered(PatchAddArgsLayer), |_| {}).await;
        assert_eq!(tool_record_args(&log), [r#"{"x":40,"y":2}"#]);
        assert_eq!(log.header.hooks, ["PatchAddArgsLayer"]);
        crate::ecs_goldens::golden_effects("anthropic_layers_patch_tool_args", &log);
    })
    .await;
}

#[tokio::test]
async fn replace_tool_result_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_hooks_cassette("corpus_hooks/replace_tool_result", |client| async move {
        let log = own_bus(client, |adder| adder.layered(ReplaceAddResultLayer), |_| {}).await;
        assert_eq!(
            tool_record_outputs(&log),
            ["42"],
            "the record holds the tool's answer"
        );
        assert_eq!(log.header.hooks, ["ReplaceAddResultLayer"]);
        crate::ecs_goldens::golden_effects("anthropic_layers_replace_tool_result", &log);
    })
    .await;
}

#[tokio::test]
async fn two_layers_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_hooks_cassette("corpus_hooks/two_hooks", |client| async move {
        // Outermost first in the header: the patch sees the dispatch
        // first, the replacement sees the answer first.
        let log = own_bus(
            client,
            |adder| {
                adder
                    .layered(ReplaceAddResultLayer)
                    .layered(PatchAddArgsLayer)
            },
            |_| {},
        )
        .await;
        assert_eq!(tool_record_args(&log), [r#"{"x":40,"y":2}"#]);
        assert_eq!(tool_record_outputs(&log), ["42"]);
        assert_eq!(
            log.header.hooks,
            ["PatchAddArgsLayer", "ReplaceAddResultLayer"]
        );
        crate::ecs_goldens::golden_effects("anthropic_layers_two_layers", &log);
    })
    .await;
}

#[tokio::test]
async fn host_deny_over_host_bus_effect_log_is_the_golden_fixture() {
    // The host's own policy on the agent's tool key, over the host's bus.
    with_anthropic_corpus_hooks_cassette("corpus_hooks/deny_tool", |client| async move {
        let mut ecs = layered_agent(client, |adder| adder.layered(DenyAddLayer));
        ecs.declare_bus_policy = false;
        let log = run_tool(&mut ecs).await;
        assert!(
            ecs.app
                .world_mut()
                .query::<(&PendingEffect, Option<&EffectOutcome>)>()
                .iter(ecs.app.world())
                .all(|(_, outcome)| outcome.is_some()),
            "host effects completed before app teardown"
        );
        drop(ecs);
        assert_eq!(log.header.bus, None);
        assert_eq!(
            families(&log),
            [EffectFamily::Completion, EffectFamily::Completion]
        );
        assert_eq!(log.header.hooks, ["DenyAddLayer"]);
        let _ = DENY_REASON;
        crate::ecs_goldens::golden_effects("anthropic_layers_host_deny_over_host_bus", &log);
    })
    .await;
}

#[tokio::test]
async fn patch_beneath_hook_patch_effect_log_is_the_golden_fixture() {
    // The agent's hook patches first (40 + 2); the host's layer beneath it
    // patches again (30 + 12): the record holds what was served.
    with_anthropic_corpus_hooks_cassette("corpus_hooks/patch_tool_args", |client| async move {
        let log = own_bus(
            client,
            |adder| adder.layered(PatchAgainLayer),
            |ecs| {
                ecs.declared_policies.insert(0, "PatchAddArgs".into());
                ecs.app
                    .add_systems(RigSchedule, patch_args.in_set(BusSet::Gate));
                ecs.app
                    .world_mut()
                    .entity_mut(ecs.agent)
                    .insert(PolicyVersion("ecs-layers/v1:patch_args".into()));
            },
        )
        .await;
        assert_eq!(tool_record_args(&log), [PATCHED_AGAIN_ARGS]);
        assert_eq!(tool_record_outputs(&log), ["42"]);
        assert_eq!(log.header.hooks, ["PatchAddArgs", "PatchAgainLayer"]);
        crate::ecs_goldens::golden_effects("anthropic_layers_patch_beneath_hook_patch", &log);
    })
    .await;
}

#[tokio::test]
async fn memory_load_replaced_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_layers_cassette(
        "corpus_layers/memory_load_replaced",
        |client| async move {
            let mut ecs = memory_agent(client);
            let response = ecs.prompt(NAME_PROMPT, false).await;
            assert!(response.contains("Ada"), "{response}");
            assert_eq!(ecs.app.world().resource::<HistoryChecks>().0, 1, "run-start history assertion executed");
            let log = ecs.effect_log();
            assert_eq!(
                families(&log),
                [
                    EffectFamily::Memory,
                    EffectFamily::Completion,
                    EffectFamily::Memory
                ]
            );
            // The record holds the store's answer: an empty conversation.
            assert!(
                matches!(
                    &log.records[0].outcome,
                    Ok(rig::effect::Outcome::Memory(rig::effect::MemoryOutcome::Loaded { messages })) if messages.is_empty()
                ),
                "{:?}",
                log.records[0].outcome
            );
            assert_eq!(log.header.hooks, ["HistoryIsReplaced", "ReplaceLoadLayer"]);
            let _ = REPLACED_RESULT;
            crate::ecs_goldens::golden_effects("anthropic_layers_memory_load_replaced", &log);
        },
    )
    .await;
}
