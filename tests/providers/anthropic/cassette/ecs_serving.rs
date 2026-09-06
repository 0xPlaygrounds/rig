//! Native serving policy, routing and host-owned bus corpus.
use super::{
    super::support::{with_anthropic_cassette, with_anthropic_corpus_serving_cassette},
    corpus_serving::{ADD_PROMPT, TWO_TOOLS},
};
use crate::{
    ecs_agent::{EcsAgent, RuntimeHandler},
    goldens::families,
    support::{
        Adder, AlphaSignal, BetaSignal, TOOLS_PREAMBLE, TWO_TOOL_STREAM_PREAMBLE,
        TWO_TOOL_STREAM_PROMPT,
    },
};
use bevy_ecs::prelude::*;
use rig::{
    effect::{EffectFamily, HandlerKey},
    prelude::*,
    providers::anthropic::completion::{CLAUDE_HAIKU_4_5, CLAUDE_SONNET_4_6},
    serve::adapters::{CompletionAdapter, MemoryAdapter},
};
use rig_ecs::{
    agent::{
        Conversation, Cursor, PolicyVersion, Remembers, Route, Temperature, ToolPolicy, UsesModel,
    },
    bus::{Bound, EffectOutcome, Handlers, PendingEffect, Policy, RigSchedule},
    systems::{Fresh, RigSet, spawn_run},
};
use std::sync::Arc;
#[derive(Resource)]
struct FastModel(Entity);
fn route_after_first(
    fresh: Query<&ChildOf, Added<Fresh>>,
    runs: Query<&Cursor>,
    fast: Res<FastModel>,
    mut commands: Commands,
) {
    for parent in &fresh {
        if runs.get(parent.parent()).expect("native run cursor").turn > 1 {
            commands.entity(parent.parent()).insert(UsesModel(fast.0));
        }
    }
}
fn routed_agent(client: &rig::providers::anthropic::Client, selected: bool) -> EcsAgent {
    let mut ecs = EcsAgent::for_golden(
        client.completion_model(CLAUDE_SONNET_4_6),
        TOOLS_PREAMBLE,
        false,
    );
    ecs.app
        .world_mut()
        .entity_mut(ecs.agent)
        .insert(Temperature(Some(0.0)));
    let fast = Handlers::with(ecs.app.world_mut(), |handlers| {
        handlers.register(
            "golden/model:fast",
            RuntimeHandler {
                inner: Arc::new(CompletionAdapter::new(
                    "fast",
                    client.completion_model(CLAUDE_HAIKU_4_5),
                )),
                runtime: tokio::runtime::Handle::current(),
            },
        )
    })
    .expect("bus installed")
    .expect("fresh route");
    ecs.app.world_mut().spawn((Route(fast), ChildOf(ecs.agent)));
    ecs.tool(Adder);
    if selected {
        ecs.app.insert_resource(FastModel(fast)).add_systems(
            RigSchedule,
            route_after_first
                .after(RigSet::Advance)
                .before(RigSet::Select),
        );
        ecs.declared_policies = vec!["RouteAfterFirstTurn".into()];
        ecs.app
            .world_mut()
            .entity_mut(ecs.agent)
            .insert(PolicyVersion("ecs-serving/v1:route_after_first".into()));
    }
    ecs
}

async fn two_tools(
    client: rig::providers::anthropic::Client,
    bus: rig::serve::ServingPolicy,
    concurrency: usize,
    events: bool,
) -> rig::effect_log::EffectLog {
    let mut ecs = EcsAgent::for_golden(
        client.completion_model(CLAUDE_SONNET_4_6),
        TWO_TOOL_STREAM_PREAMBLE,
        events,
    );
    ecs.app.world_mut().resource_mut::<Policy>().0 = bus;
    ecs.tool(AlphaSignal);
    ecs.tool(BetaSignal);
    let run = spawn_run(
        ecs.app.world_mut(),
        ecs.agent,
        &[],
        TWO_TOOL_STREAM_PROMPT,
        true,
        Some(8),
    );
    ecs.app
        .world_mut()
        .entity_mut(run)
        .insert(ToolPolicy { concurrency });
    let output = tokio::time::timeout(std::time::Duration::from_secs(5), ecs.wait_for_success(run))
        .await
        .expect("two tools never wait on each other");
    assert!(!output.is_empty());
    let log = ecs.effect_log();
    assert_eq!(families(&log), TWO_TOOLS);
    assert_eq!(
        log.header.bus,
        Some(bus),
        "the header says how it was served"
    );
    let names: Vec<_> = log
        .records
        .iter()
        .filter_map(|record| match &record.kind {
            rig::effect::EffectKind::ToolCall { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(
        names,
        ["lookup_harbor_label", "lookup_orchard_label"],
        "dispatch order, whatever the policy"
    );
    log
}

#[tokio::test]
async fn serial_concurrency_one_effect_log_is_the_golden_fixture() {
    with_anthropic_cassette(
        "streaming_tools/streaming_tool_concurrency_emits_results_as_completed_but_persists_call_order",
        |client| async move {
            let serial = rig::serve::ServingPolicy {
                serial_per_handler: true,
                ..rig::serve::ServingPolicy::default()
            };
            let log = two_tools(client, serial, 1, false).await;
            crate::ecs_goldens::golden_effects("anthropic_serving_serial_concurrency_one", &log);
        },
    )
    .await;
}

#[tokio::test]
async fn concurrent_concurrency_one_effect_log_is_the_golden_fixture() {
    with_anthropic_cassette(
        "streaming_tools/streaming_tool_concurrency_emits_results_as_completed_but_persists_call_order",
        |client| async move {
            let log = two_tools(client, rig::serve::ServingPolicy::default(), 1, false).await;
            crate::ecs_goldens::golden_effects("anthropic_serving_concurrent_concurrency_one", &log);
        },
    )
    .await;
}

#[tokio::test]
async fn concurrent_concurrency_two_effect_log_is_the_golden_fixture() {
    with_anthropic_cassette(
        "streaming_tools/streaming_tool_concurrency_emits_results_as_completed_but_persists_call_order",
        |client| async move {
            let log = two_tools(client, rig::serve::ServingPolicy::default(), 2, false).await;
            crate::ecs_goldens::golden_effects("anthropic_serving_concurrent_concurrency_two", &log);
        },
    )
    .await;
}

/// Events kept under concurrent dispatch: the stream's delivery is the
/// record, and buffering does not reorder it.
#[tokio::test]
async fn concurrent_concurrency_two_events_effect_log_is_the_golden_fixture() {
    with_anthropic_cassette(
        "streaming_tools/streaming_tool_concurrency_emits_results_as_completed_but_persists_call_order",
        |client| async move {
            let log = two_tools(client, rig::serve::ServingPolicy::default(), 2, true).await;
            crate::ecs_goldens::golden_effects(
                "anthropic_serving_concurrent_concurrency_two_events",
                &log,
            );
        },
    )
    .await;
}

/// Every buffer at one: the park points are exercised, the trace is the
/// same.
#[tokio::test]
async fn capacity_one_effect_log_is_the_golden_fixture() {
    with_anthropic_cassette(
        "streaming_tools/streaming_tool_concurrency_emits_results_as_completed_but_persists_call_order",
        |client| async move {
            let bus = rig::serve::ServingPolicy {
                command_capacity: 1,
                stream_capacity: 1,
                serial_per_handler: false,
            };
            let log = two_tools(client, bus, 2, false).await;
            crate::ecs_goldens::golden_effects("anthropic_serving_capacity_one", &log);
        },
    )
    .await;
}

/// Serial serving over memory and a tool: three keys, each served one
/// command at a time, in dispatch order.
#[tokio::test]
async fn serial_memory_tools_effect_log_is_the_golden_fixture() {
    with_anthropic_cassette("corpus_hooks/observe_everything", |client| async move {
        let mut ecs = EcsAgent::for_golden_with_setup(
            client.completion_model(CLAUDE_SONNET_4_6),
            TOOLS_PREAMBLE,
            false,
            |world| {
                Handlers::with(world, |handlers| {
                    handlers.register(
                        crate::goldens::MEMORY_KEY,
                        RuntimeHandler {
                            inner: Arc::new(MemoryAdapter::new(
                                rig::memory::InMemoryConversationMemory::new(),
                            )),
                            runtime: tokio::runtime::Handle::current(),
                        },
                    )
                })
                .expect("bus installed")
                .expect("fresh memory");
            },
        );
        let memory = ecs
            .app
            .world_mut()
            .query::<(Entity, &Bound)>()
            .iter(ecs.app.world())
            .find(|(_, bound)| bound.key.as_str() == crate::goldens::MEMORY_KEY)
            .expect("memory handler")
            .0;
        ecs.app.world_mut().entity_mut(ecs.agent).insert((
            Temperature(Some(0.0)),
            Remembers(memory),
            Conversation(crate::goldens::CONVERSATION.into()),
        ));
        ecs.app
            .world_mut()
            .resource_mut::<Policy>()
            .0
            .serial_per_handler = true;
        ecs.tool(Adder);
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
        assert_eq!(log.header.bus.map(|bus| bus.serial_per_handler), Some(true));
        crate::ecs_goldens::golden_effects("anthropic_serving_serial_memory_tools", &log);
    })
    .await;
}

/// A second model registered as the route `fast` and selected by the hook
/// on every turn after the first: the tool-call turn goes to the default
/// model, the answer to the route, and the header's required row names
/// both.
#[tokio::test]
async fn model_route_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_serving_cassette("corpus_serving/model_route", |client| async move {
        let mut ecs = routed_agent(&client, true);
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
        assert_eq!(log.records[0].key.as_str(), "golden/model:default");
        assert_eq!(log.records[2].key.as_str(), "golden/model:fast");
        assert_eq!(
            log.header
                .required
                .get(&HandlerKey::from("golden/model:fast")),
            Some(&EffectFamily::Completion),
            "the route is in the required row"
        );
        crate::ecs_goldens::golden_effects("anthropic_serving_model_route", &log);
    })
    .await;
}

/// The route registered and never selected: the required row still names
/// it, the record never dispatches to it, and the replay must advertise it
/// from the row alone.
#[tokio::test]
async fn model_route_unselected_effect_log_is_the_golden_fixture() {
    with_anthropic_cassette("effect_corpus/tool_call_turn", |client| async move {
        let mut ecs = routed_agent(&client, false);
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
        assert!(
            log.records
                .iter()
                .all(|record| record.key.as_str() != "golden/model:fast"),
            "the route was never selected"
        );
        assert_eq!(
            log.header
                .required
                .get(&HandlerKey::from("golden/model:fast")),
            Some(&EffectFamily::Completion),
            "the route is in the required row"
        );
        crate::ecs_goldens::golden_effects("anthropic_serving_model_route_unselected", &log);
    })
    .await;
}

/// The same tool-call program over a host's bus: the host registers the
/// model under the agent's key, drives the bus and records; the agent
/// stamps the log, whose header names no bus policy (the host's).
async fn over_host_bus(
    client: rig::providers::anthropic::Client,
    streamed: bool,
) -> rig::effect_log::EffectLog {
    let mut ecs = EcsAgent::for_golden(
        client.completion_model(CLAUDE_SONNET_4_6),
        TOOLS_PREAMBLE,
        streamed,
    );
    ecs.app
        .world_mut()
        .entity_mut(ecs.agent)
        .insert(Temperature(Some(0.0)));
    ecs.declare_bus_policy = false;
    ecs.tool(Adder);
    assert!(!ecs.declare_bus_policy, "the policy is the host's");
    let output = ecs
        .prompt_with_max_turns(ADD_PROMPT, streamed, Some(3))
        .await;
    assert!(output.contains("42"), "{output}");
    let log = ecs.effect_log();
    let world = ecs.app.world_mut();
    let mut pending = world.query_filtered::<Option<&EffectOutcome>, With<PendingEffect>>();
    assert!(
        pending.iter(world).all(|outcome| outcome.is_some()),
        "host work complete before teardown"
    );
    drop(ecs);
    assert_eq!(
        families(&log),
        [
            EffectFamily::Completion,
            EffectFamily::Tool,
            EffectFamily::Completion
        ]
    );
    assert_eq!(log.header.bus, None);
    assert!(log.header.run_spec.is_some(), "the agent stamped its spec");
    assert_eq!(log.header.required.len(), 2, "{:?}", log.header.required);
    log
}

#[tokio::test]
async fn host_bus_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_serving_cassette("corpus_serving/host_bus", |client| async move {
        let log = over_host_bus(client, false).await;
        crate::ecs_goldens::golden_effects("anthropic_serving_host_bus", &log);
    })
    .await;
}

#[tokio::test]
async fn host_bus_streamed_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_serving_cassette(
        "corpus_serving/host_bus_streamed",
        |client| async move {
            let log = over_host_bus(client, true).await;
            crate::ecs_goldens::golden_effects("anthropic_serving_host_bus_streamed", &log);
        },
    )
    .await;
}
