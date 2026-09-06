//! A delivered invalid name must be actionable before the producer finishes.
//! The producer gate prevents EOF from masquerading as a midstream boundary.
// Test assertions and the shared run_support fixtures intentionally panic.
#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

#[path = "run_stream_boundary/errors.rs"]
mod errors;
#[path = "run_stream_boundary/multiple.rs"]
mod multiple;
mod run_support;

use futures::channel::oneshot;
use rig_core::{
    completion::{ModelRef, ProviderCapabilities},
    effect::{EffectKind, FamilyDescriptor, HandlerDescriptor, HandlerKey},
    serve::{OutcomeSink, Serve},
    streaming::{BlockId, BlockKind, Delta, StreamEvent},
};
use rig_ecs::{
    agent::{Failed, Failure, InvalidCall},
    bus::{EffectOutcome, Streamed},
    systems::spawn_run,
};
use run_support::*;
use std::sync::{Arc, Mutex};

use bevy_ecs::prelude::*;
use rig_core::{
    completion::{CompletionResponse, Usage as ProviderUsage},
    effect::Outcome,
    message::{AssistantContent, ToolCallId},
    streaming::{BlockClose, StreamFinal, ToolCallEnd},
};
use rig_ecs::{
    agent::{Grant, MessageParts, Order, Parts, Resolution, RunResult, Settled, Usage},
    bus::RigSchedule,
    systems::RigSet,
};

// A controlled producer is necessary here: an HTTP cassette alone cannot
// guarantee that the policy runs before EOF reaches the effect collector.
struct FinishingName(Arc<Mutex<Option<oneshot::Receiver<()>>>>);

impl Serve for FinishingName {
    type Family = rig_core::effect::family::Completion;
    fn descriptor(&self) -> HandlerDescriptor {
        NameThenGate(Arc::clone(&self.0)).descriptor()
    }
    async fn serve(&self, _kind: EffectKind, mut sink: OutcomeSink) {
        let gate = self.0.lock().expect("gate lock").take();
        let Some(gate) = gate else {
            sink.resolve(Ok(Outcome::Completion(CompletionResponse::new(
                vec![AssistantContent::text("done")],
                ProviderUsage {
                    total_tokens: 3,
                    ..ProviderUsage::new()
                },
                "boundary",
            ))))
            .await;
            return;
        };
        let id = BlockId::Wire("assembly-block".into());
        for event in [
            StreamEvent::BlockStart {
                id: id.clone(),
                kind: BlockKind::ToolCall,
            },
            StreamEvent::BlockDelta {
                id: id.clone(),
                delta: Delta::ToolName {
                    name: "wrong".into(),
                },
            },
        ] {
            sink.send(Ok(event)).await.expect("stream open");
        }
        gate.await.expect("test releases producer");
        sink.send(Ok(StreamEvent::BlockEnd {
            id,
            end: BlockClose::ToolCall(
                ToolCallEnd::whole("wrong", serde_json::json!({"x": 2, "y": 3}))
                    .with_tool_id("provider-tool")
                    .with_call_id("provider-call"),
            ),
            block: None,
        }))
        .await
        .expect("stream open");
        sink.send(Ok(StreamEvent::Final(StreamFinal::new(
            "boundary",
            ProviderUsage {
                total_tokens: 7,
                ..ProviderUsage::new()
            },
        ))))
        .await
        .expect("stream open");
    }
}

#[derive(Resource, Default)]
struct RepairCount(usize);

#[derive(Resource)]
struct Decision(Resolution);

fn decide_name(
    mut commands: Commands,
    invalid: Query<Entity, Added<InvalidCall>>,
    decision: Res<Decision>,
    mut count: ResMut<RepairCount>,
) {
    for entity in &invalid {
        count.0 += 1;
        commands.entity(entity).insert(decision.0.clone());
    }
}

#[test]
fn exhausted_early_retry_fails_without_waiting_for_eof() {
    let mut app = app();
    app.init_resource::<RepairCount>();
    app.insert_resource(Decision(Resolution::Retry {
        feedback: "try again".into(),
    }));
    app.world_mut()
        .resource_mut::<Schedules>()
        .add_systems(RigSchedule, decide_name.in_set(RigSet::Judge));
    let (_release, gate) = oneshot::channel();
    let model = register(
        &mut app,
        "boundary/model",
        NameThenGate(Arc::new(Mutex::new(Some(gate)))),
    );
    let agent = spawn_agent(app.world_mut(), "boundary", model);
    let run = spawn_run(app.world_mut(), agent, &[], "add", true, Some(2));
    tick_until(&mut app, "exhausted retry fails before EOF", |world| {
        world.get::<Failed>(run).is_some()
    });
    assert!(
        matches!(app.world().get::<Failed>(run), Some(Failed(Failure::UnknownToolCall { name })) if name == "unavailable_tool")
    );
    assert_eq!(app.world().resource::<RepairCount>().0, 1);
    assert_eq!(
        app.world_mut()
            .query::<&EffectOutcome>()
            .iter(app.world())
            .count(),
        0
    );
}

#[test]
fn early_skip_retains_prefix_and_drained_usage_without_dispatching_tool() {
    let mut app = app();
    app.init_resource::<RepairCount>();
    app.insert_resource(Decision(Resolution::Skip {
        reason: "not this tool".into(),
    }));
    app.world_mut()
        .resource_mut::<Schedules>()
        .add_systems(RigSchedule, decide_name.in_set(RigSet::Judge));
    let (release, gate) = oneshot::channel();
    let model = register(
        &mut app,
        "boundary/model",
        FinishingName(Arc::new(Mutex::new(Some(gate)))),
    );
    let adder = Adder::new("boundary/add");
    let peak = Arc::clone(&adder.peak);
    let tool = register(&mut app, "boundary/add", adder);
    let agent = spawn_agent(app.world_mut(), "boundary", model);
    app.world_mut()
        .spawn((Grant(tool), Order(0), ChildOf(agent)));
    let run = spawn_run(app.world_mut(), agent, &[], "add", true, Some(2));
    tick_until(&mut app, "skip before EOF", |world| {
        world.resource::<RepairCount>().0 == 1
    });
    assert_eq!(
        app.world_mut()
            .query::<&EffectOutcome>()
            .iter(app.world())
            .count(),
        0
    );
    release.send(()).expect("producer alive");
    tick_until(&mut app, "skip recovers", |world| {
        world.get::<Settled>(run).is_some() || world.get::<Failed>(run).is_some()
    });
    assert!(
        app.world().get::<Failed>(run).is_none(),
        "{:?}",
        app.world().get::<Failed>(run)
    );
    assert_eq!(app.world().get::<RunResult>(run).expect("result").0, "done");
    assert_eq!(
        app.world().get::<Usage>(run).expect("usage").0.total_tokens,
        10
    );
    assert_eq!(peak.load(std::sync::atomic::Ordering::SeqCst), 0);
    assert_eq!(app.world().resource::<RepairCount>().0, 1);
    let calls: Vec<_> = app
        .world_mut()
        .query::<(&ChildOf, &Parts)>()
        .iter(app.world())
        .filter(|(parent, _)| parent.parent() == run)
        .flat_map(|(_, parts)| match &parts.0 {
            MessageParts::Assistant { content, .. } => content
                .iter()
                .filter_map(|part| match part {
                    AssistantContent::ToolCall(call) => Some(call.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            _ => vec![],
        })
        .collect();
    assert_eq!(calls.len(), 1);
    assert_eq!(
        calls[0].id,
        ToolCallId::from_block(&BlockId::Wire("assembly-block".into()))
    );
    assert!(calls[0].provider.is_none());
    assert_eq!(calls[0].function.name, "wrong");
    assert_eq!(
        calls[0].function.arguments,
        serde_json::Value::Null,
        "future arguments must not enter the retained prefix"
    );
}

fn repair_name(
    mut commands: Commands,
    invalid: Query<(Entity, &InvalidCall), Added<InvalidCall>>,
    mut count: ResMut<RepairCount>,
) {
    for (entity, call) in &invalid {
        assert_eq!(call.name, "wrong");
        assert_eq!(
            call.id,
            ToolCallId::from_block(&BlockId::Wire("assembly-block".into()))
        );
        assert_eq!(call.prefix.len(), 1);
        count.0 += 1;
        commands
            .entity(entity)
            .insert(Resolution::Repair { to: "add".into() });
    }
}

#[test]
fn early_repair_survives_raw_block_completion_and_provider_identity() {
    let mut app = app();
    app.init_resource::<RepairCount>();
    app.world_mut()
        .resource_mut::<Schedules>()
        .add_systems(RigSchedule, repair_name.in_set(RigSet::Judge));
    let (release, gate) = oneshot::channel();
    let model = register(
        &mut app,
        "boundary/model",
        FinishingName(Arc::new(Mutex::new(Some(gate)))),
    );
    let adder = Adder::new("boundary/add");
    let peak = Arc::clone(&adder.peak);
    let tool = register(&mut app, "boundary/add", adder);
    let agent = spawn_agent(app.world_mut(), "boundary", model);
    app.world_mut()
        .spawn((Grant(tool), Order(0), ChildOf(agent)));
    let run = spawn_run(app.world_mut(), agent, &[], "add", true, Some(2));
    tick_until(&mut app, "repair before EOF", |world| {
        world.resource::<RepairCount>().0 == 1
    });
    assert_eq!(
        app.world_mut()
            .query::<&EffectOutcome>()
            .iter(app.world())
            .count(),
        0
    );
    assert_eq!(peak.load(std::sync::atomic::Ordering::SeqCst), 0);
    // A subsequent run policy must not reinterpret this turn's repair.
    app.world_mut()
        .entity_mut(run)
        .insert(rig_ecs::agent::ToolAccess {
            allowed: Some(Default::default()),
            ..Default::default()
        });
    release.send(()).expect("producer alive");
    tick_until(&mut app, "repaired run settles", |world| {
        world.get::<Settled>(run).is_some() || world.get::<Failed>(run).is_some()
    });
    assert!(
        app.world().get::<Failed>(run).is_none(),
        "{:?}",
        app.world().get::<Failed>(run)
    );
    assert_eq!(app.world().get::<RunResult>(run).expect("result").0, "done");
    assert_eq!(app.world().resource::<RepairCount>().0, 1);
    assert_eq!(peak.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(
        app.world().get::<Usage>(run).expect("usage").0.total_tokens,
        10
    );
    let calls: Vec<_> = app
        .world_mut()
        .query::<(&ChildOf, &Parts)>()
        .iter(app.world())
        .filter(|(parent, _)| parent.parent() == run)
        .flat_map(|(_, parts)| match &parts.0 {
            MessageParts::Assistant { content, .. } => content
                .iter()
                .filter_map(|part| match part {
                    AssistantContent::ToolCall(call) => Some(call.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            _ => vec![],
        })
        .collect();
    assert_eq!(calls.len(), 1);
    assert_eq!(
        calls[0].id,
        ToolCallId::new("provider-call").expect("valid id")
    );
    assert_eq!(calls[0].function.name, "add");
    assert_eq!(
        calls[0].function.arguments,
        serde_json::json!({"x": 2, "y": 3})
    );
}

struct NameThenGate(Arc<Mutex<Option<oneshot::Receiver<()>>>>);

impl Serve for NameThenGate {
    type Family = rig_core::effect::family::Completion;
    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from("boundary/model"),
            family: FamilyDescriptor::Completion {
                model: ModelRef::new("boundary-model"),
                capabilities: ProviderCapabilities::default(),
            },
            layers: vec![],
        }
    }
    async fn serve(&self, _kind: EffectKind, mut sink: OutcomeSink) {
        let id = BlockId::Wire("invalid-call".into());
        sink.send(Ok(StreamEvent::BlockStart {
            id: id.clone(),
            kind: BlockKind::ToolCall,
        }))
        .await
        .expect("stream open");
        sink.send(Ok(StreamEvent::BlockDelta {
            id,
            delta: Delta::ToolName {
                name: "unavailable_tool".into(),
            },
        }))
        .await
        .expect("stream open");
        let gate = self
            .0
            .lock()
            .expect("gate lock")
            .take()
            .expect("one request");
        let _ = gate.await;
    }
}

#[test]
fn invalid_tool_name_is_actionable_before_completion_outcome() {
    let mut app = app();
    let (_release, gate) = oneshot::channel();
    let model = register(
        &mut app,
        "boundary/model",
        NameThenGate(Arc::new(Mutex::new(Some(gate)))),
    );
    let agent = spawn_agent(app.world_mut(), "boundary", model);
    let run = spawn_run(
        app.world_mut(),
        agent,
        &[],
        "call an unavailable tool",
        true,
        Some(1),
    );
    tick_until(
        &mut app,
        "tool name delivered while producer is gated",
        |world| {
            world.query::<&Streamed>().iter(world).any(|stream| stream.events.iter().any(|event| matches!(event, StreamEvent::BlockDelta { delta: Delta::ToolName { name }, .. } if name == "unavailable_tool")))
        },
    );
    // Give the native policy a complete subsequent schedule pass, while the
    // producer still cannot finish. No sleep or elapsed-time inference.
    app.update();
    assert_eq!(
        app.world_mut()
            .query::<&EffectOutcome>()
            .iter(app.world())
            .count(),
        0,
        "the producer has not completed"
    );
    let actionable = app
        .world_mut()
        .query::<&InvalidCall>()
        .iter(app.world())
        .any(|call| call.name == "unavailable_tool")
        || matches!(app.world().get::<Failed>(run), Some(Failed(Failure::UnknownToolCall { name })) if name == "unavailable_tool");
    assert!(
        actionable,
        "delivered invalid tool name must reach native invalid-call policy before completion outcome"
    );
}
