//! A delivered invalid name must be actionable before the producer finishes.
//! The producer gate prevents EOF from masquerading as a midstream boundary.
// Test assertions and the shared run_support fixtures intentionally panic.
use crate::run_support;

use futures::channel::oneshot;
use rig_core::{
    completion::{ModelRef, ProviderCapabilities},
    effect::{EffectKind, FamilyDescriptor, HandlerDescriptor, HandlerKey},
    serve::{Dispatch, Reply, Serve},
};
use rig_ecs::{
    agent::{Failed, Failure, InvalidCall},
    bus::EffectOutcome,
    systems::RunCommands,
};
use run_support::*;
use std::sync::{Arc, Mutex};

use bevy_ecs::prelude::*;
use rig_core::{
    completion::{CompletionResponse, Usage as ProviderUsage},
    effect::Outcome,
    message::AssistantContent,
};
use rig_ecs::{
    agent::{Grant, MessageParts, Resolution, RunResult, Settled, Usage},
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
    async fn serve(&self, _kind: EffectKind, _dispatch: Dispatch) -> Reply {
        let gate = self.0.lock().expect("gate lock").take();
        let Some(gate) = gate else {
            return Reply::Outcome(Ok(Outcome::Completion(CompletionResponse::new(
                vec![AssistantContent::text("done")],
                ProviderUsage {
                    total_tokens: Some(3),
                    ..ProviderUsage::default()
                },
                rig_core::message::Origin::new("test.api", "boundary", ""),
                serde_json::json!({}),
            ))));
        };

        Reply::written(
            rig_core::message::Origin::new("boundary", "boundary", "boundary"),
            move |mut writer| async move {
                writer
                    .tool_call("wrong", serde_json::json!({"x": 2, "y": 3}))
                    .await
                    .expect("stream open");
                gate.await.expect("test releases producer");
                writer
                    .finish(rig_core::operation::Finish {
                        usage: ProviderUsage {
                            total_tokens: Some(7),
                            ..ProviderUsage::default()
                        },
                        ..rig_core::operation::Finish::default()
                    })
                    .await
                    .expect("stream open");
            },
        )
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
    let run = app.world_mut().spawn_run(agent, &[], "add", true, Some(2));
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
    app.world_mut().spawn((Grant(tool), ChildOf(agent)));
    let run = app.world_mut().spawn_run(agent, &[], "add", true, Some(2));
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
        Some(10)
    );
    assert_eq!(peak.load(std::sync::atomic::Ordering::SeqCst), 0);
    assert_eq!(app.world().resource::<RepairCount>().0, 1);
    let calls: Vec<_> = app
        .world_mut()
        .query_filtered::<(&ChildOf, Entity), With<rig_ecs::agent::Utterance>>()
        .iter(app.world())
        .filter(|(parent, _)| parent.parent() == run)
        .flat_map(|(_, entity)| {
            match rig_ecs::agent::content::parts::read_message(app.world(), entity)
                .expect("valid history graph")
            {
                MessageParts::Assistant(rig_core::message::AssistantMessage {
                    content, ..
                }) => content
                    .iter()
                    .filter_map(|part| match part {
                        AssistantContent::ToolCall(call) => Some(call.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>(),
                _ => vec![],
            }
        })
        .collect();
    assert_eq!(calls.len(), 1);
    // The call surfaced when it ended, whole, under the id the writer issued.
    assert!(calls[0].id.is_local());
    assert_eq!(calls[0].function.name, "wrong");
    assert_eq!(
        calls[0].function.arguments_value(),
        serde_json::json!({"x": 2, "y": 3}),
        "the retained prefix holds the call as it ended"
    );
}

fn repair_name(
    mut commands: Commands,
    invalid: Query<(Entity, &InvalidCall), Added<InvalidCall>>,
    mut count: ResMut<RepairCount>,
) {
    for (entity, call) in &invalid {
        assert_eq!(call.name, "wrong");
        assert!(call.id.is_local());
        assert_eq!(call.prefix.len(), 1);
        count.0 += 1;
        commands
            .entity(entity)
            .insert(Resolution::Repair { to: "add".into() });
    }
}

#[test]
fn early_repair_survives_the_calls_completion() {
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
    app.world_mut().spawn((Grant(tool), ChildOf(agent)));
    let run = app.world_mut().spawn_run(agent, &[], "add", true, Some(2));
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
        Some(10)
    );
    let calls: Vec<_> = app
        .world_mut()
        .query_filtered::<(&ChildOf, Entity), With<rig_ecs::agent::Utterance>>()
        .iter(app.world())
        .filter(|(parent, _)| parent.parent() == run)
        .flat_map(|(_, entity)| {
            match rig_ecs::agent::content::parts::read_message(app.world(), entity)
                .expect("valid history graph")
            {
                MessageParts::Assistant(rig_core::message::AssistantMessage {
                    content, ..
                }) => content
                    .iter()
                    .filter_map(|part| match part {
                        AssistantContent::ToolCall(call) => Some(call.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>(),
                _ => vec![],
            }
        })
        .collect();
    assert_eq!(calls.len(), 1);
    assert!(calls[0].id.is_local(), "the writer issues the call's id");
    assert_eq!(calls[0].function.name, "add");
    assert_eq!(
        calls[0].function.arguments_value(),
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
    async fn serve(&self, _kind: EffectKind, _dispatch: Dispatch) -> Reply {
        let gate = self
            .0
            .lock()
            .expect("gate lock")
            .take()
            .expect("one request");

        Reply::written(
            rig_core::message::Origin::new("writer", "writer", "writer"),
            move |mut writer| async move {
                writer
                    .tool_call("unavailable_tool", serde_json::json!({}))
                    .await
                    .expect("stream open");
                let _ = gate.await;
            },
        )
    }
}
