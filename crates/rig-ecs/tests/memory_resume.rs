//! Snapshot cuts around memory finalization, using a live scripted store.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

mod run_support;

use std::sync::{Arc, Mutex};

use bevy_ecs::prelude::*;
use rig_core::{
    effect::{
        EffectId, EffectKind, FamilyDescriptor, HandlerDescriptor, HandlerKey, MemoryOp,
        MemoryOutcome, Outcome,
    },
    serve::{OutcomeSink, Serve},
};
use rig_ecs::{
    agent::{
        Conversation, Remembers, Run, Settled,
        scene::{load_world, save_world},
    },
    bus::{EffectOutcome, Held, Issued, PendingEffect, RigSchedule},
    systems::{RigSet, spawn_run},
};
use run_support::*;

const MODEL: &str = "t/model:default";
const MEMORY: &str = "t/memory";

struct Store {
    calls: Arc<Mutex<Vec<EffectId>>>,
    hold_after_write: bool,
}

impl Serve for Store {
    type Family = rig_core::effect::family::Memory;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from(MEMORY),
            family: FamilyDescriptor::Memory {},
            layers: vec![],
        }
    }

    async fn serve(&self, kind: EffectKind, sink: OutcomeSink) {
        let answer = match kind {
            EffectKind::Memory {
                op: MemoryOp::Load { .. },
            } => MemoryOutcome::Loaded { messages: vec![] },
            EffectKind::Memory {
                op: MemoryOp::Append { .. },
            } => {
                // The external write happens before an outcome is observable.
                self.calls.lock().unwrap().push(sink.id());
                if self.hold_after_write {
                    futures::future::pending::<()>().await;
                }
                MemoryOutcome::Appended
            }
            other => panic!("unexpected operation: {other:?}"),
        };
        sink.resolve(Ok(Outcome::Memory(answer))).await;
    }
}

fn setup(calls: Arc<Mutex<Vec<EffectId>>>, hold: bool) -> (bevy_app::App, Entity, Entity) {
    let mut app = app();
    let (model, _) = Capturing::new(MODEL, "ok");
    let model = register(&mut app, MODEL, model);
    let memory = register(
        &mut app,
        MEMORY,
        Store {
            calls,
            hold_after_write: hold,
        },
    );
    (app, model, memory)
}

fn start(app: &mut bevy_app::App, model: Entity, memory: Entity) -> Entity {
    let agent = spawn_agent(app.world_mut(), "t", model);
    app.world_mut()
        .entity_mut(agent)
        .insert((Remembers(memory), Conversation("conversation".into())));
    spawn_run(app.world_mut(), agent, &[], "go", false, None)
}

fn appends(world: &mut World) -> Vec<Entity> {
    world
        .query::<(Entity, &PendingEffect)>()
        .iter(world)
        .filter(|(_, effect)| {
            matches!(
                effect.kind,
                EffectKind::Memory {
                    op: MemoryOp::Append { .. }
                }
            )
        })
        .map(|(entity, _)| entity)
        .collect()
}

fn appended(world: &mut World) -> bool {
    appends(world)
        .into_iter()
        .any(|entity| world.get::<EffectOutcome>(entity).is_some())
}

#[derive(Resource)]
struct BeforeAppend(rig_ecs::agent::scene::WorldScene);

#[test]
fn settled_snapshot_before_finalization_schedules_one_append() {
    let calls = Arc::new(Mutex::new(vec![]));
    let (mut first, model, memory) = setup(calls.clone(), false);
    first.add_systems(
        RigSchedule,
        (|world: &mut World| {
            if !world.contains_resource::<BeforeAppend>()
                && world
                    .query_filtered::<Entity, With<Settled>>()
                    .iter(world)
                    .next()
                    .is_some()
            {
                assert!(appends(world).is_empty());
                let scene = save_world(world).unwrap();
                world.insert_resource(BeforeAppend(scene));
            }
        })
        .after(RigSet::Materialise)
        .before(RigSet::Settle),
    );
    start(&mut first, model, memory);
    tick_until(&mut first, "snapshot before finalization", |w| {
        w.contains_resource::<BeforeAppend>()
    });
    let scene = first
        .world_mut()
        .remove_resource::<BeforeAppend>()
        .unwrap()
        .0;
    drop(first);
    // A separate external store: this asserts what the saved branch does,
    // not that abandoning another live branch undoes its writes.
    let calls = Arc::new(Mutex::new(vec![]));
    let (mut restored, _, _) = setup(calls.clone(), false);
    load_world(&scene, restored.world_mut()).unwrap();
    tick_until(&mut restored, "new finalization completed", appended);
    assert_eq!(appends(restored.world_mut()).len(), 1);
    assert_eq!(calls.lock().unwrap().len(), 1);
}

#[test]
fn completed_append_is_not_scheduled_again_after_load() {
    let calls = Arc::new(Mutex::new(vec![]));
    let (mut first, model, memory) = setup(calls.clone(), false);
    start(&mut first, model, memory);
    tick_until(&mut first, "append completed", appended);
    let scene = save_world(first.world_mut()).unwrap();
    drop(first);
    let (mut restored, _, _) = setup(calls.clone(), false);
    load_world(&scene, restored.world_mut()).unwrap();
    for _ in 0..10 {
        restored.update();
    }
    assert_eq!(
        appends(restored.world_mut()).len(),
        1,
        "restoration scheduled a second operation"
    );
    assert_eq!(calls.lock().unwrap().len(), 1);
}

#[test]
fn queued_append_survives_without_a_second_operation() {
    let calls = Arc::new(Mutex::new(vec![]));
    let (mut first, model, memory) = setup(calls.clone(), false);
    first.add_systems(
        RigSchedule,
        (|mut commands: Commands, effects: Query<(Entity, &PendingEffect), Without<Issued>>| {
            for (entity, effect) in &effects {
                if matches!(
                    effect.kind,
                    EffectKind::Memory {
                        op: MemoryOp::Append { .. }
                    }
                ) {
                    commands.entity(entity).insert(Held);
                }
            }
        })
        .after(RigSet::Settle),
    );
    let run = start(&mut first, model, memory);
    tick_until(&mut first, "append queued", |w| {
        w.get::<Settled>(run).is_some() && !appends(w).is_empty()
    });
    assert!(calls.lock().unwrap().is_empty());
    let scene = save_world(first.world_mut()).unwrap();
    drop(first);
    let (mut restored, _, _) = setup(calls.clone(), false);
    let loaded = load_world(&scene, restored.world_mut()).unwrap();
    for effect in loaded.effects {
        restored.world_mut().entity_mut(effect).remove::<Held>();
    }
    tick_until(&mut restored, "restored append completed", appended);
    assert_eq!(appends(restored.world_mut()).len(), 1);
    assert_eq!(calls.lock().unwrap().len(), 1);
}

#[test]
fn unresolved_external_write_reissues_the_same_operation_identity() {
    let calls = Arc::new(Mutex::new(vec![]));
    let (mut first, model, memory) = setup(calls.clone(), true);
    start(&mut first, model, memory);
    tick_until(&mut first, "external write happened", |_| {
        !calls.lock().unwrap().is_empty()
    });
    assert!(!appended(first.world_mut()));
    let scene = save_world(first.world_mut()).unwrap();
    drop(first);
    let (mut restored, _, _) = setup(calls.clone(), false);
    let loaded = load_world(&scene, restored.world_mut()).unwrap();
    assert!(
        loaded
            .graph
            .iter()
            .any(|entity| restored.world().get::<Run>(*entity).is_some())
    );
    tick_until(&mut restored, "retried append completed", appended);
    assert_eq!(appends(restored.world_mut()).len(), 1);
    let calls = calls.lock().unwrap();
    assert_eq!(
        calls.len(),
        2,
        "a scene cannot prove whether an unanswered external write happened"
    );
    assert_eq!(
        calls.first(),
        calls.last(),
        "the host can deduplicate by the saved operation id within this log"
    );
}
