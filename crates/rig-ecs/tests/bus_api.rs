//! A consumer of the supported bus boundary, without worker-storage imports.

#![allow(clippy::expect_used, reason = "test assertions")]

use bevy_ecs::{prelude::*, system::RunSystemOnce};
use rig_core::effect::{EffectKind, FamilyDescriptor, Outcome};
use rig_ecs::bus::{
    Bus, BusSet, EffectOutcome, Handlers, PendingEffect, Progress, RigSchedule, WorldOutcome,
    run_to_quiescence,
};

#[derive(Resource, Default)]
struct Observed(Vec<Entity>);

type Unanswered = (
    With<PendingEffect>,
    Without<WorldOutcome>,
    Without<EffectOutcome>,
);

fn gate(mut commands: Commands, effects: Query<Entity, Unanswered>) {
    // Open world handlers can be answered by arbitrary application systems.
    for entity in &effects {
        commands
            .entity(entity)
            .insert(WorldOutcome::new(Ok(Outcome::Custom {
                payload: serde_json::json!("answer"),
            })));
    }
}

fn judge(
    outcomes: Query<(Entity, &EffectOutcome), Added<EffectOutcome>>,
    mut observed: ResMut<Observed>,
    mut progress: ResMut<Progress>,
) {
    for (entity, outcome) in &outcomes {
        assert!(outcome.0.is_ok());
        observed.0.push(entity);
        progress.mark();
    }
}

#[test]
fn host_systems_register_answer_and_observe_through_public_components_and_sets() {
    let mut world = World::new();
    Bus::default().install(&mut world);
    world.init_resource::<Observed>();
    world
        .run_system_once(|mut handlers: Handlers| {
            handlers
                .register_open(
                    "host",
                    FamilyDescriptor::Custom {
                        kind: "question".into(),
                    },
                )
                .expect("register");
        })
        .expect("system");
    world
        .resource_mut::<Schedules>()
        .get_mut(RigSchedule)
        .expect("schedule")
        .add_systems((gate.in_set(BusSet::Gate), judge.in_set(BusSet::Judge)));
    let effect = world
        .spawn(PendingEffect::new(
            "host",
            EffectKind::Custom {
                kind: "question".into(),
                payload: serde_json::Value::Null,
            },
        ))
        .id();
    run_to_quiescence(&mut world);
    assert_eq!(world.resource::<Observed>().0, [effect]);
}

#[test]
fn a_mutating_gate_can_name_its_subject_without_conflicting_query_access() {
    #[derive(Resource, Default)]
    struct SubjectsSeen(Vec<rig_core::observe::Subject>);
    let mut world = World::new();
    Bus::default().install(&mut world);
    world.init_resource::<SubjectsSeen>();
    let effect = world
        .spawn((
            PendingEffect::new(
                "host",
                EffectKind::Custom {
                    kind: "question".into(),
                    payload: serde_json::Value::Null,
                },
            ),
            rig_ecs::bus::Scope("host/run".into()),
        ))
        .id();
    world
        .run_system_once(
            |mut effects: Query<(Entity, &mut PendingEffect)>,
             walk: rig_ecs::bus::SubjectWalk,
             mut seen: ResMut<SubjectsSeen>| {
                for (entity, mut pending) in &mut effects {
                    pending.kind = EffectKind::Custom {
                        kind: "patched".into(),
                        payload: serde_json::json!("updated"),
                    };
                    seen.0
                        .push(walk.of_intent(entity, &pending.key, pending.kind.family()));
                }
            },
        )
        .expect("mutable intent access composes with the identity walk");
    let seen = &world.resource::<SubjectsSeen>().0;
    assert_eq!(seen.len(), 1);
    assert_eq!(
        seen.first().and_then(|subject| subject.scope.as_deref()),
        Some("host/run")
    );
    assert!(seen.first().is_some_and(|subject| subject.order.is_some()));
    assert!(
        matches!(&world.get::<PendingEffect>(effect).expect("effect").kind, EffectKind::Custom { kind, .. } if &**kind == "patched")
    );
}

#[allow(clippy::panic, reason = "shared fixture assertions")]
mod bus_support;

#[test]
fn worker_status_distinguishes_running_work_from_uncollected_delivery() {
    use rig_ecs::bus::{ExecutionStatus, InFlight, Streamed, Streaming, execution_status};
    use std::{sync::Arc, time::Instant};

    fn wait(mut ready: impl FnMut() -> bool) {
        let deadline = Instant::now() + bus_support::GUARD;
        while !ready() {
            assert!(
                Instant::now() < deadline,
                "worker did not reach expected state"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    let idle = ExecutionStatus {
        preparing: false,
        streaming: false,
    };
    let mut world = World::new();
    let entity = world.spawn_empty().id();
    assert_eq!(execution_status(&world, entity), None);
    Bus::default().install(&mut world);
    assert_eq!(execution_status(&world, entity), Some(idle));
    world.despawn(entity);
    assert_eq!(execution_status(&world, entity), None);

    let counters = Arc::new(bus_support::Counters::default());
    counters.hold.hold();
    let mut model = bus_support::MockModel::new(&counters);
    model.cap = 1;
    Handlers::register_in(&mut world, "model", model).expect("register model");
    let unary = world
        .spawn(PendingEffect::new("model", bus_support::completion()))
        .id();
    run_to_quiescence(&mut world);
    assert_eq!(
        execution_status(&world, unary),
        Some(ExecutionStatus {
            preparing: true,
            streaming: false,
        })
    );
    counters.hold.release();
    wait(|| execution_status(&world, unary) == Some(idle));
    assert!(
        world.get::<EffectOutcome>(unary).is_none(),
        "finished worker still awaits Collect"
    );
    run_to_quiescence(&mut world);
    assert!(world.get::<EffectOutcome>(unary).is_some());

    counters.hold.hold();
    let stream = world
        .spawn(PendingEffect::new("model", bus_support::streaming()))
        .id();
    wait(|| {
        run_to_quiescence(&mut world);
        world.get::<Streaming>(stream).is_some()
    });
    assert_eq!(
        execution_status(&world, stream),
        Some(ExecutionStatus {
            preparing: false,
            streaming: true,
        })
    );
    counters.hold.release();
    wait(|| execution_status(&world, stream) == Some(idle));
    assert!(world.get::<InFlight>(stream).is_some());
    assert!(
        world.get::<EffectOutcome>(stream).is_none(),
        "queued stream output is not yet delivered"
    );
    run_to_quiescence(&mut world);
    assert_eq!(
        world
            .get::<Streamed>(stream)
            .expect("collected stream")
            .text,
        "tick "
    );
    assert!(world.get::<EffectOutcome>(stream).is_some());
    world.despawn(stream);
    assert_eq!(execution_status(&world, stream), None);
}
