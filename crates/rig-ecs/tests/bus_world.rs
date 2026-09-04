//! Handlers that are systems, interception in the world, causality as
//! `ChildOf`: the fixture's proofs 8, 11 and 12 in the native shape, and
//! the whole-tree cancel.
//!
//! | proof | test |
//! |---|---|
//! | 8 a system answers; the serial key stays busy until it does | `a_system_answers_an_asked_effect_and_the_key_waits_for_it` |
//! | 11 a decision suspended, then made from a system next tick; a denial is no record | `a_held_effect_is_denied_or_approved_from_a_system_next_tick` |
//! | 12 nested dispatch: `ChildOf`, `parent` in the record, same-key nesting refused, despawn ⇒ `Cancelled` | `a_child_effect_records_its_parent_and_a_reentrant_one_is_refused`, `despawning_a_parent_cancels_its_children_in_flight_and_never_serves_the_queued` |
//! | — patch in `Gate`; replace in `Judge`; the record keeps the handler's answer | `gate_patches_and_judge_replaces_but_the_record_keeps_the_answer` |

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::type_complexity
)]

mod bus_support;

use std::sync::{Arc, atomic::Ordering};

use bevy_ecs::prelude::*;
use bus_support::*;
use rig_core::{
    completion::Message,
    effect::{CustomEffect, EffectId, EffectKind, HandlerKey, Outcome},
    error::{ErrorKind, ErrorReport},
    serve::Decision,
};
use rig_ecs::bus::{
    Answer, Asked, BusSet, EffectLogResource, EffectOutcome, Handlers, Held, InFlight, Issued,
    PendingEffect, RigSchedule,
};
use rig_effect_log::EffectLogRecorder;

/// A question the world answers.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct Ask {
    question: String,
}

impl CustomEffect for Ask {
    const KIND: &'static str = "ask";
    type Answer = String;
}

/// Answers every `Ask` with the question echoed, from a system.
fn answer_asks(asked: Query<(Entity, &Asked<Ask>), Without<Answer<Ask>>>, mut commands: Commands) {
    for (entity, Asked(ask)) in &asked {
        commands
            .entity(entity)
            .insert(Answer::<Ask>(format!("you asked: {}", ask.question)));
    }
}

fn add_after_judge<M>(
    app: &mut bevy_app::App,
    system: impl IntoScheduleConfigs<bevy_ecs::system::ScheduleSystem, M>,
) {
    app.world_mut()
        .resource_mut::<bevy_ecs::schedule::Schedules>()
        .add_systems(RigSchedule, system.after(BusSet::Judge));
}

fn add_in_gate<M>(
    app: &mut bevy_app::App,
    system: impl IntoScheduleConfigs<bevy_ecs::system::ScheduleSystem, M>,
) {
    app.world_mut()
        .resource_mut::<bevy_ecs::schedule::Schedules>()
        .add_systems(RigSchedule, system.in_set(BusSet::Gate));
}

fn add_in_judge<M>(
    app: &mut bevy_app::App,
    system: impl IntoScheduleConfigs<bevy_ecs::system::ScheduleSystem, M>,
) {
    app.world_mut()
        .resource_mut::<bevy_ecs::schedule::Schedules>()
        .add_systems(RigSchedule, system.in_set(BusSet::Judge));
}

#[test]
fn a_system_answers_an_asked_effect_and_the_key_waits_for_it() {
    let mut app = serial_app();
    Handlers::with(app.world_mut(), |handlers| {
        handlers
            .register_world::<Ask>("world/ask")
            .expect("a fresh key")
    })
    .expect("a bus");
    let first = app
        .world_mut()
        .spawn(
            PendingEffect::custom(
                "world/ask",
                &Ask {
                    question: "one".to_owned(),
                },
            )
            .expect("serializes"),
        )
        .id();
    let second = app
        .world_mut()
        .spawn(
            PendingEffect::custom(
                "world/ask",
                &Ask {
                    question: "two".to_owned(),
                },
            )
            .expect("serializes"),
        )
        .id();
    // No answering system yet: the first is asked, the second waits on the
    // serial key.
    app.update();
    assert!(app.world().get::<Asked<Ask>>(first).is_some(), "asked");
    assert!(app.world().get::<InFlight>(first).is_some());
    assert!(
        app.world().get::<InFlight>(second).is_none(),
        "the key is busy until the system answers"
    );
    tick(&mut app, 3);
    assert!(app.world().get::<InFlight>(second).is_none());

    add_after_judge(&mut app, answer_asks);
    tick_until(&mut app, "both answered", |world| {
        world.get::<EffectOutcome>(first).is_some() && world.get::<EffectOutcome>(second).is_some()
    });
    let world = app.world();
    let first_answer = world
        .get::<EffectOutcome>(first)
        .expect("answered")
        .custom::<Ask>()
        .expect("a string");
    assert_eq!(first_answer, "you asked: one");
    let second_answer = world
        .get::<EffectOutcome>(second)
        .expect("answered")
        .custom::<Ask>()
        .expect("a string");
    assert_eq!(second_answer, "you asked: two");
    assert!(
        world.get::<Asked<Ask>>(first).is_none(),
        "the ask is consumed"
    );
    assert!(world.get::<InFlight>(first).is_none());
}

#[derive(Resource, Default)]
struct Decisions {
    /// What the gate decides for each held effect, when it decides.
    decided: Vec<(Entity, Decision)>,
}

/// Holds every fresh pending effect until `Decisions` says.
fn gate(
    fresh: Query<
        Entity,
        (
            With<PendingEffect>,
            Without<Held>,
            Without<Issued>,
            Without<EffectOutcome>,
        ),
    >,
    mut decisions: ResMut<Decisions>,
    mut commands: Commands,
) {
    for entity in &fresh {
        commands.entity(entity).insert(Held);
    }
    for (entity, decision) in decisions.decided.drain(..) {
        let mut entity = commands.entity(entity);
        entity.remove::<Held>();
        match decision {
            Decision::Proceed => {}
            Decision::Patch(kind) => {
                entity.insert(PendingEffect::new("model", kind));
            }
            Decision::Deny(report) => {
                entity.insert(EffectOutcome(Err(report)));
            }
        }
    }
}

#[test]
fn a_held_effect_is_denied_or_approved_from_a_system_next_tick() {
    let counters = Arc::new(Counters::default());
    let mut app = app();
    EffectLogResource::install(app.world_mut(), EffectLogRecorder::new());
    register(&mut app, "model", MockModel::new(&counters));
    app.init_resource::<Decisions>();
    add_in_gate(&mut app, gate);

    let denied = app
        .world_mut()
        .spawn(PendingEffect::new("model", completion()))
        .id();
    let approved = app
        .world_mut()
        .spawn(PendingEffect::new("model", completion()))
        .id();
    tick(&mut app, 3);
    assert!(app.world().get::<Held>(denied).is_some(), "held, not taken");
    assert!(app.world().get::<Issued>(approved).is_none());
    assert_eq!(counters.unary_started.load(Ordering::SeqCst), 0);

    app.world_mut().resource_mut::<Decisions>().decided = vec![
        (denied, Decision::deny("not today")),
        (approved, Decision::Proceed),
    ];
    tick_until(&mut app, "decided", |world| {
        world.get::<EffectOutcome>(denied).is_some()
            && world.get::<EffectOutcome>(approved).is_some()
    });
    let world = app.world();
    let denial = world.get::<EffectOutcome>(denied).expect("denied");
    assert_eq!(
        denial.0.as_ref().expect_err("denied").kind,
        ErrorKind::Denied
    );
    assert!(
        world.get::<Issued>(denied).is_none(),
        "a denial is no dispatch"
    );
    let answer = world.get::<EffectOutcome>(approved).expect("approved");
    assert_eq!(text_of(&answer.0), "hello from the world");

    // Decisions are program, never record: the log has the approved
    // dispatch alone, and a despawn of the denied entity is safe.
    let log = world.resource::<EffectLogResource>().log();
    assert_eq!(log.records.len(), 1, "{:?}", log.records);
    assert!(log.records[0].outcome.is_ok());
    app.world_mut().despawn(denied);
    tick(&mut app, 2);
}

fn patch_greeting(mut fresh: Query<&mut PendingEffect, (Without<Issued>, Without<EffectOutcome>)>) {
    for mut effect in &mut fresh {
        if let EffectKind::Completion { request, .. } = &mut effect.kind
            && request.chat_history.len() == 1
        {
            request.chat_history.push(Message::user("patched"));
        }
    }
}

fn replace_answer(mut landed: Query<&mut EffectOutcome, Added<EffectOutcome>>) {
    for mut outcome in &mut landed {
        outcome.0 = Err(ErrorReport::new(
            ErrorKind::Timeout,
            "replaced by the judge",
        ));
    }
}

#[test]
fn gate_patches_and_judge_replaces_but_the_record_keeps_the_answer() {
    let counters = Arc::new(Counters::default());
    let mut app = app();
    EffectLogResource::install(app.world_mut(), EffectLogRecorder::new());
    register(&mut app, "model", MockModel::new(&counters));
    add_in_gate(&mut app, patch_greeting);
    add_in_judge(&mut app, replace_answer);
    let effect = app
        .world_mut()
        .spawn(PendingEffect::new("model", completion()))
        .id();
    tick_until(&mut app, "answered", |world| {
        world.get::<EffectOutcome>(effect).is_some()
    });
    let world = app.world();
    let seen = world.get::<EffectOutcome>(effect).expect("answered");
    assert_eq!(
        seen.0.as_ref().expect_err("replaced").kind,
        ErrorKind::Timeout,
        "the consumer sees the judge's verdict"
    );
    let log = world.resource::<EffectLogResource>().log();
    assert_eq!(log.records.len(), 1);
    let record = &log.records[0];
    assert!(
        record.outcome.is_ok(),
        "the record keeps the handler's answer"
    );
    let EffectKind::Completion { request, .. } = &record.kind else {
        panic!("a completion");
    };
    assert_eq!(
        request.chat_history.len(),
        2,
        "the record's request is what the handler served: the patched one"
    );
}

/// A question whose answer needs the model: the answering system spawns a
/// child completion `ChildOf` the ask and answers once the child has.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct Nested {
    /// The key the child goes to.
    child_key: String,
}

impl CustomEffect for Nested {
    const KIND: &'static str = "nested";
    type Answer = String;
}

#[derive(Component)]
struct ChildEffect(Entity);

fn nest(
    asked: Query<(Entity, &Asked<Nested>), (Without<ChildEffect>, Without<Answer<Nested>>)>,
    mut commands: Commands,
) {
    for (entity, Asked(nested)) in &asked {
        let child = commands
            .spawn((
                PendingEffect::new(nested.child_key.as_str(), completion()),
                ChildOf(entity),
            ))
            .id();
        commands.entity(entity).insert(ChildEffect(child));
    }
}

fn finish_nested(
    asked: Query<(Entity, &ChildEffect), (With<Asked<Nested>>, Without<Answer<Nested>>)>,
    answered: Query<&EffectOutcome>,
    mut commands: Commands,
) {
    for (entity, ChildEffect(child)) in &asked {
        if let Ok(outcome) = answered.get(*child) {
            let text = match &outcome.0 {
                Ok(_) => text_of(&outcome.0),
                Err(report) => format!("child failed: {:?}", report.kind),
            };
            commands.entity(entity).insert(Answer::<Nested>(text));
        }
    }
}

#[test]
fn a_child_effect_records_its_parent_and_a_reentrant_one_is_refused() {
    let counters = Arc::new(Counters::default());
    let mut app = serial_app();
    EffectLogResource::install(app.world_mut(), EffectLogRecorder::new());
    register(&mut app, "model", MockModel::new(&counters));
    Handlers::with(app.world_mut(), |handlers| {
        handlers
            .register_world::<Nested>("world/nested")
            .expect("a fresh key")
    })
    .expect("a bus");
    add_after_judge(&mut app, (nest, finish_nested).chain());

    let ask = app
        .world_mut()
        .spawn(
            PendingEffect::custom(
                "world/nested",
                &Nested {
                    child_key: "model".to_owned(),
                },
            )
            .expect("serializes"),
        )
        .id();
    tick_until(&mut app, "nested answered", |world| {
        world.get::<EffectOutcome>(ask).is_some()
    });
    let world = app.world();
    let answer = world
        .get::<EffectOutcome>(ask)
        .expect("answered")
        .custom::<Nested>()
        .expect("a string");
    assert_eq!(answer, "hello from the world");
    let log = world.resource::<EffectLogResource>().log();
    assert_eq!(log.records.len(), 2, "{:?}", log.records);
    let parent_id = world.get::<Issued>(ask).expect("issued").0;
    let child = log
        .records
        .iter()
        .find(|record| record.key == HandlerKey::from("model"))
        .expect("the child's record");
    assert_eq!(child.parent, Some(parent_id), "causality in the record");
    let parent = log
        .records
        .iter()
        .find(|record| record.key == HandlerKey::from("world/nested"))
        .expect("the parent's record");
    assert_eq!(parent.parent, None);

    // Same-key nesting under serial serving: the child would wait for its
    // ancestor's key, so it is refused before any dispatch, with no record.
    let reentrant = app
        .world_mut()
        .spawn(
            PendingEffect::custom(
                "world/nested",
                &Nested {
                    child_key: "world/nested".to_owned(),
                },
            )
            .expect("serializes"),
        )
        .id();
    tick_until(&mut app, "reentrant refused", |world| {
        world.get::<EffectOutcome>(reentrant).is_some()
    });
    let world = app.world();
    let refused = world
        .get::<EffectOutcome>(reentrant)
        .expect("answered")
        .custom::<Nested>()
        .expect("a string");
    assert_eq!(refused, "child failed: Request");
    let log = world.resource::<EffectLogResource>().log();
    assert_eq!(
        log.records.len(),
        3,
        "the refused child left no record: {:?}",
        log.records
    );
}

#[test]
fn despawning_a_parent_cancels_its_children_in_flight_and_never_serves_the_queued() {
    let counters = Arc::new(Counters::default());
    counters.hold.hold();
    let mut app = app_with(rig_core::serve::ServingPolicy {
        serial_per_handler: true,
        ..Default::default()
    });
    EffectLogResource::install(app.world_mut(), EffectLogRecorder::new());
    register(&mut app, "model", MockModel::new(&counters));
    Handlers::with(app.world_mut(), |handlers| {
        handlers
            .register_world::<Nested>("world/nested")
            .expect("a fresh key")
    })
    .expect("a bus");
    add_after_judge(&mut app, (nest, finish_nested).chain());

    let ask = app
        .world_mut()
        .spawn(
            PendingEffect::custom(
                "world/nested",
                &Nested {
                    child_key: "model".to_owned(),
                },
            )
            .expect("serializes"),
        )
        .id();
    tick_until(&mut app, "child in flight", |_| {
        counters.unary_started.load(Ordering::SeqCst) == 1
    });
    // A second child of the same ask, queued behind the first on the
    // serial key: begun for nothing, so never a record.
    let queued = app
        .world_mut()
        .spawn((PendingEffect::new("model", completion()), ChildOf(ask)))
        .id();
    tick(&mut app, 2);
    assert!(
        app.world().get::<Issued>(queued).is_none(),
        "queued, not taken"
    );
    let in_flight: Vec<EffectId> = app
        .world_mut()
        .query::<(&Issued, &InFlight)>()
        .iter(app.world())
        .map(|(issued, _)| issued.0)
        .collect();
    assert_eq!(in_flight.len(), 2, "the ask and its first child");

    app.world_mut().despawn(ask);
    counters.hold.release();
    tick(&mut app, 3);
    assert!(
        app.world().get_entity(queued).is_err(),
        "despawned with its parent"
    );
    assert_eq!(
        counters.unary_served.load(Ordering::SeqCst),
        0,
        "the child in flight was dropped, not answered"
    );
    let log = app.world().resource::<EffectLogResource>().log();
    let mut cancelled: Vec<EffectId> = log
        .records
        .iter()
        .filter(
            |record| matches!(&record.outcome, Err(report) if report.kind == ErrorKind::Cancelled),
        )
        .map(|record| record.id)
        .collect();
    cancelled.sort();
    let mut expected = in_flight;
    expected.sort();
    assert_eq!(
        cancelled, expected,
        "both begun dispatches record Cancelled"
    );
    assert_eq!(log.records.len(), 2, "the queued child has no record");
    let _ = Outcome::Custom(serde_json::Value::Null);
}
