//! Effects are entities: the fixture's proofs 1–7 in the native shape.
//!
//! | proof | test |
//! |---|---|
//! | 1 the bounds hold as components | `every_component_a_system_holds_is_send_sync` |
//! | 2 the handler's future is held in the entity | `a_pending_effect_is_taken_served_and_answered` |
//! | 3 answered across ticks with no waker per frame | `a_pending_effect_is_taken_served_and_answered` |
//! | 4 despawn in flight cancels; pre-dispatch despawn never serves | `despawning_an_effect_in_flight_cancels_its_handler`, `an_effect_despawned_before_dispatch_is_never_served` |
//! | 5 the intake bound blocks nobody | `the_intake_bound_leaves_the_rest_pending_and_blocks_nobody` |
//! | 6 a stream lands per tick; drop mid-stream cancels | `a_stream_accumulates_per_tick`, `despawning_mid_stream_cancels_the_handler` |
//! | 7 register, dispatch, deregister from systems | `handlers_are_registered_and_removed_from_systems` |

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::type_complexity
)]

mod bus_support;

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use bevy_ecs::prelude::*;
use bus_support::*;
use rig_core::{
    effect::{EffectFamily, HandlerKey},
    error::ErrorKind,
    serve::ServingPolicy,
};
use rig_ecs::bus::{
    Bound, BusSet, EffectOutcome, Handlers, InFlight, Issued, PendingEffect, RigSchedule, Seq,
    Serving, Streamed, Streaming,
};

#[test]
fn every_component_a_system_holds_is_send_sync() {
    fn assert_send_sync<T: Send + Sync + 'static>() {}
    assert_send_sync::<PendingEffect>();
    assert_send_sync::<Seq>();
    assert_send_sync::<Issued>();
    assert_send_sync::<InFlight>();
    assert_send_sync::<Serving>();
    assert_send_sync::<Streaming>();
    assert_send_sync::<Streamed>();
    assert_send_sync::<EffectOutcome>();
    assert_send_sync::<Bound>();
}

#[test]
fn a_pending_effect_is_taken_served_and_answered() {
    let counters = Arc::new(Counters::default());
    let mut app = app();
    register(&mut app, "model", MockModel::new(&counters));
    let effect = app
        .world_mut()
        .spawn(PendingEffect::new("model", completion()))
        .id();

    let ticks = tick_until(&mut app, "answered", |world| {
        world.get::<EffectOutcome>(effect).is_some()
    });
    let world = app.world();
    let outcome = world.get::<EffectOutcome>(effect).expect("answered");
    assert_eq!(text_of(&outcome.0), "hello from the world");
    assert!(world.get::<Issued>(effect).is_some(), "the id stays");
    assert!(world.get::<InFlight>(effect).is_none(), "settled");
    assert!(
        world.get::<Serving>(effect).is_none(),
        "the task left the entity"
    );
    assert_eq!(counters.unary_served.load(Ordering::SeqCst), 1);
    assert!(ticks >= 1);
}

#[test]
fn an_effect_despawned_before_dispatch_is_never_served() {
    let counters = Arc::new(Counters::default());
    let mut app = app();
    register(&mut app, "model", MockModel::new(&counters));
    let effect = app
        .world_mut()
        .spawn(PendingEffect::new("model", completion()))
        .id();
    app.world_mut().despawn(effect);
    tick(&mut app, 3);
    assert_eq!(counters.unary_started.load(Ordering::SeqCst), 0);
}

#[test]
fn despawning_an_effect_in_flight_cancels_its_handler() {
    let counters = Arc::new(Counters::default());
    counters.hold.store(true, Ordering::SeqCst);
    let mut app = app();
    register(&mut app, "model", MockModel::new(&counters));
    let effect = app
        .world_mut()
        .spawn(PendingEffect::new("model", completion()))
        .id();
    tick_until(&mut app, "in flight", |_| {
        counters.unary_started.load(Ordering::SeqCst) == 1
    });
    assert!(app.world().get::<InFlight>(effect).is_some());
    app.world_mut().despawn(effect);
    counters.hold.store(false, Ordering::SeqCst);
    tick(&mut app, 3);
    // The task was dropped with the entity: the handler never answered.
    assert_eq!(counters.unary_served.load(Ordering::SeqCst), 0);
}

#[test]
fn the_intake_bound_leaves_the_rest_pending_and_blocks_nobody() {
    let counters = Arc::new(Counters::default());
    counters.hold.store(true, Ordering::SeqCst);
    let mut app = app_with(ServingPolicy {
        command_capacity: 2,
        ..ServingPolicy::default()
    });
    register(&mut app, "model", MockModel::new(&counters));
    let effects: Vec<Entity> = (0..5)
        .map(|_| {
            app.world_mut()
                .spawn(PendingEffect::new("model", completion()))
                .id()
        })
        .collect();
    app.update();
    let in_flight = effects
        .iter()
        .filter(|e| app.world().get::<InFlight>(**e).is_some())
        .count();
    assert_eq!(in_flight, 2, "two taken this tick");
    app.update();
    let in_flight = effects
        .iter()
        .filter(|e| app.world().get::<InFlight>(**e).is_some())
        .count();
    assert_eq!(in_flight, 4, "two more the next tick");
    counters.hold.store(false, Ordering::SeqCst);
    tick_until(&mut app, "all answered", |world| {
        effects
            .iter()
            .all(|e| world.get::<EffectOutcome>(*e).is_some())
    });
    assert_eq!(counters.unary_served.load(Ordering::SeqCst), 5);
}

#[test]
fn a_stream_accumulates_per_tick() {
    let counters = Arc::new(Counters::default());
    let mut app = app();
    register(&mut app, "model", MockModel::new(&counters));
    let effect = app
        .world_mut()
        .spawn(PendingEffect::new("model", streaming()))
        .id();

    let mut seen = Vec::new();
    tick_until(&mut app, "stream done", |world| {
        if let Some(streamed) = world.get::<Streamed>(effect) {
            seen.push(streamed.events.len());
        }
        world.get::<EffectOutcome>(effect).is_some()
    });
    let world = app.world();
    let streamed = world.get::<Streamed>(effect).expect("the fold stays");
    assert!(
        streamed.text.starts_with("tick tick "),
        "text folded: {:?}",
        streamed.text
    );
    assert!(seen.windows(2).all(|w| w[0] <= w[1]), "monotone: {seen:?}");
    let outcome = world
        .get::<EffectOutcome>(effect)
        .expect("answered at the terminal");
    assert!(outcome.0.is_ok(), "{:?}", outcome.0);
    assert!(world.get::<Streaming>(effect).is_none());
    assert!(world.get::<InFlight>(effect).is_none());
}

#[test]
fn despawning_mid_stream_cancels_the_handler() {
    let counters = Arc::new(Counters::default());
    let mut app = app();
    register(&mut app, "model", MockModel::endless(&counters));
    let effect = app
        .world_mut()
        .spawn(PendingEffect::new("model", streaming()))
        .id();
    tick_until(&mut app, "streaming", |world| {
        world
            .get::<Streamed>(effect)
            .is_some_and(|streamed| !streamed.events.is_empty())
    });
    app.world_mut().despawn(effect);
    // The task went with the entity: the handler is dropped, never told —
    // exactly as rig-bus drops a handler future when its consumer goes —
    // so its sends stop where they were.
    tick(&mut app, 2);
    std::thread::sleep(std::time::Duration::from_millis(20));
    let sent = counters.stream_sends.load(Ordering::SeqCst);
    tick(&mut app, 20);
    std::thread::sleep(std::time::Duration::from_millis(20));
    assert_eq!(
        counters.stream_sends.load(Ordering::SeqCst),
        sent,
        "nothing sent after the despawn"
    );
}

#[derive(Resource)]
struct Runtime {
    key: HandlerKey,
    counters: Arc<Counters>,
    registered: bool,
    effect: Option<Entity>,
    answered: Arc<AtomicUsize>,
}

fn register_from_a_system(mut handlers: Handlers, mut runtime: ResMut<Runtime>) {
    if !runtime.registered {
        handlers
            .register(runtime.key.clone(), MockModel::new(&runtime.counters))
            .expect("a fresh key");
        runtime.registered = true;
    }
}

fn dispatch_from_a_system(mut commands: Commands, mut runtime: ResMut<Runtime>) {
    if runtime.effect.is_none() {
        let effect = commands
            .spawn(PendingEffect::new(runtime.key.clone(), completion()))
            .id();
        runtime.effect = Some(effect);
    }
}

fn observe_from_a_system(
    runtime: Res<Runtime>,
    answered: Query<&EffectOutcome, Added<EffectOutcome>>,
) {
    if let Some(effect) = runtime.effect
        && answered.get(effect).is_ok()
    {
        runtime.answered.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn handlers_are_registered_and_removed_from_systems() {
    let counters = Arc::new(Counters::default());
    let answered = Arc::new(AtomicUsize::new(0));
    let mut app = app();
    app.insert_resource(Runtime {
        key: HandlerKey::from("runtime/model"),
        counters: Arc::clone(&counters),
        registered: false,
        effect: None,
        answered: Arc::clone(&answered),
    });
    app.world_mut()
        .resource_mut::<bevy_ecs::schedule::Schedules>()
        .add_systems(
            RigSchedule,
            (
                register_from_a_system.before(BusSet::Gate),
                dispatch_from_a_system.in_set(BusSet::Gate),
                observe_from_a_system.after(BusSet::Judge),
            ),
        );
    tick_until(&mut app, "answered from a system", |_| {
        answered.load(Ordering::SeqCst) == 1
    });
    let key = HandlerKey::from("runtime/model");
    let bound = Handlers::with(app.world_mut(), |handlers| {
        let described = handlers.descriptor(&key).expect("bound");
        assert_eq!(described.family.family(), EffectFamily::Completion);
        assert!(handlers.keys().contains(&key));
        handlers.deregister(&key)
    })
    .expect("a bus");
    assert!(bound, "was registered");
    app.update();
    let gone =
        Handlers::with(app.world_mut(), |handlers| handlers.descriptor(&key)).expect("a bus");
    assert!(gone.is_none(), "deregistered: the entity despawned");
    let orphan = app
        .world_mut()
        .spawn(PendingEffect::new(key, completion()))
        .id();
    tick_until(&mut app, "unavailable", |world| {
        world.get::<EffectOutcome>(orphan).is_some()
    });
    let outcome = app.world().get::<EffectOutcome>(orphan).expect("answered");
    assert_eq!(
        outcome.0.as_ref().expect_err("no handler").kind,
        ErrorKind::HandlerUnavailable
    );
}
