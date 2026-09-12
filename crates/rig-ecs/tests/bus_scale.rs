//! Scale and the schedule: many effects from parallel spawners, the
//! per-tick cost with thousands in flight (proof 14's successor), the
//! quiescence cap, and proof 9's successor (nothing dies, nothing is
//! re-registered).
//!
//! | proof / behaviour | test |
//! |---|---|
//! | §4.8 a thousand pendings from four parallel spawners, `Seq` order in the log | `a_thousand_effects_from_four_parallel_spawners_resolve_in_seq_order` |
//! | 14 ten thousand effects in flight cost a bounded `Collect` | `ten_thousand_effects_in_flight_cost_one_bounded_tick` |
//! | §11.2 the quiescence cap is a diagnostic, never a hang | `the_quiescence_cap_ends_the_tick` |
//! | 9 a handler survives every effect it served; nothing is re-registered | `handlers_outlive_every_effect_and_serve_again` |

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::type_complexity
)]

mod bus_support;

use std::{
    sync::{Arc, atomic::Ordering},
    time::Instant,
};

use bevy_ecs::prelude::*;
use bus_support::*;
use rig_core::serve::ServingPolicy;
use rig_ecs::bus::{
    BusSet, EffectLogResource, EffectOutcome, InFlight, PendingEffect, Progress, RigSchedule, Seq,
};
use rig_effect_log::EffectLogRecorder;

#[derive(Resource, Default)]
struct Spawned(std::sync::Mutex<Vec<Entity>>);

/// Four spawner systems, each with its own tag component, run in parallel
/// (they share nothing but `Commands`).
macro_rules! spawner {
    ($name:ident, $tag:ident) => {
        #[derive(Component)]
        struct $tag;

        fn $name(mut commands: Commands, spawned: Res<Spawned>, done: Query<(), With<$tag>>) {
            if done.iter().count() >= 250 {
                return;
            }
            let mut mine = spawned.0.lock().expect("spawned");
            for _ in 0..25 {
                let entity = commands
                    .spawn((PendingEffect::new("model", completion()), $tag))
                    .id();
                mine.push(entity);
            }
        }
    };
}

spawner!(spawn_a, A);
spawner!(spawn_b, B);
spawner!(spawn_c, C);
spawner!(spawn_d, D);

#[test]
fn a_thousand_effects_from_four_parallel_spawners_resolve_in_seq_order() {
    let counters = Arc::new(Counters::default());
    let mut app = app_with(ServingPolicy {
        command_capacity: 1_000,
        ..ServingPolicy::default()
    });
    EffectLogResource::install(app.world_mut(), EffectLogRecorder::new());
    register(&mut app, "model", MockModel::new(&counters));
    app.init_resource::<Spawned>();
    app.world_mut()
        .resource_mut::<bevy_ecs::schedule::Schedules>()
        .add_systems(
            RigSchedule,
            (spawn_a, spawn_b, spawn_c, spawn_d).before(BusSet::Gate),
        );
    tick_until(&mut app, "a thousand answered", |world| {
        let spawned = world
            .resource::<Spawned>()
            .0
            .lock()
            .expect("spawned")
            .clone();
        spawned.len() == 1_000
            && spawned
                .iter()
                .all(|entity| world.get::<EffectOutcome>(*entity).is_some())
    });
    assert_eq!(counters.unary_served.load(Ordering::SeqCst), 1_000);
    // The log's dispatch order is the `Seq` order: strictly increasing
    // ids along strictly increasing seqs.
    let world = app.world_mut();
    let mut rows: Vec<(u64, u64)> = world
        .query::<(&Seq, &rig_ecs::bus::Issued)>()
        .iter(world)
        .map(|(seq, issued)| (seq.0, issued.0.as_u64()))
        .collect();
    rows.sort_by_key(|(seq, _)| *seq);
    assert_eq!(rows.len(), 1_000);
    assert!(rows.windows(2).all(|w| w[0].1 < w[1].1), "ids follow seqs");
    let log = world.resource::<EffectLogResource>().log();
    assert_eq!(log.records.len(), 1_000);
    assert!(
        log.records.windows(2).all(|w| w[0].id < w[1].id),
        "the log is in dispatch order"
    );
}

#[test]
fn ten_thousand_effects_in_flight_cost_one_bounded_tick() {
    let counters = Arc::new(Counters::default());
    counters.hold.hold();
    let mut app = app_with(ServingPolicy {
        command_capacity: 10_000,
        ..ServingPolicy::default()
    });
    register(&mut app, "model", MockModel::new(&counters));
    let effects: Vec<Entity> = (0..10_000)
        .map(|_| {
            app.world_mut()
                .spawn(PendingEffect::new("model", completion()))
                .id()
        })
        .collect();
    tick_until(&mut app, "all in flight", |world| {
        world.query::<&InFlight>().iter(world).count() == 10_000
    });
    // Every tick now checks ten thousand held tasks: the number the PR
    // reports (proof 14's successor — there is no inbox to compare against;
    // the check is `check_ready` per task, no waker, no probe of a future
    // the host holds).
    let ticks = 100_u32;
    let start = Instant::now();
    tick(&mut app, ticks as usize);
    let per_tick = start.elapsed() / ticks;
    eprintln!("ten thousand in flight: {per_tick:?} per tick");
    assert!(
        per_tick < std::time::Duration::from_millis(50),
        "a tick with ten thousand in flight took {per_tick:?}"
    );
    counters.hold.release();
    tick_until(&mut app, "all answered", |world| {
        effects
            .iter()
            .all(|entity| world.get::<EffectOutcome>(*entity).is_some())
    });
}

/// A system that always reports progress: the cap must end the tick.
fn always_progress(mut progress: ResMut<Progress>, mut passes: ResMut<Passes>) {
    progress.mark();
    passes.0 += 1;
}

#[derive(Resource, Default)]
struct Passes(usize);

#[test]
fn the_quiescence_cap_ends_the_tick() {
    let mut app = app();
    app.init_resource::<Passes>();
    app.world_mut()
        .resource_mut::<bevy_ecs::schedule::Schedules>()
        .add_systems(RigSchedule, always_progress.after(BusSet::Judge));
    let start = Instant::now();
    app.update();
    assert!(start.elapsed() < GUARD);
    assert_eq!(
        app.world().resource::<Passes>().0,
        rig_ecs::bus::QUIESCENCE_CAP,
        "exactly the cap's passes, then the tick ends"
    );
}

#[test]
fn handlers_outlive_every_effect_and_serve_again() {
    let counters = Arc::new(Counters::default());
    let mut app = app();
    register(&mut app, "model", MockModel::new(&counters));
    let first = app
        .world_mut()
        .spawn(PendingEffect::new("model", completion()))
        .id();
    tick_until(&mut app, "first", |world| {
        world.get::<EffectOutcome>(first).is_some()
    });
    // Every effect entity gone — the fixture's "driver dead" moment: here
    // nothing dies, because the handler is an entity of its own.
    app.world_mut().despawn(first);
    tick(&mut app, 2);
    let effects = app
        .world_mut()
        .query::<&PendingEffect>()
        .iter(app.world())
        .count();
    assert_eq!(effects, 0);
    let second = app
        .world_mut()
        .spawn(PendingEffect::new("model", completion()))
        .id();
    tick_until(&mut app, "second", |world| {
        world.get::<EffectOutcome>(second).is_some()
    });
    assert_eq!(counters.unary_served.load(Ordering::SeqCst), 2);
}
