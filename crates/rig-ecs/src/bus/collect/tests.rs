#![allow(clippy::unwrap_used)]
use super::*;
use rig_core::{effect::EffectId, streaming::UnknownPayload};
use std::sync::Arc;
#[cfg(not(target_family = "wasm"))]
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(not(target_family = "wasm"))]
struct Dropped(Arc<AtomicUsize>);
#[cfg(not(target_family = "wasm"))]
impl Drop for Dropped {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

fn item() -> Result<StreamEvent, rig_core::error::ErrorReport> {
    Ok(StreamEvent::Unknown(UnknownPayload::new(
        serde_json::Value::Null,
    )))
}

#[cfg(not(target_family = "wasm"))]
fn tracked_stream(
    pending: usize,
    polls: Arc<AtomicUsize>,
    drops: Arc<AtomicUsize>,
) -> rig_core::streaming::StreamEvents {
    let guard = Dropped(drops);
    Box::pin(futures::stream::poll_fn(move |cx| {
        let _guard = &guard;
        let poll = polls.fetch_add(1, Ordering::SeqCst);
        if poll < pending {
            cx.waker().wake_by_ref();
            Poll::Pending
        } else {
            Poll::Ready(Some(item()))
        }
    }))
}

fn world() -> World {
    let mut world = World::new();
    crate::bus::BusPlugin::with_policy(Default::default()).install(&mut world);
    world
}

fn insert(world: &mut World, streaming: Streaming, seq: u64) -> Entity {
    world
        .spawn((
            InFlight { key: "test".into() },
            super::super::Seq(seq),
            Issued(EffectId::from_raw(seq)),
            streaming,
            Streamed::default(),
        ))
        .id()
}

fn ready(world: &mut World, seq: u64, count: usize) -> Entity {
    // Prefill without a worker so scheduler tests cannot depend on OS timing.
    let (mut sender, events) = futures::channel::mpsc::channel(count);
    for _ in 0..count {
        sender.try_send(item()).unwrap();
    }
    insert(
        world,
        Streaming {
            events,
            fold: rig_core::serve::StreamTap::new(),
            delivered: 0,
        },
        seq,
    )
}

#[test]
fn ready_delivery_is_bounded_per_pass() {
    let mut world = world();
    let effect = ready(&mut world, 0, 1000);
    world.run_schedule(super::super::plugin::RigSchedule);
    assert_eq!(
        world.get::<Streamed>(effect).unwrap().events.len(),
        STREAM_ITEMS_PER_EFFECT
    );
}

#[test]
fn empty_setup_polls_do_not_rotate_a_later_ready_delivery_batch() {
    use rig_core::{effect::EffectKind, serve::Origin};
    let mut world = world();
    let recorder = rig_cassette::effect_log::EffectLogRecorder::keeping_stream_events();
    Recording::install(&mut world, recorder.clone());
    for id in 0..2 {
        world.resource::<Recording>().begin(
            EffectId::from_raw(id),
            "test".into(),
            EffectKind::Custom {
                kind: "test".into(),
                payload: serde_json::Value::Null,
            },
            Origin::default(),
        );
    }
    let (mut first_sender, first_events) = futures::channel::mpsc::channel(4);
    insert(
        &mut world,
        Streaming {
            events: first_events,
            fold: rig_core::serve::StreamTap::new(),
            delivered: 0,
        },
        0,
    );
    let mut schedule = Schedule::default();
    schedule.add_systems(collect_streams);
    // Only the first worker has installed its empty stream. No delivery or
    // work-budget exhaustion occurs before the second worker is installed.
    schedule.run(&mut world);
    assert!(recorder.header().deliveries.unwrap().is_empty());
    let (mut second_sender, second_events) = futures::channel::mpsc::channel(4);
    insert(
        &mut world,
        Streaming {
            events: second_events,
            fold: rig_core::serve::StreamTap::new(),
            delivered: 0,
        },
        1,
    );
    first_sender.try_send(item()).unwrap();
    second_sender.try_send(item()).unwrap();
    schedule.run(&mut world);
    let order: Vec<_> = recorder
        .header()
        .deliveries
        .unwrap()
        .iter()
        .map(|delivery| delivery.id.as_u64())
        .collect();
    assert_eq!(order, [0, 1]);
}

#[test]
fn whole_tick_allowance_rotates_service_across_hot_effects() {
    let mut world = world();
    let effects: Vec<_> = (0..80).map(|seq| ready(&mut world, seq, 1000)).collect();
    world.run_schedule(super::super::plugin::RigSchedule);
    let delivered = |world: &World| {
        effects
            .iter()
            .map(|entity| world.get::<Streamed>(*entity).unwrap().events.len())
            .sum::<usize>()
    };
    assert_eq!(
        delivered(&world),
        STREAM_WORK_PER_TICK,
        "one pass spends the whole allowance"
    );
    world.run_schedule(super::super::plugin::RigSchedule);
    assert_eq!(delivered(&world), 2 * STREAM_WORK_PER_TICK);
    assert!(
        effects
            .iter()
            .all(|entity| !world.get::<Streamed>(*entity).unwrap().events.is_empty()),
        "the next tick must begin with effects skipped by the previous limit"
    );
}

#[test]
/// Forty hot effects under the allowance: every pass serves each its
/// per-effect cap, so five passes give every effect five caps. (Re-recorded
/// from 512: one pass per update, no quiescence loop spending the whole
/// allowance in one update.)
fn hot_effects_under_the_allowance_get_the_per_effect_cap_every_pass() {
    let mut world = world();
    let effects: Vec<_> = (0..40).map(|seq| ready(&mut world, seq, 1000)).collect();
    for _ in 0..5 {
        world.run_schedule(super::super::plugin::RigSchedule);
    }
    let counts: Vec<_> = effects
        .iter()
        .map(|entity| world.get::<Streamed>(*entity).unwrap().events.len())
        .collect();
    assert_eq!(counts, vec![5 * STREAM_ITEMS_PER_EFFECT; 40]);
}

#[test]
fn two_hot_effects_get_equal_service_in_one_pass() {
    let mut world = world();
    let first = ready(&mut world, 0, 5000);
    let second = ready(&mut world, 1, 5000);
    world.run_schedule(super::super::plugin::RigSchedule);
    let first = world.get::<Streamed>(first).unwrap().events.len();
    let second = world.get::<Streamed>(second).unwrap().events.len();
    assert_eq!(first, STREAM_ITEMS_PER_EFFECT);
    assert_eq!(first, second, "both ready effects get equal service");
}

#[cfg(not(target_family = "wasm"))]
fn wait_for(mut done: impl FnMut() -> bool) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while !done() {
        assert!(
            std::time::Instant::now() < deadline,
            "worker did not progress"
        );
        std::thread::yield_now();
    }
}

#[cfg(not(target_family = "wasm"))]
fn worker(
    world: &mut World,
    effect: Entity,
    stream: rig_core::streaming::StreamEvents,
    capacity: usize,
) {
    let streaming = Tasks::with(world, |tasks| {
        tasks.streaming(effect, stream, capacity, Default::default())
    })
    .unwrap();
    world.entity_mut(effect).insert(streaming);
}

#[cfg(not(target_family = "wasm"))]
fn bare(world: &mut World, seq: u64) -> Entity {
    world
        .spawn((
            InFlight { key: "test".into() },
            super::super::Seq(seq),
            Issued(EffectId::from_raw(seq)),
            Streamed::default(),
        ))
        .id()
}

#[cfg(not(target_family = "wasm"))]
#[test]
fn worker_self_wakes_without_collect_and_stalls_on_full_delivery() {
    let mut world = world();
    let polls = Arc::new(AtomicUsize::new(0));
    let drops = Arc::new(AtomicUsize::new(0));
    let effect = bare(&mut world, 0);
    worker(
        &mut world,
        effect,
        tracked_stream(3, polls.clone(), drops.clone()),
        4,
    );
    wait_for(|| polls.load(Ordering::SeqCst) >= 8);
    // futures mpsc has four shared slots and one sender-reserved slot.
    assert_eq!(polls.load(Ordering::SeqCst), 8);
    assert!(world.get::<Streamed>(effect).unwrap().events.is_empty());
    // The table owns the worker: leaving flight drops the stream.
    world.entity_mut(effect).remove::<InFlight>();
    wait_for(|| drops.load(Ordering::SeqCst) == 1);
}

#[cfg(not(target_family = "wasm"))]
#[test]
fn scheduled_despawn_replacement_and_shutdown_drop_owned_workers() {
    let mut world = world();
    let polls = Arc::new(AtomicUsize::new(0));
    let drops = Arc::new(AtomicUsize::new(0));
    let effect = bare(&mut world, 0);
    worker(
        &mut world,
        effect,
        tracked_stream(0, polls.clone(), drops.clone()),
        1,
    );
    // Replacing the component drops the first worker.
    worker(
        &mut world,
        effect,
        tracked_stream(0, polls.clone(), drops.clone()),
        1,
    );
    wait_for(|| drops.load(Ordering::SeqCst) == 1);
    let mut schedule = Schedule::default();
    schedule.add_systems(move |mut commands: Commands| {
        commands.entity(effect).despawn();
    });
    schedule.run(&mut world);
    wait_for(|| drops.load(Ordering::SeqCst) == 2);
    let effect = bare(&mut world, 1);
    worker(
        &mut world,
        effect,
        tracked_stream(0, polls, drops.clone()),
        1,
    );
    drop(world);
    wait_for(|| drops.load(Ordering::SeqCst) == 3);
}

#[test]
fn unary_stream_truncation_counts_items_across_collection_passes() {
    let mut world = world();
    let log = Arc::new(rig_core::observe::ObservationLog::default());
    Witnessing::install(&mut world, log.clone());
    let count = STREAM_ITEMS_PER_EFFECT * 2 + 3;
    let effect = ready(&mut world, 0, count);
    // Exercise the collector's unary fold, with a closed, prefilled receiver.
    world.entity_mut(effect).remove::<Streamed>();
    for _ in 0..4 {
        world.run_schedule(super::super::plugin::RigSchedule);
        if world.get::<EffectOutcome>(effect).is_some() {
            break;
        }
    }
    assert!(world.get::<EffectOutcome>(effect).unwrap().0.is_err());
    let after: Vec<_> = log
        .trace()
        .observations
        .iter()
        .filter_map(|fact| match &fact.action {
            Action::StreamTruncated { delivered, .. } => Some(*delivered),
            _ => None,
        })
        .collect();
    assert_eq!(after, [count]);
}
