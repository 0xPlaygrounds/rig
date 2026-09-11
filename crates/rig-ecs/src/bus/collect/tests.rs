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
    super::super::plugin::install_bus(&mut world, Default::default());
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
fn ready_delivery_is_bounded_and_deltas_do_not_spin_quiescence() {
    let mut world = world();
    let effect = ready(&mut world, 0, 1000);
    super::super::plugin::run_to_quiescence(&mut world);
    assert_eq!(
        world.get::<Streamed>(effect).unwrap().events.len(),
        STREAM_ITEMS_PER_EFFECT
    );
    assert!(
        !world.resource::<Progress>().0,
        "deltas do not spin quiescence"
    );
}

#[test]
#[cfg(feature = "replay")]
fn empty_setup_polls_do_not_rotate_a_later_ready_delivery_batch() {
    use rig_core::{effect::EffectKind, serve::Origin};
    let mut world = world();
    let recorder = rig_effect_log::EffectLogRecorder::keeping_stream_events();
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
    world.resource_mut::<Schedules>().add_systems(
        super::super::plugin::RigSchedule,
        (|mut progress: ResMut<Progress>| progress.mark())
            .after(super::super::plugin::BusSet::Collect),
    );
    super::super::plugin::run_to_quiescence(&mut world);
    let delivered = |world: &World| {
        effects
            .iter()
            .map(|entity| world.get::<Streamed>(*entity).unwrap().events.len())
            .sum::<usize>()
    };
    assert_eq!(
        delivered(&world),
        STREAM_WORK_PER_TICK,
        "repeated quiescence passes share the streaming allowance"
    );
    super::super::plugin::run_to_quiescence(&mut world);
    assert_eq!(delivered(&world), 2 * STREAM_WORK_PER_TICK);
    assert!(
        effects
            .iter()
            .all(|entity| !world.get::<Streamed>(*entity).unwrap().events.is_empty()),
        "the next tick must begin with effects skipped by the previous limit"
    );
}

#[test]
fn partial_final_pass_rotates_fairly_across_ticks() {
    let mut world = world();
    let effects: Vec<_> = (0..40).map(|seq| ready(&mut world, seq, 1000)).collect();
    world.resource_mut::<Schedules>().add_systems(
        super::super::plugin::RigSchedule,
        (|mut progress: ResMut<Progress>| progress.mark())
            .after(super::super::plugin::BusSet::Collect),
    );
    for _ in 0..5 {
        super::super::plugin::run_to_quiescence(&mut world);
    }
    let counts: Vec<_> = effects
        .iter()
        .map(|entity| world.get::<Streamed>(*entity).unwrap().events.len())
        .collect();
    assert_eq!(counts, vec![512; 40]);
}

#[test]
fn repeated_quiescence_cannot_multiply_the_tick_allowance() {
    let mut world = world();
    let first = ready(&mut world, 0, 5000);
    let second = ready(&mut world, 1, 5000);
    world.resource_mut::<Schedules>().add_systems(
        super::super::plugin::RigSchedule,
        (|mut progress: ResMut<Progress>| progress.mark())
            .after(super::super::plugin::BusSet::Collect),
    );
    super::super::plugin::run_to_quiescence(&mut world);
    let first = world.get::<Streamed>(first).unwrap().events.len();
    let second = world.get::<Streamed>(second).unwrap().events.len();
    assert_eq!(first + second, STREAM_WORK_PER_TICK);
    assert_eq!(
        first, second,
        "both ready effects retain equal service across passes"
    );
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
#[test]
fn worker_self_wakes_without_collect_and_stalls_on_full_delivery() {
    let mut world = world();
    let polls = Arc::new(AtomicUsize::new(0));
    let drops = Arc::new(AtomicUsize::new(0));
    let (streaming, task) = Streaming::spawn(tracked_stream(3, polls.clone(), drops.clone()), 4);
    let effect = insert(&mut world, streaming, 0);
    world
        .non_send_mut::<Executions>()
        .streams
        .insert(effect, task);
    wait_for(|| polls.load(Ordering::SeqCst) >= 8);
    // futures mpsc has four shared slots and one sender-reserved slot.
    assert_eq!(polls.load(Ordering::SeqCst), 8);
    assert!(world.get::<Streamed>(effect).unwrap().events.is_empty());
    world.entity_mut(effect).remove::<InFlight>();
    wait_for(|| drops.load(Ordering::SeqCst) == 1);
    assert!(world.non_send::<Executions>().streams.is_empty());
}

#[cfg(not(target_family = "wasm"))]
#[test]
fn scheduled_despawn_replacement_and_shutdown_drop_owned_workers() {
    let mut world = world();
    let polls = Arc::new(AtomicUsize::new(0));
    let drops = Arc::new(AtomicUsize::new(0));
    let (streaming, task) = Streaming::spawn(tracked_stream(0, polls.clone(), drops.clone()), 1);
    let effect = insert(&mut world, streaming, 0);
    world
        .non_send_mut::<Executions>()
        .streams
        .insert(effect, task);
    let (streaming, task) = Streaming::spawn(tracked_stream(0, polls.clone(), drops.clone()), 1);
    world.entity_mut(effect).insert(streaming);
    world
        .non_send_mut::<Executions>()
        .streams
        .insert(effect, task);
    wait_for(|| drops.load(Ordering::SeqCst) == 1);
    let mut schedule = Schedule::default();
    schedule.add_systems(move |mut commands: Commands| {
        commands.entity(effect).despawn();
    });
    schedule.run(&mut world);
    wait_for(|| drops.load(Ordering::SeqCst) == 2);
    assert!(world.non_send::<Executions>().streams.is_empty());
    let (streaming, task) = Streaming::spawn(tracked_stream(0, polls, drops.clone()), 1);
    let effect = insert(&mut world, streaming, 1);
    world
        .non_send_mut::<Executions>()
        .streams
        .insert(effect, task);
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
        super::super::plugin::run_to_quiescence(&mut world);
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
