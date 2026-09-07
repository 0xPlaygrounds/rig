#![allow(clippy::unwrap_used)]
use super::*;
use rig_core::{effect::EffectId, streaming::UnknownPayload};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct Dropped(Arc<AtomicUsize>);
impl Drop for Dropped {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

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
            Poll::Ready(Some(Ok(StreamEvent::Unknown(UnknownPayload::new(
                serde_json::Value::Null,
            )))))
        }
    }))
}

fn world() -> World {
    let mut world = World::new();
    super::super::plugin::install_bus(&mut world, Default::default());
    world
}

fn insert(world: &mut World, stream: rig_core::streaming::StreamEvents) -> Entity {
    let entity = world
        .spawn((
            InFlight { key: "test".into() },
            Issued(EffectId::from_raw(1)),
            Streaming::default(),
            Streamed::default(),
        ))
        .id();
    world
        .non_send_mut::<Executions>()
        .streams
        .insert(entity, stream);
    entity
}

#[test]
fn self_wakes_wait_for_the_next_collect_and_ready_streams_get_one_poll() {
    let mut world = world();
    let polls = Arc::new(AtomicUsize::new(0));
    let drops = Arc::new(AtomicUsize::new(0));
    let effect = insert(&mut world, tracked_stream(3, polls.clone(), drops.clone()));
    let mut collect = Schedule::default();
    collect.add_systems(collect_streams);
    for pass in 1..=6 {
        collect.run(&mut world);
        assert_eq!(polls.load(Ordering::SeqCst), pass);
        assert_eq!(
            world.get::<Streamed>(effect).unwrap().events.len(),
            pass.saturating_sub(3)
        );
    }
    assert!(
        !world.resource::<Progress>().0,
        "deltas do not spin quiescence"
    );
    world.entity_mut(effect).remove::<InFlight>();
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert!(world.non_send::<Executions>().streams.is_empty());
}

#[test]
fn unrelated_progress_can_drive_multiple_collects_but_quiescence_stays_bounded() {
    for extra in [0, 5, 100] {
        let mut world = world();
        let polls = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        insert(&mut world, tracked_stream(0, polls.clone(), drops));
        world.resource_mut::<Schedules>().add_systems(
            super::super::plugin::RigSchedule,
            (move |mut progress: ResMut<Progress>, mut passes: Local<usize>| {
                if *passes < extra {
                    progress.mark();
                }
                *passes += 1;
            })
            .after(super::super::plugin::BusSet::Collect),
        );
        super::super::plugin::run_to_quiescence(&mut world);
        assert_eq!(polls.load(Ordering::SeqCst), (extra + 1).min(64));
    }
}

#[test]
fn scheduled_despawn_replacement_and_shutdown_drop_owned_streams() {
    let mut world = world();
    let polls = Arc::new(AtomicUsize::new(0));
    let drops = Arc::new(AtomicUsize::new(0));
    let effect = insert(&mut world, tracked_stream(0, polls.clone(), drops.clone()));
    world
        .non_send_mut::<Executions>()
        .streams
        .insert(effect, tracked_stream(0, polls.clone(), drops.clone()));
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    let mut schedule = Schedule::default();
    schedule.add_systems(move |mut commands: Commands| {
        commands.entity(effect).despawn();
    });
    schedule.run(&mut world);
    assert_eq!(drops.load(Ordering::SeqCst), 2);
    assert!(world.non_send::<Executions>().streams.is_empty());
    insert(&mut world, tracked_stream(0, polls, drops.clone()));
    drop(world);
    assert_eq!(drops.load(Ordering::SeqCst), 3);
}

#[test]
fn a_ready_initial_task_cannot_poll_its_stream_twice_in_one_collect() {
    let mut world = world();
    let polls = Arc::new(AtomicUsize::new(0));
    let drops = Arc::new(AtomicUsize::new(0));
    let stream = tracked_stream(0, polls.clone(), drops);
    let entity = world
        .spawn((
            InFlight { key: "test".into() },
            Issued(EffectId::from_raw(1)),
            Serving,
            Streamed::default(),
        ))
        .id();
    let task =
        bevy_tasks::IoTaskPool::get().spawn(async move { rig_core::serve::Reply::Stream(stream) });
    for _ in 0..100_000 {
        if task.is_finished() {
            break;
        }
        std::thread::yield_now();
    }
    assert!(task.is_finished(), "initial task did not finish");
    world
        .non_send_mut::<Executions>()
        .tasks
        .insert(entity, task);
    let mut collect = Schedule::default();
    collect.add_systems((collect_tasks, collect_streams).chain());
    collect.run(&mut world);
    assert_eq!(polls.load(Ordering::SeqCst), 1);
    assert_eq!(world.get::<Streamed>(entity).unwrap().events.len(), 1);
    collect.run(&mut world);
    assert_eq!(polls.load(Ordering::SeqCst), 2);
}
