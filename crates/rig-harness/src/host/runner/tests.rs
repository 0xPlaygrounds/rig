use std::sync::mpsc;
use std::time::{Duration, Instant};

use bevy::time::{DelayedCommandsExt, TimePlugin};
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

use super::RunnerPlugin;

/// Runs an app on the loop with nothing to do but a command delayed by
/// `delay`, which exits it. How long after the start the command ran.
fn delayed_exit(delay: Duration) -> Option<Duration> {
    let (sender, ran) = mpsc::channel();
    let started = Instant::now();
    let mut app = App::new();
    app.add_plugins((TimePlugin, RunnerPlugin)).add_systems(
        Startup,
        move |mut commands: Commands| {
            let sender = sender.clone();
            commands
                .delayed()
                .duration(delay)
                .queue(move |world: &mut World| {
                    sender.send(Instant::now()).ok();
                    world.write_message(AppExit::Success);
                });
        },
    );
    app.run();
    ran.try_recv().ok().map(|at| at.duration_since(started))
}

#[test]
fn the_loop_wakes_for_a_delayed_command_within_a_frame() {
    let delay = Duration::from_millis(300);
    let ran = delayed_exit(delay);
    // Without its deadline the loop would sleep until woken.
    assert!(
        ran.is_some_and(|ran| ran >= delay && ran < delay + Duration::from_millis(200)),
        "{ran:?}"
    );
}

#[test]
fn a_wait_across_idle_frames_ends_on_time() {
    // One long sleep: Bevy's own 250 ms cap on a frame's time would make
    // this wait many times as long.
    let delay = Duration::from_secs(2);
    let ran = delayed_exit(delay);
    assert!(
        ran.is_some_and(|ran| ran >= delay && ran < delay + Duration::from_millis(500)),
        "{ran:?}"
    );
}

#[test]
fn the_loop_wakes_at_a_wall_clock_time() {
    let delay = Duration::from_millis(300);
    let (sender, ran) = mpsc::channel();
    let started = Instant::now();
    let mut app = App::new();
    app.add_plugins((TimePlugin, RunnerPlugin, crate::clock::ClockPlugin))
        .add_systems(Startup, move |mut commands: Commands| {
            commands.spawn(crate::clock::WakeAt::after(delay));
        })
        .add_observer(
            move |_: On<Add<crate::clock::Due>>, mut exit: MessageWriter<AppExit>| {
                sender.send(Instant::now()).ok();
                exit.write(AppExit::Success);
            },
        );
    app.run();
    let ran = ran.try_recv().ok().map(|at| at.duration_since(started));
    // Without the alarm the loop would sleep until woken.
    assert!(
        ran.is_some_and(|ran| ran >= delay && ran < delay + Duration::from_millis(200)),
        "{ran:?}"
    );
}

#[test]
fn an_idle_loop_sleeps_until_woken() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

    use rig_ecs::tasks::Wake;

    let frames = Arc::new(AtomicU32::new(0));
    let done = Arc::new(AtomicBool::new(false));
    let (counted, finished) = (frames.clone(), done.clone());
    let mut app = App::new();
    app.add_plugins((TimePlugin, RunnerPlugin))
        .add_systems(Startup, move |wake: Res<Wake>| {
            let (wake, done) = (wake.clone(), done.clone());
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(3500));
                done.store(true, Ordering::Release);
                wake.wake();
            });
        })
        .add_systems(Update, move |mut exit: MessageWriter<AppExit>| {
            counted.fetch_add(1, Ordering::Relaxed);
            if finished.load(Ordering::Acquire) {
                exit.write(AppExit::Success);
            }
        });
    app.run();
    // The first frame, then the one the wake brought: no frame polls.
    let frames = frames.load(Ordering::Relaxed);
    assert!(frames <= 2, "{frames} frames");
}
