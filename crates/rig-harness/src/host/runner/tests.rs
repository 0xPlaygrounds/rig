use std::sync::mpsc;
use std::time::{Duration, Instant};

use bevy::time::{DelayedCommandsExt, TimePlugin};
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

use super::{IDLE, RunnerPlugin};

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
    // Without its deadline the loop would sleep a whole `IDLE`.
    assert!(
        ran.is_some_and(|ran| ran >= delay && ran < delay + Duration::from_millis(200)),
        "{ran:?}"
    );
}

#[test]
fn a_wait_across_idle_frames_ends_on_time() {
    // Two idle frames pass first: Bevy's own 250 ms cap on a frame's time
    // would make this wait about four times as long.
    let delay = 2 * IDLE;
    let ran = delayed_exit(delay);
    assert!(
        ran.is_some_and(|ran| ran >= delay && ran < delay + Duration::from_millis(500)),
        "{ran:?}"
    );
}
