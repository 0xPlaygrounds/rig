//! The app's loop without a window. It runs a frame, then sleeps until the
//! first of: a [`Wake`], the deadline of the next delayed command on Bevy's
//! clock (a retried model call waits on one), the shortest [`KeepAwake`]
//! interval, or a second. Frames never come faster than every 16 ms,
//! however often a streaming reply wakes the loop; the frames after a wake
//! come from [`settle`](rig_ecs::calls::settle), which works the same under
//! any loop.

use std::time::{Duration, Instant};

use bevy::time::{DelayedCommandQueue, Time};
use bevy_app::PluginsState;
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use rig_ecs::calls::{KeepAwake, Wake};

/// The shortest time between two frames.
const FRAME: Duration = Duration::from_millis(16);
/// The longest sleep without a wake, a deadline or a [`KeepAwake`]: what
/// polls without a wake, such as a `/reload` build's process, is seen
/// within it.
const IDLE: Duration = Duration::from_secs(1);

/// Sets the loop and its [`Wake`].
pub struct RunnerPlugin;

impl Plugin for RunnerPlugin {
    fn build(&self, app: &mut App) {
        // One pending wake is enough: the frame it brings sees all work.
        let (sender, wakes) = crossbeam_channel::bounded(1);
        app.insert_resource(Wake::new(move || {
            sender.try_send(()).ok();
        }));
        app.set_runner(move |mut app: App| {
            // As `ScheduleRunnerPlugin` does
            // (`bevy_app/src/schedule_runner.rs`).
            if app.plugins_state() != PluginsState::Cleaned {
                while app.plugins_state() == PluginsState::Adding {
                    bevy_tasks::tick_global_task_pools_on_main_thread();
                }
                app.finish();
                app.cleanup();
            }
            let mut due = NextFrame::new(app.world_mut());
            loop {
                let started = Instant::now();
                app.update();
                if let Some(exit) = app.should_exit() {
                    return exit;
                }
                let sleep = due.after(app.world()).saturating_sub(started.elapsed());
                wakes.recv_timeout(sleep).ok();
                if let Some(rest) = FRAME.checked_sub(started.elapsed()) {
                    std::thread::sleep(rest);
                }
            }
        });
    }
}

/// What decides when the next frame is due.
struct NextFrame {
    delayed: QueryState<&'static DelayedCommandQueue>,
    awake: QueryState<&'static KeepAwake>,
}

impl NextFrame {
    fn new(world: &mut World) -> Self {
        Self {
            delayed: world.query(),
            awake: world.query(),
        }
    }

    /// How long after the start of the frame that just ran the next one is
    /// due. Delayed commands are applied by the end of a frame, so every
    /// queue's deadline is on the clock by then, measured from the frame's
    /// start as Bevy's `Time` is.
    fn after(&mut self, world: &World) -> Duration {
        let delayed = world.get_resource::<Time>().and_then(|time| {
            self.delayed
                .iter(world)
                .map(|queue| queue.submit_at.saturating_sub(time.elapsed()))
                .min()
        });
        self.awake
            .iter(world)
            .map(|awake| awake.0)
            .chain(delayed)
            .fold(IDLE, Duration::min)
    }
}

#[cfg(test)]
mod tests;
