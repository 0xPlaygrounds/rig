//! The app's loop without a window. It runs a frame, then sleeps until the
//! first of: a [`Wake`], the deadline of the next delayed command on Bevy's
//! clock (a retried model call waits on one), the next [`WakeAt`] on the
//! wall clock, or the shortest [`KeepAwake`] interval; with none of them
//! it sleeps until woken, so an idle agent or server runs no frame. Frames
//! never come faster than every 16 ms, however often a streaming reply
//! wakes the loop; the few frames after a wake come from the kernel (see
//! [`Wake`]), the same under any loop. Every schedule runs on the main
//! thread, one system after another, set as the plugins are cleaned up.

use std::time::{Duration, Instant, SystemTime};

use bevy::time::{DelayedCommandQueue, Time, Virtual};
use bevy_app::PluginsState;
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_ecs::schedule::SingleThreadedExecutor;
use rig_ecs::tasks::Wake;

use crate::clock::{Due, KeepAwake, WakeAt};

/// The shortest time between two frames.
const FRAME: Duration = Duration::from_millis(16);
/// The longest gap between two frames that Bevy's virtual clock counts in
/// full. Bevy's default, 250 ms, suits a game's steady frames; across this
/// loop's long sleeps it would make a retry's wait many times as long. A longer gap, such as a suspended machine, is still cut short,
/// as Bevy intends: its fixed-step loop runs once for every 15.6 ms
/// counted, all in the frame after the gap.
const MAX_FRAME_GAP: Duration = Duration::from_secs(60);

/// Finishes and cleans up the app's plugins, as Bevy's runners do before
/// their first frame (`bevy_app/src/schedule_runner.rs`), which Bevy does
/// not expose.
pub(crate) fn finish_plugins(app: &mut App) {
    if app.plugins_state() != PluginsState::Cleaned {
        while app.plugins_state() == PluginsState::Adding {
            bevy_tasks::tick_global_task_pools_on_main_thread();
        }
        app.finish();
        app.cleanup();
    }
}

/// Sets the loop and its [`Wake`], and lets Bevy's clock count frames up
/// to a minute apart in full.
pub struct RunnerPlugin;

impl Plugin for RunnerPlugin {
    fn build(&self, app: &mut App) {
        // One pending wake is enough: the frame it brings sees all work.
        let (sender, wakes) = crossbeam_channel::bounded(1);
        app.insert_resource(Wake::new(move || {
            sender.try_send(()).ok();
        }));
        app.set_runner(move |mut app: App| {
            finish_plugins(&mut app);
            let mut due = NextFrame::new(app.world_mut());
            loop {
                let started = Instant::now();
                app.update();
                if let Some(exit) = app.should_exit() {
                    return exit;
                }
                match due.after(app.world()) {
                    Some(after) => {
                        wakes.recv_timeout(after.saturating_sub(started.elapsed())).ok();
                    }
                    // Nothing is due: only a wake brings the next frame.
                    None => {
                        if wakes.recv().is_err() {
                            return AppExit::Success;
                        }
                    }
                }
                if let Some(rest) = FRAME.checked_sub(started.elapsed()) {
                    std::thread::sleep(rest);
                }
            }
        });
    }

    fn finish(&self, app: &mut App) {
        if let Some(mut time) = app.world_mut().get_resource_mut::<Time<Virtual>>() {
            time.set_max_delta(MAX_FRAME_GAP);
        }
    }

    /// Runs every schedule single-threaded, under this loop or any other
    /// that drives the app: a terminal app's systems are small, and Bevy's
    /// multi-threaded executor would wake the task pool for them every
    /// frame, which costs more than they do.
    fn cleanup(&self, app: &mut App) {
        if let Some(mut schedules) = app.world_mut().get_resource_mut::<Schedules>() {
            for (_, schedule) in schedules.iter_mut() {
                schedule.set_executor(SingleThreadedExecutor::new());
            }
        }
    }
}

/// What decides when the next frame is due.
struct NextFrame {
    delayed: QueryState<&'static DelayedCommandQueue>,
    awake: QueryState<&'static KeepAwake>,
    alarms: QueryState<&'static WakeAt, Without<Due>>,
}

impl NextFrame {
    fn new(world: &mut World) -> Self {
        Self {
            delayed: world.query(),
            awake: world.query(),
            alarms: world.query_filtered(),
        }
    }

    /// How long after the start of the frame that just ran the next one is
    /// due, if anything is due. Delayed commands are applied by the end of a frame, so every
    /// queue's deadline is on the clock by then, measured from the frame's
    /// start as Bevy's `Time` is.
    fn after(&mut self, world: &World) -> Option<Duration> {
        let delayed = world.get_resource::<Time>().and_then(|time| {
            self.delayed
                .iter(world)
                .map(|queue| queue.submit_at.saturating_sub(time.elapsed()))
                .min()
        });
        let now = SystemTime::now();
        let alarms = self.alarms.iter(world).map(|alarm| alarm.left(now));
        self.awake
            .iter(world)
            .map(|awake| awake.0)
            .chain(delayed)
            .chain(alarms)
            .min()
    }
}

#[cfg(test)]
mod tests;
