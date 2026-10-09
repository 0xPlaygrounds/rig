//! The app's loop without a window. It runs a frame, then sleeps until a
//! [`Wake`] or a second passes. Frames never come faster than every 16 ms,
//! however often a streaming reply wakes the loop; the frames after a wake
//! come from [`settle`](crate::calls::settle), which works the same under
//! any loop.

use std::time::{Duration, Instant};

use bevy_app::PluginsState;
use bevy_app::prelude::*;

use crate::calls::Wake;

/// The shortest time between two frames.
const FRAME: Duration = Duration::from_millis(16);
/// The longest sleep without a wake.
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
            loop {
                let started = Instant::now();
                app.update();
                if let Some(exit) = app.should_exit() {
                    return exit;
                }
                // With its wake replaced, the loop runs every IDLE.
                wakes.recv_timeout(IDLE).ok();
                if let Some(rest) = FRAME.checked_sub(started.elapsed()) {
                    std::thread::sleep(rest);
                }
            }
        });
    }
}
