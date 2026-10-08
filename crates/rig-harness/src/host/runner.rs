//! The app's loop without a window. It runs a frame, then sleeps until a
//! [`Wake`] or a second passes. A wake keeps frames coming every 16 ms
//! for a few more, since a woken frame often leaves work for
//! the next: a command's notice, a task that finished just after it woke
//! the loop. Frames never come faster than that, however often a
//! streaming reply wakes the loop.

use std::time::{Duration, Instant};

use bevy_app::PluginsState;
use bevy_app::prelude::*;

use crate::core::calls::Wake;

/// The shortest time between two frames.
const FRAME: Duration = Duration::from_millis(16);
/// Frames run at [`FRAME`] after a wake.
const SETTLE_FRAMES: u32 = 4;
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
            let mut settling = SETTLE_FRAMES;
            loop {
                let started = Instant::now();
                app.update();
                if let Some(exit) = app.should_exit() {
                    return exit;
                }
                let wait = if settling > 0 {
                    settling -= 1;
                    FRAME
                } else {
                    IDLE
                };
                // With its wake replaced, the loop runs at FRAME.
                if wakes.recv_timeout(wait).is_ok() {
                    settling = SETTLE_FRAMES;
                }
                if let Some(rest) = FRAME.checked_sub(started.elapsed()) {
                    std::thread::sleep(rest);
                }
            }
        });
    }
}
