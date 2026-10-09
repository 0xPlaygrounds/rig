//! Work that runs off the main thread: a [`Running`] task on a call entity
//! becomes a [`Done`] component when it finishes, or a tool call's
//! [`ToolOutput`](super::tools::ToolOutput), through one generic
//! `poll_calls` system, so observers of that component carry the turn on.
//! Each finished task also calls [`Wake`], so a loop that sleeps while
//! nothing happens runs a frame for it.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use bevy_ecs::prelude::*;
use bevy_tasks::futures::check_ready;
use bevy_tasks::{AsyncComputeTaskPool, ConditionalSendFuture, Task, TaskPool};

/// Frames run right after a woken one (see [`settle`]).
const SETTLE_FRAMES: u32 = 4;

/// Wakes the app's loop from another thread: a finished call, a streamed
/// fragment, terminal input, a signal or a timer ([`Wake::after`]). The
/// runner that sleeps between frames inserts its own; the default does
/// nothing, for a loop that never sleeps for long. A windowed app inserts
/// one that sends winit's `WinitUserEvent::WakeUp` through bevy_winit's
/// `EventLoopProxyWrapper`, in its plugin's `finish`, so the window's loop
/// runs a frame for agent activity (the recipe is in rig-harness's docs).
///
/// A woken frame often leaves work for the next, such as a command's
/// notice or a task that finished just after it woke the loop, so
/// [`settle`] asks for a few more frames after each wake, whatever the loop.
#[derive(Resource, Clone)]
pub struct Wake {
    wake: Arc<dyn Fn() + Send + Sync>,
    woken: Arc<AtomicBool>,
}

impl Wake {
    /// A wake that calls `wake`, which asks the loop for a frame from any
    /// thread.
    pub fn new(wake: impl Fn() + Send + Sync + 'static) -> Self {
        Self {
            wake: Arc::new(wake),
            woken: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Asks the loop for a frame soon.
    pub fn wake(&self) {
        self.woken.store(true, Ordering::Release);
        (self.wake)();
    }

    /// Asks the loop for a frame once `delay` has passed, on
    /// futures-timer's timer thread: a plugin animates or polls without a
    /// thread of its own. Dropping the task cancels the wake; detach it to
    /// keep it. For a steady rate, gate a system with [`every`](crate::timer::every).
    /// The frame it brings is not [`settle`]d: a timer leaves no work for
    /// the next frame.
    pub fn after(&self, delay: Duration) -> Task<()> {
        let wake = Arc::clone(&self.wake);
        AsyncComputeTaskPool::get_or_init(TaskPool::default).spawn(async move {
            futures_timer::Delay::new(delay).await;
            wake();
        })
    }
}

impl Default for Wake {
    fn default() -> Self {
        Self::new(|| {})
    }
}

/// Runs [`SETTLE_FRAMES`] more frames after a frame some [`Wake`] woke,
/// in `Last`, by asking the loop for each without counting it as a wake.
pub fn settle(wake: Res<Wake>, mut left: Local<u32>) {
    if wake.woken.swap(false, Ordering::AcqRel) {
        *left = SETTLE_FRAMES;
    }
    if *left > 0 {
        *left -= 1;
        (wake.wake)();
    }
}

/// A call's task. Dropping it, or the entity, cancels the task.
#[derive(Component)]
pub struct Running<T: Send + Sync + 'static>(pub Task<T>);

impl<T: Send + Sync + 'static> Running<T> {
    /// Spawns `work` on `pool`; its end wakes the loop.
    pub fn spawn(
        pool: &TaskPool,
        wake: &Wake,
        work: impl ConditionalSendFuture<Output = T> + 'static,
    ) -> Self {
        let wake = wake.clone();
        Self(pool.spawn(async move {
            let output = work.await;
            wake.wake();
            output
        }))
    }
}

/// A finished call's output.
#[derive(Component)]
pub struct Done<T: Send + Sync + 'static>(pub T);

impl<T: Send + Sync + 'static> From<T> for Done<T> {
    fn from(output: T) -> Self {
        Self(output)
    }
}

/// Replaces each finished [`Running<T>`] task with its output as the
/// component `C`: [`Done<T>`] for most calls, a
/// [`ToolOutput`](super::tools::ToolOutput) for tool calls.
pub fn poll_calls<T: Send + Sync + 'static, C: Component + From<T>>(
    mut calls: Query<(Entity, &mut Running<T>)>,
    mut commands: Commands,
) {
    for (entity, mut running) in &mut calls {
        if let Some(output) = check_ready(&mut running.bypass_change_detection().0) {
            commands
                .entity(entity)
                .remove::<Running<T>>()
                .insert(C::from(output));
        }
    }
}
