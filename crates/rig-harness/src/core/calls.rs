//! Work that runs off the main thread: a [`Running`] task on a call entity
//! becomes a [`Done`] component when it finishes, or a tool call's
//! [`ToolOutput`](super::tools::ToolOutput), through one generic
//! `poll_calls` system, so observers of that component carry the turn on.
//! Each finished task also calls [`Wake`], so a loop that sleeps while
//! nothing happens runs a frame for it.

use std::sync::Arc;

use bevy_ecs::prelude::*;
use bevy_tasks::futures::check_ready;
use bevy_tasks::{Task, TaskPool};

/// Wakes the app's loop from another thread: a finished call, a streamed
/// fragment, terminal input or a signal. The runner that sleeps between
/// frames inserts its own; the default does nothing, for a loop that never
/// sleeps for long. A windowing plugin inserts one that sends winit's
/// `WinitUserEvent::WakeUp`.
#[derive(Resource, Clone)]
pub struct Wake(Arc<dyn Fn() + Send + Sync>);

impl Wake {
    /// A wake that calls `wake`.
    pub fn new(wake: impl Fn() + Send + Sync + 'static) -> Self {
        Self(Arc::new(wake))
    }

    /// Asks the loop for a frame soon.
    pub fn wake(&self) {
        (self.0)();
    }
}

impl Default for Wake {
    fn default() -> Self {
        Self::new(|| {})
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
        work: impl Future<Output = T> + Send + 'static,
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
