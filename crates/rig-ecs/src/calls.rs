//! Work that runs off the main thread: a [`Running`] task on a call entity
//! becomes a [`Done`] component when it finishes, or a tool call's
//! [`ToolOutput`](super::tools::ToolOutput), through one `poll_calls`
//! system, so observers of that component carry the turn on.
//! Each finished task also calls [`Wake`], so a loop that sleeps while
//! nothing happens runs a frame for it.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;
use bevy_tasks::futures::check_ready;
use bevy_tasks::{ConditionalSendFuture, Task, TaskPool};

/// Frames run right after a woken one (see [`settle`]).
const SETTLE_FRAMES: u32 = 4;

/// The longest gap between two frames that Bevy's virtual clock counts in
/// full, for a loop that sleeps while idle (a frame a second at least, in
/// rig-harness). Bevy's default, 250 ms, suits a game's steady frames; across
/// one-second frames it would make a retry's wait four times as long. A
/// longer gap, such as a suspended machine, is still cut short, as Bevy
/// intends: its fixed-step loop runs once for every 15.6 ms counted, all in
/// the frame after the gap.
pub const MAX_FRAME_GAP: Duration = Duration::from_secs(60);

/// Wakes the app's loop from another thread: a finished call, a streamed
/// fragment, terminal input or a signal. The runner that sleeps between
/// frames inserts its own; the default does nothing, for a loop that never
/// sleeps for long. A windowed app inserts
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
}

impl Default for Wake {
    fn default() -> Self {
        Self::new(|| {})
    }
}

/// Keeps a loop that sleeps between frames running a frame at least this
/// often while the entity exists: a spinner, or a panel that polls. Time
/// itself is Bevy's: a system paced with
/// `.run_if(on_real_timer(interval))` runs on time while a `KeepAwake` of
/// that interval lives, and a one-off wait is a delayed command
/// (`commands.delayed().duration(wait)`), whose deadline the loop wakes for
/// on its own.
///
/// ```
/// use std::time::Duration;
/// use rig_ecs::prelude::*;
///
/// const STEP: Duration = Duration::from_millis(200);
///
/// fn tick() {}
///
/// let mut app = App::new();
/// // `tick` runs every step while this entity lives; once it is despawned
/// // the loop sleeps until there is work.
/// app.world_mut().spawn((Name::new("spinner"), KeepAwake(STEP)));
/// app.add_systems(Update, tick.run_if(on_real_timer(STEP)));
/// ```
#[derive(Component, Reflect, Clone, Copy, Debug)]
#[reflect(Component, Clone, Debug)]
pub struct KeepAwake(pub Duration);

/// Runs four more frames after a frame some [`Wake`] woke,
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

/// What a finished task does to its entity: inserts its output.
type Finish = Box<dyn FnOnce(EntityWorldMut) + Send>;

/// Work running off the main thread for its entity, such as a call: when
/// it finishes, [`poll_calls`] replaces it with its output, as a
/// [`Done<T>`] or the component [`Running::spawn_into`] names, and
/// observers of that component carry on. Bevy's own pattern
/// (`examples/async_tasks/async_compute.rs`): one task component for every
/// kind of output, so no plugin registers a system for its own. Dropping
/// it, or the entity, cancels the task; so does the app's exit.
#[derive(Component)]
pub struct Running(Task<Finish>);

impl Running {
    /// Spawns `work` on `pool`; it ends as a [`Done<T>`] on the entity, and
    /// wakes the loop.
    pub fn spawn<T: Send + Sync + 'static>(
        pool: &TaskPool,
        wake: &Wake,
        work: impl ConditionalSendFuture<Output = T> + 'static,
    ) -> Self {
        Self::spawn_into::<Done<T>, T>(pool, wake, work)
    }

    /// [`Running::spawn`], ending as the component `C` made from the
    /// output, such as a tool call's
    /// [`ToolOutput`](super::tools::ToolOutput).
    pub fn spawn_into<C: Component + From<T>, T: Send + 'static>(
        pool: &TaskPool,
        wake: &Wake,
        work: impl ConditionalSendFuture<Output = T> + 'static,
    ) -> Self {
        let wake = wake.clone();
        Self(pool.spawn(async move {
            let output = work.await;
            wake.wake();
            Box::new(move |mut entity: EntityWorldMut| {
                entity.insert(C::from(output));
            }) as Finish
        }))
    }

    /// Takes the task off, to be cancelled: `task.cancel().await`.
    pub(crate) fn into_task(self) -> Task<Finish> {
        self.0
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

/// The system polling running work, in `Update`: a system that reads what
/// finished this frame runs after it.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PollCalls;

/// Replaces each finished [`Running`] task with its output.
pub fn poll_calls(mut calls: Query<(Entity, &mut Running)>, mut commands: Commands) {
    for (entity, mut running) in &mut calls {
        if let Some(finish) = check_ready(&mut running.bypass_change_detection().0) {
            commands.entity(entity).remove::<Running>().queue(finish);
        }
    }
}
