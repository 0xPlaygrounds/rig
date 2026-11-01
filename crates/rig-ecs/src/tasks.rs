//! Work that runs off the main thread: a [`Running`] task on an entity
//! becomes a [`Done`] component when it finishes, through one `poll_tasks`
//! system, so observers of that component carry on; on an effect, such as
//! an open tool call, it becomes the effect's
//! [`Outcome`](super::stage::Outcome) in
//! [`Stage::Arrive`](super::stage::Stage::Arrive). A long-lived source,
//! such as a socket or a thread's channel, is a [`Running::stream`] whose
//! every item becomes a [`Done`] on the entity, in order. Each finished
//! task and each item also calls [`Wake`], so a loop that sleeps while
//! nothing happens runs a frame for it.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_channel::Receiver;
use bevy_ecs::prelude::*;
use bevy_ecs::query::QueryFilter;
use bevy_tasks::futures::check_ready;
use bevy_tasks::{ConditionalSend, ConditionalSendFuture, Task, TaskPool};
use futures::{Stream, StreamExt};

use super::stage::{self, Effect};

/// Frames run right after a woken one (see [`run_settle_frames`]).
const SETTLE_FRAMES: u32 = 4;

/// The most items of a [`Running::stream`] that wait for a frame: a frame
/// that falls behind holds the source back rather than growing memory.
const STREAM_ITEMS: usize = 256;

/// Wakes the app's loop from another thread: a finished call, a streamed
/// fragment, terminal input or a signal. The runner that sleeps between
/// frames inserts its own; the default does nothing, for a loop that never
/// sleeps for long. A windowed app inserts its own too (rig-harness's
/// PLUGINS.md, "A window").
///
/// A woken frame often leaves work for the next, such as a command's
/// notice or a task that finished just after it woke the loop, so the
/// kernel asks for a few more frames after each wake, in `Last`, whatever
/// the loop.
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

/// Runs [`SETTLE_FRAMES`] more frames after a frame some [`Wake`] woke,
/// in `Last`, by asking the loop for each without counting it as a wake.
pub(crate) fn run_settle_frames(wake: Res<Wake>, mut left: Local<u32>) {
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

/// Work running off the main thread for its entity, such as a tool call's
/// work: when
/// it finishes, the kernel replaces it with its output in [`PollTasks`], as a
/// [`Done<T>`] or the component [`Running::spawn_into`] names, and
/// observers of that component carry on. Bevy's own pattern
/// (`examples/async_tasks/async_compute.rs`): one task component for every
/// kind of output, so no plugin registers a system for its own. Dropping
/// it, or the entity, cancels the task; so does the app's exit.
#[derive(Component)]
pub struct Running(Work);

/// What a [`Running`] waits for.
enum Work {
    /// One output, when the task finishes.
    Task(Task<Finish>),
    /// An output per item, as they come, until the task ends.
    Stream {
        items: Receiver<Finish>,
        task: Task<()>,
    },
}

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
    /// output.
    pub fn spawn_into<C: Component + From<T>, T: Send + 'static>(
        pool: &TaskPool,
        wake: &Wake,
        work: impl ConditionalSendFuture<Output = T> + 'static,
    ) -> Self {
        Self::finishing(pool, wake, work, |mut entity, output| {
            entity.insert(C::from(output));
        })
    }

    /// Spawns `work` on `pool` for the effect it goes on, such as an open
    /// tool call's `ToolResult`; it ends as the effect's
    /// [`stage::Outcome`], unless something answered it first.
    pub fn answering<T: Into<stage::Outcome> + Send + 'static>(
        pool: &TaskPool,
        wake: &Wake,
        work: impl ConditionalSendFuture<Output = T> + 'static,
    ) -> Self {
        Self::finishing(pool, wake, work, |mut entity, output| {
            let outcome: stage::Outcome = output.into();
            entity.insert_if_new(outcome);
        })
    }

    /// Spawns a task on `pool` that runs `items` to their end, such as a
    /// socket's messages or a thread's channel: each item is inserted on
    /// the entity as a [`Done<T>`], in order, and wakes the loop; once the
    /// last one is in, `Running` is removed. Several items may come in one
    /// frame, so a reader observes `Insert<Done<T>>`, which sees each. At
    /// most 256 wait; the task waits for room before it reads the next.
    pub fn stream<T: Send + Sync + 'static>(
        pool: &TaskPool,
        wake: &Wake,
        items: impl Stream<Item = T> + ConditionalSend + 'static,
    ) -> Self {
        let (sender, received) = async_channel::bounded(STREAM_ITEMS);
        let wake = wake.clone();
        let task = pool.spawn(async move {
            let mut items = std::pin::pin!(items);
            while let Some(item) = items.next().await {
                let insert = move |mut entity: EntityWorldMut| {
                    entity.insert(Done(item));
                };
                // Dropped with its entity: nothing reads the rest.
                if sender.send(Box::new(insert) as Finish).await.is_err() {
                    break;
                }
                wake.wake();
            }
            wake.wake();
        });
        Self(Work::Stream {
            items: received,
            task,
        })
    }

    /// Spawns `work` on `pool`; `finish` puts its output on the entity.
    pub(crate) fn finishing<T: Send + 'static>(
        pool: &TaskPool,
        wake: &Wake,
        work: impl ConditionalSendFuture<Output = T> + 'static,
        finish: fn(EntityWorldMut, T),
    ) -> Self {
        let wake = wake.clone();
        Self(Work::Task(pool.spawn(async move {
            let output = work.await;
            wake.wake();
            Box::new(move |entity: EntityWorldMut| finish(entity, output)) as Finish
        })))
    }

    /// Whether it is a [`Running::stream`], a source that may never end,
    /// rather than work that finishes.
    pub fn is_stream(&self) -> bool {
        matches!(self.0, Work::Stream { .. })
    }

    /// Cancels the task, once awaited.
    pub(crate) async fn cancel(self) {
        match self.0 {
            Work::Task(task) => {
                task.cancel().await;
            }
            Work::Stream { task, .. } => {
                task.cancel().await;
            }
        }
    }
}

/// A finished task's output, or a [`Running::stream`]'s latest item.
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
pub struct PollTasks;

/// Replaces each finished [`Running`] task with its output, but an
/// effect's, which the kernel takes in [`Stage::Arrive`](stage::Stage::Arrive).
pub(crate) fn poll_tasks(
    tasks: Query<(Entity, &mut Running), Without<Effect>>,
    commands: Commands,
) {
    finish(tasks, commands);
}

/// Replaces each finished effect's [`Running`] task with its outcome, in
/// [`Stage::Arrive`](stage::Stage::Arrive).
pub(crate) fn arrive(tasks: Query<(Entity, &mut Running), With<Effect>>, commands: Commands) {
    finish(tasks, commands);
}

/// Replaces each finished task among `tasks` with its output, and puts a
/// stream's items on its entity in order, removing it once it ended.
fn finish<F: QueryFilter>(mut tasks: Query<(Entity, &mut Running), F>, mut commands: Commands) {
    for (entity, mut running) in &mut tasks {
        match &mut running.bypass_change_detection().0 {
            Work::Task(task) => {
                if let Some(finish) = check_ready(task) {
                    commands.entity(entity).remove::<Running>().queue(finish);
                }
            }
            Work::Stream { items, task } => {
                // Every item was sent before the task ended. Only what came
                // before this look is taken, so a source that keeps sending
                // never holds the frame.
                let ended = check_ready(task).is_some();
                let came = items.len();
                let mut entity = commands.entity(entity);
                for insert in std::iter::from_fn(|| items.try_recv().ok()).take(came) {
                    entity.queue(insert);
                }
                if ended {
                    entity.remove::<Running>();
                }
            }
        }
    }
}
