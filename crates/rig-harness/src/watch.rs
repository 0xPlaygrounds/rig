//! Watching files. An entity with a [`FileWatch`] on a file or a
//! directory gets a [`FileChanges`] message for each [`WatchEvent`] of its
//! watch: a snapshot of the files there once armed, each batch of files
//! created, changed or removed under it, a listing that failed, and the
//! watch's end (also when it could not start), and the loop is woken for
//! each; despawning it, or removing the `FileWatch`, stops the watch. A
//! plugin that reloads templates or skills from a directory watches it,
//! and reads them on its snapshot, so no file made in between is missed.
//!
//! The watch is a [`Running::stream`] on its entity, so put a `FileWatch`
//! on an entity of its own. It watches through the project's workspace
//! ([`ProjectWorkspace`], this machine when there is none): on this
//! machine the system tells it of changes; another workspace may look
//! every second ([`Workspace::watch`]). It reports the files
//! [`Workspace::files`] lists with the watch's [`Listing`] (by default
//! leaving out ignored and hidden files and `target`). The plugin guide's
//! [files section](crate::plugin_guide#the-workspace-files-and-child-processes)
//! has an example.

use std::path::PathBuf;
use std::sync::Arc;

use bevy_app::prelude::*;
use bevy_ecs::lifecycle::HookContext;
use bevy_ecs::prelude::*;
use bevy_ecs::world::DeferredWorld;
use bevy_tasks::futures_lite::{Stream, StreamExt as _, stream};
use bevy_tasks::{IoTaskPool, TaskPool};
use rig_ecs::tasks::{Done, Running, Wake};
pub use rig_tools::workspace::WatchEvent;
use rig_tools::workspace::{Listing, Watch, Workspace};

use crate::workspace::ProjectWorkspace;

/// Watches `path`: the file, or the directory's files (with `recursive`,
/// the files of every directory under it). Inserting another arms it
/// again.
#[derive(Component, Clone, Debug, PartialEq, Eq)]
#[component(immutable, on_insert = arm, on_discard = disarm)]
pub struct FileWatch {
    /// The file or directory.
    pub path: PathBuf,
    /// Whether directories under it are watched too.
    pub recursive: bool,
    /// Which files are reported, at first and as they change.
    pub listing: Listing,
}

impl FileWatch {
    /// A watch of `path` alone, or of the files right in it.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            recursive: false,
            listing: Listing::Visible,
        }
    }

    /// A watch of every file under `path`.
    pub fn tree(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            recursive: true,
            listing: Listing::Visible,
        }
    }

    /// The same watch, reporting the files `listing` keeps.
    pub fn listing(mut self, listing: Listing) -> Self {
        self.listing = listing;
        self
    }
}

/// What a [`FileWatch`] told: the files there once armed, then what
/// changed, a listing that failed, and its end.
#[derive(Message, Clone, Debug)]
pub struct FileChanges {
    /// The entity with the watch.
    pub watch: Entity,
    /// What it told.
    pub event: WatchEvent,
}

/// Passes on what the watches see.
pub struct WatchPlugin;

impl Plugin for WatchPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<FileChanges>().add_observer(changed);
    }
}

/// What a watch told.
struct Seen(WatchEvent);

/// Starts the watch through the project's workspace.
fn arm(mut world: DeferredWorld, context: HookContext) {
    let Some(watch) = world.get::<FileWatch>(context.entity).cloned() else {
        return;
    };
    let workspace = world.get_resource::<ProjectWorkspace>().cloned();
    let workspace = workspace.unwrap_or_default().0;
    let wake = world.get_resource::<Wake>().cloned().unwrap_or_default();
    let pool = IoTaskPool::get_or_init(TaskPool::default);
    let running = Running::stream(pool, &wake, watch_stream(workspace, watch));
    world.commands().entity(context.entity).insert(running);
}

/// Stops the watch.
fn disarm(mut world: DeferredWorld, context: HookContext) {
    world
        .commands()
        .entity(context.entity)
        .try_remove::<(Running, Done<Seen>)>();
}

/// Writes what a watch saw.
fn changed(
    inserted: On<Insert<Done<Seen>>>,
    seen: Query<&Done<Seen>, With<FileWatch>>,
    mut changes: MessageWriter<FileChanges>,
) {
    if let Ok(Done(Seen(event))) = seen.get(inserted.entity) {
        changes.write(FileChanges {
            watch: inserted.entity,
            event: event.clone(),
        });
    }
}

/// Where a watch is.
enum WatchState {
    /// Not armed yet.
    Unarmed(Arc<dyn Workspace>, FileWatch),
    /// Armed: the events to come.
    Armed(Watch),
    /// It could not start, which was told: it ends.
    Failed,
}

/// The events of `watch` through `workspace`; one that cannot start tells
/// why, then that it closed.
fn watch_stream(workspace: Arc<dyn Workspace>, watch: FileWatch) -> impl Stream<Item = Seen> {
    stream::unfold(
        Some(WatchState::Unarmed(workspace, watch)),
        |state| async move {
            let mut armed = match state? {
                WatchState::Unarmed(workspace, watch) => {
                    match workspace.watch(&watch.path, watch.recursive, watch.listing) {
                        Ok(armed) => armed,
                        Err(error) => {
                            let why = std::io::Error::new(
                                error.kind(),
                                format!("cannot watch {}: {error}", watch.path.display()),
                            );
                            return Some((
                                Seen(WatchEvent::Unavailable(Arc::new(why))),
                                Some(WatchState::Failed),
                            ));
                        }
                    }
                }
                WatchState::Armed(armed) => armed,
                WatchState::Failed => return Some((Seen(WatchEvent::Closed), None)),
            };
            let event = armed.next().await?;
            Some((Seen(event), Some(WatchState::Armed(armed))))
        },
    )
}

#[cfg(test)]
mod tests;
