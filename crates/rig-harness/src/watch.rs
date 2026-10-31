//! Watching files. An entity with a [`FileWatch`] on a file or a
//! directory gets a [`FileChanges`] message naming the files there when it
//! was armed, then one for each batch of files created, changed or removed
//! under it, as the system reports them, and the loop is woken for each;
//! despawning it, or removing the `FileWatch`, stops the watch. A plugin
//! that reloads templates or skills from a directory watches it, and reads
//! them on its first `FileChanges`, so no file made in between is missed.
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
use bevy_log::warn;
use bevy_tasks::futures_lite::{Stream, StreamExt as _, stream};
use bevy_tasks::{IoTaskPool, TaskPool};
use rig_ecs::tasks::{Done, Running, Wake};
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

/// What changed under a [`FileWatch`]: first the files there when it was
/// armed, then what changed since it last looked.
#[derive(Message, Clone, Debug, PartialEq, Eq)]
pub struct FileChanges {
    /// The entity with the watch.
    pub watch: Entity,
    /// The files there at first, then the files created, changed or
    /// removed, in path order.
    pub paths: Vec<PathBuf>,
}

/// Passes on what the watches see.
pub struct WatchPlugin;

impl Plugin for WatchPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<FileChanges>().add_observer(changed);
    }
}

/// What a watch saw: the files there, then a batch of changes.
struct Seen(Vec<PathBuf>);

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
    if let Ok(Done(Seen(paths))) = seen.get(inserted.entity) {
        changes.write(FileChanges {
            watch: inserted.entity,
            paths: paths.clone(),
        });
    }
}

/// Where a watch is.
enum WatchState {
    /// Not armed yet.
    Unarmed(Arc<dyn Workspace>, FileWatch),
    /// Armed: the batches to come.
    Armed(Watch),
}

/// The files `watch` covers at first, then each batch of changes the
/// workspace reports.
fn watch_stream(workspace: Arc<dyn Workspace>, watch: FileWatch) -> impl Stream<Item = Seen> {
    stream::unfold(
        Some(WatchState::Unarmed(workspace, watch)),
        |state| async move {
            match state? {
                WatchState::Unarmed(workspace, watch) => {
                    // Armed before listing, so nothing made in between is missed.
                    let armed =
                        workspace
                            .clone()
                            .watch(&watch.path, watch.recursive, watch.listing);
                    let files = files_now(&*workspace, &watch);
                    let next = match armed {
                        Ok(armed) => Some(WatchState::Armed(armed)),
                        Err(error) => {
                            warn!("cannot watch {}: {error}", watch.path.display());
                            None
                        }
                    };
                    Some((Seen(files), next))
                }
                WatchState::Armed(mut armed) => {
                    let paths = armed.next().await?;
                    Some((Seen(paths), Some(WatchState::Armed(armed))))
                }
            }
        },
    )
}

/// The files `watch` covers now, in path order.
fn files_now(workspace: &dyn Workspace, watch: &FileWatch) -> Vec<PathBuf> {
    let root = watch.path.as_path();
    let mut files = workspace
        .files(root, None, watch.listing)
        .map(|listed| listed.files)
        .unwrap_or_default();
    files.retain(|path| watch.recursive || path == root || path.parent() == Some(root));
    files.sort();
    files
}

#[cfg(test)]
mod tests;
