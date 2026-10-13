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
//! [`Workspace::files`] lists, leaving out ignored and hidden files and
//! `target`, unless the watch is [`all`](FileWatch::all).
//!
//! ```no_run
//! use rig_harness::prelude::*;
//!
//! fn watch_templates(mut commands: Commands, root: Res<ProjectRoot>) {
//!     commands.spawn(FileWatch::tree(root.0.join(".rig/templates")));
//! }
//!
//! fn reload(mut changes: MessageReader<FileChanges>) {
//!     for change in changes.read() {
//!         for path in &change.paths {
//!             println!("{} changed", path.display());
//!         }
//!     }
//! }
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use bevy_app::prelude::*;
use bevy_ecs::lifecycle::HookContext;
use bevy_ecs::prelude::*;
use bevy_ecs::world::DeferredWorld;
use bevy_log::warn;
use bevy_tasks::futures_lite::{Stream, StreamExt as _, stream};
use bevy_tasks::{IoTaskPool, TaskPool};
use rig_ecs::calls::{Done, Running, Wake};
use rig_tools::workspace::{Watch, Workspace};

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
    /// Whether ignored and hidden files and `target` are reported too.
    /// The first [`FileChanges`] still lists only what
    /// [`Workspace::files`] lists.
    pub all: bool,
}

impl FileWatch {
    /// A watch of `path` alone, or of the files right in it.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            recursive: false,
            all: false,
        }
    }

    /// A watch of every file under `path`.
    pub fn tree(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            recursive: true,
            all: false,
        }
    }

    /// The same watch, reporting ignored and hidden files and `target`
    /// too.
    pub fn all(mut self) -> Self {
        self.all = true;
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
struct Looked(Vec<PathBuf>);

/// Starts the watch through the project's workspace.
fn arm(mut world: DeferredWorld, context: HookContext) {
    let Some(watch) = world.get::<FileWatch>(context.entity).cloned() else {
        return;
    };
    let workspace = world.get_resource::<ProjectWorkspace>().cloned();
    let workspace = workspace.unwrap_or_default().0;
    let wake = world.get_resource::<Wake>().cloned().unwrap_or_default();
    let pool = IoTaskPool::get_or_init(TaskPool::default);
    let running = Running::stream(pool, &wake, looks(workspace, watch));
    world.commands().entity(context.entity).insert(running);
}

/// Stops the watch.
fn disarm(mut world: DeferredWorld, context: HookContext) {
    world
        .commands()
        .entity(context.entity)
        .try_remove::<(Running, Done<Looked>)>();
}

/// Writes what a watch saw.
fn changed(
    looked: On<Insert<Done<Looked>>>,
    seen: Query<&Done<Looked>, With<FileWatch>>,
    mut changes: MessageWriter<FileChanges>,
) {
    if let Ok(Done(Looked(paths))) = seen.get(looked.entity) {
        changes.write(FileChanges {
            watch: looked.entity,
            paths: paths.clone(),
        });
    }
}

/// Where a watch is.
enum Look {
    /// Not armed yet.
    Arm(Arc<dyn Workspace>, FileWatch),
    /// Armed: the batches to come.
    Watching(Watch),
}

/// The files `watch` covers at first, then each batch of changes the
/// workspace reports.
fn looks(workspace: Arc<dyn Workspace>, watch: FileWatch) -> impl Stream<Item = Looked> {
    stream::unfold(Some(Look::Arm(workspace, watch)), |look| async move {
        match look? {
            Look::Arm(workspace, watch) => {
                // Armed before listing, so nothing made in between is missed.
                let armed = workspace
                    .clone()
                    .watch(&watch.path, watch.recursive, watch.all);
                let there = there(&*workspace, &watch);
                let next = match armed {
                    Ok(armed) => Some(Look::Watching(armed)),
                    Err(error) => {
                        warn!("cannot watch {}: {error}", watch.path.display());
                        None
                    }
                };
                Some((Looked(there), next))
            }
            Look::Watching(mut armed) => {
                let paths = armed.next().await?;
                Some((Looked(paths), Some(Look::Watching(armed))))
            }
        }
    })
}

/// The files `watch` covers now, in path order.
fn there(workspace: &dyn Workspace, watch: &FileWatch) -> Vec<PathBuf> {
    let root = watch.path.as_path();
    let mut files = workspace.files(root, None, false).unwrap_or_default();
    files.retain(|path| watch.recursive || path == root || path.parent() == Some(root));
    files.sort();
    files
}

#[cfg(test)]
mod tests;
