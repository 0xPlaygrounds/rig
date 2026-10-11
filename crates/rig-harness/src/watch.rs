//! Watching files. An entity with a [`FileWatch`] on a file or a
//! directory gets a [`FileChanges`] message naming the files there when it
//! was armed, then one naming what was created, changed or removed under
//! it, within about a second, and the loop is woken for each; despawning
//! it, or removing the `FileWatch`, stops the watch. A plugin that reloads
//! templates or skills from a directory watches it, and reads them on its
//! first `FileChanges`, so no file made in between is missed.
//!
//! The watch is a [`Running::stream`] on its entity, so put a `FileWatch`
//! on an entity of its own. It looks at the watched path's files (their
//! size and change time) each second, off the app's thread, and at most
//! [`MAX_FILES`] files.
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

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, SystemTime};

use bevy_app::prelude::*;
use bevy_ecs::lifecycle::HookContext;
use bevy_ecs::prelude::*;
use bevy_ecs::world::DeferredWorld;
use bevy_tasks::futures_lite::{Stream, stream};
use bevy_tasks::{IoTaskPool, TaskPool};
use futures_timer::Delay;
use rig_ecs::calls::{Done, Running, Wake};

/// How often a watch looks.
const EVERY: Duration = Duration::from_secs(1);

/// The most files one watch looks at; what is beyond goes unseen.
pub const MAX_FILES: usize = 10_000;

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
}

impl FileWatch {
    /// A watch of `path` alone, or of the files right in it.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            recursive: false,
        }
    }

    /// A watch of every file under `path`.
    pub fn tree(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            recursive: true,
        }
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

/// What a watch saw in one look.
struct Looked(Vec<PathBuf>);

/// Starts the watch's looks.
fn arm(mut world: DeferredWorld, context: HookContext) {
    let Some(watch) = world.get::<FileWatch>(context.entity).cloned() else {
        return;
    };
    let wake = world.get_resource::<Wake>().cloned().unwrap_or_default();
    let pool = IoTaskPool::get_or_init(TaskPool::default);
    let running = Running::stream(pool, &wake, looks(watch));
    world.commands().entity(context.entity).insert(running);
}

/// Stops the watch's looks.
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

/// The files `watch` covers at first, then each [`EVERY`] what changed,
/// when something did.
fn looks(watch: FileWatch) -> impl Stream<Item = Looked> {
    stream::unfold(None, move |before: Option<Files>| {
        let watch = watch.clone();
        async move {
            let Some(before) = before else {
                let files = files(&watch);
                return Some((Looked(files.keys().cloned().collect()), Some(files)));
            };
            let mut before = before;
            loop {
                Delay::new(EVERY).await;
                let files = files(&watch);
                let paths = differences(&before, &files);
                before = files;
                if !paths.is_empty() {
                    return Some((Looked(paths), Some(before)));
                }
            }
        }
    })
}

/// Each file's size and change time.
type Files = BTreeMap<PathBuf, (u64, Option<SystemTime>)>;

/// The files `watch` covers, at most [`MAX_FILES`].
fn files(watch: &FileWatch) -> Files {
    let mut files = Files::new();
    let mut directories = vec![(watch.path.clone(), true)];
    while let Some((path, top)) = directories.pop() {
        let Ok(metadata) = fs::metadata(&path) else {
            continue;
        };
        if !metadata.is_dir() {
            files.insert(path, (metadata.len(), metadata.modified().ok()));
        } else if top || watch.recursive {
            let entries = fs::read_dir(&path).into_iter().flatten().flatten();
            directories.extend(entries.map(|entry| (entry.path(), false)));
        }
        if files.len() >= MAX_FILES {
            break;
        }
    }
    files
}

/// The paths created, changed or removed from `before` to `after`.
fn differences(before: &Files, after: &Files) -> Vec<PathBuf> {
    let changed = after
        .iter()
        .filter(|(path, now)| before.get(path.as_path()) != Some(now))
        .map(|(path, _)| path);
    let removed = before
        .keys()
        .filter(|path| !after.contains_key(path.as_path()));
    let mut paths: Vec<PathBuf> = changed.chain(removed).cloned().collect();
    paths.sort();
    paths
}

#[cfg(test)]
mod tests;
