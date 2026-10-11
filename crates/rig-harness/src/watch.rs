//! Watching files. An entity with a [`FileWatch`] on a file or a
//! directory gets a [`FileChanges`] message naming what was created,
//! changed or removed under it, within about a second, and the loop is
//! woken for it; despawning it, or removing the `FileWatch`, stops the
//! watch. A plugin that reloads templates or skills from a directory
//! watches it instead of reading it again before every use.
//!
//! One thread looks at every watched path's files (their size and change
//! time) each second, off the app's thread, and at most
//! [`MAX_FILES`] files per watch.
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

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, SystemTime};

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use crossbeam_channel::{Receiver, Sender};
use rig_ecs::calls::Wake;

/// How often the watcher looks.
const EVERY: Duration = Duration::from_secs(1);

/// The most files one watch looks at; what is beyond goes unseen.
pub const MAX_FILES: usize = 10_000;

/// Watches `path`: the file, or the directory's files (with `recursive`,
/// the files of every directory under it).
#[derive(Component, Clone, Debug, PartialEq, Eq)]
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

/// What changed under a [`FileWatch`] since it was last looked at.
#[derive(Message, Clone, Debug, PartialEq, Eq)]
pub struct FileChanges {
    /// The entity with the watch.
    pub watch: Entity,
    /// The files created, changed or removed, in path order.
    pub paths: Vec<PathBuf>,
}

/// Runs the watcher.
pub struct WatchPlugin;

impl Plugin for WatchPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<FileChanges>()
            .add_systems(PreUpdate, watch_files);
    }
}

/// The watches the thread looks at, by entity.
type Watches = HashMap<Entity, FileWatch>;

/// The watcher thread's side in the app: what it watches, and what it saw.
#[derive(Resource)]
struct Watcher {
    watches: Arc<Mutex<Watches>>,
    changes: Receiver<FileChanges>,
}

impl Watcher {
    /// Starts the thread, which runs while the watcher lives.
    fn start(wake: Option<Wake>) -> Self {
        let watches = Arc::new(Mutex::new(Watches::new()));
        let (sender, changes) = crossbeam_channel::unbounded();
        let shared = Arc::downgrade(&watches);
        std::thread::Builder::new()
            .name("rig-watch".to_owned())
            .spawn(move || look(&shared, &sender, wake.as_ref()))
            .ok();
        Self { watches, changes }
    }
}

impl Watcher {
    /// Tells the thread about `watches` and that `gone` went.
    fn update(&self, watches: &Query<(Entity, Ref<FileWatch>)>, gone: Vec<Entity>) {
        let mut shared = self
            .watches
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        for entity in gone {
            shared.remove(&entity);
        }
        for (entity, watch) in watches {
            shared.insert(entity, FileWatch::clone(&watch));
        }
    }
}

/// Tells the thread which watches exist, and passes on what it saw.
fn watch_files(
    watches: Query<(Entity, Ref<FileWatch>)>,
    mut gone: RemovedComponents<FileWatch>,
    watcher: Option<Res<Watcher>>,
    wake: Option<Res<Wake>>,
    mut changes: MessageWriter<FileChanges>,
    mut commands: Commands,
) {
    let gone: Vec<Entity> = gone.read().collect();
    let Some(watcher) = watcher else {
        if !watches.is_empty() {
            let watcher = Watcher::start(wake.map(|wake| wake.clone()));
            watcher.update(&watches, gone);
            commands.insert_resource(watcher);
        }
        return;
    };
    if !gone.is_empty() || watches.iter().any(|(_, watch)| watch.is_changed()) {
        watcher.update(&watches, gone);
    }
    changes.write_batch(watcher.changes.try_iter());
}

/// The watcher thread: looks at every watch each [`EVERY`] until the app
/// drops the watches.
fn look(watches: &Weak<Mutex<Watches>>, changes: &Sender<FileChanges>, wake: Option<&Wake>) {
    let mut seen: HashMap<Entity, (FileWatch, Files)> = HashMap::new();
    while let Some(watches) = watches.upgrade() {
        let now: Watches = watches
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .clone();
        drop(watches);
        seen.retain(|entity, (watch, _)| now.get(entity) == Some(watch));
        let mut woke = false;
        for (entity, watch) in now {
            let files = files(&watch);
            let Some((_, before)) = seen.get(&entity) else {
                seen.insert(entity, (watch, files));
                continue;
            };
            let paths = differences(before, &files);
            seen.insert(entity, (watch, files));
            if !paths.is_empty() {
                changes
                    .send(FileChanges {
                        watch: entity,
                        paths,
                    })
                    .ok();
                woke = true;
            }
        }
        if let Some(wake) = wake.filter(|_| woke) {
            wake.wake();
        }
        std::thread::sleep(EVERY);
    }
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
