use std::ffi::{OsStr, OsString};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use bevy_app::TaskPoolPlugin;
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use rig_ecs::AgentPlugin;
use rig_tools::process::{Launcher, Process, Spawn};
use rig_tools::workspace::{Contents, Listed, Listing, Local, Stat, WatchEvent, Workspace};

use crate::workspace::ProjectWorkspace;

use super::{FileChanges, FileWatch, WatchPlugin};

/// What the app's watches saw, frame after frame.
#[derive(Resource, Default)]
struct Seen(Vec<FileChanges>);

/// What each change of `watch` told, by kind, with its paths.
fn told(app: &App, watch: Entity) -> Vec<(&'static str, Vec<PathBuf>)> {
    let changes = app.world().resource::<Seen>().0.iter();
    let kind = |event: &WatchEvent| match event {
        WatchEvent::Snapshot(_) => "snapshot",
        WatchEvent::Changed(_) => "changed",
        WatchEvent::Unavailable(_) => "unavailable",
        WatchEvent::Closed => "closed",
    };
    changes
        .filter(|change| change.watch == watch)
        .map(|change| (kind(&change.event), change.event.paths().to_vec()))
        .collect()
}

/// Runs frames until the watches saw `count` changes, or five seconds
/// passed.
fn run_until_seen(app: &mut App, count: usize) {
    let started = Instant::now();
    while app.world().resource::<Seen>().0.len() < count
        && started.elapsed() < Duration::from_secs(5)
    {
        app.update();
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn a_watch_lists_the_files_there_then_names_the_files_made_and_stops_when_despawned() {
    let root = std::env::temp_dir().join(format!("rig-watch-{}", std::process::id()));
    std::fs::create_dir_all(root.join("nested")).ok();
    let there = root.join("there.md");
    std::fs::write(&there, "before").ok();
    let mut app = App::new();
    app.add_plugins((TaskPoolPlugin::default(), AgentPlugin, WatchPlugin))
        .init_resource::<Seen>()
        .add_systems(
            Last,
            |mut changes: MessageReader<FileChanges>, mut seen: ResMut<Seen>| {
                seen.0.extend(changes.read().cloned());
            },
        );
    let watch = app.world_mut().spawn(FileWatch::tree(&root)).id();
    run_until_seen(&mut app, 1);
    let made = root.join("nested/new.md");
    std::fs::write(&made, "hello").ok();
    run_until_seen(&mut app, 2);
    assert_eq!(
        told(&app, watch),
        [
            ("snapshot", vec![there.clone()]),
            ("changed", vec![made.clone()])
        ]
    );
    app.world_mut().despawn(watch);
    app.update();
    std::fs::remove_file(&made).ok();
    std::thread::sleep(Duration::from_millis(1200));
    app.update();
    assert_eq!(app.world().resource::<Seen>().0.len(), 2);
    std::fs::remove_dir_all(&root).ok();
}

/// An app that collects what its watches tell, with `workspace` as the
/// project's.
fn watching_app(workspace: Option<Arc<dyn Workspace>>) -> App {
    let mut app = App::new();
    app.add_plugins((TaskPoolPlugin::default(), AgentPlugin, WatchPlugin))
        .init_resource::<Seen>()
        .add_systems(
            Last,
            |mut changes: MessageReader<FileChanges>, mut seen: ResMut<Seen>| {
                seen.0.extend(changes.read().cloned());
            },
        );
    if let Some(workspace) = workspace {
        app.insert_resource(ProjectWorkspace(workspace));
    }
    app
}

/// Runs frames for `wait`.
fn run_for(app: &mut App, wait: Duration) {
    let started = Instant::now();
    while started.elapsed() < wait {
        app.update();
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// A watch that cannot start never looks like a healthy empty directory:
/// it tells why and keeps trying, and once the directory opens it tells
/// the files there, then what changes.
#[cfg(unix)]
#[test]
fn a_watch_that_cannot_start_is_not_an_empty_directory() {
    use std::os::unix::fs::PermissionsExt as _;
    let root = std::env::temp_dir().join(format!("rig-watch-locked-{}", std::process::id()));
    std::fs::create_dir_all(&root).ok();
    let root = std::fs::canonicalize(&root).unwrap_or(root);
    let there = root.join("there.md");
    std::fs::write(&there, "x").ok();
    std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o000)).ok();
    // Whoever may read it anyway (root) can watch it.
    let locked = std::fs::read_dir(&root).is_err();
    let mut app = watching_app(None);
    let watch = app.world_mut().spawn(FileWatch::tree(&root)).id();
    run_for(&mut app, Duration::from_millis(1500));
    std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o755)).ok();
    run_for(&mut app, Duration::from_millis(1500));
    std::fs::write(&there, "edited").ok();
    run_for(&mut app, Duration::from_millis(1000));
    std::fs::remove_dir_all(&root).ok();
    if locked {
        assert_eq!(
            told(&app, watch),
            [
                ("unavailable", vec![]),
                ("snapshot", vec![there.clone()]),
                ("changed", vec![there])
            ]
        );
    }
}

/// A watch of a workspace whose listing fails for a while at first tells
/// why and keeps looking: once a listing works it tells the files there,
/// then what changes.
#[test]
fn a_watch_whose_first_listing_fails_starts_once_one_works() {
    let root = std::env::temp_dir().join(format!("rig-watch-flaky-{}", std::process::id()));
    std::fs::create_dir_all(&root).ok();
    let there = root.join("there.md");
    std::fs::write(&there, "x").ok();
    let flaky = Arc::new(Flaky(AtomicBool::new(true)));
    let mut app = watching_app(Some(flaky.clone()));
    let watch = app.world_mut().spawn(FileWatch::tree(&root)).id();
    run_for(&mut app, Duration::from_millis(2000));
    flaky.0.store(false, Ordering::SeqCst);
    run_for(&mut app, Duration::from_millis(1500));
    std::fs::write(&there, "edited").ok();
    run_for(&mut app, Duration::from_millis(1500));
    std::fs::remove_dir_all(&root).ok();
    assert_eq!(
        told(&app, watch),
        [
            ("unavailable", vec![]),
            ("snapshot", vec![there.clone()]),
            ("changed", vec![there])
        ]
    );
}

/// This machine's disk, whose listing fails while the flag is set.
#[derive(Debug)]
struct Flaky(AtomicBool);

impl Workspace for Flaky {
    fn host(&self) -> Option<&str> {
        Some("test://flaky")
    }

    fn stat(&self, path: &Path) -> io::Result<Stat> {
        Local.stat(path)
    }

    fn read(&self, path: &Path, limit: u64) -> io::Result<Contents> {
        Local.read(path, limit)
    }

    fn write(&self, path: &Path, bytes: &[u8]) -> io::Result<()> {
        Local.write(path, bytes)
    }

    fn remove(&self, path: &Path) -> io::Result<()> {
        Local.remove(path)
    }

    fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        Local.rename(from, to)
    }

    fn canonical(&self, path: &Path) -> PathBuf {
        Local.canonical(path)
    }

    fn files(&self, root: &Path, glob: Option<&str>, listing: Listing) -> io::Result<Listed> {
        match self.0.load(Ordering::SeqCst) {
            true => Err(io::Error::other("the listing failed")),
            false => Local.files(root, glob, listing),
        }
    }

    fn spawn(
        &self,
        launcher: &Launcher,
        dir: Option<&Path>,
        program: &OsStr,
        args: &[OsString],
        spawn: Spawn,
    ) -> io::Result<Process> {
        Local.spawn(launcher, dir, program, args, spawn)
    }
}
