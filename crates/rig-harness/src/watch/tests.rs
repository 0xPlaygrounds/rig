use std::time::{Duration, Instant};

use bevy_app::TaskPoolPlugin;
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use rig_ecs::AgentPlugin;

use rig_tools::workspace::WatchEvent;

use super::{FileChanges, FileWatch, WatchPlugin};

/// What the app's watches saw, frame after frame.
#[derive(Resource, Default)]
struct Seen(Vec<FileChanges>);

/// What each change of `watch` told, by kind, with its paths.
fn told(app: &App, watch: Entity) -> Vec<(&'static str, Vec<std::path::PathBuf>)> {
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

/// A watch that cannot start never looks like a healthy empty directory:
/// it tells why, then that it closed.
#[cfg(unix)]
#[test]
fn a_watch_that_cannot_start_is_not_an_empty_directory() {
    use std::os::unix::fs::PermissionsExt as _;
    let root = std::env::temp_dir().join(format!("rig-watch-locked-{}", std::process::id()));
    std::fs::create_dir_all(&root).ok();
    std::fs::write(root.join("there.md"), "x").ok();
    std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o000)).ok();
    // Whoever may read it anyway (root) can watch it.
    let locked = std::fs::read_dir(&root).is_err();
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
    run_until_seen(&mut app, 2);
    std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o755)).ok();
    std::fs::remove_dir_all(&root).ok();
    if locked {
        assert_eq!(
            told(&app, watch),
            [("unavailable", vec![]), ("closed", vec![])]
        );
    }
}
