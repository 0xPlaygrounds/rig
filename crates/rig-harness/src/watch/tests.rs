use std::time::{Duration, Instant};

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

use super::{FileChanges, FileWatch, WatchPlugin};

/// What the app's watches saw, frame after frame.
#[derive(Resource, Default)]
struct Seen(Vec<FileChanges>);

#[test]
fn a_watch_names_the_files_made_under_it_and_stops_when_despawned() {
    let root = std::env::temp_dir().join(format!("rig-watch-{}", std::process::id()));
    std::fs::create_dir_all(root.join("nested")).ok();
    let mut app = App::new();
    app.add_plugins(WatchPlugin)
        .init_resource::<Seen>()
        .add_systems(
            Update,
            |mut changes: MessageReader<FileChanges>, mut seen: ResMut<Seen>| {
                seen.0.extend(changes.read().cloned());
            },
        );
    let watch = app.world_mut().spawn(FileWatch::tree(&root)).id();
    app.update();
    // The watcher has looked once before anything changes.
    std::thread::sleep(Duration::from_millis(1500));
    let made = root.join("nested/new.md");
    std::fs::write(&made, "hello").ok();
    let started = Instant::now();
    while app.world().resource::<Seen>().0.is_empty() && started.elapsed() < Duration::from_secs(5)
    {
        app.update();
        std::thread::sleep(Duration::from_millis(50));
    }
    let seen = &app.world().resource::<Seen>().0;
    assert_eq!(
        seen.first(),
        Some(&FileChanges {
            watch,
            paths: vec![made.clone()],
        })
    );
    app.world_mut().despawn(watch);
    app.update();
    std::fs::remove_file(&made).ok();
    std::thread::sleep(Duration::from_millis(2200));
    app.update();
    assert_eq!(app.world().resource::<Seen>().0.len(), 1);
    std::fs::remove_dir_all(&root).ok();
}
