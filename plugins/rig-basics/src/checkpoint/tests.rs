use std::collections::HashMap;
use std::path::PathBuf;

use rig_core::test_utils::MockCompletionModel;
use rig_core::tool::{ChangeKind, FileChange, FilesChanged};
use rig_harness_test_support::{MemoryStore, app_on, calls, connect, reply, run_until};
use rig_tools::checkpoint::{Taking, snapshot};
use rig_tools::process::Launcher;
use rig_tools::workspace::Local;
use rig_tools::{Shell, Write};

use super::{CallCheckpoint, CheckpointPlugin, TurnCheckpoint};
use crate::per_call::AppPerCallExt;
use rig_harness::prelude::*;

/// The files each settled call reported changed, and those changed since
/// its own checkpoint's "before" when it settled, by call id.
#[derive(Resource, Default)]
struct Changed {
    reported: HashMap<String, Vec<(String, ChangeKind)>>,
    since_before: HashMap<String, Vec<(String, ChangeKind)>>,
}

fn note_changes(
    settled: On<Add<Settled>>,
    calls: Query<(&ToolRequest, Option<&Published>, Option<&CallCheckpoint>)>,
    mut changed: ResMut<Changed>,
    root: Res<ProjectRoot>,
) {
    let Ok((request, published, checkpoint)) = calls.get(settled.entity) else {
        return;
    };
    let named = |changes: Vec<FileChange>| changes.into_iter().map(|it| (it.path, it.kind));
    let id = request.call.id.to_string();
    if let Some(before) = checkpoint.and_then(CallCheckpoint::before) {
        let now = snapshot(&Local, &root.0, &Taking::default(), None);
        let since = now.and_then(|now| now.changes_since(&before));
        let since = named(since.unwrap_or_default()).collect();
        changed.since_before.insert(id.clone(), since);
    }
    let changes = published.and_then(|published| published.0.get::<FilesChanged>().ok().flatten());
    let changes = named(changes.map(|it| it.0).unwrap_or_default()).collect();
    changed.reported.insert(id, changes);
}

/// A directory of the test's own beside its binary, emptied.
fn scratch(name: &str) -> PathBuf {
    let exe = std::env::current_exe().unwrap_or_default();
    let dir = exe.with_file_name(format!("{name}-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).ok();
    dir.canonicalize().unwrap_or(dir)
}

/// The project directory, where the user saves a file when the first
/// `shell` call settles.
#[derive(Resource)]
struct UserSaves(PathBuf);

fn user_saves(settled: On<Add<Settled>>, calls: Query<&ToolRequest>, dir: Res<UserSaves>) {
    if calls
        .get(settled.entity)
        .is_ok_and(|request| request.call.id.to_string() == "s1")
    {
        std::fs::write(dir.0.join("user.txt"), "mine\n").ok();
    }
}

/// In one turn, each `shell` call reports the files it made, and neither
/// the file a `write` call made between them, which reports its own, nor
/// one the user saved between them: each call takes its own "before", and
/// keeps it on the call for a plugin to compare with.
#[test]
fn each_shell_call_of_a_turn_reports_what_it_changed() {
    let dir = scratch("checkpoint-turn");
    let model = MockCompletionModel::from_stream_turns([
        calls(
            "s1",
            "shell",
            serde_json::json!({ "command": "printf one > a.txt" }),
        ),
        calls(
            "s2",
            "shell",
            serde_json::json!({ "command": "printf two > c.txt" }),
        ),
        calls(
            "w1",
            "write",
            serde_json::json!({ "path": "b.txt", "content": "b\n" }),
        ),
        calls(
            "s3",
            "shell",
            serde_json::json!({ "command": "printf three > d.txt" }),
        ),
        reply("done"),
    ]);
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(ProjectRoot(dir.clone()))
        .add_plugins((AgentPlugin, CheckpointPlugin))
        .add_tool_per_call(Shell::default(), ToolOptions::default(), |place| Shell {
            launcher: Launcher::default().in_dir(&place.base),
            workspace: place.workspace,
            confine: place.confine,
            ..Shell::default()
        })
        .add_tool_per_call(Write::default(), ToolOptions::default(), |place| Write {
            workspace: place.workspace,
            base: Some(place.base),
            confine: place.confine,
        })
        .init_resource::<Changed>()
        .insert_resource(UserSaves(dir.clone()))
        .add_observer(note_changes)
        .add_observer(user_saves);
    app.finish();
    app.update();
    let Some(connection) = connect(&model) else {
        return;
    };
    let world = app.world_mut();
    let agent = world.spawn((Agent, connection)).id();
    world.trigger(Deliver::new(agent, "make files", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        world.resource::<Changed>().reported.contains_key("s3")
    });
    let Changed {
        reported: changed,
        since_before,
    } = app.world().resource::<Changed>();
    let path = |name: &str| dir.join(name).to_string_lossy().into_owned();
    let created = |name: &str| vec![(path(name), ChangeKind::Created)];
    assert_eq!(changed.get("s1"), Some(&created("a.txt")), "{changed:?}");
    assert_eq!(changed.get("w1"), Some(&created("b.txt")), "{changed:?}");
    assert_eq!(changed.get("s2"), Some(&created("c.txt")), "{changed:?}");
    assert_eq!(changed.get("s3"), Some(&created("d.txt")), "{changed:?}");
    assert_eq!(since_before.get("s2"), Some(&created("c.txt")));
    assert_eq!(since_before.get("s3"), Some(&created("d.txt")));
    assert_eq!(since_before.get("w1"), None, "a write is not checkpointed");
    std::fs::remove_dir_all(&dir).ok();
}

/// When the app starts, the content earlier runs' scans kept in the
/// session scratch goes; kept content that cannot be removed only warns,
/// and the app runs on.
#[test]
fn kept_content_goes_when_the_app_starts() {
    let started = |name: &str, blobs_is_a_file: bool| {
        let dir = scratch(name);
        let blobs = dir.join("checkpoints").join("project").join("blobs");
        match blobs_is_a_file {
            true => {
                std::fs::create_dir_all(blobs.parent().unwrap_or(&dir)).ok();
                std::fs::write(&blobs, "not a directory").ok();
            }
            false => {
                std::fs::create_dir_all(&blobs).ok();
                std::fs::write(blobs.join("0123"), "kept").ok();
            }
        }
        let (mut app, _wakes) = app_on(&MemoryStore::default());
        app.insert_resource(ProjectRoot(dir.clone()))
            .insert_resource(rig_harness::host::session::Scratch(Some(dir.clone())))
            .add_plugins(CheckpointPlugin);
        app.finish();
        app.update();
        let left = blobs.exists();
        std::fs::remove_dir_all(&dir).ok();
        left
    };
    assert!(
        !started("checkpoint-forget", false),
        "the kept content goes"
    );
    assert!(started("checkpoint-forget-fails", true), "the file stays");
}

/// A turn spawned before the plugin builds keeps turns from requiring a
/// shared checkpoint: the plugin only warns, and the app runs on with each
/// call taking its own.
#[test]
fn a_turn_spawned_before_the_plugin_builds_only_warns() {
    let (mut app, _wakes) = app_on(&MemoryStore::default());
    let agent = app.world_mut().spawn_empty().id();
    app.world_mut().spawn(TurnOf(agent));
    app.add_plugins(CheckpointPlugin);
    app.finish();
    app.update();
    let later = app.world_mut().spawn(TurnOf(agent)).id();
    assert!(app.world().get::<TurnCheckpoint>(later).is_none());
}
