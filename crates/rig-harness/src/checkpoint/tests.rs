use std::collections::HashMap;
use std::path::PathBuf;

use rig_core::test_utils::MockCompletionModel;
use rig_core::tool::{ChangeKind, FilesChanged};
use rig_harness_test_support::{MemoryStore, app_on, calls, connect, reply, run_until};
use rig_tools::process::Launcher;
use rig_tools::{Shell, Write};

use super::CheckpointPlugin;
use crate::per_call::AppPerCallExt;
use crate::prelude::*;

/// The files each settled call reported changed, by call id.
#[derive(Resource, Default)]
struct Changed(HashMap<String, Vec<(String, ChangeKind)>>);

fn note_changes(
    settled: On<Add<Settled>>,
    calls: Query<(&ToolRequest, Option<&Published>)>,
    mut changed: ResMut<Changed>,
) {
    let Ok((request, published)) = calls.get(settled.entity) else {
        return;
    };
    let changes = published.and_then(|published| published.0.get::<FilesChanged>().ok().flatten());
    let changes = changes.map(|it| it.0).unwrap_or_default();
    let changes = changes.into_iter().map(|it| (it.path, it.kind)).collect();
    changed.0.insert(request.call.id.to_string(), changes);
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
/// one the user saved between them: each call takes its own "before".
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
        .add_tool_per_call(Shell::default(), ToolOptions::default(), false, |place| {
            Shell {
                launcher: Launcher::default().in_dir(&place.base),
                workspace: place.workspace,
                confine: place.confine,
                ..Shell::default()
            }
        })
        .add_tool_per_call(Write::default(), ToolOptions::default(), false, |place| {
            Write {
                workspace: place.workspace,
                base: Some(place.base),
                confine: place.confine,
            }
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
        world.resource::<Changed>().0.contains_key("s3")
    });
    let changed = &app.world().resource::<Changed>().0;
    let path = |name: &str| dir.join(name).to_string_lossy().into_owned();
    let created = |name: &str| vec![(path(name), ChangeKind::Created)];
    assert_eq!(changed.get("s1"), Some(&created("a.txt")), "{changed:?}");
    assert_eq!(changed.get("w1"), Some(&created("b.txt")), "{changed:?}");
    assert_eq!(changed.get("s2"), Some(&created("c.txt")), "{changed:?}");
    assert_eq!(changed.get("s3"), Some(&created("d.txt")), "{changed:?}");
    std::fs::remove_dir_all(&dir).ok();
}
