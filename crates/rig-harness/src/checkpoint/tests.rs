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

/// In one turn, each `shell` call reports the files it made, and not the
/// file a `write` call made between them, which reports its own.
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
            "w1",
            "write",
            serde_json::json!({ "path": "b.txt", "content": "b\n" }),
        ),
        calls(
            "s2",
            "shell",
            serde_json::json!({ "command": "printf two > c.txt" }),
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
        .add_observer(note_changes);
    app.finish();
    app.update();
    let Some(connection) = connect(&model) else {
        return;
    };
    let world = app.world_mut();
    let agent = world.spawn((Agent, connection)).id();
    world.trigger(Deliver::new(agent, "make files", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        world.resource::<Changed>().0.contains_key("s2")
    });
    let changed = &app.world().resource::<Changed>().0;
    let path = |name: &str| dir.join(name).to_string_lossy().into_owned();
    let created = |name: &str| vec![(path(name), ChangeKind::Created)];
    assert_eq!(changed.get("s1"), Some(&created("a.txt")), "{changed:?}");
    assert_eq!(changed.get("w1"), Some(&created("b.txt")), "{changed:?}");
    assert_eq!(changed.get("s2"), Some(&created("c.txt")), "{changed:?}");
    std::fs::remove_dir_all(&dir).ok();
}
