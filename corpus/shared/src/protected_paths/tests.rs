use std::path::PathBuf;

use rig_basics::coding_tools::{ReadTool, WriteTool};
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{Ended, app_on, calls, connect, log_ended, reply, run_until};
use serde_json::json;

use super::*;

/// A project in the system's temporary directory with a secret `.env`, a
/// file its `.rig/paths.toml` read-protects and an ordinary file, each
/// holding `KEY`, and an empty `RIG_HOME` beside it.
fn project() -> std::io::Result<PathBuf> {
    let dir = std::env::temp_dir().join(format!("corpus-protected-paths-{}", std::process::id()));
    let project = dir.join("project");
    std::fs::create_dir_all(project.join("sub"))?;
    std::fs::create_dir_all(project.join("keys"))?;
    std::fs::create_dir_all(project.join(".rig"))?;
    std::fs::write(
        project.join(".rig/paths.toml"),
        "deny_read = [\"keys/**\"]\n",
    )?;
    std::fs::write(project.join("keys/api.txt"), "KEY=secret-key\n")?;
    std::fs::create_dir_all(dir.join("home"))?;
    std::fs::write(project.join(".env"), "KEY=secret\n")?;
    std::fs::write(project.join("notes.txt"), "KEY=public\n")?;
    Ok(project)
}

#[test]
fn protected_files_are_neither_read_nor_written_nor_found() {
    let project = project();
    assert!(project.is_ok(), "{project:?}");
    let Ok(project) = project else {
        return;
    };
    let dir = project.parent().unwrap_or(&project).to_path_buf();
    // A protected file the search cannot read, and so would name.
    let locked = project.join("keys/locked.txt");
    std::fs::write(&locked, "public\n").ok();
    #[cfg(unix)]
    let mode = |mode| {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(mode)).ok()
    };
    #[cfg(unix)]
    mode(0o000);
    // The built-in tools resolve paths against the working directory.
    assert!(std::env::set_current_dir(&project).is_ok());
    let store = MemoryStore::default();
    let (mut app, wakes) = app_on(&store);
    app.set_error_handler(rig_harness::error::warn)
        .insert_resource(ProjectRoot(project.clone()))
        .insert_resource(RigHome(Home::at(dir.join("home"))))
        .add_plugins((AgentPlugin, SessionLogPlugin, ReadTool, WriteTool))
        .add_tool_per_call(rig_tools::Search::default(), ToolOptions::default())
        .add_plugins(ProtectedPathsPlugin)
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let model = MockCompletionModel::from_stream_turns([
        calls("1", "read", json!({ "path": "./sub/../.env" })),
        calls(
            "2",
            "write",
            json!({ "path": "../outside.txt", "content": "x" }),
        ),
        calls("3", "search", json!({ "pattern": "KEY" })),
        calls("4", "search", json!({ "pattern": "public" })),
        reply("done"),
    ]);
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next();
    let connection = connect(&model);
    assert!(
        agent.is_some() && connection.is_some(),
        "an agent and the catalog's model"
    );
    let (Some(agent), Some(connection)) = (agent, connection) else {
        return;
    };
    world.entity_mut(agent).insert(connection);
    world.trigger(Deliver::new(agent, "Look around.", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });

    let sent: Vec<String> = model
        .requests()
        .iter()
        .map(|request| serde_json::to_string(request).unwrap_or_default())
        .collect();
    let outside = dir.join("outside.txt").exists();
    #[cfg(unix)]
    mode(0o644);
    std::fs::remove_dir_all(&dir).ok();
    assert_eq!(sent.len(), 5, "{sent:?}");
    let [_, after_read, after_write, after_search, after_public] = sent.as_slice() else {
        return;
    };
    assert!(after_read.contains("read-protected"), "{after_read}");
    assert!(sent.iter().all(|request| !request.contains("secret")));
    assert!(after_write.contains("write-protected"), "{after_write}");
    assert!(!outside);
    assert!(
        after_search.contains("notes.txt:1: KEY=public"),
        "{after_search}"
    );
    // The search never shows `.env` (it skips hidden files); the match in
    // `keys/api.txt`, which `.rig/paths.toml` protects, is dropped.
    assert!(!after_search.contains(".env:"));
    assert!(!after_search.contains("keys/api.txt"));
    assert!(after_search.contains("1 match in protected files hidden"));
    // A search with no protected match does not name the protected file it
    // could not read either.
    assert!(
        after_public.contains("notes.txt:1: KEY=public") && !after_public.contains("locked"),
        "{after_public}"
    );
    let counts = app.world().resource::<PathGuardCounts>();
    assert_eq!(counts.blocked.get("deny_read .env*"), Some(&1));
    assert_eq!(
        counts.blocked.get("deny_write outside the project"),
        Some(&1)
    );
    assert_eq!(counts.hidden.get("deny_read keys/**"), Some(&1));
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(ProtectedPathsPlugin);
}
