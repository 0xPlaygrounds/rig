use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rig_cassette::journal::MemoryStore;
use rig_coding_tools::WriteTool;
use rig_core::test_utils::MockCompletionModel;
use rig_ecs::journal::JournalPlugin;
use rig_harness_test_support::{app_on, calls, connect, reply, run_until};
use rig_tui::{AppStatus, FRONT};

use super::*;

/// What the git item says, if it is there.
fn item(world: &World) -> Option<String> {
    let place = GitStatusSettings::default().place();
    world
        .get_resource::<AppStatus>()?
        .0
        .iter()
        .find(|item| (item.side, item.order) == (place.side, place.order))
        .map(|item| item.text.clone())
}

#[test]
fn the_item_shows_the_branch_then_the_file_the_agent_wrote() {
    let base = std::env::var_os("CARGO_TARGET_DIR").map_or_else(std::env::temp_dir, PathBuf::from);
    let dir = base.join(format!("corpus-git-status-item-{}", std::process::id()));
    fs::remove_dir_all(&dir).ok();
    assert!(fs::create_dir_all(&dir).is_ok());
    let init = Command::new("git")
        .args(["init", "-q", "-b", "main"])
        .current_dir(&dir)
        .status();
    assert!(init.is_ok_and(|status| status.success()));

    let model = MockCompletionModel::from_stream_turns([
        calls(
            "1",
            "write",
            serde_json::json!({ "path": dir.join("new.txt"), "content": "new\n" }),
        ),
        reply("wrote new.txt"),
    ]);
    let connection = connect(&model);
    assert!(connection.is_some(), "the catalog lists the model");
    let Some(connection) = connection else {
        return;
    };
    let runs = Arc::new(AtomicUsize::new(0));
    let counted = runs.clone();
    let git = GitRunner::default();
    let runner = GitRunner::new(move |command| {
        counted.fetch_add(1, Ordering::SeqCst);
        git.run(command)
    });
    let (mut app, wakes) = app_on(&MemoryStore::default());
    // The terminal view's run, without rig-tui opening a terminal.
    app.insert_resource(Invoked {
        terminal: true,
        ..Invoked::default()
    })
    .add_front(Front::new(FRONT).on_terminal())
    .insert_resource(ProjectRoot(dir.clone()))
    .insert_resource(runner)
    .add_plugins((AgentPlugin, JournalPlugin, WriteTool, GitStatusItemPlugin));
    app.finish();

    run_until(&mut app, &wakes, |world| item(world).is_some());
    assert_eq!(item(app.world()).as_deref(), Some("main"));

    let world = app.world_mut();
    let agent = world.query_filtered::<Entity, With<Agent>>().single(world);
    assert!(agent.is_ok(), "one agent");
    let Ok(agent) = agent else {
        return;
    };
    world.entity_mut(agent).insert(connection);
    world.trigger(Deliver::new(agent, "Write new.txt", DeliveryMode::Queue));
    let frames = AtomicUsize::new(0);
    run_until(&mut app, &wakes, |world| {
        frames.fetch_add(1, Ordering::SeqCst);
        item(world).as_deref() == Some("main ±1")
    });
    assert_eq!(item(app.world()).as_deref(), Some("main ±1"));

    // Startup, the write and the turn's end: a look each at most, however
    // many frames ran.
    let runs = runs.load(Ordering::SeqCst);
    let frames = frames.load(Ordering::SeqCst);
    assert!(
        runs <= 3 && frames > runs,
        "git ran {runs} times in {frames} frames"
    );
    fs::remove_dir_all(&dir).ok();
}
