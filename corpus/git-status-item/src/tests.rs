use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rig_cassette::journal::MemoryStore;
use rig_coding_tools::WriteTool;
use rig_core::test_utils::MockCompletionModel;
use rig_ecs::journal::JournalPlugin;
use rig_harness_test_support::{app_on, calls, connect, reply, run_until};
use rig_tui::{AppStatus, TuiPlugin};

use super::*;

/// What the git item says, if it is there.
fn item(world: &World) -> Option<String> {
    let place = GitStatusConfig::default().place;
    world
        .resource::<AppStatus>()
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
    let inner = GitRunner::default();
    assert!(inner.run(&dir, &["init", "-q", "-b", "main"]).is_some());
    let file = dir.join("new.txt").display().to_string();

    let model = MockCompletionModel::from_stream_turns([
        calls(
            "1",
            "write",
            serde_json::json!({ "path": file, "content": "new\n" }),
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
    let runner = GitRunner::new(move |dir, args| {
        counted.fetch_add(1, Ordering::SeqCst);
        inner.run(dir, args)
    });
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(GitStatusConfig {
        dir: Some(dir.clone()),
        ..GitStatusConfig::default()
    })
    .insert_resource(runner)
    // No `Invoked`: the view's plugin draws nothing, the item is there.
    .add_plugins((
        AgentPlugin,
        JournalPlugin,
        TuiPlugin,
        WriteTool,
        GitStatusItemPlugin,
    ));
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
    assert!(dir.join("new.txt").exists());

    // Startup, the write and the turn's end: a look each at most, however
    // many frames ran.
    let runs = runs.load(Ordering::SeqCst);
    assert!(runs <= 4, "git ran {runs} times");
    assert!(
        frames.load(Ordering::SeqCst) > runs,
        "fewer frames than git runs"
    );
    fs::remove_dir_all(&dir).ok();
}
