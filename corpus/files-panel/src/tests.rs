use std::path::PathBuf;
use std::sync::mpsc::Receiver;

use rig_cassette::journal::MemoryStore;
use rig_coding_tools::{EditTool, ReadTool, WriteTool};
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::{app_on, calls, connect, reply, run_until};
use rig_tui::ratatui::Terminal;
use rig_tui::ratatui::backend::TestBackend;
use serde_json::json;

use super::*;

/// Turns ended.
#[derive(Resource, Default)]
struct Ended(usize);

/// The kernel, the coding tools and the panel on `store`, after the first
/// frame restored the session.
fn start(store: &MemoryStore) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_on(store);
    app.add_plugins((
        AgentPlugin,
        JournalPlugin,
        ReadTool,
        EditTool,
        WriteTool,
        FilesPanelPlugin,
    ))
    .init_resource::<Ended>()
    .add_observer(|_: On<TurnEnded>, mut ended: ResMut<Ended>| ended.0 += 1);
    app.finish();
    app.update();
    (app, wakes)
}

/// A fresh directory for the test's files, under the crate's ignored
/// `target`.
fn scratch() -> PathBuf {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/files-panel-test");
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).ok();
    dir
}

/// The path and counts of each changed file, and the path of each read.
fn data(files: &TouchedFiles) -> (Vec<(String, u32, u32)>, Vec<String>) {
    let name = |file: &TouchedFile| {
        let path = Path::new(&file.path).file_name();
        path.map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default()
    };
    let changed = files.changed().into_iter();
    let changed = changed.map(|(file, change)| (name(file), change.added, change.removed));
    (
        changed.collect(),
        files.read().into_iter().map(name).collect(),
    )
}

#[test]
fn reads_edits_and_writes_are_listed_drawn_and_restored() {
    let dir = scratch();
    let a = dir.join("a.txt").display().to_string();
    let b = dir.join("b.txt").display().to_string();
    std::fs::write(&a, "one\ntwo\nthree\n").ok();
    let model = MockCompletionModel::from_stream_turns([
        calls("1", "read", json!({ "path": a })),
        calls(
            "2",
            "edit",
            json!({ "path": a, "edits": [{ "old_text": "two", "new_text": "deux" }] }),
        ),
        calls("3", "write", json!({ "path": b, "content": "x\ny\nz\n" })),
        reply("Done."),
    ]);
    let store = MemoryStore::default();
    let (mut app, wakes) = start(&store);
    let world = app.world_mut();
    let agent = world.query_filtered::<Entity, With<Agent>>().single(world);
    let connection = connect(&model);
    assert!(agent.is_ok() && connection.is_some(), "no agent to run");
    let (Ok(agent), Some(connection)) = (agent, connection) else {
        return;
    };
    world.entity_mut(agent).insert(connection);
    world.trigger(Deliver::new(
        agent,
        "Change the files.",
        DeliveryMode::Steer,
    ));
    run_until(&mut app, &wakes, |world| world.resource::<Ended>().0 > 0);

    let files = app.world().resource::<TouchedFiles>().clone();
    let changed = vec![("b.txt".to_owned(), 3, 0), ("a.txt".to_owned(), 1, 1)];
    assert_eq!(data(&files), (changed, vec!["a.txt".to_owned()]));

    let mut terminal = Terminal::new(TestBackend::new(WIDTH, 10)).ok();
    let drawn = terminal.as_mut().map(|terminal| {
        terminal
            .draw(|frame| frame.render_widget(panel(&files, WIDTH), frame.area()))
            .is_ok()
    });
    assert_eq!(drawn, Some(true));
    let text: String = terminal
        .iter()
        .flat_map(|terminal| terminal.backend().buffer().content())
        .map(|cell| cell.symbol())
        .collect();
    assert!(text.contains("b.txt +3 −0"), "{text}");
    drop(app);

    let (restarted, _wakes) = start(&store);
    assert_eq!(restarted.world().resource::<TouchedFiles>(), &files);
    std::fs::remove_dir_all(&dir).ok();
}
