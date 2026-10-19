use std::sync::mpsc::Receiver;

use rig_basics::coding_tools::{EditTool, ReadTool, WriteTool};
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{
    Ended, app_on, calls, connect, first_agent, log_ended, reply, run_until,
};
use rig_tui::ratatui::Terminal;
use rig_tui::ratatui::backend::TestBackend;
use serde_json::json;

use super::*;

/// The kernel, the coding tools and the panel on `store`, working in
/// `root`, after the first frame restored the session.
fn start(store: &MemoryStore, root: &Path) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_on(store);
    app.insert_resource(ProjectRoot(root.to_owned()))
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            ReadTool,
            EditTool,
            WriteTool,
            FilesPanelPlugin,
        ))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    (app, wakes)
}

/// Presses the keys of the binding labelled `label`.
fn press(world: &mut World, label: &str) {
    let mut bindings = world.query::<(Entity, &KeyBinding)>();
    let found: Vec<Entity> = bindings
        .iter(world)
        .filter(|(_, binding)| binding.label == label)
        .map(|(entity, _)| entity)
        .collect();
    assert_eq!(found.len(), 1, "{label}");
    for entity in found {
        world.trigger(KeyPressed {
            entity,
            agent: None,
            released: false,
        });
    }
    world.flush();
}

/// The name, `+` and `−` of each changed file, and the name of each read.
fn data(files: &TouchedFiles, root: &Path) -> (Vec<String>, Vec<String>) {
    let changed = files.changed().iter().map(|f| label(f, root)).collect();
    (
        changed,
        files.read().iter().map(|f| name(f, root)).collect(),
    )
}

#[test]
fn reads_edits_and_writes_are_listed_drawn_and_restored() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/files-panel-test");
    std::fs::remove_dir_all(&root).ok();
    std::fs::create_dir_all(&root).ok();
    let (a, b) = (root.join("a.txt"), root.join("b.txt"));
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
    let (mut app, wakes) = start(&store, &root);
    let world = app.world_mut();
    let (agent, connection) = (first_agent(world), connect(&model));
    assert!(agent.is_some() && connection.is_some(), "no agent to run");
    let (Some(agent), Some(connection)) = (agent, connection) else {
        return;
    };
    world.entity_mut(agent).insert(connection);
    world.trigger(Deliver::new(
        agent,
        "Change the files.",
        DeliveryMode::Steer,
    ));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });

    let files = app.world().resource::<TouchedFiles>().clone();
    let changed = vec!["b.txt +3 −0".to_owned(), "a.txt +1 −1".to_owned()];
    assert_eq!(data(&files, &root), (changed, vec!["a.txt".to_owned()]));
    let a_path = a.display().to_string();
    let diff = files.files.get(&a_path).map(TouchedFile::session_diff);
    let diff = diff.unwrap_or_default();
    assert!(diff.contains("-two\n+deux\n"), "{diff}");

    let mut terminal = Terminal::new(TestBackend::new(WIDTH, 10)).ok();
    let drawn = terminal.as_mut().map(|terminal| {
        terminal
            .draw(|frame| {
                let area = frame.area();
                let theme = rig_tui::Theme::default();
                draw_files(&files, &root, Some(0), (area, &theme), frame.buffer_mut());
            })
            .is_ok()
    });
    assert_eq!(drawn, Some(true));
    let text: String = terminal
        .iter()
        .flat_map(|terminal| terminal.backend().buffer().content())
        .map(|cell| cell.symbol())
        .collect();
    assert!(text.contains("b.txt +3 −0"), "{text}");

    // ctrl+f opens the panel with the keys; Down then Enter shows a.txt's
    // diff, which takes the keys.
    for key in [
        "The files panel",
        "Select the file below",
        "Show the file's session diff",
    ] {
        press(app.world_mut(), key);
    }
    let world = app.world_mut();
    let shown = world
        .query_filtered::<&DiffBox, With<TuiPanel>>()
        .single(world);
    assert_eq!(shown.map(|diff| diff.path.clone()).ok(), Some(a_path));
    let focus = world.resource::<KeyFocus>().0;
    let diff_box = world
        .query_filtered::<Entity, With<DiffBox>>()
        .single(world);
    assert_eq!(focus, diff_box.ok());
    drop(app);

    let (restarted, _wakes) = start(&store, &root);
    assert_eq!(restarted.world().resource::<TouchedFiles>(), &files);
    std::fs::remove_dir_all(&root).ok();
}
