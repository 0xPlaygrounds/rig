use std::sync::mpsc::Receiver;

use rig_cassette::journal::MemoryStore;
use rig_core::test_utils::MockCompletionModel;
use rig_harness::pick::PickPlugin;
use rig_harness_test_support::{
    Ended, advance, app_on, connect, first_agent, log_ended, reply, run_until, scratch_home,
};

use super::*;

/// The kernel, the journal, the pickers' `Shown` and the plugin on
/// `store`, with `RIG_HOME` at `home`, after the first frame.
fn start(store: &MemoryStore, home: &Home) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_on(store);
    app.insert_resource(RigHome(home.clone()))
        .insert_resource(ProjectRoot(home.root().join("project")))
        .add_plugins((AgentPlugin, JournalPlugin, PickPlugin, HistoryPlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.cleanup();
    app.update();
    (app, wakes)
}

/// Presses this plugin's binding of `keys` in the input, as the terminal
/// view does on a match (there is no public headless view), then runs a
/// frame.
fn press(app: &mut App, keys: KeyPress) {
    let world = app.world_mut();
    let mut bindings = world.query::<(Entity, &KeyBinding)>();
    let bound = bindings
        .iter(world)
        .find(|(_, binding)| binding.scope == KeyScope::Input && binding.keys == keys.into())
        .map(|(entity, _)| entity);
    assert!(bound.is_some(), "{keys} is not bound");
    if let Some(entity) = bound {
        world.trigger(KeyPressed {
            entity,
            agent: None,
            released: false,
        });
    }
    app.update();
}

fn input(app: &App) -> (String, usize) {
    let input = app.world().resource::<TuiInput>();
    (input.text().to_owned(), input.cursor())
}

/// Only the first "a" is recorded; the draft typed survives a restart,
/// with the cursor at its end, and Up and Down browse past it and back.
#[test]
fn history_skips_repeats_and_secrets_and_the_draft_survives_a_restart() {
    let home = scratch_home("corpus-ho-history-and-drafts");
    let store = MemoryStore::default();
    let (mut app, wakes) = start(&store, &home);
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let model = MockCompletionModel::from_stream_turns((0..4).map(|_| reply("ok")));
    if let Some(connection) = connect(&model) {
        let choice = ModelChoice(connection.spec.reference());
        app.world_mut()
            .entity_mut(agent)
            .insert((connection, choice));
    }
    for text in ["a", "a", " secret", "key sk-abc123"] {
        let message = Deliver::new(agent, text, DeliveryMode::Queue).with_origin(Origin::user());
        app.world_mut().trigger(message);
    }
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });

    let file = app
        .world()
        .resource::<History>()
        .file()
        .map(Path::to_path_buf);
    let written = file
        .and_then(|file| fs::read_to_string(file).ok())
        .unwrap_or_default();
    let texts: Vec<String> = written
        .lines()
        .filter_map(|line| serde_json::from_str::<Entry>(line).ok())
        .map(|entry| entry.text)
        .collect();
    assert_eq!(texts, ["a"], "{written}");

    // The focused agent's draft, typed in the terminal view's input.
    app.world_mut().resource_mut::<TuiInput>().set("half writ");
    app.update();
    advance(&mut app, QUIET + Duration::from_millis(100));
    app.update();
    let saved = app.world().get::<Draft>(agent).cloned();
    assert_eq!(saved, Some(Draft("half writ".to_owned())));
    drop(app);

    let (mut app, _wakes) = start(&store, &home);
    assert_eq!(input(&app), ("half writ".to_owned(), 9));
    press(&mut app, KeyPress::key(KeyCode::Up));
    assert_eq!(input(&app).0, "a");
    press(&mut app, KeyPress::key(KeyCode::Down));
    assert_eq!(input(&app), ("half writ".to_owned(), 9));
    fs::remove_dir_all(home.root()).ok();
}
