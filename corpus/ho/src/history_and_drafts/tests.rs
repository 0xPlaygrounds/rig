use std::sync::mpsc::Receiver;

use rig_core::test_utils::MockCompletionModel;
use rig_harness::pick::PickPlugin;
use rig_harness_test_support::{
    Ended, MemoryStore, advance, app_on, connect, first_agent, log_ended, reply, run_until,
    scratch_home,
};

use rig_tui::{Editor, MainEditor};

use super::*;

/// The kernel, the journal, the pickers' `ShownAgent` and the plugin on
/// `store`, with `RIG_HOME` at `home`, after the first frame.
fn start(store: &MemoryStore, home: &Home) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_on(store);
    app.insert_resource(RigHome(home.clone()))
        .insert_resource(ProjectRoot(home.root().join("project")))
        .add_plugins((AgentPlugin, SessionLogPlugin, PickPlugin, HistoryPlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.cleanup();
    app.update();
    (app, wakes)
}

/// Up or Down as the terminal view does them, on the history it browses
/// (the view's key reading needs a terminal, so this calls the input's
/// own Up and Down the view's keys run), then a frame.
fn browse(app: &mut App, up: bool) {
    let world = app.world_mut();
    let entries = world.resource::<PromptHistory>().entries.clone();
    edit_input(app, |input| {
        if up {
            input.up(&entries);
        } else {
            input.down(&entries);
        }
    });
    app.update();
}

fn input(app: &App) -> (String, usize) {
    let world = app.world();
    let mut main = world.try_query_filtered::<&Editor, With<MainEditor>>();
    let input = main.as_mut().and_then(|main| main.single(world).ok());
    input.map_or_else(Default::default, |input| {
        (input.text().to_owned(), input.cursor())
    })
}

/// Edits the default screen's input.
fn edit_input(app: &mut App, edit: impl FnOnce(&mut Editor)) {
    let world = app.world_mut();
    let mut main = world.query_filtered::<&mut Editor, With<MainEditor>>();
    if let Ok(mut input) = main.single_mut(world) {
        edit(&mut input);
    }
}

/// Only the first "a" is recorded, and only it is browsed after a resume;
/// the draft typed survives a restart, with the cursor at its end, and Up
/// and Down browse past it and back.
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
        let message = Deliver::new(agent, text, DeliveryMode::Queue);
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
    edit_input(&mut app, |input| input.set("half writ"));
    app.update();
    advance(&mut app, QUIET + Duration::from_millis(100));
    app.update();
    let saved = app.world().get::<Draft>(agent).cloned();
    assert_eq!(saved, Some(Draft("half writ".to_owned())));
    drop(app);

    let (mut app, _wakes) = start(&store, &home);
    assert_eq!(input(&app), ("half writ".to_owned(), 9));
    // The view seeds the resumed conversation's prompts before a key,
    // the secret too; this plugin's history takes their place.
    rig_tui::add_view(&mut app);
    app.world_mut().run_schedule(rig_tui::TuiKeys);
    assert_eq!(app.world().resource::<PromptHistory>().entries, ["a"]);
    // The first Up goes to the start of the draft, the next to the history.
    browse(&mut app, true);
    browse(&mut app, true);
    assert_eq!(input(&app).0, "a");
    browse(&mut app, false);
    assert_eq!(input(&app), ("half writ".to_owned(), 9));
    fs::remove_dir_all(home.root()).ok();
}
