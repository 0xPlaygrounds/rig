use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;

use rig_basics::coding_tools::EditTool;
use rig_basics::subagents::SubagentsPlugin;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness_test_support::{
    Collected, MemoryStore, app_on, calls, collect, reply, run_until, scripted_models, tui_view,
};
use rig_tui::{ShowAgent, TuiTestScreen};
use serde_json::json;

use super::*;

const MAIN_MODEL: &str = "churn/main";
const SUB_MODEL: &str = "churn/sub";

/// A project beside the test binary with `src/a.rs` and `src/b.rs`.
fn project(name: &str) -> PathBuf {
    let dir = std::env::current_exe()
        .unwrap_or_default()
        .with_file_name(name);
    std::fs::remove_dir_all(&dir).ok();
    assert!(std::fs::create_dir_all(dir.join("src")).is_ok());
    assert!(std::fs::write(dir.join("src/a.rs"), "one\ntwo\nthree\n").is_ok());
    assert!(std::fs::write(dir.join("src/b.rs"), "fn b() {}\n").is_ok());
    std::fs::canonicalize(&dir).unwrap_or(dir)
}

/// A reply that edits `path`, replacing `old` with `new`.
fn edit(id: &str, path: &str, old: &str, new: &str) -> Vec<MockStreamEvent> {
    let edits = [json!({ "old_text": old, "new_text": new })];
    calls(id, "edit", json!({ "path": path, "edits": edits }))
}

/// Turn 1: four edits of `src/a.rs`, one adding `let x = 1;`. Turn 2:
/// three more, one removing it. 7 calls, 6 lines added, 4 removed, 1
/// reverted.
fn thrash() -> Vec<Vec<MockStreamEvent>> {
    let a = "src/a.rs";
    vec![
        edit("e1", a, "two\n", "two\nlet x = 1;\n"),
        edit("e2", a, "one\n", "one\nb\n"),
        edit("e3", a, "three\n", "three\nc\n"),
        edit("e4", a, "one\n", "ONE\n"),
        reply("Turn 1 done."),
        edit("e5", a, "two\nlet x = 1;\n", "two\n"),
        edit("e6", a, "two\n", "TWO\n"),
        edit("e7", a, "three\n", "THREE\n"),
        reply("Turn 2 done."),
    ]
}

/// An app on `store` for the project `dir` with the edit tool, subagents,
/// the plugin under `settings` and the terminal view on a test screen; its
/// main agent on the scripted main model.
fn start(
    store: &MemoryStore,
    dir: &Path,
    settings: &str,
    models: Models,
) -> (App, Receiver<()>, Entity) {
    let (mut app, wakes) = app_on(store);
    app.insert_resource(models)
        .insert_resource(ProjectRoot(dir.to_path_buf()))
        .insert_resource(Settings::from_toml(settings))
        .insert_resource(TuiTestScreen::new(140, 30))
        .add_plugins((AgentPlugin, SessionLogPlugin, EditTool, SubagentsPlugin))
        .add_plugins(ChurnMonitorPlugin);
    collect::<Notice>(&mut app);
    tui_view(&mut app);
    app.finish();
    app.cleanup();
    app.update();
    let agent = main_agent(app.world_mut());
    app.world_mut().trigger(SetModel {
        entity: agent,
        model: MAIN_MODEL.to_owned(),
        kind: ChoiceKind::ForThisRun,
    });
    // The terminal view shows the main agent, as a front does.
    app.world_mut().trigger(ShowAgent { entity: agent });
    (app, wakes, agent)
}

fn main_agent(world: &mut World) -> Entity {
    let mut agents = world.query_filtered::<Entity, (With<Agent>, Without<SpawnedBy>)>();
    agents.iter(world).next().unwrap_or(Entity::PLACEHOLDER)
}

/// Sends `text` as the user and runs until no turn runs and `main` was
/// asked `requests` times in all.
fn turn(
    app: &mut App,
    wakes: &Receiver<()>,
    agent: Entity,
    text: &str,
    done: impl Fn(&World) -> bool,
) {
    app.world_mut()
        .trigger(Deliver::new(agent, text, DeliveryMode::Steer));
    app.update();
    run_until(app, wakes, |world| idle(world) && done(world));
}

fn idle(world: &World) -> bool {
    world
        .try_query::<&TurnOf>()
        .is_some_and(|mut turns| turns.iter(world).next().is_none())
}

/// What `/churn` shows now.
fn churn_table(app: &mut App, agent: Entity) -> String {
    app.world_mut()
        .trigger(RunCommand::new(agent, "churn".to_owned()));
    for _ in 0..3 {
        app.update();
    }
    let notices = &app.world().resource::<Collected<Notice>>().0;
    let shown = notices
        .iter()
        .rev()
        .find(|notice| notice.text.starts_with("file"));
    shown.map(|notice| notice.text.clone()).unwrap_or_default()
}

/// The cells of the table's row for `file` and `agent`.
fn row(table: &str, file: &str, agent: &str) -> Vec<String> {
    let rows = table.lines().map(|line| {
        line.split_whitespace()
            .map(str::to_owned)
            .collect::<Vec<_>>()
    });
    let mut rows = rows.filter(|cells| cells.first().map(String::as_str) == Some(file));
    let found = rows.find(|cells| cells.get(1).map(String::as_str) == Some(agent));
    found.unwrap_or_default()
}

fn churn_warnings(app: &App) -> usize {
    let notices = &app.world().resource::<Collected<Notice>>().0;
    let warns = notices
        .iter()
        .filter(|notice| notice.text.starts_with("churn:"));
    warns.count()
}

fn screen(app: &App) -> String {
    app.world().resource::<TuiTestScreen>().rows().join("\n")
}

fn main_turn(world: &World) -> u64 {
    let churn = world.resource::<Churn>();
    let main = churn.agents.values().find(|agent| agent.name == MAIN);
    main.map_or(0, |agent| agent.turn)
}

#[test]
fn thrashing_a_file_shows_churn_until_the_window_moves_on_and_survives_a_resume() {
    let dir = project("sp-churn-monitor-project");
    let task = json!({ "description": "helper", "prompt": "Edit src/b.rs.", "model": SUB_MODEL });
    let mut script = thrash();
    script.extend([
        calls("t", "task", task),
        reply("Started a helper."),
        reply("The helper is done."),
        reply("Nothing to change."),
        reply("Still nothing."),
        reply("Nothing again."),
    ]);
    let main = MockCompletionModel::from_stream_turns(script);
    let sub = MockCompletionModel::from_stream_turns([
        edit(
            "s1",
            "src/b.rs",
            "fn b() {}\n",
            "fn b() -> u8 {\n    1\n}\n",
        ),
        reply("Edited b."),
    ]);
    let models = || scripted_models([(MAIN_MODEL, &main), (SUB_MODEL, &sub)]);
    let store = MemoryStore::default();
    let (mut app, wakes, agent) = start(&store, &dir, "", models());

    let asked = |n: usize| {
        let main = main.clone();
        move |_: &World| main.requests().len() >= n
    };
    turn(&mut app, &wakes, agent, "Fix the types.", asked(5));
    turn(&mut app, &wakes, agent, "Still broken.", asked(9));
    for _ in 0..3 {
        app.update();
    }
    let text = std::fs::read_to_string(dir.join("src/a.rs")).unwrap_or_default();
    assert_eq!(text, "ONE\nb\nTWO\nTHREE\nc\n");
    let shown = screen(&app);
    assert!(shown.contains("churn: src/a.rs ×7"), "{shown}");
    assert_eq!(churn_warnings(&app), 1);
    let table = churn_table(&mut app, agent);
    assert_eq!(
        row(&table, "src/a.rs", MAIN),
        ["src/a.rs", "main", "7", "+6", "−4", "1"],
        "{table}"
    );

    // The user's own change is not the agent's.
    assert!(std::fs::write(dir.join("src/a.rs"), "ONE\nb\nTWO\nTHREE\nc\nlet x = 1;\n").is_ok());
    assert_eq!(churn_table(&mut app, agent), table);

    // Turn 3 starts a helper that edits `src/b.rs` once; turns 3 to 5 of
    // the main agent change nothing.
    turn(&mut app, &wakes, agent, "Have a helper fix b.", asked(12));
    while main_turn(app.world()) < 4 && main.requests().len() < 14 {
        let before = main.requests().len();
        turn(&mut app, &wakes, agent, "Anything else?", asked(before + 1));
    }
    assert_eq!(main_turn(app.world()), 4);
    for _ in 0..3 {
        app.update();
    }
    let shown = screen(&app);
    assert!(!shown.contains("churn: src/a.rs ×"), "{shown}");
    assert_eq!(churn_warnings(&app), 1);
    let table = churn_table(&mut app, agent);
    assert_eq!(
        row(&table, "src/b.rs", "helper"),
        ["src/b.rs", "helper", "1", "+3", "−1", "0"],
        "{table}"
    );
    assert_eq!(
        row(&table, "src/a.rs", MAIN).get(2).map(String::as_str),
        Some("7"),
        "{table}"
    );
    let nudged = main.requests().iter().any(|request| {
        serde_json::to_string(request)
            .unwrap_or_default()
            .contains("[churn monitor]")
    });
    assert!(!nudged, "nothing goes to the model by default");

    // A restart that resumes the session shows the same table.
    drop(app);
    let (mut app, _wakes, agent) = start(&store, &dir, "", models());
    assert_eq!(churn_table(&mut app, agent), table);
}

#[test]
fn a_nudge_goes_with_the_first_request_of_an_episode_only() {
    let dir = project("sp-churn-monitor-nudge");
    let mut script = thrash();
    script.push(reply("Turn 3 done."));
    let main = MockCompletionModel::from_stream_turns(script);
    let models = scripted_models([(MAIN_MODEL, &main)]);
    let store = MemoryStore::default();
    let (mut app, wakes, agent) = start(&store, &dir, "[churn]\nnudge = true\n", models);
    let asked = |n: usize| {
        let main = main.clone();
        move |_: &World| main.requests().len() >= n
    };
    turn(&mut app, &wakes, agent, "Fix the types.", asked(5));
    turn(&mut app, &wakes, agent, "Still broken.", asked(9));
    turn(&mut app, &wakes, agent, "And now?", asked(10));
    let notes: Vec<usize> = main
        .requests()
        .iter()
        .map(|request| {
            let sent = serde_json::to_string(request).unwrap_or_default();
            sent.matches("[churn monitor]").count()
        })
        .collect();
    // The request after the seventh edit, and no other.
    assert_eq!(notes, [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]);
    let after = main
        .requests()
        .get(8)
        .map(|request| serde_json::to_string(request).unwrap_or_default());
    assert!(after.is_some_and(|sent| sent.contains("src/a.rs")));
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(ChurnMonitorPlugin);
}
