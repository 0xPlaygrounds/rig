use std::path::PathBuf;
use std::sync::mpsc::Receiver;

use rig_basics::approval::ApprovalPlugin;
use rig_basics::coding_tools::{ShellTool, WriteTool};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::ask::AskPlugin;
use rig_harness::clock::ClockPlugin;
use rig_harness_test_support::{
    Collected, Ended, MemoryStore, ToolOutputs, app_on, calls, collect, connect, first_agent,
    log_ended, log_tool_outputs, pass_wall_clock, reply, run_until,
};
use serde_json::json;

use super::*;

/// The stand-in approval rule: every `shell` and `write` call asks.
fn ask_always(
    calls: Query<(Entity, &ToolRequest), (ReadyToStart, ToolCalls, Without<Approved>)>,
    mut commands: Commands,
) {
    for (call, request) in &calls {
        if matches!(request.call.function.name.as_str(), "shell" | "write") {
            commands
                .entity(call)
                .queue(NeedsApproval::add("ask-always", "It asks."));
        }
    }
}

/// The stand-in protected-path rule: nothing under `src/locked` is written.
fn protect(mut calls: Query<(&Footprint, &mut Decision), (ReadyToStart, ToolCalls)>) {
    for (footprint, mut decision) in &mut calls {
        let locked = written(Some(&footprint.0)).is_some_and(|paths| {
            paths
                .iter()
                .any(|path| path.components().any(|part| part.as_os_str() == "locked"))
        });
        if locked {
            decision.decide(
                Decider::plugin("protect"),
                Outcome::refused("protected path"),
            );
        }
    }
}

fn project(name: &str) -> PathBuf {
    let exe = std::env::current_exe().unwrap_or_default();
    let dir = exe.with_file_name(format!("timed-grants-{name}-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    for sub in ["src/locked", "docs"] {
        std::fs::create_dir_all(dir.join(sub)).ok();
    }
    std::fs::canonicalize(&dir).unwrap_or(dir)
}

fn write(id: &str, path: &str) -> MockStreamEvent {
    MockStreamEvent::tool_call(id, "write", json!({ "path": path, "content": "x\n" }))
}

fn start(
    name: &str,
    model: &MockCompletionModel,
    store: &MemoryStore,
) -> (App, Receiver<()>, Entity, PathBuf) {
    let dir = project(name);
    let (mut app, wakes) = app_on(store);
    app.insert_resource(ProjectRoot(dir.clone()))
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            ClockPlugin,
            AskPlugin,
            ApprovalPlugin,
            ShellTool,
            WriteTool,
            TimedGrantsPlugin,
        ))
        .add_systems(AgentPass, (ask_always, protect).in_set(Stage::Gate))
        .init_resource::<Ended>()
        .add_observer(log_ended)
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs);
    collect::<Notice>(&mut app);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    if let Some(connection) = connect(model) {
        app.world_mut().entity_mut(agent).insert(connection);
    }
    (app, wakes, agent, dir)
}

fn turn(app: &mut App, wakes: &Receiver<()>, agent: Entity, text: &str) {
    let ended = app.world().resource::<Ended>().0.len();
    app.world_mut()
        .trigger(Deliver::new(agent, text, DeliveryMode::Queue));
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() > ended
    });
}

fn command(app: &mut App, agent: Entity, line: &str) {
    app.world_mut().trigger(RunCommand::new(agent, line));
    app.update();
}

fn output(app: &App, id: &str) -> String {
    let outputs = &app.world().resource::<ToolOutputs>().0;
    outputs.get(id).cloned().unwrap_or_default()
}

fn noticed(app: &App, text: &str) -> usize {
    let notices = &app.world().resource::<Collected<Notice>>().0;
    notices
        .iter()
        .filter(|notice| notice.text.contains(text))
        .count()
}

fn live(app: &mut App) -> usize {
    let mut grants = app.world_mut().query::<&Grant>();
    grants.iter(app.world()).count()
}

const ASKED: &str = "nobody can answer";

#[test]
fn grants_cover_calls_until_they_run_out() {
    let shell = |id: &str| calls(id, "shell", json!({ "command": "echo hi" }));
    let model = MockCompletionModel::from_stream_turns([
        shell("s1"),
        reply("one"),
        shell("s2"),
        reply("two"),
        shell("s3"),
        reply("three"),
        vec![
            write("a", "src/a.rs"),
            write("x", "docs/x.md"),
            write("k", "src/locked/k.rs"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        reply("four"),
        calls(
            "b",
            "write",
            json!({ "path": "src/b.rs", "content": "x\n" }),
        ),
        reply("five"),
    ]);
    let (mut app, wakes, agent, dir) = start("cover", &model, &MemoryStore::default());

    // Two turns of `shell` run unasked; the third asks, and is denied
    // headless.
    command(&mut app, agent, "grant shell for 2 turns");
    for n in 1..=3 {
        turn(&mut app, &wakes, agent, &format!("turn {n}"));
    }
    assert!(output(&app, "s1").contains("hi"), "{}", output(&app, "s1"));
    assert!(output(&app, "s2").contains("hi"), "{}", output(&app, "s2"));
    assert!(output(&app, "s3").contains(ASKED), "{}", output(&app, "s3"));
    assert_eq!(noticed(&app, "Grant shell ran out"), 1);

    // Edits under `src/` until the turn ends: `docs/` asks, and the
    // protected path stays refused; the next turn asks again.
    command(&mut app, agent, "grant edits:src/** until turn-end");
    turn(&mut app, &wakes, agent, "turn 4");
    assert!(dir.join("src/a.rs").exists(), "{}", output(&app, "a"));
    assert!(output(&app, "x").contains(ASKED), "{}", output(&app, "x"));
    assert!(
        output(&app, "k").contains("protected path"),
        "{}",
        output(&app, "k")
    );
    assert!(!dir.join("src/locked/k.rs").exists());
    turn(&mut app, &wakes, agent, "turn 5");
    assert!(output(&app, "b").contains(ASKED), "{}", output(&app, "b"));
    assert_eq!(live(&mut app), 0);

    // Ten minutes, counted down; eleven later it ran out, said once.
    command(&mut app, agent, "grant shell for 10m");
    let id = app
        .world()
        .get::<AgentId>(agent)
        .cloned()
        .unwrap_or_default();
    let mut grants = app.world_mut().query::<(&Grant, Option<&WakeAt>)>();
    let status = soonest(&id, grants.iter(app.world()), SystemTime::now());
    assert!(
        status
            .as_deref()
            .is_some_and(|it| it.starts_with("grant shell 9:5")),
        "{status:?}"
    );
    let saved = app
        .world()
        .get::<SavedGrants>(agent)
        .cloned()
        .unwrap_or_default();
    assert!(
        saved
            .0
            .first()
            .is_some_and(|grant| grant.until_ms.is_some()),
        "{saved:?}"
    );
    pass_wall_clock(app.world_mut(), Duration::from_secs(11 * 60));
    for _ in 0..3 {
        app.update();
    }
    command(&mut app, agent, "grants");
    assert_eq!(live(&mut app), 0);
    assert_eq!(noticed(&app, "No live grants."), 1);
    assert_eq!(noticed(&app, "Grant shell ran out"), 2);
    assert_eq!(
        app.world()
            .get::<SavedGrants>(agent)
            .map(|saved| saved.0.len()),
        Some(0)
    );
}

#[test]
fn resume_keeps_turns_left_and_drops_passed_time() {
    let model = MockCompletionModel::from_stream_turns([reply("one")]);
    let store = MemoryStore::default();
    let (mut app, wakes, agent, _) = start("resume", &model, &store);
    command(&mut app, agent, "grant shell for 3 turns");
    turn(&mut app, &wakes, agent, "turn 1");
    command(&mut app, agent, "grant write for 1s");
    command(&mut app, agent, "grant edits for 1h");
    assert_eq!(live(&mut app), 3);
    drop(app);
    // The second's time passes while the session is closed.
    std::thread::sleep(Duration::from_millis(1100));

    let (mut app, _wakes, _, _) = start("resume", &model, &store);
    let mut grants = app.world_mut().query::<&Grant>();
    let mut kept: Vec<(String, Lasts)> = grants
        .iter(app.world())
        .map(|grant| (grant.cover.label(), grant.lasts))
        .collect();
    kept.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(
        kept,
        [
            ("edits".to_owned(), Lasts::Time),
            ("shell".to_owned(), Lasts::Turns(2))
        ]
    );
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(TimedGrantsPlugin);
}
