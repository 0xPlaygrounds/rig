use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;

use rig_basics::coding_tools::{EditTool, WriteTool};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::ask::AskPlugin;
use rig_harness::rig_ecs::AgentPlugin;
use rig_harness::rig_ecs::session_log::SessionLogPlugin;
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{
    AnswerAsks, Answers, Ended, ToolOutputs, answer_open, app_on, calls, connect, first_agent,
    log_ended, log_tool_outputs, reply, run_until,
};
use serde_json::json;

use super::*;

/// A project directory named `name` with `big.rs` (200 lines) and
/// `small.rs` (5 lines).
fn project(name: &str) -> PathBuf {
    let dir = std::env::current_exe()
        .unwrap_or_default()
        .with_file_name(name);
    std::fs::remove_dir_all(&dir).ok();
    assert!(std::fs::create_dir_all(&dir).is_ok());
    let dir = std::fs::canonicalize(&dir).unwrap_or(dir);
    assert!(std::fs::write(dir.join("big.rs"), big()).is_ok());
    assert!(std::fs::write(dir.join("small.rs"), "a\nb\nc\nd\ne\n").is_ok());
    dir
}

fn big() -> String {
    (1..=200)
        .map(|n| format!("fn helper_{n}() {{}}\n"))
        .collect()
}

/// `big.rs` cut to its first 10 lines and 10 new ones: −190 / +10.
fn rewrite() -> String {
    let kept = (1..=10).map(|n| format!("fn helper_{n}() {{}}\n"));
    kept.chain((1..=10).map(|n| format!("fn merged_{n}() {{}}\n")))
        .collect()
}

fn text(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_default()
}

fn write_call(id: &str, path: &Path, content: &str) -> MockStreamEvent {
    MockStreamEvent::tool_call(id, "write", json!({ "path": path, "content": content }))
}

/// An app on `store` in the project `dir` with the file tools and the
/// guard, its agent on `model`; with `answers`, someone answers questions
/// from that script.
fn start(
    store: &MemoryStore,
    dir: &Path,
    model: &MockCompletionModel,
    answers: Option<Answers>,
) -> (App, Receiver<()>, Entity) {
    let (mut app, wakes) = app_on(store);
    app.insert_resource(ProjectRoot(dir.to_path_buf()))
        .add_plugins((AgentPlugin, SessionLogPlugin, AskPlugin))
        .add_plugins((WriteTool, EditTool))
        .add_plugins(LargeChangeGuardPlugin)
        .init_resource::<Ended>()
        .add_observer(log_ended)
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs);
    if let Some(answers) = answers {
        app.insert_resource(answers).add_plugins(AnswerAsks);
    }
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = first_agent(world).unwrap_or(Entity::PLACEHOLDER);
    if let Some(connection) = connect(model) {
        world.entity_mut(agent).insert(connection);
    }
    (app, wakes, agent)
}

fn asked(app: &App) -> Vec<String> {
    app.world().resource::<Answers>().asked.clone()
}

fn ended(world: &World) -> usize {
    world.resource::<Ended>().0.len()
}

#[test]
fn a_large_change_waits_for_the_user_and_small_ones_do_not() {
    let dir = project("sn-large-change-guard");
    let (big_rs, small_rs) = (dir.join("big.rs"), dir.join("small.rs"));
    let edit = json!({ "path": small_rs, "edits": [{ "old_text": "c", "new_text": "C" }] });
    let model = MockCompletionModel::from_stream_turns([
        vec![
            write_call("1", &big_rs, &rewrite()),
            MockStreamEvent::tool_call("2", "edit", edit),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        calls(
            "3",
            "write",
            json!({ "path": big_rs, "content": rewrite() }),
        ),
        reply("Rewrote big.rs."),
        calls(
            "4",
            "write",
            json!({ "path": big_rs, "content": "fn one() {}\n" }),
        ),
        reply("Rewrote it again."),
    ]);
    let answers = Answers::new([
        Answer::choice("deny_reason"),
        Answer::text("keep the helpers"),
    ]);
    let store = MemoryStore::default();
    let (mut app, wakes, agent) = start(&store, &dir, &model, Some(answers));
    app.world_mut()
        .trigger(Deliver::new(agent, "Simplify big.rs.", DeliveryMode::Steer));

    // Reply 1: the write is denied with a reason; the small edit ran.
    run_until(&mut app, &wakes, |world| {
        world.resource::<Answers>().asked.len() == 3
    });
    let titles = asked(&app);
    let first = titles.first().cloned().unwrap_or_default();
    assert!(
        first.ends_with("big.rs: −190 / +10 lines (95% of file replaced), 1 KB"),
        "{first}"
    );
    assert_eq!(text(&big_rs), big());
    assert_eq!(text(&small_rs), "a\nb\nC\nd\ne\n");
    let second = format!("{:?}", model.requests().get(1));
    assert!(
        second.contains("−190") && second.contains("keep the helpers"),
        "{second}"
    );

    // Reply 2: the same write, allowed for the session.
    let third = titles.get(2).cloned().unwrap_or_default();
    let allow_file = Answer::choice("allow_file");
    assert!(answer_open(app.world_mut(), &third, &allow_file));
    run_until(&mut app, &wakes, |world| ended(world) == 1);
    assert_eq!(text(&big_rs), rewrite());

    // A later turn rewrites it whole: nobody is asked.
    app.world_mut()
        .trigger(Deliver::new(agent, "Again.", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| ended(world) == 2);
    assert_eq!(text(&big_rs), "fn one() {}\n");
    assert_eq!(asked(&app).len(), 3);

    // The allowed file is saved with the session.
    drop(app);
    let unused = MockCompletionModel::from_stream_turns([reply("unused")]);
    let (resumed, _, _) = start(&store, &dir, &unused, None);
    let allowed = &resumed.world().resource::<AllowedFiles>().files;
    assert_eq!(allowed, &[big_rs.display().to_string()]);
}

#[test]
fn a_run_nobody_answers_denies_a_large_change() {
    let dir = project("sn-large-change-guard-headless");
    let big_rs = dir.join("big.rs");
    let model = MockCompletionModel::from_stream_turns([
        calls(
            "1",
            "write",
            json!({ "path": big_rs, "content": rewrite() }),
        ),
        reply("Could not."),
    ]);
    let (mut app, wakes, agent) = start(&MemoryStore::default(), &dir, &model, None);
    app.world_mut()
        .trigger(Deliver::new(agent, "Simplify big.rs.", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| ended(world) == 1);
    let outputs = &app.world().resource::<ToolOutputs>().0;
    let output = outputs.get("1").cloned().unwrap_or_default();
    assert!(
        output.contains(HEADLESS) && output.contains("−190"),
        "{output}"
    );
    assert_eq!(text(&big_rs), big());
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(LargeChangeGuardPlugin);
}
