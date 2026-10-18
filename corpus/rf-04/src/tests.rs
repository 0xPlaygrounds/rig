use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;

use rig_basics::approval::ApprovalPlugin;
use rig_basics::coding_tools::{EditTool, ReadTool, ShellTool};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::ask::AskPlugin;
use rig_harness_test_support::{
    AnswerAsks, Answers, MemoryStore, ToolOutputs, answer_open, app_on, calls, connect,
    log_tool_outputs, reply, run_until, start_call,
};
use serde_json::json;

use super::*;

/// A fresh, canonical directory for a test's files, beside the test's
/// binary, with `files` written in it.
fn scratch(name: &str, files: &[(&str, &str)]) -> PathBuf {
    let exe = std::env::current_exe().unwrap_or_default();
    let dir = exe.with_file_name(format!("rig-policy-{name}"));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).ok();
    let dir = std::fs::canonicalize(&dir).unwrap_or(dir);
    for (path, text) in files {
        let path = dir.join(path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        std::fs::write(path, text).ok();
    }
    dir
}

/// The kernel, questions, the stock approval service, the coding tools
/// and the policy, in the project `root` with `settings`, on `store`;
/// someone at it who answers from `script` when there is one, else nobody
/// (a `--print` run without `--answer`). After its first frame.
fn start(
    store: &MemoryStore,
    root: &Path,
    settings: &str,
    script: Option<Vec<Answer>>,
) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_on(store);
    app.insert_resource(Settings::from_toml(settings))
        .insert_resource(ProjectRoot(root.to_path_buf()))
        .add_plugins((AgentPlugin, SessionLogPlugin, AskPlugin, ApprovalPlugin))
        .add_plugins((ReadTool, EditTool, ShellTool, PolicyPlugin))
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs);
    if let Some(script) = script {
        app.add_plugins(AnswerAsks)
            .insert_resource(Answers::new(script));
    }
    app.finish();
    app.update();
    (app, wakes)
}

/// What the model read as call `id`'s result, once it settled.
fn output(world: &World, id: &str) -> Option<String> {
    world.resource::<ToolOutputs>().0.get(id).cloned()
}

fn settled(id: &'static str) -> impl Fn(&World) -> bool {
    move |world| output(world, id).is_some()
}

/// A new agent answered by `model`.
fn agent_on(app: &mut App, model: &MockCompletionModel) -> Entity {
    let mut agent = app.world_mut().spawn(Agent);
    if let Some(connection) = connect(model) {
        let choice = ModelChoice(connection.spec.reference());
        agent.insert((connection, choice));
    }
    agent.id()
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_default()
}

#[test]
fn run_tests_runs_where_each_calling_agent_works_and_a_confined_one_stays_inside() {
    let dir = scratch("places", &[("a/x", ""), ("b/x", ""), ("outside/x", "")]);
    let settings = r#"
        [policy]
        rules = [{ tool = "run_tests", action = "allow" }]
        [run_tests]
        command = ["sh", "-c", "echo tests ran in $(pwd)"]
    "#;
    let (mut app, wakes) = start(&MemoryStore::default(), &dir, settings, None);
    let world = app.world_mut();
    let a = world.spawn((Agent, WorkDir(dir.join("a")))).id();
    let b = world.spawn((Agent, WorkDir(dir.join("b")))).id();
    app.update();
    // Both agents' calls in one frame, and the confined agent reaching out.
    let world = app.world_mut();
    start_call(world, a, ("run_tests", "ta"), json!({}));
    start_call(world, b, ("run_tests", "tb"), json!({}));
    start_call(
        world,
        a,
        ("run_tests", "tx"),
        json!({ "path": "../outside" }),
    );
    run_until(&mut app, &wakes, |world| {
        ["ta", "tb", "tx"]
            .iter()
            .all(|id| output(world, id).is_some())
    });

    let world = app.world();
    let ran_in = |id, sub: &str| {
        let text = output(world, id).unwrap_or_default();
        let want = format!("tests ran in {}", dir.join(sub).display());
        assert!(text.contains(&want), "{id}: {text}");
    };
    ran_in("ta", "a");
    ran_in("tb", "b");
    let outside = output(world, "tx").unwrap_or_default();
    assert!(
        outside.contains("the only directory this tool works in") && !outside.contains("ran in"),
        "{outside}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn deny_refuses_before_anything_runs_and_ask_with_nobody_to_answer_refuses_at_once() {
    let dir = scratch("deny", &[("src/a.rs", "one\n")]);
    let (file, marker) = (dir.join("src/a.rs"), dir.join("marker"));
    let settings = r#"
        [policy]
        rules = [{ tool = "shell", command = "rm -rf", action = "deny", reason = "Deleting trees is blocked by policy." }]
    "#;
    let (mut app, wakes) = start(&MemoryStore::default(), &dir, settings, None);
    let agent = app.world_mut().spawn(Agent).id();
    app.update();
    let world = app.world_mut();
    let shell = json!({ "command": format!("touch {} && rm -rf junk", marker.display()) });
    start_call(world, agent, ("shell", "s1"), shell);
    let edit = json!({
        "path": file.display().to_string(),
        "edits": [{ "old_text": "one", "new_text": "two" }],
    });
    start_call(world, agent, ("edit", "e1"), edit);
    // One frame: both are refused at Start, nothing held, nothing run.
    app.update();
    let world = app.world();
    let denied = output(world, "s1").unwrap_or_default();
    assert!(
        denied.contains("Deleting trees is blocked by policy."),
        "{denied}"
    );
    let unasked = output(world, "e1").unwrap_or_default();
    assert!(unasked.contains("nobody can answer"), "{unasked}");
    assert!(!marker.exists());
    assert_eq!(read(&file), "one\n");
    // A read runs unasked.
    let path = json!({ "path": file.display().to_string() });
    start_call(app.world_mut(), agent, ("read", "r1"), path);
    run_until(&mut app, &wakes, settled("r1"));
    let text = output(app.world(), "r1").unwrap_or_default();
    assert!(text.contains("one"), "{text}");

    // Settings that do not read leave the policy at its default, which
    // still asks before a write, and are shown at startup.
    let broken = "[policy]\nrules = \"allow everything\"";
    let (mut app, _wakes) = start(&MemoryStore::default(), &dir, broken, None);
    assert!(!app.world().resource::<Settings>().problems().is_empty());
    let agent = app.world_mut().spawn(Agent).id();
    app.update();
    let edit = json!({
        "path": file.display().to_string(),
        "edits": [{ "old_text": "one", "new_text": "two" }],
    });
    start_call(app.world_mut(), agent, ("edit", "e2"), edit);
    app.update();
    let unasked = output(app.world(), "e2").unwrap_or_default();
    assert!(unasked.contains("nobody can answer"), "{unasked}");
    assert_eq!(read(&file), "one\n");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_held_edit_holds_back_only_what_depends_on_it_and_for_this_session_is_remembered() {
    let dir = scratch("held", &[("src/a.rs", "one\n"), ("src/b.rs", "bee\n")]);
    let (a, b) = (dir.join("src/a.rs"), dir.join("src/b.rs"));
    let edit = |from: &str, to: &str| {
        json!({
            "path": a.display().to_string(),
            "edits": [{ "old_text": from, "new_text": to }],
        })
    };
    // One reply: an edit of a.rs, a read of b.rs, a read of a.rs; then,
    // in the next turn, another edit.
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("e1", "edit", edit("one", "two")),
            MockStreamEvent::tool_call("rb", "read", json!({ "path": b.display().to_string() })),
            MockStreamEvent::tool_call("ra", "read", json!({ "path": a.display().to_string() })),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        reply("Edited."),
        calls("e2", "edit", edit("two", "three")),
        reply("Edited again."),
    ]);
    let (mut app, wakes) = start(&MemoryStore::default(), &dir, "", Some(Vec::new()));
    let agent = agent_on(&mut app, &model);
    let tell = |app: &mut App, text: &str| {
        app.world_mut()
            .trigger(Deliver::new(agent, text, DeliveryMode::Queue));
    };
    tell(&mut app, "Edit a.rs, read both.");
    run_until(&mut app, &wakes, settled("rb"));
    app.update();
    let world = app.world();
    assert!(output(world, "rb").unwrap_or_default().contains("bee"));
    assert_eq!(output(world, "e1"), None, "the edit waits for its answer");
    assert_eq!(
        output(world, "ra"),
        None,
        "the read of a.rs waits for the edit"
    );
    assert_eq!(world.resource::<Answers>().asked, ["Policy: allow `edit`?"]);

    let session = Answer::choice(SESSION);
    assert!(answer_open(
        app.world_mut(),
        "Policy: allow `edit`?",
        &session
    ));
    run_until(&mut app, &wakes, |world| {
        model.request_count() == 2 && world.get::<ActiveTurn>(agent).is_none()
    });
    assert!(
        output(app.world(), "ra")
            .unwrap_or_default()
            .contains("two")
    );

    // The same kind of call again runs without asking.
    tell(&mut app, "Edit it again.");
    run_until(&mut app, &wakes, |world| {
        model.request_count() == 4 && world.get::<ActiveTurn>(agent).is_none()
    });
    assert_eq!(read(&a), "three\n");
    assert_eq!(app.world().resource::<Answers>().asked.len(), 1);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_restart_while_a_call_waits_asks_again_and_keeps_only_this_sessions_approvals() {
    let dir = scratch("restart", &[("src/a.rs", "one\n")]);
    let file = dir.join("src/a.rs");
    let edit = json!({
        "path": file.display().to_string(),
        "edits": [{ "old_text": "one", "new_text": "two" }],
    });
    let model = MockCompletionModel::from_stream_turns([calls("e1", "edit", edit), reply("Ok.")]);
    let store = MemoryStore::default();
    let asked = |world: &World| !world.resource::<Answers>().asked.is_empty();
    let remembered = "shell default".to_owned();
    {
        let (mut app, wakes) = start(&store, &dir, "", Some(Vec::new()));
        let agent = agent_on(&mut app, &model);
        let world = app.world_mut();
        world
            .resource_mut::<SessionApprovals>()
            .0
            .push(remembered.clone());
        world.trigger(Deliver::new(agent, "Edit a.rs.", DeliveryMode::Queue));
        run_until(&mut app, &wakes, asked);
        // The question is open and nobody answered when the app goes.
        app.update();
    }
    assert_eq!(read(&file), "one\n");

    let (mut app, wakes) = start(&store, &dir, "", Some(Vec::new()));
    run_until(&mut app, &wakes, asked);
    let world = app.world();
    assert_eq!(world.resource::<Answers>().asked, ["Policy: allow `edit`?"]);
    assert_eq!(output(world, "e1"), None, "the restored call waits again");
    assert_eq!(read(&file), "one\n");
    assert_eq!(world.resource::<SessionApprovals>().0, [remembered]);

    // A new session remembers nothing.
    let (app, _wakes) = start(&MemoryStore::default(), &dir, "", Some(Vec::new()));
    assert!(app.world().resource::<SessionApprovals>().0.is_empty());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered((ApprovalPlugin, PolicyPlugin));
}
