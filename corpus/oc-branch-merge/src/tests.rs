use std::fs;
use std::path::PathBuf;
use std::sync::mpsc::Receiver;

use rig_coding_tools::WriteTool;
use rig_core::test_utils::MockCompletionModel;
use rig_harness::ask::AskPlugin;
use rig_harness::harness_protocol::SessionId;
use rig_harness::rig_ecs::journal::JournalPlugin;
use rig_harness_test_support::{
    AnswerAsks, Answers, Collected, Ended, app_in, calls, collect, connect, first_agent, log_ended,
    reply, run_until, scratch_home,
};
use serde_json::json;

use super::*;

/// What the scripted `shell` prints.
const TEST_OK: &str = "test result: ok. 12 passed\n[exit code 0]";

/// A `shell` that only says the tests passed.
fn shell(called: On<ToolCalled<serde_json::Value>>, mut calls: Query<&mut Decision, ToolCalls>) {
    if let Ok(mut call) = calls.get_mut(called.call) {
        call.decide(
            Decider::plugin("shell"),
            ToolResult::success(TEST_OK.into()),
        );
    }
}

fn project(home: &Home) -> PathBuf {
    home.root().join("project")
}

fn start(home: &Home, id: &SessionId) -> (App, Receiver<()>) {
    let (mut app, wakes) = app_in(home, id);
    app.insert_resource(ProjectRoot(project(home)))
        .add_plugins((AgentPlugin, JournalPlugin, AskPlugin, WriteTool, AnswerAsks))
        .add_plugins(BranchMergePlugin)
        .add_open_tool("shell", "Run a command.", ToolOptions::default(), shell)
        .init_resource::<Ended>()
        .add_observer(log_ended);
    collect::<Recalled>(&mut app);
    app.finish();
    app.update();
    (app, wakes)
}

fn ask(app: &mut App, wakes: &Receiver<()>, agent: Entity, text: &str) {
    let ended = app.world().resource::<Ended>().0.len();
    let typed = Deliver::new(agent, text, DeliveryMode::Queue).with_author(Author::user());
    app.world_mut().trigger(typed);
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() > ended
    });
}

/// Runs `/line` and a few frames after it.
fn run(app: &mut App, wakes: &Receiver<()>, agent: Entity, line: &str) {
    app.world_mut().trigger(RunCommand::new(agent, line));
    for _ in 0..3 {
        app.update();
    }
    wakes.try_iter().count();
}

fn branches(app: &App, agent: Entity) -> Branches {
    app.world()
        .get::<Branches>(agent)
        .cloned()
        .unwrap_or_default()
}

/// Waits for merge number `n` to be applied.
fn merged(app: &mut App, wakes: &Receiver<()>, agent: Entity, n: usize) {
    run_until(app, wakes, |world| {
        world
            .get::<Branches>(agent)
            .is_some_and(|branches| branches.merges.len() >= n)
    });
    for _ in 0..3 {
        app.update();
    }
}

/// The text of branch `name`'s messages.
fn line_text(app: &App, agent: Entity, name: &str) -> String {
    let Some(conversation) = app.world().get::<Conversation>(agent) else {
        return String::new();
    };
    let live = entries(conversation);
    format!("{:?}", branches(app, agent).line(name, Some(&live)))
}

fn last_request(model: &MockCompletionModel) -> String {
    format!("{:?}", model.requests().last())
}

/// Why each refused command was refused.
fn errors(app: &App) -> Vec<String> {
    let recalled = &app.world().resource::<Collected<Recalled>>().0;
    recalled
        .iter()
        .filter_map(|line| line.why.clone())
        .collect()
}

#[test]
fn a_side_branch_is_merged_as_one_note_and_only_once_per_message() {
    let home = scratch_home("oc-branch-merge");
    fs::create_dir_all(project(&home)).ok();
    let x = project(&home).join("x.rs");
    fs::write(&x, "broken\n").ok();
    let model = MockCompletionModel::from_stream_turns([
        reply("shared one"),
        reply("shared two"),
        reply("feature step on A"),
        calls(
            "w1",
            "write",
            json!({ "path": "x.rs", "content": "fixed\n" }),
        ),
        reply("wrote x.rs"),
        calls(
            "s1",
            "shell",
            json!({ "command": "cargo test -- --test-threads=1" }),
        ),
        reply("root cause: shared temp dir"),
        reply("third B reply"),
        reply("- root cause: shared temp dir\n- worked: cargo test -- --test-threads=1\n- x.rs"),
        reply("noted on A"),
        reply("fourth B reply"),
        reply("- fourth turn: nothing new"),
        reply("C reply"),
        reply("- C summary"),
    ]);
    let id = SessionId::generate();
    let (mut app, wakes) = start(&home, &id);
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    if let Some(connection) = connect(&model) {
        app.world_mut().entity_mut(agent).insert(connection);
    }
    *app.world_mut().resource_mut::<Answers>() =
        Answers::new([Answer::Choice("accept".to_owned())]);

    ask(&mut app, &wakes, agent, "first shared");
    ask(&mut app, &wakes, agent, "second shared");
    run(&mut app, &wakes, agent, "branch A");
    ask(&mut app, &wakes, agent, "work on the feature");
    run(&mut app, &wakes, agent, "switch main");
    run(&mut app, &wakes, agent, "branch B");
    ask(&mut app, &wakes, agent, "why is CI flaky?");
    ask(&mut app, &wakes, agent, "run the tests serially");
    ask(&mut app, &wakes, agent, "anything else?");
    assert_eq!(fs::read_to_string(&x).ok().as_deref(), Some("fixed\n"));
    run(&mut app, &wakes, agent, "switch A");
    // A never wrote x.rs: the switch put it back as A has it.
    assert_eq!(fs::read_to_string(&x).ok().as_deref(), Some("broken\n"));

    run(&mut app, &wakes, agent, "merge B");
    merged(&mut app, &wakes, agent, 1);
    let summarizer = last_request(&model);
    assert!(summarizer.contains("why is CI flaky?"), "{summarizer}");
    assert!(!summarizer.contains("work on the feature"), "{summarizer}");
    ask(&mut app, &wakes, agent, "go on with the feature");
    let next = last_request(&model);
    assert!(
        next.contains(MERGED) && next.contains("shared temp dir"),
        "{next}"
    );
    assert!(
        next.contains("first shared") && next.contains("work on the feature"),
        "{next}"
    );
    assert!(
        !next.contains("why is CI flaky?") && !next.contains("third B reply"),
        "{next}"
    );

    // A fourth turn on B; the second merge summarizes it alone.
    run(&mut app, &wakes, agent, "switch B");
    ask(&mut app, &wakes, agent, "a fourth question");
    run(&mut app, &wakes, agent, "switch A");
    let written = branches(&app, agent)
        .get("B")
        .and_then(|b| b.files.first().cloned());
    let written = written.map(|record| record.path).unwrap_or_default();
    let mut answers = app.world_mut().resource_mut::<Answers>();
    answers
        .script
        .push_back(Answer::Choice("accept".to_owned()));
    answers.script.push_back(Answer::Choices(vec![written]));
    run(&mut app, &wakes, agent, "merge B --with-files");
    merged(&mut app, &wakes, agent, 2);
    let summarizer = last_request(&model);
    assert!(summarizer.contains("a fourth question"), "{summarizer}");
    assert!(!summarizer.contains("why is CI flaky?"), "{summarizer}");
    run_until(&mut app, &wakes, |world| {
        world.resource::<Answers>().asked.len() >= 3
    });
    run_until(&mut app, &wakes, |_| {
        fs::read_to_string(&x).is_ok_and(|it| it == "fixed\n")
    });
    assert_eq!(fs::read_to_string(&x).ok().as_deref(), Some("fixed\n"));

    // C off B: into B is a merge; B into C or into itself is refused.
    run(&mut app, &wakes, agent, "switch B");
    run(&mut app, &wakes, agent, "branch C");
    ask(&mut app, &wakes, agent, "try C");
    let refused = errors(&app).len();
    let mut answers = app.world_mut().resource_mut::<Answers>();
    answers
        .script
        .push_back(Answer::Choice("cancel".to_owned()));
    run(&mut app, &wakes, agent, "merge C into B");
    run_until(&mut app, &wakes, |world| {
        world.resource::<Answers>().asked.len() >= 4
    });
    assert_eq!(errors(&app).len(), refused, "{:?}", errors(&app));
    run(&mut app, &wakes, agent, "merge B into C");
    run(&mut app, &wakes, agent, "merge B into B");
    let errors = errors(&app);
    let refusals = errors.get(refused..).unwrap_or_default();
    let descendant = refusals.iter().any(|it| it.contains("descendant"));
    assert!(refusals.len() == 2 && descendant, "{errors:?}");

    // Reloaded, the notes are on A alone.
    drop(app);
    let (mut app, wakes) = start(&home, &id);
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    run_until(&mut app, &wakes, |world| {
        world
            .get::<Branches>(agent)
            .is_some_and(|branches| branches.list.len() == 4)
    });
    assert_eq!(branches(&app, agent).active, "C");
    let a = line_text(&app, agent, "A");
    assert_eq!(a.matches(MERGED).count(), 2, "{a}");
    for other in ["main", "B", "C"] {
        let line = line_text(&app, agent, other);
        assert!(!line.contains(MERGED), "{other}: {line}");
    }
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(BranchMergePlugin);
}
