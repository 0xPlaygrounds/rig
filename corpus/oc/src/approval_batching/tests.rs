use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;

use rig_basics::coding_tools::{ReadTool, ShellTool};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::ask::AskPlugin;
use rig_harness::pick::PickPlugin;
use rig_harness_test_support::{
    Ended, MemoryStore, app_on, connect, first_agent, log_ended, reply, run_until, tui_view,
};
use serde_json::json;

use super::*;

const RULES: &str = r#"
[[ask]]
tool = "shell"
argument = "command"
patterns = ["rm *", "git push*"]
"#;

/// Each settled tool call's outcome, by call id.
#[derive(Resource, Default)]
struct Results(HashMap<String, Outcome>);

fn keep_result(
    settled: On<Add<Settled>>,
    calls: Query<(&ToolRequest, &Outcome)>,
    mut results: ResMut<Results>,
) {
    if let Ok((request, outcome)) = calls.get(settled.entity) {
        results
            .0
            .insert(request.call.id.to_string(), outcome.clone());
    }
}

/// How many questions this plugin asked.
#[derive(Resource, Default)]
struct Opened(usize);

fn count_opened(_: On<Add<ApprovalBatch>>, mut opened: ResMut<Opened>) {
    opened.0 += 1;
}

/// A project with `README.md`, `a.txt` and `b.txt`, and a `bin/git` that
/// records its arguments in `git.log`.
fn project(name: &str) -> PathBuf {
    let exe = std::env::current_exe().unwrap_or_default();
    let dir = exe.with_file_name(format!("approval-batching-{name}-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(dir.join("bin")).ok();
    for file in ["README.md", "a.txt", "b.txt"] {
        std::fs::write(dir.join(file), "x\n").ok();
    }
    let git = format!("#!/bin/sh\necho \"$@\" >> {}/git.log\n", dir.display());
    std::fs::write(dir.join("bin/git"), git).ok();
    std::process::Command::new("chmod")
        .args(["+x", "bin/git"])
        .current_dir(&dir)
        .status()
        .ok();
    dir
}

/// Reply 1: a read and three commands the rules ask about.
fn reply_one() -> Vec<MockStreamEvent> {
    vec![
        MockStreamEvent::tool_call("read", "read", json!({ "path": "README.md" })),
        MockStreamEvent::tool_call("rm-a", "shell", json!({ "command": "rm a.txt" })),
        MockStreamEvent::tool_call("rm-b", "shell", json!({ "command": "rm b.txt" })),
        MockStreamEvent::tool_call("push", "shell", json!({ "command": "git push" })),
        MockStreamEvent::final_response_with_default_usage(),
    ]
}

/// The tools, this plugin and the agent in `dir`, with the fake `git`
/// first on `PATH`; with the terminal view when `at_terminal`.
fn start(
    dir: &Path,
    model: &MockCompletionModel,
    at_terminal: bool,
) -> (App, Receiver<()>, Entity) {
    let (mut app, wakes) = app_on(&MemoryStore::default());
    let path = format!(
        "{}:{}",
        dir.join("bin").display(),
        std::env::var("PATH").unwrap_or_default()
    );
    let processes = Processes::default();
    processes.set(processes.launcher().with_env("PATH", path));
    app.insert_resource(processes)
        .insert_resource(ProjectRoot(dir.to_owned()))
        .insert_resource(ApproveRules::from_toml(RULES).unwrap_or_default())
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            AskPlugin,
            PickPlugin,
            ReadTool,
            ShellTool,
            ApprovalBatchingPlugin,
        ))
        .init_resource::<Ended>()
        .init_resource::<Results>()
        .init_resource::<Opened>()
        .add_observer(log_ended)
        .add_observer(keep_result)
        .add_observer(count_opened);
    if at_terminal {
        tui_view(&mut app);
    }
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    if let Some(connection) = connect(model) {
        app.world_mut().entity_mut(agent).insert(connection);
    }
    (app, wakes, agent)
}

fn output(app: &App, id: &str) -> String {
    let results = &app.world().resource::<Results>().0;
    results
        .get(id)
        .map(|outcome| outcome.tool_result().output().render())
        .unwrap_or_default()
}

#[cfg(feature = "tui")]
fn rows(app: &mut App) -> Vec<String> {
    let mut batches = app.world_mut().query::<&ApprovalBatch>();
    batches
        .iter(app.world())
        .flat_map(|batch| batch.rows.iter().map(|row| row.summary.clone()))
        .collect()
}

/// The terminal view answers with the question's own keys (`tui`).
#[cfg(feature = "tui")]
#[test]
fn one_question_for_a_reply_denies_the_unticked_and_an_interrupt_cancels() {
    let dir = project("terminal");
    let model = MockCompletionModel::from_stream_turns([
        reply_one(),
        reply("Removed a.txt and pushed; kept b.txt."),
        vec![
            MockStreamEvent::tool_call("rm-b2", "shell", json!({ "command": "rm b.txt" })),
            MockStreamEvent::tool_call("rm-r", "shell", json!({ "command": "rm README.md" })),
            MockStreamEvent::final_response_with_default_usage(),
        ],
    ]);
    let (mut app, wakes, agent) = start(&dir, &model, true);
    app.world_mut()
        .trigger(Deliver::new(agent, "clean up", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| {
        world.resource::<Results>().0.contains_key("read") && world.resource::<Opened>().0 > 0
    });

    // The read ran before any answer; one question holds the three commands.
    assert!(
        output(&app, "read").contains('x'),
        "{}",
        output(&app, "read")
    );
    assert_eq!(app.world().resource::<Opened>().0, 1);
    assert_eq!(rows(&mut app), ["rm a.txt", "rm b.txt", "git push"]);

    // In the terminal view, every row starts ticked: untick `rm b.txt`
    // with Space, type a note for it, answer.
    rig_harness_test_support::press(&mut app, "down space k e e p space b enter");
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() == 1
    });
    assert!(!dir.join("a.txt").exists(), "a.txt is still there");
    assert!(dir.join("b.txt").exists(), "b.txt was removed");
    let denied = output(&app, "rm-b");
    assert!(
        denied.contains(DENIED) && denied.contains("keep b"),
        "{denied}"
    );
    let pushed = std::fs::read_to_string(dir.join("git.log")).unwrap_or_default();
    assert_eq!(pushed.trim(), "push");
    assert_eq!(app.world().resource::<Opened>().0, 1);

    // Reply 2's question, interrupted: both calls cancelled, nothing removed.
    app.world_mut()
        .trigger(Deliver::new(agent, "and the rest", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| world.resource::<Opened>().0 == 2);
    assert_eq!(rows(&mut app), ["rm b.txt", "rm README.md"]);
    app.world_mut().trigger(StopTurn::new(agent));
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() == 2
    });
    let results = &app.world().resource::<Results>().0;
    for id in ["rm-b2", "rm-r"] {
        let kind = results
            .get(id)
            .and_then(|outcome| outcome.0.as_ref().err())
            .map(|e| e.kind);
        assert_eq!(
            kind,
            Some(rig_core::error::ErrorKind::Cancelled),
            "{id}: {:?}",
            results.get(id)
        );
    }
    assert!(dir.join("b.txt").exists() && dir.join("README.md").exists());
    assert!(rows(&mut app).is_empty(), "the question is still open");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn with_nobody_to_answer_every_call_needing_approval_is_denied_without_a_question() {
    let dir = project("headless");
    let model = MockCompletionModel::from_stream_turns([reply_one(), reply("Nothing removed.")]);
    let (mut app, wakes, agent) = start(&dir, &model, false);
    app.world_mut()
        .trigger(Deliver::new(agent, "clean up", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() == 1
    });

    assert_eq!(app.world().resource::<Opened>().0, 0);
    assert!(
        output(&app, "read").contains('x'),
        "{}",
        output(&app, "read")
    );
    for id in ["rm-a", "rm-b", "push"] {
        assert!(
            output(&app, id).contains("nobody can answer"),
            "{id}: {}",
            output(&app, id)
        );
    }
    assert!(dir.join("a.txt").exists() && dir.join("b.txt").exists());
    assert!(!dir.join("git.log").exists());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(ApprovalBatchingPlugin);
}
