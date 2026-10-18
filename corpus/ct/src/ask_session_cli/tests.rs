use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use rig_basics::telemetry::UsagePlugin;
use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::{Invocation, SessionDir};
use rig_harness_test_support::{
    Ended, app_in, calls, first_agent, log_ended, reply, run_until, scratch_home, scripted_models,
};
use serde_json::{Value, json};

use super::*;

const ANSWER: &str = "Bumping the timeout fixed it (turn 4).";

/// A `shell` that only says the tests passed.
fn shell(called: On<ToolCalled<Value>>, mut calls: Query<&mut Decision, ToolCalls>) {
    if let Ok(mut call) = calls.get_mut(called.call) {
        call.decide(
            Decider::plugin("test"),
            ToolResult::success("test result: ok".into()),
        );
    }
}

/// The saved session `fixture` of `home`: five turns, the last (turn 4)
/// running `cargo test -p core` and saying it was fixed by bumping the
/// timeout. Its app stays open, as another rig process would.
fn fixture(home: &Home) -> (App, SessionDir) {
    let Ok((id, dir)) = home.new_session() else {
        return (App::new(), SessionDir::at(home.sessions()));
    };
    let model = MockCompletionModel::from_stream_turns([
        reply("Read the readme."),
        reply("Looked at the tests."),
        reply("Found the slow test."),
        reply("It waits on the network."),
        calls("c1", "shell", json!({ "command": "cargo test -p core" })),
        reply("Tests pass: fixed by bumping the timeout."),
    ]);
    let (mut app, wakes) = app_in(home, &id);
    app.insert_resource(scripted_models([("test/fixture", &model)]))
        .add_plugins((AgentPlugin, SessionLogPlugin))
        .add_open_tool("shell", "Run a command.", ToolOptions::default(), shell)
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    app.world_mut().trigger(SetModel {
        entity: agent,
        model: "test/fixture".to_owned(),
    });
    let asks = [
        "Start.",
        "Go on.",
        "And?",
        "Why so slow?",
        "Make core's tests pass.",
    ];
    for (turns, ask) in asks.into_iter().enumerate() {
        app.world_mut()
            .trigger(Deliver::new(agent, ask, DeliveryMode::Steer));
        run_until(&mut app, &wakes, |world| {
            world.resource::<Ended>().0.len() > turns
        });
    }
    app.update();
    dir.set_name(Some("fixture")).ok();
    (app, dir)
}

/// What `rig ask-session` with `args` did in `home`, its model scripted.
fn ask(home: &Home, args: &[&str], model: &MockCompletionModel) -> AskRun {
    let Ok((id, _)) = home.new_session() else {
        return AskRun::default();
    };
    let invocation = Invocation {
        command: Some(COMMAND.to_owned()),
        arguments: args.iter().map(|arg| (*arg).to_owned()).collect(),
        ..Invocation::default()
    };
    let (mut app, wakes) = app_in(home, &id);
    app.insert_resource(Invoked {
        args: invocation,
        terminal: false,
    })
    .insert_resource(scripted_models([("test/asker", model)]))
    .add_plugins((AgentPlugin, SessionLogPlugin, UsagePlugin, AskSessionPlugin));
    app.finish();
    run_until(&mut app, &wakes, |world| {
        world.contains_resource::<AskRun>()
    });
    app.world_mut()
        .remove_resource::<AskRun>()
        .unwrap_or_default()
}

/// Every file under `dir` with its bytes and when it last changed.
fn files(dir: &Path) -> BTreeMap<PathBuf, (Vec<u8>, Option<SystemTime>)> {
    let mut files = BTreeMap::new();
    let mut dirs = vec![dir.to_owned()];
    while let Some(dir) = dirs.pop() {
        for entry in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                dirs.push(path);
            } else {
                let changed = entry.metadata().ok().and_then(|meta| meta.modified().ok());
                files.insert(
                    path.clone(),
                    (std::fs::read(&path).unwrap_or_default(), changed),
                );
            }
        }
    }
    files
}

fn asker() -> MockCompletionModel {
    MockCompletionModel::from_stream_turns([
        calls("s1", SEARCH, json!({ "query": "fixed timeout" })),
        reply(ANSWER),
    ])
}

#[test]
fn a_question_about_an_open_session_is_answered_and_changes_nothing() {
    let home = scratch_home("ct-ask-session-cli");
    let (_open, dir) = fixture(&home);
    // Locked by "another process", as a running launcher holds it.
    let lock = File::options()
        .create(true)
        .truncate(false)
        .write(true)
        .open(dir.launcher_lock());
    assert!(lock.as_ref().is_ok_and(|lock| lock.try_lock().is_ok()));
    assert!(dir.runs_elsewhere());
    let before = files(dir.path());
    assert!(
        before
            .keys()
            .any(|path| path.extension().is_some_and(|ext| ext == "jsonl"))
    );

    let model = asker();
    let plain = ask(
        &home,
        &["fixture", "what fixed it?", "--model", "test/asker"],
        &model,
    );
    // It saw the session, and could call only the two read-only tools.
    let first = model.requests().into_iter().next();
    let seen = first
        .as_ref()
        .map(|first| serde_json::to_string(&first.chat_history));
    let seen = seen.and_then(Result::ok).unwrap_or_default();
    assert!(seen.contains("cargo test -p core"), "{seen}");
    let tools = first.iter().flat_map(|first| &first.tools);
    let mut tools: Vec<String> = tools.map(|tool| tool.name.to_string()).collect();
    tools.sort();
    assert_eq!(tools, [FETCH, SEARCH]);
    assert_eq!(
        plain,
        AskRun {
            stdout: format!("{ANSWER}\n"),
            status: 0
        }
    );

    let path = dir.path().to_string_lossy().into_owned();
    let args = [
        path.as_str(),
        "what fixed it?",
        "--json",
        "--model=test/asker",
    ];
    let run = ask(&home, &args, &asker());
    assert_eq!(run.status, 0);
    let json: Value = serde_json::from_str(&run.stdout).unwrap_or_default();
    let field = |name: &str| json.get(name).cloned().unwrap_or_default();
    assert_eq!(field("answer"), ANSWER);
    assert_eq!(field("model"), "test/asker");
    assert!(
        field("cited_turns")
            .as_array()
            .is_some_and(|turns| turns.contains(&json!(4))),
        "{json}"
    );
    assert!(field("tokens").is_u64(), "{json}");

    let unknown = ask(
        &home,
        &["no-such-session", "anything?", "--model", "test/asker"],
        &asker(),
    );
    assert_eq!(
        unknown,
        AskRun {
            stdout: String::new(),
            status: 2
        }
    );

    assert_eq!(files(dir.path()), before);
}
