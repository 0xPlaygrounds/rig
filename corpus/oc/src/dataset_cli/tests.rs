use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use rig_basics::coding_tools::ReadTool;
use rig_basics::telemetry::effect_log::EffectLogPlugin;
use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::Invocation;
use rig_harness_test_support::{
    Ended, app_in, calls, first_agent, log_ended, reply, run_until, scratch_home, scripted_models,
};
use serde_json::{Value, json};

use super::*;

/// One message the user sends a scripted session.
enum Step {
    /// Sent, and its turn runs to its end.
    Say(&'static str),
    /// Sent, and its turn stopped once a tool call runs.
    Stop(&'static str),
}

/// A `slow` tool whose calls never end.
fn slow(_: On<ToolCalled<Value>>) {}

/// A saved session of `home` in `project`, its agent answered by the
/// scripted `model`, the user sending `steps`; its id.
fn session(
    home: &Home,
    project: &Path,
    model: (&str, &MockCompletionModel),
    steps: &[Step],
) -> String {
    let Ok((id, dir)) = home.new_session() else {
        return String::new();
    };
    std::fs::write(dir.directory(), project.to_string_lossy().as_bytes()).ok();
    let (mut app, wakes) = app_in(home, &id);
    app.insert_resource(scripted_models([model]))
        .insert_resource(ProjectRoot(project.to_owned()))
        .add_plugins((AgentPlugin, SessionLogPlugin, EffectLogPlugin, ReadTool))
        .add_open_tool("slow", "Wait.", ToolOptions::default(), slow)
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    app.world_mut().trigger(SetModel {
        entity: agent,
        model: model.0.to_owned(),
        kind: ChoiceKind::ForThisRun,
    });
    for (done, step) in steps.iter().enumerate() {
        let (Step::Say(text) | Step::Stop(text)) = step;
        app.world_mut()
            .trigger(Deliver::new(agent, *text, DeliveryMode::Steer));
        if let Step::Stop(_) = step {
            run_until(&mut app, &wakes, |world| {
                world
                    .try_query::<(&ToolRequest, &Started)>()
                    .is_some_and(|mut calls| calls.iter(world).next().is_some())
            });
            app.world_mut().trigger(StopTurn::new(agent));
        }
        run_until(&mut app, &wakes, |world| {
            world.resource::<Ended>().0.len() > done
        });
    }
    app.update();
    id.to_string()
}

/// What `rig dataset` with `args` makes in `home`, run from `cwd`.
fn dataset(home: &Home, cwd: &Path, args: &[&str]) -> DatasetRun {
    let invocation = Invocation {
        command: Some(COMMAND.to_owned()),
        arguments: args.iter().map(|arg| (*arg).to_owned()).collect(),
        ..Invocation::default()
    };
    let mut app = App::new();
    app.insert_resource(RigHome(home.clone()))
        .insert_resource(ProjectRoot(cwd.to_owned()))
        .insert_resource(Invoked {
            args: invocation,
            terminal: false,
        })
        .add_plugins(DatasetPlugin);
    app.finish();
    app.update();
    app.world_mut()
        .remove_resource::<DatasetRun>()
        .unwrap_or_default()
}

/// Every file under `dir`, with its content and modification time.
fn files(dir: &Path) -> BTreeMap<PathBuf, (Vec<u8>, Option<SystemTime>)> {
    let mut files = BTreeMap::new();
    let mut dirs = vec![dir.to_owned()];
    while let Some(dir) = dirs.pop() {
        for entry in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                dirs.push(path);
            } else {
                let modified = entry.metadata().and_then(|meta| meta.modified()).ok();
                let bytes = std::fs::read(&path).unwrap_or_default();
                files.insert(path, (bytes, modified));
            }
        }
    }
    files
}

fn lines(path: &Path) -> Vec<Value> {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    text.lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect()
}

fn messages(line: &Value) -> Vec<Value> {
    line.get("messages")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

#[test]
fn exports_the_matching_sessions_as_chat_lines() {
    let home = scratch_home("corpus-oc-dataset-cli");
    let project = home.root().join("work");
    std::fs::create_dir_all(&project).ok();
    let notes = project.join("notes.txt");
    std::fs::write(&notes, "the notes").ok();

    let read = json!({ "path": notes.display().to_string() });
    let a_model = MockCompletionModel::from_stream_turns([
        calls("r1", "read", read),
        reply("Read it."),
        reply("Two."),
        reply("Three."),
    ]);
    let a = session(
        &home,
        &project,
        ("fake/fake-a", &a_model),
        &[
            Step::Say("Read notes."),
            Step::Say("Again."),
            Step::Say("Once more."),
        ],
    );
    let b_model = MockCompletionModel::from_stream_turns([reply("Only.")]);
    let b = session(
        &home,
        &project,
        ("fake/fake-b", &b_model),
        &[Step::Say("Hi.")],
    );
    let c_model = MockCompletionModel::from_stream_turns([
        reply("Logged in."),
        calls("s1", "slow", json!({})),
        reply("Fine."),
    ]);
    let c = session(
        &home,
        &project,
        ("fake/fake-a", &c_model),
        &[
            Step::Say("Log in with TOKEN=abc123 please."),
            Step::Stop("Wait for it."),
            Step::Say("Never mind."),
        ],
    );
    let before = files(&home.sessions());

    // A and C were answered by fake-a; C's stopped turn is dropped.
    let args = [
        "export",
        "--model",
        "fake-a",
        "--exclude-errors",
        "--redact",
        "-o",
        "out",
    ];
    let run = dataset(&home, &project, &args);
    assert_eq!(run.status, 0, "{run:?}");
    let dropped = "turns dropped: 1 (failed 0, stopped 1, cut short 0)";
    assert!(run.stdout.contains(dropped), "{run:?}");
    let written = std::fs::read_to_string(project.join("out/dataset.jsonl")).unwrap_or_default();
    assert!(!written.contains("abc123"), "{written}");
    assert!(written.contains(dataset::REDACTED), "{written}");
    let exported = lines(&project.join("out/dataset.jsonl"));
    let of = |id: &str| {
        exported
            .iter()
            .find(|line| line.pointer("/metadata/session") == Some(&json!(id)))
            .cloned()
            .unwrap_or_default()
    };
    let sessions: Vec<_> = exported
        .iter()
        .filter_map(|line| line.pointer("/metadata/session")?.as_str())
        .collect();
    assert_eq!(sessions.len(), 2, "{written}");
    assert!(
        sessions.contains(&a.as_str()) && sessions.contains(&c.as_str()),
        "{written}"
    );
    assert!(!sessions.contains(&b.as_str()), "{written}");

    let a_messages = messages(&of(&a));
    let call_id = a_messages
        .iter()
        .find_map(|message| message.pointer("/tool_calls/0/id"))
        .cloned();
    assert!(call_id.is_some(), "{a_messages:?}");
    let answered = a_messages.iter().any(|message| {
        message.get("role") == Some(&json!("tool"))
            && message.get("tool_call_id") == call_id.as_ref()
    });
    assert!(answered, "{a_messages:?}");
    let users = a_messages
        .iter()
        .filter(|message| message.get("role") == Some(&json!("user")))
        .count();
    assert_eq!(users, 3, "{a_messages:?}");
    let tools = of(&a).get("tools").cloned().unwrap_or_default();
    let names: Vec<_> = tools
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|tool| tool.pointer("/function/name")?.as_str())
        .collect();
    assert!(names.contains(&"read"), "{tools}");

    let c_text = of(&c).to_string();
    assert!(
        c_text.contains("Never mind.") && c_text.contains("Logged in."),
        "{c_text}"
    );
    assert!(
        !c_text.contains("Wait for it.")
            && !c_text.contains("tool_calls")
            && !c_text.contains(r#""role":"tool""#),
        "{c_text}"
    );

    // The same seed splits the same way.
    let split = |out: &str| {
        let args = ["export", "--split", "0.5", "--seed", "7", "-o", out];
        let run = dataset(&home, &project, &args);
        assert_eq!(run.status, 0, "{run:?}");
        ["train.jsonl", "valid.jsonl"]
            .map(|file| std::fs::read_to_string(project.join(out).join(file)).unwrap_or_default())
    };
    let first = split("one");
    assert_eq!(first, split("two"));
    assert_eq!(
        first.iter().map(|text| text.lines().count()).sum::<usize>(),
        3
    );

    let none = dataset(
        &home,
        &project,
        &["export", "--model", "nope", "-o", "none"],
    );
    assert_eq!(none.status, 2, "{none:?}");

    assert!(
        before == files(&home.sessions()),
        "a session's files changed"
    );
    std::fs::remove_dir_all(home.root()).ok();
}
