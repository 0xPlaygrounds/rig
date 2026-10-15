use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::{Invocation, SessionId};
use rig_harness::host::exit::ExitPlugin;
use rig_harness_test_support::{
    app_in, calls, connect, first_agent, reply, run_until, scratch_home,
};

use super::client::{Words, ask, parse, read_ask, suggest};
use super::*;

/// What a test's clients printed and how their requests ended.
type Answered = (String, std::io::Result<Option<String>>);

/// Every [`ShellReplied`].
#[derive(Resource, Default)]
struct Replied(Vec<ShellReplied>);

/// The `wait` call the first request's turn makes, held until the test
/// answers it, so the agent stays busy.
#[derive(Resource, Default)]
struct WaitCall(Option<Entity>);

fn record(replied: On<ShellReplied>, mut log: ResMut<Replied>) {
    log.0.push(replied.event().clone());
}

/// Asks the session on `socket` from a thread of its own.
fn client(socket: &Path, request: ShellAsk) -> mpsc::Receiver<Answered> {
    let (sender, answered) = mpsc::channel();
    let socket = socket.to_path_buf();
    std::thread::spawn(move || {
        let mut out = Vec::new();
        let result = ask(&socket, &request, &mut out);
        sender
            .send((String::from_utf8_lossy(&out).into_owned(), result))
            .ok();
    });
    answered
}

/// Runs frames until `done`, at most ten seconds.
fn drive(app: &mut App, wakes: &mpsc::Receiver<()>, mut done: impl FnMut(&mut App) -> bool) {
    let started = Instant::now();
    while !done(app) {
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "the shell clients hung"
        );
        app.update();
        wakes.recv_timeout(Duration::from_millis(5)).ok();
    }
}

/// Runs frames until `answered` has the client's answer.
fn wait_for(
    app: &mut App,
    wakes: &mpsc::Receiver<()>,
    answered: &mpsc::Receiver<Answered>,
) -> Answered {
    let mut got = None;
    drive(app, wakes, |_| {
        got = got.take().or_else(|| answered.try_recv().ok());
        got.is_some()
    });
    got.unwrap_or_else(|| (String::new(), Err(std::io::Error::other("no answer"))))
}

/// The texts of the agent's user messages from the shell.
fn shell_messages(app: &App, agent: Entity) -> Vec<String> {
    let Some(conversation) = app.world().get::<Conversation>(agent) else {
        return Vec::new();
    };
    conversation
        .messages()
        .iter()
        .filter_map(|message| match &**message {
            message::Message::User { content } => content.iter().find_map(|part| match part {
                UserContent::Text(text) if text.text.contains("[from my shell]") => {
                    Some(text.text.clone())
                }
                _ => None,
            }),
            _ => None,
        })
        .collect()
}

/// Runs the bash snippet non-interactively: `false` with the output
/// "boom" through `rig-run`, then `?? why`, with a `rig` that only keeps
/// the words it was given. The request the client would send for them.
fn shell_side(project: &Path, state: &Path) -> Option<(Words, ShellAsk)> {
    let fake = project.join("fake-rig");
    let args = project.join("args");
    std::fs::write(
        &fake,
        format!("#!/bin/sh\nprintf '%s\\n' \"$@\" > '{}'\n", args.display()),
    )
    .ok()?;
    std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).ok()?;
    let snippet = project.join("snippet.bash");
    std::fs::write(&snippet, snippet::BASH).ok()?;
    let script = "shopt -s expand_aliases\nsource \"$1\"\nrig-run sh -c 'echo boom; echo key hunter2-xyz; false'\n?? why\n";
    let status = std::process::Command::new("bash")
        .args(["-c", script, "shell", &snippet.display().to_string()])
        .current_dir(project)
        .env("RIG_BIN", &fake)
        .env("RIG_SHELL_STATE_DIR", state)
        .stdout(std::process::Stdio::null())
        .status()
        .ok()?;
    assert!(status.success(), "the snippet ran");
    let words: Vec<String> = std::fs::read_to_string(&args)
        .ok()?
        .lines()
        .map(str::to_owned)
        .collect();
    let (command, rest) = words.split_first()?;
    assert_eq!(command, SHELL_ASK);
    let words = parse(rest);
    let request = read_ask(words.state.as_deref()?, &words);
    Some((words, request))
}

#[test]
fn shells_ask_the_running_session_one_at_a_time() {
    let home = scratch_home("ct-shell-companion");
    let project = home.root().join("project");
    let state = project.join(".state");
    assert!(std::fs::create_dir_all(&state).is_ok());
    let marker = project.join("MARKER");
    let model = MockCompletionModel::from_stream_turns([
        calls("w1", "wait", serde_json::json!({})),
        reply("It failed because boom."),
        reply("one"),
        reply("two"),
        reply(&format!(
            "Run this:\n```sh\ntouch {}\n```",
            marker.display()
        )),
    ]);
    let (mut app, wakes) = app_in(&home, &SessionId::generate());
    app.insert_resource(Settings::from_toml(
        "[shell-companion]\nredact = [\"hunter2-xyz\"]\n",
    ))
    .insert_resource(ProjectRoot(project.clone()))
    .add_plugins((AgentPlugin, SessionLogPlugin, ShellCompanionPlugin))
    .init_resource::<WaitCall>()
    .add_open_tool(
        "wait",
        "Wait.",
        ToolOptions::default(),
        |called: On<ToolCalled<serde_json::Value>>, mut wait: ResMut<WaitCall>| {
            wait.0 = Some(called.call);
        },
    )
    .init_resource::<Replied>()
    .add_observer(record);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let connection = connect(&model);
    assert!(connection.is_some());
    if let Some(connection) = connection {
        app.world_mut().entity_mut(agent).insert(connection);
    }

    let shell = shell_side(&project, &state);
    assert!(shell.is_some(), "the snippet recorded the command");
    let Some((words, why)) = shell else { return };
    assert_eq!((words.question.as_str(), why.exit), ("why", Some(1)));
    assert!(
        why.command.contains("echo boom") && why.tail.contains("boom"),
        "{why:?}"
    );
    assert_eq!(
        Path::new(&why.cwd).canonicalize().ok(),
        project.canonicalize().ok()
    );

    let socket = client::find_socket(home.root(), &project.join("src"));
    assert!(
        socket.is_some(),
        "the session serves the project, below its root too"
    );
    let Some(socket) = socket else { return };

    // The first request's turn waits on `wait`; two shells ask meanwhile.
    let first = client(&socket, why);
    drive(&mut app, &wakes, |app| {
        app.world().resource::<WaitCall>().0.is_some()
    });
    let asks = |question: &str| ShellAsk {
        question: question.to_owned(),
        cwd: project.display().to_string(),
        ..ShellAsk::default()
    };
    let second = client(&socket, asks("first"));
    drive(&mut app, &wakes, |app| {
        app.world().resource::<Server>().queue.len() == 1
    });
    let third = client(&socket, asks("second"));
    drive(&mut app, &wakes, |app| {
        app.world().resource::<Server>().queue.len() == 2
    });
    let call = app
        .world()
        .resource::<WaitCall>()
        .0
        .unwrap_or(Entity::PLACEHOLDER);
    if let Some(mut decision) = app.world_mut().get_mut::<Decision>(call) {
        decision.decide(
            Decider::plugin("test"),
            ToolResult::success("waited".into()),
        );
    }
    let answers = [&first, &second, &third].map(|answered| wait_for(&mut app, &wakes, answered));
    let [
        (why_out, why_result),
        (one_out, one_result),
        (two_out, two_result),
    ] = answers;
    assert!(matches!(why_result, Ok(None)), "{why_result:?}");
    assert!(
        why_out.contains("> wait") && why_out.contains("It failed because boom."),
        "{why_out}"
    );
    assert!(
        matches!(one_result, Ok(None)) && one_out.contains("1 ahead") && one_out.contains("one"),
        "{one_out}"
    );
    assert!(
        matches!(two_result, Ok(None)) && two_out.contains("2 ahead") && two_out.contains("two"),
        "{two_out}"
    );
    assert!(
        !one_out.contains("two") && !two_out.contains("one"),
        "each its own reply"
    );

    // `?? --cmd`: the command comes back for the edit buffer, unrun.
    let request = ShellAsk {
        cmd: true,
        ..asks("list big files")
    };
    let suggesting = client(&socket, request);
    let (cmd_out, command) = wait_for(&mut app, &wakes, &suggesting);
    let expected = format!("touch {}", marker.display());
    assert_eq!(
        command.as_ref().ok().cloned().flatten(),
        Some(expected.clone()),
        "{cmd_out}"
    );
    suggest(&state, command.ok().flatten().as_deref());
    assert_eq!(
        std::fs::read_to_string(state.join(client::SUGGESTED)).ok(),
        Some(expected)
    );
    assert!(!marker.exists(), "nothing ran the command");

    let messages = shell_messages(&app, agent);
    assert_eq!(messages.len(), 4, "{messages:#?}");
    let questions: Vec<bool> = ["why", "first", "second", "list big files"]
        .iter()
        .zip(&messages)
        .map(|(question, message)| message.starts_with(&format!("[from my shell] {question}")))
        .collect();
    assert_eq!(questions, [true; 4], "delivered in order: {messages:#?}");
    let why = messages.first().cloned().unwrap_or_default();
    for part in [
        "```shell\n$ sh -c",
        "# exit code: 1",
        &format!("# cwd: {}", project.display()),
        "boom",
        "[REDACTED]",
    ] {
        assert!(why.contains(part), "{part} in {why}");
    }
    assert!(
        !why.contains("hunter2-xyz"),
        "the secret is redacted: {why}"
    );
    assert_eq!(app.world().resource::<Replied>().0.len(), 4);
    std::fs::remove_dir_all(home.root()).ok();
}

#[test]
fn without_a_session_a_one_shot_session_answers() {
    let home = scratch_home("ct-shell-companion-one-shot");
    let state = home.root().join("state");
    assert!(std::fs::create_dir_all(&state).is_ok());
    for (name, text) in [("command", "false"), ("exit", "1"), ("output", "boom")] {
        assert!(std::fs::write(state.join(name), text).is_ok());
    }
    assert!(std::fs::write(state.join("cwd"), home.root().display().to_string()).is_ok());
    let model = MockCompletionModel::from_stream_turns([reply("false always fails.")]);
    let (mut app, wakes) = app_in(&home, &SessionId::generate());
    let arguments = ["--state", &state.display().to_string()]
        .map(str::to_owned)
        .to_vec();
    let args = Invocation {
        model: Some("deepseek/deepseek-flash".to_owned()),
        command: Some(SHELL_ASK.to_owned()),
        arguments,
        ..Invocation::default()
    };
    app.insert_resource(Invoked {
        args,
        terminal: false,
    })
    .insert_resource(ProjectRoot(home.root().to_path_buf()))
    .add_plugins((
        AgentPlugin,
        SessionLogPlugin,
        ExitPlugin,
        ShellCompanionPlugin,
    ))
    .init_resource::<Replied>()
    .add_observer(record);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    if let Some(connection) = connect(&model) {
        app.world_mut().entity_mut(agent).insert(connection);
    }
    run_until(&mut app, &wakes, |world| {
        world.contains_resource::<ExitReason>()
    });

    let done = Event::Done {
        answer: "false always fails.".to_owned(),
        command: None,
    };
    let replied: Vec<&Event> = app
        .world()
        .resource::<Replied>()
        .0
        .iter()
        .map(|replied| &replied.event)
        .collect();
    assert_eq!(replied, [&done]);
    assert_eq!(
        app.world().get_resource::<ExitReason>(),
        Some(&ExitReason::Quit)
    );
    let messages = shell_messages(&app, agent);
    assert!(
        messages
            .first()
            .is_some_and(|message| message.contains(WHY) && message.contains("boom")),
        "{messages:?}"
    );
    std::fs::remove_dir_all(home.root()).ok();
}
