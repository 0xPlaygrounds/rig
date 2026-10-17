use std::fmt::Display;

use rig_core::message::{AssistantMessage, ToolName};
use rig_core::test_utils::MockCompletionModel;
use rig_harness::prelude::{
    Agent, AgentPlugin, AppToolsExt, Deliver, DeliveryMode, Entity, On, SessionLogPlugin,
    SpawnedBy, ToolCalled, ToolOptions, TurnOutcome, With, Without,
};
use rig_harness_test_support::{
    Ended, app_in, log_ended, reply, run_until, scratch_home, scripted_models,
};

use rig_harness_protocol::SessionId;
use rig_memory::{MODEL, TITLE};

use super::*;

/// `result`'s value, failing the test with its error.
fn ok<T>(result: Result<T, impl Display>) -> Option<T> {
    let result = result.map_err(|error| error.to_string());
    assert_eq!(result.as_ref().err(), None);
    result.ok()
}

/// Runs `session-edit` with `args` against `home`: its output, or why it
/// stopped.
fn edit(home: &Home, args: &[&str]) -> Result<String, Error> {
    let args: Vec<String> = args.iter().map(|arg| (*arg).to_owned()).collect();
    let mut out = Vec::new();
    run(&args, home, &mut out)?;
    Ok(String::from_utf8_lossy(&out).into_owned())
}

/// The 10 000-line output of the fixture's tool call.
fn huge() -> String {
    let lines: Vec<String> = (1..=10_000).map(|n| format!("auth.log line {n}")).collect();
    lines.join("\n")
}

/// A session made with the library alone, named `fix-auth`: a user
/// message, a call of `shell`, its 10 000-line result, a second exchange,
/// and one subagent.
fn fixture(home: &Home) -> Result<(SessionId, Vec<Message>), Box<dyn std::error::Error>> {
    let shell = ToolName::new("shell")?;
    let messages = vec![
        Message::user("The login fails; fix it."),
        Message::Assistant(AssistantMessage::new(vec![
            AssistantContent::text("Reading the log."),
            AssistantContent::tool_call(
                "call_1",
                shell.clone(),
                serde_json::json!({ "command": "cat auth.log" }),
            ),
        ])),
        Message::tool_result(CallId::from_wire("call_1"), shell, huge()),
        Message::user("What was it?"),
        Message::assistant("An expired token; fixed."),
    ];
    let main = SavedAgent {
        id: "main".to_owned(),
        conversation: Conversation::from_messages(messages.clone()),
        saved: [(MODEL.to_owned(), Value::from("test/model"))].into(),
        ..SavedAgent::default()
    };
    let sub = SavedAgent {
        id: "sub".to_owned(),
        parent: Some("main".to_owned()),
        conversation: Conversation::from_messages([
            Message::user("Count the failed logins."),
            Message::assistant("42."),
        ]),
        saved: [(TITLE.to_owned(), Value::from("Count failures"))].into(),
        ..SavedAgent::default()
    };
    let (id, dir) = home.new_session()?;
    let store = DirStore::open(dir.path())?;
    let mut session = Session {
        agents: vec![main, sub],
        ..Session::default()
    };
    session.save(&store)?;
    dir.set_name(Some("fix-auth"))?;
    dir.mark_saved()?;
    Ok((id, messages))
}

/// The main conversation of the session `named` in `home`, read with the
/// library's normal load path.
fn reopen(home: &Home, named: &str) -> Option<Vec<Message>> {
    let opened = ok(locate(home, named).and_then(open))?;
    Some(
        opened
            .session
            .main_agent()?
            .conversation
            .messages()
            .iter()
            .map(|message| (**message).clone())
            .collect(),
    )
}

/// The text of the tool result in `message`.
fn result_text(message: Option<&Message>) -> String {
    let Some(Message::User { content }) = message else {
        return String::new();
    };
    let texts = content.iter().filter_map(|part| match part {
        UserContent::ToolResult(result) => result.content.first()?.as_text(),
        _ => None,
    });
    texts.collect()
}

#[test]
fn lists_shrinks_drops_resumes_exports_and_refuses() {
    let home = scratch_home("oc-session-edit-library");
    let Some((id, original)) = ok(fixture(&home)) else {
        return;
    };

    // `list`: five messages, the tool's name, the subagent.
    let listed = ok(edit(&home, &["fix-auth", "list"])).unwrap_or_default();
    assert!(listed.contains("messages: 5"), "{listed}");
    assert!(listed.contains("model: test/model"), "{listed}");
    assert_eq!(listed.matches(" chars").count(), 5, "{listed}");
    assert_eq!(
        listed.matches("shell").count(),
        2,
        "the call and its result: {listed}"
    );
    assert!(listed.contains("Count failures: 2 messages"), "{listed}");

    // `shrink --over 4000 --as small`: the result keeps 40 lines and a
    // marker; every other message is as it was.
    let shrunk = ok(edit(
        &home,
        &["fix-auth", "shrink", "--over", "4000", "--as", "small"],
    ));
    assert!(shrunk.is_some_and(|out| out.contains("rig --resume")));
    let small = reopen(&home, "small").unwrap_or_default();
    let text = result_text(small.get(2));
    assert_eq!(text.lines().count(), 41, "{text}");
    assert!(text.contains("[9960 lines removed by session-edit]"));
    assert!(text.starts_with("auth.log line 1\n") && text.ends_with("auth.log line 10000"));
    let others = |messages: &[Message]| -> Vec<String> {
        let kept = messages.iter().enumerate().filter(|(at, _)| *at != 2);
        kept.map(|(_, message)| serde_json::to_string(message).unwrap_or_default())
            .collect()
    };
    assert_eq!(others(&small), others(&original));
    assert_eq!(small.len(), 5);
    // The copy was rewritten by the shrink, so no saved record of it
    // still holds what was cut.
    let small_dir = ok(locate(&home, "small"));
    let small_log = small_dir.map(|dir| dir.path().join("logs").join("main.jsonl"));
    let small_log = small_log.and_then(|log| fs::read_to_string(log).ok());
    assert!(small_log.is_some_and(|log| !log.contains("auth.log line 5000")));

    // `drop 2..2` on the original: the call goes with its result.
    let dropped = ok(edit(&home, &["fix-auth", "drop", "2..2", "--as", "cut"])).unwrap_or_default();
    assert!(
        dropped.contains("removed message 2 (tool result of shell)"),
        "{dropped}"
    );
    assert!(
        dropped.contains("removed tool call shell (call_1) from message 1"),
        "{dropped}"
    );
    let cut = reopen(&home, "cut").unwrap_or_default();
    assert_eq!(cut.len(), 4);
    assert!(
        cut.iter().all(|message| call_ids(message).is_empty()),
        "{cut:?}"
    );
    assert!(pairing::validate(&cut).is_ok());
    // The original is untouched.
    assert_eq!(reopen(&home, "fix-auth"), Some(original.clone()));

    // `small` resumes in the harness, and its request carries the shrunk
    // result.
    let small_id = ok(locate(&home, "small")).and_then(|dir| dir.id());
    let model = MockCompletionModel::from_stream_turns([reply("Glad to help.")]);
    if let Some(small_id) = small_id {
        let (mut app, wakes) = app_in(&home, &small_id);
        app.insert_resource(scripted_models([("test/model", &model)]))
            .add_plugins((AgentPlugin, SessionLogPlugin))
            // A `shell`, so the request declares tools and keeps the call
            // and its result as such.
            .add_open_tool(
                "shell",
                "A stand-in.",
                ToolOptions::default(),
                |_: On<ToolCalled<Value>>| {},
            )
            .init_resource::<Ended>()
            .add_observer(log_ended);
        app.finish();
        app.update();
        let world = app.world_mut();
        let mut main = world.query_filtered::<Entity, (With<Agent>, Without<SpawnedBy>)>();
        let agent = main.iter(world).next().unwrap_or(Entity::PLACEHOLDER);
        world.trigger(Deliver::new(agent, "Thanks!", DeliveryMode::Steer));
        run_until(&mut app, &wakes, |world| {
            !world.resource::<Ended>().0.is_empty()
        });
        let ended = &app.world().resource::<Ended>().0;
        assert!(
            matches!(ended.first(), Some(TurnOutcome::Answered(_))),
            "{ended:?}"
        );
    }
    let requests = model.requests();
    let sent = requests
        .first()
        .map(|request| request.chat_history.clone())
        .unwrap_or_default();
    assert!(
        sent.iter()
            .any(|message| result_text(Some(message)).contains("9960 lines removed")),
        "{sent:?}"
    );

    // `export --markdown`: a heading per message, the call's JSON.
    let file = home.root().join("fix-auth.md");
    let path = file.display().to_string();
    assert!(ok(edit(&home, &["fix-auth", "export", "--markdown", &path])).is_some());
    let markdown = ok(fs::read_to_string(&file)).unwrap_or_default();
    for heading in [
        "## 0 · User",
        "## 1 · Assistant",
        "## 2 · Tool result",
        "## 3 · User",
        "## 4 · Assistant",
    ] {
        assert!(markdown.contains(heading), "{heading}: {markdown}");
    }
    assert!(
        markdown.contains("```json\n{\n  \"command\": \"cat auth.log\"\n}\n```"),
        "{markdown}"
    );
    assert!(markdown.contains("# Subagent: Count failures"));

    // `--in-place`: the session is rewritten, its old self kept in `.bak`.
    let in_place = ok(edit(
        &home,
        &["fix-auth", "shrink", "--over", "4000", "--in-place"],
    ));
    assert!(in_place.is_some_and(|out| out.contains(".bak")));
    let rewritten = reopen(&home, "fix-auth").unwrap_or_default();
    assert_eq!(result_text(rewritten.get(2)).lines().count(), 41);
    let session = home.session(&id);
    let log = fs::read_to_string(session.path().join("logs").join("main.jsonl"));
    assert!(log.is_ok_and(|log| !log.contains("auth.log line 5000")));
    let backup = session
        .path()
        .with_extension("bak")
        .join("logs")
        .join("main.jsonl");
    assert!(fs::read_to_string(backup).is_ok_and(|log| log.contains("auth.log line 5000")));

    // A truncated session is refused, with why: `cut` still holds the
    // whole result in its history (a drop is a recorded splice), so half
    // its log ends inside that line.
    let cut_dir = ok(locate(&home, "cut")).map(|dir| dir.path().to_owned());
    let log = cut_dir.unwrap_or_default().join("logs").join("main.jsonl");
    let bytes = fs::read(&log).unwrap_or_default();
    fs::write(&log, bytes.get(..bytes.len() / 2).unwrap_or_default()).ok();
    let refused = edit(&home, &["cut", "list"]);
    assert!(
        refused
            .as_ref()
            .is_err_and(|why| why.0.contains("truncated")),
        "{refused:?}"
    );
    // So is a session that is not there.
    assert!(edit(&home, &["no-such-session", "list"]).is_err());
}
