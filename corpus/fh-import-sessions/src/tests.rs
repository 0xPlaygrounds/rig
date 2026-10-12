use std::fs;
use std::path::{Path, PathBuf};

use rig_core::message::{Message, ToolResultContent};
use rig_core::test_utils::MockCompletionModel;
use rig_harness::prelude::read_session;
use rig_harness_test_support::{
    Collected, Ended, app_in, collect, first_agent, log_ended, reply, run_until, scratch_home,
    scripted_models,
};

use super::*;

const CLAUDE: &str = r#"{"type":"user","uuid":"u1","parentUuid":null,"isSidechain":false,"sessionId":"cc-1","cwd":"/work/myapp","timestamp":"2026-10-01T10:00:00Z","message":{"role":"user","content":"List the files"}}
{"type":"assistant","uuid":"a1","parentUuid":"u1","isSidechain":false,"sessionId":"cc-1","timestamp":"2026-10-01T10:00:05Z","message":{"role":"assistant","model":"claude-x","content":[{"type":"text","text":"Listing."},{"type":"tool_use","id":"toolu_1","name":"Bash","input":{"command":"ls -la"}}]}}
{"type":"user","uuid":"u2","parentUuid":"a1","isSidechain":false,"sessionId":"cc-1","timestamp":"2026-10-01T10:00:06Z","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"main.rs\nCargo.toml"}]}}
{"type":"assistant","uuid":"a2","parentUuid":"u2","isSidechain":false,"sessionId":"cc-1","timestamp":"2026-10-01T10:00:09Z","message":{"role":"assistant","model":"claude-x","content":[{"type":"text","text":"Two files."}]}}
{"type":"user","uuid":"s1","parentUuid":null,"isSidechain":true,"sessionId":"cc-1","timestamp":"2026-10-01T10:00:07Z","message":{"role":"user","content":"Count the files"}}
{"type":"assistant","uuid":"s2","parentUuid":"s1","isSidechain":true,"sessionId":"cc-1","timestamp":"2026-10-01T10:00:08Z","message":{"role":"assistant","model":"claude-x","content":[{"type":"text","text":"Two."}]}}
{"type":"file-history-snapshot","messageId":"a2","snapshot":{}}
"#;

const CODEX: &str = r#"{"timestamp":"2026-10-02T09:00:00Z","type":"session_meta","payload":{"id":"cx-1","cwd":"/work/other"}}
{"timestamp":"2026-10-02T09:00:01Z","type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"Run the tests"}]}}
{"timestamp":"2026-10-02T09:00:02Z","type":"response_item","payload":{"type":"function_call","name":"shell","arguments":"{\"command\":[\"bash\",\"-lc\",\"cargo test\"]}","call_id":"call_1"}}
{"timestamp":"2026-10-02T09:00:09Z","type":"response_item","payload":{"type":"function_call_output","call_id":"call_1","output":"ok"}}
{"timestamp":"2026-10-02T09:00:10Z","type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"They pass."}]}}
"#;

/// Runs `/import <tool> <path>` for the app's agent; the sessions it
/// reported so far.
fn import_with(app: &mut App, tool: &str, path: &Path) -> Vec<SessionImported> {
    let world = app.world_mut();
    let agent = first_agent(world).unwrap_or(Entity::PLACEHOLDER);
    world.trigger(RunCommand::new(
        agent,
        format!("import {tool} {}", path.display()),
    ));
    app.update();
    app.world()
        .resource::<Collected<SessionImported>>()
        .0
        .clone()
}

/// The conversation of the main agent of `session`, as its logs left it.
fn conversation_of(home: &Home, session: &SessionId) -> Vec<Message> {
    let store = JsonlDirStore::new(home.session(session).path());
    let saved = read_session(&store).unwrap_or_default();
    let main = saved
        .agents
        .into_iter()
        .find(|agent| agent.parent.is_none());
    main.map(|agent| agent.conversation.messages().to_vec())
        .unwrap_or_default()
}

/// The tool call `id`'s place in `messages`, and whether the next message
/// holds its result.
fn paired(messages: &[Message], id: &str) -> Option<(usize, bool)> {
    let at = messages.iter().position(|message| {
        matches!(message, Message::Assistant(turn)
            if turn.tool_calls().any(|call| call.id.wire() == id))
    })?;
    let answered = matches!(messages.get(at + 1), Some(Message::User { content })
        if content.iter().any(|item| matches!(item, UserContent::ToolResult(result) if result.call.wire() == id)));
    Some((at, answered))
}

#[test]
fn imports_resumes_and_reimports() {
    let home = scratch_home("fh-import-sessions");
    let fixtures = home.root().join("fixtures");
    fs::create_dir_all(&fixtures).ok();
    let claude: PathBuf = fixtures.join("cc-1.jsonl");
    let codex: PathBuf = fixtures.join("rollout-cx-1.jsonl");
    fs::write(&claude, CLAUDE).ok();
    fs::write(&codex, CODEX).ok();

    let current = SessionId::generate();
    let (mut app, _wakes) = app_in(&home, &current);
    app.add_plugins((AgentPlugin, JournalPlugin, ImportSessionsPlugin));
    collect::<SessionImported>(&mut app);
    app.finish();
    app.update();
    import_with(&mut app, "claude-code", &claude);
    let reported = import_with(&mut app, "codex", &codex);

    // Two sessions, which `/resume` lists.
    let current_dir = home.session(&current);
    let listed = rig_sessions::list(&home, current_dir.path());
    assert_eq!(listed.len(), 2, "{reported:?}");
    assert_eq!(reported.len(), 2, "{reported:?}");
    let (Some(first), Some(second)) = (reported.first(), reported.get(1)) else {
        return;
    };
    assert!(first.created && second.created);
    assert_eq!(first.unknown(), 1, "{:?}", first.counts);
    assert_eq!(second.unknown(), 0, "{:?}", second.counts);

    // The Claude Code one: `Bash` became `shell`, with its result and a
    // note of the unknown entry.
    let messages = conversation_of(&home, &first.session);
    let shell = messages.iter().find_map(|message| match message {
        Message::Assistant(turn) => turn
            .tool_calls()
            .find(|call| call.function.name == "shell")
            .cloned(),
        _ => None,
    });
    let command = shell
        .as_ref()
        .and_then(|call| call.function.arguments.get("command")?.as_str());
    assert_eq!(command, Some("ls -la"));
    let (_, answered) = paired(&messages, "toolu_1").unwrap_or_default();
    // The sidechain is a subagent of it, and each agent keeps where it came
    // from with the source's times.
    let store = JsonlDirStore::new(home.session(&first.session).path());
    let agents = read_session(&store).unwrap_or_default().agents;
    assert_eq!(
        agents.iter().filter(|agent| agent.parent.is_some()).count(),
        1
    );
    let from = agents.iter().filter(|agent| {
        agent
            .saved
            .keys()
            .any(|path| path.ends_with("ImportedFrom"))
    });
    assert_eq!(from.count(), 2);
    assert!(answered, "the call is followed by its result");
    let result_text = messages.iter().flat_map(|message| match message {
        Message::User { content } => content.clone(),
        Message::Assistant(_) | Message::System { .. } => Vec::new(),
    });
    let texts: Vec<String> = result_text
        .filter_map(|item| match item {
            UserContent::ToolResult(result) => result
                .content
                .first()
                .and_then(ToolResultContent::as_text)
                .map(str::to_owned),
            UserContent::Text(text) => Some(text.text),
            _ => None,
        })
        .collect();
    assert!(
        texts.iter().any(|text| text.contains("main.rs")),
        "{texts:?}"
    );
    assert!(
        texts
            .iter()
            .any(|text| text.contains("file-history-snapshot")),
        "{texts:?}"
    );

    // Resumed, it continues on the current model with a valid request.
    let model = MockCompletionModel::from_stream_turns([reply("Done.")]);
    {
        let (mut resumed, wakes) = app_in(&home, &first.session);
        resumed
            .insert_resource(scripted_models([("test/model", &model)]))
            .add_plugins((AgentPlugin, JournalPlugin, ImportSessionsPlugin))
            // A `shell` here, so the request declares tools and keeps the
            // imported calls as calls.
            .add_open_tool(
                "shell",
                "A stand-in.",
                ToolOptions::default(),
                |_: On<ToolCalled<serde_json::Value>>| {},
            )
            .init_resource::<Ended>()
            .add_observer(log_ended);
        resumed.finish();
        resumed.update();
        let world = resumed.world_mut();
        let mut main = world.query_filtered::<Entity, (With<Agent>, Without<SpawnedBy>)>();
        let agent = main.iter(world).next().unwrap_or(Entity::PLACEHOLDER);
        let world = resumed.world_mut();
        world.trigger(SetModel {
            entity: agent,
            model: "test/model".to_owned(),
        });
        world.trigger(Deliver::new(agent, "continue", DeliveryMode::Steer));
        run_until(&mut resumed, &wakes, |world| {
            !world.resource::<Ended>().0.is_empty()
        });
        resumed.update();
    }
    let requests = model.requests();
    let sent = requests
        .first()
        .map(|request| request.chat_history.clone())
        .unwrap_or_default();
    let (call_at, answered) = paired(&sent, "toolu_1").unwrap_or_default();
    assert!(
        answered,
        "the call and its result are paired in the request: {sent:?}"
    );
    let user_at = sent.iter().position(|message| matches!(message, Message::User { content }
        if content.iter().any(|item| matches!(item, UserContent::Text(text) if text.text == "List the files"))));
    assert!(user_at.is_some_and(|at| at < call_at), "in order: {sent:?}");
    assert!(
        matches!(sent.last(), Some(Message::User { .. })),
        "ends with the new message"
    );

    // Imported again with a new message: still one session, the message
    // added.
    let more = r#"{"type":"user","uuid":"u3","parentUuid":"a2","isSidechain":false,"sessionId":"cc-1","timestamp":"2026-10-03T08:00:00Z","message":{"role":"user","content":"One more thing"}}"#;
    fs::write(&claude, format!("{CLAUDE}{more}\n")).ok();
    let reported = import_with(&mut app, "claude-code", &claude);
    let again = reported.last();
    assert!(
        again.is_some_and(|again| !again.created
            && again.session == first.session
            && again.added == 1),
        "{again:?}"
    );
    assert_eq!(rig_sessions::list(&home, current_dir.path()).len(), 2);
    let messages = conversation_of(&home, &first.session);
    let last = messages.last().and_then(|message| match message {
        Message::User { content } => content.iter().find_map(|item| match item {
            UserContent::Text(text) => Some(text.text.clone()),
            _ => None,
        }),
        _ => None,
    });
    assert_eq!(last.as_deref(), Some("One more thing"));
}
