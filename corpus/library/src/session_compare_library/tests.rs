use std::fmt::Display;
use std::fs;
use std::path::{Path, PathBuf};

use rig_core::conversation::Conversation;
use rig_core::message::{AssistantContent, Message, ToolName, ToolResult, ToolResultContent};
use rig_memory::{DirStore, MODEL, SavedAgent, Session, Store};
use serde_json::{Value, json};

use super::*;

/// The main agent's log in every session here.
const AGENT: &str = "agent";

/// `result`'s value, failing the test with its error.
fn ok<T>(result: Result<T, impl Display>) -> Option<T> {
    let result = result.map_err(|error| error.to_string());
    assert_eq!(result.as_ref().err(), None);
    result.ok()
}

/// A fresh directory beside the test binary.
fn scratch(name: &str) -> PathBuf {
    let exe = std::env::current_exe().unwrap_or_default();
    let dir = exe.with_file_name(format!("ct-session-compare-{name}"));
    fs::remove_dir_all(&dir).ok();
    dir
}

/// The model's reply calling `tool` (call `id`) with `args`.
fn call(id: &str, tool: &str, args: Value) -> Option<Message> {
    let name = ok(ToolName::new(tool))?;
    Some(Message::from(vec![AssistantContent::tool_call(
        id, name, args,
    )]))
}

/// The result of the call `reply` makes, failed with `failed`.
fn result(reply: &Message, failed: bool) -> Option<Message> {
    let Message::Assistant(reply) = reply else {
        return None;
    };
    let call = reply.tool_calls().next()?;
    let content = vec![ToolResultContent::text(if failed {
        "no match"
    } else {
        "ok"
    })];
    let result = if failed {
        ToolResult {
            is_error: true,
            ..call.result(content)
        }
    } else {
        call.result(content)
    };
    Some(Message::tool_results(vec![result]))
}

/// Each of `calls` followed by its result (the one at `failing` failed),
/// then the model's last word `done`.
fn work(calls: Vec<Option<Message>>, failing: Option<usize>, done: &str) -> Option<Vec<Message>> {
    let mut messages = Vec::new();
    for (at, reply) in calls.into_iter().enumerate() {
        let reply = reply?;
        let result = result(&reply, failing == Some(at))?;
        messages.extend([reply, result]);
    }
    messages.push(Message::assistant(done));
    Some(messages)
}

/// The shared first exchange: three messages, the last the user's ask.
fn shared() -> Vec<Message> {
    vec![
        Message::user("What does this repository do?"),
        Message::assistant("It parses configuration files."),
        Message::user("Fix the off-by-one in the parser."),
    ]
}

/// Side a: reads and edits `src/a.rs`, two calls.
fn side_a() -> Option<Vec<Message>> {
    let edit = json!({"path": "src/a.rs", "edits": [{"old_text": "<=", "new_text": "<"}]});
    work(
        vec![
            call("a1", "read", json!({"path": "src/a.rs"})),
            call("a2", "edit", edit),
        ],
        None,
        "Fixed in src/a.rs.",
    )
}

/// Side b: edits `src/a.rs`, writes `src/b.rs`, then fails to edit it.
fn side_b() -> Option<Vec<Message>> {
    let edit = |path: &str| json!({"path": path, "edits": [{"old_text": "<=", "new_text": "<"}]});
    work(
        vec![
            call("b1", "edit", edit("src/a.rs")),
            call(
                "b2",
                "write",
                json!({"path": "src/b.rs", "content": "// tests\n"}),
            ),
            call("b3", "edit", edit("src/b.rs")),
        ],
        Some(2),
        "Fixed; one edit of src/b.rs failed.",
    )
}

/// Saves a session whose main agent holds the [`shared`] exchange, then
/// `rest`, and ran on `model`, with what its calls used, in `dir`.
fn save_session(dir: &Path, rest: Vec<Message>, model: &str) -> Option<DirStore> {
    let store = ok(DirStore::open(dir))?;
    let mut conversation = Conversation::new();
    conversation.extend(shared());
    conversation.extend(rest);
    let mut agent = SavedAgent {
        id: AGENT.to_owned(),
        conversation,
        ..SavedAgent::default()
    };
    agent.saved.insert(MODEL.to_owned(), json!(model).into());
    let mut session = Session {
        agents: vec![agent],
        ..Session::default()
    };
    let spent = json!({
        "total": {"tokens": {"input_tokens": 1200, "output_tokens": 300}, "cost": 0.05, "unpriced": 0, "calls": 3},
        "models": {model: {"tokens": {}, "cost": 0.05, "unpriced": 0, "calls": 3}},
    });
    session
        .saved
        .insert(SESSION_SPENDING.to_owned(), spent.into());
    ok(session.save(&store))?;
    Some(store)
}

/// Continues the main agent of the session in `store` with `messages`,
/// on `model`.
fn continue_session(store: &DirStore, messages: Vec<Message>, model: &str) -> Option<()> {
    let mut session = ok(Session::read(store))?;
    session.saved.remove(SESSION_SPENDING);
    let agent = session.agent_mut(AGENT)?;
    agent.saved.insert(MODEL.to_owned(), json!(model).into());
    agent.conversation.extend(messages);
    ok(session.save(store))
}

/// Runs the program with `args`.
fn compare(args: &[&str]) -> Result<String, String> {
    let args: Vec<String> = args.iter().map(|arg| (*arg).to_owned()).collect();
    run(&args)
}

fn scenario() -> Option<()> {
    let root = scratch("scenario");
    let (a, b, text) = (
        root.join("opus.session"),
        root.join("local.session"),
        root.join("notes.txt"),
    );
    save_session(&a, side_a()?, "anthropic/claude-opus")?;

    // Side b starts with the same exchange, then went on another way.
    let local = save_session(&b, Vec::new(), "anthropic/claude-opus")?;
    continue_session(&local, side_b()?, "local/qwen")?;

    // A log a newer rig wrote: left out with a warning, not refused.
    let newer = br#"{"seq":0,"n":99,"t":0,"type":"header","format":"rig-session/99","log":"helper","lineage":7}"#;
    ok(local.append_log("helper", &[newer.as_slice(), b"\n"].concat()))?;

    let paths = [&a, &b, &text].map(|path| path.display().to_string());
    let [a, b, text] = paths.each_ref().map(String::as_str);

    let report = ok(compare(&[a, b]))?;
    assert!(
        report.contains("diverge at message 3 (3 shared)"),
        "{report}"
    );
    assert!(report.contains("files both: src/a.rs"), "{report}");
    assert!(report.contains("files only b: src/b.rs"), "{report}");

    let json: Value = ok(serde_json::from_str(&ok(compare(&["--json", a, b]))?))?;
    let at = |pointer: &str| json.pointer(pointer).cloned().unwrap_or_default();
    assert_eq!(at("/divergence/index"), json!(3));
    assert_eq!(at("/divergence/a/role"), json!("assistant"));
    assert_eq!(at("/divergence/a/text"), json!("calls read"));
    assert_eq!(at("/divergence/b/text"), json!("calls edit src/a.rs"));
    assert_eq!(at("/a/tool_calls"), json!({"read": 1, "edit": 1}));
    assert_eq!(at("/b/tool_calls"), json!({"edit": 2, "write": 1}));
    assert_eq!(
        (at("/a/failed_tool_calls"), at("/b/failed_tool_calls")),
        (json!(0), json!(1))
    );
    assert_eq!(
        (
            at("/a/user_turns"),
            at("/a/model_replies"),
            at("/b/model_replies")
        ),
        (json!(2), json!(4), json!(5))
    );
    assert_eq!(
        (at("/a/tokens_in"), at("/a/cost"), at("/b/tokens_in")),
        (json!(1200), json!(0.05), Value::Null)
    );
    assert_eq!(
        (at("/a/models"), at("/b/models")),
        (json!(["anthropic/claude-opus"]), json!(["local/qwen"]))
    );
    assert_eq!((at("/a/warnings"), at("/b/warnings")), (json!(0), json!(1)));
    assert_eq!(
        at("/files"),
        json!({"both": ["src/a.rs"], "only_a": [], "only_b": ["src/b.rs"]})
    );

    // Anything but a session is refused with a reason, not a panic.
    ok(fs::write(text, "just some notes\n"))?;
    let refused = compare(&[text, a]).err()?;
    assert!(refused.contains("is not a rig session"), "{refused}");
    fs::remove_dir_all(&root).ok()
}

#[test]
fn two_sessions_compare() {
    assert!(scenario().is_some());
}
