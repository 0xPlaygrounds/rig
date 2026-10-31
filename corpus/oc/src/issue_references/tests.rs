use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex, PoisonError};

use rig_core::message::Message as ChatMessage;
use rig_core::test_utils::MockCompletionModel;
use rig_harness::settings::Settings;
use rig_harness_test_support::{
    Ended, MemoryStore, ToolOutputs, app_on, calls, first_agent, log_ended, log_tool_outputs,
    press, reply, run_until, scripted_models, tui_view,
};
use rig_tui::{Editor, InputContext, MainEditor};
use serde_json::{Value, json};

use super::*;

const TYPED: &str = "fix #142 and see !87, ignore `#9` in code, and #9";

/// A GitHub API on a local port: issue 142 with 12 comments, pull request
/// 87 with two unresolved review threads and a resolved one, 404 for
/// anything else. It logs each request's method and path.
struct FakeTracker {
    url: String,
    log: Arc<Mutex<Vec<String>>>,
}

impl FakeTracker {
    fn start() -> Option<Self> {
        let listener = TcpListener::bind("127.0.0.1:0").ok()?;
        let url = format!("http://{}", listener.local_addr().ok()?);
        let log = Arc::new(Mutex::new(Vec::new()));
        let kept = log.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                serve(stream, &kept);
            }
        });
        Some(Self { url, log })
    }

    /// How many requests asked for exactly `path`.
    fn asked(&self, path: &str) -> usize {
        let log = self.log.lock().unwrap_or_else(PoisonError::into_inner);
        log.iter()
            .filter(|line| line.split(' ').nth(1) == Some(path))
            .count()
    }
}

/// Answers one request and closes the connection.
fn serve(stream: TcpStream, log: &Mutex<Vec<String>>) {
    let mut reader = BufReader::new(&stream);
    let mut request_line = String::new();
    reader.read_line(&mut request_line).ok();
    let mut length = 0;
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header).unwrap_or(0) == 0 || header.trim().is_empty() {
            break;
        }
        if let Some(value) = header.to_ascii_lowercase().strip_prefix("content-length:") {
            length = value.trim().parse().unwrap_or(0);
        }
    }
    let mut body = vec![0; length];
    reader.read_exact(&mut body).ok();
    let mut words = request_line.split_whitespace();
    let (method, target) = (
        words.next().unwrap_or_default(),
        words.next().unwrap_or_default(),
    );
    let path = target.split('?').next().unwrap_or_default();
    log.lock()
        .unwrap_or_else(PoisonError::into_inner)
        .push(format!("{method} {path}"));
    let (status, answer) = route(method, path);
    let text = answer.to_string();
    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{text}",
        text.len()
    );
    (&stream).write_all(response.as_bytes()).ok();
}

fn route(method: &str, path: &str) -> (&'static str, Value) {
    let comment = |n: usize| json!({ "user": { "login": "bob" }, "body": format!("c{n:02}") });
    let thread = |resolved: bool, path: &str, line: u64, author: &str, body: &str| {
        let comments = json!({ "nodes": [{ "author": { "login": author }, "body": body }] });
        json!({ "isResolved": resolved, "path": path, "line": line, "comments": comments })
    };
    match (method, path) {
        ("GET", "/repos/acme/widget/issues") => (
            "200 OK",
            json!([
                { "number": 142, "title": "Crash on empty input" },
                { "number": 87, "title": "Handle empty input", "pull_request": {} },
                { "number": 140, "title": "Docs typo" },
            ]),
        ),
        ("GET", "/repos/acme/widget/issues/142") => (
            "200 OK",
            json!({
                "title": "Crash on empty input", "state": "open", "labels": [{ "name": "bug" }],
                "user": { "login": "alice" }, "comments": 12,
                "body": "thread 'main' panicked at src/parse.rs:12:5",
            }),
        ),
        ("GET", "/repos/acme/widget/issues/142/comments") => {
            ("200 OK", (1..=12).map(comment).collect())
        }
        ("GET", "/repos/acme/widget/pulls/87") => (
            "200 OK",
            json!({
                "title": "Handle empty input", "state": "open", "labels": [],
                "user": { "login": "carol" }, "comments": 0, "body": "Fixes #142",
            }),
        ),
        ("GET", "/repos/acme/widget/issues/87/comments") => ("200 OK", json!([])),
        ("GET", "/repos/acme/widget/pulls/87/files") => (
            "200 OK",
            json!([{ "filename": "src/parse.rs" }, { "filename": "tests/parse.rs" }]),
        ),
        ("POST", "/graphql") => {
            let threads = [
                thread(false, "src/parse.rs", 12, "dave", "handle None too"),
                thread(true, "src/lib.rs", 1, "erin", "already fixed"),
                thread(false, "tests/parse.rs", 3, "erin", "add an empty case"),
            ];
            let pull = json!({ "reviewThreads": { "nodes": threads } });
            (
                "200 OK",
                json!({ "data": { "repository": { "pullRequest": pull } } }),
            )
        }
        _ => ("404 Not Found", json!({ "message": "Not Found" })),
    }
}

/// The texts of the last user message of `messages`.
fn last_user_texts(messages: &[impl std::borrow::Borrow<ChatMessage>]) -> Vec<String> {
    let content = messages
        .iter()
        .map(std::borrow::Borrow::<ChatMessage>::borrow)
        .rev()
        .find_map(|message| match message {
            ChatMessage::User { content } => Some(content.clone()),
            _ => None,
        });
    content
        .unwrap_or_default()
        .iter()
        .filter_map(|item| match item {
            UserContent::Text(text) => Some(text.text().to_owned()),
            _ => None,
        })
        .collect()
}

fn send(
    app: &mut App,
    wakes: &std::sync::mpsc::Receiver<()>,
    agent: Entity,
    text: &str,
    turns: usize,
) {
    app.world_mut()
        .trigger(Deliver::new(agent, text, DeliveryMode::Steer));
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() >= turns
    });
}

#[test]
fn references_go_with_the_message_once_and_through_the_tool() {
    let server = FakeTracker::start();
    assert!(server.is_some(), "no local port for the fake tracker");
    let Some(server) = server else { return };
    let model = MockCompletionModel::from_stream_turns([
        reply("on it"),
        calls("t1", TRACKER_GET, json!({ "ref": "#142" })),
        reply("done"),
    ]);
    let store = MemoryStore::default();
    let (mut app, wakes) = app_on(&store);
    app.insert_resource(settings(&server))
        .insert_resource(scripted_models([("test/model", &model)]))
        .add_plugins((AgentPlugin, SessionLogPlugin, IssueReferencesPlugin))
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs)
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    app.world_mut().trigger(SetModel {
        entity: agent,
        model: "test/model".to_owned(),
        kind: ChoiceKind::ForThisRun,
    });
    app.update();

    send(&mut app, &wakes, agent, TYPED, 1);
    let requests = model.requests();
    let first = requests
        .first()
        .map(|request| request.chat_history.as_slice());
    let sent = last_user_texts(first.unwrap_or_default());
    assert_eq!(
        sent.first().map(String::as_str),
        Some(TYPED),
        "the text first, as typed"
    );
    let block = sent
        .iter()
        .find(|text| text.contains("Crash on empty input"))
        .cloned()
        .unwrap_or_default();
    assert!(block.contains("comments (last 10 of 12)"), "{block}");
    assert!(
        block.contains("c03") && block.contains("c12") && !block.contains("c02"),
        "{block}"
    );
    let pull = sent
        .iter()
        .find(|text| text.contains("Handle empty input"))
        .cloned()
        .unwrap_or_default();
    assert!(
        pull.contains("- src/parse.rs:12 dave: handle None too"),
        "{pull}"
    );
    assert!(
        pull.contains("- tests/parse.rs:3 erin: add an empty case"),
        "{pull}"
    );
    assert!(
        !pull.contains("already fixed") && pull.contains("changed files (2)"),
        "{pull}"
    );
    assert!(
        sent.iter()
            .any(|text| text.starts_with("could not load #9: 404")),
        "{sent:?}"
    );
    assert_eq!(sent.len(), 4, "the text and three references: {sent:?}");
    assert_eq!(
        server.asked("/repos/acme/widget/issues/9"),
        1,
        "the `#9` in code is not loaded"
    );

    let stored = app
        .world()
        .get::<Conversation>(agent)
        .and_then(|conversation| {
            conversation
                .messages()
                .iter()
                .find_map(|message| match &**message {
                    ChatMessage::User { content } => match content.first() {
                        Some(UserContent::Text(text)) => Some(text.text().to_owned()),
                        _ => None,
                    },
                    _ => None,
                })
        });
    assert_eq!(stored.as_deref(), Some(TYPED), "stored as typed");

    send(&mut app, &wakes, agent, "again #142", 2);
    assert_eq!(
        server.asked("/repos/acme/widget/issues/142"),
        1,
        "cached for the session"
    );
    let outputs = &app.world().resource::<ToolOutputs>().0;
    assert_eq!(
        outputs.get("t1"),
        Some(&block),
        "the tool gives the same block"
    );
}

/// The `[tracker]` table for `server`.
fn settings(server: &FakeTracker) -> Settings {
    Settings::from_toml(&format!(
        "[tracker]\nkind = \"github\"\napi = \"{}\"\nrepo = \"acme/widget\"\nhost = \"github.com\"\n",
        server.url
    ))
}

#[test]
fn hash_offers_the_open_issues_by_title() {
    let server = FakeTracker::start();
    assert!(server.is_some(), "no local port for the fake tracker");
    let Some(server) = server else { return };
    let store = MemoryStore::default();
    let (mut app, wakes) = app_on(&store);
    app.insert_resource(settings(&server)).add_plugins((
        AgentPlugin,
        SessionLogPlugin,
        IssueReferencesPlugin,
    ));
    tui_view(&mut app);
    app.finish();
    app.update();
    press(&mut app, "f i x space #");
    run_until(&mut app, &wakes, |world| {
        world.resource::<OpenIssues>().issues.is_some()
    });
    press(&mut app, "1 4");
    let listed: Vec<(String, String)> = app
        .world()
        .resource::<InputContext>()
        .completion()
        .map(|open| open.items.iter())
        .into_iter()
        .flatten()
        .map(|item| (item.text.clone(), item.detail.clone()))
        .collect();
    let issue = |number: &str, title: &str| (number.to_owned(), title.to_owned());
    assert_eq!(
        listed,
        [
            issue("142", "Crash on empty input"),
            issue("140", "Docs typo")
        ],
        "open issues by title, not the pull request"
    );
    press(&mut app, "tab");
    let world = app.world_mut();
    let mut main = world.query_filtered::<&Editor, With<MainEditor>>();
    let typed = main.single(world).map(Editor::text);
    assert_eq!(typed.ok(), Some("fix #142 "));
    assert_eq!(server.asked("/repos/acme/widget/issues"), 1, "loaded once");
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(IssueReferencesPlugin);
}
