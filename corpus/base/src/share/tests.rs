use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{
    AnswerAsks, Answers, Collected, Ended, app_on, calls, collect, connect, first_agent, log_ended,
    reply, run_until,
};

use super::*;

/// The key the tool result holds.
const KEY: &str = "sk-test1234567890abcdef1234567890abcd";

/// What the user asks.
const ASKED: &str = "Which key does the environment hold?";

/// The requests the stub server got: method, path and body.
type Received = Arc<Mutex<Vec<(String, String, String)>>>;

/// A paste server on 127.0.0.1 that records each request and answers
/// `{"url":"http://x/1"}`; its URL.
fn stub(received: &Received) -> Option<String> {
    let listener = TcpListener::bind("127.0.0.1:0").ok()?;
    let url = format!("http://{}/", listener.local_addr().ok()?);
    let received = Arc::clone(received);
    std::thread::spawn(move || {
        for stream in listener.incoming().take(2).flatten() {
            let mut reader = BufReader::new(stream);
            let mut line = String::new();
            reader.read_line(&mut line).ok();
            let mut words = line.split_whitespace();
            let method = words.next().unwrap_or_default().to_owned();
            let path = words.next().unwrap_or_default().to_owned();
            let mut length = 0;
            loop {
                let mut header = String::new();
                if reader.read_line(&mut header).unwrap_or(0) == 0 || header.trim().is_empty() {
                    break;
                }
                if let Some((name, value)) = header.split_once(':')
                    && name.eq_ignore_ascii_case("content-length")
                {
                    length = value.trim().parse().unwrap_or(0);
                }
            }
            let mut body = vec![0; length];
            reader.read_exact(&mut body).ok();
            let body = String::from_utf8_lossy(&body).into_owned();
            if let Ok(mut received) = received.lock() {
                received.push((method, path, body));
            }
            let answer = r#"{"url":"http://x/1"}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\
                 connection: close\r\n\r\n{answer}",
                answer.len()
            );
            reader.get_mut().write_all(response.as_bytes()).ok();
        }
    });
    Some(url)
}

/// The bodies of the questions asked.
#[derive(Resource, Default)]
struct Bodies(Vec<String>);

fn shared_links(world: &World) -> usize {
    let notices = &world.resource::<Collected<Notice>>().0;
    notices
        .iter()
        .filter(|notice| notice.text.starts_with("Shared:"))
        .count()
}

/// Types `/share`, which the front's script confirms, and waits for the
/// link.
fn share(app: &mut App, wakes: &std::sync::mpsc::Receiver<()>, agent: Entity) {
    let links = shared_links(app.world());
    app.world_mut()
        .trigger(Deliver::new(agent, "/share", DeliveryMode::Queue));
    run_until(app, wakes, |world| shared_links(world) > links);
}

#[test]
fn share_posts_a_redacted_page_then_updates_the_same_one() {
    let received = Received::default();
    let url = stub(&received).unwrap_or_default();
    assert!(!url.is_empty(), "the stub server listens");
    let settings = format!("[share]\ntarget = \"http\"\nurl = \"{url}\"\n");
    let store = MemoryStore::default();
    let (mut app, wakes) = app_on(&store);
    let confirm = Answer::choice("share");
    app.insert_resource(Settings::from_toml(&settings))
        .insert_resource(Answers::new([confirm.clone(), confirm]))
        .init_resource::<Bodies>()
        .init_resource::<Ended>()
        .add_plugins((AgentPlugin, SessionLogPlugin, AnswerAsks, SharePlugin))
        .init_resource::<rig_harness::worker::WorkerOutbox>()
        .add_observer(log_ended)
        .add_observer(
            |asked: On<Add<Ask>>, asks: Query<&Ask>, mut bodies: ResMut<Bodies>| {
                bodies
                    .0
                    .extend(asks.get(asked.entity).map(|ask| ask.body.clone()));
            },
        )
        .add_open_tool(
            "env",
            "The environment.",
            ToolOptions::default(),
            |called: On<ToolCalled<serde_json::Value>>, mut commands: Commands| {
                let text = format!("OPENAI_API_KEY={KEY}\nHOME=/home/u");
                let result = ToolResult::success(text.into());
                commands
                    .entity(called.call)
                    .insert_if_new(Outcome::from(result));
            },
        );
    collect::<Notice>(&mut app);
    collect::<CopyToClipboard>(&mut app);
    app.finish();
    app.update();
    let model = MockCompletionModel::from_stream_turns([
        calls("1", "env", serde_json::json!({})),
        reply("It holds an OpenAI key."),
    ]);
    let world = app.world_mut();
    let agent = first_agent(world).unwrap_or(Entity::PLACEHOLDER);
    if let Some(connection) = connect(&model) {
        world.entity_mut(agent).insert(connection);
    }
    // The kernel redacts the key the tool prints before it is committed;
    // the one the user pasted stays in the conversation for share to catch.
    let asked = format!("{ASKED} Mine is {KEY}.");
    world.trigger(Deliver::new(agent, asked, DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });

    share(&mut app, &wakes, agent);
    let notices = &app.world().resource::<Collected<Notice>>().0;
    assert!(
        notices
            .iter()
            .any(|notice| notice.text == "Shared: http://x/1"),
        "{notices:?}"
    );
    let copied = &app.world().resource::<Collected<CopyToClipboard>>().0;
    assert_eq!(copied, &[CopyToClipboard("http://x/1".to_owned())]);
    // The views are sent the link to copy.
    let outbox = app.world().resource::<rig_harness::worker::WorkerOutbox>();
    let links: Vec<&serde_json::Value> = outbox
        .events
        .iter()
        .filter(|event| event.kind == SharedLink::KIND)
        .map(|event| &event.data)
        .collect();
    assert_eq!(links, [&serde_json::json!({ "url": "http://x/1" })]);
    let body = app
        .world()
        .resource::<Bodies>()
        .0
        .first()
        .cloned()
        .unwrap_or_default();
    assert!(
        body.contains("sk-tes… (37 chars)") && !body.contains(KEY),
        "{body}"
    );
    let id = app
        .world()
        .get::<Shared>(agent)
        .map(|shared| shared.id.clone());
    assert_eq!(id.as_deref(), Some("1"));

    share(&mut app, &wakes, agent);
    let asked = &app.world().resource::<Answers>().asked;
    assert_eq!(asked.len(), 2, "each /share asks: {asked:?}");
    let received = received.lock().map(|got| got.clone()).unwrap_or_default();
    let requests: Vec<(&str, &str)> = received
        .iter()
        .map(|(method, path, _)| (method.as_str(), path.as_str()))
        .collect();
    assert_eq!(
        requests,
        [("POST", "/"), ("PUT", "/1")],
        "a create, then an update"
    );
    let page = received
        .first()
        .map(|(.., page)| page.as_str())
        .unwrap_or_default();
    assert!(page.contains("<html>") && page.contains(ASKED), "{page}");
    assert!(page.contains("OPENAI_API_KEY=[REDACTED]"), "{page}");
    assert!(!page.contains(KEY), "the key is not shared");
}
