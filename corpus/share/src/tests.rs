use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

use rig_cassette::journal::MemoryStore;
use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::Invocation;
use rig_harness_test_support::{app_on, calls, connect, reply, run_until};

use super::*;

/// The key the tool result holds.
const KEY: &str = "sk-test1234567890abcdef1234";

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

/// What the plugin showed the user.
#[derive(Resource, Default)]
struct Shown {
    picks: Vec<PickRequest>,
    notices: Vec<String>,
    ended: usize,
}

fn record(
    mut picks: MessageReader<PickRequest>,
    mut notices: MessageReader<Notice>,
    mut shown: ResMut<Shown>,
) {
    shown.picks.extend(picks.read().cloned());
    shown
        .notices
        .extend(notices.read().map(|notice| notice.text.clone()));
}

/// Runs `/share`, then picks the picker's first item, as the user would.
fn share_and_confirm(app: &mut App, wakes: &std::sync::mpsc::Receiver<()>, agent: Entity) {
    let before = app.world().resource::<Shown>().picks.len();
    app.world_mut().trigger(RunCommand {
        entity: agent,
        line: "share".to_owned(),
    });
    run_until(app, wakes, |world| {
        world.resource::<Shown>().picks.len() > before
    });
    let shown = app.world().resource::<Shown>();
    let confirm = shown
        .picks
        .last()
        .and_then(|pick| pick.items.first())
        .map(|item| item.command.clone());
    assert_eq!(confirm.as_deref(), Some("share confirm"));
    let label = shown
        .picks
        .last()
        .and_then(|pick| pick.items.first())
        .map(|item| item.label.clone())
        .unwrap_or_default();
    assert!(label.contains("1 secrets redacted: sk-tes…"), "{label}");
    assert!(!label.contains(KEY), "{label}");
    app.world_mut().trigger(RunCommand {
        entity: agent,
        line: confirm.unwrap_or_default(),
    });
}

#[test]
fn share_posts_a_redacted_page_then_updates_the_same_one() {
    let received = Received::default();
    let url = stub(&received);
    assert!(url.is_some(), "the stub server listens");
    let store = MemoryStore::default();
    let (mut app, wakes) = app_on(&store);
    app.set_error_handler(bevy_ecs::error::warn)
        .insert_resource(Invoked {
            args: Invocation::default(),
            terminal: true,
        })
        .insert_resource(ShareConfig {
            target: ShareTarget::Http {
                url: url.unwrap_or_default(),
            },
            clipboard: false,
        })
        .init_resource::<Shown>()
        .add_plugins((AgentPlugin, JournalPlugin, SharePlugin))
        .add_systems(Last, record)
        .add_observer(|_: On<TurnEnded>, mut shown: ResMut<Shown>| shown.ended += 1)
        .add_open_tool(
            "env",
            "The environment.",
            ToolOptions::default(),
            |called: On<ToolCalled<serde_json::Value>>, mut commands: Commands| {
                let text = format!("OPENAI_API_KEY={KEY}\nHOME=/home/u");
                let result = ToolResult::success(rig_core::tool::ToolOutput::text(text));
                commands.entity(called.call).insert(ToolOutput(result));
            },
        );
    app.finish();
    app.update();
    let model = MockCompletionModel::from_stream_turns([
        calls("1", "env", serde_json::json!({})),
        reply("It holds an OpenAI key."),
    ]);
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next();
    assert!(agent.is_some(), "the agent");
    let (Some(agent), Some(connection)) = (agent, connect(&model)) else {
        return;
    };
    world.entity_mut(agent).insert(connection);
    world.trigger(Deliver::new(agent, ASKED, DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        world.resource::<Shown>().ended > 0
    });

    share_and_confirm(&mut app, &wakes, agent);
    run_until(&mut app, &wakes, |world| {
        world.get::<Shared>(agent).is_some()
    });
    let notices = &app.world().resource::<Shown>().notices;
    assert!(
        notices.iter().any(|notice| notice == "Shared: http://x/1"),
        "{notices:?}"
    );

    share_and_confirm(&mut app, &wakes, agent);
    run_until(&mut app, &wakes, |world| {
        let shown = world.resource::<Shown>();
        shown
            .notices
            .iter()
            .filter(|notice| notice.starts_with("Shared:"))
            .count()
            == 2
    });
    let received = received.lock().map(|got| got.clone()).unwrap_or_default();
    assert_eq!(received.len(), 2, "a create and an update: {received:?}");
    let [(first, first_path, page), (second, second_path, _)] = received.as_slice() else {
        return;
    };
    assert_eq!((first.as_str(), first_path.as_str()), ("POST", "/"));
    assert!(page.contains("<html>") && page.contains(ASKED), "{page}");
    assert!(page.contains("OPENAI_API_KEY=[REDACTED]"), "{page}");
    assert!(!page.contains(KEY), "the key is not shared");
    assert_eq!((second.as_str(), second_path.as_str()), ("PUT", "/1"));
}
