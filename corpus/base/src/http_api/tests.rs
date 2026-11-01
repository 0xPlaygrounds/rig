use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::pick::PickPlugin;
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{app_on, connect, scratch_home};

use super::*;

/// An app serving on a free localhost port, its token written under a
/// scratch `RIG_HOME`.
fn serving(test: &str) -> (App, Receiver<()>, Home) {
    let home = scratch_home(&format!("corpus-http-api-{test}"));
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(Settings::from_toml("[serve]\naddr = \"127.0.0.1:0\""))
        .insert_resource(RigHome(home.clone()))
        .add_plugins((AgentPlugin, SessionLogPlugin, PickPlugin, HttpApiPlugin));
    app.finish();
    app.update();
    (app, wakes, home)
}

/// Runs frames until the client thread sends what it saw.
fn drive<T>(app: &mut App, wakes: &Receiver<()>, client: impl FnOnce() -> T + Send + 'static) -> T
where
    T: Send + 'static,
{
    let (sender, seen) = mpsc::channel();
    std::thread::spawn(move || sender.send(client()).ok());
    let started = Instant::now();
    loop {
        app.update();
        wakes.recv_timeout(Duration::from_millis(20)).ok();
        if let Ok(seen) = seen.try_recv() {
            return seen;
        }
        assert!(
            started.elapsed() < Duration::from_secs(20),
            "the client hung"
        );
    }
}

/// Sends a request and returns the response's status line and the rest.
fn send(addr: SocketAddr, request: &str) -> std::io::Result<(String, BufReader<TcpStream>)> {
    let mut stream = TcpStream::connect(addr)?;
    stream.set_read_timeout(Some(Duration::from_secs(10)))?;
    stream.write_all(request.as_bytes())?;
    let mut reader = BufReader::new(stream);
    let mut status = String::new();
    reader.read_line(&mut status)?;
    Ok((status.trim_end().to_owned(), reader))
}

/// The status line of an authorized `POST` of `body` to `path`.
fn post(api: &HttpApi, path: &str, body: &str) -> std::io::Result<String> {
    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {}\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        api.token,
        body.len()
    );
    let (status, mut rest) = send(api.addr, &request)?;
    rest.read_to_end(&mut Vec::new())?;
    Ok(status)
}

/// The authorized event stream.
fn events(api: &HttpApi) -> std::io::Result<BufReader<TcpStream>> {
    let request = format!(
        "GET /events HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {}\r\n\r\n",
        api.token
    );
    Ok(send(api.addr, &request)?.1)
}

/// Reads events until `done` says the stream told enough.
fn read_events(
    stream: &mut BufReader<TcpStream>,
    mut done: impl FnMut(&str, &Value) -> bool,
) -> std::io::Result<()> {
    let (mut kind, mut line) = (String::new(), String::new());
    while stream.read_line(&mut line)? > 0 {
        if let Some(name) = line.trim_end().strip_prefix("event: ") {
            kind = name.to_owned();
        } else if let Some(data) = line.trim_end().strip_prefix("data: ")
            && done(&kind, &serde_json::from_str(data).unwrap_or_default())
        {
            return Ok(());
        }
        line.clear();
    }
    Ok(())
}

/// What the client saw.
#[derive(Debug, Default)]
struct Seen {
    unauthorized: String,
    posted: String,
    deltas: String,
    answer: Option<String>,
    turn_ended: bool,
}

#[test]
fn a_posted_message_streams_its_reply_and_the_token_is_required() {
    let (mut app, wakes, home) = serving("message");
    // A key the reply streams in pieces reaches no event.
    let pieces = [
        "Use GITHUB_TOKEN=gh",
        "p_abcdefghijklmnopqrst",
        "uvwxyz0123456789 now.",
    ];
    let mut streamed: Vec<_> = pieces
        .iter()
        .map(|piece| MockStreamEvent::text(*piece))
        .collect();
    streamed.push(MockStreamEvent::final_response_with_default_usage());
    let model = MockCompletionModel::from_stream_turns([streamed]);
    let world = app.world_mut();
    let agent = world
        .query_filtered::<(Entity, &AgentId), With<Agent>>()
        .iter(world)
        .next()
        .map(|(agent, id)| (agent, id.0.clone()));
    let ready = (
        agent,
        world.get_resource::<HttpApi>().cloned(),
        connect(&model),
    );
    assert!(
        matches!(ready, (Some(_), Some(_), Some(_))),
        "an agent, the server and the model"
    );
    let (Some((agent, id)), Some(api), Some(connection)) = ready else {
        return;
    };
    world.entity_mut(agent).insert(connection);
    assert_ne!(api.addr.port(), 0, "the bound port is exposed");
    let token = std::fs::read_to_string(home.root().join("serve-token")).ok();
    std::fs::remove_dir_all(home.root()).ok();
    assert_eq!(token.as_deref(), Some(api.token.as_str()));

    let seen = drive(&mut app, &wakes, move || {
        let mut seen = Seen {
            unauthorized: send(api.addr, "GET /agents HTTP/1.1\r\nHost: x\r\n\r\n")?.0,
            ..Seen::default()
        };
        let mut stream = events(&api)?;
        let body = r#"{"text":"hi","mode":"queue"}"#;
        seen.posted = post(&api, &format!("/agents/{id}/messages"), body)?;
        read_events(&mut stream, |kind, data| {
            if kind == "text_delta" {
                let text = data.get("text").and_then(Value::as_str);
                seen.deltas.push_str(text.unwrap_or_default());
            }
            seen.turn_ended = kind == "turn_end";
            if seen.turn_ended {
                let answer = data.get("answer").and_then(Value::as_str);
                seen.answer = answer.map(str::to_owned);
            }
            seen.turn_ended
        })?;
        std::io::Result::Ok(seen)
    });
    assert!(seen.is_ok(), "the client failed: {seen:?}");
    let Ok(seen) = seen else {
        return;
    };
    assert_eq!(seen.unauthorized, "HTTP/1.1 401 Unauthorized");
    assert_eq!(seen.posted, "HTTP/1.1 202 Accepted");
    let redacted = "Use GITHUB_TOKEN=[REDACTED] now.";
    assert_eq!(seen.deltas, redacted, "{seen:?}");
    assert_eq!(seen.answer.as_deref(), Some(redacted), "{seen:?}");
}

/// How each question and picker was answered.
#[derive(Resource, Default)]
struct Answers {
    asked: Vec<Option<Answer>>,
    picked: Vec<Option<Chosen>>,
}

#[test]
fn a_plugins_question_and_picker_are_answered_over_http() {
    let (mut app, wakes, home) = serving("ask");
    std::fs::remove_dir_all(home.root()).ok();
    let api = app.world().get_resource::<HttpApi>().cloned();
    assert!(api.is_some(), "the server runs");
    let Some(api) = api else {
        return;
    };
    let world = app.world_mut();
    let answerers = world.query::<&Answerer>().iter(world).count();
    assert_eq!(answerers, 1, "the server answers questions");

    // A question and a picker open once the client listens; their answers
    // are kept.
    app.init_resource::<Answers>().add_observer(
        |answered: On<Add<Outcome>>,
         asks: Query<&Outcome, With<Ask>>,
         mut answers: ResMut<Answers>| {
            if let Ok(outcome) = asks.get(answered.entity) {
                answers.asked.push(Answer::of(outcome));
            }
        },
    );
    app.add_systems(
        Update,
        |subscribers: Res<Subscribers>, mut asked: Local<bool>, mut commands: Commands| {
            if !*asked && !subscribers.clients.is_empty() {
                *asked = true;
                commands.spawn(Ask::yes_no("Deploy?"));
                let picker = Picker::new("Branch").item("main", "").item("dev", "");
                commands.spawn(picker).observe(
                    |picked: On<Picked>, mut answers: ResMut<Answers>| {
                        answers.picked.push(picked.chosen.clone());
                    },
                );
            }
        },
    );
    let status = drive(&mut app, &wakes, move || {
        let mut stream = events(&api)?;
        let mut ids = Vec::new();
        read_events(&mut stream, |kind, data| {
            if kind == "approval_request" {
                ids.extend(data.get("id").and_then(Value::as_u64));
            }
            ids.len() == 2
        })?;
        let paths = ids.iter().map(|id| format!("/approvals/{id}"));
        let [ask, picker] = <[String; 2]>::try_from(paths.collect::<Vec<_>>()).unwrap_or_default();
        let statuses = [
            post(&api, &ask, r#"{"choice":"maybe"}"#)?,
            post(&api, &ask, r#"{"choice":"yes"}"#)?,
            post(&api, &picker, r#"{"choice":"dev"}"#)?,
        ];
        let mut closed = 0;
        read_events(&mut stream, |kind, _| {
            closed += usize::from(kind == "approval_closed");
            closed == 2
        })?;
        std::io::Result::Ok((statuses, closed))
    });
    assert!(status.is_ok(), "the client failed: {status:?}");
    let Ok((statuses, closed)) = status else {
        return;
    };
    let accepted = "HTTP/1.1 202 Accepted";
    assert_eq!(statuses, ["HTTP/1.1 400 Bad Request", accepted, accepted]);
    assert_eq!(closed, 2, "the stream says both closed");
    let answers = app.world().resource::<Answers>();
    assert_eq!(answers.asked, [Some(Answer::choice("yes"))]);
    assert_eq!(answers.picked, [Some(Chosen::item(1))]);
}
