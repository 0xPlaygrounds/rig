use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use rig_cassette::journal::MemoryStore;
use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::{app_on, connect, reply};

use super::*;

/// What the client saw.
#[derive(Debug, Default)]
struct Seen {
    unauthorized: Option<String>,
    posted: Option<String>,
    deltas: String,
    turn_ended: bool,
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

/// The test's client: a request without the token, the event stream, and
/// "hi" posted to `agent`.
fn client(api: &HttpApi, agent: &str) -> std::io::Result<Seen> {
    let mut seen = Seen::default();
    let (status, _) = send(api.addr, "GET /agents HTTP/1.1\r\nHost: x\r\n\r\n")?;
    seen.unauthorized = Some(status);
    let auth = format!("Authorization: Bearer {}", api.token);
    let (_, mut events) = send(
        api.addr,
        &format!("GET /events HTTP/1.1\r\nHost: x\r\n{auth}\r\n\r\n"),
    )?;
    let body = r#"{"text":"hi","mode":"queue"}"#;
    let post = format!(
        "POST /agents/{agent}/messages HTTP/1.1\r\nHost: x\r\n{auth}\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    let (status, mut rest) = send(api.addr, &post)?;
    rest.read_to_end(&mut Vec::new())?;
    seen.posted = Some(status);
    let mut kind = String::new();
    let mut line = String::new();
    while events.read_line(&mut line)? > 0 {
        if let Some(name) = line.trim_end().strip_prefix("event: ") {
            kind = name.to_owned();
        } else if let Some(data) = line.trim_end().strip_prefix("data: ") {
            let data: Value = serde_json::from_str(data).unwrap_or_default();
            match kind.as_str() {
                "text_delta" => {
                    let text = data.get("text").and_then(Value::as_str);
                    seen.deltas.push_str(text.unwrap_or_default());
                }
                "turn_end" => {
                    seen.turn_ended = true;
                    break;
                }
                _ => {}
            }
        }
        line.clear();
    }
    Ok(seen)
}

#[test]
fn a_posted_message_streams_its_reply_and_the_token_is_required() {
    let store = MemoryStore::default();
    let (mut app, wakes) = app_on(&store);
    let serve = Serve {
        addr: "127.0.0.1:0".to_owned(),
        allow_remote: false,
        token_file: None,
    };
    app.add_plugins((
        AgentPlugin,
        JournalPlugin,
        HttpApiPlugin { serve: Some(serve) },
    ));
    app.finish();
    app.update();
    let model = MockCompletionModel::from_stream_turns([reply("hello")]);
    let world = app.world_mut();
    let agent = world
        .query_filtered::<(Entity, &AgentId), With<Agent>>()
        .iter(world)
        .next()
        .map(|(agent, id)| (agent, id.0.clone()));
    let api = world.get_resource::<HttpApi>().cloned();
    let ready = (agent, api, connect(&model));
    assert!(
        matches!(ready, (Some(_), Some(_), Some(_))),
        "an agent, the server and the model"
    );
    let (Some((agent, id)), Some(api), Some(connection)) = ready else {
        return;
    };
    world.entity_mut(agent).insert(connection);
    assert_ne!(api.addr.port(), 0, "the bound port is exposed");

    let (sender, seen) = mpsc::channel();
    std::thread::spawn(move || sender.send(client(&api, &id)).ok());
    let started = Instant::now();
    let seen = loop {
        app.update();
        wakes.recv_timeout(Duration::from_millis(50)).ok();
        if let Ok(seen) = seen.try_recv() {
            break seen;
        }
        assert!(
            started.elapsed() < Duration::from_secs(20),
            "the client hung"
        );
    };
    assert!(seen.is_ok(), "the client failed: {seen:?}");
    let Ok(seen) = seen else {
        return;
    };
    assert_eq!(
        seen.unauthorized.as_deref(),
        Some("HTTP/1.1 401 Unauthorized")
    );
    assert_eq!(seen.posted.as_deref(), Some("HTTP/1.1 202 Accepted"));
    assert!(seen.deltas.contains("hello"), "{seen:?}");
    assert!(seen.turn_ended, "{seen:?}");
}

/// The commands run, by their arguments.
#[derive(Resource, Default)]
struct Ran(Vec<String>);

/// What the pickers reported.
#[derive(Resource, Default)]
struct Reported(Vec<Option<Chosen>>);

/// The status line of `POST /approvals/0` choosing `choice`.
fn approve(api: &HttpApi, choice: &str) -> std::io::Result<String> {
    let body = format!(r#"{{"choice":"{choice}"}}"#);
    let post = format!(
        "POST /approvals/0 HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {}\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        api.token,
        body.len()
    );
    let (status, mut rest) = send(api.addr, &post)?;
    rest.read_to_end(&mut Vec::new())?;
    Ok(status)
}

#[test]
fn a_plugins_picker_is_answered_over_http_and_closed() {
    let (mut app, wakes) = app_on(&MemoryStore::default());
    let serve = Serve {
        addr: "127.0.0.1:0".to_owned(),
        allow_remote: false,
        token_file: None,
    };
    app.add_plugins((AgentPlugin, HttpApiPlugin { serve: Some(serve) }))
        .init_resource::<Ran>()
        .init_resource::<Reported>()
        .add_command(
            "note",
            "Notes its arguments",
            |In(args): In<CommandArgs>, mut ran: ResMut<Ran>| ran.0.push(args.args),
        )
        .add_observer(|picked: On<Picked>, mut reported: ResMut<Reported>| {
            reported.0.push(picked.chosen.clone());
        });
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = world.spawn(Agent).id();
    let picker = Picker::new("Size")
        .for_agent(agent)
        .item("small", "note small")
        .item("large", "note large");
    let picker = world.spawn(picker).id();
    let api = world.get_resource::<HttpApi>().cloned();
    assert!(api.is_some(), "the server runs");
    let Some(api) = api else {
        return;
    };
    app.update();

    let (sender, statuses) = mpsc::channel();
    std::thread::spawn(move || {
        let choices = ["huge", "large", "small"];
        let statuses: std::io::Result<Vec<String>> =
            choices.iter().map(|choice| approve(&api, choice)).collect();
        sender.send(statuses).ok();
    });
    let started = Instant::now();
    let statuses = loop {
        app.update();
        wakes.recv_timeout(Duration::from_millis(50)).ok();
        if let Ok(statuses) = statuses.try_recv() {
            break statuses;
        }
        assert!(
            started.elapsed() < Duration::from_secs(20),
            "the client hung"
        );
    };
    let statuses = statuses.unwrap_or_default();
    assert_eq!(
        statuses,
        [
            "HTTP/1.1 400 Bad Request",
            "HTTP/1.1 202 Accepted",
            "HTTP/1.1 404 Not Found"
        ]
    );
    assert_eq!(app.world().resource::<Ran>().0, ["large"]);
    assert_eq!(
        app.world().resource::<Reported>().0,
        [Some(Chosen::Item(1))]
    );
    assert!(app.world().get_entity(picker).is_err(), "closed");
}
