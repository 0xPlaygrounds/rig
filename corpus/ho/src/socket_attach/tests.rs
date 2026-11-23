use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::SessionId;
use rig_harness::pick::PickPlugin;
use rig_harness_test_support::{
    Collected, Ended, app_in, collect, commit_message, first_agent, log_ended, reply, scratch_home,
    scripted_models,
};

use super::*;

/// A client's connection, read line by line.
struct Connection {
    reader: BufReader<UnixStream>,
    writer: UnixStream,
}

impl Connection {
    fn open(path: &PathBuf) -> std::io::Result<Self> {
        let writer = UnixStream::connect(path)?;
        writer.set_read_timeout(Some(Duration::from_secs(10)))?;
        let reader = BufReader::new(writer.try_clone()?);
        Ok(Self { reader, writer })
    }

    fn event(&mut self) -> std::io::Result<Value> {
        let mut line = String::new();
        self.reader.read_line(&mut line)?;
        serde_json::from_str(&line).map_err(std::io::Error::other)
    }

    fn type_line(&mut self, text: &str) -> std::io::Result<()> {
        writeln!(self.writer, "{}", json!({ "text": text }))
    }

    /// The events up to the next turn's end or the session's, by type,
    /// with what streamed as one `text` event.
    fn until_turn_end(&mut self) -> std::io::Result<Vec<(String, String)>> {
        let mut seen: Vec<(String, String)> = Vec::new();
        loop {
            let event = self.event()?;
            let kind = event
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            let text = event
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default();
            match seen.last_mut() {
                Some((last, streamed)) if kind == "text" && last == "text" => {
                    streamed.push_str(text);
                }
                _ => seen.push((kind.clone(), text.to_owned())),
            }
            if kind == "turn_end" || kind == "ended" {
                return Ok(seen);
            }
        }
    }
}

/// Runs frames until the client thread sends what it saw.
fn drive<T: Send + 'static>(
    app: &mut App,
    wakes: &mpsc::Receiver<()>,
    client: impl FnOnce() -> T + Send + 'static,
) -> T {
    let (sender, seen) = mpsc::channel();
    std::thread::spawn(move || sender.send(client()).ok());
    let started = Instant::now();
    loop {
        app.update();
        wakes.recv_timeout(Duration::from_millis(10)).ok();
        if let Ok(seen) = seen.try_recv() {
            return seen;
        }
        assert!(
            started.elapsed() < Duration::from_secs(30),
            "the clients hung"
        );
    }
}

/// What the clients saw.
#[derive(Debug, Default)]
struct Seen {
    backlogs: Vec<Value>,
    first_turn: Vec<Vec<(String, String)>>,
    big_turns: usize,
}

#[test]
fn clients_watch_and_type_and_one_that_never_reads_is_dropped() {
    let big = "x".repeat(800_000);
    let model =
        MockCompletionModel::from_stream_turns([reply("hi there"), reply(&big), reply(&big)]);
    let home = scratch_home("attach");
    let id = SessionId::generate();
    let (mut app, wakes) = app_in(&home, &id);
    app.insert_resource(scripted_models([("test/model", &model)]))
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            PickPlugin,
            SocketAttachPlugin,
        ))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    collect::<Notice>(&mut app);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let model = "test/model".to_owned();
    app.world_mut().trigger(SetModel {
        entity: agent,
        model,
        kind: ChoiceKind::ForThisRun,
    });
    let earlier = message::Message::user("an earlier question");
    assert_eq!(commit_message(app.world_mut(), agent, earlier), Ok(()));
    app.update();

    let path = home.root().join("sockets").join(format!("{id}.sock"));
    let mode = std::fs::metadata(&path).map(|meta| meta.permissions().mode() & 0o777);
    assert_eq!(mode.ok(), Some(0o600), "the socket is the user's alone");

    let socket = path.clone();
    let seen = drive(&mut app, &wakes, move || {
        let mut seen = Seen::default();
        let mut clients = [Connection::open(&socket)?, Connection::open(&socket)?];
        for client in &mut clients {
            seen.backlogs.push(client.event()?);
        }
        let [first, second] = &mut clients;
        first.type_line("hello")?;
        seen.first_turn = vec![first.until_turn_end()?, second.until_turn_end()?];
        // A client that reads its backlog, and nothing after.
        let mut idle = Connection::open(&socket)?;
        idle.event()?;
        for text in ["big", "bigger"] {
            first.type_line(text)?;
            for client in [&mut *first, &mut *second] {
                let events = client.until_turn_end()?;
                let streamed = events.iter().find(|(kind, _)| kind == "text");
                seen.big_turns +=
                    usize::from(streamed.is_some_and(|(_, text)| text.len() == 800_000));
            }
        }
        std::io::Result::Ok((seen, clients, idle))
    });
    assert!(seen.is_ok(), "a client failed: {:?}", seen.as_ref().err());
    let Ok((seen, [mut first, _], _idle)) = seen else {
        return;
    };

    for backlog in &seen.backlogs {
        assert_eq!(backlog.get("type"), Some(&json!("backlog")));
        let first = backlog.pointer("/items/0/text");
        assert_eq!(first, Some(&json!("an earlier question")), "{backlog}");
    }
    let expected = [
        ("delivery", "hello"),
        ("text", "hi there"),
        ("turn_end", ""),
    ]
    .map(|(kind, text)| (kind.to_owned(), text.to_owned()));
    for turn in &seen.first_turn {
        assert_eq!(turn.as_slice(), expected.as_slice());
    }
    assert_eq!(seen.big_turns, 4, "both readers got both large replies");
    assert_eq!(
        app.world().resource::<Ended>().0.len(),
        3,
        "every turn finished"
    );
    let notices = &app.world().resource::<Collected<Notice>>().0;
    let dropped = notices
        .iter()
        .filter(|notice| notice.text.contains("fell 1 MB behind"));
    assert_eq!(
        dropped.count(),
        1,
        "only the client that never reads is dropped"
    );

    app.world_mut().write_message(AppExit::Success);
    app.update();
    assert!(!path.exists(), "the socket is removed on exit");
    let rest = (0..3).map(|_| first.event()).find_map(|event| {
        event
            .ok()
            .filter(|event| event.get("type") == Some(&json!("ended")))
    });
    assert_eq!(
        rest.and_then(|event| event.get("text").cloned()),
        Some(json!(ENDED))
    );
    std::fs::remove_dir_all(home.root()).ok();
}
