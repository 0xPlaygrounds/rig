//! The client against a fake server that speaks just enough of the wire:
//! hello with a token, an idempotent `rig/session/send`, a journal per
//! session with subscribe/replay/follow, and a switch that drops every
//! connection, as a crashing `rigd` would.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use rig_protocol::methods::{
    Disposition, Empty, EventNotice, HelloResult, Limits, List, Send, Sent, ServerInfo, Subscribe,
    Subscribed, SubscriptionId,
};
use rig_protocol::{
    ClientId, CommandSeq, Conn, Endpoint, EventRecord, Frame, Key, Listener, Method, Notification,
    Request, Response, RpcError, Seq, SessionId, codes, read_frame, write_frame,
};
use serde_json::{Value, json};

use std::io::Write as _;

use super::*;

const TOKEN: &str = "fake-token";

fn home(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("rcl-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).ok();
    dir
}

type Writer = Arc<Mutex<Box<dyn std::io::Write + core::marker::Send>>>;

#[derive(Default)]
struct World {
    journals: HashMap<SessionId, Vec<EventRecord>>,
    /// (connection, subscription, session) of every following subscription.
    follows: Vec<(u64, SubscriptionId, SessionId)>,
    conns: HashMap<u64, (Writer, Conn)>,
    sent: HashMap<String, Sent>,
    applied: u64,
}

/// A fake `rigd` on `home`'s endpoint.
#[derive(Clone)]
struct Fake {
    world: Arc<Mutex<World>>,
    next: Arc<AtomicU64>,
    /// Drop the connection on the first `send`, after applying it.
    drop_on_send: Arc<AtomicBool>,
    /// Replay one event before `afterSeq`, which the client must drop.
    sloppy_replay: Arc<AtomicBool>,
    clients: Arc<AtomicU64>,
    /// Close this many connections on their hello, unanswered, as a
    /// restarting server can.
    reset_hellos: Arc<AtomicU64>,
    /// The wire version it speaks (frames and hello): a newer server's.
    speaks: Arc<AtomicU64>,
}

impl Fake {
    fn start(home: &Path) -> Fake {
        let run = home.join("run");
        std::fs::create_dir_all(&run).ok();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&run, std::fs::Permissions::from_mode(0o700)).ok();
        }
        std::fs::write(run.join("token"), TOKEN).ok();
        let listener = Listener::bind(&Endpoint::for_home(home)).expect("bind");
        let fake = Fake {
            world: Arc::default(),
            next: Arc::new(AtomicU64::new(1)),
            drop_on_send: Arc::default(),
            sloppy_replay: Arc::default(),
            clients: Arc::default(),
            reset_hellos: Arc::default(),
            speaks: Arc::new(AtomicU64::new(1)),
        };
        let server = fake.clone();
        std::thread::spawn(move || {
            while let Ok((conn, _peer)) = listener.accept() {
                let server = server.clone();
                std::thread::spawn(move || server.serve(conn));
            }
        });
        fake
    }

    /// Every connection closes, as when `rigd` dies.
    fn drop_all(&self) {
        let mut world = self.world.lock().expect("world");
        for (_, (_, conn)) in world.conns.drain() {
            conn.shutdown().ok();
        }
        world.follows.clear();
    }

    /// Appends an event to `session`'s journal and sends it to followers.
    fn append(&self, session: &str, kind: &str) -> Seq {
        let session = SessionId(session.into());
        let mut world = self.world.lock().expect("world");
        let journal = world.journals.entry(session.clone()).or_default();
        let seq = Seq(journal.len() as u64 + 1);
        let record = EventRecord {
            session: session.clone(),
            seq,
            at_ms: 0,
            kind: kind.into(),
            data: json!({}),
            source: None,
        };
        journal.push(record.clone());
        let follows = world.follows.clone();
        for (conn, subscription, followed) in follows {
            if followed == session
                && let Some((writer, _)) = world.conns.get(&conn)
            {
                notify(writer, subscription, record.clone());
            }
        }
        seq
    }

    fn serve(&self, conn: Conn) {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        let closer = conn.try_clone().expect("clone");
        let (mut reader, writer) = conn.split().expect("split");
        let writer: Writer = Arc::new(Mutex::new(Box::new(writer)));
        self.world
            .lock()
            .expect("world")
            .conns
            .insert(id, (Arc::clone(&writer), closer));
        let mut line = Vec::new();
        let mut hello = false;
        while let Ok(Some(frame)) = read_frame(&mut reader, &mut line) {
            let Frame::Request(request) = frame else {
                continue;
            };
            if !hello
                && self
                    .reset_hellos
                    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_sub(1))
                    .is_ok()
            {
                break;
            }
            if !hello {
                let token = request.params.get("token").and_then(Value::as_str);
                if request.method != "rig/hello" || token != Some(TOKEN) {
                    reply(
                        &writer,
                        request,
                        Err(RpcError::new(codes::UNAUTHORIZED, "bad token")),
                    );
                    break;
                }
                hello = true;
                self.clients.fetch_add(1, Ordering::Relaxed);
                let speaks = self.speaks.load(Ordering::Relaxed);
                if speaks != 1 {
                    // A newer server: it negotiated a version the client
                    // doesn't speak and writes every frame at it.
                    let line = json!({"jsonrpc": "2.0", "rig": speaks, "id": request.id, "result": {"rig": speaks}});
                    let mut writer = writer.lock().expect("writer");
                    writeln!(writer, "{line}").ok();
                    writer.flush().ok();
                    continue;
                }
                let result = HelloResult {
                    rig: 1,
                    server: ServerInfo {
                        version: "fake".into(),
                        pid: std::process::id(),
                    },
                    client_id: ClientId(format!("c-{id}")),
                    methods: Vec::new(),
                    events: Vec::new(),
                    limits: Limits { frame: 1 << 20 },
                };
                reply(&writer, request, Ok(json!(result)));
                continue;
            }
            let speaks = self.speaks.load(Ordering::Relaxed);
            if speaks != 1 {
                let line = json!({"jsonrpc": "2.0", "rig": speaks, "id": request.id, "result": []});
                let mut writer = writer.lock().expect("writer");
                writeln!(writer, "{line}").ok();
                writer.flush().ok();
                continue;
            }
            match request.method.as_str() {
                m if m == List::NAME => reply(&writer, request, Ok(json!([]))),
                m if m == Send::NAME => {
                    let Ok(send) = serde_json::from_value::<Send>(request.params.clone()) else {
                        reply(
                            &writer,
                            request,
                            Err(RpcError::new(codes::INVALID_PARAMS, "")),
                        );
                        continue;
                    };
                    let mut world = self.world.lock().expect("world");
                    let sent = match world.sent.get(&send.key.0) {
                        Some(sent) => sent.clone(),
                        None => {
                            world.applied += 1;
                            let sent = Sent {
                                command_seq: CommandSeq(world.applied),
                                disposition: Disposition::Started,
                            };
                            world.sent.insert(send.key.0.clone(), sent.clone());
                            sent
                        }
                    };
                    drop(world);
                    if self.drop_on_send.swap(false, Ordering::Relaxed) {
                        self.drop_all();
                        break;
                    }
                    reply(&writer, request, Ok(json!(sent)));
                }
                m if m == Subscribe::NAME => {
                    let Ok(sub) = serde_json::from_value::<Subscribe>(request.params.clone())
                    else {
                        continue;
                    };
                    let subscription = SubscriptionId(self.next.fetch_add(1, Ordering::Relaxed));
                    let mut world = self.world.lock().expect("world");
                    let journal = world
                        .journals
                        .get(&sub.session_id)
                        .cloned()
                        .unwrap_or_default();
                    let from = match self.sloppy_replay.load(Ordering::Relaxed) {
                        true => sub.after_seq.0.saturating_sub(1),
                        false => sub.after_seq.0,
                    };
                    let next_seq = Seq(journal.len() as u64 + 1);
                    reply(
                        &writer,
                        request,
                        Ok(json!(Subscribed {
                            subscription,
                            next_seq
                        })),
                    );
                    for record in journal.into_iter().filter(|r| r.seq.0 > from) {
                        notify(&writer, subscription, record);
                    }
                    if sub.follow {
                        world.follows.push((id, subscription, sub.session_id));
                    }
                }
                // A plugin method with no key: applied, then the connection drops
                // before the answer when `drop_on_send` is set.
                "x/test/bump" => {
                    self.world.lock().expect("world").applied += 1;
                    if self.drop_on_send.swap(false, Ordering::Relaxed) {
                        self.drop_all();
                        break;
                    }
                    reply(&writer, request, Ok(json!(Empty {})));
                }
                _ => reply(&writer, request, Ok(json!(Empty {}))),
            }
        }
        self.world.lock().expect("world").conns.remove(&id);
    }
}

fn reply(writer: &Writer, request: Request, outcome: Result<Value, RpcError>) {
    let mut writer = writer.lock().expect("writer");
    write_frame(
        &mut *writer,
        &Frame::Response(Response::new(request.id, outcome)),
    )
    .ok();
}

fn notify(writer: &Writer, subscription: SubscriptionId, record: EventRecord) {
    let notice = EventNotice {
        subscription,
        record,
    };
    let mut writer = writer.lock().expect("writer");
    let frame = Frame::Notification(Notification::new(methods::EVENT, json!(notice)));
    write_frame(&mut *writer, &frame).ok();
}

fn connect(home: &Path) -> Client {
    let options = Options {
        autostart: Autostart::Off,
        give_up: Duration::from_secs(5),
        ..Options::default()
    };
    Client::connect_with(
        &Target::Local {
            rig_home: home.to_owned(),
        },
        "test",
        options,
    )
    .expect("connect")
}

fn seqs(events: &mut Events, n: usize) -> Vec<u64> {
    (0..n)
        .map(|_| {
            events
                .next_timeout(Duration::from_secs(5))
                .expect("an event in time")
                .expect("an event")
                .seq
                .0
        })
        .collect()
}

#[test]
fn a_client_says_hello_and_calls() {
    let home = home("hello");
    let _fake = Fake::start(&home);
    let client = connect(&home);
    assert_eq!(
        client.hello().map(|h| h.server.version),
        Some("fake".into())
    );
    let sessions = client.call::<List>(&List::default()).expect("list");
    assert!(sessions.is_empty());
}

#[test]
fn a_wrong_token_is_refused_and_not_retried() {
    let home = home("token");
    let _fake = Fake::start(&home);
    let options = Options {
        token: Some("wrong".into()),
        autostart: Autostart::Off,
        ..Options::default()
    };
    let target = Target::Local { rig_home: home };
    let error = Client::connect_with(&target, "test", options)
        .err()
        .expect("refused");
    assert!(
        matches!(&error, ClientError::Rpc(e) if e.code == codes::UNAUTHORIZED),
        "{error:?}"
    );
    assert_eq!(error.exit(), Exit::NotAllowed);
}

#[test]
fn no_server_and_no_autostart_says_what_to_run() {
    let home = home("absent");
    let options = Options {
        autostart: Autostart::Off,
        ..Options::default()
    };
    let target = Target::Local { rig_home: home };
    let error = Client::connect_with(&target, "test", options)
        .err()
        .expect("unavailable");
    assert_eq!(error.exit(), Exit::Unavailable);
    assert!(
        error.next().is_some_and(|next| next.contains("rig serve")),
        "{error:?}"
    );
}

#[test]
fn a_stream_resumes_after_a_lost_connection_without_loss_or_repeat() {
    let home = home("resume");
    let fake = Fake::start(&home);
    fake.sloppy_replay.store(true, Ordering::Relaxed);
    let client = connect(&home);
    for _ in 0..3 {
        fake.append("s-1", "notice");
    }
    let mut events = client
        .subscribe(&SessionId("s-1".into()), Seq(1))
        .expect("subscribe");
    assert_eq!(seqs(&mut events, 2), [2, 3]);
    fake.drop_all();
    // Written while the client is away: the resubscribe replays them.
    fake.append("s-1", "notice");
    fake.append("s-1", "notice");
    assert_eq!(seqs(&mut events, 2), [4, 5]);
    fake.append("s-1", "notice");
    assert_eq!(seqs(&mut events, 1), [6]);
    assert!(events.next_timeout(Duration::from_millis(100)).is_none());
    assert_eq!(events.last_seq(), Seq(6));
    assert!(fake.clients.load(Ordering::Relaxed) >= 2, "it reconnected");
}

#[test]
fn two_streams_on_one_session_each_get_their_own_replay() {
    let home = home("two");
    let fake = Fake::start(&home);
    let client = connect(&home);
    for _ in 0..4 {
        fake.append("s-1", "notice");
    }
    let session = SessionId("s-1".into());
    let mut late = client.subscribe(&session, Seq(3)).expect("late");
    assert_eq!(seqs(&mut late, 1), [4]);
    fake.append("s-1", "notice");
    let mut early = client.subscribe(&session, Seq(0)).expect("early");
    assert_eq!(seqs(&mut early, 5), [1, 2, 3, 4, 5]);
    assert_eq!(seqs(&mut late, 1), [5]);
}

#[test]
fn a_call_lost_with_its_connection_is_resent_and_applied_once() {
    let home = home("resend");
    let fake = Fake::start(&home);
    fake.drop_on_send.store(true, Ordering::Relaxed);
    let client = connect(&home);
    let send = Send {
        session_id: SessionId("s-1".into()),
        text: Some("hi".into()),
        content: None,
        mode: None,
        key: Key("k-1".into()),
    };
    let sent = client.call::<Send>(&send).expect("sent after a reconnect");
    assert_eq!(sent.command_seq, CommandSeq(1));
    assert_eq!(fake.world.lock().expect("world").applied, 1);
}

#[test]
fn a_dropped_connection_without_reconnect_fails_and_says_so() {
    let home = home("noreconnect");
    let fake = Fake::start(&home);
    let options = Options {
        autostart: Autostart::Off,
        reconnect: false,
        ..Options::default()
    };
    let target = Target::Local { rig_home: home };
    let client = Client::connect_with(&target, "worker", options).expect("connect");
    fake.drop_all();
    let deadline = Instant::now() + Duration::from_secs(5);
    let error = loop {
        match client.call::<List>(&List::default()) {
            Err(error) => break error,
            Ok(_) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(10)),
            Ok(_) => panic!("the call kept working"),
        }
    };
    assert_eq!(error.exit(), Exit::Unavailable);
}

/// Not a test of its own: the fake `rigd` process autostart runs. Under a
/// plain test run (no `RIG_ROLE`) it returns at once.
#[test]
fn fake_server_process() {
    if std::env::var_os("RIG_ROLE").is_none_or(|role| role != "server") {
        return;
    }
    let Some(home) = std::env::var_os("RIG_HOME").map(PathBuf::from) else {
        return;
    };
    let run = home.join("run");
    let lock = std::fs::File::options()
        .create(true)
        .write(true)
        .truncate(false)
        .open(run.join("server.lock"))
        .expect("lock file");
    if lock.try_lock().is_err() {
        // Another server won the start race.
        return;
    }
    let fake = Fake::start(&home);
    // Exits once no client has connected for a while, as `rigd` does.
    let mut seen = 0;
    let mut quiet = Instant::now();
    loop {
        std::thread::sleep(Duration::from_millis(50));
        let now = fake.clients.load(Ordering::Relaxed);
        let open = !fake.world.lock().expect("world").conns.is_empty();
        if now != seen || open {
            seen = now;
            quiet = Instant::now();
        }
        if quiet.elapsed() > Duration::from_secs(2) {
            return;
        }
    }
}

#[test]
fn a_missing_server_is_started_once_for_racing_clients() {
    let home = home("autostart");
    let exe = std::env::current_exe().expect("exe");
    let options = Options {
        autostart: Autostart::Program {
            program: exe,
            args: ["--exact", "tests::fake_server_process", "--nocapture"]
                .map(OsString::from)
                .to_vec(),
        },
        ..Options::default()
    };
    let target = Target::Local {
        rig_home: home.clone(),
    };
    let racers: Vec<_> = (0..3)
        .map(|_| {
            let (target, options) = (target.clone(), options.clone());
            std::thread::spawn(move || {
                let client = Client::connect_with(&target, "racer", options).expect("connect");
                client.call::<List>(&List::default()).expect("list");
                client.hello().map(|h| h.server.pid)
            })
        })
        .collect();
    let pids: Vec<_> = racers
        .into_iter()
        .map(|r| r.join().expect("racer"))
        .collect();
    let first = pids.first().copied().flatten().expect("a pid");
    assert_ne!(first, std::process::id(), "a separate server process");
    assert!(pids.iter().all(|pid| *pid == Some(first)), "{pids:?}");
    // The server exits on its own once idle; don't leave it behind.
    std::process::Command::new("kill")
        .arg(first.to_string())
        .status()
        .ok();
}

#[test]
fn the_client_has_no_bevy() {
    let output = std::process::Command::new(env!("CARGO"))
        .args([
            "tree",
            "--offline",
            "-p",
            "rig-client",
            "-e",
            "normal",
            "--prefix",
            "none",
        ])
        .output()
        .expect("cargo tree");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let tree = String::from_utf8_lossy(&output.stdout);
    assert!(!tree.contains("bevy"), "{tree}");
}

#[test]
fn a_stream_that_does_not_follow_ends_at_the_journal_end() {
    let home = home("nofollow");
    let fake = Fake::start(&home);
    let client = connect(&home);
    for _ in 0..3 {
        fake.append("s-1", "notice");
    }
    let session = SessionId("s-1".into());
    let events = client
        .subscribe_with(&session, Seq(1), false)
        .expect("subscribe");
    let empty = client
        .subscribe_with(&session, Seq(3), false)
        .expect("subscribe");
    // On a thread: a stream that never ends must fail the test, not hang it.
    let (tx, rx) = crossbeam_channel::bounded(1);
    std::thread::spawn(move || {
        let seqs: Vec<u64> = events.map(|e| e.expect("event").seq.0).collect();
        tx.send((seqs, empty.count())).ok();
    });
    assert_eq!(
        rx.recv_timeout(Duration::from_secs(5)).ok(),
        Some((vec![2, 3], 0))
    );
}

#[test]
fn a_reconnect_survives_hellos_the_restarting_server_resets() {
    let home = home("resethello");
    let fake = Fake::start(&home);
    let client = connect(&home);
    fake.append("s-1", "notice");
    let mut events = client
        .subscribe(&SessionId("s-1".into()), Seq(0))
        .expect("subscribe");
    assert_eq!(seqs(&mut events, 1), [1]);
    fake.reset_hellos.store(2, Ordering::Relaxed);
    fake.drop_all();
    fake.append("s-1", "notice");
    assert_eq!(seqs(&mut events, 1), [2]);
    assert_eq!(
        fake.reset_hellos.load(Ordering::Relaxed),
        0,
        "both resets were met"
    );
    client.call::<List>(&List::default()).expect("calls go on");
}

#[test]
fn a_call_without_a_key_is_not_resent_after_a_lost_connection() {
    let home = home("nokey");
    let fake = Fake::start(&home);
    fake.drop_on_send.store(true, Ordering::Relaxed);
    let client = connect(&home);
    let error = client
        .call_raw("x/test/bump", json!({"sessionId": "s-1"}))
        .expect_err("its outcome is unknown");
    assert!(
        matches!(&error, ClientError::Unavailable { next } if next.contains("may have run")),
        "{error:?}"
    );
    assert_eq!(fake.world.lock().expect("world").applied, 1, "applied once");
    // The client goes on: the next call reconnects.
    client
        .call::<List>(&List::default())
        .expect("list after a reconnect");
}

#[test]
fn a_newer_server_is_refused_naming_this_client_as_the_side_to_update() {
    let home = home("newer");
    let fake = Fake::start(&home);
    fake.speaks.store(2, Ordering::Relaxed);
    let options = Options {
        autostart: Autostart::Off,
        ..Options::default()
    };
    let target = Target::Local { rig_home: home };
    let error = Client::connect_with(&target, "test", options)
        .err()
        .expect("refused");
    assert!(
        matches!(&error, ClientError::Version { update: Side::Client, supported } if supported == &[1]),
        "{error:?}"
    );
    assert!(error.next().is_some_and(|next| next.contains("newer rig")));
}

#[test]
fn a_frame_of_another_version_after_hello_fails_the_call_instead_of_hanging() {
    let home = home("laterversion");
    let fake = Fake::start(&home);
    let client = connect(&home);
    fake.speaks.store(2, Ordering::Relaxed);
    let (tx, rx) = crossbeam_channel::bounded(1);
    std::thread::spawn(move || {
        tx.send(client.call::<List>(&List::default()).err()).ok();
    });
    let error = rx
        .recv_timeout(Duration::from_secs(5))
        .expect("an answer in time");
    assert!(
        matches!(
            &error,
            Some(ClientError::Version {
                update: Side::Client,
                ..
            })
        ),
        "{error:?}"
    );
}
