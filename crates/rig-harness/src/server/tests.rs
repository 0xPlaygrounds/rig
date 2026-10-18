//! `rigd` over its real socket, with a scripted worker: this test binary
//! itself, run as [`fake_worker`]. The fake applies each `Send` by
//! answering `echo <text>`; `panic-once` makes it exit 70 the first time,
//! `hang` makes it never answer.
//!
//! Each test returns its setup failures with `?` and asserts the rest.
#![allow(clippy::panic_in_result_fn)]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use rig_client::{Autostart, Client, ClientError, Options as ClientOptions, Target};
use rig_protocol::event::{Message, TurnEnded, TurnStarted, TurnStatus};
use rig_protocol::methods::{
    Command, List, ReadEvents, Send, Start, WORKER_COMMAND, Wait, WaitStatus, WorkerApplied,
    WorkerCommand, WorkerEvent, WorkerEvents, WorkerHello, WorkerQuiescent,
};
use rig_protocol::{AgentId, BuildId, CommandSeq, Key, Seq, SessionId, codes};

use super::{Options, bind, serve, server_app};

type Result<T = (), E = Box<dyn std::error::Error>> = std::result::Result<T, E>;
use crate::harness_protocol::{Home, env};

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("rigd-{name}-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).ok();
    dir
}

/// The tests' server options: short idle times, a one-second watchdog,
/// and the scripted worker.
fn options() -> Result<Options> {
    Ok(Options {
        idle_exit: Duration::from_millis(300),
        idle_unload: Duration::from_millis(200),
        watchdog: Duration::from_secs(1),
        crash_limit: 3,
        worker: Some((
            std::env::current_exe()?,
            [
                "server::tests::fake_worker",
                "--exact",
                "--ignored",
                "--nocapture",
            ]
            .map(Into::into)
            .to_vec(),
        )),
    })
}

/// A server on its own thread, for one test.
struct Rigd {
    home: PathBuf,
    thread: Option<JoinHandle<bevy_app::AppExit>>,
}

impl Rigd {
    fn start(name: &str) -> Result<Rigd> {
        Rigd::at(scratch(name), options()?)
    }

    /// A server for an existing home, as a restarted one finds it.
    fn at(home: PathBuf, options: Options) -> Result<Rigd> {
        let at = home.clone();
        let (ready, started) = crossbeam_channel::bounded(1);
        let thread = std::thread::spawn(move || {
            let home = Home::at(&at);
            let Ok((_lock, token, listener)) = bind(&home) else {
                return bevy_app::AppExit::error();
            };
            let mut app = server_app();
            serve(&mut app, home, token, listener, options);
            ready.send(()).ok();
            app.run()
        });
        started.recv_timeout(Duration::from_secs(10))?;
        Ok(Rigd {
            home,
            thread: Some(thread),
        })
    }

    fn client(&self) -> Result<Client> {
        let options = ClientOptions {
            autostart: Autostart::Off,
            give_up: Duration::from_secs(5),
            ..ClientOptions::default()
        };
        Client::connect_with(
            &Target::Local {
                rig_home: self.home.clone(),
            },
            "test",
            options,
        )
        .map_err(Into::into)
    }

    /// Waits for the server to exit on its own once idle.
    fn exits(mut self) -> Result {
        let thread = self.thread.take().ok_or("no server thread")?;
        let by = Instant::now() + Duration::from_secs(20);
        while !thread.is_finished() {
            if Instant::now() > by {
                return Err("rigd did not exit when idle".into());
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        let exit = thread.join().map_err(|_| "the server thread panicked")?;
        assert_eq!(exit, bevy_app::AppExit::Success);
        Ok(())
    }

    /// [`exits`](Rigd::exits), then removes its home.
    fn ends(self) -> Result {
        let home = self.home.clone();
        self.exits()?;
        std::fs::remove_dir_all(&home).ok();
        Ok(())
    }
}

fn key() -> Key {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    Key(format!(
        "k-{}",
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ))
}

fn start(client: &Client, cwd: &Path, prompt: &str) -> Result<SessionId> {
    let started = client.call::<Start>(&Start {
        cwd: cwd.to_owned(),
        prompt: Some(prompt.to_owned()),
        name: None,
        host: None,
        build: None,
        grant: None,
        policy: None,
        budget: None,
        context: None,
        unload: None,
        view: false,
        key: key(),
    })?;
    Ok(started.session_id)
}

fn send(session: &SessionId, text: &str, key: Key) -> Send {
    Send {
        session_id: session.clone(),
        text: Some(text.to_owned()),
        content: None,
        mode: None,
        key,
    }
}

fn wait(
    client: &Client,
    session: &SessionId,
    timeout: Duration,
) -> Result<rig_protocol::methods::WaitResult> {
    Ok(client.call::<Wait>(&Wait {
        session_id: session.clone(),
        after_command: None,
        timeout_ms: Some(timeout.as_millis() as u64),
    })?)
}

fn events(client: &Client, session: &SessionId) -> Result<Vec<rig_protocol::EventRecord>> {
    Ok(client
        .call::<ReadEvents>(&ReadEvents {
            session_id: session.clone(),
            after_seq: Seq(0),
            limit: None,
        })?
        .events)
}

#[test]
fn start_send_wait_and_read_events() -> Result {
    let rigd = Rigd::start("basic")?;
    let client = rigd.client()?;
    let session = start(&client, &rigd.home, "hi")?;

    let first = wait(&client, &session, Duration::from_secs(20))?;
    assert_eq!(first.status, WaitStatus::Answered);
    assert_eq!(first.answer.as_deref(), Some("echo hi"));

    let again = key();
    let sent = client.call::<Send>(&send(&session, "again", again.clone()))?;
    // A repeated key with the same command answers what it did.
    let repeat = client.call::<Send>(&send(&session, "again", again.clone()))?;
    assert_eq!(repeat.command_seq, sent.command_seq);
    let conflict = client.call::<Send>(&send(&session, "other", again));
    assert!(
        matches!(&conflict, Err(ClientError::Rpc(e)) if e.code == codes::KEY_CONFLICT),
        "{conflict:?}"
    );
    let second = wait(&client, &session, Duration::from_secs(20))?;
    assert_eq!(second.answer.as_deref(), Some("echo again"));
    assert_eq!(second.command_seq, sent.command_seq);

    let records = events(&client, &session)?;
    let seqs: Vec<u64> = records.iter().map(|r| r.seq.0).collect();
    assert_eq!(seqs, (1..=seqs.len() as u64).collect::<Vec<_>>());
    let kinds: Vec<&str> = records.iter().map(|r| r.kind.as_str()).collect();
    assert_eq!(kinds.first(), Some(&"session.started"));
    assert_eq!(kinds.iter().filter(|k| **k == "turn.ended").count(), 2);
    assert!(kinds.contains(&"worker.ready"));

    let listed = client.call::<List>(&List::default())?;
    assert_eq!(listed.len(), 1);
    drop(client);
    rigd.ends()
}

#[test]
fn a_crashed_worker_restarts_and_its_neighbour_never_notices() -> Result {
    let rigd = Rigd::start("crash")?;
    let client = rigd.client()?;
    let crashing = start(&client, &rigd.home, "panic-once")?;
    let calm = start(&client, &rigd.home, "calm")?;

    let calm_answer = wait(&client, &calm, Duration::from_secs(20))?;
    assert_eq!(calm_answer.answer.as_deref(), Some("echo calm"));
    let after_crash = wait(&client, &crashing, Duration::from_secs(20))?;
    assert_eq!(after_crash.status, WaitStatus::Answered);
    assert_eq!(after_crash.answer.as_deref(), Some("echo panic-once"));
    let kinds: Vec<String> = events(&client, &crashing)?
        .into_iter()
        .map(|r| r.kind)
        .collect();
    assert_eq!(
        kinds.iter().filter(|k| *k == "worker.exited").count(),
        1,
        "{kinds:?}"
    );
    assert_eq!(
        kinds.iter().filter(|k| *k == "turn.ended").count(),
        1,
        "{kinds:?}"
    );
    drop(client);
    rigd.ends()
}

/// One record of the scripted worker's saved log: the events a command
/// produced, saved with the command it applied in the same line, as the
/// real worker saves `Applied` beside the `Incoming`.
#[derive(serde::Serialize, serde::Deserialize)]
struct Saved {
    applied: CommandSeq,
    events: Vec<WorkerEvent>,
}

const FAKE_LOG: &str = "fake";

/// The scripted worker: run by the server tests' `rigd` as its workers,
/// inert under a normal test run. It keeps a saved log
/// (`sessions/<id>/fake.jsonl`), says hello with what it applied, and
/// re-exports what `rigd` has not journaled, like the real worker.
#[test]
#[ignore = "run by the server tests as their worker"]
fn fake_worker() -> Result {
    let (Ok(token), Ok(session), Ok(home)) = (
        std::env::var(env::WORKER_TOKEN),
        std::env::var(env::SESSION),
        std::env::var(env::HOME),
    ) else {
        return Ok(());
    };
    let session = SessionId(session);
    // The real worker handles SIGHUP (an exit it may take its time over),
    // so an orphaned, stopped worker isn't simply hung up on.
    #[cfg(unix)]
    signal_hook::flag::register(
        signal_hook::consts::SIGHUP,
        Arc::new(std::sync::atomic::AtomicBool::new(false)),
    )?;
    let (tx, commands) = crossbeam_channel::unbounded::<WorkerCommand>();
    // Plugin calls are answered at once, as the real worker's frame loop
    // answers them while a tool call runs; a hung loop answers none.
    let (ping_tx, pings) = crossbeam_channel::unbounded::<u64>();
    let options = ClientOptions {
        token: Some(token),
        autostart: Autostart::Off,
        reconnect: false,
        on_notification: Some(Arc::new(move |n: rig_protocol::Notification| {
            if n.method == WORKER_COMMAND
                && let Ok(command) = serde_json::from_value::<WorkerCommand>(n.params)
            {
                match command.command {
                    Command::Plugin { reply, .. } => {
                        ping_tx.send(reply).ok();
                    }
                    _ => {
                        tx.send(command).ok();
                    }
                }
            }
        })),
        ..ClientOptions::default()
    };
    let target = Target::Local {
        rig_home: PathBuf::from(&home),
    };
    let client = Arc::new(Client::connect_with(&target, "fake-worker", options)?);
    let hung = Arc::new(std::sync::atomic::AtomicBool::new(false));
    {
        let (client, hung) = (client.clone(), hung.clone());
        std::thread::spawn(move || {
            for reply in pings {
                if hung.load(std::sync::atomic::Ordering::Acquire) {
                    continue;
                }
                let answer = rig_protocol::methods::WorkerReply {
                    reply,
                    outcome: rig_protocol::Outcome::Result(serde_json::json!({})),
                };
                client.call::<rig_protocol::methods::WorkerReply>(&answer).ok();
            }
        });
    }
    let dir = PathBuf::from(&home).join("sessions").join(&session.0);
    let marker = dir.join("crashed-once");
    let log_path = dir.join("fake.jsonl");
    let saved: Vec<Saved> = std::fs::read_to_string(&log_path)
        .unwrap_or_default()
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();
    let applied = saved.last().map_or(CommandSeq(0), |s| s.applied);
    let welcome = client.call::<WorkerHello>(&WorkerHello {
        rig: rig_protocol::PROTOCOL,
        session_id: session.clone(),
        build: BuildId("dev".into()),
        applied,
        logs: vec![rig_protocol::methods::LogLen {
            log: FAKE_LOG.into(),
            len: saved.len() as u64,
        }],
    })?;
    // Re-export what was saved but never journaled.
    let known = welcome.known_through.get(FAKE_LOG).copied().unwrap_or(0);
    for (index, record) in (1..).zip(&saved) {
        if index > known {
            client.call::<WorkerEvents>(&WorkerEvents {
                events: record.events.clone(),
            })?;
        }
    }
    let mut index = saved.len() as u64;
    let mut log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)?;
    let mut queue: std::collections::VecDeque<WorkerCommand> = welcome.replay.into();
    loop {
        let command = match queue.pop_front() {
            Some(command) => command,
            None => match commands.recv_timeout(Duration::from_secs(30)) {
                Ok(command) => command,
                Err(_) => std::process::exit(0),
            },
        };
        let send = match command.command {
            Command::Send(send) => send,
            // The real worker saves, then turns quiescent for rigd to
            // restart it on the new build.
            // A stop or a reload applies at once; nothing runs to stop.
            Command::Reload(_) | Command::Stop(_) | Command::Answer(_) => {
                client.call::<WorkerApplied>(&WorkerApplied {
                    through: command.command_seq,
                })?;
                client.call::<WorkerQuiescent>(&WorkerQuiescent { quiescent: true })?;
                continue;
            }
            _ => continue,
        };
        let text = send.text.unwrap_or_default();
        if text == "panic-once" && !marker.exists() {
            std::fs::write(&marker, "").ok();
            std::process::exit(70);
        }
        if text == "hang" {
            hung.store(true, std::sync::atomic::Ordering::Release);
            std::thread::sleep(Duration::from_secs(3600));
        }
        // A long tool call that prints nothing: the loop stays alive.
        if text == "silent" {
            std::thread::sleep(Duration::from_secs(3));
        }
        let agent = AgentId("main".into());
        let turn = format!("t-{}", command.command_seq.0);
        let live = |kind: &str, data: serde_json::Value| WorkerEvent {
            kind: kind.to_owned(),
            data,
            source: None,
        };
        client.call::<WorkerEvents>(&WorkerEvents {
            events: vec![live(
                "turn.started",
                serde_json::to_value(TurnStarted {
                    agent_id: agent.clone(),
                    turn_id: turn.clone(),
                })?,
            )],
        })?;
        // `flood N`: N live deltas first, in frames of 500.
        if let Some(n) = text
            .strip_prefix("flood ")
            .and_then(|n| n.parse::<usize>().ok())
        {
            let delta = live(
                "message.delta",
                serde_json::json!({"agentId": "main", "messageId": turn, "text": "x".repeat(64)}),
            );
            for _ in 0..n.div_ceil(500) {
                client.call::<WorkerEvents>(&WorkerEvents {
                    events: vec![delta.clone(); 500],
                })?;
            }
        }
        if text.starts_with("slow") {
            std::thread::sleep(Duration::from_millis(300));
        }
        index += 1;
        let source = Some(rig_protocol::SourceKey {
            log: FAKE_LOG.into(),
            index,
        });
        let answer = format!("echo {text}");
        // One saved record exports two events under one key.
        let durable = vec![
            WorkerEvent {
                kind: "message".into(),
                data: serde_json::to_value(Message {
                    agent_id: agent.clone(),
                    message_id: turn.clone(),
                    role: "assistant".into(),
                    text: answer.clone(),
                })?,
                source: source.clone(),
            },
            WorkerEvent {
                kind: "turn.ended".into(),
                data: serde_json::to_value(TurnEnded {
                    agent_id: agent,
                    turn_id: turn,
                    status: TurnStatus::Answered,
                    answer: Some(answer),
                    error: None,
                    through: command.command_seq,
                })?,
                source,
            },
        ];
        let record = Saved {
            applied: command.command_seq,
            events: durable.clone(),
        };
        let mut line = serde_json::to_vec(&record)?;
        line.push(b'\n');
        std::io::Write::write_all(&mut log, &line)?;
        client.call::<WorkerEvents>(&WorkerEvents { events: durable })?;
        client.call::<WorkerApplied>(&WorkerApplied {
            through: command.command_seq,
        })?;
        client.call::<WorkerQuiescent>(&WorkerQuiescent { quiescent: true })?;
    }
}

/// `rigd` as a process of its own, with the tests' options: what the
/// signal and restart tests start, and what a client autostarts.
#[test]
#[ignore = "run by the server tests as their rigd process"]
fn rigd_process() -> Result {
    if std::env::var(env::ROLE).as_deref() != Ok("server") {
        return Ok(());
    }
    let home = Home::from_env();
    let (lock, token, listener) = match bind(&home) {
        Ok(bound) => bound,
        Err(super::StartError::Running) => return Ok(()),
        Err(super::StartError::Io(e)) => return Err(e.into()),
    };
    let mut app = server_app();
    serve(&mut app, home, token, listener, options()?);
    let exit = app.run();
    drop(lock);
    std::process::exit(if exit.is_success() { 0 } else { 1 });
}

const RIGD_PROCESS: [&str; 3] = ["server::tests::rigd_process", "--exact", "--ignored"];

#[test]
fn a_server_started_with_signals_ignored_still_exits_on_sigterm() -> Result {
    let home = scratch("signals");
    let exe = std::env::current_exe()?;
    // What a driver's `nohup` or `&` gives its children: SIGINT and
    // SIGTERM ignored, inherited across exec.
    let mut child = std::process::Command::new("sh")
        .arg("-c")
        .arg("trap '' INT TERM; exec \"$0\" server::tests::rigd_process --exact --ignored")
        .arg(&exe)
        .env(env::HOME, &home)
        .env(env::ROLE, "server")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()?;
    let socket = rig_protocol::Endpoint::for_home(&home).path;
    let by = Instant::now() + Duration::from_secs(20);
    while !socket.exists() && Instant::now() < by {
        std::thread::sleep(Duration::from_millis(20));
    }
    std::process::Command::new("kill")
        .args(["-TERM", &child.id().to_string()])
        .status()?;
    let by = Instant::now() + Duration::from_secs(5);
    let mut exited = None;
    while exited.is_none() && Instant::now() < by {
        exited = child.try_wait()?;
        std::thread::sleep(Duration::from_millis(20));
    }
    if exited.is_none() {
        child.kill().ok();
        child.wait().ok();
    }
    std::fs::remove_dir_all(&home).ok();
    assert!(exited.is_some(), "SIGTERM did not stop rigd");
    Ok(())
}

#[test]
fn a_restarted_server_knows_every_journaled_session() -> Result {
    let rigd = Rigd::start("restore")?;
    let home = rigd.home.clone();
    let client = rigd.client()?;
    let session = start(&client, &home, "hi")?;
    wait(&client, &session, Duration::from_secs(20))?;
    drop(client);
    rigd.exits()?;
    // The scripted worker never saves: a session with a journal comes back
    // all the same.
    let id: crate::harness_protocol::SessionId = session.0.parse()?;
    assert!(!Home::at(&home).session(&id).is_saved());

    let rigd = Rigd::at(home, options()?)?;
    let client = rigd.client()?;
    let sent = client.call::<Send>(&send(&session, "back", key()))?;
    let back = wait(&client, &session, Duration::from_secs(20))?;
    assert_eq!(back.answer.as_deref(), Some("echo back"));
    assert_eq!(back.command_seq, sent.command_seq);
    drop(client);
    rigd.ends()
}

#[test]
fn a_hung_worker_is_killed_and_faults_its_session_alone() -> Result {
    let rigd = Rigd::start("hang")?;
    let client = rigd.client()?;
    let hung = start(&client, &rigd.home, "hang")?;
    let calm = start(&client, &rigd.home, "calm")?;

    let started = Instant::now();
    let answer = wait(&client, &calm, Duration::from_secs(20))?;
    assert_eq!(answer.answer.as_deref(), Some("echo calm"));
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "{:?}",
        started.elapsed()
    );

    // Killed by the watchdog each time it hangs again, until it faults.
    let faulted = wait(&client, &hung, Duration::from_secs(20))?;
    assert_eq!(faulted.status, WaitStatus::Failed);
    let kinds: Vec<String> = events(&client, &hung)?
        .into_iter()
        .map(|r| r.kind)
        .collect();
    assert_eq!(
        kinds.iter().filter(|k| *k == "worker.exited").count(),
        3,
        "{kinds:?}"
    );
    assert!(kinds.iter().any(|k| k == "session.faulted"), "{kinds:?}");
    drop(client);
    rigd.ends()
}

/// The pid of the worker `session` started last.
fn worker_pid(client: &Client, session: &SessionId) -> Result<Option<u32>> {
    Ok(events(client, session)?
        .iter()
        .rev()
        .find(|r| r.kind == "worker.ready")
        .and_then(|r| r.data.get("pid")?.as_u64())
        .and_then(|pid| u32::try_from(pid).ok()))
}

fn kill9(pid: u32) -> Result {
    std::process::Command::new("kill")
        .args(["-KILL", &pid.to_string()])
        .stderr(std::process::Stdio::null())
        .status()?;
    Ok(())
}

/// Seqs run 1, 2, 3…, and each message id has one final `message`.
fn exactly_once(records: &[rig_protocol::EventRecord]) {
    let seqs: Vec<u64> = records.iter().map(|r| r.seq.0).collect();
    assert_eq!(seqs, (1..=seqs.len() as u64).collect::<Vec<_>>());
    let mut ids: Vec<String> = records
        .iter()
        .filter(|r| r.kind == "message")
        .filter_map(|r| r.data.get("messageId")?.as_str().map(str::to_owned))
        .collect();
    let all = ids.len();
    ids.sort();
    ids.dedup();
    assert_eq!(ids.len(), all, "a message was journaled twice");
}

#[test]
fn killing_the_worker_anywhere_loses_and_repeats_nothing() -> Result {
    // Every kill is a crash: no limit faults the session here.
    let rigd = Rigd::at(
        scratch("kill9"),
        Options {
            crash_limit: usize::MAX,
            ..options()?
        },
    )?;
    let client = rigd.client()?;
    let session = start(&client, &rigd.home, "slow 0")?;
    wait(&client, &session, Duration::from_secs(20))?;
    // A fixed seed: the same kill points every run.
    let mut seed: u64 = 0x5eed;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) % 450
    };
    let rounds = 20;
    for round in 1..=rounds {
        let text = format!("slow {round}");
        let sent = client.call::<Send>(&send(&session, &text, key()))?;
        std::thread::sleep(Duration::from_millis(next()));
        if let Some(pid) = worker_pid(&client, &session)? {
            kill9(pid)?;
        }
        let done = wait(&client, &session, Duration::from_secs(20))?;
        assert_eq!(done.status, WaitStatus::Answered, "round {round}: {done:?}");
        assert_eq!(done.answer, Some(format!("echo {text}")), "round {round}");
        assert!(done.command_seq >= sent.command_seq);
    }
    let records = events(&client, &session)?;
    exactly_once(&records);
    let ended = records.iter().filter(|r| r.kind == "turn.ended").count();
    assert_eq!(ended, rounds + 1);
    drop(client);
    rigd.ends()
}

#[test]
fn a_subscriber_that_never_reads_slows_nobody_else() -> Result {
    let rigd = Rigd::start("slow-reader")?;
    let client = rigd.client()?;
    let session = start(&client, &rigd.home, "flood 2000")?;
    wait(&client, &session, Duration::from_secs(20))?;
    let round = |client: &Client| -> Result<Duration> {
        let started = Instant::now();
        client.call::<Send>(&send(&session, "flood 2000", key()))?;
        wait(client, &session, Duration::from_secs(20))?;
        Ok(started.elapsed())
    };
    let solo = (0..5).map(|_| round(&client)).collect::<Result<Vec<_>>>()?;

    // A client that subscribes and then never reads a byte.
    let endpoint = rig_protocol::Endpoint::for_home(&rigd.home);
    let (reader, mut writer) = rig_protocol::Conn::connect(&endpoint)?.split()?;
    let hello = serde_json::json!({
        "token": rig_protocol::read_token(&rigd.home)?,
        "client": {"name": "stuck", "version": "0"},
        "rig": {"min": 1, "max": 1},
    });
    let subscribe = serde_json::json!({"sessionId": session.0, "afterSeq": 0, "follow": true});
    for (id, method, params) in [
        (1, "rig/hello", hello),
        (2, "rig/events/subscribe", subscribe),
    ] {
        let request = rig_protocol::Request::new(rig_protocol::RequestId::Num(id), method, params);
        rig_protocol::write_frame(&mut writer, &rig_protocol::Frame::Request(request))?;
    }
    let stuck = (0..5).map(|_| round(&client)).collect::<Result<Vec<_>>>()?;

    let worst = |times: &[Duration]| times.iter().max().copied().unwrap_or_default();
    // Twice the solo worst, and a little room for a busy machine.
    let bound = 2 * worst(&solo) + Duration::from_millis(250);
    assert!(
        worst(&stuck) < bound,
        "solo {solo:?}, beside a stuck reader {stuck:?}"
    );
    drop((reader, writer));
    drop(client);
    rigd.ends()
}

#[test]
fn an_unloaded_worker_comes_back_for_the_next_command() -> Result {
    let rigd = Rigd::start("unload")?;
    let client = rigd.client()?;
    let session = start(&client, &rigd.home, "hi")?;
    wait(&client, &session, Duration::from_secs(20))?;
    let by = Instant::now() + Duration::from_secs(10);
    while client
        .call::<List>(&List::default())?
        .iter()
        .any(|s| s.worker)
    {
        assert!(Instant::now() < by, "the idle worker was never unloaded");
        std::thread::sleep(Duration::from_millis(50));
    }
    client.call::<Send>(&send(&session, "back", key()))?;
    let back = wait(&client, &session, Duration::from_secs(20))?;
    assert_eq!(back.answer.as_deref(), Some("echo back"));
    let records = events(&client, &session)?;
    let ready = records.iter().filter(|r| r.kind == "worker.ready").count();
    assert_eq!(ready, 2);
    exactly_once(&records);
    drop(client);
    rigd.ends()
}

#[test]
fn a_killed_server_comes_back_and_its_clients_miss_nothing() -> Result {
    let home = scratch("rigd-kill9");
    let target = Target::Local {
        rig_home: home.clone(),
    };
    let options = ClientOptions {
        autostart: Autostart::Program {
            program: std::env::current_exe()?,
            args: RIGD_PROCESS.map(Into::into).to_vec(),
        },
        give_up: Duration::from_secs(20),
        ..ClientOptions::default()
    };
    let client = Client::connect_with(&target, "test", options)?;
    let first = client.hello().map(|h| h.server.pid).ok_or("no hello")?;
    let session = start(&client, &home, "slow mid-turn")?;
    let mut stream = client.subscribe(&session, Seq(0))?;
    // Mid-turn: the turn started, its answer not yet saved.
    loop {
        let record = stream
            .next_timeout(Duration::from_secs(20))
            .ok_or("no turn started")??;
        if record.kind == "turn.started" {
            break;
        }
    }
    kill9(first)?;
    let done = wait(&client, &session, Duration::from_secs(30))?;
    assert_eq!(done.status, WaitStatus::Answered, "{done:?}");
    assert_eq!(done.answer.as_deref(), Some("echo slow mid-turn"));
    let second = client.hello().map(|h| h.server.pid).ok_or("no hello")?;
    assert_ne!(first, second, "the client reconnected to a new server");
    // The new server's worker ran the turn to its end, never killed.
    let exits: Vec<_> = events(&client, &session)?
        .into_iter()
        .filter(|r| r.kind == "worker.exited")
        .collect();
    assert!(exits.is_empty(), "{exits:?}");

    // The stream the client kept across the restart has every record once.
    let journal = events(&client, &session)?;
    let mut seen: Vec<rig_protocol::EventRecord> = Vec::new();
    while seen.last().is_none_or(|r| r.kind != "turn.ended") {
        let record = stream
            .next_timeout(Duration::from_secs(20))
            .ok_or("the stream stopped")??;
        seen.push(record);
    }
    let after: Vec<u64> = journal
        .iter()
        .map(|r| r.seq.0)
        .filter(|seq| {
            *seq > journal
                .iter()
                .position(|r| r.kind == "turn.started")
                .unwrap_or(0) as u64
                + 1
        })
        .take(seen.len())
        .collect();
    let got: Vec<u64> = seen.iter().map(|r| r.seq.0).collect();
    assert_eq!(got, after, "the resumed stream skipped or repeated records");
    exactly_once(&journal);
    drop(stream);
    drop(client);
    // The new server exits once idle, leaving nothing behind.
    let by = Instant::now() + Duration::from_secs(20);
    while rig_protocol::Endpoint::for_home(&home).path.exists() {
        assert!(Instant::now() < by, "the restarted server never exited");
        std::thread::sleep(Duration::from_millis(50));
    }
    std::fs::remove_dir_all(&home).ok();
    Ok(())
}

#[test]
fn a_wait_after_a_stop_waits_for_the_message_not_the_stop() -> Result {
    let rigd = Rigd::start("wait-stop")?;
    let client = rigd.client()?;
    let session = start(&client, &rigd.home, "hi")?;
    wait(&client, &session, Duration::from_secs(20))?;
    client.call::<rig_protocol::methods::Stop>(&rig_protocol::methods::Stop {
        session_id: session.clone(),
        scope: rig_protocol::methods::StopScope::Turn,
        key: key(),
    })?;
    // The stop changes the turn the message started; that turn's end
    // answers the wait.
    let after = wait(&client, &session, Duration::from_secs(3))?;
    assert_eq!(after.status, WaitStatus::Answered, "{after:?}");
    drop(client);
    rigd.ends()
}

#[test]
fn a_reload_restarts_the_session_on_its_new_build() -> Result {
    let rigd = Rigd::start("reload")?;
    let client = rigd.client()?;
    let session = start(&client, &rigd.home, "hi")?;
    wait(&client, &session, Duration::from_secs(20))?;
    client.call::<rig_protocol::methods::Reload>(&rig_protocol::methods::Reload {
        session_id: session.clone(),
        build: BuildId("next".into()),
        key: key(),
    })?;
    let by = Instant::now() + Duration::from_secs(20);
    let done = loop {
        let records = events(&client, &session)?;
        if let Some(done) = records.iter().find(|r| r.kind == "reload.done") {
            break done.data.clone();
        }
        assert!(Instant::now() < by, "no reload.done: {records:?}");
        std::thread::sleep(Duration::from_millis(50));
    };
    assert_eq!(done.get("build").and_then(|b| b.as_str()), Some("next"));
    client.call::<Send>(&send(&session, "after", key()))?;
    let after = wait(&client, &session, Duration::from_secs(20))?;
    assert_eq!(after.answer.as_deref(), Some("echo after"));
    let kinds: Vec<String> = events(&client, &session)?
        .into_iter()
        .map(|r| r.kind)
        .collect();
    assert_eq!(
        kinds.iter().filter(|k| *k == "worker.ready").count(),
        2,
        "{kinds:?}"
    );
    assert!(!kinds.iter().any(|k| k == "reload.failed"), "{kinds:?}");
    drop(client);
    rigd.ends()
}

/// Whether `pid` is a live process (not gone, not a zombie).
#[cfg(target_os = "linux")]
fn alive(pid: u32) -> bool {
    std::fs::read_to_string(format!("/proc/{pid}/stat"))
        .ok()
        .and_then(|stat| {
            let state = stat
                .rsplit(')')
                .next()?
                .split_whitespace()
                .next()?
                .to_owned();
            Some(state != "Z" && state != "X")
        })
        .unwrap_or(false)
}

#[cfg(target_os = "linux")]
#[test]
fn no_worker_outlives_its_server_even_a_stopped_one() -> Result {
    let home = scratch("orphan");
    let mut rigd = std::process::Command::new(std::env::current_exe()?)
        .args(RIGD_PROCESS)
        .env(env::HOME, &home)
        .env(env::ROLE, "server")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()?;
    let socket = rig_protocol::Endpoint::for_home(&home).path;
    let by = Instant::now() + Duration::from_secs(20);
    while !socket.exists() && Instant::now() < by {
        std::thread::sleep(Duration::from_millis(20));
    }
    let client = Client::connect_with(
        &Target::Local {
            rig_home: home.clone(),
        },
        "test",
        ClientOptions {
            autostart: Autostart::Off,
            reconnect: false,
            ..ClientOptions::default()
        },
    )?;
    // Busy in a tool call: it won't read the end of its connection soon.
    let session = start(&client, &home, "hang")?;
    let by = Instant::now() + Duration::from_secs(20);
    let pid = loop {
        if let Some(pid) = worker_pid(&client, &session)? {
            break pid;
        }
        assert!(Instant::now() < by, "no worker started");
        std::thread::sleep(Duration::from_millis(20));
    };
    // A stopped worker never reads the end of its connection.
    std::process::Command::new("kill")
        .args(["-STOP", &pid.to_string()])
        .status()?;
    rigd.kill()?;
    rigd.wait()?;
    let by = Instant::now() + Duration::from_secs(1);
    while alive(pid) && Instant::now() < by {
        std::thread::sleep(Duration::from_millis(20));
    }
    let outlived = alive(pid);
    if outlived {
        kill9(pid)?;
    }
    std::fs::remove_dir_all(&home).ok();
    assert!(!outlived, "worker {pid} outlived its server");
    Ok(())
}

#[test]
fn a_silent_but_live_worker_is_never_killed_and_a_stopped_one_is() -> Result {
    // The watchdog is a second here: a three-second silent call outlives it.
    let rigd = Rigd::start("silent")?;
    let client = rigd.client()?;
    let quiet = start(&client, &rigd.home, "silent")?;
    let answer = wait(&client, &quiet, Duration::from_secs(20))?;
    assert_eq!(answer.answer.as_deref(), Some("echo silent"));
    let kinds: Vec<String> = events(&client, &quiet)?.into_iter().map(|r| r.kind).collect();
    assert!(!kinds.iter().any(|k| k == "worker.exited"), "{kinds:?}");

    // Stopped mid-call, the same worker is dead to the watchdog.
    client.call::<Send>(&send(&quiet, "silent", key()))?;
    std::thread::sleep(Duration::from_millis(300));
    let pid = worker_pid(&client, &quiet)?.ok_or("no worker")?;
    std::process::Command::new("kill").args(["-STOP", &pid.to_string()]).status()?;
    let again = wait(&client, &quiet, Duration::from_secs(20))?;
    assert_eq!(again.answer.as_deref(), Some("echo silent"));
    let kinds: Vec<String> = events(&client, &quiet)?.into_iter().map(|r| r.kind).collect();
    assert_eq!(kinds.iter().filter(|k| *k == "worker.exited").count(), 1, "{kinds:?}");
    drop(client);
    rigd.ends()
}

#[test]
fn a_session_saved_before_the_server_is_listed_and_driven() -> Result {
    let home = scratch("pre-rigd");
    // What a rig before the server left: a saved session, no journal.
    let id: crate::harness_protocol::SessionId = "1700000000-7".parse()?;
    let dir = Home::at(&home).session(&id);
    std::fs::create_dir_all(dir.path())?;
    dir.mark_saved()?;
    std::fs::write(dir.directory(), home.as_os_str().as_encoded_bytes())?;
    dir.set_name(Some("old"))?;

    let rigd = Rigd::at(home.clone(), options()?)?;
    let client = rigd.client()?;
    let listed = client.call::<List>(&List {
        tree: false,
        all: true,
    })?;
    let old = listed
        .iter()
        .find(|s| s.session_id.0 == id.as_str())
        .ok_or("the saved session is not listed")?;
    assert_eq!(old.name.as_deref(), Some("old"));
    assert_eq!(old.cwd, home);

    let session = SessionId(id.as_str().to_owned());
    client.call::<Send>(&send(&session, "again", key()))?;
    let answer = wait(&client, &session, Duration::from_secs(20))?;
    assert_eq!(answer.answer.as_deref(), Some("echo again"));
    let records = events(&client, &session)?;
    assert_eq!(records.first().map(|r| r.kind.as_str()), Some("session.started"));
    exactly_once(&records);
    drop(client);
    rigd.ends()
}
