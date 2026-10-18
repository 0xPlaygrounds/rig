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

/// A server on its own thread, for one test.
struct Rigd {
    home: PathBuf,
    thread: Option<JoinHandle<bevy_app::AppExit>>,
}

impl Rigd {
    fn start(name: &str) -> Result<Rigd> {
        Rigd::at(scratch(name))
    }

    /// A server for an existing home, as a restarted one finds it.
    fn at(home: PathBuf) -> Result<Rigd> {
        let options = Options {
            idle_exit: Duration::from_millis(300),
            idle_unload: Duration::from_millis(200),
            watchdog: Duration::from_secs(1),
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
        };
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

/// The scripted worker: run by the server tests' `rigd` as its workers,
/// inert under a normal test run.
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
    let (tx, commands) = crossbeam_channel::unbounded::<WorkerCommand>();
    let options = ClientOptions {
        token: Some(token),
        autostart: Autostart::Off,
        reconnect: false,
        on_notification: Some(Arc::new(move |n: rig_protocol::Notification| {
            if n.method == WORKER_COMMAND
                && let Ok(command) = serde_json::from_value(n.params)
            {
                tx.send(command).ok();
            }
        })),
        ..ClientOptions::default()
    };
    let target = Target::Local {
        rig_home: PathBuf::from(&home),
    };
    let client = Client::connect_with(&target, "fake-worker", options)?;
    let marker = PathBuf::from(&home)
        .join("sessions")
        .join(&session.0)
        .join("crashed-once");
    let welcome = client.call::<WorkerHello>(&WorkerHello {
        rig: rig_protocol::PROTOCOL,
        session_id: session.clone(),
        build: BuildId("dev".into()),
        applied: CommandSeq(0),
        logs: Vec::new(),
    })?;
    let mut queue: std::collections::VecDeque<WorkerCommand> = welcome.replay.into();
    loop {
        let command = match queue.pop_front() {
            Some(command) => command,
            None => match commands.recv_timeout(Duration::from_secs(30)) {
                Ok(command) => command,
                Err(_) => std::process::exit(0),
            },
        };
        let Command::Send(send) = command.command else {
            continue;
        };
        let text = send.text.unwrap_or_default();
        if text == "panic-once" && !marker.exists() {
            std::fs::write(&marker, "").ok();
            std::process::exit(70);
        }
        if text == "hang" {
            std::thread::sleep(Duration::from_secs(3600));
        }
        let agent = AgentId("main".into());
        let event = |kind: &str, data: serde_json::Value| WorkerEvent {
            kind: kind.to_owned(),
            data,
            source: None,
        };
        let turn = format!("t-{}", command.command_seq.0);
        let answer = format!("echo {text}");
        let batch = vec![
            event(
                "turn.started",
                serde_json::to_value(TurnStarted {
                    agent_id: agent.clone(),
                    turn_id: turn.clone(),
                })?,
            ),
            event(
                "message",
                serde_json::to_value(Message {
                    agent_id: agent.clone(),
                    message_id: turn.clone(),
                    role: "assistant".into(),
                    text: answer.clone(),
                })?,
            ),
            event(
                "turn.ended",
                serde_json::to_value(TurnEnded {
                    agent_id: agent,
                    turn_id: turn,
                    status: TurnStatus::Answered,
                    answer: Some(answer),
                    error: None,
                    through: command.command_seq,
                })?,
            ),
        ];
        client.call::<WorkerEvents>(&WorkerEvents { events: batch })?;
        client.call::<WorkerApplied>(&WorkerApplied {
            through: command.command_seq,
        })?;
        client.call::<WorkerQuiescent>(&WorkerQuiescent { quiescent: true })?;
    }
}

#[test]
fn a_started_session_is_kept_even_if_its_worker_never_saves() -> Result {
    let rigd = Rigd::start("kept")?;
    let client = rigd.client()?;
    let session = start(&client, &rigd.home, "hi")?;
    let id: crate::harness_protocol::SessionId = session.0.parse()?;
    // A worker that exits before its first save discards an unsaved
    // session's directory, the journal in it included.
    assert!(Home::at(&rigd.home).session(&id).is_saved());
    wait(&client, &session, Duration::from_secs(20))?;
    drop(client);
    rigd.ends()
}

/// `rigd` as a process of its own: run by the signal test.
#[test]
#[ignore = "run by a_server_started_with_signals_ignored_still_exits_on_sigterm"]
fn rigd_process() {
    if std::env::var_os(env::HOME).is_some() && std::env::var_os("RIGD_PROCESS").is_some() {
        let exit = super::run();
        std::process::exit(if exit.is_success() { 0 } else { 1 });
    }
}

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
        .env("RIGD_PROCESS", "1")
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
    // Saved by its worker or not, a session with a journal comes back.
    let id: crate::harness_protocol::SessionId = session.0.parse()?;
    std::fs::remove_file(Home::at(&home).session(&id).path().join("saved"))?;

    let rigd = Rigd::at(home)?;
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
