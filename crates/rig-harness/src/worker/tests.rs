use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use rig_core::test_utils::MockCompletionModel;
use rig_ecs::AgentPlugin;
use rig_ecs::agent::Agent;
use rig_ecs::session_log::SessionLogPlugin;
use rig_harness_test_support::{app_in, connect, reply, run_until, scratch_home};
use rig_protocol::methods::{
    Command, HelloResult, Limits, Send as SendCommand, ServerInfo, WorkerCommand, WorkerWelcome,
};
use rig_protocol::{
    ClientId, CommandSeq, Endpoint, Frame, Key, Listener, Response,
    read_frame, write_frame,
};

use super::*;
use crate::harness_protocol::{Home, SessionId as HarnessId};
use crate::host::exit::ExitPlugin;

/// What the fake `rigd` heard from the worker.
#[derive(Default)]
struct Heard {
    events: Vec<WorkerEvent>,
    applied: Vec<u64>,
    hello: Option<WorkerHello>,
}

/// A `rigd` that answers one worker: its welcome says `known` and replays
/// `replay`; every other request is answered `{}`, and what came is kept.
fn fake_rigd(
    endpoint: &Endpoint,
    known: BTreeMap<String, u64>,
    replay: Vec<WorkerCommand>,
) -> (Arc<Mutex<Heard>>, JoinHandle<()>) {
    let listener = Listener::bind(endpoint);
    assert!(listener.is_ok(), "bind: {:?}", listener.as_ref().err());
    let heard = Arc::new(Mutex::new(Heard::default()));
    let kept = heard.clone();
    let server = std::thread::spawn(move || {
        let Some((mut reader, mut writer)) = listener
            .ok()
            .and_then(|listener| listener.accept().ok())
            .and_then(|(conn, _)| conn.split().ok())
        else {
            return;
        };
        let mut line = Vec::new();
        let mut welcome = Some(WorkerWelcome {
            known_through: known,
            replay,
        });
        while let Ok(Some(frame)) = read_frame(&mut reader, &mut line) {
            let Frame::Request(request) = frame else {
                continue;
            };
            let mut heard = lock(&kept);
            let result = match request.method.as_str() {
                "rig/hello" => serde_json::to_value(HelloResult {
                    rig: 1,
                    server: ServerInfo {
                        version: "test".into(),
                        pid: 0,
                    },
                    client_id: ClientId("c-1".into()),
                    methods: Vec::new(),
                    events: Vec::new(),
                    limits: Limits { frame: 1 << 20 },
                }),
                "worker/hello" => {
                    heard.hello = serde_json::from_value(request.params).ok();
                    serde_json::to_value(welcome.take())
                }
                "worker/events" => {
                    let batch: Option<WorkerEvents> = serde_json::from_value(request.params).ok();
                    heard.events.extend(batch.into_iter().flat_map(|b| b.events));
                    Ok(serde_json::json!({}))
                }
                "worker/applied" => {
                    let applied: Option<WorkerApplied> = serde_json::from_value(request.params).ok();
                    heard.applied.extend(applied.map(|a| a.through.0));
                    Ok(serde_json::json!({}))
                }
                _ => Ok(serde_json::json!({})),
            };
            let response = Response::new(request.id, Ok(result.unwrap_or_default()));
            if write_frame(&mut writer, &Frame::Response(response)).is_err() {
                return;
            }
        }
    });
    (heard, server)
}

fn lock(heard: &Mutex<Heard>) -> std::sync::MutexGuard<'_, Heard> {
    heard.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// A socket path short enough for `sun_path`, under the test's temp dir.
fn endpoint(name: &str) -> Endpoint {
    let dir: PathBuf = std::env::temp_dir().join(format!("rw-{}-{name}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    Endpoint {
        path: dir.join("rig.sock"),
    }
}

fn send(seq: u64, text: &str) -> WorkerCommand {
    WorkerCommand {
        command_seq: CommandSeq(seq),
        command: Command::Send(SendCommand {
            session_id: SessionId("s".into()),
            text: Some(text.into()),
            content: None,
            mode: None,
            key: Key(format!("k-{seq}")),
        }),
    }
}

/// A worker app on session `id` of `home`, reaching `endpoint`, its model
/// `model`.
fn worker(
    home: &Home,
    id: &HarnessId,
    endpoint: &Endpoint,
    model: &MockCompletionModel,
) -> (App, std::sync::mpsc::Receiver<()>) {
    let (mut app, wakes) = app_in(home, id);
    app.insert_resource(WorkerConfig {
        endpoint: Endpoint {
            path: endpoint.path.clone(),
        },
        token: "t".into(),
        session: SessionId(id.to_string()),
        build: BuildId("b".into()),
    })
    .add_plugins((AgentPlugin, SessionLogPlugin, ExitPlugin, WorkerPlugin));
    let model = model.clone();
    app.add_systems(
        PostStartup,
        move |agents: Query<Entity, With<Agent>>, mut commands: Commands| {
            for agent in &agents {
                if let Some(connection) = connect(&model) {
                    commands.entity(agent).insert(connection);
                }
            }
        },
    );
    app.finish();
    app.cleanup();
    (app, wakes)
}

fn kinds(heard: &Heard) -> Vec<&str> {
    heard.events.iter().map(|e| e.kind.as_str()).collect()
}

fn turn_ended(heard: &Heard) -> Vec<rig_protocol::event::TurnEnded> {
    heard
        .events
        .iter()
        .filter(|e| e.kind == "turn.ended")
        .filter_map(|e| serde_json::from_value(e.data.clone()).ok())
        .collect()
}

#[test]
fn a_command_runs_one_turn_and_its_events_are_exported_once() {
    let home = scratch_home("worker-one-turn");
    let id = HarnessId::generate();
    let model = MockCompletionModel::from_stream_turns([reply("hello")]);

    // The first run: rigd replays command 1, the worker answers it.
    let at = endpoint("one");
    let (heard, server) = fake_rigd(&at, BTreeMap::new(), vec![send(1, "hi")]);
    let (mut app, wakes) = worker(&home, &id, &at, &model);
    run_until(&mut app, &wakes, |_| {
        !turn_ended(&lock(&heard)).is_empty()
            && lock(&heard).applied.contains(&1)
    });
    drop(app);
    drop(server);
    let first = lock(&heard);
    let seen = kinds(&first);
    assert!(seen.contains(&"turn.started"), "{seen:?}");
    assert!(seen.contains(&"conversation.change"), "{seen:?}");
    let ended = turn_ended(&first);
    let ends: Vec<_> = ended
        .iter()
        .map(|e| (e.answer.as_deref(), e.through))
        .collect();
    assert_eq!(ends, [(Some("hello"), CommandSeq(1))]);
    let messages: Vec<String> = first
        .events
        .iter()
        .filter(|e| e.kind == "message")
        .filter_map(|e| Some(e.data.get("text")?.as_str()?.to_owned()))
        .collect();
    assert_eq!(messages, ["hi", "hello"]);
    // Every durable event names its record.
    assert!(
        first
            .events
            .iter()
            .filter(|e| e.kind != "message.delta")
            .all(|e| e.source.is_some())
    );
    let known: BTreeMap<String, u64> = first
        .events
        .iter()
        .filter_map(|e| e.source.as_ref())
        .fold(BTreeMap::new(), |mut known, source| {
            let at = known.entry(source.log.clone()).or_insert(0);
            *at = (*at).max(source.index);
            known
        });
    let turn_log = ended
        .first()
        .map(|e| e.agent_id.0.clone())
        .unwrap_or_default();
    drop(first);

    // Restarted, rigd replays command 1 again and has journaled all but
    // the turn's end: the command is not applied twice, and only the end
    // is exported again.
    let mut short = known.clone();
    let ended_at = lock(&heard)
        .events
        .iter()
        .find(|e| e.kind == "turn.ended")
        .and_then(|e| e.source.clone())
        .map(|s| s.index)
        .unwrap_or_default();
    short.insert(turn_log, ended_at.saturating_sub(1));
    let again = endpoint("two");
    let (heard, server) = fake_rigd(&again, short, vec![send(1, "hi")]);
    let (mut app, wakes) = worker(&home, &id, &again, &model);
    run_until(&mut app, &wakes, |_| {
        lock(&heard).hello.is_some()
    });
    for _ in 0..20 {
        app.update();
        wakes.recv_timeout(std::time::Duration::from_millis(20)).ok();
    }
    drop(app);
    drop(server);
    let second = lock(&heard);
    assert_eq!(second.hello.as_ref().map(|h| h.applied), Some(CommandSeq(1)));
    let kinds = kinds(&second);
    assert!(!kinds.contains(&"turn.started"), "{kinds:?}");
    assert_eq!(turn_ended(&second).len(), 1, "{kinds:?}");
}
