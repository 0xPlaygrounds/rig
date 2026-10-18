use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use rig_core::test_utils::MockCompletionModel;
use rig_protocol::Notification;
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
use crate::ask::{Answer, Ask};
use rig_protocol::AskId;
use crate::harness_protocol::{Home, SessionId as HarnessId};
use crate::host::exit::ExitPlugin;

/// What the fake `rigd` heard from the worker.
#[derive(Default)]
struct Heard {
    /// Where `rigd` writes to the worker, once it connected.
    to_worker: Option<Box<dyn std::io::Write + Send>>,
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
        let Some((mut reader, writer)) = listener
            .ok()
            .and_then(|listener| listener.accept().ok())
            .and_then(|(conn, _)| conn.split().ok())
        else {
            return;
        };
        lock(&kept).to_worker = Some(Box::new(writer));
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
            let Some(writer) = heard.to_worker.as_mut() else {
                return;
            };
            if write_frame(writer, &Frame::Response(response)).is_err() {
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
        grant: None,
        cwd: PathBuf::from("/"),
        ask_timeout: Duration::from_secs(2),
    })
    .add_plugins((
        AgentPlugin,
        SessionLogPlugin,
        ExitPlugin,
        crate::clock::ClockPlugin,
        WorkerPlugin,
    ));
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

/// Sends `command` to the worker as `rigd` does.
fn command(heard: &Mutex<Heard>, command: WorkerCommand) {
    let note = Notification::new(
        methods::WORKER_COMMAND,
        serde_json::to_value(command).unwrap_or_default(),
    );
    if let Some(writer) = lock(heard).to_worker.as_mut() {
        write_frame(writer, &Frame::Notification(note)).ok();
    }
}

fn opened(heard: &Heard) -> Vec<rig_protocol::event::AskOpened> {
    heard
        .events
        .iter()
        .filter(|e| e.kind == "ask.opened")
        .filter_map(|e| serde_json::from_value(e.data.clone()).ok())
        .collect()
}

#[test]
fn an_ask_reaches_the_channel_an_answer_decides_it_and_one_nobody_answers_is_denied() {
    let home = scratch_home("worker-asks");
    let id = HarnessId::generate();
    let model = MockCompletionModel::from_stream_turns([reply("unused")]);
    let at = endpoint("asks");
    let (heard, server) = fake_rigd(&at, BTreeMap::new(), Vec::new());
    let (mut app, wakes) = worker(&home, &id, &at, &model);
    run_until(&mut app, &wakes, |world| world.contains_resource::<WorkerChannel>());
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next()
        .unwrap_or(Entity::PLACEHOLDER);
    let keep = world
        .spawn((Ask::yes_no("Keep the changes?"), ForAgent(agent)))
        .id();
    run_until(&mut app, &wakes, |_| opened(&lock(&heard)).len() == 1);
    let asked = opened(&lock(&heard));
    let ask_id = asked.first().map(|a| a.ask_id.clone()).unwrap_or(AskId(String::new()));
    assert_eq!(
        asked.first().map(|a| (a.title.as_str(), a.agent_id.0.is_empty())),
        Some(("Keep the changes?", false))
    );

    command(
        &heard,
        WorkerCommand {
            command_seq: CommandSeq(1),
            command: Command::Answer(methods::Answer {
                session_id: SessionId(id.to_string()),
                ask_id,
                option_id: Some("yes".into()),
                text: None,
                key: Key("k-1".into()),
            }),
        },
    );
    run_until(&mut app, &wakes, |world| world.get::<Outcome>(keep).is_some());
    let answer = app.world().get::<Outcome>(keep).and_then(Answer::of);
    assert!(answer.is_some_and(|answer| answer.is("yes")));

    // Nobody answers the next one: it is denied at its deadline.
    let world = app.world_mut();
    let wait = world
        .spawn((Ask::yes_no("Delete it?"), ForAgent(agent)))
        .id();
    run_until(&mut app, &wakes, |world| world.get::<Outcome>(wait).is_some());
    assert!(app.world().get::<Outcome>(wait).and_then(Answer::of).is_none());
    run_until(&mut app, &wakes, |_| {
        lock(&heard)
            .events
            .iter()
            .any(|e| e.kind == "ask.resolved" && e.data.get("by") == Some(&"timeout".into()))
    });
    drop(app);
    drop(server);
}
