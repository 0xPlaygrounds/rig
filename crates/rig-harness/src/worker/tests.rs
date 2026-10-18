use std::collections::BTreeMap;
use std::fs;
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
    replies: Vec<WorkerReply>,
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
                "worker/reply" => {
                    heard.replies.extend(serde_json::from_value(request.params).ok());
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

/// The socket of a rig home short enough for `sun_path`, under the
/// test's temp dir: `<home>/run/rig.sock`, as the client finds it.
fn endpoint(name: &str) -> Endpoint {
    let dir: PathBuf = std::env::temp_dir().join(format!("rw-{}-{name}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    Endpoint::for_home(&dir)
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
    worker_with(home, id, endpoint, model, None)
}

/// [`worker`] of a session with `grant`.
fn worker_with(
    home: &Home,
    id: &HarnessId,
    endpoint: &Endpoint,
    model: &MockCompletionModel,
    grant: Option<Grant>,
) -> (App, std::sync::mpsc::Receiver<()>) {
    let (mut app, wakes) = app_in(home, id);
    app.insert_resource(WorkerConfig {
        endpoint: Endpoint {
            path: endpoint.path.clone(),
        },
        token: "t".into(),
        session: SessionId(id.to_string()),
        build: BuildId("b".into()),
        grant,
        cwd: PathBuf::from("/"),
        ask_timeout: Duration::from_secs(2),
    })
    .add_plugins((
        AgentPlugin,
        SessionLogPlugin,
        ExitPlugin,
        crate::clock::ClockPlugin,
        crate::commands::CommandPlugin,
        crate::ask::AskPlugin,
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
    // Every durable event names its record (deltas and the context are live).
    assert!(
        first
            .events
            .iter()
            .filter(|e| e.kind != "message.delta" && e.kind != "context")
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
    // The turn the ask held answers the command that answered it.
    assert_eq!(app.world().get::<Through>(agent), Some(&Through(1)));

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

#[derive(serde::Serialize, serde::Deserialize)]
struct Shout {
    text: String,
}

impl Method for Shout {
    const NAME: &'static str = "x/test/shout";
    type Params = Shout;
    type Result = String;
}

#[test]
fn a_plugin_method_runs_in_the_worker_and_its_answer_goes_back() {
    use crate::channel::{Call, ChannelAppExt};
    let home = scratch_home("worker-plugin");
    let id = HarnessId::generate();
    let model = MockCompletionModel::from_stream_turns([reply("unused")]);
    let at = endpoint("plugin");
    let call = |method: &str, reply: u64| WorkerCommand {
        command_seq: CommandSeq(reply),
        command: Command::Plugin {
            method: method.into(),
            params: serde_json::json!({"text": "hi"}),
            reply,
            owner: false,
        },
    };
    let replay = vec![call(Shout::NAME, 1), call("x/test/none", 2)];
    let (heard, server) = fake_rigd(&at, BTreeMap::new(), replay);
    let (mut app, wakes) = worker(&home, &id, &at, &model);
    app.add_method::<Shout, _, _>(|call: In<Call<Shout>>| -> Result<String, RpcError> {
        Ok(call.params.text.to_uppercase())
    });
    run_until(&mut app, &wakes, |_| lock(&heard).replies.len() == 2);
    let replies: Vec<(u64, bool)> = lock(&heard)
        .replies
        .iter()
        .map(|r| (r.reply, matches!(&r.outcome, rig_protocol::Outcome::Result(v) if v == "HI")))
        .collect();
    assert_eq!(replies, [(1, true), (2, false)]);
    drop(app);
    drop(server);
}

#[test]
fn a_spent_budget_starts_no_model_call() {
    let home = scratch_home("worker-budget");
    let id = HarnessId::generate();
    let model = MockCompletionModel::from_stream_turns([reply("too much")]);
    let at = endpoint("budget");
    let spent = WorkerCommand {
        command_seq: CommandSeq(1),
        command: Command::Plugin {
            method: SetBudget::NAME.into(),
            params: serde_json::json!({"exhausted": true}),
            reply: 1,
            owner: false,
        },
    };
    let (heard, server) = fake_rigd(&at, BTreeMap::new(), vec![spent, send(2, "hi")]);
    let (mut app, wakes) = worker(&home, &id, &at, &model);
    run_until(&mut app, &wakes, |_| !turn_ended(&lock(&heard)).is_empty());
    let ends: Vec<_> = turn_ended(&lock(&heard))
        .iter()
        .map(|e| (e.status, e.answer.clone()))
        .collect();
    assert_eq!(ends.len(), 1);
    assert!(ends.iter().all(|(status, answer)| *status != rig_protocol::event::TurnStatus::Answered
        && answer.is_none()), "{ends:?}");
    assert!(model.requests().is_empty());
    drop(app);
    drop(server);
}

#[test]
fn each_turn_ends_through_the_command_that_started_it() {
    let home = scratch_home("worker-through");
    let id = HarnessId::generate();
    let turns = 20;
    let model = MockCompletionModel::from_stream_turns((1..=turns).map(|n| reply(&format!("{n}"))));
    let at = endpoint("through");
    let (heard, server) = fake_rigd(&at, BTreeMap::new(), Vec::new());
    let (mut app, wakes) = worker(&home, &id, &at, &model);
    run_until(&mut app, &wakes, |world| world.contains_resource::<WorkerChannel>());
    for seq in 1..=turns {
        command(&heard, send(seq, "ping"));
        let ended = usize::try_from(seq).unwrap_or(usize::MAX);
        run_until(&mut app, &wakes, |_| turn_ended(&lock(&heard)).len() == ended);
    }
    let through: Vec<u64> = turn_ended(&lock(&heard)).iter().map(|e| e.through.0).collect();
    assert_eq!(through, (1..=turns).collect::<Vec<_>>());
    drop(app);
    drop(server);
}

#[test]
fn a_reply_that_calls_a_tool_exports_the_call_and_its_result() {
    use rig_core::conversation::{Change, ChangeRecord, Entry};
    use rig_core::message::{AssistantContent, CallId, Message, ToolName};
    let Ok(tool) = ToolName::new("look") else {
        return;
    };
    let call = Message::from(AssistantContent::tool_call(
        "c-1",
        tool.clone(),
        serde_json::json!({"path": "a"}),
    ));
    let result = Message::tool_result(CallId::from_wire("c-1"), tool, "seen");
    let push = |message: Message| ChangeRecord {
        change: Change::Push {
            entries: vec![Entry::new(message)],
        },
        by: None,
        at: None,
    };
    let mut exporter = export::Exporter::default();
    let mut out = Vec::new();
    for (seq, message) in [(1, call), (2, result)] {
        let record = rig_memory::LogRecord::Change(push(message));
        exporter.record("agent", seq, &record, true, &mut out);
    }
    let calls: Vec<(String, serde_json::Value)> = out
        .iter()
        .filter(|e| e.kind.starts_with("call."))
        .map(|e| (e.kind.clone(), e.data.clone()))
        .collect();
    assert_eq!(
        calls,
        [
            (
                "call.started".to_owned(),
                serde_json::json!({"agentId": "agent", "callId": r#"{"provider":"c-1"}"#, "tool": "look", "args": {"path": "a"}})
            ),
            (
                "call.ended".to_owned(),
                serde_json::json!({"callId": r#"{"provider":"c-1"}"#, "ok": true, "result": "seen"})
            ),
        ]
    );
}

#[cfg(unix)]
#[test]
fn a_worker_whose_log_stops_saving_exits_to_be_restarted() {
    use std::os::unix::fs::PermissionsExt;
    let home = scratch_home("worker-readonly");
    let id = HarnessId::generate();
    let model = MockCompletionModel::from_stream_turns([reply("one"), reply("two")]);
    let at = endpoint("readonly");
    let (heard, server) = fake_rigd(&at, BTreeMap::new(), vec![send(1, "hi")]);
    let (mut app, wakes) = worker(&home, &id, &at, &model);
    run_until(&mut app, &wakes, |_| turn_ended(&lock(&heard)).len() == 1);
    let logs = home.session(&id).path().join("logs");
    let mode = |path: &std::path::Path, mode| {
        fs::set_permissions(path, fs::Permissions::from_mode(mode)).ok();
    };
    for entry in fs::read_dir(&logs).into_iter().flatten().flatten() {
        mode(&entry.path(), 0o400);
    }
    mode(&logs, 0o500);
    command(&heard, send(2, "again"));
    let started = std::time::Instant::now();
    while app.should_exit().is_none() && started.elapsed() < Duration::from_secs(10) {
        app.update();
        wakes.recv_timeout(Duration::from_millis(20)).ok();
    }
    mode(&logs, 0o700);
    for entry in fs::read_dir(&logs).into_iter().flatten().flatten() {
        mode(&entry.path(), 0o600);
    }
    let exit = app.should_exit();
    assert_eq!(
        exit.map(|exit| exit.is_success()),
        Some(false),
        "a worker that can no longer save exits for rigd to restart it"
    );
    drop(app);
    drop(server);
}

#[test]
fn the_context_the_model_is_given_is_journaled_once_per_change() {
    use rig_ecs::agent::SystemPrompt;
    let home = scratch_home("worker-context");
    let id = HarnessId::generate();
    let model = MockCompletionModel::from_stream_turns([reply("1"), reply("2"), reply("3")]);
    let at = endpoint("context");
    let (heard, server) = fake_rigd(&at, BTreeMap::new(), Vec::new());
    let (mut app, wakes) = worker(&home, &id, &at, &model);
    run_until(&mut app, &wakes, |world| world.contains_resource::<WorkerChannel>());
    let prompt = |app: &mut App, text: &str| {
        let world = app.world_mut();
        let agents: Vec<Entity> = world
            .query_filtered::<Entity, With<Agent>>()
            .iter(world)
            .collect();
        for agent in agents {
            world.entity_mut(agent).insert(SystemPrompt(text.to_owned()));
        }
    };
    prompt(&mut app, "You are A.");
    for seq in 1..=3 {
        if seq == 3 {
            prompt(&mut app, "You are B.");
        }
        command(&heard, send(seq, "ping"));
        let ended = usize::try_from(seq).unwrap_or(usize::MAX);
        run_until(&mut app, &wakes, |_| turn_ended(&lock(&heard)).len() == ended);
    }
    let systems: Vec<String> = lock(&heard)
        .events
        .iter()
        .filter(|e| e.kind == "context")
        .filter_map(|e| Some(e.data.get("system")?.as_str()?.to_owned()))
        .collect();
    assert_eq!(systems.len(), 2, "{systems:?}");
    assert!(systems.first().is_some_and(|s| s.contains("You are A.")), "{systems:?}");
    assert!(systems.last().is_some_and(|s| s.contains("You are B.")), "{systems:?}");
    drop(app);
    drop(server);
}

#[test]
fn a_session_granted_some_tools_lets_its_agents_call_only_those() {
    use rig_ecs::agent::ToolAccess;
    let home = scratch_home("worker-tools");
    let id = HarnessId::generate();
    let model = MockCompletionModel::from_stream_turns([reply("unused")]);
    let at = endpoint("tools");
    let (_heard, server) = fake_rigd(&at, BTreeMap::new(), Vec::new());
    let grant = Grant {
        tools: Some(vec!["read".into()]),
        net: true,
        ..Grant::default()
    };
    let (mut app, wakes) = worker_with(&home, &id, &at, &model, Some(grant));
    run_until(&mut app, &wakes, |world| world.contains_resource::<WorkerChannel>());
    let world = app.world_mut();
    let access: Vec<bool> = world
        .query_filtered::<&ToolAccess, With<Agent>>()
        .iter(world)
        .map(|access| access.allows("read") && !access.allows("shell"))
        .collect();
    assert_eq!(access, [true]);
    drop(app);
    drop(server);
}

#[test]
fn each_model_call_exports_what_it_used_once() {
    use rig_core::test_utils::MockStreamEvent;
    let home = scratch_home("worker-usage");
    let id = HarnessId::generate();
    let used = |n| vec![MockStreamEvent::text("ok"), MockStreamEvent::final_response_with_total_tokens(n)];
    let model = MockCompletionModel::from_stream_turns([used(40), used(2)]);
    let at = endpoint("usage");
    let (heard, server) = fake_rigd(&at, BTreeMap::new(), vec![send(1, "a")]);
    let (mut app, wakes) = worker(&home, &id, &at, &model);
    run_until(&mut app, &wakes, |_| turn_ended(&lock(&heard)).len() == 1);
    command(&heard, send(2, "b"));
    run_until(&mut app, &wakes, |_| turn_ended(&lock(&heard)).len() == 2);
    for _ in 0..5 {
        app.update();
    }
    let tokens: Vec<u64> = lock(&heard)
        .events
        .iter()
        .filter(|e| e.kind == "usage")
        .filter_map(|e| e.data.get("tokens")?.as_u64())
        .collect();
    assert_eq!(tokens, [40, 2]);
    assert!(lock(&heard).events.iter().filter(|e| e.kind == "usage").all(|e| e.source.is_some()));
    drop(app);
    drop(server);
}

/// Calls of each tool the approval test's tools ran.
#[derive(Resource, Default)]
struct Ran(Vec<String>);

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct NoArgs {}

#[test]
fn a_session_granted_ask_shell_asks_before_a_shell_call_and_never_for_a_read() {
    use rig_core::tool::ToolResult;
    use rig_ecs::tools::{AppToolsExt, ToolCalled, ToolOptions};
    use rig_harness_test_support::calls;
    let home = scratch_home("worker-gate");
    let id = HarnessId::generate();
    let model = MockCompletionModel::from_stream_turns([
        calls("c-1", "shell", serde_json::json!({})),
        reply("ran it"),
        calls("c-2", "read", serde_json::json!({})),
        reply("read it"),
    ]);
    let at = endpoint("gate");
    let (heard, server) = fake_rigd(&at, BTreeMap::new(), vec![send(1, "run")]);
    let grant = Grant {
        ask: vec!["shell".into()],
        net: true,
        ..Grant::default()
    };
    let (mut app, wakes) = worker_with(&home, &id, &at, &model, Some(grant));
    app.init_resource::<Ran>();
    for tool in ["shell", "read"] {
        app.add_open_tool(
            tool,
            "A tool.",
            ToolOptions::default(),
            move |called: On<ToolCalled<NoArgs>>, mut ran: ResMut<Ran>, mut commands: Commands| {
                ran.0.push(tool.to_owned());
                let done = Outcome::from(ToolResult::success("ok".into()));
                commands.entity(called.call).insert_if_new(done);
            },
        );
    }
    run_until(&mut app, &wakes, |_| opened(&lock(&heard)).len() == 1);
    assert!(app.world().resource::<Ran>().0.is_empty(), "nothing runs before the answer");
    let ask = opened(&lock(&heard)).first().map(|a| a.ask_id.clone());
    let tool = lock(&heard)
        .events
        .iter()
        .find(|e| e.kind == "ask.opened")
        .and_then(|e| Some(e.data.get("call")?.get("tool")?.as_str()?.to_owned()));
    assert_eq!(tool.as_deref(), Some("shell"));
    command(
        &heard,
        WorkerCommand {
            command_seq: CommandSeq(2),
            command: Command::Answer(methods::Answer {
                session_id: SessionId(id.to_string()),
                ask_id: ask.unwrap_or(AskId(String::new())),
                option_id: Some("allow".into()),
                text: None,
                key: Key("k-2".into()),
            }),
        },
    );
    run_until(&mut app, &wakes, |_| turn_ended(&lock(&heard)).len() == 1);
    command(&heard, send(3, "read"));
    run_until(&mut app, &wakes, |_| turn_ended(&lock(&heard)).len() == 2);
    assert_eq!(app.world().resource::<Ran>().0, ["shell", "read"]);
    assert_eq!(opened(&lock(&heard)).len(), 1, "a read never asks");
    drop(app);
    drop(server);
}

#[test]
fn a_reply_bigger_than_a_frame_still_reaches_rigd_whole() {
    let home = scratch_home("worker-big");
    let id = HarnessId::generate();
    let big = "x".repeat(400 * 1024);
    let model = MockCompletionModel::from_stream_turns([reply(&big)]);
    let at = endpoint("big");
    let (heard, server) = fake_rigd(&at, BTreeMap::new(), vec![send(1, "write a lot")]);
    let (mut app, wakes) = worker(&home, &id, &at, &model);
    run_until(&mut app, &wakes, |_| turn_ended(&lock(&heard)).len() == 1);
    let longest = lock(&heard)
        .events
        .iter()
        .filter(|e| e.kind == "message")
        .filter_map(|e| Some(e.data.get("text")?.as_str()?.len()))
        .max();
    assert_eq!(longest, Some(big.len()));
    drop(app);
    drop(server);
}

#[test]
fn an_event_too_big_for_any_frame_goes_with_its_text_cut_and_said_so() {
    let huge = WorkerEvent {
        kind: "message".into(),
        data: serde_json::json!({"text": "y".repeat(3 << 20)}),
        source: None,
    };
    let small = WorkerEvent {
        kind: "notice".into(),
        data: serde_json::json!({"text": "after"}),
        source: None,
    };
    let batches = export::batches(vec![huge, small]);
    let sizes: Vec<usize> = batches
        .iter()
        .map(|b| serde_json::to_vec(b).map_or(usize::MAX, |v| v.len()))
        .collect();
    assert!(sizes.iter().all(|size| *size < rig_protocol::MAX_FRAME), "{sizes:?}");
    let texts: Vec<String> = batches
        .iter()
        .flatten()
        .filter_map(|e| Some(e.data.get("text")?.as_str()?.to_owned()))
        .collect();
    assert_eq!(texts.len(), 2);
    assert!(texts.first().is_some_and(|t| t.ends_with("the session's log holds the whole text]")));
    assert_eq!(texts.last().map(String::as_str), Some("after"));
}

#[test]
fn a_message_sent_while_an_approval_is_open_leaves_it_open() {
    let home = scratch_home("worker-send-open");
    let id = HarnessId::generate();
    let model = MockCompletionModel::from_stream_turns([reply("noted")]);
    let at = endpoint("sendopen");
    let (heard, server) = fake_rigd(&at, BTreeMap::new(), Vec::new());
    let (mut app, wakes) = worker(&home, &id, &at, &model);
    run_until(&mut app, &wakes, |world| world.contains_resource::<WorkerChannel>());
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next()
        .unwrap_or(Entity::PLACEHOLDER);
    let approval = Ask::new("Allow `shell`?")
        .choice("deny", "Deny")
        .choice("allow", "Allow")
        .free_text();
    let open = world.spawn((approval, ForAgent(agent))).id();
    run_until(&mut app, &wakes, |_| opened(&lock(&heard)).len() == 1);
    command(&heard, send(1, "also check the tests"));
    run_until(&mut app, &wakes, |_| turn_ended(&lock(&heard)).len() == 1);
    assert!(
        app.world().get::<Outcome>(open).is_none(),
        "the message did not answer the approval"
    );
    drop(app);
    drop(server);
}

#[test]
fn a_message_with_an_attachment_keeps_it_in_order() {
    use rig_core::message::{ImageMediaType, UserContent};
    let home = scratch_home("worker-attach");
    let id = HarnessId::generate();
    let model = MockCompletionModel::from_stream_turns([reply("a cat")]);
    let at = endpoint("attach");
    let image = UserContent::image_base64("iVBORw0KGgo=", Some(ImageMediaType::PNG), None);
    let content = vec![
        serde_json::json!({"type": "text", "text": "What is this?"}),
        serde_json::json!({"type": "attachment", "label": "cat.png", "content": image}),
        serde_json::json!({"type": "attachment", "label": "bad", "content": {"nonsense": 1}}),
    ];
    let with_image = WorkerCommand {
        command_seq: CommandSeq(1),
        command: Command::Send(SendCommand {
            session_id: SessionId("s".into()),
            text: None,
            content: Some(content),
            mode: None,
            key: Key("k-1".into()),
        }),
    };
    let (heard, server) = fake_rigd(&at, BTreeMap::new(), vec![with_image]);
    let (mut app, wakes) = worker(&home, &id, &at, &model);
    run_until(&mut app, &wakes, |_| turn_ended(&lock(&heard)).len() == 1);
    let sent = model
        .requests()
        .first()
        .map(|request| format!("{request:?}"))
        .unwrap_or_default();
    assert!(sent.contains("What is this?"), "{sent}");
    assert!(sent.contains("iVBORw0KGgo="), "the image went to the model: {sent}");
    drop(app);
    drop(server);
}
