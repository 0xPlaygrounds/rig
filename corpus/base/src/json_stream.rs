#![doc = include_str!("json_stream/README.md")]

use std::io::Write;

use async_channel::{Receiver, Sender};
use bevy_ecs::system::SystemParam;
use bevy_tasks::{IoTaskPool, TaskPool};
use rig_basics::headless::{AppHeadlessExt, HeadlessEnded};
use rig_harness::prelude::*;
use serde::Deserialize;
use serde_json::{Map, Value, json};

/// The schema version every line carries as `v`.
pub const VERSION: u32 = 1;

/// The front's name and its option, `--json`.
pub const FRONT: &str = "json";

/// The option that keeps a `--json` run reading commands, `--rpc`.
pub const RPC: &str = "rpc";

/// Where the lines go: stdout unless the app inserted another.
#[derive(Resource)]
pub struct JsonSink(Box<dyn Write + Send + Sync>);

impl JsonSink {
    /// Lines to `out`.
    pub fn new(out: impl Write + Send + Sync + 'static) -> Self {
        Self(Box::new(out))
    }
}

/// The command lines coming in: stdin's unless the app inserted another;
/// they end when every sender dropped.
#[derive(Resource)]
pub struct RpcInput(Receiver<String>);

impl RpcInput {
    /// An input fed by the returned sender; it ends when the sender drops.
    pub fn channel() -> (Sender<String>, Self) {
        let (sender, receiver) = async_channel::unbounded();
        (sender, Self(receiver))
    }
}

/// The front's Review system, which writes what each turn's model call
/// streamed, for other plugins to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct JsonStreamSystems;

/// The JSON event stream and RPC front.
#[derive(Default)]
pub struct JsonStreamPlugin;

impl Plugin for JsonStreamPlugin {
    fn build(&self, app: &mut App) {
        app.add_front(Front::new(FRONT).on_flag(FRONT, "Print every event as a JSON line"))
            .add_flag(Flag::new(
                RPC,
                "With --json, read commands as JSON lines on stdin",
            ));
    }

    fn finish(&self, app: &mut App) {
        if !app.runs_front(FRONT) {
            return;
        }
        let invoked = app.world().get_resource::<Invoked>().cloned();
        let prompt = invoked
            .as_ref()
            .and_then(|invoked| invoked.args.print.clone());
        let rpc = prompt.is_none() || invoked.as_ref().is_some_and(|invoked| invoked.flag(RPC));
        if !app.world().contains_resource::<JsonSink>() {
            app.insert_resource(JsonSink::new(std::io::stdout()));
        }
        app.add_observer(turn_started)
            .add_observer(started)
            .add_observer(settled)
            .add_observer(picker_opened)
            .add_systems(
                Update,
                (
                    session_started.run_if(resource_added::<SessionRestored>),
                    notices,
                ),
            )
            .configure_sets(AgentPass, JsonStreamSystems.in_set(Stage::Review))
            .add_systems(AgentPass, stream_deltas.in_set(JsonStreamSystems));
        if !rpc {
            // Nobody answers questions: with no `Answerer`, plugins refuse.
            if let Some(prompt) = prompt {
                app.run_headless(Some(prompt)).add_observer(headless_ended);
            }
            return;
        }
        app.world_mut().spawn(Answerer::default());
        if !app.world().contains_resource::<RpcInput>() {
            let (sender, input) = RpcInput::channel();
            // A `-p` prompt is the first command.
            if let Some(text) = prompt {
                let first = json!({ "type": "prompt", "id": "print", "text": text });
                sender.try_send(first.to_string()).ok();
            }
            app.insert_resource(input)
                .add_systems(Startup, move || read_stdin(sender.clone()));
        }
        app.add_systems(Startup, listen)
            .add_observer(command)
            .add_observer(input_ended);
    }
}

/// Writes lines, each naming its agent and the agent's parent.
#[derive(SystemParam)]
struct Out<'w, 's> {
    sink: ResMut<'w, JsonSink>,
    agents: Query<'w, 's, (&'static AgentId, Option<&'static SpawnedBy>)>,
}

impl Out<'_, '_> {
    /// The ids naming `agent` on a line: its own and its parent's.
    fn ids(&self, agent: Option<Entity>) -> (Value, Option<Value>) {
        let Some((id, spawned)) = agent.and_then(|agent| self.agents.get(agent).ok()) else {
            return (Value::Null, None);
        };
        let parent = spawned
            .and_then(|SpawnedBy(parent)| self.agents.get(*parent).ok())
            .map(|(parent, _)| Value::String(parent.0.clone()));
        (Value::String(id.0.clone()), parent)
    }

    /// One `kind` line about `agent` with `fields`.
    fn emit(&mut self, kind: &str, agent: Option<Entity>, fields: Value) {
        let (agent, parent) = self.ids(agent);
        let mut line = Map::new();
        line.insert("v".to_owned(), VERSION.into());
        line.insert("type".to_owned(), kind.into());
        line.insert("agent".to_owned(), agent);
        if let Some(parent) = parent {
            line.insert("parent".to_owned(), parent);
        }
        if let Value::Object(fields) = fields {
            line.extend(fields);
        }
        let out = &mut self.sink.0;
        writeln!(out, "{}", Value::Object(line)).ok();
        out.flush().ok();
    }
}

fn session_started(agents: Query<(Entity, &Name), With<Agent>>, mut out: Out) {
    let listed: Vec<Value> = agents
        .iter()
        .map(|(agent, name)| {
            let (id, parent) = out.ids(Some(agent));
            json!({ "agent": id, "parent": parent, "name": name.as_str() })
        })
        .collect();
    out.emit("session_start", None, json!({ "agents": listed }));
}

fn turn_started(turn: On<Add<TurnOf>>, turns: Query<&TurnOf>, mut out: Out) {
    if let Ok(TurnOf(agent)) = turns.get(turn.entity) {
        out.emit("turn_start", Some(*agent), json!({}));
    }
}

/// A tool call starts, or a question waits for its answer.
fn started(
    start: On<Add<Started>>,
    calls: Query<(&ToolRequest, &ForAgent)>,
    asks: Query<(&Ask, Option<&ForAgent>)>,
    mut out: Out,
) {
    if let Ok((request, &ForAgent(agent))) = calls.get(start.entity) {
        let call = &request.call;
        let fields = json!({
            "id": call.id.wire(),
            "name": call.function.name.as_str(),
            "args": call.function.arguments_value(),
        });
        out.emit("tool_call_start", Some(agent), fields);
    } else if let Ok((ask, agent)) = asks.get(start.entity) {
        let choices: Vec<&str> = ask.choices.iter().map(|c| c.label.as_str()).collect();
        let request = (start.entity, "ask", ask.title.as_str(), ask.body.as_str());
        emit_request(
            &mut out,
            agent.map(|agent| agent.0),
            request,
            choices,
            ask.free_text,
        );
    }
}

/// A `request` line about `entity`, which an `answer` names.
fn emit_request(
    out: &mut Out,
    agent: Option<Entity>,
    (entity, kind, title, body): (Entity, &str, &str, &str),
    choices: Vec<&str>,
    free_text: bool,
) {
    let fields = json!({
        "request_id": entity.to_bits(),
        "kind": kind,
        "title": title,
        "body": body,
        "choices": choices,
        "free_text": free_text,
    });
    out.emit("request", agent, fields);
}

fn picker_opened(opened: On<Insert<Picker>>, pickers: Query<&Picker>, mut out: Out) {
    if let Ok(picker) = pickers.get(opened.entity) {
        let choices: Vec<&str> = picker.items.iter().map(|i| i.label.as_str()).collect();
        let request = (
            opened.entity,
            "picker",
            picker.title.as_str(),
            picker.body.as_str(),
        );
        emit_request(&mut out, picker.agent, request, choices, picker.free_text);
    }
}

/// A tool call, model call or turn settled.
fn settled(
    settled: On<Add<Settled>>,
    effects: Query<(&ForAgent, Option<&Outcome>, Option<&TurnOutcome>)>,
    calls: Query<&ToolRequest>,
    models: Query<(), With<ModelRequest>>,
    mut out: Out,
) {
    let entity = settled.entity;
    let Ok((&ForAgent(agent), outcome, ended)) = effects.get(entity) else {
        return;
    };
    if let (Ok(request), Some(outcome)) = (calls.get(entity), outcome) {
        let result = outcome.tool_result();
        let fields = json!({
            "id": request.call.id.wire(),
            "name": request.call.function.name.as_str(),
            "result": serde_json::to_value(result.output()).unwrap_or_default(),
            "is_error": !result.is_success(),
        });
        out.emit("tool_call_end", Some(agent), fields);
    } else if let Some(response) = outcome
        .filter(|_| models.contains(entity))
        .and_then(Outcome::completion)
    {
        let usage = serde_json::to_value(response.usage).unwrap_or_default();
        out.emit("usage", Some(agent), json!({ "usage": usage }));
    } else if let Some(ended) = ended {
        let fields = match ended {
            TurnOutcome::Answered(message) => {
                json!({ "outcome": "answered", "text": message.final_answer() })
            }
            TurnOutcome::Failed(why) => json!({ "outcome": "failed", "error": why }),
            TurnOutcome::Stopped(why) => json!({ "outcome": "stopped", "error": why }),
        };
        out.emit("turn_end", Some(agent), fields);
    }
}

/// Sends what each turn's model call streamed, in Review right after the
/// kernel read it, so it goes out before the reply's settling events.
fn stream_deltas(
    mut deltas: MessageReader<StreamDelta>,
    turn_calls: Query<(), With<TurnCall>>,
    mut out: Out,
) {
    for delta in deltas.read() {
        if !turn_calls.contains(delta.effect) {
            continue;
        }
        let (kind, text) = match &delta.event {
            StreamEvent::Text { text, .. } => ("text_delta", text),
            StreamEvent::Reasoning { text, .. } => ("reasoning_delta", text),
            _ => continue,
        };
        out.emit(kind, delta.agent, json!({ "text": text }));
    }
}

fn notices(mut notices: MessageReader<Notice>, mut out: Out) {
    for notice in notices.read() {
        let level = match notice.level {
            NoticeLevel::Info => "info",
            NoticeLevel::Warn => "warn",
            NoticeLevel::Error => "error",
        };
        let fields = json!({ "level": level, "text": notice.text });
        out.emit("notice", notice.agent, fields);
    }
}

/// The command lines of an RPC run, a stream on this entity.
#[derive(Component)]
struct CommandLines;

/// Starts the thread that sends stdin's lines to `sender`.
fn read_stdin(sender: Sender<String>) {
    std::thread::spawn(move || {
        for line in std::io::stdin().lines() {
            let Ok(line) = line else {
                break;
            };
            if sender.send_blocking(line).is_err() {
                break;
            }
        }
    });
}

/// Reads the command lines as a stream on an entity of their own.
fn listen(input: Option<Res<RpcInput>>, wake: Option<Res<Wake>>, mut commands: Commands) {
    let Some(input) = input else {
        return;
    };
    let wake = wake.map(|wake| wake.clone()).unwrap_or_default();
    let pool = IoTaskPool::get_or_init(TaskPool::default);
    commands.spawn((CommandLines, Running::stream(pool, &wake, input.0.clone())));
    commands.remove_resource::<RpcInput>();
}

/// The end of the command lines ends the run.
fn input_ended(
    ended: On<Remove<Running>>,
    lines: Query<(), With<CommandLines>>,
    mut commands: Commands,
) {
    if lines.contains(ended.entity) {
        commands.trigger(Exit(ExitReason::Quit));
    }
}

/// A `-p` run could not send its prompt.
fn headless_ended(ended: On<HeadlessEnded>, mut out: Out) {
    if let Some(why) = &ended.why {
        let fields = json!({ "level": "error", "text": why });
        out.emit("notice", ended.agent, fields);
    }
}

/// A command line.
#[derive(Deserialize)]
struct Command {
    #[serde(default)]
    id: Value,
    /// The agent's `AgentId`; the agent shown without one.
    #[serde(default)]
    agent: Option<String>,
    #[serde(flatten)]
    kind: Kind,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Kind {
    Prompt {
        text: String,
    },
    Interrupt,
    SetModel {
        model: String,
    },
    Answer {
        request_id: u64,
        choice: Option<usize>,
        text: Option<String>,
    },
}

/// What a command needs of the app.
#[derive(SystemParam)]
struct Rpc<'w, 's> {
    agents: Query<'w, 's, (Entity, &'static AgentId)>,
    shown: Res<'w, ShownAgent>,
    asks: Query<'w, 's, &'static Ask, (Without<Outcome>, Without<Settled>)>,
    pickers: Query<'w, 's, &'static Picker>,
    commands: Commands<'w, 's>,
}

impl Rpc<'_, '_> {
    /// The agent named `id`, else the one shown.
    fn agent(&self, id: Option<String>) -> Result<Entity, String> {
        let agent = match id {
            Some(id) => AgentId(id).find_in(self.agents.iter()),
            None => self.shown.0,
        };
        agent.ok_or_else(|| "no such agent".to_owned())
    }
}

/// Runs a command line that came in and answers it with a `response`.
fn command(
    line: On<Insert<Done<String>>>,
    lines: Query<&Done<String>, With<CommandLines>>,
    mut rpc: Rpc,
    mut out: Out,
) {
    let Ok(Done(line)) = lines.get(line.entity) else {
        return;
    };
    if line.trim().is_empty() {
        return;
    }
    let (id, done) = match serde_json::from_str::<Command>(line) {
        Ok(command) => (command.id.clone(), run(&mut rpc, command)),
        Err(error) => (Value::Null, Err(format!("not a command: {error}"))),
    };
    let fields = match done {
        Ok(()) => json!({ "id": id, "ok": true }),
        Err(error) => json!({ "id": id, "ok": false, "error": error }),
    };
    out.emit("response", None, fields);
}

/// Runs `command`, or says why it cannot.
fn run(rpc: &mut Rpc, command: Command) -> Result<(), String> {
    match command.kind {
        Kind::Prompt { text } => {
            let agent = rpc.agent(command.agent)?;
            let prompt = Deliver::new(agent, text, DeliveryMode::Queue);
            rpc.commands.trigger(prompt);
        }
        Kind::Interrupt => {
            let agent = rpc.agent(command.agent)?;
            rpc.commands.trigger(StopTurn::new(agent));
        }
        Kind::SetModel { model } => {
            let entity = rpc.agent(command.agent)?;
            rpc.commands.trigger(SetModel { entity, model });
        }
        Kind::Answer {
            request_id,
            choice,
            text,
        } => {
            let entity = Entity::try_from_bits(request_id).ok_or("no such request")?;
            let (count, free_text) = match (rpc.asks.get(entity), rpc.pickers.get(entity)) {
                (Ok(ask), _) => (ask.choices.len(), ask.free_text),
                (_, Ok(picker)) => (picker.items.len(), picker.free_text),
                _ => return Err("no such request".to_owned()),
            };
            let chosen = match (choice, text) {
                (Some(index), _) if index < count => Chosen::item(index),
                (Some(_), _) => return Err("no such choice".to_owned()),
                (None, Some(text)) if free_text => Chosen::text(text),
                _ => return Err("the answer needs a choice".to_owned()),
            };
            rpc.commands.trigger(Pick::person(entity, Some(chosen)));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
