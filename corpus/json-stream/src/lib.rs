//! Plugin corpus: 48. JSON event stream and RPC mode (spec `48-json-stream.md`).
//!
//! With `RIG_JSON=1` (stream) or `RIG_JSON=rpc`, every event of every
//! agent goes to stdout as one JSON line, and in RPC mode stdin takes
//! commands as JSON lines. The launcher refuses `--json` and `--rpc`
//! (`Invocation::parse` knows only `-p` and `-m`), so the mode comes from
//! the environment; an app may insert [`JsonMode`] itself.
//!
//! # Schema, version 1
//!
//! Every line is an object with `v` (1), `type`, `agent` (the `AgentId`,
//! or `null` for app-wide lines) and, for a subagent, `parent`.
//!
//! Events: `session_start {agents: [{agent, parent, name}]}`,
//! `turn_start`, `text_delta {text}`, `reasoning_delta {text}`,
//! `tool_call_start {id, name, args}`, `tool_call_end {id, name, result,
//! is_error}`, `notice {level, text}`, `usage {usage}`, `turn_end {outcome:
//! "answered" | "failed" | "stopped", text?, error?}`, `request
//! {request_id, title, choices, selected}` (a picker another plugin opened)
//! and `response {id, ok, error?}`.
//!
//! Commands (RPC): `{"type":"prompt","id":…,"agent"?:…,"text":…}`,
//! `{"type":"interrupt","id":…,"agent"?:…}`,
//! `{"type":"set_model","id":…,"agent"?:…,"model":"vendor/model"}`,
//! `{"type":"answer","id":…,"request_id":…,"choice":<index>}`. Without
//! `agent` a command goes to the primary agent. End of stdin exits.
//!
//! `rig-print` must be left out of `plugins.toml` for this front: it
//! drives a non-interactive run itself, reads stdin and prints the answer.

use std::collections::HashMap;
use std::io::{BufRead as _, Write};

use bevy_ecs::system::SystemParam;
use crossbeam_channel::{Receiver, Sender, TryRecvError};
use rig_harness::prelude::*;
use rig_harness::rig_ecs::calls::Done;
use rig_harness::rig_ecs::turn::ModelReply;
use rig_tui::{PickItem, PickRequest};
use serde::Deserialize;
use serde_json::{Map, Value, json};

/// The schema version every line carries as `v`.
pub const VERSION: u32 = 1;

/// Which JSON front runs.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub enum JsonMode {
    /// Events out; with `-p`, one prompt then exit.
    Stream,
    /// Events out and commands in, until stdin ends.
    Rpc,
}

impl JsonMode {
    /// From `RIG_JSON`: `rpc`, or any other non-empty value for a stream.
    pub fn from_env() -> Option<Self> {
        match std::env::var("RIG_JSON").ok()?.as_str() {
            "" | "0" => None,
            "rpc" => Some(Self::Rpc),
            _ => Some(Self::Stream),
        }
    }
}

/// Where the lines go: stdout unless the app inserted another.
#[derive(Resource)]
pub struct JsonSink(Box<dyn Write + Send + Sync>);

impl JsonSink {
    /// Lines to `out`.
    pub fn new(out: impl Write + Send + Sync + 'static) -> Self {
        Self(Box::new(out))
    }
}

/// The command lines coming in: stdin's in RPC mode unless the app
/// inserted another.
#[derive(Resource)]
pub struct RpcInput(Receiver<String>);

impl RpcInput {
    /// An input fed by the returned sender.
    pub fn channel() -> (Sender<String>, Self) {
        let (sender, receiver) = crossbeam_channel::unbounded();
        (sender, Self(receiver))
    }
}

/// The JSON event stream and RPC front.
#[derive(Default)]
pub struct JsonStreamPlugin;

impl Plugin for JsonStreamPlugin {
    fn build(&self, app: &mut App) {
        let mode = match app.world().get_resource::<JsonMode>() {
            Some(mode) => *mode,
            None => match JsonMode::from_env() {
                Some(mode) => mode,
                None => return,
            },
        };
        app.insert_resource(mode)
            .add_message::<PickRequest>()
            .init_resource::<Pickers>()
            .register_required_components::<Partial, Streamed>()
            .add_observer(turn_started)
            .add_observer(tool_started)
            .add_observer(tool_ended)
            .add_observer(model_done)
            .add_observer(turn_ended)
            .add_systems(
                Update,
                (
                    session_started.run_if(resource_added::<SessionRestored>),
                    stream_deltas.after(PollCalls),
                    notices,
                    pickers,
                    commands.run_if(resource_exists::<RpcInput>),
                ),
            );
        if !app.world().contains_resource::<JsonSink>() {
            app.insert_resource(JsonSink::new(std::io::stdout()));
        }
        if mode == JsonMode::Rpc && !app.world().contains_resource::<RpcInput>() {
            app.add_systems(Startup, read_stdin);
        }
        let prompt = app
            .world()
            .get_resource::<Invoked>()
            .and_then(|invoked| invoked.args.print.clone());
        if let (JsonMode::Stream, Some(prompt)) = (mode, prompt) {
            app.insert_resource(PrintRun {
                prompt,
                step: Step::Start,
                answered: false,
            })
            .add_systems(Update, drive);
        }
    }
}

/// What a model call's stream sent out so far, in bytes.
#[derive(Component, Default)]
struct Streamed {
    text: usize,
    reasoning: usize,
}

/// Pickers waiting for an `answer`, by request id.
#[derive(Resource, Default)]
struct Pickers {
    next: u64,
    open: HashMap<String, (Entity, Vec<PickItem>)>,
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

/// The agent whose turn made the call `call`, if any.
fn call_agent(call: Entity, of: &Query<&CallOf>, turns: &Query<&TurnOf>) -> Option<Entity> {
    let CallOf(turn) = of.get(call).ok()?;
    turns.get(*turn).ok().map(|TurnOf(agent)| *agent)
}

fn tool_started(
    add: On<Add<ToolCallRun>>,
    runs: Query<&ToolCallRun>,
    of: Query<&CallOf>,
    turns: Query<&TurnOf>,
    mut out: Out,
) {
    let Ok(run) = runs.get(add.entity) else {
        return;
    };
    let call = &run.call;
    let fields = json!({
        "id": call.id.wire(),
        "name": call.function.name.as_str(),
        "args": call.function.arguments_value(),
    });
    out.emit(
        "tool_call_start",
        call_agent(add.entity, &of, &turns),
        fields,
    );
}

fn tool_ended(
    add: On<Add<ToolOutput>>,
    calls: Query<(&ToolCallRun, &ToolOutput)>,
    of: Query<&CallOf>,
    turns: Query<&TurnOf>,
    mut out: Out,
) {
    let Ok((run, ToolOutput(result))) = calls.get(add.entity) else {
        return;
    };
    let fields = json!({
        "id": run.call.id.wire(),
        "name": run.call.function.name.as_str(),
        "result": serde_json::to_value(result.output()).unwrap_or_default(),
        "is_error": !result.is_success(),
    });
    out.emit("tool_call_end", call_agent(add.entity, &of, &turns), fields);
}

/// Sends what each streaming model call added since the last frame.
fn stream_deltas(
    mut calls: Query<(Entity, &Partial, &mut Streamed), Changed<Partial>>,
    of: Query<&CallOf>,
    turns: Query<&TurnOf>,
    mut out: Out,
) {
    for (call, partial, mut streamed) in &mut calls {
        let agent = call_agent(call, &of, &turns);
        send_new(
            &mut out,
            agent,
            &partial.text,
            &mut streamed.text,
            "text_delta",
        );
        let reasoning = &partial.reasoning;
        send_new(
            &mut out,
            agent,
            reasoning,
            &mut streamed.reasoning,
            "reasoning_delta",
        );
    }
}

/// Sends `full` past the `sent` bytes as a `kind` line.
fn send_new(out: &mut Out, agent: Option<Entity>, full: &str, sent: &mut usize, kind: &str) {
    if let Some(new) = full.get(*sent..).filter(|new| !new.is_empty()) {
        out.emit(kind, agent, json!({ "text": new }));
        *sent = full.len();
    }
}

/// A model call ended: sends the rest of its text the stream did not
/// show yet (the call is gone before the next frame's deltas), and its
/// usage.
fn model_done(
    done: On<Add<Done<ModelReply>>>,
    mut calls: Query<(&Done<ModelReply>, Option<&mut Streamed>)>,
    of: Query<&CallOf>,
    turns: Query<&TurnOf>,
    mut out: Out,
) {
    let Ok((Done(reply), streamed)) = calls.get_mut(done.entity) else {
        return;
    };
    let Ok(response) = reply else {
        return;
    };
    let agent = call_agent(done.entity, &of, &turns);
    if let Some(mut streamed) = streamed {
        let (mut text, mut reasoning) = (String::new(), String::new());
        for item in &response.choice {
            match item {
                AssistantContent::Text(part) => text.push_str(part.text()),
                AssistantContent::Reasoning(part) => reasoning.push_str(&part.text),
                _ => {}
            }
        }
        send_new(&mut out, agent, &text, &mut streamed.text, "text_delta");
        let sent = &mut streamed.reasoning;
        send_new(&mut out, agent, &reasoning, sent, "reasoning_delta");
    }
    let usage = serde_json::to_value(response.usage).unwrap_or_default();
    out.emit("usage", agent, json!({ "usage": usage }));
}

fn turn_ended(ended: On<TurnEnded>, run: Option<ResMut<PrintRun>>, mut out: Out) {
    let fields = match &ended.outcome {
        TurnOutcome::Answered(message) => {
            json!({ "outcome": "answered", "text": final_answer(message) })
        }
        TurnOutcome::Failed(why) => json!({ "outcome": "failed", "error": why }),
        TurnOutcome::Stopped => json!({ "outcome": "stopped" }),
    };
    if let Some(mut run) = run
        && let Step::Sent { agent } = run.step
        && agent == ended.entity
    {
        run.answered = matches!(ended.outcome, TurnOutcome::Answered(_));
    }
    out.emit("turn_end", Some(ended.entity), fields);
}

fn notices(mut notices: MessageReader<Notice>, mut out: Out) {
    for notice in notices.read() {
        let level = match notice.level {
            NoticeLevel::Info => "info",
            NoticeLevel::Error => "error",
        };
        let fields = json!({ "level": level, "text": notice.text });
        out.emit("notice", notice.agent, fields);
    }
}

/// Turns each picker another plugin opens into a `request`.
fn pickers(mut requests: MessageReader<PickRequest>, mut open: ResMut<Pickers>, mut out: Out) {
    for request in requests.read() {
        open.next += 1;
        let id = format!("pick-{}", open.next);
        let choices: Vec<&str> = request
            .items
            .iter()
            .map(|item| item.label.as_str())
            .collect();
        let fields = json!({
            "request_id": id,
            "title": request.title,
            "choices": choices,
            "selected": request.selected,
        });
        out.emit("request", Some(request.agent), fields);
        open.open.insert(id, (request.agent, request.items.clone()));
    }
}

/// Starts the thread that feeds stdin's lines in.
fn read_stdin(wake: Res<Wake>, mut commands: Commands) {
    let (sender, input) = RpcInput::channel();
    let wake = wake.clone();
    std::thread::spawn(move || {
        for line in std::io::stdin().lock().lines() {
            let Ok(line) = line else {
                break;
            };
            if sender.send(line).is_err() {
                break;
            }
            wake.wake();
        }
        // The sender drops: the input ends.
        drop(sender);
        wake.wake();
    });
    commands.insert_resource(input);
}

/// A command line.
#[derive(Deserialize)]
struct Command {
    #[serde(default)]
    id: Value,
    /// The agent's `AgentId`; the primary agent without one.
    #[serde(default)]
    agent: Option<String>,
    #[serde(flatten)]
    kind: Kind,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Kind {
    Prompt { text: String },
    Interrupt,
    SetModel { model: String },
    Answer { request_id: String, choice: usize },
}

/// What a command line needs of the app.
#[derive(SystemParam)]
struct Rpc<'w, 's> {
    input: Res<'w, RpcInput>,
    primary: PrimaryQuery<'w, 's>,
    pickers: ResMut<'w, Pickers>,
    commands: Commands<'w, 's>,
    exits: MessageWriter<'w, AppExit>,
}

/// Runs each command line that came in, answering each with a `response`.
fn commands(mut rpc: Rpc, mut out: Out) {
    loop {
        let line = match rpc.input.0.try_recv() {
            Ok(line) => line,
            Err(TryRecvError::Empty) => return,
            Err(TryRecvError::Disconnected) => {
                rpc.exits.write(AppExit::Success);
                return;
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        let (id, done) = match serde_json::from_str::<Command>(&line) {
            Ok(command) => (command.id.clone(), run(&mut rpc, command)),
            Err(error) => (Value::Null, Err(format!("not a command: {error}"))),
        };
        let fields = match done {
            Ok(()) => json!({ "id": id, "ok": true }),
            Err(error) => json!({ "id": id, "ok": false, "error": error }),
        };
        out.emit("response", None, fields);
    }
}

/// Runs `command`, or says why it cannot.
fn run(rpc: &mut Rpc, command: Command) -> Result<(), String> {
    let agent = match &command.agent {
        Some(id) => {
            let agents = rpc.primary.iter().map(|(agent, id, _)| (agent, id));
            AgentId(id.clone()).find_in(agents)
        }
        None => primary(&rpc.primary),
    };
    let agent = agent.ok_or("no such agent")?;
    match command.kind {
        Kind::Prompt { text } => {
            let deliver = Deliver::new(agent, text, DeliveryMode::Queue);
            rpc.commands.trigger(deliver);
        }
        Kind::Interrupt => rpc.commands.trigger(Interrupt::new(agent)),
        Kind::SetModel { model } => rpc.commands.trigger(SetModel {
            entity: agent,
            model,
        }),
        Kind::Answer { request_id, choice } => {
            let (picker, items) = rpc
                .pickers
                .open
                .remove(&request_id)
                .ok_or("no such request")?;
            let item = items.get(choice).ok_or("no such choice")?;
            rpc.commands.trigger(RunCommand {
                entity: picker,
                line: item.command.clone(),
            });
        }
    }
    Ok(())
}

/// A `-p` run in stream mode.
#[derive(Resource)]
struct PrintRun {
    prompt: String,
    step: Step,
    /// Whether the prompt's turn ended with an answer.
    answered: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Step {
    Start,
    Connecting { agent: Entity },
    Sent { agent: Entity },
    Done,
}

/// Chooses a model when none was, sends the prompt, and exits once no
/// agent works: 0 when the prompt's turn was answered, 1 otherwise.
fn drive(
    mut run: ResMut<PrintRun>,
    agents: PrimaryQuery,
    choices: Query<(Option<&ModelChoice>, Has<Connection>)>,
    busy: Query<(), Or<(With<TurnOf>, With<Running>)>>,
    models: Res<Models>,
    mut commands: Commands,
    mut exits: MessageWriter<AppExit>,
) {
    match run.step {
        Step::Start => {
            let Some(agent) = primary(&agents) else {
                return;
            };
            if let Ok((None, false)) = choices.get(agent)
                && let Some(spec) = models.0.reachable().first()
            {
                let model = spec.reference().to_string();
                commands.trigger(SetModel {
                    entity: agent,
                    model,
                });
            }
            run.step = Step::Connecting { agent };
        }
        Step::Connecting { agent } => {
            if !choices.get(agent).is_ok_and(|(_, connected)| connected) {
                run.step = Step::Done;
                exits.write(AppExit::from_code(1));
                return;
            }
            let text = run.prompt.clone();
            commands.trigger(Deliver::new(agent, text, DeliveryMode::Steer));
            run.step = Step::Sent { agent };
        }
        Step::Sent { .. } => {
            if busy.is_empty() {
                run.step = Step::Done;
                exits.write(match run.answered {
                    true => AppExit::Success,
                    false => AppExit::from_code(1),
                });
            }
        }
        Step::Done => {}
    }
}

#[cfg(test)]
mod tests;
