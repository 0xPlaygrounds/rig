#![doc = include_str!("../README.md")]

use std::io::{BufRead as _, Write};

use bevy_ecs::system::SystemParam;
use crossbeam_channel::{Receiver, Sender, TryRecvError};
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

/// The command lines coming in: stdin's unless the app inserted another.
#[derive(Resource)]
pub struct RpcInput(Receiver<String>);

impl RpcInput {
    /// An input fed by the returned sender; it ends when the sender drops.
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
        let args = invoked.as_ref().map(|invoked| &invoked.args);
        let prompt = args.and_then(|args| args.print.clone());
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
            .add_systems(AgentPass, stream_deltas.in_set(Stage::Review));
        if rpc {
            app.world_mut().spawn(Answerer::default());
            if !app.world().contains_resource::<RpcInput>() {
                app.add_systems(Startup, read_stdin);
            }
            app.add_systems(
                Update,
                run_commands
                    .before(AgentLoop)
                    .run_if(resource_exists::<RpcInput>),
            );
        } else {
            app.add_systems(AgentPass, refuse_asks.in_set(Stage::Gate));
        }
        if let Some(prompt) = prompt {
            let model = args.is_some_and(|args| args.model.is_some());
            app.insert_resource(PrintRun {
                prompt,
                model,
                step: Step::Start,
                exit: !rpc,
                answered: false,
            })
            .add_systems(
                Update,
                drive
                    .before(AgentLoop)
                    .run_if(resource_exists::<SessionRestored>),
            );
        }
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
    run: Option<ResMut<PrintRun>>,
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
                json!({ "outcome": "answered", "text": final_answer(message) })
            }
            TurnOutcome::Failed(why) => json!({ "outcome": "failed", "error": why }),
            TurnOutcome::Stopped(why) => json!({ "outcome": "stopped", "error": why }),
        };
        if let Some(mut run) = run
            && run.step == Step::Sent(agent)
        {
            run.answered = matches!(ended, TurnOutcome::Answered(_));
        }
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
        let kind = match delta.kind {
            DeltaKind::Text => "text_delta",
            DeltaKind::Reasoning => "reasoning_delta",
        };
        out.emit(kind, delta.agent, json!({ "text": delta.text }));
    }
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

/// Nobody answers a `-p` run's questions: each is refused.
fn refuse_asks(asks: Query<Entity, (With<Ask>, Undecided)>, mut commands: Commands) {
    for ask in &asks {
        let why = "nobody answers questions in a --json -p run";
        commands.entity(ask).insert_if_new(Outcome::refused(why));
    }
}

/// Starts the thread that feeds stdin's lines in, waking the loop for each.
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
        // The input ends with the sender.
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
    /// The agent's `AgentId`; the agent the user talks to without one.
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
    input: Res<'w, RpcInput>,
    agents: PrimaryQuery<'w, 's>,
    asks: Query<'w, 's, &'static Ask, (Without<Outcome>, Without<Settled>)>,
    pickers: Query<'w, 's, &'static Picker>,
    commands: Commands<'w, 's>,
    exits: MessageWriter<'w, AppExit>,
}

/// Runs each command line that came in, answering each with a `response`;
/// the end of the input ends the run.
fn run_commands(mut rpc: Rpc, mut out: Out) {
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
        Some(id) => AgentId(id.clone()).find_in(rpc.agents.iter().map(|(e, id, _)| (e, id))),
        None => primary(&rpc.agents),
    };
    let agent = agent.ok_or("no such agent")?;
    match command.kind {
        Kind::Prompt { text } => {
            let prompt = Deliver::new(agent, text, DeliveryMode::Queue);
            rpc.commands.trigger(prompt);
        }
        Kind::Interrupt => rpc.commands.trigger(Interrupt::new(agent)),
        Kind::SetModel { model } => rpc.commands.trigger(SetModel {
            entity: agent,
            model,
        }),
        Kind::Answer {
            request_id,
            choice,
            text,
        } => {
            let request = Entity::try_from_bits(request_id).ok_or("no such request")?;
            answer(rpc, agent, request, (choice, text))?;
        }
    }
    Ok(())
}

/// Answers the open question or picker `request` with the item at
/// `choice`, else with `text`.
fn answer(
    rpc: &mut Rpc,
    agent: Entity,
    request: Entity,
    (choice, text): (Option<usize>, Option<String>),
) -> Result<(), String> {
    if let Ok(ask) = rpc.asks.get(request) {
        let answer = match (choice, text) {
            (Some(index), _) => {
                let choice = ask.choices.get(index).ok_or("no such choice")?;
                Answer::Choice(choice.id.clone())
            }
            (None, Some(text)) if ask.free_text => Answer::Text(text),
            _ => return Err("the answer needs a choice".to_owned()),
        };
        rpc.commands.entity(request).insert_if_new(answer.outcome());
        return Ok(());
    }
    // A picker is the terminal view's: picking runs the item's command,
    // then tells its plugin and closes it, as the view does.
    let picker = rpc.pickers.get(request).map_err(|_| "no such request")?;
    let chosen = match (choice, text) {
        (Some(index), _) => {
            let item = picker.items.get(index).ok_or("no such choice")?;
            if !item.command.is_empty() {
                rpc.commands.trigger(RunCommand::new(
                    picker.agent.unwrap_or(agent),
                    item.command.clone(),
                ));
            }
            Chosen::Item(index)
        }
        (None, Some(text)) if picker.free_text => Chosen::Text(text),
        _ => return Err("the answer needs a choice".to_owned()),
    };
    rpc.commands.trigger(Picked {
        entity: request,
        chosen: Some(chosen),
    });
    rpc.commands.entity(request).despawn();
    Ok(())
}

/// A `-p` run: its prompt, then, unless commands keep it running, the exit
/// once no agent works.
#[derive(Resource)]
struct PrintRun {
    prompt: String,
    /// Whether `--model` named the model, which rig-models chooses.
    model: bool,
    step: Step,
    /// Whether the run exits after the prompt.
    exit: bool,
    /// Whether the prompt's last turn ended with an answer.
    answered: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Step {
    Start,
    /// The prompt went to this agent.
    Sent(Entity),
    Done,
}

/// What keeps a `-p` run from exiting: turns, other work such as a
/// sign-in, and messages on their way to an agent.
#[derive(SystemParam)]
struct Busy<'w, 's> {
    turns: Query<'w, 's, (), With<TurnOf>>,
    running: Query<'w, 's, (), With<Running>>,
    messages: Query<'w, 's, (), (With<Incoming>, Without<Settled>)>,
}

impl Busy<'_, '_> {
    fn any(&self) -> bool {
        !self.turns.is_empty() || !self.running.is_empty() || !self.messages.is_empty()
    }
}

/// Sends the prompt, with the first model that can be reached when none
/// was chosen, and exits once no agent works: 0 when the prompt's turn was
/// answered, 1 otherwise.
fn drive(
    mut run: ResMut<PrintRun>,
    (agents, chosen): (PrimaryQuery, Query<(), With<ModelChoice>>),
    busy: Busy,
    models: Res<Models>,
    mut commands: Commands,
    mut exits: MessageWriter<AppExit>,
) {
    match run.step {
        Step::Start => {
            let Some(agent) = primary(&agents) else {
                return;
            };
            if !run.model
                && !chosen.contains(agent)
                && let Some(spec) = models.0.reachable().first()
            {
                let model = spec.reference().to_string();
                commands.trigger(SetModel {
                    entity: agent,
                    model,
                });
            }
            let prompt = Deliver::new(agent, run.prompt.clone(), DeliveryMode::Steer);
            commands.trigger(prompt);
            run.step = Step::Sent(agent);
        }
        Step::Sent(_) if run.exit && !busy.any() => {
            run.step = Step::Done;
            exits.write(match run.answered {
                true => AppExit::Success,
                false => AppExit::from_code(1),
            });
        }
        Step::Sent(_) | Step::Done => {}
    }
}

#[cfg(test)]
mod tests;
