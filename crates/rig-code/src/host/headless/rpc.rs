//! `--rpc`: requests as lines of JSON on stdin, answers and the
//! [event stream](super::events) as lines of JSON on stdout, until stdin
//! closes or a `quit` request comes.
//!
//! Every request has a `type` and may carry an `id`, which its answer
//! (`{"type": "response", "id", "ok", "data" | "error"}`) repeats:
//!
//! - `trigger`: `event` (a type path, or its short name such as `Submit`)
//!   and `value`: any reflectable request event of the app, the same ones
//!   the terminal view sends, triggered through Bevy's `ReflectEvent`:
//!   `Submit`, `FollowUp`, `Interrupt`, `Retry`, `Compact`, `SetModel`,
//!   `SetEffort`, `Approve`, `Rewind`, `Fork`, `Focus`, a plugin's own. An
//!   `entity` field may be left out (the primary agent), given as an
//!   [`AgentId`] string, or as entity bits from an event.
//! - `agents`: every agent, with its entity, model, state and spending.
//! - `messages`: an `agent`'s conversation (default: the primary agent).
//! - `events`: the request events `trigger` takes.
//! - `commands`: the slash commands, which `Submit` runs.
//! - `quit`: exit, saving the session.

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_ecs::reflect::{AppTypeRegistry, ReflectEvent};
use bevy_ecs::system::SystemState;
use bevy_reflect::serde::TypedReflectDeserializer;
use bevy_reflect::{TypeInfo, TypeRegistration};
use crossbeam_channel::Receiver;
use serde::Deserialize;
use serde::de::DeserializeSeed;
use serde_json::{Value, json};

use super::events::to_json;
use super::{PrimaryQuery, emit, primary};
use crate::core::agent::{ActiveTurn, Agent, AgentId, Conversation, ModelChoice};
use crate::core::calls::Wake;
use crate::core::commands::SlashCommand;
use crate::core::subagents::SubagentOf;
use crate::core::usage::Spending;
use crate::core::workdir::WorkDir;

/// Reads requests from stdin and answers them.
pub(super) struct RpcPlugin;

impl Plugin for RpcPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, read_stdin)
            .add_systems(PreUpdate, serve.run_if(resource_exists::<Requests>));
    }
}

/// What the stdin thread read: a line, or the end of stdin.
enum Input {
    Line(String),
    Closed,
}

/// The lines read from stdin.
#[derive(Resource)]
struct Requests(Receiver<Input>);

/// One request.
#[derive(Deserialize)]
struct Request {
    #[serde(default)]
    id: Value,
    #[serde(flatten)]
    command: Command,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Command {
    Trigger {
        event: String,
        #[serde(default)]
        value: Value,
    },
    Agents,
    Messages {
        #[serde(default)]
        agent: Value,
    },
    Events,
    Commands,
    Quit,
}

/// Starts the thread that reads stdin a line at a time and wakes the loop
/// for each.
fn read_stdin(wake: Res<Wake>, mut commands: Commands) {
    let (sender, receiver) = crossbeam_channel::unbounded();
    let wake = wake.clone();
    let started = std::thread::Builder::new()
        .name("rig-code-rpc".to_owned())
        .spawn(move || {
            for line in std::io::stdin().lines() {
                let Ok(line) = line else {
                    break;
                };
                if sender.send(Input::Line(line)).is_err() {
                    return;
                }
                wake.wake();
            }
            sender.send(Input::Closed).ok();
            wake.wake();
        });
    match started {
        Ok(_) => {
            commands.insert_resource(Requests(receiver));
        }
        Err(failure) => {
            emit(&json!({
                "type": "response",
                "id": null,
                "ok": false,
                "error": format!("could not read stdin: {failure}"),
            }));
            commands.write_message(AppExit::from_code(1));
        }
    }
}

/// Answers every request read since the last frame.
fn serve(world: &mut World) {
    let Some(requests) = world.get_resource::<Requests>().map(|r| r.0.clone()) else {
        return;
    };
    for input in requests.try_iter() {
        let line = match input {
            Input::Line(line) => line,
            Input::Closed => {
                world.write_message(AppExit::Success);
                return;
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        let (id, answer) = match serde_json::from_str::<Request>(&line) {
            Ok(request) => (request.id, answer(world, request.command)),
            Err(failure) => (Value::Null, Err(format!("not a request: {failure}"))),
        };
        emit(&match answer {
            Ok(data) => json!({"type": "response", "id": id, "ok": true, "data": data}),
            Err(error) => json!({"type": "response", "id": id, "ok": false, "error": error}),
        });
    }
    // Commands queued by the requests' observers run now, before the
    // frame's systems look.
    world.flush();
}

fn answer(world: &mut World, command: Command) -> Result<Value, String> {
    match command {
        Command::Trigger { event, value } => trigger(world, &event, value),
        Command::Agents => Ok(agents(world)),
        Command::Messages { agent } => {
            let agent = agent_entity(world, &agent)?;
            let conversation = world
                .get::<Conversation>(agent)
                .ok_or("that entity is not an agent")?;
            Ok(to_json(&conversation.0))
        }
        Command::Events => Ok(events(world)),
        Command::Commands => {
            let mut query = world.query::<&SlashCommand>();
            let mut commands: Vec<Value> = query
                .iter(world)
                .map(|command| json!({"name": command.name, "help": command.help}))
                .collect();
            commands.sort_by_key(|command| command.get("name").map(Value::to_string));
            Ok(Value::Array(commands))
        }
        Command::Quit => {
            world.write_message(AppExit::Success);
            Ok(Value::Null)
        }
    }
}

/// Triggers the reflected event `event` with `value`, as Bevy's remote
/// protocol does (`bevy_remote`'s `world.trigger_event`).
fn trigger(world: &mut World, event: &str, value: Value) -> Result<Value, String> {
    let registry = world
        .get_resource::<AppTypeRegistry>()
        .ok_or("the app has no type registry")?
        .clone();
    let registry = registry.read();
    let registration = registry
        .get_with_type_path(event)
        .or_else(|| registry.get_with_short_type_path(event))
        .ok_or_else(|| format!("no event `{event}`; `events` lists them"))?;
    let reflect_event = registration
        .data::<ReflectEvent>()
        .ok_or_else(|| format!("`{event}` is not a reflectable event"))?
        .clone();
    let value = with_entity(world, registration, value)?;
    let payload = TypedReflectDeserializer::new(registration, &registry)
        .deserialize(value)
        .map_err(|failure| format!("`{event}` is invalid: {failure}"))?;
    reflect_event.trigger(world, payload.as_partial_reflect(), &registry);
    Ok(Value::Null)
}

/// `value` with its `entity` field, when the event has one, as entity
/// bits: the primary agent when it is missing, the agent with that id when
/// it is a string.
fn with_entity(
    world: &mut World,
    registration: &TypeRegistration,
    value: Value,
) -> Result<Value, String> {
    let TypeInfo::Struct(info) = registration.type_info() else {
        return Ok(value);
    };
    if info.field("entity").is_none() {
        return Ok(value);
    }
    let mut fields = match value {
        Value::Object(fields) => fields,
        Value::Null => serde_json::Map::new(),
        other => return Ok(other),
    };
    let named = fields.get("entity").cloned().unwrap_or(Value::Null);
    if !named.is_number() {
        let agent = agent_entity(world, &named)?;
        fields.insert("entity".to_owned(), Value::from(agent.to_bits()));
    }
    Ok(Value::Object(fields))
}

/// The agent `named`: the primary agent for `null`, by [`AgentId`] (or a
/// unique prefix of at least 8 characters) for a string, by entity bits
/// for a number.
fn agent_entity(world: &mut World, named: &Value) -> Result<Entity, String> {
    match named {
        Value::Null => {
            let mut state = SystemState::<PrimaryQuery>::new(world);
            let agents = state
                .get(world)
                .map_err(|failure| format!("cannot read the agents: {failure}"))?;
            primary(&agents).ok_or_else(|| "there is no agent".to_owned())
        }
        Value::String(id) => {
            let mut agents = world.query_filtered::<(Entity, &AgentId), With<Agent>>();
            let found: Vec<Entity> = agents
                .iter(world)
                .filter(|(_, agent)| {
                    agent.0 == *id || (id.len() >= 8 && agent.0.starts_with(id.as_str()))
                })
                .map(|(entity, _)| entity)
                .collect();
            match found.as_slice() {
                [agent] => Ok(*agent),
                [] => Err(format!("no agent `{id}`")),
                _ => Err(format!("`{id}` names several agents")),
            }
        }
        Value::Number(bits) => bits
            .as_u64()
            .and_then(Entity::try_from_bits)
            .filter(|entity| world.get::<Agent>(*entity).is_some())
            .ok_or_else(|| format!("{bits} is not an agent's entity")),
        other => Err(format!("{other} does not name an agent")),
    }
}

/// Every agent, for `agents`.
fn agents(world: &mut World) -> Value {
    let mut query = world.query_filtered::<(
        Entity,
        &AgentId,
        Option<&ModelChoice>,
        Has<ActiveTurn>,
        &Spending,
        Option<&SubagentOf>,
        Option<&WorkDir>,
    ), With<Agent>>();
    let ids: Vec<(Entity, String)> = world
        .query::<(Entity, &AgentId)>()
        .iter(world)
        .map(|(entity, id)| (entity, id.0.clone()))
        .collect();
    let parent_id = |parent: Entity| {
        ids.iter()
            .find(|(entity, _)| *entity == parent)
            .map(|(_, id)| id.clone())
    };
    let agents: Vec<Value> = query
        .iter(world)
        .map(|(entity, id, model, busy, spending, parent, dir)| {
            json!({
                "agent": id.0,
                "entity": entity.to_bits(),
                "model": model.map(|model| &model.0),
                "busy": busy,
                "parent": parent.and_then(|of| parent_id(of.0)),
                "directory": dir.map(|dir| &dir.0),
                "spending": to_json(spending),
            })
        })
        .collect();
    Value::Array(agents)
}

/// The reflectable events `trigger` takes, by type path and short name.
fn events(world: &World) -> Value {
    let Some(registry) = world.get_resource::<AppTypeRegistry>() else {
        return Value::Array(Vec::new());
    };
    let registry = registry.read();
    let mut events: Vec<Value> = registry
        .iter()
        .filter(|registration| registration.data::<ReflectEvent>().is_some())
        .map(|registration| {
            let path = registration.type_info().type_path_table();
            json!({"path": path.path(), "name": path.short_path()})
        })
        .collect();
    events.sort_by_key(|event| event.get("path").map(Value::to_string));
    Value::Array(events)
}
