//! The mirror: a session's [`EventRecord`]s applied to entities that carry
//! the kernel's own components, so the views read a mirrored agent as they
//! read a local one. An agent is an [`Agent`] with its [`AgentId`] and
//! [`Conversation`], its running turn a [`TurnOf`] it with the turn's
//! [`Calls`](rig_ecs::agent::Calls): the reply streaming in as a
//! [`Partial`], each tool call as a [`ToolRequest`] that has [`Started`],
//! with its [`ToolProgress`]. The conversation changes only by the
//! records its agent saved (`conversation.change`), applied in order, each
//! once.

use std::collections::HashMap;
use std::time::SystemTime;

use bevy_ecs::prelude::*;
use bevy_log::warn;
use rig_core::conversation::ChangeRecord;
use rig_core::message::{CallId, ToolCall, ToolFunction, ToolName};
use rig_ecs::agent::{Agent, AgentId, CallOf, Conversation, Notice, Partial, TurnOf};
use rig_ecs::restore::Restored;
use rig_ecs::stage::{ForAgent, Settled, Started};
use rig_ecs::tools::ToolRequest;
use rig_protocol::event::{self, Event};
use rig_protocol::{EventRecord, Seq};

use crate::ask::{Ask, Choice};
use crate::progress::ToolProgress;
use crate::status::{AppStatus, StatusEvent};

/// On a mirror entity: the agent it mirrors.
#[derive(Component, Clone, Debug)]
pub struct Mirrored {
    /// The agent's id in its session.
    pub agent: rig_protocol::AgentId,
}

/// What the mirror knows of the session: which entity mirrors what, and
/// the last event applied.
#[derive(Resource, Default, Debug)]
pub struct Mirror {
    /// The last event applied; a repeat or an older one is skipped.
    pub last: Seq,
    agents: HashMap<String, Entity>,
    /// Per agent, the last conversation record applied, by its log.
    applied: HashMap<String, u64>,
    /// Per agent, its running turn's entity, once its conversation opened
    /// the turn.
    turns: HashMap<String, Entity>,
    /// Agents whose turn started but whose conversation has not opened
    /// it yet.
    starting: Vec<String>,
    /// Per agent, the reply streaming in and its message id.
    streams: HashMap<String, (String, Entity)>,
    /// Running tool calls by id, with their agent.
    calls: HashMap<String, (String, Entity)>,
    /// The agent a session shows before its worker named it: what the
    /// user types to a new session goes to it, and the first agent the
    /// session names is it.
    provisional: Option<Entity>,
    /// Open questions by id.
    asks: HashMap<String, Entity>,
    /// The session's driver, as the server last said.
    pub driver: Option<rig_protocol::ClientId>,
}

impl Mirror {
    /// The entity mirroring agent `id`, if any.
    pub fn agent(&self, id: &str) -> Option<Entity> {
        self.agents.get(id).copied()
    }
}

/// Applies `records` in order to the mirror in `world`, skipping any at or
/// before the last applied.
pub fn apply(world: &mut World, records: impl IntoIterator<Item = EventRecord>) {
    world.init_resource::<Mirror>();
    world.init_resource::<AppStatus>();
    for record in records {
        if record.seq <= world.resource::<Mirror>().last {
            continue;
        }
        world.resource_mut::<Mirror>().last = record.seq;
        if record.kind == StatusEvent::KIND {
            status(world, &record);
        }
        match Event::decode(&record) {
            Some(event) => apply_event(world, event),
            None if is_builtin(&record.kind) => {
                warn!(
                    "event {} {} does not fit its kind",
                    record.seq.0, record.kind
                );
            }
            // A plugin's own kind: a view plugin reads it as a message.
            None => {}
        }
        world.write_message(Applied(record));
    }
    start_turns(world);
    attach_calls(world);
}

/// Each event the mirror applied, for a view plugin that reads its own
/// kinds (`x/<plugin>/<name>`) or the record itself.
#[derive(Message, Clone, Debug)]
pub struct Applied(pub EventRecord);

fn is_builtin(kind: &str) -> bool {
    Event::KINDS.contains(&kind)
}

fn apply_event(world: &mut World, event: Event) {
    match event {
        Event::SessionStarted(_) => {
            let mirror = world.resource::<Mirror>();
            if mirror.agents.is_empty() && mirror.provisional.is_none() {
                let mirrored = Mirrored {
                    agent: rig_protocol::AgentId(String::new()),
                };
                let entity = world.spawn((Agent, AgentId(String::new()), mirrored)).id();
                world.resource_mut::<Mirror>().provisional = Some(entity);
            }
        }
        Event::ConversationChange(change) => conversation_change(world, change),
        Event::TurnStarted(started) => {
            let id = started.agent_id.0;
            agent_entity(world, &id);
            let mut mirror = world.resource_mut::<Mirror>();
            if !mirror.turns.contains_key(&id) && !mirror.starting.contains(&id) {
                mirror.starting.push(id);
            }
        }
        Event::TurnEnded(ended) => end_turn(world, &ended.agent_id.0),
        Event::MessageDelta(delta) => stream(world, delta),
        // The reply's final text: its stream (named by a stream id of its
        // own) is done, and its entry comes as the conversation's record.
        Event::Message(message) if message.role == "assistant" => {
            let stream = world
                .resource_mut::<Mirror>()
                .streams
                .remove(&message.agent_id.0);
            if let Some((_, entity)) = stream {
                despawn(world, entity);
            }
        }
        Event::CallStarted(started) => call_started(world, started),
        Event::CallProgress(progress) => {
            let call = world
                .resource::<Mirror>()
                .calls
                .get(&progress.call_id)
                .cloned();
            if let Some((_, entity)) = call {
                let line = progress
                    .text
                    .lines()
                    .next_back()
                    .unwrap_or_default()
                    .to_owned();
                world.entity_mut(entity).insert(ToolProgress(line));
            }
        }
        Event::CallEnded(ended) => {
            let call = world.resource_mut::<Mirror>().calls.remove(&ended.call_id);
            if let Some((_, entity)) = call {
                despawn(world, entity);
            }
        }
        Event::WorkerExited(exited) => {
            // A reply streaming when the worker stopped is never finished:
            // its saved part comes again as the restarted worker's records.
            let streams: Vec<_> = world.resource_mut::<Mirror>().streams.drain().collect();
            for (_, (_, entity)) in streams {
                despawn(world, entity);
            }
            if !exited.restarting {
                notice(world, Notice::warn(None, "The agent stopped."));
            }
        }
        Event::Notice(said) => {
            let text = said.text;
            let note = match said.level.as_str() {
                "error" => Notice::error(None, text),
                "warn" | "warning" => Notice::warn(None, text),
                _ => Notice::info(None, text),
            };
            notice(world, note);
        }
        Event::SessionFaulted(faulted) => {
            let text = format!("The session faulted: {}.", faulted.reason);
            notice(world, Notice::error(None, text));
        }
        Event::SessionStopped(_) => {
            let agents: Vec<_> = world.resource::<Mirror>().turns.keys().cloned().collect();
            for agent in agents {
                end_turn(world, &agent);
            }
            notice(world, Notice::info(None, "The session stopped."));
        }
        Event::AskOpened(opened) => ask_opened(world, opened),
        Event::AskResolved(resolved) => {
            let ask = world
                .resource_mut::<Mirror>()
                .asks
                .remove(&resolved.ask_id.0);
            if let Some(entity) = ask {
                despawn(world, entity);
            }
        }
        Event::DriverChanged(changed) => {
            world.resource_mut::<Mirror>().driver = changed.driver;
        }
        Event::ReloadFailed(failed) => {
            let text = format!("Reload failed: {}", failed.error);
            notice(world, Notice::error(None, text));
        }
        Event::BudgetExhausted(_) => {
            let text = "The session's budget is spent; its turns were stopped.";
            notice(world, Notice::warn(None, text));
        }
        _ => {}
    }
}

fn notice(world: &mut World, notice: Notice) {
    world.write_message(notice);
}

/// The entity mirroring agent `id`, spawned the first time it is named.
fn agent_entity(world: &mut World, id: &str) -> Entity {
    if let Some(entity) = world.resource::<Mirror>().agent(id) {
        return entity;
    }
    let mirrored = Mirrored {
        agent: rig_protocol::AgentId(id.to_owned()),
    };
    let entity = world.spawn((Agent, AgentId(id.to_owned()), mirrored)).id();
    world
        .resource_mut::<Mirror>()
        .agents
        .insert(id.to_owned(), entity);
    entity
}

fn conversation_change(world: &mut World, change: event::ConversationChange) {
    let id = change.agent_id.0;
    let agent = agent_entity(world, &id);
    let key = format!("{id}\u{0}{}", change.log);
    let last = world.resource::<Mirror>().applied.get(&key).copied();
    if last.is_some_and(|last| change.index <= last) {
        return;
    }
    let record: ChangeRecord = match serde_json::from_value(change.record) {
        Ok(record) => record,
        Err(why) => {
            warn!(
                "conversation record {} of {id} unreadable: {why}",
                change.index
            );
            return;
        }
    };
    let Some(mut conversation) = world.get_mut::<Conversation>(agent) else {
        return;
    };
    match conversation.apply(&record) {
        Ok(()) => {
            world
                .resource_mut::<Mirror>()
                .applied
                .insert(key, change.index);
        }
        Err(why) => {
            warn!(
                "conversation record {} of {id} does not apply: {why:?}",
                change.index
            );
        }
    }
}

/// Spawns the turn of each agent that started one, once its conversation
/// opened it: the turn is spawned as restored, so it takes the open record
/// its agent saved rather than opening one of its own.
fn start_turns(world: &mut World) {
    let starting = std::mem::take(&mut world.resource_mut::<Mirror>().starting);
    let mut waiting = Vec::new();
    for id in starting {
        let Some(agent) = world.resource::<Mirror>().agent(&id) else {
            continue;
        };
        let open = world
            .get::<Conversation>(agent)
            .is_some_and(|conversation| {
                let last = conversation.turns().next_back();
                last.is_some_and(|(_, record)| record.end.is_none())
            });
        if !open {
            waiting.push(id);
            continue;
        }
        let mirrored = Mirrored {
            agent: rig_protocol::AgentId(id.clone()),
        };
        let turn = world.spawn((TurnOf(agent), Restored, mirrored)).id();
        world.resource_mut::<Mirror>().turns.insert(id, turn);
    }
    world.resource_mut::<Mirror>().starting = waiting;
}

/// Puts each stream and call whose agent's turn runs now among that
/// turn's calls.
fn attach_calls(world: &mut World) {
    let mirror = world.resource::<Mirror>();
    let streams = mirror
        .streams
        .iter()
        .map(|(agent, (_, entity))| (agent, *entity));
    let calls = mirror
        .calls
        .values()
        .map(|(agent, entity)| (agent, *entity));
    let attach: Vec<_> = streams
        .chain(calls)
        .filter_map(|(agent, entity)| Some((entity, *mirror.turns.get(agent)?)))
        .collect();
    for (entity, turn) in attach {
        if world.get::<CallOf>(entity).is_none_or(|of| of.0 != turn) {
            world.entity_mut(entity).insert(CallOf(turn));
        }
    }
}

fn end_turn(world: &mut World, id: &str) {
    let mut mirror = world.resource_mut::<Mirror>();
    mirror.starting.retain(|starting| starting != id);
    let turn = mirror.turns.remove(id);
    let stream = mirror.streams.remove(id);
    let calls: Vec<_> = mirror
        .calls
        .iter()
        .filter(|(_, (agent, _))| agent == id)
        .map(|(call, (_, entity))| (call.clone(), *entity))
        .collect();
    for (call, _) in &calls {
        mirror.calls.remove(call);
    }
    for (_, entity) in calls {
        despawn(world, entity);
    }
    if let Some((_, entity)) = stream {
        despawn(world, entity);
    }
    if let Some(turn) = turn {
        despawn(world, turn);
    }
}

fn stream(world: &mut World, delta: event::MessageDelta) {
    let id = delta.agent_id.0.clone();
    let agent = agent_entity(world, &id);
    let current = world.resource::<Mirror>().streams.get(&id).cloned();
    let entity = match current {
        Some((message, entity)) if message == delta.message_id => entity,
        old => {
            if let Some((_, entity)) = old {
                despawn(world, entity);
            }
            let mirrored = Mirrored {
                agent: delta.agent_id.clone(),
            };
            let entity = world
                .spawn((Partial::default(), ForAgent(agent), mirrored))
                .id();
            let stream = (delta.message_id.clone(), entity);
            world.resource_mut::<Mirror>().streams.insert(id, stream);
            entity
        }
    };
    if let Some(mut partial) = world.get_mut::<Partial>(entity) {
        if let Some(text) = &delta.text {
            partial.text.push_str(text);
        }
        if let Some(reasoning) = &delta.reasoning {
            partial.reasoning.push_str(reasoning);
        }
    }
}

fn call_started(world: &mut World, started: event::CallStarted) {
    let id = started.agent_id.0.clone();
    let agent = agent_entity(world, &id);
    let Ok(name) = ToolName::try_from(started.tool.clone()) else {
        warn!("call {} has no tool name", started.call_id);
        return;
    };
    let call_id = serde_json::from_value(serde_json::Value::String(started.call_id.clone()))
        .unwrap_or_else(|_| CallId::from_wire(started.call_id.clone()));
    let call = ToolCall::new(call_id, ToolFunction::new(name, started.args));
    let mirrored = Mirrored {
        agent: started.agent_id.clone(),
    };
    let request = ToolRequest { call };
    let at = SystemTime::now();
    let entity = world
        .spawn((request, Started { at }, ForAgent(agent), mirrored))
        .id();
    let old = world
        .resource_mut::<Mirror>()
        .calls
        .insert(started.call_id, (id, entity));
    if let Some((_, old)) = old {
        despawn(world, old);
    }
}

/// A `status` event's items, on the agent, its turn or the app.
fn status(world: &mut World, record: &EventRecord) {
    let Ok(event) = serde_json::from_value::<StatusEvent>(record.data.clone()) else {
        warn!("status {} unreadable", record.seq.0);
        return;
    };
    let Some(id) = event.agent_id else {
        world.resource_mut::<AppStatus>().0 = event.items;
        return;
    };
    let agent = agent_entity(world, &id);
    let target = match event.turn {
        true => world.resource::<Mirror>().turns.get(&id).copied(),
        false => Some(agent),
    };
    if let Some(target) = target {
        world.entity_mut(target).insert(event.items);
    }
}

/// On a mirrored question: its id in the session, which an answer names,
/// and the kind of each of its options (ACP's: `allow_once`,
/// `reject_once`, ...), in the order of its choices.
#[derive(Component, Clone, Debug)]
pub struct MirroredAsk {
    pub id: rig_protocol::AskId,
    pub kinds: Vec<String>,
}

fn ask_opened(world: &mut World, opened: event::AskOpened) {
    let agent = agent_entity(world, &opened.agent_id.0);
    let choices = opened.options.iter();
    let choices = choices.map(|option| Choice::new(option.option_id.clone(), option.label.clone()));
    let ask = Ask {
        title: opened.title,
        body: opened.body,
        choices: choices.collect(),
        free_text: opened.kind == event::AskKind::Question,
        initial: None,
        multiline: false,
        inline: false,
        multiple: false,
    };
    let mirrored = Mirrored {
        agent: opened.agent_id,
    };
    let id = opened.ask_id.0.clone();
    let kinds = opened
        .options
        .into_iter()
        .map(|option| option.kind)
        .collect();
    let asked = MirroredAsk {
        id: opened.ask_id,
        kinds,
    };
    let entity = world.spawn((ask, ForAgent(agent), mirrored, asked)).id();
    let old = world.resource_mut::<Mirror>().asks.insert(id, entity);
    if let Some(old) = old {
        despawn(world, old);
    }
}

/// Despawns a mirror entity, settled first so an effect's going is not
/// recorded as a cancellation.
fn despawn(world: &mut World, entity: Entity) {
    if let Ok(mut gone) = world.get_entity_mut(entity) {
        gone.insert(Settled);
        gone.despawn();
    }
}

/// Forgets every mirrored entity, so the session's events can be applied
/// again from the start.
pub fn clear(world: &mut World) {
    let mirror = world.remove_resource::<Mirror>().unwrap_or_default();
    let entities = (mirror.asks.into_values())
        .chain(mirror.provisional)
        .chain(mirror.calls.into_values().map(|(_, entity)| entity))
        .chain(mirror.streams.into_values().map(|(_, entity)| entity))
        .chain(mirror.turns.into_values())
        .chain(mirror.agents.into_values());
    for entity in entities.collect::<Vec<_>>() {
        despawn(world, entity);
    }
    world.init_resource::<Mirror>();
}
