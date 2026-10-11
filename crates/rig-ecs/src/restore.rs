//! Restoring a session from its agent logs at startup, then one reconcile
//! pass that settles what a crash or a restart left half done, with
//! ordinary records, so the next restore finds nothing left to fix. Each
//! log is read as [`read_session`](super::journal::read_session) reads it,
//! a torn last line cut off; a retired
//! agent and the agents it spawned are not restored.
//!
//! A call left without a result whose tool may change something is
//! answered as interrupted; the work an agent carries on is spawned as
//! ordinary stage entities marked [`Restored`] and gated like any, as
//! nothing held or approved before the restart is kept. A plugin re-arms
//! its own obligations from observers of the saved components the restore
//! inserts (`On<Insert<MyWait>>`, before [`SessionRestored`] exists):
//!
//! ```
//! use rig_ecs::prelude::*;
//!
//! #[derive(Component)]
//! struct Paused;
//!
//! // A paused agent does not carry its work on after a restart.
//! fn stay_paused(
//!     turns: Query<(Entity, &TurnOf), (With<Restored>, Without<TurnOutcome>)>,
//!     paused: Query<(), With<Paused>>,
//!     mut commands: Commands,
//! ) {
//!     for (turn, &TurnOf(agent)) in &turns {
//!         if paused.contains(agent) {
//!             let stopped = TurnOutcome::Stopped(Some("restart".to_owned()));
//!             commands.entity(turn).insert(stopped);
//!         }
//!     }
//! }
//!
//! App::new().add_systems(AgentPass, stay_paused.in_set(Stage::Gate));
//! ```

use std::collections::{HashMap, HashSet};
use std::error::Error;

use bevy_ecs::prelude::*;
use bevy_ecs::reflect::{AppTypeRegistry, ReflectComponent};
use bevy_log::warn;
use bevy_reflect::prelude::ReflectDefault;
use bevy_reflect::serde::TypedReflectDeserializer;
use bevy_reflect::{Reflect, ReflectFromReflect, TypeRegistry};
use rig_cassette::journal::{JournalStore, LogLines};
use rig_core::message::ToolCall;
use rig_core::transcript::{close_pending_with, pending_calls};
use serde::de::DeserializeSeed;
use serde_json::Value;

use super::agent::{Agent, AgentId, CallOf, Conversation, Notice, SpawnedBy, Turn, TurnOf};
use super::effects::Effects;
use super::journal::{
    AgentLog, Commit, Folded, ReflectSaved, SESSION, SessionLog, SessionRestored, SessionStore,
    fold, load_images,
};
use super::stage::{Effect, ForAgent};
use super::tools::{Declared, Footprint, Serves, ToolDef, ToolRequest, footprint_of};
use super::turn::TurnCall;

/// The result of a call that may change something and was running when
/// the session stopped.
const INTERRUPTED: &str = "interrupted by a restart; it may have partly run";

/// The agents [`restore_session`] spawned, each with its depth, until
/// [`reconcile`] took them.
#[derive(Resource)]
pub(crate) struct RestoredAgents(Vec<(usize, Entity)>);

/// On the work a restore carries on for an agent: its turn, the turn's
/// read-only calls that run again and its model calls. Gate sees each as it
/// sees new work; a turn stopped there
/// (`TurnOutcome::Stopped(Some("restart"))`) starts none of it, and its
/// calls are answered as stopped.
#[derive(Component, Reflect, Clone, Copy, Debug, Default)]
#[reflect(Component, Default)]
pub struct Restored;

/// Spawns the agents of the session's logs, with their conversations and
/// saved components, links each agent to the agent that
/// spawned it, restores the saved resources, and starts logging. Anything that does not load is skipped
/// with a notice.
pub(crate) fn restore_session(world: &mut World) {
    let (Some(log), Some(SessionStore(store))) = (
        world.get_resource::<SessionLog>().cloned(),
        world.get_resource::<SessionStore>().cloned(),
    ) else {
        return;
    };
    let mut notices: Vec<String> = Vec::new();
    let mut folded: Vec<Folded> = Vec::new();
    let agents = store.agents().unwrap_or_else(|failure| {
        notices.push(format!("Could not list the agent logs: {failure}."));
        Vec::new()
    });
    for agent in agents {
        match read_log(&*store, &agent) {
            Ok(agent) => folded.push(agent),
            Err(failure) => notices.push(format!(
                "Could not restore the log of agent {agent}: {failure}."
            )),
        }
    }
    let next_n = folded.iter().map(|agent| agent.next_n).max();
    let retired = retired(&folded);
    folded.retain(|agent| !retired.contains(&agent.header.agent));
    let registry = world.get_resource::<AppTypeRegistry>().cloned();
    let registry = registry.unwrap_or_default();
    let registry = registry.read();
    let mut entities = HashMap::new();
    let mut links = Vec::new();
    let mut logs = Vec::new();
    let mut session = None;
    for agent in folded {
        let log = AgentLog {
            next_seq: agent.next_seq,
            started: true,
            components: agent.components,
            ..AgentLog::default()
        };
        let Folded {
            header,
            conversation,
            ..
        } = agent;
        if header.agent == SESSION {
            for (path, value) in &log.components {
                let restored = saved(path, value, &registry).and_then(|(component, value)| {
                    let mut resource = resource_entity(world, component)?;
                    component.insert(&mut resource, value.as_partial_reflect(), &registry);
                    Ok(())
                });
                if let Err(failure) = restored {
                    notices.push(format!("Skipped saved resource `{path}`: {failure}."));
                }
            }
            session = Some((header.agent, log));
            continue;
        }
        let id = AgentId(header.agent.clone());
        let mut spawned = world.spawn((Name::new("agent"), Agent, id.clone(), conversation));
        for (path, value) in &log.components {
            let restored = saved(path, value, &registry).map(|(component, value)| {
                component.insert(&mut spawned, value.as_partial_reflect(), &registry);
            });
            if let Err(failure) = restored {
                notices.push(format!("Skipped saved component `{path}`: {failure}."));
            }
        }
        let entity = spawned.id();
        entities.insert(id.0.clone(), entity);
        links.push((entity, header.parent));
        logs.push((entity, id.0, log));
    }
    for (child, parent) in links {
        if let Some(&parent) = parent.and_then(|parent| entities.get(&parent))
            && let Ok(mut child) = world.get_entity_mut(child)
        {
            child.insert(SpawnedBy(parent));
        }
    }
    let mut parents = world.query::<&SpawnedBy>();
    let parents = parents.query(world);
    let bound = logs.len();
    let restored = logs
        .iter_mut()
        .map(|(entity, _, log)| {
            // A loop cannot happen, but a bound costs nothing.
            log.depth = parents.iter_ancestors(*entity).take(bound).count();
            (log.depth, *entity)
        })
        .collect();
    // What restoring set off, such as connecting each model, is not logged.
    world.flush();
    let logs = logs.into_iter().map(|(_, id, log)| (id, log));
    log.resume(
        store,
        logs.chain(session).collect(),
        next_n.unwrap_or_default(),
    );
    world.insert_resource(RestoredAgents(restored));
    world.insert_resource(SessionRestored);
    for notice in notices {
        world.write_message(Notice::error(None, notice));
    }
}

/// The saved component of the type path `path` from its logged `value`,
/// with the reflection that inserts it, in place of an immutable one too.
fn saved<'r>(
    path: &str,
    value: &Value,
    registry: &'r TypeRegistry,
) -> Result<(&'r ReflectComponent, Box<dyn Reflect>), String> {
    let registration = registry
        .get_with_type_path(path)
        .filter(|registration| registration.data::<ReflectSaved>().is_some())
        .ok_or("no plugin saves it in this build")?;
    let component = registration.data::<ReflectComponent>();
    let component = component.ok_or("it is not a reflected component")?;
    let reflected = TypedReflectDeserializer::new(registration, registry)
        .deserialize(value)
        .map_err(|failure| format!("its saved value no longer fits: {failure}"))?;
    let value = registration
        .data::<ReflectFromReflect>()
        .and_then(|from| from.from_reflect(&*reflected))
        .ok_or("its saved value no longer fits")?;
    Ok((component, value))
}

/// The entity of the resource `component` reflects: the one its plugin
/// inserted, else a new one, which the resource's insert makes its own.
fn resource_entity<'w>(
    world: &'w mut World,
    component: &ReflectComponent,
) -> Result<EntityWorldMut<'w>, String> {
    let id = component.register_component(world);
    let entity = match world.resource_entities().get(id) {
        Some(entity) => entity,
        None => world.spawn_empty().id(),
    };
    world
        .get_entity_mut(entity)
        .map_err(|failure| failure.to_string())
}

/// Reads and folds the log of `agent` in `store`, cutting off a torn last
/// line there first. Images come back from the store's blobs.
fn read_log(store: &dyn JournalStore, agent: &str) -> Result<Folded, Box<dyn Error>> {
    let log = LogLines::read(store, agent)?;
    if log.torn() {
        store.truncate(agent, log.len())?;
    }
    let mut folded = fold(&log)?;
    load_images(&mut folded, store);
    Ok(folded)
}

/// The ids of the agents a restore skips: those retired, and the agents
/// they spawned, all the way down.
fn retired(agents: &[Folded]) -> HashSet<String> {
    let parents: HashMap<&str, Option<&str>> = agents
        .iter()
        .map(|agent| (agent.header.agent.as_str(), agent.header.parent.as_deref()))
        .collect();
    let gone: HashSet<&str> = agents
        .iter()
        .filter(|agent| agent.retired)
        .map(|agent| agent.header.agent.as_str())
        .collect();
    let mut skipped = HashSet::new();
    for agent in agents {
        let mut at = Some(agent.header.agent.as_str());
        // A loop cannot happen, but a bound costs nothing.
        for _ in 0..=agents.len() {
            let Some(id) = at else {
                break;
            };
            if gone.contains(id) {
                skipped.insert(agent.header.agent.clone());
                break;
            }
            at = parents.get(id).copied().flatten();
        }
    }
    skipped
}

/// Settles what the restored agents left half done, parents before the
/// agents they spawned: by appending records, a tool call without a
/// result whose tool may change something is answered as interrupted, and
/// the agent carries on with [`Restored`] work: a turn that runs again the
/// calls of read-only tools, or that calls the model when the conversation
/// ends in the user's message that was not halted, or in a full set of
/// tool results.
pub(crate) fn reconcile(world: &mut World) {
    let Some(RestoredAgents(mut agents)) = world.remove_resource::<RestoredAgents>() else {
        return;
    };
    agents.sort_by_key(|&(depth, _)| depth);
    for (_, agent) in agents {
        if let Err(error) = world.run_system_cached_with(settle, agent) {
            warn!("could not reconcile a restored agent: {error}");
        }
    }
}

/// Settles one restored agent, as [`reconcile`] says. The calls run again
/// are a reply of a fresh effect id, as the model call that asked for them
/// is gone.
fn settle(
    In(agent): In<Entity>,
    mut agents: Query<&mut Conversation>,
    tools: Query<(&ToolDef, &Footprint, Has<Declared>, &Serves)>,
    effects: Res<Effects>,
    mut commit: Commit,
    mut commands: Commands,
) {
    let Ok(mut conversation) = agents.get_mut(agent) else {
        return;
    };
    // A call runs again when it only reads, as its tool says of it.
    let reruns = |call: &ToolCall| {
        let name = call.function.name.as_str();
        let tool = tools.iter().find(|(def, ..)| def.0.name.as_str() == name);
        let handler = tool.and_then(|(.., serves)| match serves {
            Serves::Handler(handler) => Some(handler),
            Serves::Open(_) => None,
        });
        let tool = tool.map(|(_, footprint, declared, _)| (footprint, declared));
        footprint_of(call, handler, tool).0.reads_only()
    };
    let (reruns, interrupted): (Vec<_>, Vec<_>) = pending_calls(conversation.messages())
        .into_iter()
        .partition(reruns);
    if !interrupted.is_empty() {
        let results = close_pending_with(&interrupted, INTERRUPTED);
        commit.message(agent, &mut conversation, results);
    }
    if !reruns.is_empty() {
        let reply = effects.next_id();
        let state = Turn {
            reply: Some(reply),
            ..Turn::default()
        };
        let turn = commands.spawn((TurnOf(agent), state, Restored)).id();
        for call in reruns {
            commands.spawn((
                ToolRequest { call },
                Effect::child_of(Some(reply)),
                CallOf(turn),
                ForAgent(agent),
                Restored,
            ));
        }
    } else if conversation.awaits_model() {
        let turn = commands.spawn((TurnOf(agent), Restored)).id();
        commands.spawn((TurnCall, CallOf(turn), ForAgent(agent), Restored));
    }
}

#[cfg(test)]
mod tests;
