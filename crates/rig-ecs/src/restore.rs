//! Restoring a session from its agent logs at startup, then one reconcile
//! pass that settles what a crash or a restart left half done. Each log is
//! read from its newest [`Condensed`] record on, with a torn last line cut
//! off and
//! records of unknown types skipped. Each fix the reconcile pass makes is an
//! ordinary record, so the next restore finds nothing left to fix; the
//! request-build repair that answers any call still without a result stays
//! as the backstop.
//!
//! A call left without a result whose tool may change something is
//! answered as interrupted at once; the work an agent carries on is spawned
//! as ordinary stage entities marked [`Restored`]: a turn, the read-only
//! calls it runs again and the model call after them, or its model call
//! when the model owes an answer. They are admitted, prepared and gated
//! like any work, as nothing held or approved before the restart is kept.
//! A Gate system keeps an agent idle by stopping its restored turn, and a
//! plugin re-arms its own obligations from observers of the saved
//! components the restore inserts (`On<Insert<MyWait>>`, before
//! [`SessionRestored`] exists):
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

use std::collections::HashMap;
use std::error::Error;

use bevy_ecs::prelude::*;
use bevy_ecs::reflect::{AppTypeRegistry, ReflectComponent};
use bevy_log::warn;
use bevy_reflect::prelude::ReflectDefault;
use bevy_reflect::serde::TypedReflectDeserializer;
use bevy_reflect::{Reflect, ReflectFromReflect, TypeRegistry};
use rig_cassette::journal::{JournalStore, load_images};
use rig_core::transcript::{close_pending_with, pending_calls};
use serde::Deserialize;
use serde::de::{DeserializeSeed, IgnoredAny};
use serde_json::Value;

use super::agent::{
    Agent, AgentId, CallOf, Condensed, Conversation, Notice, SpawnedBy, Turn, TurnOf,
};
use super::effects::Effects;
use super::journal::{
    AgentLog, Commit, Header, Line, Record, ReflectSaved, SESSION, SessionLog, SessionRestored,
    SessionStore,
};
use super::stage::{Effect, ForAgent};
use super::tools::{Footprint, ToolDef, ToolRequest};
use super::turn::TurnCall;

/// The result of a call that may change something and was running when
/// the session stopped.
const INTERRUPTED: &str = "interrupted by a restart; it may have partly run";

/// The first fields of a line, read from every line to find the newest
/// condensed record and the newest record of each saved component.
#[derive(Deserialize)]
struct Envelope {
    seq: u64,
    #[serde(rename = "type")]
    kind: String,
    component: Option<String>,
}

/// An agent log folded into the agent's state, and the log to carry on.
struct Folded {
    header: Header,
    conversation: Conversation,
    condensed: Option<Condensed>,
    log: AgentLog,
}

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

/// Spawns the agents of the session's logs, with their conversations,
/// condensed summaries and saved components, links each agent to the agent that
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
    let registry = world.get_resource::<AppTypeRegistry>().cloned();
    let registry = registry.unwrap_or_default();
    let registry = registry.read();
    let mut entities = HashMap::new();
    let mut links = Vec::new();
    let mut logs = Vec::new();
    let mut session = None;
    for agent in folded {
        let Folded {
            header,
            conversation,
            condensed,
            log,
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
        if let Some(condensed) = condensed {
            spawned.insert(condensed);
        }
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
    log.resume(store, logs.chain(session).collect());
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
    let bytes = store.read(agent)?;
    let complete = bytes
        .iter()
        .rposition(|byte| *byte == b'\n')
        .map_or(0, |at| at + 1);
    let mut lines: Vec<&[u8]> = bytes
        .get(..complete)
        .unwrap_or_default()
        .split(|byte| *byte == b'\n')
        .collect();
    if lines.last().is_some_and(|line| line.is_empty()) {
        lines.pop();
    }
    let mut keep = complete;
    if let Some(last) = lines.last()
        && serde_json::from_slice::<IgnoredAny>(last).is_err()
    {
        keep = keep.saturating_sub(last.len() + 1);
        lines.pop();
    }
    if keep < bytes.len() {
        store.truncate(agent, u64::try_from(keep)?)?;
    }
    lines.retain(|line| !line.is_empty());
    let header = match lines
        .first()
        .and_then(|first| serde_json::from_slice::<Line>(first).ok())
    {
        Some(Line {
            record: Record::Header(header),
            ..
        }) => header,
        _ => return Err("it does not start with a header".into()),
    };
    let envelopes: Vec<Option<Envelope>> = lines
        .iter()
        .map(|line| serde_json::from_slice(line).ok())
        .collect();
    let next_seq = envelopes.iter().flatten().map(|envelope| envelope.seq + 1);
    let mut folded = Folded {
        header,
        conversation: Conversation::default(),
        condensed: None,
        log: AgentLog {
            next_seq: next_seq.max().unwrap_or(1),
            started: true,
            ..AgentLog::default()
        },
    };
    // Messages are read from the first one the newest condensed record
    // kept.
    let condensed = envelopes
        .iter()
        .rposition(|envelope| matches!(envelope, Some(envelope) if envelope.kind == "condensed"))
        .and_then(|at| match serde_json::from_slice(lines.get(at)?).ok()? {
            Line {
                record:
                    Record::Condensed {
                        summary,
                        first_kept,
                    },
                ..
            } => Some((at, summary, first_kept)),
            _ => None,
        });
    let from = match condensed {
        Some((at, summary, first_kept)) => {
            folded.condensed = Some(Condensed { upto: 0, summary });
            envelopes
                .iter()
                .enumerate()
                .skip(1)
                .find(|(_, envelope)| {
                    matches!(envelope, Some(envelope) if envelope.seq >= first_kept)
                })
                .map_or(at, |(index, _)| index.min(at))
        }
        None => 1,
    };
    for (line, envelope) in lines.iter().zip(&envelopes).skip(from) {
        // Saved components are read below; unknown record types are
        // skipped and left on disk.
        if envelope
            .as_ref()
            .is_some_and(|envelope| envelope.component.is_some())
        {
            continue;
        }
        let Ok(line) = serde_json::from_slice::<Line>(line) else {
            continue;
        };
        match line.record {
            Record::Message { message, origin } => {
                let mut message = message.into_owned();
                if let Err(failure) = load_images(&mut message, store) {
                    warn!("an image agent {agent} logged is gone: {failure}");
                }
                folded.conversation.append(message, origin, Some(line.seq));
            }
            Record::Retract => {
                folded.conversation.retract();
            }
            // Only a conversation waiting for its model is halted, so a
            // halt after a halt means it was asked again, as by a retry.
            Record::Halt { reason } => {
                folded.conversation.resume();
                folded.conversation.halt(reason);
            }
            Record::Header(_) | Record::Component { .. } | Record::Condensed { .. } => {}
        }
    }
    // Only the newest record of each saved component is read, wherever it
    // is in the log.
    let newest: HashMap<&str, usize> = envelopes
        .iter()
        .enumerate()
        .filter_map(|(at, envelope)| Some((envelope.as_ref()?.component.as_deref()?, at)))
        .collect();
    for at in newest.into_values() {
        let line = lines
            .get(at)
            .and_then(|line| serde_json::from_slice(line).ok());
        if let Some(Line {
            record: Record::Component { component, value },
            ..
        }) = line
        {
            folded
                .log
                .components
                .extend(value.map(|value| (component, value)));
        }
    }
    Ok(folded)
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
    tools: Query<(&ToolDef, &Footprint)>,
    effects: Res<Effects>,
    mut commit: Commit,
    mut commands: Commands,
) {
    let Ok(mut conversation) = agents.get_mut(agent) else {
        return;
    };
    // A call runs again when its tool, of any kind, only reads.
    let reruns = |name: &str| {
        tools.iter().any(|(def, footprint)| {
            def.0.name.as_str() == name && *footprint == Footprint::ReadOnly
        })
    };
    let (reruns, interrupted): (Vec<_>, Vec<_>) = pending_calls(conversation.messages())
        .into_iter()
        .partition(|call| reruns(call.function.name.as_str()));
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
