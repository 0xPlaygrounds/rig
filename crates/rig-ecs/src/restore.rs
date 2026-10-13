//! Restoring a session from its agent logs at startup, then one reconcile
//! pass that settles what a crash or a restart left half done, with
//! ordinary records, so the next restore finds nothing left to fix. The
//! session is read as [`open_session`] reads it: strictly, refused whole
//! before anything is written when it cannot be read, a torn last line set
//! aside; a retired agent and the agents it spawned are not restored.
//!
//! The work an agent carries on is spawned as ordinary stage entities
//! marked [`Restored`]: every call left without a result, as one reply,
//! and gated like any, as nothing held or approved before the restart is
//! kept. A call that may change something is marked on its agent's
//! conversation (an aside of kind `rig.call`) before it starts, and the
//! mark replaced by its result when it settles while its reply still runs;
//! the mark goes when the reply's results are in. A restored call with a
//! mark is [`Interrupted`]: no gate that holds or asks sees it, a plugin
//! answers it in Gate, or lets it run again by removing `Interrupted` in
//! Prepare, and the kernel answers it after every gate otherwise. The
//! restore commits no answer; it only removes the marks of calls that have
//! a result. A plugin re-arms its own obligations from observers of the
//! saved components the restore inserts (`On<Insert<MyWait>>`, before
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

use bevy_ecs::prelude::*;
use bevy_ecs::reflect::{AppTypeRegistry, ReflectComponent};
use bevy_log::warn;
use bevy_reflect::prelude::ReflectDefault;
use bevy_reflect::serde::TypedReflectDeserializer;
use bevy_reflect::{Reflect, ReflectFromReflect, TypeRegistry};
use rig_core::completion::Message;
use rig_core::conversation::{Aside, AsideId, Author};
use rig_core::effect::Decider;
use rig_core::message::{CallId, UserContent};
use rig_core::tool::{ToolExecutionError, ToolResult};
use rig_memory::Session;
use serde::de::DeserializeSeed;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::agent::{Agent, AgentId, CallOf, Conversation, Notice, SpawnedBy, Turn, TurnOf};
use super::effects::Effects;
use super::journal::{
    Opened, ReflectSaved, SessionLog, SessionRestored, SessionStore, open_session,
};
use super::stage::{Admitted, Decision, Effect, ForAgent, Outcome, Settled, Started};
use super::tools::ToolRequest;
use super::turn::TurnCall;

/// The result of a call that may change something and was running when
/// the session stopped.
const INTERRUPTED: &str = "interrupted by a restart; it may have partly run";

/// The kind of the aside that marks a call that may change something,
/// from its start until its reply's results are in.
const CALL_MARK: &str = "rig.call";

/// A `rig.call` aside's value.
#[derive(Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
enum CallMark {
    /// The call started.
    Started { call: CallId },
    /// The call settled with `result` while its reply still ran.
    Settled { call: CallId, result: ToolResult },
}

impl CallMark {
    fn call(&self) -> &CallId {
        match self {
            Self::Started { call } | Self::Settled { call, .. } => call,
        }
    }
}

/// The `rig.call` marks of `conversation`.
fn marks(conversation: &rig_core::conversation::Conversation) -> Vec<(AsideId, CallMark)> {
    let marks = conversation
        .asides()
        .filter(|(_, aside)| aside.kind == CALL_MARK);
    marks
        .filter_map(|(id, aside)| Some((id, CallMark::deserialize(&aside.value).ok()?)))
        .collect()
}

/// Marks the call `call` started on `conversation`, after its last entry,
/// in place of a mark it had, such as one a restart let run again.
pub(crate) fn mark_started(conversation: &mut rig_core::conversation::Conversation, call: &CallId) {
    let after = conversation.entries().next_back().map(|entry| entry.id);
    let mark = CallMark::Started { call: call.clone() };
    let value = serde_json::to_value(mark).unwrap_or_default();
    let aside = Aside::new(CALL_MARK, Author::rig(), value).after(after);
    let had = marks(conversation)
        .into_iter()
        .find(|(_, mark)| mark.call() == call);
    if let Some((id, _)) = had {
        conversation.replace_aside(id, aside);
    } else {
        conversation.add_aside(aside);
    }
}

/// Replaces the mark of the call `call`, if it is marked started, with its
/// `result`.
pub(crate) fn mark_settled(
    conversation: &mut rig_core::conversation::Conversation,
    call: &CallId,
    result: &ToolResult,
) {
    let started = marks(conversation).into_iter().find_map(|(id, mark)| {
        matches!(&mark, CallMark::Started { call: marked } if marked == call).then_some(id)
    });
    let Some((id, aside)) = started.and_then(|id| Some((id, conversation.aside(id)?.clone())))
    else {
        return;
    };
    let mark = CallMark::Settled {
        call: call.clone(),
        result: result.clone(),
    };
    let value = serde_json::to_value(mark).unwrap_or_default();
    conversation.replace_aside(id, Aside { value, ..aside });
}

/// Pushes `results`, a message of tool results, to `conversation`, and
/// removes the marks of the calls they answer.
pub(crate) fn commit_results(
    conversation: &mut rig_core::conversation::Conversation,
    results: Message,
) {
    let answered: Vec<CallId> = match &results {
        Message::User { content } => content
            .iter()
            .filter_map(|item| match item {
                UserContent::ToolResult(result) => Some(result.call.clone()),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    };
    conversation.push(results);
    for (id, mark) in marks(conversation) {
        if answered.contains(mark.call()) {
            conversation.remove_aside(id);
        }
    }
}

/// On a restored call that may have changed something before the session
/// stopped: what its mark said. The kernel answers it after every gate,
/// [`Started`](Self::Started) with an error saying a restart interrupted
/// it and [`Settled`](Self::Settled) with its result, unless a plugin
/// decided it first, in Gate, or removed this, in Prepare, to let it run
/// again as ordinary work every gate sees. Not
/// [`Undecided`](super::stage::Undecided): no gate that holds or asks
/// sees it.
///
/// ```
/// use rig_ecs::prelude::*;
///
/// // Answers a restored call itself, with what the call left behind.
/// fn answer(mut calls: Query<(&ToolRequest, &Interrupted, &mut Decision), Without<Outcome>>) {
///     for (request, interrupted, mut decision) in &mut calls {
///         let answer = match interrupted {
///             Interrupted::Settled(result) => result.clone(),
///             Interrupted::Started => ToolResult::skipped("rerun it if it is still needed"),
///         };
///         decision.decide(Decider::plugin("crash-resume"), answer);
///     }
/// }
///
/// App::new().add_systems(AgentPass, answer.in_set(Stage::Gate));
/// ```
#[derive(Component, Clone, Debug)]
pub enum Interrupted {
    /// It started and had no result.
    Started,
    /// It settled with this result while its reply still ran.
    Settled(ToolResult),
}

/// Answers each restored call still [`Interrupted`] and undecided, once
/// every gate ran, as `rig-ecs`.
pub(crate) fn answer_interrupted(
    mut calls: Query<
        (&Interrupted, &mut Decision),
        (
            With<Admitted>,
            Without<Outcome>,
            Without<Started>,
            Without<Settled>,
        ),
    >,
) {
    for (interrupted, mut decision) in &mut calls {
        if decision.is_decided() {
            continue;
        }
        let result = match interrupted {
            Interrupted::Settled(result) => result.clone(),
            Interrupted::Started => ToolResult::failed(ToolExecutionError::other(INTERRUPTED)),
        };
        decision.decide(Decider::plugin("rig-ecs"), result);
    }
}

/// The agents [`restore_session`] spawned, each with its depth, until
/// [`reconcile`] took them.
#[derive(Resource)]
pub(crate) struct RestoredAgents(Vec<(usize, Entity)>);

/// On the work a restore carries on for an agent: its turn, the turn's
/// calls left without a result and its model calls. Gate sees each as it
/// sees new work; a turn stopped there
/// (`TurnOutcome::Stopped(Some("restart"))`) starts none of it, and its
/// calls are answered as stopped.
#[derive(Component, Reflect, Clone, Copy, Debug, Default)]
#[reflect(Component, Default)]
pub struct Restored;

/// Spawns the agents of the session's logs, with their conversations and
/// saved components, links each agent to the agent that spawned it,
/// restores the saved resources, and starts logging. The session is the
/// one [`open_session`] read, or, when only a [`SessionStore`] was
/// inserted, the one this opens; a session it refuses is not restored nor
/// logged to, with a notice. A saved value that does not load is skipped
/// with a notice; what reading repaired is shown as a notice too.
pub(crate) fn restore_session(world: &mut World) {
    let session = match world.remove_resource::<Opened>() {
        Some(Opened(session)) => session,
        None => {
            let Some(SessionStore(store)) = world.remove_resource::<SessionStore>() else {
                return;
            };
            if let Err(failure) = open_session(world, store) {
                let refused = format!(
                    "The session cannot be resumed, so nothing of it is changed and this run is \
                     not saved: {failure}."
                );
                world.write_message(Notice::error(None, refused));
                return;
            }
            match world.remove_resource::<Opened>() {
                Some(Opened(session)) => session,
                None => return,
            }
        }
    };
    let Some(SessionLog(log)) = world.get_resource::<SessionLog>().cloned() else {
        return;
    };
    let Session {
        agents,
        state,
        repairs,
    } = session;
    let mut notices: Vec<Notice> = repairs
        .iter()
        .map(|repair| Notice::warn(None, format!("While resuming the session: {repair}.")))
        .collect();
    let registry = world.get_resource::<AppTypeRegistry>().cloned();
    let registry = registry.unwrap_or_default();
    let registry = registry.read();
    for (path, value) in &state {
        let restored = saved(path, value, &registry).and_then(|(component, value)| {
            let mut resource = resource_entity(world, component)?;
            component.insert(&mut resource, value.as_partial_reflect(), &registry);
            Ok(())
        });
        if let Err(failure) = restored {
            let skipped = format!("Skipped saved resource `{path}`: {failure}.");
            notices.push(Notice::error(None, skipped));
        }
    }
    let mut entities = HashMap::new();
    let mut links = Vec::new();
    for agent in agents {
        let conversation = Conversation(agent.conversation);
        let id = AgentId(agent.id);
        let mut spawned = world.spawn((Name::new("agent"), Agent, id.clone(), conversation));
        for (path, value) in &agent.state {
            let restored = saved(path, value, &registry).map(|(component, value)| {
                component.insert(&mut spawned, value.as_partial_reflect(), &registry);
            });
            if let Err(failure) = restored {
                let skipped = format!("Skipped saved component `{path}`: {failure}.");
                notices.push(Notice::error(None, skipped));
            }
        }
        let entity = spawned.id();
        entities.insert(id.0, entity);
        links.push((entity, agent.parent));
    }
    for (child, parent) in &links {
        if let Some(&parent) = parent.as_ref().and_then(|parent| entities.get(parent))
            && let Ok(mut child) = world.get_entity_mut(*child)
        {
            child.insert(SpawnedBy(parent));
        }
    }
    let mut parents = world.query::<&SpawnedBy>();
    let parents = parents.query(world);
    let bound = links.len();
    let restored = links
        .into_iter()
        .map(|(entity, _)| {
            // A loop cannot happen, but a bound costs nothing.
            let depth = parents.iter_ancestors(entity).take(bound).count();
            (depth, entity)
        })
        .collect();
    // What restoring set off, such as connecting each model, is not logged.
    world.flush();
    log.start();
    world.insert_resource(RestoredAgents(restored));
    world.insert_resource(SessionRestored { repairs });
    for notice in notices {
        world.write_message(notice);
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

/// Carries on what the restored agents left half done, parents before the
/// agents they spawned, with [`Restored`] work: a turn whose reply is every
/// tool call without a result, each marked [`Interrupted`] as its mark
/// says, or that calls the model when the conversation ends in the user's
/// message that was not halted, or in a full set of tool results.
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

/// Carries on one restored agent, as [`reconcile`] says, and removes the
/// marks of calls that have a result. The calls are a reply of a fresh
/// effect id, as the model call that asked for them is gone.
fn settle(
    In(agent): In<Entity>,
    mut agents: Query<&mut Conversation>,
    effects: Res<Effects>,
    mut commands: Commands,
) {
    let Ok(mut conversation) = agents.get_mut(agent) else {
        return;
    };
    let pending = conversation.pending_calls();
    let mut marked = HashMap::new();
    for (id, mark) in marks(&conversation) {
        if pending.iter().any(|call| &call.id == mark.call()) {
            marked.insert(mark.call().clone(), mark);
        } else {
            conversation.remove_aside(id);
        }
    }
    if !pending.is_empty() {
        let reply = effects.next_id();
        let state = Turn {
            reply: Some(reply),
            ..Turn::default()
        };
        let turn = commands.spawn((TurnOf(agent), state, Restored)).id();
        for call in pending {
            let interrupted = marked.remove(&call.id).map(|mark| match mark {
                CallMark::Started { .. } => Interrupted::Started,
                CallMark::Settled { result, .. } => Interrupted::Settled(result),
            });
            let mut restored = commands.spawn((
                ToolRequest { call },
                Effect::child_of(Some(reply)),
                CallOf(turn),
                ForAgent(agent),
                Restored,
            ));
            if let Some(interrupted) = interrupted {
                restored.insert(interrupted);
            }
        }
    } else if conversation.awaits_model() {
        let turn = commands.spawn((TurnOf(agent), Restored)).id();
        commands.spawn((TurnCall, CallOf(turn), ForAgent(agent), Restored));
    }
}

#[cfg(test)]
mod tests;
