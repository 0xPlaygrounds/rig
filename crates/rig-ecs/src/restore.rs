//! Restoring a session from its agent logs at startup, then one reconcile
//! pass that settles what a crash or a restart left half done, with
//! ordinary records, so the next restore finds nothing left to fix. The
//! session is read as [`open_session`] reads it: strictly, refused whole
//! before anything is written when it cannot be read, a torn last line set
//! aside; a retired agent and the agents it spawned are not restored.
//!
//! The work an agent carries on is spawned as ordinary stage entities marked
//! [`Restored`], gated like any: nothing held or approved before the restart
//! is kept. A tool call that had started or settled when the session stopped
//! carries [`Interrupted`], which says how the kernel answers it. The restore
//! commits no answer. A plugin re-arms its own obligations from observers of
//! the saved components the restore inserts (`On<Insert<MyWait>>`, before
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

use std::collections::{BTreeMap, BTreeSet, HashMap};

use bevy_ecs::prelude::*;
use bevy_ecs::reflect::{AppTypeRegistry, ReflectComponent};
use bevy_log::warn;
use bevy_reflect::prelude::ReflectDefault;
use bevy_reflect::serde::TypedReflectDeserializer;
use bevy_reflect::{Reflect, ReflectFromReflect, TypeRegistration, TypeRegistry};
use rig_core::completion::Message;
use rig_core::conversation::{Aside, AsideId, Author};
use rig_core::effect::Decider;
use rig_core::message::{CallId, UserContent};
use rig_core::tool::ToolResult;
use rig_memory::Session;
use serde::de::DeserializeSeed;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use web_time::SystemTime;

use super::agent::{Agent, AgentId, CallOf, Conversation, Notice, SpawnedBy, Turn, TurnOf};
use super::effects::Effects;
use super::inbox::{Unsent, kept};
use super::session_log::{
    Opened, ReflectSaved, SessionLog, SessionRestored, SessionStore, open_session,
};
use super::stage::{Admitted, Decision, Effect, ForAgent, Holds, Outcome, Settled, Started};
use super::tools::{Footprint, ToolRequest};
use super::turn::{TurnCall, not_finished};

/// Why a call that may change something and was running when the session
/// stopped did not finish, as rig-core's
/// [`not_finished`](rig_core::tool::feedback::not_finished) says it.
pub(crate) const RESTART: &str = "the session restarted";

/// The kind of the aside that marks a call that may change something from
/// its start, and any call once it settled, until its reply's results are
/// in.
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

/// Marks the call `call` settled with `result` on `conversation`: in place
/// of its started mark, else after its last entry; a call marked settled
/// already is left as it is.
pub(crate) fn mark_settled(
    conversation: &mut rig_core::conversation::Conversation,
    call: &CallId,
    result: &ToolResult,
) {
    let had = marks(conversation)
        .into_iter()
        .find(|(_, mark)| mark.call() == call);
    let mark = CallMark::Settled {
        call: call.clone(),
        result: result.clone(),
    };
    let value = serde_json::to_value(mark).unwrap_or_default();
    match had {
        Some((_, CallMark::Settled { .. })) => {}
        Some((id, CallMark::Started { .. })) => {
            if let Some(aside) = conversation.aside(id).cloned() {
                conversation.replace_aside(id, Aside { value, ..aside });
            }
        }
        None => {
            let after = conversation.entries().next_back().map(|entry| entry.id);
            conversation.add_aside(Aside::new(CALL_MARK, Author::rig(), value).after(after));
        }
    }
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

/// On a restored call that had started, or had finished, when the session
/// stopped: what its mark said. The kernel answers it after every gate,
/// [`Started`](Self::Started) with an error saying a restart interrupted
/// it and [`Settled`](Self::Settled) with its result, unless a plugin
/// decided it first, in Gate, or removed this, in Prepare, to let it run
/// again as ordinary work every gate sees. A plugin that must ask someone
/// first holds it ([`HoldOf`](super::stage::HoldOf)) and answers it later;
/// the kernel answers it once no hold is left and no plugin decided it.
/// Not [`ReadyToStart`](super::stage::ReadyToStart): no gate that holds or asks
/// for ordinary work sees it.
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

/// Answers each restored call still [`Interrupted`], undecided and unheld,
/// once every gate ran, as `rig-ecs`.
pub(crate) fn answer_interrupted(
    mut calls: Query<
        (&Interrupted, Option<&Footprint>, &mut Decision),
        (
            With<Admitted>,
            Without<Outcome>,
            Without<Started>,
            Without<Settled>,
            Without<Holds>,
        ),
    >,
) {
    for (interrupted, footprint, mut decision) in &mut calls {
        if decision.is_decided() {
            continue;
        }
        let result = match interrupted {
            Interrupted::Settled(result) => result.clone(),
            Interrupted::Started => not_finished(RESTART, true, footprint),
        };
        decision.decide(Decider::plugin("rig-ecs"), result);
    }
}

/// The agents [`restore_session`] spawned, each with its depth, until
/// [`reconcile`] took them.
#[derive(Resource)]
pub(crate) struct RestoredAgents(Vec<(usize, Entity)>);

/// On the work a restore carries on for an agent: its turn, the turn's
/// calls left without a result and its model calls, and the messages in
/// its inbox. Gate sees each call as it sees new work; a turn stopped there
/// (`TurnOutcome::Stopped(Some("restart"))`) starts none of it, and its
/// calls are answered as stopped. A message is accepted already
/// ([`Started`]): it goes with the agent's next model call, or, to an idle
/// agent, is [`Unsent`].
#[derive(Component, Reflect, Clone, Copy, Debug, Default)]
#[reflect(Component, Default)]
pub struct Restored;

/// Spawns the agents of the session's logs, with their conversations and
/// saved components, links each agent to the agent that spawned it,
/// restores the saved resources, and starts logging. The session is the
/// one [`open_session`] read, or, when only a [`SessionStore`] was
/// inserted, the one this opens; a session it refuses is not restored nor
/// logged to, with a notice. A saved value that does not load is skipped
/// with a notice, and its next save replaces it; what reading repaired is
/// shown as a notice too.
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
        saved: state,
        repairs,
    } = session;
    let mut notices: Vec<Notice> = repairs
        .iter()
        .map(|repair| Notice::warn(None, format!("While resuming the session: {repair}.")))
        .collect();
    let registry = world.get_resource::<AppTypeRegistry>().cloned();
    let registry = registry.unwrap_or_default();
    let registry = registry.read();
    let reader = SavedReader::new(world);
    let skipped = |what: &str, key: &str, failure: String| {
        let skipped =
            format!("Skipped saved {what} `{key}`: {failure}; its next save replaces it.");
        Notice::error(None, skipped)
    };
    for (key, value) in &state {
        let restored = reader
            .read(&registry, key, value)
            .and_then(|(component, value)| {
                let mut resource = resource_entity(world, component)?;
                component.insert(&mut resource, value.as_partial_reflect(), &registry);
                Ok(())
            });
        if let Err(failure) = restored {
            notices.push(skipped("resource", key, failure));
        }
    }
    let mut entities = HashMap::new();
    let mut links = Vec::new();
    for agent in agents {
        let conversation = Conversation(agent.conversation);
        let id = AgentId(agent.id);
        let mut read = Vec::new();
        for (key, value) in &agent.saved {
            match reader.read(&registry, key, value) {
                Ok(value) => read.push(value),
                Err(failure) => notices.push(skipped("component", key, failure)),
            }
        }
        let mut spawned = world.spawn((Name::new("agent"), Agent, id.clone(), conversation));
        for (component, value) in read {
            component.insert(&mut spawned, value.as_partial_reflect(), &registry);
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

/// The stable keys saved types are logged and restored under instead of
/// their type paths, such as `goals.goal` (rig-harness
/// `AppKeepsExt::saved_as`), and which of them are listed: copied into the
/// session's listing (rig-memory's `SessionMeta::listed`) as they change.
#[derive(Resource, Default, Debug, Clone, PartialEq, Eq)]
pub struct SavedKeys {
    /// Each key by type path.
    keys: BTreeMap<String, String>,
    /// Each type path by key.
    paths: BTreeMap<String, String>,
    /// The listed keys.
    listed: BTreeSet<String>,
}

impl SavedKeys {
    /// Saves the type of path `path` under `key`; again for the same pair
    /// it does nothing.
    ///
    /// # Errors
    ///
    /// Recording nothing, when another type has `key` or the type has
    /// another key.
    pub fn add(&mut self, path: &str, key: &str) -> Result<(), String> {
        match (self.paths.get(key), self.keys.get(path)) {
            (Some(taken), _) if taken != path => {
                Err(format!("the saved key `{key}` is taken by `{taken}`"))
            }
            (_, Some(other)) if other != key => {
                Err(format!("`{path}` is saved under `{other}` already"))
            }
            _ => {
                self.keys.insert(path.to_owned(), key.to_owned());
                self.paths.insert(key.to_owned(), path.to_owned());
                Ok(())
            }
        }
    }

    /// Lists `key`'s values in the session's listing.
    pub fn list(&mut self, key: &str) {
        self.listed.insert(key.to_owned());
    }

    /// The key the type of path `path` is saved under, when it has one.
    pub fn key(&self, path: &str) -> Option<&str> {
        self.keys.get(path).map(String::as_str)
    }

    /// The path of the type saved under `key`, when one is.
    pub fn path(&self, key: &str) -> Option<&str> {
        self.paths.get(key).map(String::as_str)
    }

    /// Whether `key`'s values are listed.
    pub fn is_listed(&self, key: &str) -> bool {
        self.listed.contains(key)
    }

    /// The listed keys.
    pub fn listed(&self) -> &BTreeSet<String> {
        &self.listed
    }
}

/// The one reader of saved values, for a restore: each value is read under
/// its key (a stable key, or a type path) and reflected as its type.
pub(crate) struct SavedReader {
    keys: SavedKeys,
}

impl SavedReader {
    /// A reader with the saved keys of `world`.
    pub(crate) fn new(world: &World) -> Self {
        Self {
            keys: world
                .get_resource::<SavedKeys>()
                .cloned()
                .unwrap_or_default(),
        }
    }

    /// The saved value `saved` of `key`, with the reflection that inserts
    /// it.
    ///
    /// # Errors
    ///
    /// Why it is not read: it no longer fits its type, or no plugin saves
    /// its type.
    pub(crate) fn read<'r>(
        &self,
        registry: &'r TypeRegistry,
        key: &str,
        saved: &Value,
    ) -> Result<(&'r ReflectComponent, Box<dyn Reflect>), String> {
        let (registration, component) = saved_type(self.path(key), registry)?;
        Ok((component, reflected(registration, saved, registry)?))
    }

    /// The type path of the values saved under `key`.
    pub(crate) fn path<'k>(&'k self, key: &'k str) -> &'k str {
        self.keys.path(key).unwrap_or(key)
    }
}

/// `value` reflected as the type of `registration`.
fn reflected(
    registration: &TypeRegistration,
    value: &Value,
    registry: &TypeRegistry,
) -> Result<Box<dyn Reflect>, String> {
    let reflected = TypedReflectDeserializer::new(registration, registry)
        .deserialize(value)
        .map_err(|failure| format!("its saved value no longer fits: {failure}"))?;
    registration
        .data::<ReflectFromReflect>()
        .and_then(|from| from.from_reflect(&*reflected))
        .ok_or_else(|| "its saved value no longer fits".to_owned())
}

/// The registration of the saved component of the type path
/// `path`, with the reflection that inserts and removes it.
pub(crate) fn saved_type<'r>(
    path: &str,
    registry: &'r TypeRegistry,
) -> Result<(&'r TypeRegistration, &'r ReflectComponent), String> {
    let registration = registry
        .get_with_type_path(path)
        .filter(|registration| registration.data::<ReflectSaved>().is_some())
        .ok_or("no plugin saves it in this build")?;
    let component = registration.data::<ReflectComponent>();
    let component = component.ok_or("it is not a reflected component")?;
    Ok((registration, component))
}

/// The entity of the resource `component` reflects: the one its plugin
/// inserted, else a new one, which the resource's insert makes its own.
pub(crate) fn resource_entity<'w>(
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
        if let Err(error) = world.run_system_cached_with(carry_on_restored, agent) {
            warn!("could not reconcile a restored agent: {error}");
        }
    }
}

/// Carries on one restored agent, as [`reconcile`] says, removes the
/// marks of calls that have a result, and puts back the messages its inbox
/// kept. The calls are a reply of a fresh effect id, as the model call
/// that asked for them is gone.
fn carry_on_restored(
    In(agent): In<Entity>,
    mut agents: Query<&mut Conversation>,
    effects: Res<Effects>,
    mut commands: Commands,
) {
    let Ok(mut conversation) = agents.get_mut(agent) else {
        return;
    };
    let pending = conversation.pending_calls();
    let idle = pending.is_empty() && !conversation.awaits_model();
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
                Effect::with_parent_id(Some(reply)),
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
    let at = SystemTime::now();
    for (mark, message) in kept(&conversation) {
        let mut message =
            commands.spawn((message, ForAgent(agent), Started { at }, Restored, mark));
        if idle {
            message.insert(Unsent { ended: None });
        }
    }
}

#[cfg(test)]
mod tests;
