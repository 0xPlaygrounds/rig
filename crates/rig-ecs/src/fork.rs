//! Forking an agent: a [`ForkAgent`] request makes a new agent in the same
//! session that goes on by itself from a copy of another's log, where an
//! earlier compaction can be undone. The request gets its [`ForkResult`], then goes.
//!
//! ```
//! use rig_ecs::prelude::*;
//!
//! // Forks the agent at its end, and keeps the fork it made.
//! fn fork(agent: Entity, mut commands: Commands) {
//!     let keep = |made: On<Add<ForkResult>>, results: Query<&ForkResult>, mut commands: Commands| {
//!         if let Ok(ForkResult(Ok(fork))) = results.get(made.entity) {
//!             commands.entity(*fork).insert(Name::new("fork"));
//!         }
//!     };
//!     commands.spawn(ForkAgent::new(agent, None)).observe(keep);
//! }
//! ```

use bevy_ecs::prelude::*;
use bevy_ecs::reflect::{AppTypeRegistry, ReflectComponent};
use bevy_reflect::PartialReflect;
use bevy_reflect::prelude::*;
use rig_core::conversation::MessageId;
use rig_memory::SessionError;
use serde::{Deserialize, Serialize};

use super::agent::{Agent, AgentId, CacheKey, ContextStale, Conversation, LastUsage};
use super::model::Connection;
use super::requests::RequestOf;
use super::restore::{SavedKeys, saved_type};
use super::session_log::{Fork, ReflectSaved, SessionLog, save_conversation};

/// A request to fork the agent `agent`: a new agent in the same session
/// whose log is a copy of `agent`'s up to a turn boundary at or before
/// `through` (its end with `None`; rig-memory's `Writer::fork_log`). A user
/// message that asked for a turn is at no boundary, so a fork through it
/// ends just before it. The fork has `agent`'s saved components as their
/// fork rules say ([`ReflectSaved`]), its connection and prompt cache
/// ([`CacheKey`]), a [`ForkedFrom`], and no parent. A fork of `agent`'s
/// whole conversation has its [`LastUsage`]; one cut earlier is
/// [`ContextStale`] until its first reply. The kernel makes it as the pass
/// enters [`Stage::Arrive`](super::stage::Stage::Arrive), after `agent`'s
/// unsaved changes are saved, announces it ([`AgentForked`]), inserts the
/// [`ForkResult`] and despawns the request, which is [`RequestOf`] `agent`.
///
/// It is no event:
///
/// ```compile_fail,E0277
/// use rig_ecs::prelude::*;
///
/// fn fork(agent: Entity, mut commands: Commands) {
///     commands.trigger(ForkAgent::new(agent, None));
/// }
/// ```
#[derive(Bundle)]
pub struct ForkAgent(RequestOf, Forking);

impl ForkAgent {
    /// Forks `agent` through `through`, its last message the fork may keep
    /// (`None` for its end).
    pub fn new(agent: Entity, through: Option<MessageId>) -> Self {
        Self(RequestOf(agent), Forking(through))
    }
}

/// On a [`ForkAgent`] request not made yet: the message it is through.
#[derive(Component)]
pub(crate) struct Forking(Option<MessageId>);

/// On a [`ForkAgent`] request, once the kernel tried it: the new agent, or
/// why there is none.
#[derive(Component, Debug)]
pub struct ForkResult(pub Result<Entity, ForkError>);

/// Why a [`ForkAgent`] made no fork.
#[derive(Debug, thiserror::Error)]
pub enum ForkError {
    /// The entity forked is no agent.
    #[error("it is no agent")]
    NoAgent,
    /// The session is not logged.
    #[error("the session is not logged")]
    NoLog,
    /// The app has no type registry.
    #[error("the app has no type registry")]
    NoRegistry,
    /// The log could not be copied, such as through a message the agent
    /// never had.
    #[error("{0}")]
    Session(#[source] SessionError),
}

/// On an agent made by a [`ForkAgent`]: the agents its conversation came
/// from, nearest first. A fork of a fork lists its source, then the
/// source's own. Saved with the session.
///
/// ```
/// use rig_core::conversation::Conversation;
/// use rig_core::message::Message;
/// use rig_ecs::fork::{Ancestor, ForkedFrom};
///
/// // A fork of `main` made after two messages, which then went on.
/// let mut fork = Conversation::from_messages([Message::user("Hi"), Message::assistant("Hello")]);
/// let since = fork.next_id();
/// let forked = ForkedFrom(vec![Ancestor { agent: "main".to_owned(), since }]);
/// let own = fork.push(Message::user("And now?"));
/// assert_eq!(fork.id(1).and_then(|id| forked.made_by(id)), Some("main"));
/// assert_eq!(forked.made_by(own), None);
/// ```
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[reflect(
    opaque,
    Component,
    Saved,
    Default,
    Clone,
    Debug,
    PartialEq,
    Serialize,
    Deserialize
)]
pub struct ForkedFrom(pub Vec<Ancestor>);

/// An agent a fork's conversation came from ([`ForkedFrom`]).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ancestor {
    /// Its id.
    pub agent: String,
    /// The first message id made after the copy from it: the entries of
    /// the fork with a lower id came from it, or from an ancestor further
    /// up whose `since` is above them too.
    pub since: MessageId,
}

impl ForkedFrom {
    /// The id of the agent that made the fork's message `id`, which the
    /// fork copied: the farthest ancestor whose `since` is above it. `None`
    /// for a message the fork made itself.
    pub fn made_by(&self, id: MessageId) -> Option<&str> {
        let older = self.0.iter().take_while(|ancestor| id < ancestor.since);
        older.last().map(|ancestor| ancestor.agent.as_str())
    }

    /// The ids of the agents it came from, nearest first.
    pub fn agents(&self) -> impl Iterator<Item = &str> {
        self.0.iter().map(|ancestor| ancestor.agent.as_str())
    }
}

/// An agent made by a [`ForkAgent`], announced before its request's
/// [`ForkResult`].
#[derive(Event, Clone, Debug)]
pub struct AgentForked {
    /// The agent forked.
    pub from: Entity,
    /// The new agent.
    pub fork: Entity,
    /// The message the fork was asked through, as [`ForkAgent`] said.
    pub through: Option<MessageId>,
}

/// Makes each fork asked for ([`ForkAgent`]), gives its request its
/// [`ForkResult`] and despawns it.
pub(crate) fn make_forks(
    world: &mut World,
    asked: &mut QueryState<(Entity, &RequestOf, &Forking)>,
) {
    let asked: Vec<(Entity, Entity, Option<MessageId>)> = asked
        .iter(world)
        .map(|(request, &RequestOf(from), fork)| (request, from, fork.0))
        .collect();
    for (request, from, through) in asked {
        // The observer of an earlier result may have ended it.
        if world.get::<Forking>(request).is_none() {
            continue;
        }
        let made = fork_now(world, from, through);
        if let Ok(fork) = made {
            world.trigger(AgentForked {
                from,
                fork,
                through,
            });
        }
        if let Ok(mut request) = world.get_entity_mut(request) {
            request.insert(ForkResult(made));
        }
        world.try_despawn(request).ok();
    }
}

/// Forks `from` through `through`; the new agent.
fn fork_now(
    world: &mut World,
    from: Entity,
    through: Option<MessageId>,
) -> Result<Entity, ForkError> {
    let id = world
        .get::<AgentId>(from)
        .cloned()
        .ok_or(ForkError::NoAgent)?;
    let log = world
        .get_resource::<SessionLog>()
        .filter(|log| log.0.is_live())
        .cloned()
        .ok_or(ForkError::NoLog)?;
    let registry = world
        .get_resource::<AppTypeRegistry>()
        .cloned()
        .ok_or(ForkError::NoRegistry)?;
    if let Some(mut conversation) = world.get_mut::<Conversation>(from) {
        let saved = save_conversation(&log, from, &id, conversation.bypass_change_detection());
        world.write_message_batch(saved);
    }
    let to = AgentId::default();
    let copy = log
        .0
        .fork_log(&id.0, &to.0, through)
        .map_err(ForkError::Session)?;
    let registry = registry.read();
    let (mut resets, mut removals) = (Vec::new(), Vec::new());
    let saved: Vec<(&ReflectComponent, Box<dyn PartialReflect>)> = registry
        .iter_with_data::<ReflectSaved>()
        .filter_map(|(registration, saved)| {
            match saved.fork {
                Fork::Copy => {}
                Fork::Reset(reset) => {
                    resets.push(reset);
                    return None;
                }
                Fork::Never(remove) => {
                    removals.push(remove);
                    return None;
                }
            }
            let component = registration.data::<ReflectComponent>()?;
            let value = component.reflect(world.get_entity(from).ok()?)?;
            let value = match value.reflect_clone() {
                Ok(value) => value.into_partial_reflect(),
                Err(_) => value.to_dynamic().ok()?,
            };
            Some((component, value))
        })
        .collect();
    let source = Ancestor {
        agent: id.0.clone(),
        since: copy.conversation.next_id(),
    };
    let older = world
        .get::<ForkedFrom>(from)
        .map(|from| from.0.iter().cloned());
    let lineage = ForkedFrom(
        std::iter::once(source)
            .chain(older.into_iter().flatten())
            .collect(),
    );
    // The source's context figure measures the fork's conversation only
    // when the fork has the source's conversation as it is now: the same
    // entries, not only as many.
    let whole = world.get::<Conversation>(from).is_some_and(|conversation| {
        conversation.entries().len() == copy.conversation.entries().len()
            && (conversation.entries().zip(copy.conversation.entries())).all(|(own, copied)| {
                own.id == copied.id && own.made == copied.made && own.message == copied.message
            })
    });
    let measured = world.get::<LastUsage>(from).filter(|_| whole).cloned();
    let key = CacheKey(CacheKey::of(world.get::<CacheKey>(from), &id).to_owned());
    let connection = world.get::<Connection>(from).cloned();
    let conversation = Conversation(copy.conversation);
    let empty = conversation.messages().is_empty();
    let mut spawned = world.spawn((Name::new("agent"), Agent::new(), to.clone(), conversation));
    for (component, value) in &saved {
        component.insert(&mut spawned, value.as_ref(), &registry);
    }
    spawned.insert((key, lineage));
    if let Some(connection) = connection {
        spawned.insert(connection);
    }
    // Removals last, so no component inserted before brings back a
    // required one.
    for rule in resets.into_iter().chain(removals) {
        rule(&mut spawned);
    }
    match measured {
        Some(measured) => {
            spawned.insert(measured);
        }
        // No reply measured the conversation it starts with.
        None if !empty => {
            spawned.insert(ContextStale::default());
        }
        None => {}
    }
    let fork = spawned.id();
    // A value the copied log holds that the source has no more is gone in
    // the fork too.
    let keys = world.get_resource::<SavedKeys>();
    for key in copy.saved.keys() {
        let path = keys.and_then(|keys| keys.path(key)).unwrap_or(key);
        let component = saved_type(path, &registry)
            .ok()
            .map(|(_, component)| component);
        let held = |component: &ReflectComponent| {
            let entity = world.get_entity(fork).ok();
            entity.is_some_and(|entity| component.contains(entity))
        };
        if component.is_some_and(|component| !held(component)) {
            log.0.save_value(&to.0, key, None, false);
        }
    }
    Ok(fork)
}

#[cfg(test)]
mod tests;
