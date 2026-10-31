//! Forking an agent: [`ForkAgent`] makes a new agent in the same session
//! whose log is a copy of another's up to a turn boundary, so it goes on
//! from there by itself, a compaction made before the cut can be undone in
//! it, and its requests share the source's prompt cache.
//!
//! ```
//! use rig_ecs::prelude::*;
//!
//! // Shows each fork as it is made.
//! fn show(forked: On<AgentForked>, mut shown: Local<Option<Entity>>) {
//!     *shown = Some(forked.fork);
//! }
//!
//! App::new().add_observer(show);
//! ```

use bevy_ecs::prelude::*;
use bevy_ecs::reflect::{AppTypeRegistry, ReflectComponent};
use bevy_reflect::PartialReflect;
use bevy_reflect::prelude::*;
use rig_core::conversation::MessageId;

use super::agent::{Agent, AgentId, CacheKey, ContextStale, Conversation, Notice};
use super::model::Connection;
use super::restore::SavedKeys;
use super::session_log::{Fork, ReflectSaved, SessionLog, save_conversation};

/// Fork the agent `agent`: a new agent in the same session whose log is a
/// copy of `agent`'s up to a turn boundary at or before `through` (its
/// end with `None`; rig-memory's `Writer::fork_log`), its conversation that
/// copy. A user message that asked for a turn is at no boundary, so a fork
/// through it ends just before it. The new agent has the saved components
/// `agent` has now (its model, reasoning setting, tools, system prompt and
/// working directory among them) as their fork rules say
/// ([`ReflectSaved`]), its connection, and a [`ForkedFrom`] naming it, and
/// shares its prompt cache ([`CacheKey`]). Its context is [`ContextStale`]
/// until its first reply. It has no parent. [`AgentForked`]
/// names it; a fork that cannot be made, such as without a session log or
/// through a message the agent never had, is a notice. It is made once the
/// commands run, after the source's unsaved changes are saved.
#[derive(EntityEvent, Clone, Debug)]
pub struct ForkAgent {
    /// The agent forked.
    #[event_target]
    pub agent: Entity,
    /// The last message the fork may keep; `None` for the end.
    pub through: Option<MessageId>,
}

/// On an agent made by a [`ForkAgent`]: the id of the agent it was forked
/// from. Saved with the session.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Component, Saved, Default, Clone, Debug, PartialEq)]
pub struct ForkedFrom(pub String);

/// An agent made by a [`ForkAgent`].
#[derive(Event, Clone, Debug)]
pub struct AgentForked {
    /// The agent forked.
    pub from: Entity,
    /// The new agent.
    pub fork: Entity,
    /// The message the fork was asked through, as [`ForkAgent`] said.
    pub through: Option<MessageId>,
}

/// Forks an agent, as [`ForkAgent`] says.
pub(crate) fn fork_agent(fork: On<ForkAgent>, mut commands: Commands) {
    let (from, through) = (fork.agent, fork.through);
    commands.queue(
        move |world: &mut World| match fork_now(world, from, through) {
            Ok(fork) => world.trigger(AgentForked {
                from,
                fork,
                through,
            }),
            Err(why) => {
                let why = format!("Could not fork the agent: {why}.");
                world.write_message(Notice::error(from, why));
            }
        },
    );
}

/// Forks `from` through `through`; the new agent.
fn fork_now(world: &mut World, from: Entity, through: Option<MessageId>) -> Result<Entity, String> {
    let id = world
        .get::<AgentId>(from)
        .cloned()
        .ok_or("it is no agent")?;
    let log = world
        .get_resource::<SessionLog>()
        .filter(|log| log.0.is_live())
        .cloned()
        .ok_or("the session is not logged")?;
    let registry = world
        .get_resource::<AppTypeRegistry>()
        .cloned()
        .ok_or("the app has no type registry")?;
    if let Some(mut conversation) = world.get_mut::<Conversation>(from) {
        let saved = save_conversation(&log, from, &id, conversation.bypass_change_detection());
        world.write_message_batch(saved);
    }
    let to = AgentId::default();
    let copy = log
        .0
        .fork_log(&id.0, &to.0, through)
        .map_err(|failure| failure.to_string())?;
    let registry = registry.read();
    let mut set = Vec::new();
    let saved: Vec<(&ReflectComponent, Box<dyn PartialReflect>)> = registry
        .iter_with_data::<ReflectSaved>()
        .filter_map(|(registration, saved)| {
            if let Fork::Set(rule) = saved.fork {
                set.push(rule);
                return None;
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
    let key = CacheKey(CacheKey::of(world.get::<CacheKey>(from), &id).to_owned());
    let connection = world.get::<Connection>(from).cloned();
    let conversation = Conversation(copy.conversation);
    let empty = conversation.messages().is_empty();
    let mut spawned = world.spawn((Name::new("agent"), Agent, to.clone(), conversation));
    for (component, value) in &saved {
        component.insert(&mut spawned, value.as_ref(), &registry);
    }
    for rule in set {
        rule(&mut spawned);
    }
    spawned.insert((key, ForkedFrom(id.0.clone())));
    // No reply of the fork measured the conversation it starts with.
    if !empty {
        spawned.insert(ContextStale {});
    }
    if let Some(connection) = connection {
        spawned.insert(connection);
    }
    let fork = spawned.id();
    // A value the copied log holds that the source has no more is gone in
    // the fork too.
    let keys = world.get_resource::<SavedKeys>();
    for key in copy.saved.keys() {
        let path = keys.and_then(|keys| keys.path(key)).unwrap_or(key);
        let registered = registry.get_with_type_path(path);
        let component = registered.filter(|it| it.data::<ReflectSaved>().is_some());
        let component = component.and_then(|it| it.data::<ReflectComponent>());
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
