//! Agent state that follows the line of its conversation: a component
//! reflected with [`ReflectFollows`] is kept in the agent's conversation, so
//! every split, switch, move, merge, fork and restore gives the agent the
//! value its line had at that point, and a sibling line never sees it.
//!
//! A value is kept in the conversation as `rig.follows` asides
//! and line values; a removed component is kept as `null`, so it stays
//! removed on that line.
//!
//! The kernel mirrors each change of a followed component into the
//! conversation whenever it saves (before `Update`, between the stages of a
//! pass, at the end of the frame), and after a change that may switch the
//! line gives the agent its line's value, upgraded as a saved value is. A
//! value it gave is never mirrored back.

use std::collections::HashMap;

use bevy_app::prelude::*;
use bevy_ecs::change_detection::Tick;
use bevy_ecs::message::MessageCursor;
use bevy_ecs::prelude::*;
use bevy_ecs::reflect::AppTypeRegistry;
use bevy_log::warn;
use bevy_reflect::serde::TypedReflectSerializer;
use bevy_reflect::{CreateTypeData, Reflect, TypePath, TypeRegistry};
use rig_core::conversation::{BRANCH, Change, LINE_VALUES, MessageId};
use serde_json::Value;

use line::{FOLLOWS, anchor_line_values, follow, followed, refollow_after_splice};

use super::agent::{Agent, AgentId, Conversation};
use super::restore::{SavedKeys, SavedReader};
use super::session_log::{ConversationChanged, SaveConversations, SessionLog, save_conversation};

/// Keeps an agent component with the line of the agent's conversation it was
/// set on: derive `Reflect` and add `Follows` (`#[reflect(Component,
/// Follows)]`, or opaque with serde as `Saved` allows). A split, a switch, a
/// move, a merge, a fork and a restore give the agent the value its line
/// had at that point; with none set on the line, the component is removed.
/// Keyed by its type path or the stable key its plugin gives it: a line
/// value that no longer reads is left on the line. It takes the place of
/// `Saved` for its type.
#[derive(Clone)]
pub struct ReflectFollows {
    pub(crate) watch: fn(&mut App),
}

impl<T: Component + Reflect + TypePath> CreateTypeData<T> for ReflectFollows {
    fn create_type_data(_: ()) -> Self {
        let watch = |app: &mut App| {
            app.init_resource::<Given>()
                .world_mut()
                .get_resource_or_init::<Followed>()
                .0
                .push(T::type_path().to_owned());
            app.add_systems(SaveConversations, mirror::<T>.in_set(Mirroring));
        };
        Self { watch }
    }
}

/// The systems that mirror followed components into their conversations
/// before each save of them ([`SaveConversations`]).
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Mirroring;

/// The values the kernel gave agents from their lines, by agent and key,
/// with the tick of the insert (`None` for a removal): a mirror never writes
/// them back, into whatever line is active by then.
#[derive(Resource, Default)]
pub(crate) struct Given(HashMap<(Entity, String), Option<Tick>>);

/// The type paths of the followed types.
#[derive(Resource, Default)]
pub(crate) struct Followed(pub(crate) Vec<String>);

/// Mirrors each change of the followed component `T` on an agent, and its
/// removal, into the agent's conversation, unless the line has that value.
fn mirror<T: Component + Reflect + TypePath>(
    mut changed: Query<(Entity, Ref<T>, &mut Conversation), (With<Agent>, Changed<T>)>,
    mut removed: RemovedComponents<T>,
    mut agents: Query<&mut Conversation, (With<Agent>, Without<T>)>,
    registry: Res<AppTypeRegistry>,
    keys: Res<SavedKeys>,
    mut given: ResMut<Given>,
) {
    let path = T::type_path();
    let key = keys.key(path).unwrap_or(path);
    let registry = registry.read();
    for (entity, component, mut conversation) in &mut changed {
        let gave = given.0.remove(&(entity, key.to_owned()));
        if gave == Some(Some(component.last_changed())) {
            continue;
        }
        let value = TypedReflectSerializer::new(component.as_partial_reflect(), &registry);
        match serde_json::to_value(value) {
            Ok(value) => {
                let saved = (!value.is_null()).then_some(value);
                follow(&mut conversation, key, saved.as_ref());
            }
            Err(failure) => warn!("not keeping {path} with the conversation: {failure}"),
        }
    }
    for entity in removed.read() {
        let gave = given.0.remove(&(entity, key.to_owned()));
        if gave == Some(None) {
            continue;
        }
        if let Ok(mut conversation) = agents.get_mut(entity)
            && followed(&conversation, key).is_some()
        {
            follow(&mut conversation, key, None);
        }
    }
}

/// What the changes committed for one agent ask of the sync.
#[derive(Default)]
struct Work {
    /// Each splice's removed and put ids.
    splices: Vec<(Vec<MessageId>, Vec<MessageId>)>,
    /// Whether the line may have changed.
    scan: bool,
    /// Whether anything changed.
    any: bool,
}

/// The followed types, each with its key.
pub(crate) fn followed_types(world: &World) -> Vec<(String, String)> {
    let Some(followed) = world.get_resource::<Followed>() else {
        return Vec::new();
    };
    let keys = world.get_resource::<SavedKeys>();
    followed
        .0
        .iter()
        .map(|path| {
            let key = keys.and_then(|keys| keys.key(path)).unwrap_or(path);
            (key.to_owned(), path.clone())
        })
        .collect()
}

/// Keeps followed values with their lines, after each save of the
/// conversations ([`SaveConversations`]): collapses the values of entries a splice took out,
/// anchors line values once a line has an entry of its own, and, after a
/// change that may change the line, gives each agent its line's values.
pub(crate) fn sync(world: &mut World, mut cursor: Local<MessageCursor<ConversationChanged>>) {
    let Some(messages) = world.get_resource::<Messages<ConversationChanged>>() else {
        return;
    };
    let mut work: HashMap<Entity, Work> = HashMap::new();
    for committed in cursor.read(messages) {
        let work = work.entry(committed.agent).or_default();
        work.any = true;
        match &committed.change.change {
            Change::Splice { removed, with, .. } => {
                let with = with.iter().filter_map(|entry| entry.id).collect();
                work.splices.push((removed.clone(), with));
                work.scan = true;
            }
            Change::Reset { .. } => work.scan = true,
            Change::Aside {
                aside: Some(aside), ..
            } if [BRANCH, LINE_VALUES, FOLLOWS].contains(&aside.kind.as_str()) => work.scan = true,
            _ => {}
        }
    }
    if work.is_empty() {
        return;
    }
    let followed = followed_types(world);
    if followed.is_empty() {
        return;
    }
    let keys: Vec<String> = followed.iter().map(|(key, _)| key.clone()).collect();
    let log = world
        .get_resource::<SessionLog>()
        .cloned()
        .unwrap_or_default();
    let mut agents: Vec<(Entity, Work)> = work.into_iter().collect();
    agents.sort_by_key(|(agent, _)| *agent);
    let mut saved = Vec::new();
    let mut scans = Vec::new();
    for (agent, work) in agents {
        let Some(id) = world.get::<AgentId>(agent).cloned() else {
            continue;
        };
        let Some(mut conversation) = world.get_mut::<Conversation>(agent) else {
            continue;
        };
        let conversation = conversation.bypass_change_detection();
        let mut edited = false;
        for (removed, with) in &work.splices {
            edited |= refollow_after_splice(conversation, &keys, removed, with);
        }
        if work.any {
            edited |= anchor_line_values(conversation, &keys);
        }
        if edited {
            saved.extend(save_conversation(&log, agent, &id, conversation));
        }
        if work.scan || edited {
            scans.push(agent);
        }
    }
    world.write_message_batch(saved);
    if scans.is_empty() {
        return;
    }
    let registry = world.get_resource::<AppTypeRegistry>().cloned();
    let registry = registry.unwrap_or_default();
    let registry = registry.read();
    let reader = SavedReader::new(world);
    for agent in scans {
        give_line_values(world, &registry, &reader, agent, &followed);
    }
}

/// Gives `agent` the value of each `followed` (key, type path) its line has:
/// inserted when it differs from the component, removed when the line has
/// none or a removal. A value that does not read is left on the line.
pub(crate) fn give_line_values(
    world: &mut World,
    registry: &TypeRegistry,
    reader: &SavedReader,
    agent: Entity,
    followed: &[(String, String)],
) {
    let Some(conversation) = world.get::<Conversation>(agent) else {
        return;
    };
    let lines: Vec<(String, String, Option<Value>)> = followed
        .iter()
        .map(|(key, path)| {
            let value = line::followed(conversation, key);
            (key.clone(), path.clone(), value)
        })
        .collect();
    for (key, path, value) in lines {
        let Ok((_, component)) = super::restore::saved_type(&path, registry) else {
            continue;
        };
        let now = world
            .get_entity(agent)
            .ok()
            .and_then(|entity| component.reflect(entity));
        let now = now.and_then(|now| {
            let now = TypedReflectSerializer::new(now.as_partial_reflect(), registry);
            serde_json::to_value(now).ok()
        });
        let Some(value) = value else {
            if now.is_some()
                && let Ok(mut entity) = world.get_entity_mut(agent)
            {
                component.remove(&mut entity);
                world
                    .get_resource_or_init::<Given>()
                    .0
                    .insert((agent, key), None);
            }
            continue;
        };
        if now.as_ref() == Some(&value) {
            continue;
        }
        match reader.read_followed(registry, &path, &value) {
            Ok(read) => {
                if let Ok(mut entity) = world.get_entity_mut(agent) {
                    component.insert(&mut entity, read.as_partial_reflect(), registry);
                    let tick = world.change_tick();
                    world
                        .get_resource_or_init::<Given>()
                        .0
                        .insert((agent, key), Some(tick));
                }
            }
            Err(why) => warn!("not giving the agent its line's `{key}`: {why}"),
        }
    }
}

mod line;
#[cfg(test)]
mod tests;
