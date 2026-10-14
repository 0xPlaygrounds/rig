//! Agent state that follows the line of its conversation: a component
//! reflected with [`ReflectFollows`] is kept in the agent's conversation, so
//! every split, switch, move, merge, fork and restore gives the agent the
//! value its line had at that point, and a sibling line never sees it.
//!
//! A value set on a line is kept as an aside of kind [`FOLLOWS`] by rig,
//! following the line's last entry, at most one per key and entry. While
//! the line has no entry of its own since it parted from a kept branch (or
//! has no entry at all), the value is one of the line's own values
//! (rig-core's `Conversation::set_line_value`), which travel with the line
//! as its title does, and is anchored after the line's first entry of its
//! own once it has one. A removed component is kept as a `null` value, so
//! it stays removed on that line.
//!
//! Whenever the kernel saves the conversations, between the stages of each
//! agent pass and at the end of the frame, it first mirrors each change of
//! a followed component into the conversation, and after a change that may
//! change the line (a splice, a reset, a branch, line value or followed
//! value put) gives the agent its line's value, read as a saved value is
//! (upgraded through its plugin's steps); a value it gave is never
//! mirrored back. So a change is kept at the entry it was made after, give
//! or take a stage, and a line changed before the agent loop or in a stage
//! gives its value to the next stage. Pushes and tool calls cost nothing.

use std::collections::{HashMap, HashSet};

use bevy_app::prelude::*;
use bevy_ecs::change_detection::Tick;
use bevy_ecs::message::MessageCursor;
use bevy_ecs::prelude::*;
use bevy_ecs::reflect::AppTypeRegistry;
use bevy_log::warn;
use bevy_reflect::serde::TypedReflectSerializer;
use bevy_reflect::{CreateTypeData, Reflect, TypePath, TypeRegistry};
use rig_core::conversation::{Aside, AsideId, Author, BRANCH, Change, LINE, MessageId};
use rig_memory::Versioned;
use serde_json::{Map, Value};

use super::agent::{Agent, AgentId, Conversation};
use super::journal::{Committed, SaveConversations, SessionLog, save_conversation};
use super::restore::{SavedKeys, SavedReader, SavedUpgrades};

/// The kind of the asides holding followed values:
/// `{"key": <key>, "v": n, "value": …}`, by rig.
pub const FOLLOWS: &str = "rig.follows";

/// Keeps an agent component with the line of the agent's conversation it was
/// set on: derive `Reflect` and add `Follows` (`#[reflect(Component,
/// Follows)]`, or opaque with serde as `Saved` allows). A split, a switch, a
/// move, a merge, a fork and a restore give the agent the value its line
/// had at that point; with none set on the line, the component is removed.
/// Versioned like a saved value, and keyed by its type path or the stable
/// key its plugin gives it. It takes the place of `Saved` for its type.
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

/// The agents and keys whose line value could not be read this run: their
/// component is not mirrored, so the value is never written over.
#[derive(Resource, Default)]
pub(crate) struct Unmirrored(HashSet<(Entity, String)>);

/// A followed value as kept: `{"v": n, "value": …}`, or `null` when the
/// component was removed.
fn envelope(saved: Option<&Versioned>) -> Value {
    let Some(saved) = saved else {
        return Value::Null;
    };
    let mut envelope = Map::new();
    envelope.insert("v".to_owned(), Value::from(saved.v));
    envelope.insert("value".to_owned(), saved.value.clone());
    Value::Object(envelope)
}

/// A kept value read back: `None` for `null` (removed) or no value.
fn versioned(envelope: &Value) -> Option<Versioned> {
    let value = envelope.get("value").filter(|value| !value.is_null())?;
    let v = envelope.get("v").and_then(Value::as_u64);
    let v = v.and_then(|v| u32::try_from(v).ok()).unwrap_or(1);
    Some(Versioned {
        v,
        value: value.clone(),
    })
}

/// The followed value of `key` an aside holds, as an envelope.
fn of_aside(aside: &Aside, key: &str) -> Option<Value> {
    if aside.kind != FOLLOWS || aside.value.get("key").and_then(Value::as_str) != Some(key) {
        return None;
    }
    let mut envelope = aside.value.as_object()?.clone();
    envelope.remove("key");
    let removed = envelope.get("value").is_none_or(Value::is_null);
    Some(if removed {
        Value::Null
    } else {
        Value::Object(envelope)
    })
}

/// An aside holding `envelope` as the value of `key`, after `after`.
fn follows_aside(key: &str, envelope: &Value, after: Option<MessageId>) -> Aside {
    let mut value = match envelope {
        Value::Object(envelope) => envelope.clone(),
        _ => Map::new(),
    };
    value.insert("key".to_owned(), Value::String(key.to_owned()));
    value.entry("value").or_insert(Value::Null);
    Aside::new(FOLLOWS, Author::rig(), Value::Object(value)).after(after)
}

/// The asides holding values of `key`, each with its id and position.
fn asides_of(
    conversation: &rig_core::conversation::Conversation,
    key: &str,
) -> Vec<(AsideId, usize, Value)> {
    conversation
        .asides()
        .filter_map(|(id, aside)| {
            let value = of_aside(aside, key)?;
            Some((id, conversation.aside_position(aside), value))
        })
        .collect()
}

/// The value of `key` the active line of `conversation` has: its line value,
/// else the newest of its asides (the one drawn last); `None` when it has
/// none.
pub(crate) fn current(
    conversation: &rig_core::conversation::Conversation,
    key: &str,
) -> Option<Value> {
    if let Some(value) = conversation
        .line_values()
        .and_then(|values| values.get(key))
    {
        return Some(value.clone());
    }
    let asides = asides_of(conversation, key).into_iter().enumerate();
    let newest = asides.max_by_key(|(order, (_, position, _))| (*position, *order));
    newest.map(|(_, (_, _, value))| value)
}

/// Whether a value set now belongs to the line itself: it has no entry, or
/// a kept branch parts at its end.
fn at_shared_end(conversation: &rig_core::conversation::Conversation) -> bool {
    conversation.messages().is_empty() || conversation.parts_at_end()
}

/// Puts `envelope` as the value of `key` after `after`: in place of the
/// aside of `key` that follows exactly that entry, else a new one. Unless
/// `over`, an aside there is kept as it is.
fn anchor(
    conversation: &mut rig_core::conversation::Conversation,
    key: &str,
    envelope: &Value,
    after: Option<MessageId>,
    over: bool,
) {
    let there = conversation
        .asides()
        .find(|(_, aside)| aside.after == after && of_aside(aside, key).is_some())
        .map(|(id, aside)| (id, of_aside(aside, key)));
    let aside = follows_aside(key, envelope, after);
    match there {
        Some((_, Some(value))) if value == *envelope => {}
        Some((id, _)) if over => drop(conversation.replace_aside(id, aside)),
        Some(_) => {}
        None => drop(conversation.add_aside(aside)),
    }
}

/// Sets the value of `key` on the active line to `envelope`, unless it has
/// it: a line value at a shared end, else after the last entry.
pub(crate) fn set(
    conversation: &mut rig_core::conversation::Conversation,
    key: &str,
    envelope: Value,
) {
    if current(conversation, key).as_ref() == Some(&envelope) {
        return;
    }
    if at_shared_end(conversation) {
        conversation.set_line_value(key, Some(envelope), Author::rig());
        return;
    }
    if conversation
        .line_values()
        .is_some_and(|values| values.contains_key(key))
    {
        conversation.set_line_value(key, None, Author::rig());
    }
    let last = conversation.entries().next_back().map(|entry| entry.id);
    anchor(conversation, key, &envelope, last, true);
}

/// Anchors each of `keys` the line values hold after the last entry, once
/// the line has an entry of its own; whether it changed anything.
fn anchor_line_values(
    conversation: &mut rig_core::conversation::Conversation,
    keys: &[String],
) -> bool {
    if at_shared_end(conversation) {
        return false;
    }
    let Some(values) = conversation.line_values().cloned() else {
        return false;
    };
    let last = conversation.entries().next_back().map(|entry| entry.id);
    let mut changed = false;
    for key in keys {
        if let Some(envelope) = values.get(key) {
            conversation.set_line_value(key, None, Author::rig());
            anchor(conversation, key, envelope, last, true);
            changed = true;
        }
    }
    changed
}

/// After a splice took out `removed` and put in `with`: each of `keys`
/// whose asides followed a removed entry gets one, the value of the one
/// that followed the latest removed entry, after the entry drawn before
/// them now (the line value when that entry is the line's shared end).
/// One already there that follows an entry the splice put in is newer and
/// kept. Whether it changed anything.
fn collapse(
    conversation: &mut rig_core::conversation::Conversation,
    keys: &[String],
    removed: &[MessageId],
    with: &[MessageId],
) -> bool {
    let mut changed = false;
    for key in keys {
        let gone: Vec<(AsideId, usize, Value, usize)> = conversation
            .asides()
            .filter_map(|(id, aside)| {
                let order = removed.iter().position(|id| Some(*id) == aside.after)?;
                let value = of_aside(aside, key)?;
                Some((id, conversation.aside_position(aside), value, order))
            })
            .collect();
        let Some((_, position, value, _)) = gone.iter().max_by_key(|gone| gone.3).cloned() else {
            continue;
        };
        for (id, ..) in &gone {
            conversation.remove_aside(*id);
            changed = true;
        }
        let target = position.checked_sub(1).and_then(|at| conversation.id(at));
        let last = conversation.entries().next_back().map(|entry| entry.id);
        if target.is_some() && target == last && conversation.parts_at_end() {
            if current(conversation, key).as_ref() != Some(&value) {
                conversation.set_line_value(key, Some(value), Author::rig());
            }
            continue;
        }
        let newer = target.is_some_and(|target| with.contains(&target));
        anchor(conversation, key, &value, target, !newer);
    }
    changed
}

/// Mirrors each change of the followed component `T` on an agent, and its
/// removal, into the agent's conversation, unless the line has that value.
fn mirror<T: Component + Reflect + TypePath>(
    mut changed: Query<(Entity, Ref<T>, &mut Conversation), (With<Agent>, Changed<T>)>,
    mut removed: RemovedComponents<T>,
    mut agents: Query<&mut Conversation, (With<Agent>, Without<T>)>,
    registry: Res<AppTypeRegistry>,
    (keys, upgrades): (Res<SavedKeys>, Res<SavedUpgrades>),
    (unmirrored, mut given): (Option<Res<Unmirrored>>, ResMut<Given>),
) {
    let path = T::type_path();
    let key = keys.key(path).unwrap_or(path);
    let skipped = |entity: Entity| {
        unmirrored
            .as_ref()
            .is_some_and(|unmirrored| unmirrored.0.contains(&(entity, key.to_owned())))
    };
    let v = upgrades.0.version(path);
    let registry = registry.read();
    for (entity, component, mut conversation) in &mut changed {
        let gave = given.0.remove(&(entity, key.to_owned()));
        if skipped(entity) || gave == Some(Some(component.last_changed())) {
            continue;
        }
        let value = TypedReflectSerializer::new(component.as_partial_reflect(), &registry);
        match serde_json::to_value(value) {
            Ok(value) => {
                let saved = (!value.is_null()).then_some(Versioned { v, value });
                let envelope = envelope(saved.as_ref());
                if current(&conversation, key).as_ref() != Some(&envelope) {
                    set(&mut conversation, key, envelope);
                }
            }
            Err(failure) => warn!("not keeping {path} with the conversation: {failure}"),
        }
    }
    for entity in removed.read() {
        let gave = given.0.remove(&(entity, key.to_owned()));
        if skipped(entity) || gave == Some(None) {
            continue;
        }
        if let Ok(mut conversation) = agents.get_mut(entity)
            && current(&conversation, key).is_some_and(|value| !value.is_null())
        {
            set(&mut conversation, key, Value::Null);
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
pub(crate) fn followed(world: &World) -> Vec<(String, String)> {
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
pub(crate) fn sync(world: &mut World, mut cursor: Local<MessageCursor<Committed>>) {
    let Some(messages) = world.get_resource::<Messages<Committed>>() else {
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
            } if [BRANCH, LINE, FOLLOWS].contains(&aside.kind.as_str()) => work.scan = true,
            _ => {}
        }
    }
    let followed = followed(world);
    if followed.is_empty() || work.is_empty() {
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
            edited |= collapse(conversation, &keys, removed, with);
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
    let mut reader = SavedReader::new(world);
    for agent in scans {
        give_line_values(world, &registry, &mut reader, agent, &followed);
    }
    reader.finish(world);
}

/// Gives `agent` the value of each `followed` (key, type path) its line has:
/// inserted when it differs from the component, removed when the line has
/// none or a removal. A value no step reads is set aside, and the key is not
/// mirrored on that agent this run.
pub(crate) fn give_line_values(
    world: &mut World,
    registry: &TypeRegistry,
    reader: &mut SavedReader,
    agent: Entity,
    followed: &[(String, String)],
) {
    let Some(conversation) = world.get::<Conversation>(agent) else {
        return;
    };
    let id = world.get::<AgentId>(agent).cloned();
    let lines: Vec<(String, String, Option<Versioned>)> = followed
        .iter()
        .map(|(key, path)| {
            let value = current(conversation, key).and_then(|value| versioned(&value));
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
        match reader.read_followed(registry, id.as_ref(), &path, &value) {
            Ok((read, upgraded)) => {
                if now.as_ref() == Some(&upgraded) {
                    continue;
                }
                if let Ok(mut entity) = world.get_entity_mut(agent) {
                    component.insert(&mut entity, read.as_partial_reflect(), registry);
                    let tick = world.change_tick();
                    world
                        .get_resource_or_init::<Given>()
                        .0
                        .insert((agent, key), Some(tick));
                }
            }
            Err(why) => {
                warn!("not giving the agent its line's `{key}`: {why}");
                world
                    .get_resource_or_init::<Unmirrored>()
                    .0
                    .insert((agent, key));
            }
        }
    }
}

#[cfg(test)]
mod tests;
