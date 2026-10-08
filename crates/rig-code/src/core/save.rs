//! Saving and restoring agents through reflection. Every agent component
//! whose type registers [`ReflectSaved`] is written to the session's
//! `state.json`, keyed by its type path. On restore, a component whose type
//! is no longer registered, such as one from a removed plugin, is skipped.
//!
//! ```no_run
//! use rig_code::bevy_app::prelude::*;
//! use rig_code::bevy_ecs::prelude::*;
//! use rig_code::prelude::*;
//! # use bevy_reflect::prelude::*;
//!
//! #[derive(Component, Reflect, Default)]
//! #[reflect(Component, Default, Saved)]
//! struct Greetings {
//!     count: u32,
//! }
//!
//! fn greetings_plugin(app: &mut App) {
//!     app.register_type::<Greetings>();
//! }
//! ```

use std::collections::BTreeMap;
use std::fs;
use std::sync::atomic::Ordering;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::warn;
use bevy_reflect::serde::{TypedReflectDeserializer, TypedReflectSerializer};
use bevy_reflect::{
    CreateTypeData, Reflect, ReflectFromReflect, TypePath, TypeRegistration, TypeRegistry,
};
use serde::de::DeserializeSeed;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::agent::Agent;
use super::dispatch::Effects;
use super::registry::Notice;
use super::session::Session;
use super::turn::TurnEnded;

/// Marks a component type as part of an agent's saved state. Register it
/// with `#[reflect(Component, Saved)]` and `App::register_type`.
#[derive(Clone)]
pub struct ReflectSaved;

impl<T: Component + Reflect + TypePath> CreateTypeData<T> for ReflectSaved {
    fn create_type_data(_input: ()) -> Self {
        Self
    }

    fn insert_dependencies(registration: &mut TypeRegistration) {
        registration.register_type_data::<ReflectComponent, T>();
    }
}

/// Restores the resumed session's agents at startup, saves after every
/// turn, and saves on exit.
#[derive(Default)]
pub struct SavePlugin;

impl Plugin for SavePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PreStartup, restore)
            .add_observer(autosave)
            .add_systems(Last, save_on_exit);
    }
}

/// The session's `state.json`.
#[derive(Serialize, Deserialize)]
struct SavedState {
    version: u32,
    next_effect_id: u64,
    agents: Vec<SavedAgent>,
}

/// One agent: its saved components, by type path.
#[derive(Serialize, Deserialize)]
struct SavedAgent {
    components: BTreeMap<String, Value>,
}

/// Why a saved component was not restored.
#[derive(Debug, thiserror::Error)]
enum Skipped {
    #[error("no registered type has this path")]
    Unknown,
    #[error("the type is not marked as saved")]
    NotSaved,
    #[error(transparent)]
    Invalid(#[from] serde_json::Error),
    #[error("the saved value is incomplete")]
    Incomplete,
}

/// Writes every agent's saved components and the next effect id to the
/// session's `state.json`, replacing it atomically.
pub fn save(world: &mut World) -> Result {
    let registry = registry(world)?;
    let registry = registry.read();
    let mut agents = Vec::new();
    let mut query = world.query_filtered::<Entity, With<Agent>>();
    for entity in query.iter(world) {
        let mut components = BTreeMap::new();
        let agent = world.get_entity(entity)?;
        for (_, info) in world.inspect_entity(entity)? {
            let Some(registration) = info.type_id().and_then(|id| registry.get(id)) else {
                continue;
            };
            if registration.data::<ReflectSaved>().is_none() {
                continue;
            }
            let Some(value) = registration
                .data::<ReflectComponent>()
                .and_then(|component| component.reflect(agent))
            else {
                continue;
            };
            let value = serde_json::to_value(TypedReflectSerializer::new(
                value.as_partial_reflect(),
                &registry,
            ))?;
            components.insert(registration.type_info().type_path().to_owned(), value);
        }
        agents.push(SavedAgent { components });
    }
    let state = SavedState {
        version: 1,
        next_effect_id: effects(world)?.next_id.load(Ordering::Relaxed),
        agents,
    };
    let path = session(world)?.state_path();
    let partial = path.with_extension("json.partial");
    fs::write(&partial, serde_json::to_vec_pretty(&state)?)?;
    fs::rename(&partial, &path)?;
    Ok(())
}

/// Spawns the agents saved in a resumed session, before the first agent
/// would be spawned. Each agent starts complete and idle; its saved
/// components replace the defaults one by one.
fn restore(world: &mut World) -> Result {
    let session = session(world)?;
    if !session.resumed {
        return Ok(());
    }
    let text = match fs::read_to_string(session.state_path()) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    let state: SavedState = serde_json::from_str(&text)?;
    effects(world)?
        .next_id
        .fetch_max(state.next_effect_id, Ordering::Relaxed);
    let registry = registry(world)?;
    let registry = registry.read();
    for saved in state.agents {
        let agent = world.spawn(Agent).id();
        let mut skipped = 0;
        for (path, value) in &saved.components {
            if let Err(reason) = restore_component(world, agent, &registry, path, value) {
                warn!("skipped the saved component {path}: {reason}");
                skipped += 1;
            }
        }
        let mut text = "Session restored.".to_owned();
        if skipped > 0 {
            text.push_str(&format!(
                " Skipped {skipped} saved component(s) whose type is gone, such as one from a \
                 removed plugin."
            ));
        }
        world.trigger(Notice::info(agent, text));
    }
    Ok(())
}

fn registry(world: &World) -> Result<AppTypeRegistry> {
    Ok(world
        .get_resource::<AppTypeRegistry>()
        .ok_or("the app has no type registry")?
        .clone())
}

fn effects(world: &World) -> Result<&Effects> {
    Ok(world
        .get_resource::<Effects>()
        .ok_or("the app has no effect recorder")?)
}

fn session(world: &World) -> Result<&Session> {
    Ok(world
        .get_resource::<Session>()
        .ok_or("the app has no session")?)
}

/// Inserts one saved component into `agent`.
fn restore_component(
    world: &mut World,
    agent: Entity,
    registry: &TypeRegistry,
    path: &str,
    value: &Value,
) -> Result<(), Skipped> {
    let registration = registry.get_with_type_path(path).ok_or(Skipped::Unknown)?;
    let (Some(_), Some(component), Some(from_reflect)) = (
        registration.data::<ReflectSaved>(),
        registration.data::<ReflectComponent>(),
        registration.data::<ReflectFromReflect>(),
    ) else {
        return Err(Skipped::NotSaved);
    };
    let partial = TypedReflectDeserializer::new(registration, registry).deserialize(value)?;
    let value = from_reflect
        .from_reflect(partial.as_ref())
        .ok_or(Skipped::Incomplete)?;
    if let Ok(mut agent) = world.get_entity_mut(agent) {
        component.insert(&mut agent, value.as_partial_reflect(), registry);
    }
    Ok(())
}

/// Saves after every turn, so a crash loses at most the turn in progress.
fn autosave(_ended: On<TurnEnded>, mut commands: Commands) {
    commands.queue(save);
}

/// Saves when the app is about to exit, whether it quits or reloads.
fn save_on_exit(mut exits: MessageReader<AppExit>, mut commands: Commands) {
    if exits.read().next().is_some() {
        commands.queue(save);
    }
}
