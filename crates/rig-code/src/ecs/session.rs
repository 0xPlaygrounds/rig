//! Saving and restoring the agents through reflection.
//!
//! A component is saved when its type registration carries [`ReflectSave`],
//! which `#[reflect(Save)]` adds next to `#[derive(Reflect)]`. Each agent is
//! stored in `state.json` under its [`AgentId`], one entry per component keyed
//! by type path, so a component whose plugin is gone is skipped on load
//! instead of failing it. The session is saved after every turn and on exit,
//! and restored at startup.
//!
//! ```
//! use rig_code::{bevy::prelude::*, ecs::session::ReflectSave};
//!
//! #[derive(Component, Reflect, Default)]
//! #[reflect(Component, Default, Save)]
//! struct Greetings {
//!     count: u32,
//! }
//! ```

use std::fs;

use bevy::{
    app::OnAppExitSystems,
    prelude::*,
    reflect::{
        CreateTypeData, TypeRegistry,
        serde::{TypedReflectDeserializer, TypedReflectSerializer},
        std_traits::ReflectDefault,
    },
};
use serde::de::DeserializeSeed;
use serde_json::{Map, Value, json};

use super::{
    Notice, TurnEnded,
    agent::{Agent, AgentId},
    paths,
};

/// The version of the `state.json` layout.
const FORMAT: u64 = 1;

/// Type data marking a component as part of the saved session. Saving and
/// loading go through the type's [`ReflectComponent`] and [`ReflectDefault`]:
/// a load starts from the default and applies the saved value onto it, so
/// fields added since the save keep their default.
#[derive(Clone)]
pub struct ReflectSave;

impl<T: Component + Reflect + Default> CreateTypeData<T> for ReflectSave {
    fn create_type_data(_input: ()) -> Self {
        Self
    }
}

/// Restores the session at startup and saves it after each turn and on exit.
pub struct SessionPlugin;

impl Plugin for SessionPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, restore)
            .add_systems(
                Last,
                save.in_set(OnAppExitSystems).run_if(on_message::<AppExit>),
            )
            .add_observer(|_: On<TurnEnded>, mut commands: Commands| commands.queue(save));
    }
}

/// Write every agent's saved components to `state.json`, replacing it
/// atomically.
pub fn save(world: &mut World) {
    let Some(registry) = world.get_resource::<AppTypeRegistry>().cloned() else {
        return;
    };
    let state = state(world, &registry.read());
    let file = paths::state_file();
    let temporary = file.with_extension("json.tmp");
    let written = serde_json::to_vec_pretty(&state)
        .map_err(std::io::Error::from)
        .and_then(|text| fs::write(&temporary, text))
        .and_then(|()| fs::rename(&temporary, &file));
    if let Err(error) = written {
        error!("cannot save the session to {}: {error}", file.display());
    }
}

fn state(world: &mut World, registry: &TypeRegistry) -> Value {
    let saved = registry
        .iter()
        .filter_map(|registration| {
            registration.data::<ReflectSave>()?;
            let component = registration.data::<ReflectComponent>()?;
            Some((registration.type_info().type_path(), component))
        })
        .collect::<Vec<_>>();
    let mut agents = world.query_filtered::<(Entity, &AgentId), With<Agent>>();
    let world: &World = world;
    let agents = agents
        .iter(world)
        .map(|(entity, id)| {
            let components = saved
                .iter()
                .filter_map(|(path, component)| {
                    let value = component.reflect(world.get_entity(entity).ok()?)?;
                    let serializer =
                        TypedReflectSerializer::new(value.as_partial_reflect(), registry);
                    match serde_json::to_value(serializer) {
                        Ok(value) => Some(((*path).to_owned(), value)),
                        Err(error) => {
                            error!("cannot save `{path}`: {error}");
                            None
                        }
                    }
                })
                .collect::<Map<_, _>>();
            json!({ "id": &*id.0, "components": components })
        })
        .collect::<Vec<_>>();
    json!({ "format": FORMAT, "agents": agents })
}

/// Spawn the agents saved in `state.json`, if the session has one. A
/// component whose type is not registered, or whose value no longer fits
/// it, is skipped with a notice.
fn restore(world: &mut World) {
    let file = paths::state_file();
    let Ok(text) = fs::read_to_string(&file) else {
        return;
    };
    let state = match serde_json::from_str::<Value>(&text) {
        Ok(state) => state,
        Err(error) => {
            error!("cannot read the saved session {}: {error}", file.display());
            return;
        }
    };
    let Some(registry) = world.get_resource::<AppTypeRegistry>().cloned() else {
        return;
    };
    let registry = registry.read();
    let agents = state.get("agents").and_then(Value::as_array);
    for agent in agents.into_iter().flatten() {
        let Some(id) = agent.get("id").and_then(Value::as_str) else {
            continue;
        };
        let entity = world.spawn((Agent, AgentId(id.into()))).id();
        let components = agent.get("components").and_then(Value::as_object);
        for (path, value) in components.into_iter().flatten() {
            if let Err(reason) = load(world, entity, &registry, path, value) {
                world.write_message(Notice::error(
                    entity,
                    format!("Skipped saved `{path}`: {reason}"),
                ));
            }
        }
    }
}

fn load(
    world: &mut World,
    entity: Entity,
    registry: &TypeRegistry,
    path: &str,
    value: &Value,
) -> Result<(), String> {
    let registration = registry
        .get_with_type_path(path)
        .ok_or("its plugin is not loaded")?;
    registration
        .data::<ReflectSave>()
        .ok_or("it is no longer saved")?;
    let (Some(component), Some(default)) = (
        registration.data::<ReflectComponent>(),
        registration.data::<ReflectDefault>(),
    ) else {
        return Err("it is not a component with a default".to_owned());
    };
    let value = TypedReflectDeserializer::new(registration, registry)
        .deserialize(value)
        .map_err(|error| error.to_string())?;
    let mut loaded = default.default();
    loaded
        .try_apply(value.as_ref())
        .map_err(|error| error.to_string())?;
    let mut entity = world
        .get_entity_mut(entity)
        .map_err(|error| error.to_string())?;
    component.insert(&mut entity, loaded.as_partial_reflect(), registry);
    Ok(())
}
