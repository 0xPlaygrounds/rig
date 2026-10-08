//! The session on disk: where it lives, the log file, and saving and
//! restoring agents through reflection. A component is saved when its type
//! is reflected with `#[reflect(Component, Saved)]`; a saved component whose
//! type is gone is skipped on restore.

use std::collections::BTreeMap;
use std::error::Error;
use std::fs::{self, OpenOptions};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use bevy_app::OnAppExitSystems;
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::tracing_subscriber::fmt;
use bevy_log::{BoxedFmtLayer, error};
use bevy_reflect::serde::{TypedReflectDeserializer, TypedReflectSerializer};
use bevy_reflect::{CreateTypeData, ReflectFromReflect, TypeRegistry};
use serde::de::DeserializeSeed;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::agent::{Agent, AgentId, Notice, TurnFinished};
use super::effects::Effects;

/// Type data marking a component as part of the saved session. Derive it
/// with `#[reflect(Component, Saved)]`.
#[derive(Clone)]
pub struct ReflectSaved;

impl<T> CreateTypeData<T> for ReflectSaved {
    fn create_type_data(_input: ()) -> Self {
        Self
    }
}

/// Where the session's files live: `$RIG_HOME/sessions/$RIG_SESSION/`.
#[derive(Resource, Clone, Debug)]
pub struct SessionPaths {
    /// The session id.
    pub id: String,
    /// The session directory.
    pub dir: PathBuf,
}

impl SessionPaths {
    /// The paths from `RIG_HOME` (default `$HOME/.rig`) and `RIG_SESSION`
    /// (default a new id), with the directory created.
    pub fn from_env() -> Self {
        let home = std::env::var_os("RIG_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".rig")))
            .unwrap_or_else(|| PathBuf::from(".rig"));
        // The launcher's own default and id format (`src/launcher` in the
        // `rig` crate); these apply when the agent runs without it.
        let id = std::env::var("RIG_SESSION").unwrap_or_else(|_| {
            let seconds = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|elapsed| elapsed.as_secs())
                .unwrap_or_default();
            format!("{seconds}-{}", std::process::id())
        });
        let dir = home.join("sessions").join(&id);
        // A directory that cannot be created shows up as a failed save.
        fs::create_dir_all(&dir).ok();
        Self { id, dir }
    }

    /// The saved agents.
    pub fn state(&self) -> PathBuf {
        self.dir.join("state.json")
    }

    /// The effect log, one effect record per line.
    pub fn effects(&self) -> PathBuf {
        self.dir.join("effects.jsonl")
    }

    /// The text log.
    pub fn log(&self) -> PathBuf {
        self.dir.join("agent.log")
    }
}

/// Owns the session paths, restore at startup, autosave after each turn and
/// save on exit, and routes panics to the log.
pub struct SessionPlugin;

impl Plugin for SessionPlugin {
    fn build(&self, app: &mut App) {
        std::panic::set_hook(Box::new(|info| {
            error!("{info}\n{}", std::backtrace::Backtrace::capture());
        }));
        app.insert_resource(SessionPaths::from_env())
            .add_systems(PreStartup, restore_session)
            .add_systems(
                Last,
                save_session
                    .in_set(OnAppExitSystems)
                    .run_if(on_message::<TurnFinished>.or_eager(on_message::<AppExit>)),
            );
    }
}

/// The `LogPlugin` formatter: plain text appended to the session's log, so
/// nothing is written to stderr.
pub fn log_layer(app: &mut App) -> Option<BoxedFmtLayer> {
    let file = app
        .world()
        .get_resource::<SessionPaths>()
        .and_then(|paths| {
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(paths.log())
                .ok()
        });
    let layer = fmt::Layer::default().with_ansi(false);
    Some(match file {
        Some(file) => Box::new(layer.with_writer(Mutex::new(file))),
        None => Box::new(layer.with_writer(std::io::sink)),
    })
}

#[derive(Serialize, Deserialize)]
struct SavedState {
    format: u32,
    session: String,
    agents: Vec<SavedAgent>,
}

#[derive(Serialize, Deserialize)]
struct SavedAgent {
    id: String,
    components: BTreeMap<String, Value>,
}

/// Writes every agent's saved components to `state.json` and appends the
/// resolved effects to `effects.jsonl`.
pub fn save_session(world: &mut World) {
    let (Some(paths), Some(registry)) = (
        world.get_resource::<SessionPaths>().cloned(),
        world.get_resource::<AppTypeRegistry>().cloned(),
    ) else {
        return;
    };
    let registry = registry.read();
    let mut agents: Vec<(Entity, String)> = world
        .query_filtered::<(Entity, &AgentId), With<Agent>>()
        .iter(world)
        .map(|(entity, id)| (entity, id.0.clone()))
        .collect();
    agents.sort_by(|a, b| a.1.cmp(&b.1));
    let state = SavedState {
        format: 1,
        session: paths.id.clone(),
        agents: agents
            .into_iter()
            .map(|(entity, id)| SavedAgent {
                id,
                components: saved_components(world, entity, &registry),
            })
            .collect(),
    };
    let written =
        write_state(&paths, &state).and_then(|()| match world.get_resource::<Effects>() {
            Some(effects) => effects.flush(&paths.effects()).map_err(Into::into),
            None => Ok(()),
        });
    if let Err(failure) = written {
        error!("saving the session failed: {failure}");
        world.write_message(Notice(format!("Saving the session failed: {failure}")));
    }
}

fn saved_components(
    world: &World,
    entity: Entity,
    registry: &TypeRegistry,
) -> BTreeMap<String, Value> {
    let Ok(entity) = world.get_entity(entity) else {
        return BTreeMap::new();
    };
    registry
        .iter_with_data::<ReflectSaved>()
        .filter_map(|(registration, _)| {
            let value = registration.data::<ReflectComponent>()?.reflect(entity)?;
            let path = registration.type_info().type_path().to_owned();
            match serde_json::to_value(TypedReflectSerializer::new(
                value.as_partial_reflect(),
                registry,
            )) {
                Ok(value) => Some((path, value)),
                Err(failure) => {
                    error!("not saving {path}: {failure}");
                    None
                }
            }
        })
        .collect()
}

fn write_state(paths: &SessionPaths, state: &SavedState) -> Result<(), Box<dyn Error>> {
    let temporary = paths.dir.join("state.json.tmp");
    fs::write(&temporary, serde_json::to_vec_pretty(state)?)?;
    fs::rename(&temporary, paths.state())?;
    Ok(())
}

/// Spawns the agents of `state.json`, if there is one, with every saved
/// component whose type is still registered. Anything that does not load
/// is skipped with a notice.
pub fn restore_session(world: &mut World) {
    let (Some(paths), Some(registry)) = (
        world.get_resource::<SessionPaths>().cloned(),
        world.get_resource::<AppTypeRegistry>().cloned(),
    ) else {
        return;
    };
    let text = match fs::read_to_string(paths.state()) {
        Ok(text) => text,
        Err(failure) if failure.kind() == std::io::ErrorKind::NotFound => return,
        Err(failure) => {
            world.write_message(Notice(format!(
                "Could not read the saved session: {failure}"
            )));
            return;
        }
    };
    let state: SavedState = match serde_json::from_str(&text) {
        Ok(state) => state,
        Err(failure) => {
            world.write_message(Notice(format!(
                "The saved session does not load ({failure}); starting fresh."
            )));
            return;
        }
    };
    let registry = registry.read();
    for saved in state.agents {
        let entity = world.spawn((Agent, AgentId(saved.id))).id();
        for (path, value) in saved.components {
            if let Err(failure) = restore_component(world, entity, &registry, &path, value) {
                world.write_message(Notice(format!(
                    "Skipped saved component `{path}`: {failure}."
                )));
            }
        }
    }
}

fn restore_component(
    world: &mut World,
    entity: Entity,
    registry: &TypeRegistry,
    path: &str,
    value: Value,
) -> Result<(), Box<dyn Error>> {
    let registration = registry
        .get_with_type_path(path)
        .ok_or("its plugin is not loaded")?;
    let component = registration
        .data::<ReflectComponent>()
        .filter(|_| registration.data::<ReflectSaved>().is_some())
        .ok_or("it is no longer a saved component")?;
    let value = TypedReflectDeserializer::new(registration, registry).deserialize(value)?;
    let value = match registration.data::<ReflectFromReflect>() {
        Some(from_reflect) => from_reflect
            .from_reflect(value.as_ref())
            .ok_or("its saved value no longer fits the type")?
            .into_partial_reflect(),
        None => value,
    };
    let mut entity = world.get_entity_mut(entity)?;
    component.insert(&mut entity, value.as_ref(), registry);
    Ok(())
}
