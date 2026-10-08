//! The session: its id, its directory, the log file in it, and saving and
//! restoring its agents.
//!
//! Every session writes to `<data>/sessions/<id>/`. The data directory is
//! `RIG_DATA_DIR` when set, then `$RIG_HOME/data`, then the platform's data
//! directory under `rig`. Logs go to `rig-code.log` there, never to the
//! terminal the view draws on. Agents are saved to `state.json` through
//! reflection, one entry per component keyed by type path, so a component
//! whose plugin is gone is skipped on restore instead of failing it.

use std::{
    path::PathBuf,
    sync::Mutex,
    time::{SystemTime, UNIX_EPOCH},
};

use bevy_app::{App, Last, Plugin};
use bevy_ecs::{prelude::*, reflect::AppTypeRegistry, reflect::ReflectComponent};
use bevy_log::{BoxedFmtLayer, tracing_subscriber};
use bevy_reflect::{
    ReflectFromReflect,
    serde::{TypedReflectDeserializer, TypedReflectSerializer},
};
use serde::de::DeserializeSeed as _;
use serde_json::{Map, Value};

use crate::{
    agent::{Agent, Conversation, Notice, TurnEnded},
    effects::EffectHub,
};

/// The version of the `state.json` layout.
const STATE_FORMAT: u64 = 1;

/// The running session.
#[derive(Resource, Debug, Clone)]
pub struct Session {
    /// The session id: `RIG_SESSION` when set, else `<unix seconds>-<pid>`.
    pub id: String,
    /// The directory holding the session's files.
    pub dir: PathBuf,
}

impl Session {
    /// The session the environment names, with its directory created.
    pub fn from_env() -> Self {
        let id = std::env::var("RIG_SESSION").unwrap_or_else(|_| {
            let secs = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |elapsed| elapsed.as_secs());
            format!("{secs}-{}", std::process::id())
        });
        let dir = data_dir().join("sessions").join(&id);
        // A session that cannot create its directory still runs; its files
        // fail to open and are skipped.
        let _ = std::fs::create_dir_all(&dir);
        Self { id, dir }
    }
}

/// The directory for rig's binaries and sessions.
pub fn data_dir() -> PathBuf {
    let var = |name: &str| std::env::var_os(name).filter(|value| !value.is_empty());
    if let Some(dir) = var("RIG_DATA_DIR") {
        return dir.into();
    }
    if let Some(home) = var("RIG_HOME") {
        return PathBuf::from(home).join("data");
    }
    if let Some(data) = var("XDG_DATA_HOME") {
        return PathBuf::from(data).join("rig");
    }
    match var("HOME") {
        Some(home) => PathBuf::from(home).join(".local/share/rig"),
        None => PathBuf::from(".rig"),
    }
}

/// Inserts [`Session`], autosaves after every turn, and marks the session
/// ready after the first frame. Comes before `LogPlugin`, whose file layer
/// reads the session.
pub struct SessionPlugin;

impl Plugin for SessionPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(Session::from_env())
            .add_observer(autosave)
            .add_systems(Last, mark_ready.run_if(run_once));
    }
}

fn autosave(_: On<TurnEnded>, mut commands: Commands) {
    commands.queue(save);
}

/// Tell the launcher the app started: restore ran and a frame completed.
/// A crash before this is a startup crash, which the launcher rolls back.
fn mark_ready(session: Res<Session>) {
    let path = session.dir.join("ready");
    if let Err(error) = std::fs::write(&path, "") {
        bevy_log::warn!("cannot write {}: {error}", path.display());
    }
}

/// `state.json`.
#[derive(serde::Serialize, serde::Deserialize)]
struct SavedState {
    format: u64,
    next_effect_id: u64,
    /// Per agent, each saved component by type path.
    agents: Vec<Map<String, Value>>,
}

/// Save every agent to the session's `state.json`: each component on an
/// [`Agent`] entity whose type is registered with `ReflectComponent`.
/// Written to a temporary file first, then renamed over the old one.
pub fn save(world: &mut World) {
    let Some(session) = world.get_resource::<Session>().cloned() else {
        return;
    };
    let Some(registry) = world.get_resource::<AppTypeRegistry>().cloned() else {
        return;
    };
    let registry = registry.read();
    let next_effect_id = world
        .get_resource::<EffectHub>()
        .map_or(0, |hub| hub.next_id);
    let agents: Vec<Entity> = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .collect();
    let mut saved = Vec::new();
    for agent in agents {
        let Ok(entity) = world.get_entity(agent) else {
            continue;
        };
        let mut components = Map::new();
        for &id in entity.archetype().components() {
            let Some(registration) = world
                .components()
                .get_info(id)
                .and_then(|info| info.type_id())
                .and_then(|type_id| registry.get(type_id))
            else {
                continue;
            };
            let Some(value) = registration
                .data::<ReflectComponent>()
                .and_then(|component| component.reflect(entity))
            else {
                continue;
            };
            let path = registration.type_info().type_path();
            let serializer = TypedReflectSerializer::new(value.as_partial_reflect(), &registry);
            match serde_json::to_value(serializer) {
                Ok(value) => {
                    components.insert(path.to_owned(), value);
                }
                Err(error) => bevy_log::warn!("not saving {path}: {error}"),
            }
        }
        saved.push(components);
    }
    let state = SavedState {
        format: STATE_FORMAT,
        next_effect_id,
        agents: saved,
    };
    let path = session.dir.join("state.json");
    let temporary = session.dir.join("state.json.partial");
    let written = serde_json::to_vec_pretty(&state)
        .map_err(std::io::Error::other)
        .and_then(|bytes| std::fs::write(&temporary, bytes))
        .and_then(|()| std::fs::rename(&temporary, &path));
    if let Err(error) = written {
        bevy_log::warn!("cannot save {}: {error}", path.display());
    }
}

/// Restore the agents saved in the session's `state.json`, or spawn one new
/// agent when there is none. A component whose type no plugin registers, or
/// whose shape changed, is skipped and named in a notice. The launcher's
/// `RIG_NOTICE` is shown too.
pub(crate) fn restore(world: &mut World) {
    let mut notices = Vec::new();
    if let Some(text) = std::env::var("RIG_NOTICE")
        .ok()
        .filter(|text| !text.is_empty())
    {
        notices.push(text);
    }
    let state = world.get_resource::<Session>().and_then(|session| {
        let path = session.dir.join("state.json");
        let text = std::fs::read_to_string(&path).ok()?;
        match serde_json::from_str::<SavedState>(&text) {
            Ok(state) if state.format == STATE_FORMAT => Some(state),
            Ok(state) => {
                notices.push(format!(
                    "Not restoring {}: format {} is not {STATE_FORMAT}.",
                    path.display(),
                    state.format
                ));
                None
            }
            Err(error) => {
                notices.push(format!("Not restoring {}: {error}", path.display()));
                None
            }
        }
    });
    let mut agents = Vec::new();
    if let Some(state) = state {
        if let Some(mut hub) = world.get_resource_mut::<EffectHub>() {
            hub.next_id = hub.next_id.max(state.next_effect_id);
        }
        let mut skipped = Vec::new();
        for components in state.agents {
            agents.push(restore_agent(world, components, &mut skipped));
        }
        let messages: usize = agents
            .iter()
            .filter_map(|agent| world.get::<Conversation>(*agent))
            .map(|conversation| conversation.0.len())
            .sum();
        notices.push(format!("Restored the session ({messages} messages)."));
        notices.extend(skipped);
    }
    let first = match agents.first() {
        Some(agent) => *agent,
        None => world.spawn(Agent).id(),
    };
    for text in notices {
        world.write_message(Notice { agent: first, text });
    }
}

/// Spawn one saved agent, inserting each component that still fits.
fn restore_agent(
    world: &mut World,
    components: Map<String, Value>,
    skipped: &mut Vec<String>,
) -> Entity {
    let Some(registry) = world.get_resource::<AppTypeRegistry>().cloned() else {
        return world.spawn(Agent).id();
    };
    let registry = registry.read();
    let mut entity = world.spawn(Agent);
    for (path, value) in components {
        let Some(registration) = registry.get_with_type_path(&path) else {
            skipped.push(format!("Skipped {path}: no plugin provides it."));
            continue;
        };
        let (Some(component), Some(from_reflect)) = (
            registration.data::<ReflectComponent>(),
            registration.data::<ReflectFromReflect>(),
        ) else {
            skipped.push(format!(
                "Skipped {path}: it is no longer a saved component."
            ));
            continue;
        };
        let restored = TypedReflectDeserializer::new(registration, &registry)
            .deserialize(value)
            .ok()
            .and_then(|value| from_reflect.from_reflect(value.as_ref()));
        match restored {
            Some(value) => component.insert(&mut entity, value.as_partial_reflect(), &registry),
            None => skipped.push(format!("Skipped {path}: its shape changed.")),
        }
    }
    entity.id()
}

/// The `LogPlugin` formatter: plain text appended to the session's
/// `rig-code.log`, or nowhere when that file cannot be opened.
pub(crate) fn file_log_layer(app: &mut App) -> Option<BoxedFmtLayer> {
    let file = app.world().get_resource::<Session>().and_then(|session| {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(session.dir.join("rig-code.log"))
            .ok()
    });
    let layer = tracing_subscriber::fmt::Layer::default().with_ansi(false);
    Some(match file {
        Some(file) => Box::new(layer.with_writer(Mutex::new(file))),
        None => Box::new(layer.with_writer(std::io::sink)),
    })
}

/// Route panics to the log instead of stderr. Bevy catches panics in
/// systems, observers and commands; this keeps their messages off the
/// terminal.
pub(crate) fn log_panics() {
    std::panic::set_hook(Box::new(|info| {
        bevy_log::error!("panic: {info}");
    }));
}
