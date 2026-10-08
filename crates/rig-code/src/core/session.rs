//! Saving and restoring agents through reflection.
//!
//! Every reflected component on an agent entity is written to
//! `sessions/<id>/state.json` as `{type path: value}`. Loading inserts the
//! entries one by one, so a component whose plugin is gone, or whose shape
//! changed, is skipped with a notice instead of failing the load.
//!
//! The launcher handshake lives here too: `data/resume` names the running
//! session from its start until a clean exit, so the next binary restores
//! it after a reload or a crash, and the first finished frame writes
//! `RIG_READY_FILE` so the launcher keeps this binary.

use std::path::{Path, PathBuf};

use bevy::ecs::reflect::{AppTypeRegistry, ReflectComponent};
use bevy::prelude::*;
use bevy::reflect::serde::{TypedReflectDeserializer, TypedReflectSerializer};
use bevy::reflect::{ReflectFromReflect, TypeRegistry};
use serde::de::DeserializeSeed;
use serde_json::{Map, Value};

use super::agent::{Agent, EffortChoice, ModelChoice, random_u64};
use super::app::DataDir;
use super::dispatch::Effects;
use super::models::AgentDefaults;
use super::registry::Notice;

/// Version of the `state.json` layout.
const FORMAT: u64 = 1;

/// The exit code that asks the `rig` launcher to start the newest build.
pub const RELOAD_EXIT_CODE: u8 = 75;

/// The running session: its id and directory.
#[derive(Resource, Clone)]
pub struct Session {
    /// `<unix seconds>-<4 hex digits>`.
    pub id: String,
    /// `<data>/sessions/<id>`.
    pub dir: PathBuf,
}

impl Session {
    fn open(data: &Path, id: String) -> Self {
        Self {
            dir: data.join("sessions").join(&id),
            id,
        }
    }

    fn mint(data: &Path) -> Self {
        let seconds = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_secs());
        Self::open(data, format!("{seconds}-{:04x}", random_u64() as u16))
    }
}

/// Writes `bytes` to `path` through a temporary file and a rename, creating
/// the directory.
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("tmp");
    std::fs::write(&temporary, bytes)?;
    std::fs::rename(&temporary, path)
}

/// Starts the session: restores the one named by `data/resume` when that
/// file exists, or starts a new one with one agent. Either way the session
/// is named in `data/resume` until the app exits cleanly.
pub(crate) fn start_session(world: &mut World) {
    let Some(data) = world.get_resource::<DataDir>().map(|data| data.0.clone()) else {
        return;
    };
    let session = resume(world, &data).unwrap_or_else(|| {
        let defaults = world.resource::<AgentDefaults>().clone();
        world.spawn((
            Agent,
            ModelChoice(defaults.model),
            EffortChoice(defaults.effort),
        ));
        Session::mint(&data)
    });
    if let Err(error) = write_atomic(&data.join("resume"), session.id.as_bytes()) {
        error!("cannot write the resume marker: {error}");
    }
    world.insert_resource(session);
}

/// Restores the session `data/resume` names, if it has saved state.
fn resume(world: &mut World, data: &Path) -> Option<Session> {
    let id = std::fs::read_to_string(data.join("resume")).ok()?;
    let session = Session::open(data, id.trim().to_owned());
    // A session that ended before its first save has nothing to restore.
    if session.id.is_empty() || !session.dir.join("state.json").exists() {
        return None;
    }
    match restore(world, &session) {
        Ok(()) => Some(session),
        Err(error) => {
            world.write_message(Notice::error(
                None,
                format!("cannot restore session {}: {error}", session.id),
            ));
            None
        }
    }
}

fn restore(world: &mut World, session: &Session) -> Result<(), Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(session.dir.join("state.json"))?;
    let state: Value = serde_json::from_str(&text)?;
    if state.get("format").and_then(Value::as_u64) != Some(FORMAT) {
        return Err("unknown state format".into());
    }
    if let Some(next) = state.get("next_effect_id").and_then(Value::as_u64) {
        world.resource::<Effects>().resume_at(next);
    }
    let registry = world.resource::<AppTypeRegistry>().clone();
    let registry = registry.read();
    let agents = state
        .get("agents")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut skipped = Vec::new();
    for components in agents {
        let Value::Object(components) = components else {
            continue;
        };
        let entity = world.spawn(Agent).id();
        for (path, value) in components {
            if let Err(reason) = insert_entry(world, entity, &registry, &path, value) {
                skipped.push(format!("skipped {path}: {reason}"));
            }
        }
    }
    drop(registry);
    for line in skipped {
        warn!("{line}");
        world.write_message(Notice::error(None, line));
    }
    world.write_message(Notice::info(
        None,
        format!("restored session {}", session.id),
    ));
    Ok(())
}

/// Inserts one saved component on `entity`, or says why it cannot.
fn insert_entry(
    world: &mut World,
    entity: Entity,
    registry: &TypeRegistry,
    path: &str,
    value: Value,
) -> Result<(), String> {
    let registration = registry
        .get_with_type_path(path)
        .ok_or("its plugin is not loaded")?;
    let component = registration
        .data::<ReflectComponent>()
        .ok_or("it is not a component")?;
    let partial = TypedReflectDeserializer::new(registration, registry)
        .deserialize(value)
        .map_err(|error| error.to_string())?;
    let value = registration
        .data::<ReflectFromReflect>()
        .and_then(|from| from.from_reflect(partial.as_ref()))
        .ok_or("its saved shape no longer fits")?;
    let mut entity = world
        .get_entity_mut(entity)
        .map_err(|error| error.to_string())?;
    component.insert(&mut entity, value.as_partial_reflect(), registry);
    Ok(())
}

/// Writes every agent's reflected components to the session's
/// `state.json`.
pub(crate) fn save_session(world: &mut World) {
    let Some(session) = world.get_resource::<Session>().cloned() else {
        return;
    };
    let registry = world.resource::<AppTypeRegistry>().clone();
    let registry = registry.read();
    let mut agents = Vec::new();
    let mut query = world.query_filtered::<EntityRef, With<Agent>>();
    for entity in query.iter(world) {
        let mut components = Map::new();
        for (registration, component) in registry.iter_with_data::<ReflectComponent>() {
            let Some(value) = component.reflect(entity) else {
                continue;
            };
            let serializer = TypedReflectSerializer::new(value.as_partial_reflect(), &registry);
            match serde_json::to_value(serializer) {
                Ok(value) => {
                    components.insert(registration.type_info().type_path().to_owned(), value);
                }
                Err(error) => debug!(
                    "not saving {}: {error}",
                    registration.type_info().type_path()
                ),
            }
        }
        agents.push(Value::Object(components));
    }
    drop(registry);
    let state = serde_json::json!({
        "format": FORMAT,
        "session": session.id,
        "next_effect_id": world.resource::<Effects>().next_id(),
        "agents": agents,
    });
    let written = serde_json::to_vec_pretty(&state)
        .map_err(std::io::Error::from)
        .and_then(|bytes| write_atomic(&session.dir.join("state.json"), &bytes));
    if let Err(error) = written {
        error!("cannot save session {}: {error}", session.id);
    }
}

/// On a clean exit, forgets the resume marker, so the next start begins a
/// new session. Any other exit (a reload, an error) keeps it.
pub(crate) fn clear_resume(mut exits: MessageReader<AppExit>, data: Res<DataDir>) {
    if exits.read().any(AppExit::is_success)
        && let Err(error) = std::fs::remove_file(data.0.join("resume"))
        && error.kind() != std::io::ErrorKind::NotFound
    {
        error!("cannot remove the resume marker: {error}");
    }
}

/// After the first full frame, tells the launcher this binary works by
/// creating `RIG_READY_FILE`.
pub(crate) fn mark_ready() {
    if let Some(ready) = std::env::var_os("RIG_READY_FILE").filter(|value| !value.is_empty())
        && let Err(error) = write_atomic(Path::new(&ready), b"ready")
    {
        error!("cannot write the ready file: {error}");
    }
}

/// Shows the launcher's `RIG_NOTICE`, such as a rollback, to the user.
pub(crate) fn launcher_notice(mut notices: MessageWriter<Notice>) {
    if let Ok(text) = std::env::var("RIG_NOTICE")
        && !text.is_empty()
    {
        notices.write(Notice::error(None, text));
    }
}
