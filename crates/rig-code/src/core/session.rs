//! Saving and restoring agents through reflection.
//!
//! Every reflected component on an agent entity is written to
//! `sessions/<id>/state.json` as `{type path: value}`. Loading inserts the
//! entries one by one, so a component whose plugin is gone, or whose shape
//! changed, is skipped with a notice instead of failing the load.
//!
//! `data/resume/<key of the working directory>` names the session running
//! in that directory from its start until a clean exit, so the next binary
//! started there restores it after a reload or a crash. A running session
//! holds a lock on `sessions/<id>/lock`; a second agent in the same
//! directory finds it held and starts a session of its own.

use std::fs::File;
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

/// Held while the session runs; released when the process ends.
#[derive(Resource)]
struct SessionLock(#[expect(dead_code, reason = "held for its lock")] File);

/// Locks `session`, or `None` when another process holds it.
fn lock(session: &Session) -> Option<File> {
    let file = std::fs::create_dir_all(&session.dir).and_then(|()| {
        File::options()
            .create(true)
            .truncate(false)
            .write(true)
            .open(session.dir.join("lock"))
    });
    match file {
        Ok(file) => match file.try_lock() {
            Ok(()) => Some(file),
            Err(std::fs::TryLockError::WouldBlock) => None,
            Err(std::fs::TryLockError::Error(error)) => {
                warn!("cannot lock session {}: {error}", session.id);
                Some(file)
            }
        },
        Err(error) => {
            warn!("cannot lock session {}: {error}", session.id);
            None
        }
    }
}

/// `data/resume/<key>`, where the key is a hash of the working directory,
/// so a session comes back only in the directory it ran in.
fn resume_marker(data: &Path) -> PathBuf {
    let directory = std::env::current_dir().unwrap_or_default();
    // FNV-1a: stable across builds and toolchains, unlike std's hasher.
    let key = directory
        .as_os_str()
        .as_encoded_bytes()
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
        });
    data.join("resume").join(format!("{key:016x}"))
}

/// Starts the session: restores the one the working directory's resume
/// marker names when it has saved state and no other process runs it, or
/// starts a new one with one agent. The marker names the session until the
/// app exits cleanly, unless it names a session another process runs.
pub(crate) fn start_session(world: &mut World) {
    let Some(data) = world.get_resource::<DataDir>().map(|data| data.0.clone()) else {
        return;
    };
    let marker = resume_marker(&data);
    let previous = std::fs::read_to_string(&marker)
        .ok()
        .map(|id| Session::open(&data, id.trim().to_owned()))
        .filter(|session| !session.id.is_empty());
    let mut taken = false;
    let resumed = previous.and_then(|session| {
        let Some(lock) = lock(&session) else {
            taken = true;
            return None;
        };
        resume(world, &session).then_some((session, Some(lock)))
    });
    let (session, lock) = resumed.unwrap_or_else(|| {
        let defaults = world.resource::<AgentDefaults>().clone();
        world.spawn((
            Agent,
            ModelChoice(defaults.model),
            EffortChoice(defaults.effort),
        ));
        let session = Session::mint(&data);
        let lock = lock(&session);
        (session, lock)
    });
    if !taken && let Err(error) = write_atomic(&marker, session.id.as_bytes()) {
        error!("cannot write the resume marker: {error}");
    }
    if let Some(lock) = lock {
        world.insert_resource(SessionLock(lock));
    }
    world.insert_resource(session);
}

/// Restores `session` if it has saved state.
fn resume(world: &mut World, session: &Session) -> bool {
    // A session that ended before its first save has nothing to restore.
    if !session.dir.join("state.json").exists() {
        return false;
    }
    match restore(world, session) {
        Ok(()) => true,
        Err(error) => {
            world.write_message(Notice::error(
                None,
                format!("cannot restore session {}: {error}", session.id),
            ));
            false
        }
    }
}

/// One past the highest effect id in the session's `effects.jsonl`: a
/// crash can leave effects there that `state.json` never counted.
fn next_logged_effect(session: &Session) -> u64 {
    #[derive(serde::Deserialize)]
    struct Logged {
        id: u64,
    }
    let text = std::fs::read_to_string(session.dir.join("effects.jsonl")).unwrap_or_default();
    text.lines()
        .filter_map(|line| serde_json::from_str::<Logged>(line).ok())
        .map(|logged| logged.id.saturating_add(1))
        .max()
        .unwrap_or(0)
}

fn restore(world: &mut World, session: &Session) -> Result<(), Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(session.dir.join("state.json"))?;
    let state: Value = serde_json::from_str(&text)?;
    if state.get("format").and_then(Value::as_u64) != Some(FORMAT) {
        return Err("unknown state format".into());
    }
    let saved = state
        .get("next_effect_id")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    world
        .resource::<Effects>()
        .resume_at(saved.max(next_logged_effect(session)));
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

/// On a clean exit, forgets the resume marker when it names this session,
/// so the next start begins a new one. Any other exit (a reload, an error)
/// keeps it.
pub(crate) fn clear_resume(
    mut exits: MessageReader<AppExit>,
    data: Res<DataDir>,
    session: Res<Session>,
) {
    if !exits.read().any(AppExit::is_success) {
        return;
    }
    let marker = resume_marker(&data.0);
    if std::fs::read_to_string(&marker).is_ok_and(|id| id.trim() == session.id)
        && let Err(error) = std::fs::remove_file(&marker)
    {
        error!("cannot remove the resume marker: {error}");
    }
}
