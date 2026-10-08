//! Saving and restoring agents through reflection, under the session
//! directory the host names in [`SessionPaths`]. A component is saved when
//! its type is reflected with `#[reflect(Component, Saved)]`; a saved
//! component whose type is gone is skipped on restore.

use std::any::TypeId;
use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::ops::Deref;

use bevy_app::OnAppExitSystems;
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::error;
use bevy_reflect::serde::{TypedReflectDeserializer, TypedReflectSerializer};
use bevy_reflect::{CreateTypeData, PartialReflect, ReflectFromReflect, TypeRegistry};
use serde::de::DeserializeSeed;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use rig::code_protocol::SessionDir;
use rig_core::message::ToolResult;

use super::agent::{
    ActiveTurn, Agent, AgentId, Calls, Conversation, Effort, ModelChoice, Notice, ToolCallRun,
    TurnFinished,
};
use super::calls::Done;
use super::effects::Effects;
use super::turn::{STOPPED, stopped_results};

/// Type data marking a component as part of the saved session. Derive it
/// with `#[reflect(Component, Saved)]`.
#[derive(Clone)]
pub struct ReflectSaved;

impl<T> CreateTypeData<T> for ReflectSaved {
    fn create_type_data(_input: ()) -> Self {
        Self
    }
}

/// The session's directory, the only place the core writes: the saved
/// agents in [`SessionDir::state`] and the effect log in
/// [`SessionDir::effects`]. Inserted before the agent plugins are built.
#[derive(Resource, Clone, Debug)]
pub struct SessionPaths(pub SessionDir);

impl Deref for SessionPaths {
    type Target = SessionDir;

    fn deref(&self) -> &SessionDir {
        &self.0
    }
}

/// The version of `state.json`'s layout. A file of another version is not
/// restored.
const FORMAT: u32 = 1;

/// Restores the session at startup, and saves it after each turn, after a
/// model or reasoning change, and on exit.
pub struct SavePlugin;

impl Plugin for SavePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PreStartup, restore_session).add_systems(
            Last,
            save_session.in_set(OnAppExitSystems).run_if(
                on_message::<TurnFinished>
                    .or_eager(on_message::<AppExit>)
                    .or_eager(settings_changed),
            ),
        );
    }
}

/// Whether an idle agent's model or reasoning setting changed, so a crash
/// before the next turn ends does not lose it. Both are refused while a
/// turn runs. The first frame's restored or new agents are not a change.
fn settings_changed(
    agents: Query<
        (),
        (
            With<Agent>,
            Without<ActiveTurn>,
            Or<(Changed<ModelChoice>, Changed<Effort>)>,
        ),
    >,
    mut started: Local<bool>,
) -> bool {
    let changed = *started && !agents.is_empty();
    *started = true;
    changed
}

#[derive(Serialize, Deserialize)]
struct SavedState {
    format: u32,
    agents: Vec<SavedAgent>,
}

#[derive(Serialize, Deserialize)]
struct SavedAgent {
    id: String,
    components: BTreeMap<String, Value>,
}

/// Writes every agent's saved components to `state.json` and appends the
/// resolved effects to `effects.jsonl`. An agent in a turn is saved as an
/// interrupt would leave it, so a crash before its turn ends restores a
/// conversation whose tool calls all have results.
pub(crate) fn save_session(world: &mut World) {
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
    let mut runs = world.query::<(&ToolCallRun, Option<&Done<ToolResult>>)>();
    let state = SavedState {
        format: FORMAT,
        agents: agents
            .into_iter()
            .map(|(entity, id)| {
                let settled = settled_conversation(world, entity, &mut runs);
                SavedAgent {
                    id,
                    components: saved_components(world, entity, &registry, settled.as_ref()),
                }
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
        world.write_message(Notice::error(
            None,
            format!("Saving the session failed: {failure}"),
        ));
    }
}

/// The conversation of `agent` with results for its running turn's tool
/// calls, when it has any.
fn settled_conversation(
    world: &World,
    agent: Entity,
    runs: &mut QueryState<(&ToolCallRun, Option<&Done<ToolResult>>)>,
) -> Option<Conversation> {
    let entity = world.get_entity(agent).ok()?;
    let turn = entity.get::<ActiveTurn>()?.turn();
    let calls: Vec<Entity> = world.get::<Calls>(turn)?.iter().collect();
    let mut conversation = entity.get::<Conversation>()?.clone();
    conversation.0.extend(stopped_results(
        runs.iter_many(world, calls).flatten(),
        STOPPED,
    ));
    Some(conversation)
}

/// The saved components of `entity`, with `conversation` in place of its
/// own when given.
fn saved_components(
    world: &World,
    entity: Entity,
    registry: &TypeRegistry,
    conversation: Option<&Conversation>,
) -> BTreeMap<String, Value> {
    let Ok(entity) = world.get_entity(entity) else {
        return BTreeMap::new();
    };
    registry
        .iter_with_data::<ReflectSaved>()
        .filter_map(|(registration, _)| {
            // The settled conversation stands in for the agent's own.
            let value: &dyn PartialReflect = match conversation {
                Some(conversation) if registration.type_id() == TypeId::of::<Conversation>() => {
                    conversation
                }
                _ => registration
                    .data::<ReflectComponent>()?
                    .reflect(entity)?
                    .as_partial_reflect(),
            };
            let path = registration.type_info().type_path().to_owned();
            match serde_json::to_value(TypedReflectSerializer::new(value, registry)) {
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
    let temporary = paths.state().with_extension("json.tmp");
    fs::write(&temporary, serde_json::to_vec_pretty(state)?)?;
    fs::rename(&temporary, paths.state())?;
    Ok(())
}

/// Spawns the agents of `state.json`, if there is one, with every saved
/// component whose type is still registered. Anything that does not load
/// is skipped with a notice.
pub(crate) fn restore_session(world: &mut World) {
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
            world.write_message(Notice::error(
                None,
                format!("Could not read the saved session: {failure}"),
            ));
            return;
        }
    };
    let state: SavedState = match serde_json::from_str(&text) {
        Ok(state) => state,
        Err(failure) => {
            world.write_message(Notice::error(
                None,
                format!("The saved session does not load ({failure}); starting fresh."),
            ));
            return;
        }
    };
    if state.format != FORMAT {
        world.write_message(Notice::error(
            None,
            format!(
                "The saved session has format {}, this build reads {FORMAT}; starting fresh.",
                state.format
            ),
        ));
        return;
    }
    let registry = registry.read();
    for saved in state.agents {
        let entity = world.spawn((Agent, AgentId(saved.id))).id();
        for (path, value) in saved.components {
            if let Err(failure) = restore_component(world, entity, &registry, &path, value) {
                world.write_message(Notice::error(
                    entity,
                    format!("Skipped saved component `{path}`: {failure}."),
                ));
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
