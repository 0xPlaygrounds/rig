//! The model and reasoning setting a new session starts with: the last ones
//! chosen for an agent the user talks to, kept in
//! [`Home::defaults`](rig::harness_protocol::Home::defaults). A restored
//! session keeps its own; subagents never change the defaults.

use std::fs;
use std::path::PathBuf;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::warn;
use rig::harness_protocol::Home;
use serde::{Deserialize, Serialize};

use crate::core::agent::{Agent, Effort, ModelChoice, SpawnedBy};

/// Remembers the last chosen model and reasoning setting and gives them to
/// an agent that starts without a model.
pub struct DefaultsPlugin;

impl Plugin for DefaultsPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(DefaultsFile(Home::from_env().defaults()))
            .add_systems(PostStartup, apply_defaults)
            .add_observer(remember_model)
            .add_observer(remember_effort);
    }
}

/// Where the defaults are kept.
#[derive(Resource)]
struct DefaultsFile(PathBuf);

#[derive(Serialize, Deserialize, Default)]
struct Defaults {
    model: Option<String>,
    #[serde(default)]
    effort: Effort,
}

impl DefaultsFile {
    fn read(&self) -> Defaults {
        fs::read_to_string(&self.0)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default()
    }

    /// Writes `defaults` through a temporary file, so a crash never leaves
    /// half a file.
    fn write(&self, defaults: &Defaults) {
        let written = serde_json::to_vec_pretty(defaults)
            .map_err(std::io::Error::other)
            .and_then(|bytes| {
                if let Some(parent) = self.0.parent() {
                    fs::create_dir_all(parent)?;
                }
                let temporary = self.0.with_extension("json.tmp");
                fs::write(&temporary, bytes)?;
                fs::rename(&temporary, &self.0)
            });
        if let Err(failure) = written {
            warn!(
                "could not remember the model in {}: {failure}",
                self.0.display()
            );
        }
    }
}

/// Gives each agent the user talks to that has no model yet, such as the
/// first agent of a new session, the remembered model and reasoning.
fn apply_defaults(
    file: Res<DefaultsFile>,
    agents: Query<Entity, (With<Agent>, Without<ModelChoice>, Without<SpawnedBy>)>,
    mut commands: Commands,
) {
    if agents.is_empty() {
        return;
    }
    let defaults = file.read();
    let Some(model) = defaults.model else {
        return;
    };
    for agent in &agents {
        // Effort first: choosing the model checks the effort against it.
        commands
            .entity(agent)
            .insert(defaults.effort)
            .insert(ModelChoice(model.clone()));
    }
}

/// Remembers a model chosen for an agent the user talks to.
fn remember_model(
    chosen: On<Insert<ModelChoice>>,
    agents: Query<(&ModelChoice, &Effort), Without<SpawnedBy>>,
    file: Res<DefaultsFile>,
) {
    if let Ok((model, effort)) = agents.get(chosen.entity) {
        file.write(&Defaults {
            model: Some(model.0.clone()),
            effort: *effort,
        });
    }
}

/// Remembers a reasoning setting chosen for an agent the user talks to.
fn remember_effort(
    chosen: On<Insert<Effort>>,
    agents: Query<(Option<&ModelChoice>, &Effort), Without<SpawnedBy>>,
    file: Res<DefaultsFile>,
) {
    if let Ok((Some(model), effort)) = agents.get(chosen.entity) {
        file.write(&Defaults {
            model: Some(model.0.clone()),
            effort: *effort,
        });
    }
}
