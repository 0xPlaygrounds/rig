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
use rig_tools::fs::write_atomic;
use serde::{Deserialize, Serialize};

use rig_ecs::agent::{Agent, Effort, ModelChoice, SpawnedBy};
use rig_ecs::journal::SessionLog;

/// Remembers the last chosen model and reasoning setting and gives them to
/// an agent that starts without a model.
pub struct DefaultsPlugin;

impl Plugin for DefaultsPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(DefaultsFile(Home::from_env().defaults()))
            .add_systems(PostStartup, apply_defaults)
            .add_observer(remember);
    }
}

/// Where the defaults are kept.
#[derive(Resource)]
struct DefaultsFile(PathBuf);

#[derive(Serialize, Deserialize, Default, PartialEq)]
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

    /// Writes `defaults` with [`write_atomic`], so a crash never leaves
    /// half a file.
    fn write(&self, defaults: &Defaults) {
        let written = serde_json::to_vec_pretty(defaults)
            .map_err(std::io::Error::other)
            .and_then(|bytes| write_atomic(&self.0, &bytes));
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
        commands
            .entity(agent)
            .insert((defaults.effort, ModelChoice(model.clone())));
    }
}

/// Remembers the model and reasoning of an agent the user talks to when
/// either changes, such as by `/model` or `/effort`. Restoring a session
/// inserts them before the session is logged, which is not a choice, and
/// subagents are not remembered.
fn remember(
    chosen: On<Insert<(ModelChoice, Effort)>>,
    agents: Query<(&ModelChoice, &Effort), Without<SpawnedBy>>,
    log: Res<SessionLog>,
    file: Res<DefaultsFile>,
) {
    let Ok((model, effort)) = agents.get(chosen.entity) else {
        return;
    };
    let defaults = Defaults {
        model: Some(model.0.clone()),
        effort: *effort,
    };
    if log.is_live() && file.read() != defaults {
        file.write(&defaults);
    }
}
