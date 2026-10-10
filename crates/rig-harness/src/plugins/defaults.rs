//! The model and reasoning setting a new session starts with: the last ones
//! chosen for an agent the user talks to, kept in
//! [`Home::defaults`](rig::harness_protocol::Home::defaults). A restored
//! session keeps its own; subagents never change the defaults.

use std::fs;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::warn;
use rig::harness_protocol::Home;
use rig_tools::fs::write_atomic;
use serde::{Deserialize, Serialize};

use rig_ecs::agent::{Agent, SpawnedBy};
use rig_ecs::journal::SessionLog;
use rig_ecs::model::{Effort, ModelChoice};

/// Remembers the last chosen model and reasoning setting and gives them to
/// an agent that starts without a model.
#[derive(Default)]
pub struct DefaultsPlugin;

impl Plugin for DefaultsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PostStartup, apply_defaults)
            .add_observer(remember);
    }
}

#[derive(Serialize, Deserialize, Default, PartialEq)]
struct Defaults {
    model: Option<ModelChoice>,
    #[serde(default)]
    effort: Effort,
}

impl Defaults {
    fn read() -> Self {
        fs::read_to_string(Home::from_env().defaults())
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default()
    }

    /// Writes them with [`write_atomic`], so a crash never leaves half a
    /// file.
    fn write(&self) {
        let path = Home::from_env().defaults();
        let written = serde_json::to_vec_pretty(self)
            .map_err(std::io::Error::other)
            .and_then(|bytes| write_atomic(&path, &bytes));
        if let Err(failure) = written {
            warn!(
                "could not remember the model in {}: {failure}",
                path.display()
            );
        }
    }
}

/// Gives each agent the user talks to that has no model yet, such as the
/// first agent of a new session, the remembered model and reasoning.
fn apply_defaults(
    agents: Query<Entity, (With<Agent>, Without<ModelChoice>, Without<SpawnedBy>)>,
    mut commands: Commands,
) {
    if agents.is_empty() {
        return;
    }
    let defaults = Defaults::read();
    let Some(model) = defaults.model else {
        return;
    };
    for agent in &agents {
        commands
            .entity(agent)
            .insert((defaults.effort, model.clone()));
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
) {
    let Ok((model, effort)) = agents.get(chosen.entity) else {
        return;
    };
    let defaults = Defaults {
        model: Some(model.clone()),
        effort: *effort,
    };
    if log.is_live() && Defaults::read() != defaults {
        defaults.write();
    }
}
