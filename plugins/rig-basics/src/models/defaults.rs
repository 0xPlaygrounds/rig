//! The model and reasoning setting a new session starts with: the last ones
//! the user chose for an agent they talk to, kept in the app's
//! [`Home::defaults`] ([`RigHome`]) once that agent's model answered with
//! them. A choice made for one run, such as by `--model`, an editor or an
//! automatic pick, is never kept; a restored session keeps its own, and
//! subagents never change the defaults.

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::warn;
use rig_harness::harness_protocol::Home;
use rig_harness::host::launcher::Invoked;
use rig_harness::services::RigHome;
use rig_tools::fs::{read_json, write_json};
use serde::{Deserialize, Serialize};

use rig_ecs::agent::{Agent, SpawnedBy, TurnOutcome};
use rig_ecs::model::{ChoiceKind, ChoiceTaken, ModelChoice, ReasoningSetting};
use rig_ecs::stage::{ForAgent, Settled};

/// Remembers the model and reasoning setting the user last chose, once
/// they answered, and gives them to an agent that starts without a model.
#[derive(Default)]
pub struct DefaultsPlugin;

impl Plugin for DefaultsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PostStartup, apply_defaults)
            .add_observer(hold)
            .add_observer(remember);
    }
}

#[derive(Serialize, Deserialize, Default, PartialEq)]
struct Defaults {
    model: Option<ModelChoice>,
    #[serde(default)]
    effort: ReasoningSetting,
}

impl Defaults {
    /// The ones kept in `home`.
    fn read(home: &Home) -> Self {
        read_json(&home.defaults()).unwrap_or_default()
    }

    /// Keeps them in `home` with [`write_json`], so a crash never leaves
    /// half a file.
    fn write(&self, home: &Home) {
        let path = home.defaults();
        if let Err(failure) = write_json(&path, self) {
            warn!(
                "could not remember the model in {}: {failure}",
                path.display()
            );
        }
    }
}

/// On an agent the user chose a model or reasoning setting for that its
/// model has not answered with yet: the model the user chose in this run,
/// if one is still the agent's.
#[derive(Component, Debug)]
struct Unsaved {
    model: Option<ModelChoice>,
}

/// Gives each agent the user talks to that has no model yet, such as the
/// first agent of a new session, the remembered model and reasoning; only
/// the reasoning when `--model` names the model.
fn apply_defaults(
    agents: Query<Entity, (With<Agent>, Without<ModelChoice>, Without<SpawnedBy>)>,
    invoked: Option<Res<Invoked>>,
    home: Res<RigHome>,
    mut commands: Commands,
) {
    if agents.is_empty() {
        return;
    }
    let defaults = Defaults::read(&home.0);
    let Some(model) = defaults.model else {
        return;
    };
    let named = invoked.is_some_and(|invoked| invoked.args.model.is_some());
    for agent in &agents {
        let mut agent = commands.entity(agent);
        agent.insert(defaults.effort);
        if !named {
            agent.insert(model.clone());
        }
    }
}

/// Holds a choice the user made for an agent they talk to until its model
/// answers; a model chosen for this run only replaces the user's.
fn hold(
    taken: On<ChoiceTaken>,
    agents: Query<Option<&Unsaved>, (With<Agent>, Without<SpawnedBy>)>,
    mut commands: Commands,
) {
    let Ok(unsaved) = agents.get(taken.entity) else {
        return;
    };
    let held = unsaved.and_then(|unsaved| unsaved.model.clone());
    let model = match (taken.kind, &taken.model) {
        (ChoiceKind::ByUser, Some(model)) => Some(model.clone()),
        (ChoiceKind::ByUser, None) => held,
        (ChoiceKind::ForThisRun, Some(_)) => None,
        (ChoiceKind::ForThisRun, None) => return,
    };
    if taken.kind == ChoiceKind::ByUser || unsaved.is_some() {
        commands.entity(taken.entity).insert(Unsaved { model });
    }
}

/// Once a turn of an agent holding a choice is answered, remembers its
/// model and reasoning setting if its model is the user's: the one chosen
/// in this run, else the one remembered. A failed or stopped turn keeps
/// the choice held. A failed write does not stop it.
fn remember(
    settled: On<Add<Settled>>,
    turns: Query<(&TurnOutcome, &ForAgent)>,
    agents: Query<(&Unsaved, &ModelChoice, &ReasoningSetting)>,
    home: Res<RigHome>,
    mut commands: Commands,
) {
    let Ok((TurnOutcome::Answered(_), &ForAgent(agent))) = turns.get(settled.entity) else {
        return;
    };
    let Ok((unsaved, model, effort)) = agents.get(agent) else {
        return;
    };
    commands.entity(agent).remove::<Unsaved>();
    let kept = Defaults::read(&home.0);
    let users = unsaved.model.as_ref().or(kept.model.as_ref());
    if users != Some(model) {
        return;
    }
    let defaults = Defaults {
        model: Some(model.clone()),
        effort: *effort,
    };
    if kept != defaults {
        defaults.write(&home.0);
    }
}
