//! The model and reasoning setting a new session starts with: `[model]`
//! in the app's `settings.toml` ([`Home::settings`], [`RigHome`]), whose
//! `default` and `effort` the user may set by hand, and which the last
//! choice of the user for an agent they talk to rewrites once that agent's
//! model answered with it. A choice made for one run, such as by
//! `--model`, an editor or an automatic pick, is never kept; a restored
//! session keeps its own, and subagents never change it. Only the user's
//! own file is read: a project cannot pick the model.
//!
//! ```toml
//! [model]
//! default = "openai/gpt-5.5"
//! effort = "high"   # default, off, a level, or "1024 tokens"
//! ```

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::warn;
use rig_core::completion::{Effort, Reasoning};
use rig_harness::harness_protocol::Home;
use rig_harness::harness_protocol::config::{Error, SettingsFile, edit, set_text, table};
use rig_harness::host::launcher::Invoked;
use rig_harness::services::RigHome;
use rig_harness::settings::AppSettingsExt;
use serde::Deserialize;

use rig_ecs::agent::{Agent, SpawnedBy, TurnOutcome};
use rig_ecs::model::{ChoiceKind, ChoiceTaken, ModelChoice, ReasoningSetting};
use rig_ecs::stage::{ForAgent, Settled};

/// Remembers the model and reasoning setting the user last chose, once
/// they answered, and gives them to an agent that starts without a model.
#[derive(Default)]
pub struct DefaultsPlugin;

impl Plugin for DefaultsPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<ModelTable>("model")
            .add_systems(PostStartup, apply_defaults)
            .add_observer(hold)
            .add_observer(remember);
    }
}

/// `[model]` in `settings.toml`, as written; read as a setting so a table
/// that does not read, such as an `effort` no setting is named, refuses the
/// run.
#[derive(Resource, Deserialize, Default, Clone)]
#[serde(default, deny_unknown_fields)]
struct ModelTable {
    default: Option<String>,
    #[serde(deserialize_with = "effort_word")]
    effort: Option<String>,
}

/// An `effort` that names a reasoning setting.
fn effort_word<'de, D: serde::Deserializer<'de>>(words: D) -> Result<Option<String>, D::Error> {
    let word = String::deserialize(words)?;
    match effort(&word) {
        Some(_) => Ok(Some(word)),
        None => Err(serde::de::Error::custom(format!(
            "`{word}` is not `default`, `off`, a level or `<n> tokens`"
        ))),
    }
}

#[derive(Default, PartialEq)]
struct Defaults {
    model: Option<ModelChoice>,
    effort: ReasoningSetting,
}

impl From<&ModelTable> for Defaults {
    /// An effort that does not read is the provider's default.
    fn from(table: &ModelTable) -> Self {
        Self {
            model: table.default.clone().map(ModelChoice),
            effort: table.effort.as_deref().and_then(effort).unwrap_or_default(),
        }
    }
}

/// The reasoning setting `word` names, as [`ReasoningSetting`] shows it.
fn effort(word: &str) -> Option<ReasoningSetting> {
    let word = word.trim();
    let reasoning = match word {
        "default" => return Some(ReasoningSetting(None)),
        "off" => Reasoning::Off,
        _ => match word.strip_suffix(" tokens") {
            Some(tokens) => Reasoning::Budget {
                tokens: tokens.trim().parse().ok()?,
            },
            None => Reasoning::Effort(serde_json::from_value::<Effort>(word.into()).ok()?),
        },
    };
    Some(ReasoningSetting(Some(reasoning)))
}

impl Defaults {
    /// The ones in `home`'s `settings.toml` now.
    fn read(home: &Home) -> Self {
        let file = SettingsFile::read(&home.settings()).ok();
        let table = file.as_ref().and_then(|file| file.document.get("model"));
        let text = |key: &str| {
            let value = table.and_then(|table| table.get(key));
            value.and_then(|value| value.as_str()).map(str::to_owned)
        };
        Self::from(&ModelTable {
            default: text("default"),
            effort: text("effort"),
        })
    }

    /// Writes them into `[model]` of `home`'s `settings.toml`, keeping the
    /// rest of the file and the comments beside them.
    fn write(&self, home: &Home) {
        let path = home.settings();
        let written = edit(&path, |file| {
            let Some(table) = table(&mut file.document, "model") else {
                return Err(Error::Table {
                    path: file.path.clone(),
                    table: "model".to_owned(),
                    problem: "not a table".to_owned(),
                });
            };
            let mut changed = false;
            let keys = [
                ("default", self.model.as_ref().map(|model| model.0.clone())),
                ("effort", Some(self.effort.to_string())),
            ];
            for (key, value) in keys {
                let Some(value) = value else { continue };
                changed |= set_text(table, key, &value);
            }
            Ok((changed, ()))
        });
        if let Err(failure) = written {
            warn!(
                "could not remember the model in {}: {failure}",
                path.display()
            );
        }
    }
}

/// On an agent the user chose a model or reasoning setting for that its
/// model has not answered with yet: the model the user chose in this run,
/// if they chose one.
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
/// answers. A model chosen for this run only drops what was held: those
/// choices were made for the model it replaces.
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
        (ChoiceKind::ForThisRun, Some(_)) => {
            commands.entity(taken.entity).remove::<Unsaved>();
            return;
        }
        (ChoiceKind::ForThisRun, None) => return,
    };
    commands.entity(taken.entity).insert(Unsaved { model });
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
