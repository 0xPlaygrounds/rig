//! Models from rig-core's catalog: which ones have a credential, how an
//! agent's choice becomes a live endpoint, and which reasoning settings a
//! model takes.

use std::path::PathBuf;

use bevy::prelude::*;
use rig_core::catalog::{Catalog, ModelSpec};
use rig_core::completion::{Effort, Reasoning};
use rig_core::providers::registry::ModelSelector;
use rig_core::serve::ErasedHandler;
use rig_core::serve::adapters::ModelAdapter;
use serde::{Deserialize, Serialize};

use super::agent::{EffortChoice, ModelChoice};
use super::registry::Notice;
use super::session::write_atomic;

/// The live model behind an agent's [`ModelChoice`]: its catalog entry and
/// the handler that serves its completions. Rebuilt whenever `ModelChoice`
/// is inserted, so it is never saved.
#[derive(Component, Clone)]
pub struct ModelEndpoint {
    /// The catalog entry.
    pub spec: &'static ModelSpec,
    /// Serves completions, keyed `model:<vendor/model>`.
    pub handler: ErasedHandler,
}

/// The model and effort a new agent starts with: the last ones picked.
/// Saved as `defaults.json` in the data directory.
#[derive(Resource, Default, Clone, Serialize, Deserialize)]
pub struct AgentDefaults {
    /// Catalog reference of the model.
    pub model: Option<String>,
    /// Reasoning setting.
    pub effort: Option<Reasoning>,
}

impl AgentDefaults {
    fn path(data: &std::path::Path) -> PathBuf {
        data.join("defaults.json")
    }

    /// The saved defaults, or none when there are none.
    pub(crate) fn load(data: &std::path::Path) -> Self {
        std::fs::read(Self::path(data))
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .unwrap_or_default()
    }
}

/// Writes the defaults whenever a command changed them.
pub(crate) fn save_defaults(defaults: Res<AgentDefaults>, data: Res<super::app::DataDir>) {
    let written = serde_json::to_vec_pretty(&*defaults)
        .map_err(std::io::Error::from)
        .and_then(|bytes| write_atomic(&AgentDefaults::path(&data.0), &bytes));
    if let Err(error) = written {
        error!("cannot save the agent defaults: {error}");
    }
}

/// The catalog reference of `spec`: `vendor/model`.
pub fn model_reference(spec: &ModelSpec) -> String {
    format!("{}/{}", spec.provider.vendor(), spec.id)
}

/// Catalog models that call tools and whose provider's API key variable is
/// set, in catalog order.
pub fn available_models() -> impl Iterator<Item = &'static ModelSpec> {
    Catalog::builtin().iter().filter(|spec| {
        spec.tools
            && spec.provider.is_registered()
            && spec
                .provider
                .api_key_env()
                .and_then(std::env::var_os)
                .is_some_and(|value| !value.is_empty())
    })
}

/// Token budgets a budget-only model gets for named levels.
const BUDGETS: [(&str, u32); 4] = [
    ("minimal", 1024),
    ("low", 2048),
    ("medium", 8192),
    ("high", 16384),
];

/// The reasoning settings `spec` takes, labelled: `off` when reasoning can
/// be turned off, then its effort levels, or for a model that takes a token
/// budget instead, named levels mapped to budgets that fit its range and
/// leave at least 1024 tokens for the answer.
pub fn effort_options(spec: &ModelSpec) -> Vec<(String, Reasoning)> {
    let reasoning = &spec.reasoning;
    if !reasoning.supported {
        return Vec::new();
    }
    let mut options = Vec::new();
    if reasoning.can_disable {
        options.push(("off".to_owned(), Reasoning::Off));
    }
    for effort in &reasoning.levels {
        options.push((effort.as_str().to_owned(), Reasoning::Effort(*effort)));
    }
    if let (true, Some(range)) = (reasoning.levels.is_empty(), &reasoning.budget) {
        let ceiling = spec
            .max_output_tokens
            .map_or(*range.end(), |max| max.saturating_sub(1024));
        for (label, tokens) in BUDGETS {
            let tokens = tokens.clamp(*range.start(), *range.end()).min(ceiling);
            if range.contains(&tokens)
                && options
                    .iter()
                    .all(|(_, known)| *known != Reasoning::Budget { tokens })
            {
                options.push((label.to_owned(), Reasoning::Budget { tokens }));
            }
        }
    }
    options
}

/// How `reasoning` is shown for `spec`.
pub fn effort_label(spec: &ModelSpec, reasoning: &Reasoning) -> String {
    effort_options(spec)
        .into_iter()
        .find(|(_, option)| option == reasoning)
        .map(|(label, _)| label)
        .unwrap_or_else(|| match reasoning {
            Reasoning::Budget { tokens } => format!("{tokens} tokens"),
            _ => "custom".to_owned(),
        })
}

/// The setting a model starts with: its documented default effort when it
/// takes it.
fn default_effort(spec: &ModelSpec) -> Option<Reasoning> {
    spec.reasoning
        .default
        .filter(|effort| spec.reasoning.levels.contains(effort))
        .map(Reasoning::Effort)
        .or_else(|| {
            spec.reasoning
                .levels
                .contains(&Effort::Medium)
                .then_some(Reasoning::Effort(Effort::Medium))
        })
}

/// Builds the [`ModelEndpoint`] for an agent's new [`ModelChoice`], with
/// credentials from the environment, and resets an effort the new model
/// does not take.
pub(crate) fn connect_model(
    insert: On<Insert<ModelChoice>>,
    mut commands: Commands,
    agents: Query<(&ModelChoice, &EffortChoice)>,
    mut notices: MessageWriter<Notice>,
) {
    let agent = insert.entity;
    let Ok((choice, effort)) = agents.get(agent) else {
        return;
    };
    let Some(reference) = &choice.0 else {
        commands.entity(agent).remove::<ModelEndpoint>();
        return;
    };
    let Some(spec) = Catalog::builtin().resolve(reference) else {
        commands.entity(agent).remove::<ModelEndpoint>();
        notices.write(Notice::error(
            agent,
            format!("{reference} is not in the model catalog; pick one with /model"),
        ));
        return;
    };
    let model = match ModelSelector::from(spec)
        .provider_ref()
        .map_err(|error| error.to_string())
        .and_then(|provider| {
            provider
                .completion_model()
                .map_err(|error| error.to_string())
        }) {
        Ok(model) => model,
        Err(error) => {
            commands.entity(agent).remove::<ModelEndpoint>();
            notices.write(Notice::error(
                agent,
                format!("cannot connect to {reference}: {error}"),
            ));
            return;
        }
    };
    if effort
        .0
        .as_ref()
        .is_none_or(|effort| spec.reasoning.refusal(effort).is_some())
    {
        commands
            .entity(agent)
            .insert(EffortChoice(default_effort(spec)));
    }
    commands.entity(agent).insert(ModelEndpoint {
        spec,
        handler: ErasedHandler::new(ModelAdapter::new(reference.as_str(), model)),
    });
}
