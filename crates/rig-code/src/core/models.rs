//! The model catalog as the agent uses it: which models have credentials,
//! which efforts a model takes, and connecting to one.

use rig_core::catalog::{Catalog, ModelSpec};
use rig_core::client::env::EnvError;
use rig_core::completion::options::Reasoning;
use rig_core::effect::model_key;
use rig_core::providers::registry::{ConnectError, ModelSelector};
use rig_core::serve::ErasedHandler;
use rig_core::serve::adapters::ModelAdapter;

use super::agent::Connection;

/// Why a model could not be connected.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    /// The catalog has no such model.
    #[error("`{0}` is not a catalog model; pick one with /model")]
    Unknown(String),
    /// The registry cannot reach the model's provider.
    #[error(transparent)]
    Connect(#[from] ConnectError),
    /// The provider's credentials are missing from the environment.
    #[error(transparent)]
    Env(#[from] EnvError),
}

/// A model's catalog id, `vendor/model`.
pub fn model_id(spec: &ModelSpec) -> String {
    format!("{}/{}", spec.provider.vendor(), spec.id)
}

/// The catalog models whose provider's credential variable is set and not
/// empty, by vendor then id.
pub fn available_models() -> Vec<&'static ModelSpec> {
    Catalog::builtin()
        .iter()
        .filter(|spec| {
            spec.provider
                .api_key_env()
                .and_then(std::env::var_os)
                .is_some_and(|value| !value.is_empty())
        })
        .collect()
}

/// The named token budgets offered to a model that takes a budget instead
/// of effort levels.
const BUDGETS: [(&str, u32); 4] = [
    ("minimal", 1024),
    ("low", 2048),
    ("medium", 8192),
    ("high", 16384),
];

/// The reasoning settings a model takes, with the names they are picked
/// by: `off` when it can stop reasoning, then each effort level it lists.
/// A model that takes a token budget and lists no levels gets the
/// [`BUDGETS`], clamped into its range. A model that does not reason takes
/// none.
pub fn effort_options(spec: &ModelSpec) -> Vec<(String, Reasoning)> {
    let support = &spec.reasoning;
    if !support.supported {
        return Vec::new();
    }
    let mut options = Vec::new();
    if support.can_disable {
        options.push(("off".to_owned(), Reasoning::Off));
    }
    options.extend(
        support
            .levels
            .iter()
            .map(|level| (level.as_str().to_owned(), Reasoning::Effort(*level))),
    );
    if support.levels.is_empty()
        && let Some(range) = &support.budget
    {
        for (name, tokens) in BUDGETS {
            let budget = Reasoning::Budget {
                tokens: tokens.clamp(*range.start(), *range.end()),
            };
            if !options.iter().any(|(_, option)| *option == budget) {
                options.push((name.to_owned(), budget));
            }
        }
    }
    options
}

/// How an effort setting is shown: `default`, `off`, a level name, or a
/// token budget.
pub fn effort_label(effort: Option<Reasoning>) -> String {
    match effort {
        None => "default".to_owned(),
        Some(Reasoning::Off) => "off".to_owned(),
        Some(Reasoning::Effort(effort)) => effort.as_str().to_owned(),
        Some(Reasoning::Budget { tokens }) => format!("{tokens} tokens"),
        Some(_) => "other".to_owned(),
    }
}

/// Connects to the catalog model `id`, with credentials from the
/// environment.
pub fn connect(id: &str) -> Result<Connection, ModelError> {
    let spec = Catalog::builtin()
        .resolve(id)
        .ok_or_else(|| ModelError::Unknown(id.to_owned()))?;
    let model = ModelSelector::from(spec)
        .provider_ref()?
        .completion_model()?;
    let label = model_id(spec);
    Ok(Connection {
        spec: spec.clone(),
        key: model_key(&label),
        handler: ErasedHandler::new(ModelAdapter::new(label, model)),
    })
}
