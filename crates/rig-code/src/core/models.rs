//! Catalog models the agent can use, and the reasoning settings each takes.

use std::collections::HashMap;

use rig_core::catalog::{Catalog, ModelSpec, ReasoningSupport};
use rig_core::completion::{GenerationOptions, Reasoning, UnsupportedOption};
use rig_core::operation::Completion;
use rig_core::providers::registry::{self, ConnectError, ProviderId};
use rig_core::serve::ErasedHandler;
use rig_core::serve::adapters::ModelAdapter;

/// Token budgets for the named levels on models that take a budget instead
/// of levels, clamped into the model's range.
const BUDGETS: [(&str, u32); 3] = [("low", 2048), ("medium", 8192), ("high", 16384)];

/// The catalog spec a `vendor/model` reference names.
pub fn resolve(reference: &str) -> Option<&'static ModelSpec> {
    Catalog::builtin()
        .resolve(reference)
        .ok()
        .map(|resolved| resolved.spec)
}

/// The `vendor/model` reference of `spec`.
pub fn reference(spec: &ModelSpec) -> String {
    format!("{}/{}", spec.provider.vendor(), spec.id)
}

/// Builds `spec`'s provider client from the environment and wraps the
/// model as an effect handler.
/// [`Effects::model_handler`](super::effects::Effects::model_handler) keeps
/// one per model.
pub(crate) fn handler(spec: &'static ModelSpec) -> Result<ErasedHandler, ConnectError> {
    let model = registry::connect(spec)?;
    Ok(ErasedHandler::new(ModelAdapter::<Completion>::new(
        reference(spec),
        model,
    )))
}

/// Catalog models that call tools and whose provider can be reached from
/// the environment: first those of providers with a key set, then those
/// of providers that need none (local servers such as Ollama), which a
/// view marks "no key needed". The check builds the provider's client
/// exactly as a request would, once per provider.
pub fn available_models() -> Vec<&'static ModelSpec> {
    let mut usable: HashMap<ProviderId, bool> = HashMap::new();
    let mut models: Vec<&'static ModelSpec> = Catalog::builtin()
        .iter()
        .filter(|spec| spec.tools)
        .filter(|spec| {
            *usable
                .entry(spec.provider)
                .or_insert_with(|| registry::connect(*spec).is_ok())
        })
        .collect();
    // Stable: catalog order within each group.
    models.sort_by_key(|spec| !spec.provider.requires_credential());
    models
}

/// A reasoning setting a model takes: what `/effort` calls it (`default`,
/// `off`, a level or a budget's name) and the setting, `None` for the
/// provider default.
#[derive(Clone, Copy, Debug)]
pub struct EffortOption(pub &'static str, pub Option<Reasoning>);

impl EffortOption {
    /// The name, with a budget's tokens, for a picker.
    pub fn label(&self) -> String {
        match self.1 {
            Some(Reasoning::Budget { tokens }) => format!("{} ({tokens} tokens)", self.0),
            _ => self.0.to_owned(),
        }
    }
}

/// The reasoning settings `spec` takes: the provider default first, then
/// `off` when reasoning can be disabled, then each effort level, or named
/// budgets on a model that takes a budget. A model whose controls the
/// catalog does not list offers only the default.
pub fn effort_options(spec: &ModelSpec) -> Vec<EffortOption> {
    let mut options = vec![EffortOption("default", None)];
    let ReasoningSupport::Listed {
        levels,
        budget,
        can_disable,
        ..
    } = &spec.reasoning
    else {
        return options;
    };
    if *can_disable {
        options.push(EffortOption("off", Some(Reasoning::Off)));
    }
    options.extend(
        levels
            .iter()
            .map(|level| EffortOption(level.as_str(), Some(Reasoning::Effort(*level)))),
    );
    if levels.is_empty()
        && let Some(range) = budget
    {
        options.extend(BUDGETS.iter().map(|(name, tokens)| {
            // Not `clamp`, which panics on an inverted range.
            let tokens = (*tokens).max(*range.start()).min(*range.end());
            EffortOption(name, Some(Reasoning::Budget { tokens }))
        }));
    }
    options
}

/// A short label for a reasoning setting.
pub fn effort_label(effort: Option<Reasoning>) -> String {
    match effort {
        None => "default".to_owned(),
        Some(Reasoning::Off) => "off".to_owned(),
        Some(Reasoning::Effort(level)) => level.as_str().to_owned(),
        Some(Reasoning::Budget { tokens }) => format!("{tokens} tokens"),
        Some(_) => "custom".to_owned(),
    }
}

/// The generation options a request carries for `effort`.
pub(crate) fn generation_options(effort: Option<Reasoning>) -> GenerationOptions {
    match effort {
        Some(reasoning) => GenerationOptions::new().reasoning(reasoning),
        None => GenerationOptions::new(),
    }
}

/// Whether `spec` takes `effort`, or why not.
pub(crate) fn check_effort(
    spec: &ModelSpec,
    effort: Option<Reasoning>,
) -> Result<(), UnsupportedOption> {
    spec.validate(&generation_options(effort))
}
