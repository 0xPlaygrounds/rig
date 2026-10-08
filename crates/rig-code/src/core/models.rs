//! Catalog models the agent can use, and the reasoning settings each takes.

use std::collections::HashMap;
use std::error::Error;

use rig_core::catalog::{Catalog, ModelSpec, ReasoningSupport};
use rig_core::completion::{GenerationOptions, Reasoning, UnsupportedOption};
use rig_core::operation::Completion;
use rig_core::providers::registry::{self, ProviderId};
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
pub fn handler(spec: &'static ModelSpec) -> Result<ErasedHandler, Box<dyn Error + Send + Sync>> {
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

/// The reasoning settings `spec` takes, labelled for a picker: the provider
/// default first, then `off` when reasoning can be disabled, then each
/// effort level, or named budgets on a model that takes a budget. A model
/// whose controls the catalog does not list offers only the default.
pub fn effort_options(spec: &ModelSpec) -> Vec<(String, Option<Reasoning>)> {
    let mut options = vec![("default".to_owned(), None)];
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
        options.push(("off".to_owned(), Some(Reasoning::Off)));
    }
    options.extend(
        levels
            .iter()
            .map(|level| (level.as_str().to_owned(), Some(Reasoning::Effort(*level)))),
    );
    if levels.is_empty()
        && let Some(range) = budget
    {
        options.extend(BUDGETS.iter().map(|(name, tokens)| {
            // Not `clamp`, which panics on an inverted range.
            let tokens = (*tokens).max(*range.start()).min(*range.end());
            (
                format!("{name} ({tokens} tokens)"),
                Some(Reasoning::Budget { tokens }),
            )
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
pub fn generation_options(effort: Option<Reasoning>) -> GenerationOptions {
    match effort {
        Some(reasoning) => GenerationOptions::new().reasoning(reasoning),
        None => GenerationOptions::new(),
    }
}

/// Whether `spec` takes `effort`, or why not.
pub fn check_effort(spec: &ModelSpec, effort: Option<Reasoning>) -> Result<(), UnsupportedOption> {
    spec.validate(&generation_options(effort))
}
