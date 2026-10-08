//! Catalog models the agent can use, and the reasoning settings each takes.

use std::collections::HashMap;

use rig_core::catalog::{Catalog, ModelSpec};
use rig_core::completion::{GenerationOptions, Reasoning, UnsupportedOption};
use rig_core::providers::registry::{ModelSelector, ProviderId};

/// Token budgets for the named levels on models that take a budget instead
/// of levels, clamped into the model's range.
const BUDGETS: [(&str, u32); 3] = [("low", 2048), ("medium", 8192), ("high", 16384)];

/// The catalog spec a `vendor/model` reference names.
pub fn resolve(reference: &str) -> Option<&'static ModelSpec> {
    Catalog::builtin().resolve(reference)
}

/// The `vendor/model` reference of `spec`.
pub fn reference(spec: &ModelSpec) -> String {
    format!("{}/{}", spec.provider.vendor(), spec.id)
}

/// Catalog models that call tools and whose provider has a credential in
/// the environment. The check builds the provider's client from the
/// environment exactly as a request would, once per provider.
pub fn available_models() -> Vec<&'static ModelSpec> {
    let mut usable: HashMap<ProviderId, bool> = HashMap::new();
    Catalog::builtin()
        .iter()
        .filter(|spec| spec.tools)
        .filter(|spec| {
            *usable.entry(spec.provider).or_insert_with(|| {
                spec.provider.requires_credential()
                    && ModelSelector::Spec(spec)
                        .provider_ref()
                        .is_ok_and(|reference| reference.completion_model().is_ok())
            })
        })
        .collect()
}

/// The reasoning settings `spec` takes, labelled for a picker: the provider
/// default first, then `off` when reasoning can be disabled, then each
/// effort level, or named budgets on a model that takes a budget.
pub fn effort_options(spec: &ModelSpec) -> Vec<(String, Option<Reasoning>)> {
    let support = &spec.reasoning;
    let mut options = vec![("default".to_owned(), None)];
    if !support.supported {
        return options;
    }
    if support.can_disable {
        options.push(("off".to_owned(), Some(Reasoning::Off)));
    }
    options.extend(
        support
            .levels
            .iter()
            .map(|level| (level.as_str().to_owned(), Some(Reasoning::Effort(*level)))),
    );
    if support.levels.is_empty()
        && let Some(range) = &support.budget
    {
        options.extend(BUDGETS.iter().map(|(name, tokens)| {
            let tokens = (*tokens).clamp(*range.start(), *range.end());
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
