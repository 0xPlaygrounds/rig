//! Models from rig-core's catalog: which ones have a credential, what effort
//! each one offers, and the generation options an effort maps to.

use std::collections::HashMap;

use bevy::prelude::*;
use rig_core::{
    catalog::{Catalog, ModelSpec},
    completion::{Effort, GenerationOptions, Reasoning},
    providers::registry::{ModelSelector, ProviderId},
};

use super::agent::EffortChoice;

/// Whether each catalog provider can be reached with the credentials in the
/// environment, checked once at startup.
#[derive(Resource, Debug, Default)]
pub struct Providers(HashMap<ProviderId, bool>);

impl Providers {
    /// Check every provider in the catalog. Building a model reads the
    /// provider's variables and does no network IO, and fails without a
    /// required key. A provider that needs no key, such as a local server,
    /// always counts.
    pub fn from_env() -> Self {
        let mut available = HashMap::new();
        for spec in Catalog::builtin().iter() {
            available
                .entry(spec.provider)
                .or_insert_with(|| credential_available(spec));
        }
        Self(available)
    }

    /// Whether `spec`'s provider has a credential.
    pub fn available(&self, spec: &ModelSpec) -> bool {
        self.0.get(&spec.provider).copied().unwrap_or(false)
    }

    /// The models an agent can use: they call tools and their provider has a
    /// credential, by vendor then id.
    pub fn models(&self) -> impl Iterator<Item = &'static ModelSpec> + '_ {
        Catalog::builtin()
            .iter()
            .filter(|spec| spec.tools && self.available(spec))
    }
}

fn credential_available(spec: &ModelSpec) -> bool {
    ModelSelector::from(spec)
        .provider_ref()
        .is_ok_and(|reference| reference.completion_model().is_ok())
}

/// The catalog reference `/model` takes for `spec`: `vendor/id`.
pub fn reference(spec: &ModelSpec) -> String {
    format!("{}/{}", spec.provider.vendor(), spec.id)
}

/// The model a [`super::agent::ModelChoice`] reference names.
pub fn resolve(reference: &str) -> Option<&'static ModelSpec> {
    Catalog::builtin().resolve(reference)
}

/// The efforts `spec` takes, in the order `/effort` offers them: `default`,
/// `off` when reasoning can be turned off, then each named level. A model
/// that takes a token budget instead offers `low`, `medium` and `high`.
pub fn effort_options(spec: &ModelSpec) -> Vec<EffortChoice> {
    let reasoning = &spec.reasoning;
    let mut options = vec![EffortChoice::Default];
    if !reasoning.supported {
        return options;
    }
    if reasoning.can_disable {
        options.push(EffortChoice::Off);
    }
    if !reasoning.levels.is_empty() {
        options.extend(reasoning.levels.iter().copied().map(EffortChoice::Level));
    } else if reasoning.budget.is_some() {
        options.extend([Effort::Low, Effort::Medium, Effort::High].map(EffortChoice::Level));
    }
    options
}

/// Room for the answer above a reasoning budget.
const ANSWER_TOKENS: u32 = 8192;

/// The generation options and the `max_tokens` an effort maps to on `spec`.
/// A budget model's level becomes a budget inside its range, and
/// `max_tokens` is raised above that budget, which the wire requires.
pub fn request_options(spec: &ModelSpec, effort: EffortChoice) -> (GenerationOptions, Option<u64>) {
    let mut options = GenerationOptions::default();
    let mut max_tokens = None;
    options.reasoning = match effort {
        EffortChoice::Default => None,
        EffortChoice::Off => Some(Reasoning::Off),
        EffortChoice::Level(level) => match &spec.reasoning.budget {
            Some(range) if spec.reasoning.levels.is_empty() => {
                let tokens = match level {
                    Effort::Medium => 8192,
                    Effort::High | Effort::XHigh | Effort::Max => 24576,
                    _ => *range.start(),
                }
                .max(*range.start())
                .min(*range.end());
                let limit = spec.max_output_tokens.unwrap_or(u32::MAX);
                let answer = tokens.saturating_add(ANSWER_TOKENS).min(limit);
                max_tokens = Some(u64::from(answer.max(tokens.saturating_add(1))));
                Some(Reasoning::Budget { tokens })
            }
            _ => Some(Reasoning::Effort(level)),
        },
    };
    (options, max_tokens)
}
