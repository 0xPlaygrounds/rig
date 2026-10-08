//! Models from rig's catalog: which ones have credentials, which reasoning
//! each takes, and the request an agent sends.

use std::collections::HashMap;

use bevy_ecs::prelude::*;
use rig_core::{
    catalog::{Catalog, ModelSpec},
    completion::{
        CompletionRequest, Effort as Level, GenerationOptions, Reasoning, ToolDefinition,
    },
    message::Message,
    providers::registry::ModelSelector,
};

use crate::agent::Choice;

/// Budget presets offered for models that take a token budget instead of
/// effort levels, clamped into each model's range.
const BUDGET_PRESETS: &[(&str, u32)] = &[("low", 2048), ("medium", 8192), ("high", 16384)];
/// Answer room kept above a reasoning budget in `max_tokens`.
const ANSWER_TOKENS: u32 = 8192;

/// Per vendor, whether a client can be built from the environment and
/// whether its key variable is set, checked once per vendor.
#[derive(Resource, Debug, Default)]
pub struct Credentials(HashMap<&'static str, (bool, bool)>);

impl Credentials {
    fn check(&mut self, spec: &ModelSpec) -> (bool, bool) {
        *self.0.entry(spec.provider.vendor()).or_insert_with(|| {
            let configured = ModelSelector::from(spec)
                .provider_ref()
                .ok()
                .is_some_and(|provider| provider.completion_model().is_ok());
            let key_set = spec
                .provider
                .api_key_env()
                .is_some_and(|name| std::env::var_os(name).is_some_and(|key| !key.is_empty()));
            (configured, key_set)
        })
    }

    /// Whether `/model` lists `spec`: its provider can be reached with the
    /// environment's credentials. A provider whose key is optional (a local
    /// server) is listed only when its key variable is set.
    pub fn available(&mut self, spec: &ModelSpec) -> bool {
        let (configured, key_set) = self.check(spec);
        configured && (spec.provider.requires_credential() || key_set)
    }

    /// Whether `spec` can be picked by name: its provider can be reached,
    /// with or without a key when the key is optional.
    pub fn usable(&mut self, spec: &ModelSpec) -> bool {
        self.check(spec).0
    }
}

/// The catalog reference of `spec`: `vendor/model`.
pub fn reference(spec: &ModelSpec) -> String {
    format!("{}/{}", spec.provider.vendor(), spec.id)
}

/// The catalog row for `reference`.
pub fn resolve(reference: &str) -> Option<&'static ModelSpec> {
    Catalog::builtin().resolve(reference)
}

/// `/model` options: every current, tool-calling catalog model whose
/// provider has a credential.
pub fn model_choices(credentials: &mut Credentials) -> Vec<Choice> {
    Catalog::builtin()
        .iter()
        .filter(|spec| spec.tools && !spec.deprecated && credentials.available(spec))
        .map(|spec| {
            let reference = reference(spec);
            Choice {
                label: format!("{reference}  {}", spec.display_name),
                command: format!("/model {reference}"),
            }
        })
        .collect()
}

/// `/effort` options for `spec`: `off` when reasoning can be disabled, then
/// each effort level, or budget presets for a budget-only model.
pub fn effort_choices(spec: &ModelSpec) -> Vec<Choice> {
    let support = &spec.reasoning;
    let mut choices = Vec::new();
    let default = |level: Level| {
        if support.default == Some(level) {
            " (default)"
        } else {
            ""
        }
    };
    if support.can_disable {
        choices.push(choice("off", "off".to_owned()));
    }
    for level in &support.levels {
        choices.push(choice(
            level.as_str(),
            format!("{}{}", level.as_str(), default(*level)),
        ));
    }
    if let Some(range) = support
        .budget
        .as_ref()
        .filter(|_| support.levels.is_empty())
    {
        for (name, tokens) in BUDGET_PRESETS {
            let tokens = (*tokens).clamp(*range.start(), *range.end());
            choices.push(choice(
                &tokens.to_string(),
                format!("{name} ({tokens} reasoning tokens)"),
            ));
        }
    }
    choices
}

fn choice(argument: &str, label: String) -> Choice {
    Choice {
        label,
        command: format!("/effort {argument}"),
    }
}

/// Parse an `/effort` argument: `off`, an effort level, or a token budget.
pub fn parse_effort(argument: &str) -> Option<Reasoning> {
    if argument == "off" {
        return Some(Reasoning::Off);
    }
    if let Ok(tokens) = argument.parse() {
        return Some(Reasoning::Budget { tokens });
    }
    serde_json::from_value::<Level>(serde_json::Value::String(argument.to_owned()))
        .ok()
        .map(Reasoning::Effort)
}

/// A display word for a reasoning setting.
pub fn describe(reasoning: Option<Reasoning>) -> String {
    match reasoning {
        None => "default".to_owned(),
        Some(Reasoning::Off) => "off".to_owned(),
        Some(Reasoning::Effort(level)) => level.as_str().to_owned(),
        Some(Reasoning::Budget { tokens }) => format!("{tokens} tokens"),
        Some(_) => "custom".to_owned(),
    }
}

/// Why `spec` refuses `reasoning`, or `None` when it takes it.
pub fn refusal(spec: &ModelSpec, reasoning: Reasoning) -> Option<String> {
    spec.validate(&GenerationOptions::new().reasoning(reasoning))
        .err()
        .map(|error| error.to_string())
}

/// The request for the next reply: the conversation behind `system`, the
/// tools if the model calls tools, and the reasoning `effort` or the
/// model's default. The options are validated against `spec` first.
pub fn build_request(
    spec: &ModelSpec,
    system: &str,
    conversation: &[Message],
    effort: Option<Reasoning>,
    tools: Vec<ToolDefinition>,
) -> Result<CompletionRequest, String> {
    let reasoning = effort.or(spec.reasoning.default.map(Reasoning::Effort));
    let mut options = GenerationOptions::new();
    if let Some(reasoning) = reasoning {
        options = options.reasoning(reasoning);
    }
    spec.validate(&options).map_err(|error| error.to_string())?;
    // A budget needs room for an answer under the model's output limit.
    let max_tokens = match reasoning {
        Some(Reasoning::Budget { tokens }) => {
            let ceiling = spec.max_output_tokens.unwrap_or(u32::MAX);
            if tokens >= ceiling {
                return Err(format!(
                    "a {tokens}-token reasoning budget leaves no room for an answer under \
                     {}'s {ceiling}-token output limit. Pick a smaller /effort.",
                    spec.display_name
                ));
            }
            Some(u64::from(tokens.saturating_add(ANSWER_TOKENS).min(ceiling)))
        }
        _ => None,
    };
    let (last, prior) = conversation
        .split_last()
        .ok_or_else(|| "the conversation is empty".to_owned())?;
    let mut request = CompletionRequest::new(last.clone())
        .preamble(system)
        .messages(prior.iter().cloned())
        .options(options);
    if spec.tools {
        request = request.tools(tools);
    }
    if let Some(max_tokens) = max_tokens {
        request = request.max_tokens(max_tokens);
    }
    Ok(request)
}
