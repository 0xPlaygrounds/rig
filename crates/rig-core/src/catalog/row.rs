//! One model's row as models.dev spells it, plus the facts rig adds under
//! `rig`, and how a row becomes a [`ModelSpec`].

use serde::Deserialize;

use super::spec::{
    CacheSupport, Compat, Modalities, ModelSpec, Pricing, ReasoningSupport, Sampling,
};
use crate::completion::{CacheRetention, Effort};
use crate::providers::registry::ProviderId;

/// A models.dev model row. Every field is optional, so an override row
/// names only what it changes; fields rig does not read are ignored.
#[derive(Clone, Debug, Default, Deserialize)]
pub(super) struct Row {
    name: Option<String>,
    reasoning: Option<bool>,
    reasoning_options: Option<Vec<ReasoningOption>>,
    tool_call: Option<bool>,
    structured_output: Option<bool>,
    temperature: Option<bool>,
    modalities: Option<RowModalities>,
    #[serde(default)]
    limit: Limit,
    #[serde(default)]
    cost: Cost,
    status: Option<String>,
    #[serde(default)]
    rig: Facts,
}

/// One entry of `reasoning_options`: `effort` with its `values`,
/// `budget_tokens` with its `min` and `max`, or `toggle`.
#[derive(Clone, Debug, Deserialize)]
struct ReasoningOption {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    values: Vec<String>,
    min: Option<u32>,
    max: Option<u32>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct RowModalities {
    #[serde(default)]
    input: Vec<String>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct Limit {
    context: Option<u64>,
    output: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct Cost {
    input: Option<f64>,
    output: Option<f64>,
    cache_read: Option<f64>,
    cache_write: Option<f64>,
}

/// The facts models.dev does not carry, entered by hand under `rig`. An
/// unknown key is an error, so a misspelt fact is never silently dropped.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct Facts {
    reasoning_default: Option<Effort>,
    cache: Option<Vec<CacheRetention>>,
    sampling: Option<Sampling>,
    reasoning_field: Option<String>,
    adaptive_thinking: Option<bool>,
    thinking_off: Option<String>,
    mid_conversation_system: Option<bool>,
    rejects_forced_tool_choice: Option<bool>,
    binds_context: Option<bool>,
    prompt_cache_options: Option<bool>,
    chat_tools_need_reasoning_off: Option<bool>,
}

impl Row {
    /// `self` with every field `over` sets put on top. `limit`, `cost` and
    /// `rig` merge field by field; any other field `over` sets replaces.
    pub(super) fn overlay(self, over: Row) -> Row {
        Row {
            name: over.name.or(self.name),
            reasoning: over.reasoning.or(self.reasoning),
            reasoning_options: over.reasoning_options.or(self.reasoning_options),
            tool_call: over.tool_call.or(self.tool_call),
            structured_output: over.structured_output.or(self.structured_output),
            temperature: over.temperature.or(self.temperature),
            modalities: over.modalities.or(self.modalities),
            limit: Limit {
                context: over.limit.context.or(self.limit.context),
                output: over.limit.output.or(self.limit.output),
            },
            cost: Cost {
                input: over.cost.input.or(self.cost.input),
                output: over.cost.output.or(self.cost.output),
                cache_read: over.cost.cache_read.or(self.cost.cache_read),
                cache_write: over.cost.cache_write.or(self.cost.cache_write),
            },
            status: over.status.or(self.status),
            rig: self.rig.overlay(over.rig),
        }
    }

    /// The spec of model `id` served by `provider`.
    pub(super) fn spec(&self, provider: ProviderId, id: &str) -> ModelSpec {
        let max_output_tokens = positive(self.limit.output);
        ModelSpec {
            id: id.to_owned(),
            provider,
            display_name: self.name.clone().unwrap_or_else(|| id.to_owned()),
            context_window: positive(self.limit.context),
            max_output_tokens,
            input: self.input(),
            reasoning: self.reasoning(max_output_tokens),
            caching: CacheSupport {
                retention: self.rig.cache.clone().unwrap_or_default(),
            },
            tools: self.tool_call.unwrap_or(false),
            structured_output: self.structured_output.unwrap_or(false),
            pricing: self.pricing(),
            deprecated: self.status.as_deref() == Some("deprecated"),
            sampling: self.rig.sampling.or_else(|| {
                self.temperature
                    .filter(|takes| *takes)
                    .map(|_| Sampling::Any)
            }),
            compat: self.rig.compat(),
        }
    }

    fn input(&self) -> Modalities {
        let mut input = Modalities::default();
        for modality in self.modalities.iter().flat_map(|m| &m.input) {
            match modality.as_str() {
                "text" => input.text = true,
                "image" => input.image = true,
                "audio" => input.audio = true,
                "video" => input.video = true,
                "pdf" => input.pdf = true,
                _ => {}
            }
        }
        input
    }

    /// The reasoning `reasoning_options` describes: effort `values` but
    /// `none` (and any word that is no [`Effort`], such as Groq's
    /// `default`), a `none` value or a `toggle` entry for turning it off,
    /// and `budget_tokens`, whose absent `max` is the output limit.
    fn reasoning(&self, max_output_tokens: Option<u32>) -> ReasoningSupport {
        let supported = self.reasoning.unwrap_or(false);
        let options = self.reasoning_options.as_deref().unwrap_or_default();
        let mut levels = Vec::new();
        let mut can_disable = false;
        let mut budget = None;
        for option in options {
            match option.kind.as_str() {
                "effort" => {
                    for value in &option.values {
                        match effort(value) {
                            Some(level) if !levels.contains(&level) => levels.push(level),
                            Some(_) => {}
                            None => can_disable |= value == "none",
                        }
                    }
                }
                "toggle" => can_disable = true,
                "budget_tokens" => {
                    let min = option.min.unwrap_or(0);
                    let max = option.max.or(max_output_tokens).unwrap_or(u32::MAX);
                    budget = Some(min..=max);
                }
                _ => {}
            }
        }
        ReasoningSupport {
            supported,
            levels: if supported { levels } else { Vec::new() },
            budget: budget.filter(|_| supported),
            can_disable: supported && can_disable,
            default: self.rig.reasoning_default.filter(|_| supported),
        }
    }

    fn pricing(&self) -> Option<Pricing> {
        Some(Pricing {
            input: self.cost.input?,
            output: self.cost.output?,
            cache_read: self.cost.cache_read,
            cache_write: self.cost.cache_write,
        })
    }
}

impl Facts {
    fn overlay(self, over: Facts) -> Facts {
        Facts {
            reasoning_default: over.reasoning_default.or(self.reasoning_default),
            cache: over.cache.or(self.cache),
            sampling: over.sampling.or(self.sampling),
            reasoning_field: over.reasoning_field.or(self.reasoning_field),
            adaptive_thinking: over.adaptive_thinking.or(self.adaptive_thinking),
            thinking_off: over.thinking_off.or(self.thinking_off),
            mid_conversation_system: over
                .mid_conversation_system
                .or(self.mid_conversation_system),
            rejects_forced_tool_choice: over
                .rejects_forced_tool_choice
                .or(self.rejects_forced_tool_choice),
            binds_context: over.binds_context.or(self.binds_context),
            prompt_cache_options: over.prompt_cache_options.or(self.prompt_cache_options),
            chat_tools_need_reasoning_off: over
                .chat_tools_need_reasoning_off
                .or(self.chat_tools_need_reasoning_off),
        }
    }

    fn compat(&self) -> Compat {
        Compat {
            reasoning_field: self.reasoning_field.clone(),
            adaptive_thinking: self.adaptive_thinking.unwrap_or(false),
            thinking_off: self.thinking_off.clone(),
            mid_conversation_system: self.mid_conversation_system.unwrap_or(false),
            rejects_forced_tool_choice: self.rejects_forced_tool_choice.unwrap_or(false),
            binds_context: self.binds_context.unwrap_or(false),
            prompt_cache_options: self.prompt_cache_options.unwrap_or(false),
            chat_tools_need_reasoning_off: self.chat_tools_need_reasoning_off.unwrap_or(false),
        }
    }
}

/// The effort level `word` names, as [`Effort`]'s serde names spell it.
fn effort(word: &str) -> Option<Effort> {
    Some(match word {
        "minimal" => Effort::Minimal,
        "low" => Effort::Low,
        "medium" => Effort::Medium,
        "high" => Effort::High,
        "xhigh" => Effort::XHigh,
        "max" => Effort::Max,
        _ => return None,
    })
}

/// A limit models.dev records, unless it is `0` (its spelling of "none") or
/// does not fit a `u32`.
fn positive(limit: Option<u64>) -> Option<u32> {
    limit
        .filter(|limit| *limit > 0)
        .and_then(|limit| u32::try_from(limit).ok())
}
