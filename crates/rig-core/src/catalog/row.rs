//! One model's row as models.dev spells it, plus the facts rig adds under
//! `rig`, and how a row becomes a [`ModelSpec`].

use std::collections::BTreeMap;

use serde::de::{DeserializeOwned, Visitor};
use serde::{Deserialize, Serialize};

use super::details::{
    CostDetails, CostTier, Interleaved, Mode, ModelDetails, ProviderOverride, Rates, Tier,
};
use super::spec::{
    CacheSupport, Compat, Modalities, ModelSpec, Pricing, ReasoningSupport, Sampling,
};
use crate::completion::{CacheRetention, Effort};
use crate::providers::registry::{Format, ProviderId};

/// A models.dev model row, every field models.dev publishes but its `id`,
/// which is the row's key. Every field is optional, so an override row
/// names only what it changes.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub(super) struct Row {
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    family: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    kind: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    canonical_model_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_options: Option<Vec<ReasoningOption>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    structured_output: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    attachment: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    interleaved: Option<Interleaved>,
    #[serde(skip_serializing_if = "Option::is_none")]
    modalities: Option<RowModalities>,
    #[serde(default, skip_serializing_if = "is_default")]
    limit: Limit,
    #[serde(default, skip_serializing_if = "is_default")]
    cost: Cost,
    #[serde(skip_serializing_if = "Option::is_none")]
    knowledge: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    release_date: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_updated: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    open_weights: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    provider: Option<ProviderOverride>,
    #[serde(skip_serializing_if = "Option::is_none")]
    experimental: Option<Experimental>,
    #[serde(default, skip_serializing_if = "is_default")]
    rig: Facts,
}

/// A row's `experimental` features: the service modes it is offered in.
#[derive(Clone, Debug, Default, PartialEq, Deserialize, Serialize)]
struct Experimental {
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    modes: BTreeMap<String, Mode>,
}

/// One entry of `reasoning_options`: `effort` with its `values`,
/// `budget_tokens` with its `min` and `max`, or `toggle`.
#[derive(Clone, Debug, Deserialize, Serialize)]
struct ReasoningOption {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    values: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    min: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max: Option<u32>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
struct RowModalities {
    #[serde(default)]
    input: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    output: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Deserialize, Serialize)]
struct Limit {
    #[serde(skip_serializing_if = "Option::is_none")]
    context: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output: Option<u64>,
}

#[derive(Clone, Debug, Default, PartialEq, Deserialize, Serialize)]
struct Cost {
    #[serde(skip_serializing_if = "Option::is_none")]
    input: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_read: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_write: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_audio: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_audio: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tiers: Option<Vec<CostTier>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    context_over_200k: Option<Rates>,
}

/// The facts models.dev does not carry, entered by hand under `rig`.
/// [`Catalog::from_overrides`](super::Catalog::from_overrides) rejects a key
/// it does not name.
#[derive(Clone, Debug, Default, PartialEq, Deserialize, Serialize)]
struct Facts {
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<Format>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_default: Option<Effort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_control: Option<ReasoningControl>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache: Option<Vec<CacheRetention>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sampling: Option<Sampling>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_field: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    adaptive_thinking: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_off: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mid_conversation_system: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rejects_forced_tool_choice: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    binds_context: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_options: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    chat_tools_need_reasoning_off: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pinned: Option<Vec<Pinned>>,
}

/// A row key that rig's review or the generator's joins set, written by
/// `cargo xtask catalog sync` under `rig.pinned`.
/// [`Catalog::with_models_dev`](super::Catalog::with_models_dev) leaves a
/// pinned key as it is, so a refresh from models.dev keeps it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub(super) enum Pinned {
    #[serde(rename = "name")]
    Name,
    #[serde(rename = "description")]
    Description,
    #[serde(rename = "family")]
    Family,
    #[serde(rename = "type")]
    Type,
    #[serde(rename = "reasoning")]
    Reasoning,
    #[serde(rename = "reasoning_options")]
    ReasoningOptions,
    #[serde(rename = "tool_call")]
    ToolCall,
    #[serde(rename = "structured_output")]
    StructuredOutput,
    #[serde(rename = "temperature")]
    Temperature,
    #[serde(rename = "attachment")]
    Attachment,
    #[serde(rename = "interleaved")]
    Interleaved,
    #[serde(rename = "modalities")]
    Modalities,
    #[serde(rename = "limit.context")]
    LimitContext,
    #[serde(rename = "limit.input")]
    LimitInput,
    #[serde(rename = "limit.output")]
    LimitOutput,
    #[serde(rename = "cost.input")]
    CostInput,
    #[serde(rename = "cost.output")]
    CostOutput,
    #[serde(rename = "cost.cache_read")]
    CostCacheRead,
    #[serde(rename = "cost.cache_write")]
    CostCacheWrite,
    #[serde(rename = "cost.reasoning")]
    CostReasoning,
    #[serde(rename = "cost.input_audio")]
    CostInputAudio,
    #[serde(rename = "cost.output_audio")]
    CostOutputAudio,
    #[serde(rename = "cost.tiers")]
    CostTiers,
    #[serde(rename = "cost.context_over_200k")]
    CostContextOver200k,
    #[serde(rename = "knowledge")]
    Knowledge,
    #[serde(rename = "release_date")]
    ReleaseDate,
    #[serde(rename = "last_updated")]
    LastUpdated,
    #[serde(rename = "open_weights")]
    OpenWeights,
    #[serde(rename = "status")]
    Status,
    #[serde(rename = "provider")]
    Provider,
    #[serde(rename = "experimental")]
    Experimental,
}

/// What a reasoning row that lists no `reasoning_options` means, entered by
/// hand under `rig`. Without it such a row is unknown.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum ReasoningControl {
    /// The catalog does not know which controls the model takes.
    Unknown,
    /// The model's API rejects every reasoning control.
    None,
}

impl Row {
    /// The row that builds `spec` again: every fact the spec knows, set
    /// explicitly, so laid over another row it replaces that row's value.
    pub(super) fn from_spec(spec: &ModelSpec) -> Row {
        let reasoning = &spec.reasoning;
        let reasoning_options = reasoning.supported().then(|| {
            let mut options = Vec::new();
            let levels = reasoning.levels().unwrap_or_default();
            if !levels.is_empty() {
                options.push(ReasoningOption {
                    kind: "effort".to_owned(),
                    values: levels
                        .iter()
                        .map(|level| level.as_str().to_owned())
                        .collect(),
                    min: None,
                    max: None,
                });
            }
            if reasoning.can_disable() == Some(true) {
                options.push(ReasoningOption {
                    kind: "toggle".to_owned(),
                    values: Vec::new(),
                    min: None,
                    max: None,
                });
            }
            if let Some(budget) = reasoning.budget() {
                options.push(ReasoningOption {
                    kind: "budget_tokens".to_owned(),
                    values: Vec::new(),
                    min: Some(*budget.start()),
                    max: Some(*budget.end()),
                });
            }
            options
        });
        // An empty option list reads back as unknown unless the row says
        // the model takes no control.
        let reasoning_control = match reasoning {
            ReasoningSupport::Unknown { .. } => Some(ReasoningControl::Unknown),
            _ if reasoning_options.as_ref().is_some_and(Vec::is_empty) => {
                Some(ReasoningControl::None)
            }
            _ => None,
        };
        let modalities = words(spec.input);
        let compat = &spec.compat;
        let details = &spec.details;
        let cost = &details.cost;
        Row {
            name: Some(spec.display_name.clone()),
            description: details.description.clone(),
            family: details.family.clone(),
            kind: details.kind.clone(),
            canonical_model_id: details.canonical_model_id.clone(),
            reasoning: Some(reasoning.supported()),
            reasoning_options,
            tool_call: Some(spec.tools),
            structured_output: Some(spec.structured_output),
            temperature: None,
            attachment: details.attachment,
            interleaved: details.interleaved.clone(),
            modalities: Some(RowModalities {
                input: modalities,
                output: details.output.map(words).unwrap_or_default(),
            }),
            limit: Limit {
                context: spec.context_window.map(u64::from),
                input: spec.max_input_tokens.map(u64::from),
                output: spec.max_output_tokens.map(u64::from),
            },
            cost: Cost {
                input: spec.pricing.map(|pricing| pricing.input),
                output: spec.pricing.map(|pricing| pricing.output),
                cache_read: spec.pricing.and_then(|pricing| pricing.cache_read),
                cache_write: spec.pricing.and_then(|pricing| pricing.cache_write),
                reasoning: cost.reasoning,
                input_audio: cost.input_audio,
                output_audio: cost.output_audio,
                tiers: (!cost.tiers.is_empty()).then(|| cost.tiers.clone()),
                context_over_200k: cost.context_over_200k,
            },
            knowledge: details.knowledge.clone(),
            release_date: details.release_date.clone(),
            last_updated: details.last_updated.clone(),
            open_weights: details.open_weights,
            // A listed lifecycle other than deprecation stays unless the spec
            // was deprecated.
            status: Some(match (spec.deprecated, details.status.as_deref()) {
                (true, _) => "deprecated".to_owned(),
                (false, Some(status)) if status != "deprecated" => status.to_owned(),
                (false, _) => "active".to_owned(),
            }),
            provider: details.provider.clone(),
            experimental: (!details.modes.is_empty()).then(|| Experimental {
                modes: details.modes.clone(),
            }),
            rig: Facts {
                format: spec.provider.format(),
                reasoning_default: reasoning.default_effort(),
                reasoning_control,
                cache: (!spec.caching.retention.is_empty()).then(|| spec.caching.retention.clone()),
                sampling: spec.sampling,
                reasoning_field: compat.reasoning_field.clone(),
                adaptive_thinking: Some(compat.adaptive_thinking),
                thinking_off: compat.thinking_off.clone(),
                mid_conversation_system: Some(compat.mid_conversation_system),
                rejects_forced_tool_choice: Some(compat.rejects_forced_tool_choice),
                binds_context: Some(compat.binds_context),
                prompt_cache_options: Some(compat.prompt_cache_options),
                chat_tools_need_reasoning_off: Some(compat.chat_tools_need_reasoning_off),
                prefill: Some(compat.prefill),
                pinned: None,
            },
        }
    }

    /// The row as JSON, as an override file spells it.
    pub(super) fn to_json(&self) -> serde_json::Value {
        // A row holds only strings, numbers, booleans and lists, and no map
        // keyed by anything but a string, so it always converts.
        serde_json::to_value(self).unwrap_or_default()
    }

    /// `self` with every field `over` sets put on top. `limit`, `cost` and
    /// `rig` merge field by field; any other field `over` sets replaces.
    pub(super) fn overlay(self, over: Row) -> Row {
        Row {
            name: over.name.or(self.name),
            description: over.description.or(self.description),
            family: over.family.or(self.family),
            kind: over.kind.or(self.kind),
            canonical_model_id: over.canonical_model_id.or(self.canonical_model_id),
            reasoning: over.reasoning.or(self.reasoning),
            reasoning_options: over.reasoning_options.or(self.reasoning_options),
            tool_call: over.tool_call.or(self.tool_call),
            structured_output: over.structured_output.or(self.structured_output),
            temperature: over.temperature.or(self.temperature),
            attachment: over.attachment.or(self.attachment),
            interleaved: over.interleaved.or(self.interleaved),
            modalities: both(self.modalities, over.modalities, RowModalities::overlay),
            limit: Limit {
                context: over.limit.context.or(self.limit.context),
                input: over.limit.input.or(self.limit.input),
                output: over.limit.output.or(self.limit.output),
            },
            cost: Cost {
                input: over.cost.input.or(self.cost.input),
                output: over.cost.output.or(self.cost.output),
                cache_read: over.cost.cache_read.or(self.cost.cache_read),
                cache_write: over.cost.cache_write.or(self.cost.cache_write),
                reasoning: over.cost.reasoning.or(self.cost.reasoning),
                input_audio: over.cost.input_audio.or(self.cost.input_audio),
                output_audio: over.cost.output_audio.or(self.cost.output_audio),
                tiers: over.cost.tiers.or(self.cost.tiers),
                context_over_200k: both(
                    self.cost.context_over_200k,
                    over.cost.context_over_200k,
                    rates,
                ),
            },
            knowledge: over.knowledge.or(self.knowledge),
            release_date: over.release_date.or(self.release_date),
            last_updated: over.last_updated.or(self.last_updated),
            open_weights: over.open_weights.or(self.open_weights),
            status: over.status.or(self.status),
            provider: both(self.provider, over.provider, provider),
            experimental: both(self.experimental, over.experimental, Experimental::overlay),
            rig: self.rig.overlay(over.rig),
        }
    }

    /// The keys the row's `rig` facts pin against a refresh.
    pub(super) fn pinned(&self) -> &[Pinned] {
        self.rig.pinned.as_deref().unwrap_or_default()
    }

    /// `self` without the keys in `pinned`, so laid over a row it leaves
    /// that row's values of them.
    pub(super) fn without(mut self, pinned: &[Pinned]) -> Row {
        for key in pinned {
            match key {
                Pinned::Name => self.name = None,
                Pinned::Description => self.description = None,
                Pinned::Family => self.family = None,
                Pinned::Type => self.kind = None,
                Pinned::Reasoning => self.reasoning = None,
                Pinned::ReasoningOptions => self.reasoning_options = None,
                Pinned::ToolCall => self.tool_call = None,
                Pinned::StructuredOutput => self.structured_output = None,
                Pinned::Temperature => self.temperature = None,
                Pinned::Attachment => self.attachment = None,
                Pinned::Interleaved => self.interleaved = None,
                Pinned::Modalities => self.modalities = None,
                Pinned::LimitContext => self.limit.context = None,
                Pinned::LimitInput => self.limit.input = None,
                Pinned::LimitOutput => self.limit.output = None,
                Pinned::CostInput => self.cost.input = None,
                Pinned::CostOutput => self.cost.output = None,
                Pinned::CostCacheRead => self.cost.cache_read = None,
                Pinned::CostCacheWrite => self.cost.cache_write = None,
                Pinned::CostReasoning => self.cost.reasoning = None,
                Pinned::CostInputAudio => self.cost.input_audio = None,
                Pinned::CostOutputAudio => self.cost.output_audio = None,
                Pinned::CostTiers => self.cost.tiers = None,
                Pinned::CostContextOver200k => self.cost.context_over_200k = None,
                Pinned::Knowledge => self.knowledge = None,
                Pinned::ReleaseDate => self.release_date = None,
                Pinned::LastUpdated => self.last_updated = None,
                Pinned::OpenWeights => self.open_weights = None,
                Pinned::Status => self.status = None,
                Pinned::Provider => self.provider = None,
                Pinned::Experimental => self.experimental = None,
            }
        }
        self
    }

    /// The protocol family the row's `rig` facts name, if any.
    pub(super) fn format(&self) -> Option<Format> {
        self.rig.format
    }

    /// The protocol family of the AI SDK package models.dev reaches the
    /// model with, where it names one apart from OpenAI's.
    pub(super) fn sdk_format(&self) -> Option<Format> {
        match self.provider.as_ref()?.npm.as_deref()? {
            "@ai-sdk/anthropic" => Some(Format::Anthropic),
            "@ai-sdk/google" => Some(Format::Gemini),
            _ => None,
        }
    }

    /// The spec of model `id` served by `provider`'s vendor, in the family
    /// the row names (its `rig` facts', else its AI SDK package's) when that
    /// family is registered for the vendor, else in the vendor's primary.
    pub(super) fn spec(&self, provider: ProviderId, id: &str) -> ModelSpec {
        let vendor = provider.vendor();
        let provider = self
            .rig
            .format
            .or(self.sdk_format())
            .and_then(|format| ProviderId::new(vendor, format))
            .or_else(|| ProviderId::catalog(vendor).filter(ProviderId::is_registered))
            .unwrap_or(provider);
        let max_output_tokens = positive(self.limit.output);
        ModelSpec {
            id: id.to_owned(),
            provider,
            display_name: self.name.clone().unwrap_or_else(|| id.to_owned()),
            context_window: positive(self.limit.context),
            max_input_tokens: positive(self.limit.input),
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
            details: self.details(),
        }
    }

    fn input(&self) -> Modalities {
        modalities(self.modalities.iter().flat_map(|m| &m.input))
    }

    fn details(&self) -> ModelDetails {
        let cost = &self.cost;
        ModelDetails {
            description: self.description.clone(),
            family: self.family.clone(),
            kind: self.kind.clone(),
            canonical_model_id: self.canonical_model_id.clone(),
            attachment: self.attachment,
            interleaved: self.interleaved.clone(),
            output: self
                .modalities
                .as_ref()
                .filter(|listed| !listed.output.is_empty())
                .map(|listed| modalities(&listed.output)),
            knowledge: self.knowledge.clone(),
            release_date: self.release_date.clone(),
            last_updated: self.last_updated.clone(),
            open_weights: self.open_weights,
            // `active` is how a row from a spec says it is not deprecated.
            status: self.status.clone().filter(|status| status != "active"),
            cost: CostDetails {
                reasoning: cost.reasoning,
                input_audio: cost.input_audio,
                output_audio: cost.output_audio,
                tiers: cost.tiers.clone().unwrap_or_default(),
                context_over_200k: cost.context_over_200k,
            },
            provider: self.provider.clone(),
            modes: (self.experimental.as_ref())
                .map(|experimental| experimental.modes.clone())
                .unwrap_or_default(),
        }
    }

    /// The reasoning `reasoning_options` describes: effort `values` but
    /// `none` (and any word that is no [`Effort`], such as Groq's
    /// `default`), a `none` value or a `toggle` entry for turning it off,
    /// and `budget_tokens`, whose absent `max` is the output limit. A
    /// reasoning row whose options list no control rig reads is unknown,
    /// unless its `rig` facts say `"reasoning_control": "none"`.
    fn reasoning(&self, max_output_tokens: Option<u32>) -> ReasoningSupport {
        if !self.reasoning.unwrap_or(false) {
            return ReasoningSupport::None;
        }
        let default = self.rig.reasoning_default;
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
        let lists_nothing = levels.is_empty() && budget.is_none() && !can_disable;
        if lists_nothing && self.rig.reasoning_control != Some(ReasoningControl::None) {
            return ReasoningSupport::Unknown { default };
        }
        ReasoningSupport::Listed {
            levels,
            budget,
            can_disable,
            default,
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
            format: over.format.or(self.format),
            reasoning_default: over.reasoning_default.or(self.reasoning_default),
            reasoning_control: over.reasoning_control.or(self.reasoning_control),
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
            prefill: over.prefill.or(self.prefill),
            pinned: over.pinned.or(self.pinned),
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
            prefill: self.prefill.unwrap_or(false),
        }
    }
}

/// The keys an object at `path` inside a row may hold, where `path` is the
/// keys leading to it (`[]` for the row, `["limit"]`, `["reasoning_options",
/// "0"]`), or `None` when that object's keys are free, such as a provider
/// body or the names of its modes.
pub(super) fn known_fields(path: &[&str]) -> Option<Vec<&'static str>> {
    Some(match path {
        [] => fields::<Row>().to_vec(),
        ["limit"] => fields::<Limit>().to_vec(),
        ["cost"] => fields::<Cost>().to_vec(),
        ["rig"] => fields::<Facts>().to_vec(),
        ["modalities"] => fields::<RowModalities>().to_vec(),
        ["reasoning_options", _] => fields::<ReasoningOption>().to_vec(),
        // `interleaved` is `true`, `false`, or this object.
        ["interleaved"] => vec!["field"],
        ["cost", "context_over_200k"] | ["experimental", "modes", _, "cost"] => {
            fields::<Rates>().to_vec()
        }
        // A tier's rates are flattened beside it, which serde does not list.
        ["cost", "tiers", _] => [fields::<Rates>(), &["tier"]].concat(),
        ["cost", "tiers", _, "tier"] => fields::<Tier>().to_vec(),
        ["provider"] | ["experimental", "modes", _, "provider"] => {
            fields::<ProviderOverride>().to_vec()
        }
        ["experimental"] => fields::<Experimental>().to_vec(),
        ["experimental", "modes", _] => fields::<Mode>().to_vec(),
        _ => return None,
    })
}

/// The fields `T`'s derived `Deserialize` reads, as the input spells them.
fn fields<T: DeserializeOwned>() -> &'static [&'static str] {
    let mut fields: &'static [&'static str] = &[];
    // The deserializer only records the field list and then fails, so the
    // error is expected.
    let _ = T::deserialize(FieldNames(&mut fields));
    fields
}

/// A deserializer that records the field names a struct asks for.
struct FieldNames<'a>(&'a mut &'static [&'static str]);

impl<'de> serde::Deserializer<'de> for FieldNames<'_> {
    type Error = serde::de::value::Error;

    fn deserialize_any<V: Visitor<'de>>(self, _: V) -> Result<V::Value, Self::Error> {
        Err(serde::de::Error::custom(
            "only a struct's field names are read",
        ))
    }

    fn deserialize_struct<V: Visitor<'de>>(
        self,
        _: &'static str,
        fields: &'static [&'static str],
        _: V,
    ) -> Result<V::Value, Self::Error> {
        *self.0 = fields;
        Err(serde::de::Error::custom(
            "only a struct's field names are read",
        ))
    }

    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string
        bytes byte_buf option unit unit_struct newtype_struct seq tuple
        tuple_struct map enum identifier ignored_any
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

/// The modalities `words` names, as models.dev spells them.
fn modalities<'a>(words: impl IntoIterator<Item = &'a String>) -> Modalities {
    let mut read = Modalities::default();
    for word in words {
        match word.as_str() {
            "text" => read.text = true,
            "image" => read.image = true,
            "audio" => read.audio = true,
            "video" => read.video = true,
            "pdf" => read.pdf = true,
            _ => {}
        }
    }
    read
}

/// `modalities` as models.dev spells them.
fn words(modalities: Modalities) -> Vec<String> {
    [
        (modalities.text, "text"),
        (modalities.image, "image"),
        (modalities.audio, "audio"),
        (modalities.video, "video"),
        (modalities.pdf, "pdf"),
    ]
    .into_iter()
    .filter(|(listed, _)| *listed)
    .map(|(_, word)| word.to_owned())
    .collect()
}

fn is_default<T: Default + PartialEq>(value: &T) -> bool {
    *value == T::default()
}

/// A limit models.dev records, unless it is `0` (its spelling of "none") or
/// does not fit a `u32`.
fn positive(limit: Option<u64>) -> Option<u32> {
    limit
        .filter(|limit| *limit > 0)
        .and_then(|limit| u32::try_from(limit).ok())
}

/// `over` laid over `base` by `merge` when both are given, else whichever
/// is.
fn both<T>(base: Option<T>, over: Option<T>, merge: impl FnOnce(T, T) -> T) -> Option<T> {
    match (base, over) {
        (Some(base), Some(over)) => Some(merge(base, over)),
        (base, over) => over.or(base),
    }
}

impl RowModalities {
    /// `over` laid over these, input and output apart: an empty list is
    /// one the override does not name.
    fn overlay(self, over: RowModalities) -> RowModalities {
        let pick = |base: Vec<String>, over: Vec<String>| match over.is_empty() {
            true => base,
            false => over,
        };
        RowModalities {
            input: pick(self.input, over.input),
            output: pick(self.output, over.output),
        }
    }
}

impl Experimental {
    /// `over` laid over this, mode by mode.
    fn overlay(mut self, over: Experimental) -> Experimental {
        for (name, mode) in over.modes {
            let merged = match self.modes.remove(&name) {
                Some(base) => Mode {
                    cost: both(base.cost, mode.cost, rates),
                    provider: both(base.provider, mode.provider, provider),
                },
                None => mode,
            };
            self.modes.insert(name, merged);
        }
        self
    }
}

/// `over` laid over `base`, price by price.
fn rates(base: Rates, over: Rates) -> Rates {
    Rates {
        input: over.input.or(base.input),
        output: over.output.or(base.output),
        cache_read: over.cache_read.or(base.cache_read),
        cache_write: over.cache_write.or(base.cache_write),
        reasoning: over.reasoning.or(base.reasoning),
        input_audio: over.input_audio.or(base.input_audio),
        output_audio: over.output_audio.or(base.output_audio),
    }
}

/// `over` laid over `base` field by field, its headers and body key by key.
fn provider(base: ProviderOverride, over: ProviderOverride) -> ProviderOverride {
    let mut headers = base.headers;
    headers.extend(over.headers);
    ProviderOverride {
        npm: over.npm.or(base.npm),
        api: over.api.or(base.api),
        shape: over.shape.or(base.shape),
        body: both(base.body, over.body, json),
        headers,
    }
}

/// `over` laid over `base`: objects key by key, anything else replaced.
fn json(base: serde_json::Value, over: serde_json::Value) -> serde_json::Value {
    match (base, over) {
        (serde_json::Value::Object(mut base), serde_json::Value::Object(over)) => {
            for (key, value) in over {
                match base.get_mut(&key) {
                    Some(slot) => *slot = json(slot.take(), value),
                    None => {
                        base.insert(key, value);
                    }
                }
            }
            serde_json::Value::Object(base)
        }
        (_, over) => over,
    }
}
