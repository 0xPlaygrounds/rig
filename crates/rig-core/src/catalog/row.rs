//! One model's row as models.dev spells it, plus the facts rig adds under
//! `rig`, and how a row becomes a [`ModelSpec`].

use serde::de::{DeserializeOwned, Visitor};
use serde::{Deserialize, Serialize};

use super::spec::{
    CacheSupport, Compat, Modalities, ModelSpec, Pricing, ReasoningSupport, Sampling,
};
use crate::completion::{CacheRetention, Effort};
use crate::providers::registry::{Format, ProviderId};

/// A models.dev model row. Every field is optional, so an override row
/// names only what it changes; fields rig does not read are ignored.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub(super) struct Row {
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
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
    modalities: Option<RowModalities>,
    #[serde(default, skip_serializing_if = "is_default")]
    limit: Limit,
    #[serde(default, skip_serializing_if = "is_default")]
    cost: Cost,
    #[serde(skip_serializing_if = "Option::is_none")]
    status: Option<String>,
    #[serde(default, skip_serializing_if = "is_default")]
    rig: Facts,
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
}

#[derive(Clone, Debug, Default, PartialEq, Deserialize, Serialize)]
struct Limit {
    #[serde(skip_serializing_if = "Option::is_none")]
    context: Option<u64>,
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
    #[serde(rename = "modalities")]
    Modalities,
    #[serde(rename = "limit.context")]
    LimitContext,
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
    #[serde(rename = "status")]
    Status,
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
        let input = &spec.input;
        let modalities = [
            (input.text, "text"),
            (input.image, "image"),
            (input.audio, "audio"),
            (input.video, "video"),
            (input.pdf, "pdf"),
        ]
        .into_iter()
        .filter(|(reads, _)| *reads)
        .map(|(_, word)| word.to_owned())
        .collect();
        let compat = &spec.compat;
        Row {
            name: Some(spec.display_name.clone()),
            reasoning: Some(reasoning.supported()),
            reasoning_options,
            tool_call: Some(spec.tools),
            structured_output: Some(spec.structured_output),
            temperature: None,
            modalities: Some(RowModalities { input: modalities }),
            limit: Limit {
                context: spec.context_window.map(u64::from),
                output: spec.max_output_tokens.map(u64::from),
            },
            cost: spec.pricing.map_or_else(Cost::default, |pricing| Cost {
                input: Some(pricing.input),
                output: Some(pricing.output),
                cache_read: pricing.cache_read,
                cache_write: pricing.cache_write,
            }),
            status: Some(
                if spec.deprecated {
                    "deprecated"
                } else {
                    "active"
                }
                .to_owned(),
            ),
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
                Pinned::Reasoning => self.reasoning = None,
                Pinned::ReasoningOptions => self.reasoning_options = None,
                Pinned::ToolCall => self.tool_call = None,
                Pinned::StructuredOutput => self.structured_output = None,
                Pinned::Temperature => self.temperature = None,
                Pinned::Modalities => self.modalities = None,
                Pinned::LimitContext => self.limit.context = None,
                Pinned::LimitOutput => self.limit.output = None,
                Pinned::CostInput => self.cost.input = None,
                Pinned::CostOutput => self.cost.output = None,
                Pinned::CostCacheRead => self.cost.cache_read = None,
                Pinned::CostCacheWrite => self.cost.cache_write = None,
                Pinned::Status => self.status = None,
            }
        }
        self
    }

    /// The protocol family the row's `rig` facts name, if any.
    pub(super) fn format(&self) -> Option<Format> {
        self.rig.format
    }

    /// The spec of model `id` served by `provider`, in the family the row
    /// names when that family is registered for `provider`'s vendor.
    pub(super) fn spec(&self, provider: ProviderId, id: &str) -> ModelSpec {
        let provider = self
            .rig
            .format
            .and_then(|format| ProviderId::new(provider.vendor(), format))
            .unwrap_or(provider);
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
        }
    }
}

/// Keys of a row that rig's own data carries for the xtask and rig-core
/// does not read: `canonical_model_id` joins a gateway row to the model it
/// serves when the catalog is generated.
const UNREAD: [&str; 1] = ["canonical_model_id"];

/// The keys an object at `path` inside a row may hold, where `path` is the
/// keys leading to it (`[]` for the row, `["limit"]`, `["reasoning_options",
/// "0"]`), or `None` when that object is not one rig reads field by field.
pub(super) fn known_fields(path: &[&str]) -> Option<Vec<&'static str>> {
    Some(match path {
        [] => {
            let mut row = fields::<Row>().to_vec();
            row.extend(UNREAD);
            row
        }
        ["limit"] => fields::<Limit>().to_vec(),
        ["cost"] => fields::<Cost>().to_vec(),
        ["rig"] => fields::<Facts>().to_vec(),
        ["modalities"] => fields::<RowModalities>().to_vec(),
        ["reasoning_options", _] => fields::<ReasoningOption>().to_vec(),
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
