//! What models.dev records about a model beyond the facts requests are
//! checked against, under its own field names.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::spec::Modalities;

/// What models.dev records about a model that rig does not act on: its
/// description, lifecycle dates, extra prices, request overrides and
/// service modes. Each field is `None` or empty when the row does not say.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelDetails {
    /// What the model is for, in a sentence.
    pub description: Option<String>,
    /// The family it belongs to, such as `gemini`.
    pub family: Option<String>,
    /// A specialized kind, models.dev's `type`, such as `image` or
    /// `embedding`; `None` for a chat model.
    #[serde(rename = "type")]
    pub kind: Option<String>,
    /// The `vendor/model` this row serves, when it is another vendor's.
    pub canonical_model_id: Option<String>,
    /// Whether it takes file attachments.
    pub attachment: Option<bool>,
    /// Whether, and under which field, it takes reasoning interleaved with
    /// its messages.
    pub interleaved: Option<Interleaved>,
    /// What it writes, models.dev's `modalities.output`.
    pub output: Option<Modalities>,
    /// Its knowledge cutoff, `YYYY-MM` or `YYYY-MM-DD`.
    pub knowledge: Option<String>,
    /// When it was first released, `YYYY-MM` or `YYYY-MM-DD`.
    pub release_date: Option<String>,
    /// When it last changed, `YYYY-MM` or `YYYY-MM-DD`.
    pub last_updated: Option<String>,
    /// Whether its weights are public.
    pub open_weights: Option<bool>,
    /// Its lifecycle status as listed: `alpha`, `beta` or `deprecated`.
    pub status: Option<String>,
    /// Prices beyond [`Pricing`](super::Pricing), in USD per million tokens.
    pub cost: CostDetails,
    /// How a request reaches it, when that differs from its vendor's.
    pub provider: Option<ProviderOverride>,
    /// The service modes it is offered in, such as `fast` or `priority`, by
    /// name: models.dev's `experimental.modes`.
    pub modes: BTreeMap<String, Mode>,
}

/// Interleaved reasoning support: `true` in general, or the message field
/// it is carried in, such as `reasoning_content`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Interleaved {
    /// Supported, or not, with no field named.
    Supported(bool),
    /// Supported under this message field.
    Field {
        /// The field's name.
        field: String,
    },
}

/// Prices beyond the base [`Pricing`](super::Pricing).
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CostDetails {
    /// Reasoning tokens, when billed apart from output.
    pub reasoning: Option<f64>,
    /// Audio input tokens, when billed apart.
    pub input_audio: Option<f64>,
    /// Audio output tokens, when billed apart.
    pub output_audio: Option<f64>,
    /// Prices that apply past a size of request.
    pub tiers: Vec<CostTier>,
    /// Prices for a request of more than 200k tokens.
    pub context_over_200k: Option<Rates>,
}

/// One set of prices in USD per million tokens; each `None` when not
/// listed.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Rates {
    /// Uncached input.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<f64>,
    /// Output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<f64>,
    /// Input read from the cache.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read: Option<f64>,
    /// Input written to the cache.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_write: Option<f64>,
    /// Reasoning tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<f64>,
    /// Audio input tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_audio: Option<f64>,
    /// Audio output tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_audio: Option<f64>,
}

/// Prices that apply to a request past the size its [`Tier`] names.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CostTier {
    /// The prices.
    #[serde(flatten)]
    pub rates: Rates,
    /// Where they start.
    pub tier: Tier,
}

/// Where a [`CostTier`] starts: past `size` of what `kind` measures.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tier {
    /// What is measured, such as `context`.
    #[serde(rename = "type")]
    pub kind: String,
    /// The size, in tokens.
    pub size: u64,
}

/// How a request reaches a model, or runs in a mode, where that differs
/// from its vendor's: models.dev's per-model `provider`.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ProviderOverride {
    /// The AI SDK package that reaches it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub npm: Option<String>,
    /// The API base URL, which may name environment variables as
    /// `${NAME}`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api: Option<String>,
    /// The API shape, such as `responses`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shape: Option<String>,
    /// Fields added to the request body.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<serde_json::Value>,
    /// Headers added to the request.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub headers: BTreeMap<String, String>,
}

/// A service mode a model is offered in.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Mode {
    /// Its prices, when they differ.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost: Option<Rates>,
    /// What a request in it changes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderOverride>,
}
