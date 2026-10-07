//! What the catalog knows about one model, and the checks it makes against
//! [`GenerationOptions`].

use std::ops::RangeInclusive;

use serde::{Deserialize, Serialize};

use crate::completion::{
    CacheRetention, Cost, Effort, GenerationOptions, Reasoning, UnsupportedOption, Usage,
};
use crate::providers::registry::ProviderId;

/// One model's facts: limits, input modalities, the reasoning and caching it
/// takes, and its prices. Built by [`Catalog`](super::Catalog) from its data.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct ModelSpec {
    /// The provider's own model id.
    pub id: String,
    /// The provider serving the model under [`Self::id`].
    pub provider: ProviderId,
    /// A human-readable name.
    pub display_name: String,
    /// The most tokens the model reads and writes in one request, if known.
    pub context_window: Option<u32>,
    /// The most tokens the model writes in one reply, if known.
    pub max_output_tokens: Option<u32>,
    /// What the model reads.
    pub input: Modalities,
    /// The reasoning the model takes.
    pub reasoning: ReasoningSupport,
    /// The cache retentions the model honours.
    pub caching: CacheSupport,
    /// Whether the model calls tools.
    pub tools: bool,
    /// Whether the model constrains its output to a JSON schema.
    pub structured_output: bool,
    /// Prices in USD per million tokens, if known.
    pub pricing: Option<Pricing>,
    /// Whether the provider has deprecated the model.
    pub deprecated: bool,
    /// When the model takes sampling parameters (`temperature`, `top_p`,
    /// `top_logprobs`, `logprobs`), or `None` when unknown.
    pub sampling: Option<Sampling>,
    /// Facts the encoders read that no portable field holds.
    pub compat: Compat,
}

/// What a model reads.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Modalities {
    /// Text.
    pub text: bool,
    /// Images.
    pub image: bool,
    /// Audio.
    pub audio: bool,
    /// Video.
    pub video: bool,
    /// PDF documents.
    pub pdf: bool,
}

/// The reasoning a model takes. A row that marks a model as reasoning but
/// lists no reasoning options (193 rows of the built-in catalog, most of
/// them gateway rows on HuggingFace, OpenRouter, Venice and Bedrock) has no
/// levels, no budget and cannot disable, so [`ModelSpec::validate`] refuses
/// every effort, every budget and `Off` on it; unlike [`CacheSupport`],
/// empty here does not mean unknown.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReasoningSupport {
    /// Whether the model reasons at all.
    pub supported: bool,
    /// The effort levels it takes, empty when it takes none.
    pub levels: Vec<Effort>,
    /// The reasoning-token budgets it takes, if it takes one.
    pub budget: Option<RangeInclusive<u32>>,
    /// Whether a request can turn its reasoning off.
    pub can_disable: bool,
    /// The effort it uses when a request names none, if documented.
    pub default: Option<Effort>,
}

/// The cache retentions a model honours.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CacheSupport {
    /// The [`CacheRetention`] values the model honours. Empty when the
    /// catalog does not know, in which case nothing is refused.
    pub retention: Vec<CacheRetention>,
}

/// Prices in USD per million tokens.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pricing {
    /// Uncached input.
    pub input: f64,
    /// Output, reasoning included.
    pub output: f64,
    /// Input read from the cache, or `None` when unknown.
    pub cache_read: Option<f64>,
    /// Input written to the cache, or `None` when unknown.
    pub cache_write: Option<f64>,
}

impl Pricing {
    /// What `usage` costs at these prices, or `None` unless it reports both
    /// its input and output tokens. Uncached input is the input tokens less
    /// those read from and written to the cache; cache reads and writes
    /// with no price of their own are charged at [`Self::input`]. Every
    /// cache write is charged at one price, so a provider that bills longer
    /// retention higher costs more than this says. It is the standard-tier
    /// list price of the tokens: the service tier, long-context price tiers
    /// and hosted-tool fees (web search, code execution) are not in it.
    pub fn cost(&self, usage: &Usage) -> Option<Cost> {
        let input = usage.input_tokens?;
        let output = usage.output_tokens?;
        let read = usage.cached_input_tokens.unwrap_or(0);
        let written = usage.cache_creation_input_tokens.unwrap_or(0);
        let uncached = input.saturating_sub(read).saturating_sub(written);
        // Token counts stay far below 2^53, so the conversion is exact.
        let price = |tokens: u64, per_million: f64| tokens as f64 * per_million / 1_000_000.0;
        Some(Cost::from_parts(
            price(uncached, self.input),
            price(output, self.output),
            price(read, self.cache_read.unwrap_or(self.input)),
            price(written, self.cache_write.unwrap_or(self.input)),
        ))
    }
}

/// When a model takes sampling parameters.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Sampling {
    /// Always.
    Any,
    /// Only while its reasoning is off.
    ReasoningOff,
    /// Never.
    Never,
}

/// Model facts the encoders read that no portable field holds. Each defaults
/// to `false` or `None`, which is what a model the field does not concern
/// has.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Compat {
    /// The field an OpenAI Chat Completions assistant message must carry
    /// its reasoning under, such as `reasoning_content`.
    pub reasoning_field: Option<String>,
    /// Anthropic: an effort level goes with `thinking: {"type": "adaptive"}`.
    pub adaptive_thinking: bool,
    /// Anthropic: the `thinking.type` that turns reasoning off, where it is
    /// not `disabled`.
    pub thinking_off: Option<String>,
    /// Anthropic: the model takes `role: "system"` inside `messages`.
    pub mid_conversation_system: bool,
    /// Anthropic: the model answers a forced `tool_choice` with an error.
    pub rejects_forced_tool_choice: bool,
    /// Anthropic: the model binds its thinking to the request's tools and
    /// system prompt.
    pub binds_context: bool,
    /// OpenAI: the prompt cache takes `prompt_cache_options`, not
    /// `prompt_cache_retention`.
    pub prompt_cache_options: bool,
    /// OpenAI: Chat Completions takes the model's tools only while its
    /// reasoning is off.
    pub chat_tools_need_reasoning_off: bool,
}

impl ModelSpec {
    /// Checks `options` against what the model takes: `reasoning` against
    /// [`Self::reasoning`] and `cache` against [`Self::caching`]. The error
    /// names the option, the provider's vendor and this model. A reasoning
    /// row with no reasoning options refuses every `reasoning` value (see
    /// [`ReasoningSupport`]).
    pub fn validate(&self, options: &GenerationOptions) -> Result<(), UnsupportedOption> {
        let refuse = |option: &'static str, reason: String| {
            UnsupportedOption::new(option, self.provider.vendor(), &self.id, reason)
        };
        if let Some(reason) = options
            .reasoning
            .as_ref()
            .and_then(|reasoning| self.reasoning.refusal(reasoning))
        {
            return Err(refuse("reasoning", reason));
        }
        if let Some(reason) = options
            .cache
            .as_ref()
            .and_then(|cache| self.caching.refusal(cache))
        {
            return Err(refuse("cache", reason));
        }
        Ok(())
    }
}

impl ReasoningSupport {
    /// Why the model cannot take `reasoning`, or `None` when it can.
    pub fn refusal(&self, reasoning: &Reasoning) -> Option<String> {
        match reasoning {
            Reasoning::Off if !self.supported || self.can_disable => None,
            Reasoning::Off => Some("reasoning cannot be turned off on this model".to_owned()),
            _ if !self.supported => Some("the model does not reason".to_owned()),
            Reasoning::Effort(effort) if self.levels.contains(effort) => None,
            Reasoning::Effort(_) if self.levels.is_empty() && self.budget.is_some() => {
                Some("the model takes a reasoning budget, not an effort level".to_owned())
            }
            Reasoning::Effort(_) if self.levels.is_empty() => {
                Some("the model takes no effort level".to_owned())
            }
            Reasoning::Effort(effort) => Some(format!(
                "the model has no `{}` effort level",
                effort.as_str()
            )),
            Reasoning::Budget { tokens } => match &self.budget {
                Some(range) if range.contains(tokens) => None,
                Some(range) => Some(format!(
                    "the model takes a reasoning budget from {} to {} tokens",
                    range.start(),
                    range.end()
                )),
                None if self.levels.is_empty() => {
                    Some("the model takes no reasoning budget".to_owned())
                }
                None => Some("the model takes an effort level, not a reasoning budget".to_owned()),
            },
        }
    }
}

impl CacheSupport {
    /// Why the model cannot honour `cache`, or `None` when it can or the
    /// catalog does not know.
    pub fn refusal(&self, cache: &CacheRetention) -> Option<String> {
        (!self.retention.is_empty() && !self.retention.contains(cache)).then(|| {
            format!(
                "the model does not honour `{}` cache retention",
                retention_word(cache)
            )
        })
    }
}

/// The lower-case word `cache` serializes as.
pub(super) fn retention_word(cache: &CacheRetention) -> &'static str {
    match cache {
        CacheRetention::None => "none",
        CacheRetention::Short => "short",
        CacheRetention::Long => "long",
    }
}
