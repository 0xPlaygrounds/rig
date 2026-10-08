//! Portable generation options: provider-neutral knobs a request sets once
//! and every completion wire maps to its own JSON, or refuses. They hold only
//! what [`CompletionRequest`](super::CompletionRequest) has no field for, so
//! `temperature`, `max_tokens`, `tool_choice` and `output_schema` stay on the
//! request.
//!
//! ```
//! use rig_core::completion::{CacheRetention, CompletionRequest, Effort, GenerationOptions};
//!
//! let options = GenerationOptions::default()
//!     .reasoning(Effort::High)
//!     .cache(CacheRetention::Long);
//! let request = CompletionRequest::new("Plan the refactor.").options(options);
//! assert!(!request.options.is_default());
//! ```

use std::borrow::Cow;

use serde::{Deserialize, Serialize};

mod mapping;
mod merge;

pub use mapping::{Mapping, OptionFields, OptionMap};
pub use merge::{BaseInput, FinalBody, RawAt, Rewrite, check, param, request_params};
pub(crate) use merge::{CatalogRefusal, catalog_refusals, collect_refusals};

/// Provider-neutral generation knobs for one request. An unset field leaves
/// the provider's default. A field the wire or model cannot honour is
/// reported through [`Self::unsupported_policy`], never silently dropped.
///
/// This is the reusable value: build it once and hand it to an agent
/// (`AgentBuilder::options`), overlay a run's on an agent's
/// ([`Self::overlay`]), or store it serialized with a request. For a single
/// request, `CompletionRequest`, `AgentBuilder` and `AgentRunner` also have
/// a shortcut per field (`.seed(7)`) that writes that one field into their
/// options.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GenerationOptions {
    /// How much the model reasons before answering.
    #[serde(default)]
    pub reasoning: Option<Reasoning>,
    /// How long the provider keeps the prompt prefix cached.
    #[serde(default)]
    pub cache: Option<CacheRetention>,
    /// The processing tier the request asks for.
    #[serde(default)]
    pub service_tier: Option<ServiceTier>,
    /// How long the answer should be.
    #[serde(default)]
    pub verbosity: Option<Verbosity>,
    /// Whether the model may call several tools in one turn.
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    /// Nucleus sampling probability mass.
    #[serde(default)]
    pub top_p: Option<f64>,
    /// Sampling seed, for providers that offer best-effort determinism.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Sequences that end generation. Empty means none.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
    /// What happens to an option the wire or model cannot honour. `None`
    /// leaves the policy unset, which acts as [`OnUnsupported::Error`]
    /// ([`Self::unsupported_policy`]); a set policy counts as setting an
    /// option.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub on_unsupported: Option<OnUnsupported>,
}

impl GenerationOptions {
    /// No option set and no policy set; the same as [`Self::default`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether every field holds its default: no option set, and no
    /// policy set. A request whose options are default is not checked
    /// against its model's catalog rules; setting any field, the policy
    /// included, turns the checks on.
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    /// The policy in effect: the one set, or [`OnUnsupported::Error`].
    pub fn unsupported_policy(&self) -> OnUnsupported {
        self.on_unsupported.unwrap_or_default()
    }

    /// Set the reasoning level or budget.
    pub fn reasoning(mut self, reasoning: impl Into<Reasoning>) -> Self {
        self.reasoning = Some(reasoning.into());
        self
    }

    /// Set the cache retention.
    pub fn cache(mut self, cache: CacheRetention) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Set the service tier.
    pub fn service_tier(mut self, tier: ServiceTier) -> Self {
        self.service_tier = Some(tier);
        self
    }

    /// Set the answer verbosity.
    pub fn verbosity(mut self, verbosity: Verbosity) -> Self {
        self.verbosity = Some(verbosity);
        self
    }

    /// Allow or forbid several tool calls in one turn.
    pub fn parallel_tool_calls(mut self, parallel: bool) -> Self {
        self.parallel_tool_calls = Some(parallel);
        self
    }

    /// Set the nucleus sampling probability mass.
    pub fn top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Set the sampling seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Replace the stop sequences.
    pub fn stop<S: Into<String>>(mut self, stop: impl IntoIterator<Item = S>) -> Self {
        self.stop = stop.into_iter().map(Into::into).collect();
        self
    }

    /// Set what happens to an option the wire or model cannot honour.
    pub fn on_unsupported(mut self, policy: OnUnsupported) -> Self {
        self.on_unsupported = Some(policy);
        self
    }

    /// Every option but the policy, borrowed, as a wire maps it.
    /// Destructures `self` with no `..`, so a new field fails to compile
    /// here first.
    pub fn fields(&self) -> OptionFields<'_> {
        let Self {
            reasoning,
            cache,
            service_tier,
            verbosity,
            parallel_tool_calls,
            top_p,
            seed,
            stop,
            on_unsupported: _,
        } = self;
        OptionFields {
            reasoning: reasoning.as_ref(),
            cache: cache.as_ref(),
            service_tier: service_tier.as_ref(),
            verbosity: verbosity.as_ref(),
            parallel_tool_calls: *parallel_tool_calls,
            top_p: *top_p,
            seed: *seed,
            stop,
        }
    }

    /// `self` with every field `over` sets put on top: a `Some` option, a
    /// non-empty `stop` list, a set `on_unsupported` (so a run can restore
    /// `Error` over an agent's `Ignore`). Every other
    /// field keeps `self`'s value. An agent's options overlaid with a run's
    /// give the run's where it sets one.
    pub fn overlay(self, over: &GenerationOptions) -> GenerationOptions {
        let GenerationOptions {
            reasoning,
            cache,
            service_tier,
            verbosity,
            parallel_tool_calls,
            top_p,
            seed,
            stop,
            on_unsupported,
        } = over;
        GenerationOptions {
            reasoning: reasoning.or(self.reasoning),
            cache: cache.or(self.cache),
            service_tier: service_tier.or(self.service_tier),
            verbosity: verbosity.or(self.verbosity),
            parallel_tool_calls: parallel_tool_calls.or(self.parallel_tool_calls),
            top_p: top_p.or(self.top_p),
            seed: seed.or(self.seed),
            stop: if stop.is_empty() {
                self.stop
            } else {
                stop.clone()
            },
            on_unsupported: on_unsupported.or(self.on_unsupported),
        }
    }
}

/// How much the model reasons before answering.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Reasoning {
    /// No reasoning.
    Off,
    /// A portable effort level.
    Effort(Effort),
    /// An explicit reasoning-token budget.
    Budget {
        /// The most tokens the model may spend reasoning.
        tokens: u32,
    },
}

impl From<Effort> for Reasoning {
    fn from(effort: Effort) -> Self {
        Self::Effort(effort)
    }
}

/// A portable reasoning effort level, lowest first.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Effort {
    /// The least reasoning the model offers.
    Minimal,
    /// Low effort.
    Low,
    /// Medium effort.
    Medium,
    /// High effort.
    High,
    /// Extra-high effort.
    XHigh,
    /// The most reasoning the model offers.
    Max,
}

impl Effort {
    /// The level's lower-case wire word, as its serde name spells it:
    /// `"minimal"`, `"low"`, `"medium"`, `"high"`, `"xhigh"` or `"max"`.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
            Self::Max => "max",
        }
    }
}

/// How long the provider keeps the prompt prefix cached.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CacheRetention {
    /// Do not cache.
    None,
    /// The provider's short retention, typically minutes.
    Short,
    /// The provider's long retention, typically an hour or more.
    Long,
}

/// The processing tier a request asks for.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ServiceTier {
    /// The provider chooses.
    Auto,
    /// The standard tier.
    Default,
    /// Cheaper, slower processing.
    Flex,
    /// Faster, dearer processing.
    Priority,
}

/// How long the answer should be.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Verbosity {
    /// Terse.
    Low,
    /// Balanced.
    Medium,
    /// Detailed.
    High,
}

impl Verbosity {
    /// The level's lower-case wire word: `"low"`, `"medium"` or `"high"`.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }
}

/// What happens to an option the wire or model cannot honour.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OnUnsupported {
    /// The request fails with [`UnsupportedOption`].
    #[default]
    Error,
    /// The option is skipped with a warning.
    Ignore,
}

/// An option the wire or model cannot honour. Carried by
/// [`ProviderError::UnsupportedOption`](crate::error::ProviderError::UnsupportedOption).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("`{option}` is not supported by {provider} model `{model}`: {reason}")]
pub struct UnsupportedOption {
    /// The [`GenerationOptions`] field name, such as `"reasoning"`, a
    /// provider option's `"<provider>.<section>.<field>"`, such as
    /// `"openrouter.*.provider"`, or the body key a model's catalog entry
    /// refuses, such as `"temperature"` or `"tools"`.
    pub option: Cow<'static, str>,
    /// The provider that refused it.
    pub provider: String,
    /// The model the request resolved to.
    pub model: String,
    /// Why it cannot be honoured.
    pub reason: String,
}

impl UnsupportedOption {
    /// The refusal of `option` by `provider` for `model`, because of `reason`.
    pub fn new(
        option: impl Into<Cow<'static, str>>,
        provider: impl Into<String>,
        model: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            option: option.into(),
            provider: provider.into(),
            model: model.into(),
            reason: reason.into(),
        }
    }
}

/// Why a dry run of a completion request failed: what
/// [`DynModel::check`](crate::DynModel::check) returns.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum CheckError {
    /// Every option the wire or model cannot honour, in the order the wire
    /// meets them. The request builds once they are removed.
    #[error("{}", unsupported_list(.0))]
    Unsupported(Vec<UnsupportedOption>),
    /// The request cannot be built for another reason, such as an empty
    /// history. Refusals met before it are not listed, since the build
    /// stopped there.
    #[error(transparent)]
    Invalid(crate::error::ProviderError),
}

/// The refusals, one per clause.
fn unsupported_list(refusals: &[UnsupportedOption]) -> String {
    refusals
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("; ")
}

#[cfg(test)]
mod tests;
