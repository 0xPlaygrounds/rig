//! OpenRouter routing preferences, model identifiers, and typed completion metadata.
//!
//! ```
//! use rig_core::providers::openrouter::ProviderPreferences;
//! let params = ProviderPreferences::new().cheapest().to_json();
//! assert!(params.get("provider").is_some());
//! ```

use serde::{Deserialize, Serialize};

use crate::completion;
use crate::providers::internal::openai_chat_completions_compatible::{
    map_native_finish_reason, map_openai_finish_reason,
};
use crate::providers::openai::completion::Message;

/// The `qwen/qwq-32b` model. Find more models at <https://openrouter.ai/models>.
pub const QWEN_QWQ_32B: &str = "qwen/qwq-32b";
/// The `anthropic/claude-3.7-sonnet` model. Find more models at <https://openrouter.ai/models>.
pub const CLAUDE_3_7_SONNET: &str = "anthropic/claude-3.7-sonnet";
/// The `perplexity/sonar-pro` model. Find more models at <https://openrouter.ai/models>.
pub const PERPLEXITY_SONAR_PRO: &str = "perplexity/sonar-pro";
/// The `google/gemini-2.0-flash-001` model. Find more models at <https://openrouter.ai/models>.
pub const GEMINI_FLASH_2_0: &str = "google/gemini-2.0-flash-001";

/// Data collection policy for providers.
///
/// Controls whether providers are allowed to collect and store request data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum DataCollection {
    /// Allow providers that may collect data (default)
    #[default]
    Allow,
    /// Restrict routing to providers that do not store user data non-transiently
    Deny,
}

/// Model quantization levels supported by OpenRouter.
///
/// Restrict routing to providers serving a specific quantization level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Quantization {
    /// 4-bit integer quantization
    #[serde(rename = "int4")]
    Int4,
    /// 8-bit integer quantization
    #[serde(rename = "int8")]
    Int8,
    /// 16-bit floating point
    #[serde(rename = "fp16")]
    Fp16,
    /// Brain floating point 16-bit
    #[serde(rename = "bf16")]
    Bf16,
    /// 32-bit floating point (full precision)
    #[serde(rename = "fp32")]
    Fp32,
    /// 8-bit floating point
    #[serde(rename = "fp8")]
    Fp8,
    /// Unknown or custom quantization level
    #[serde(rename = "unknown")]
    Unknown,
}

/// Simple sorting strategy for providers.
///
/// Determines how providers should be prioritized when multiple are available.
/// If you set `sort`, default load balancing is disabled and providers are tried
/// deterministically in the resulting order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderSortStrategy {
    /// Sort by price (cheapest first)
    Price,
    /// Sort by throughput (higher tokens/sec first)
    Throughput,
    /// Sort by latency (lower latency first)
    Latency,
}

/// Partition strategy for multi-model requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SortPartition {
    /// Sort providers within each model group (default)
    Model,
    /// Sort providers globally across all models
    None,
}

/// Complex sorting configuration with partition support.
///
/// For multi-model requests, allows control over how providers are sorted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderSortConfig {
    /// Sorting strategy
    pub by: ProviderSortStrategy,

    /// Partition strategy (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub partition: Option<SortPartition>,
}

impl ProviderSortConfig {
    /// Create a new sort config with the given strategy
    pub fn new(by: ProviderSortStrategy) -> Self {
        Self {
            by,
            partition: None,
        }
    }

    /// Set partition strategy for multi-model requests
    pub fn partition(mut self, partition: SortPartition) -> Self {
        self.partition = Some(partition);
        self
    }
}

/// Sort configuration - can be a simple string or a complex object.
///
/// Use `ProviderSort::Simple` for basic sorting, or `ProviderSort::Complex`
/// for multi-model requests with partition control.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ProviderSort {
    /// Simple sorting by a single strategy
    Simple(ProviderSortStrategy),
    /// Complex sorting with partition support
    Complex(ProviderSortConfig),
}

impl From<ProviderSortStrategy> for ProviderSort {
    fn from(strategy: ProviderSortStrategy) -> Self {
        ProviderSort::Simple(strategy)
    }
}

impl From<ProviderSortConfig> for ProviderSort {
    fn from(config: ProviderSortConfig) -> Self {
        ProviderSort::Complex(config)
    }
}

/// Throughput threshold configuration with percentile support.
///
/// Endpoints not meeting the threshold are deprioritized (moved later), not excluded.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ThroughputThreshold {
    /// Simple threshold in tokens/sec
    Simple(f64),
    /// Percentile-based thresholds
    Percentile(PercentileThresholds),
}

/// Latency threshold configuration with percentile support.
///
/// Endpoints not meeting the threshold are deprioritized, not excluded.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum LatencyThreshold {
    /// Simple threshold in seconds
    Simple(f64),
    /// Percentile-based thresholds
    Percentile(PercentileThresholds),
}

/// Percentile-based thresholds for throughput or latency.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PercentileThresholds {
    /// 50th percentile threshold
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p50: Option<f64>,
    /// 75th percentile threshold
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p75: Option<f64>,
    /// 90th percentile threshold
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p90: Option<f64>,
    /// 99th percentile threshold
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p99: Option<f64>,
}

impl PercentileThresholds {
    /// Create new empty percentile thresholds
    pub fn new() -> Self {
        Self::default()
    }

    /// Set p50 threshold
    pub fn p50(mut self, value: f64) -> Self {
        self.p50 = Some(value);
        self
    }

    /// Set p75 threshold
    pub fn p75(mut self, value: f64) -> Self {
        self.p75 = Some(value);
        self
    }

    /// Set p90 threshold
    pub fn p90(mut self, value: f64) -> Self {
        self.p90 = Some(value);
        self
    }

    /// Set p99 threshold
    pub fn p99(mut self, value: f64) -> Self {
        self.p99 = Some(value);
        self
    }
}

/// Maximum price configuration for hard ceiling on costs.
///
/// If no eligible provider is at or under the ceiling, the request fails.
/// Units are OpenRouter pricing units (e.g., dollars per million tokens).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MaxPrice {
    /// Maximum price per prompt token
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<f64>,
    /// Maximum price per completion token
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion: Option<f64>,
    /// Maximum price per request
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request: Option<f64>,
    /// Maximum price per image
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<f64>,
}

impl MaxPrice {
    /// Create new empty max price config
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum price per prompt token
    pub fn prompt(mut self, price: f64) -> Self {
        self.prompt = Some(price);
        self
    }

    /// Set maximum price per completion token
    pub fn completion(mut self, price: f64) -> Self {
        self.completion = Some(price);
        self
    }

    /// Set maximum price per request
    pub fn request(mut self, price: f64) -> Self {
        self.request = Some(price);
        self
    }

    /// Set maximum price per image
    pub fn image(mut self, price: f64) -> Self {
        self.image = Some(price);
        self
    }
}

/// Request-side provider eligibility and ordering preferences. Unset fields are
/// omitted, leaving gateway defaults in effect. Rig serializes these preferences
/// without validating provider slugs or numeric limits.
///
/// See <https://openrouter.ai/docs/guides/routing/provider-selection>.
///
/// ```rust
/// use rig_core::providers::openrouter::{ProviderPreferences, ProviderSortStrategy, Quantization};
///
/// let prefs = ProviderPreferences::new()
///     .sort(ProviderSortStrategy::Throughput)
///     .zdr(true)
///     .quantizations([Quantization::Int8])
///     .only(["anthropic", "openai"]);
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ProviderPreferences {
    /// Try these provider slugs in the given order first.
    /// If `allow_fallbacks: true`, OpenRouter may try other providers after this list is exhausted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub order: Option<Vec<String>>,

    /// Hard allowlist. Only these provider slugs are eligible.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub only: Option<Vec<String>>,

    /// Blocklist. These provider slugs are never used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ignore: Option<Vec<String>>,

    /// If `false`, the router will not use any providers outside what your constraints permit.
    /// Default is `true`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allow_fallbacks: Option<bool>,

    /// If `true`, only route to providers that support all parameters in your request.
    ///
    /// This is recommended for structured outputs so OpenRouter only selects
    /// providers that support the generated `response_format` parameter.
    /// Default is `false`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub require_parameters: Option<bool>,

    /// Data collection policy. If [`DataCollection::Deny`], restrict routing to providers
    /// that do not store user data non-transiently. Default is [`DataCollection::Allow`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_collection: Option<DataCollection>,

    /// If `true`, restrict routing to Zero Data Retention endpoints only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub zdr: Option<bool>,

    /// Sorting strategy. Affects ordering, not strict exclusion.
    /// If set, default load balancing is disabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sort: Option<ProviderSort>,

    /// Throughput threshold. Endpoints not meeting the threshold are deprioritized.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preferred_min_throughput: Option<ThroughputThreshold>,

    /// Latency threshold. Endpoints not meeting the threshold are deprioritized.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preferred_max_latency: Option<LatencyThreshold>,

    /// Hard price ceiling. If no provider is at or under, the request fails.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_price: Option<MaxPrice>,

    /// Restrict routing to providers serving specific quantization levels.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantizations: Option<Vec<Quantization>>,
}

impl ProviderPreferences {
    /// Create a new empty provider preferences struct
    pub fn new() -> Self {
        Self::default()
    }

    /// Try these provider slugs in the given order first.
    ///
    /// If `allow_fallbacks` is true (default), OpenRouter may try other providers
    /// after this list is exhausted.
    ///
    /// ```rust
    /// use rig_core::providers::openrouter::ProviderPreferences;
    ///
    /// let prefs = ProviderPreferences::new()
    ///     .order(["anthropic", "openai"]);
    /// ```
    pub fn order(mut self, providers: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.order = Some(
            providers
                .into_iter()
                .map(std::convert::Into::into)
                .collect(),
        );
        self
    }

    /// Restrict eligibility to these provider slugs.
    ///
    /// ```rust
    /// use rig_core::providers::openrouter::ProviderPreferences;
    ///
    /// let prefs = ProviderPreferences::new()
    ///     .only(["azure", "together"])
    ///     .allow_fallbacks(false);
    /// ```
    pub fn only(mut self, providers: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.only = Some(
            providers
                .into_iter()
                .map(std::convert::Into::into)
                .collect(),
        );
        self
    }

    /// Exclude these provider slugs.
    ///
    /// ```rust
    /// use rig_core::providers::openrouter::ProviderPreferences;
    ///
    /// let prefs = ProviderPreferences::new()
    ///     .ignore(["deepinfra"]);
    /// ```
    pub fn ignore(mut self, providers: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.ignore = Some(
            providers
                .into_iter()
                .map(std::convert::Into::into)
                .collect(),
        );
        self
    }

    /// Control whether fallbacks are allowed.
    ///
    /// If `false`, the router will not use any providers outside what your constraints permit.
    /// Default is `true`.
    pub fn allow_fallbacks(mut self, allow: bool) -> Self {
        self.allow_fallbacks = Some(allow);
        self
    }

    /// If `true`, only route to providers that support all parameters in your request.
    ///
    /// Default is `false`, meaning providers may ignore unsupported parameters.
    pub fn require_parameters(mut self, require: bool) -> Self {
        self.require_parameters = Some(require);
        self
    }

    /// Set data collection policy.
    ///
    /// If `Deny`, restrict routing to providers that do not store user data non-transiently.
    pub fn data_collection(mut self, policy: DataCollection) -> Self {
        self.data_collection = Some(policy);
        self
    }

    /// Restrict routing to Zero Data Retention endpoints when enabled.
    ///
    /// ```rust
    /// use rig_core::providers::openrouter::ProviderPreferences;
    ///
    /// let prefs = ProviderPreferences::new()
    ///     .zdr(true);
    /// ```
    pub fn zdr(mut self, enable: bool) -> Self {
        self.zdr = Some(enable);
        self
    }

    /// Set the sorting strategy for providers.
    ///
    /// Disable default load balancing and try providers in the resulting order.
    ///
    /// ```rust
    /// use rig_core::providers::openrouter::{ProviderPreferences, ProviderSortStrategy};
    ///
    /// let prefs = ProviderPreferences::new()
    ///     .sort(ProviderSortStrategy::Latency);
    /// ```
    pub fn sort(mut self, sort: impl Into<ProviderSort>) -> Self {
        self.sort = Some(sort.into());
        self
    }

    /// Set preferred minimum throughput threshold.
    ///
    /// Endpoints below the threshold are deprioritized, not excluded.
    ///
    /// ```rust
    /// use rig_core::providers::openrouter::{ProviderPreferences, ThroughputThreshold, PercentileThresholds};
    ///
    /// let prefs = ProviderPreferences::new()
    ///     .preferred_min_throughput(ThroughputThreshold::Simple(50.0));
    ///
    /// let prefs = ProviderPreferences::new()
    ///     .preferred_min_throughput(ThroughputThreshold::Percentile(
    ///         PercentileThresholds::new().p90(50.0)
    ///     ));
    /// ```
    pub fn preferred_min_throughput(mut self, threshold: ThroughputThreshold) -> Self {
        self.preferred_min_throughput = Some(threshold);
        self
    }

    /// Set preferred maximum latency threshold.
    ///
    /// Endpoints not meeting the threshold are deprioritized, not excluded.
    pub fn preferred_max_latency(mut self, threshold: LatencyThreshold) -> Self {
        self.preferred_max_latency = Some(threshold);
        self
    }

    /// Set maximum price ceiling.
    ///
    /// If no eligible provider is at or under the ceiling, the request fails.
    pub fn max_price(mut self, price: MaxPrice) -> Self {
        self.max_price = Some(price);
        self
    }

    /// Restrict routing to providers serving these quantization levels.
    ///
    /// ```rust
    /// use rig_core::providers::openrouter::{ProviderPreferences, Quantization};
    ///
    /// let prefs = ProviderPreferences::new()
    ///     .quantizations([Quantization::Int8, Quantization::Fp16]);
    /// ```
    pub fn quantizations(mut self, quantizations: impl IntoIterator<Item = Quantization>) -> Self {
        self.quantizations = Some(quantizations.into_iter().collect());
        self
    }

    /// Convenience: Sort by price (cheapest first)
    pub fn cheapest(self) -> Self {
        self.sort(ProviderSortStrategy::Price)
    }

    /// Convert to JSON value for use in additional_params
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "provider": self
        })
    }
}

fn deserialize_openrouter_choices_dropping_incomplete_tool_calls<'de, D>(
    deserializer: D,
) -> Result<Vec<Choice>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    crate::providers::internal::openai_chat_completions_compatible::deserialize_choices_dropping_incomplete_tool_calls_when(
        deserializer,
        |choice| {
            let normalized = choice
                .get("finish_reason")
                .and_then(serde_json::Value::as_str)
                .filter(|reason| !reason.is_empty());
            if let Some(reason) = normalized {
                return matches!(map_openai_finish_reason(reason), completion::FinishReason::Length);
            }

            choice
                .get("native_finish_reason")
                .and_then(serde_json::Value::as_str)
                .filter(|reason| !reason.is_empty())
                .is_some_and(|reason| {
                    matches!(map_native_finish_reason(reason), completion::FinishReason::Length)
                })
        },
    )
}

/// Typed OpenRouter completion response.
///
/// For more information, see the
/// [OpenRouter Chat Completions reference](https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    #[serde(deserialize_with = "deserialize_openrouter_choices_dropping_incomplete_tool_calls")]
    pub choices: Vec<Choice>,
    pub system_fingerprint: Option<String>,
    /// Upstream provider selected by OpenRouter for this response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// Service tier reported by the routed provider, when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    pub usage: Option<Usage>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Choice {
    pub index: usize,
    pub native_finish_reason: Option<String>,
    pub message: Message,
    pub finish_reason: Option<String>,
    /// Per-token probability metadata returned when `logprobs` is requested.
    ///
    /// Normalized completions intentionally omit provider-native
    /// probabilities; they stay readable through `CompletionResponse::raw`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<serde_json::Value>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    #[serde(default)]
    pub completion_tokens: usize,
    pub total_tokens: usize,
    #[serde(default)]
    pub cost: f64,
    /// Prompt-token cache breakdown, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    /// Completion-token breakdown, including the reasoning share when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

/// Prompt-token breakdown reported by OpenRouter for cached requests.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Default)]
pub struct PromptTokensDetails {
    /// Tokens served from cache (cache hit).
    #[serde(default)]
    pub cached_tokens: usize,
    /// Tokens written to cache on this call (cache miss that populated the cache).
    #[serde(default)]
    pub cache_write_tokens: usize,
}

/// Completion-token breakdown containing the reported reasoning share.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Default)]
pub struct CompletionTokensDetails {
    /// Tokens the upstream spent on hidden reasoning, counted inside
    /// `completion_tokens`.
    #[serde(default)]
    pub reasoning_tokens: usize,
}

#[cfg(test)]
mod tests;
