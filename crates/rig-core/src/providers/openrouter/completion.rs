use super::client::{OpenRouterExt, Usage};
use crate::message::{self, DocumentMediaType, DocumentSourceKind, MimeType};
use crate::telemetry::ProviderResponseExt;
use crate::{
    completion::{self, CompletionError, CompletionRequest},
    json_utils,
    providers::internal::openai_chat_completions_compatible::map_openai_finish_reason,
    providers::openai,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ================================================================
// OpenRouter Completion API
// ================================================================

/// The `qwen/qwq-32b` model. Find more models at <https://openrouter.ai/models>.
pub const QWEN_QWQ_32B: &str = "qwen/qwq-32b";
/// The `anthropic/claude-3.7-sonnet` model. Find more models at <https://openrouter.ai/models>.
pub const CLAUDE_3_7_SONNET: &str = "anthropic/claude-3.7-sonnet";
/// The `perplexity/sonar-pro` model. Find more models at <https://openrouter.ai/models>.
pub const PERPLEXITY_SONAR_PRO: &str = "perplexity/sonar-pro";
/// The `google/gemini-2.0-flash-001` model. Find more models at <https://openrouter.ai/models>.
pub const GEMINI_FLASH_2_0: &str = "google/gemini-2.0-flash-001";

/// Stable descriptor name recorded on telemetry spans and on every normalized
/// response produced by this provider.
pub(crate) const PROVIDER_NAME: &str = "openrouter";

// ================================================================
// Provider Selection and Prioritization
// ================================================================
// See: https://openrouter.ai/docs/guides/routing/provider-selection

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

/// Provider preferences for OpenRouter routing.
///
/// This struct allows you to control which providers are used and how they are prioritized
/// when making requests through OpenRouter.
///
/// See: <https://openrouter.ai/docs/guides/routing/provider-selection>
///
/// # Example
///
/// ```rust
/// use rig_core::providers::openrouter::{ProviderPreferences, ProviderSortStrategy, Quantization};
///
/// // Create preferences for zero data retention providers, sorted by throughput
/// let prefs = ProviderPreferences::new()
///     .sort(ProviderSortStrategy::Throughput)
///     .zdr(true)
///     .quantizations([Quantization::Int8])
///     .only(["anthropic", "openai"]);
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ProviderPreferences {
    // === Provider Selection Controls ===
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

    // === Compatibility and Policy Filters ===
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

    // === Performance and Cost Preferences ===
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

    // === Quantization Filter ===
    /// Restrict routing to providers serving specific quantization levels.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantizations: Option<Vec<Quantization>>,
}

impl ProviderPreferences {
    /// Create a new empty provider preferences struct
    pub fn new() -> Self {
        Self::default()
    }

    // === Provider Selection Controls ===

    /// Try these provider slugs in the given order first.
    ///
    /// If `allow_fallbacks` is true (default), OpenRouter may try other providers
    /// after this list is exhausted.
    ///
    /// # Example
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

    /// Hard allowlist. Only these provider slugs are eligible.
    ///
    /// # Example
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

    /// Blocklist. These provider slugs are never used.
    ///
    /// # Example
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

    // === Compatibility and Policy Filters ===

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

    /// If `true`, restrict routing to Zero Data Retention endpoints only.
    ///
    /// # Example
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

    // === Performance and Cost Preferences ===

    /// Set the sorting strategy for providers.
    ///
    /// If set, default load balancing is disabled and providers are tried
    /// deterministically in the resulting order.
    ///
    /// # Example
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
    /// Endpoints not meeting the threshold are deprioritized (moved later), not excluded.
    ///
    /// # Example
    ///
    /// ```rust
    /// use rig_core::providers::openrouter::{ProviderPreferences, ThroughputThreshold, PercentileThresholds};
    ///
    /// // Simple threshold
    /// let prefs = ProviderPreferences::new()
    ///     .preferred_min_throughput(ThroughputThreshold::Simple(50.0));
    ///
    /// // Percentile threshold
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

    // === Quantization Filter ===

    /// Restrict routing to providers serving specific quantization levels.
    ///
    /// # Example
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

    // === Convenience Methods ===

    /// Convenience: Enable Zero Data Retention
    pub fn zero_data_retention(self) -> Self {
        self.zdr(true)
    }

    /// Convenience: Sort by throughput (higher tokens/sec first)
    pub fn fastest(self) -> Self {
        self.sort(ProviderSortStrategy::Throughput)
    }

    /// Convenience: Sort by price (cheapest first)
    pub fn cheapest(self) -> Self {
        self.sort(ProviderSortStrategy::Price)
    }

    /// Convenience: Sort by latency (lower latency first)
    pub fn lowest_latency(self) -> Self {
        self.sort(ProviderSortStrategy::Latency)
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

/// A openrouter completion object.
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

/// Normalize OpenRouter's terminal reason for a choice.
///
/// OpenRouter reports two fields: `finish_reason`, normalized by OpenRouter to
/// the OpenAI Chat Completions vocabulary, and `native_finish_reason`, which is
/// whatever the upstream provider said (e.g. Gemini's `"STOP"`). The normalized
/// field wins; the native one is only consulted when OpenRouter omitted the
/// normalized field, so a reason the gateway could not translate is still
/// reported rather than lost. The native value must not be read with the OpenAI
/// vocabulary — routing Gemini's `STOP` or Anthropic's `end_turn` through
/// [`map_openai_finish_reason`] would report a plain natural stop as
/// [`completion::FinishReason::Other`]. Either way an unrecognized value is
/// preserved verbatim in [`FinishReason::Other`](completion::FinishReason::Other).
pub(crate) fn map_finish_reason(choice: &Choice) -> Option<completion::FinishReason> {
    if let Some(reason) = choice
        .finish_reason
        .as_deref()
        .filter(|reason| !reason.is_empty())
    {
        return Some(map_openai_finish_reason(reason));
    }

    choice
        .native_finish_reason
        .as_deref()
        .filter(|reason| !reason.is_empty())
        .map(map_native_finish_reason)
}

/// Map an upstream provider's own terminal reason, as forwarded by OpenRouter.
///
/// This covers the vocabularies OpenRouter routes to, matched
/// case-insensitively because they disagree on casing (Gemini screams,
/// Anthropic does not). Anything unrecognized is still carried verbatim in the
/// spelling the upstream provider used.
pub(crate) fn map_native_finish_reason(reason: &str) -> completion::FinishReason {
    match reason.to_ascii_lowercase().as_str() {
        // OpenAI-compatible upstreams, plus Anthropic's `end_turn`/`stop_sequence`
        // and Gemini's `STOP`.
        "stop" | "end_turn" | "stop_sequence" | "complete" | "completed" => {
            completion::FinishReason::Stop
        }
        "length" | "max_tokens" | "max_output_tokens" | "model_length" => {
            completion::FinishReason::Length
        }
        "tool_calls" | "function_call" | "tool_use" => completion::FinishReason::ToolCalls,
        "content_filter" | "safety" | "blocklist" | "prohibited_content" | "spii" => {
            completion::FinishReason::ContentFilter
        }
        _ => completion::FinishReason::Other(reason.to_owned()),
    }
}

/// Normalize an OpenRouter chat completion response.
///
/// The provider descriptor name is an *input* because the message model is the
/// shared OpenAI one; taking it as part of the conversion keeps the shape
/// consistent with the OpenAI-compatible path even though only OpenRouter
/// produces this envelope.
impl crate::completion::NormalizeCompletionResponse for CompletionResponse {
    fn normalize(self, provider: &str) -> Result<completion::CompletionResponse, CompletionError> {
        let response = self;
        let choice = response.choices.first().ok_or_else(|| {
            CompletionError::ResponseError("Response contained no choices".to_owned())
        })?;
        let finish_reason = map_finish_reason(choice);

        let content = match &choice.message {
            Message::Assistant {
                content: message_content,
                tool_calls,
                reasoning,
                reasoning_details,
                images,
                refusal,
                ..
            } => {
                // A structured-output refusal arrives as a *sibling* of
                // `content` (`{"content": null, "refusal": "…"}`), not as the
                // `refusal` content part below — that spelling belongs to the
                // Responses API, which this wire never sends. Reading only
                // `content` dropped the refusal outright and normalized the
                // turn to nothing, while `text_response` already fell back
                // to the field and the streaming path already delivered it.
                // The rule is shared with the OpenAI chat path so no two
                // readers of this wire can disagree about it (#2332).
                let refusal_fallback = openai::completion::assistant_refusal_fallback(
                    message_content,
                    refusal.as_deref(),
                );

                // Match the shared streaming adapter's canonical turn order:
                // the model reasons, speaks, then acts. OpenRouter may place
                // reasoning beside text and tool calls in one blocking
                // message, so appending it after the calls made blocking and
                // streaming histories disagree for the same provider turn.
                let mut normalized_content = Vec::new();
                let mut grouped_reasoning: HashMap<
                    Option<String>,
                    Vec<(usize, usize, message::ReasoningContent)>,
                > = HashMap::new();
                let mut reasoning_order: Vec<Option<String>> = Vec::new();
                for (position, detail) in reasoning_details.iter().enumerate() {
                    let (reasoning_id, sort_index, parsed_content) = match detail {
                        ReasoningDetails::Summary {
                            id, index, summary, ..
                        } => (
                            id.clone(),
                            *index,
                            Some(message::ReasoningContent::Summary(summary.clone())),
                        ),
                        ReasoningDetails::Encrypted {
                            id, index, data, ..
                        } => (
                            id.clone(),
                            *index,
                            Some(message::ReasoningContent::Encrypted(data.clone())),
                        ),
                        ReasoningDetails::Text {
                            id,
                            index,
                            text,
                            signature,
                            ..
                        } => (
                            id.clone(),
                            *index,
                            text.as_ref().map(|text| message::ReasoningContent::Text {
                                text: text.clone(),
                                signature: signature.clone(),
                            }),
                        ),
                    };

                    let Some(parsed_content) = parsed_content else {
                        continue;
                    };
                    let sort_index = sort_index.unwrap_or(position);

                    let entry = grouped_reasoning.entry(reasoning_id.clone());
                    if matches!(entry, std::collections::hash_map::Entry::Vacant(_)) {
                        reasoning_order.push(reasoning_id);
                    }
                    entry
                        .or_default()
                        .push((sort_index, position, parsed_content));
                }

                if grouped_reasoning.is_empty() {
                    if let Some(reasoning) = reasoning {
                        normalized_content.push(completion::AssistantContent::reasoning(reasoning));
                    }
                } else {
                    for reasoning_id in reasoning_order {
                        let Some(mut blocks) = grouped_reasoning.remove(&reasoning_id) else {
                            continue;
                        };
                        blocks.sort_by_key(|(index, position, _)| (*index, *position));
                        normalized_content.push(completion::AssistantContent::Reasoning(
                            message::Reasoning {
                                id: reasoning_id,
                                content: blocks
                                    .into_iter()
                                    .map(|(_, _, content)| content)
                                    .collect::<Vec<_>>(),
                            },
                        ));
                    }
                }

                normalized_content.extend(message_content.iter().map(|part| match part {
                    openai::AssistantContent::Text { text, .. } => {
                        completion::AssistantContent::text(text)
                    }
                    openai::AssistantContent::Refusal { refusal } => {
                        completion::AssistantContent::text(refusal)
                    }
                }));

                if let Some(refusal) = refusal_fallback {
                    normalized_content.push(completion::AssistantContent::text(refusal));
                }

                normalized_content.extend(tool_calls.iter().map(|call| {
                    completion::AssistantContent::tool_call(
                        &call.id,
                        &call.function.name,
                        call.function.arguments.clone(),
                    )
                }));

                normalized_content.extend(images.iter().map(response_image_to_assistant_content));

                Ok(normalized_content)
            }
            _ => Err(CompletionError::ResponseError(
                "Response did not contain a valid message or tool call".into(),
            )),
        }?;

        // A provider-truncated turn can legitimately have no surviving
        // content (for example, its only tool call was cut off before the
        // first usable argument token). Preserve the terminal diagnostic and
        // metadata exactly as the shared OpenAI-compatible normalizer does;
        // completed empty turns remain errors.
        let choice = match &finish_reason {
            Some(reason) if reason.truncated_output() => content,
            _ => crate::message::require_non_empty_response(content)?,
        };

        let usage = response
            .usage
            .as_ref()
            .map(completion::Usage::from)
            .unwrap_or_default();

        Ok(
            // OpenRouter's `id` identifies the generation, not an assistant
            // message, so it is carried as a response ID rather than a
            // message ID.
            completion::CompletionResponse::new(choice, usage, provider)
                .with_response_id(response.id)
                .with_model(response.model)
                .with_optional_finish_reason(finish_reason),
        )
    }
}

impl ProviderResponseExt for CompletionResponse {
    type Usage = Usage;

    fn response_id(&self) -> Option<&str> {
        Some(self.id.as_str())
    }

    fn response_model_name(&self) -> Option<&str> {
        Some(self.model.as_str())
    }

    fn text_response(&self) -> Option<String> {
        let response = self
            .choices
            .iter()
            .filter_map(|choice| {
                openai::completion::assistant_message_text_response(&choice.message)
            })
            .collect::<Vec<_>>()
            .join("\n");

        (!response.is_empty()).then_some(response)
    }

    fn usage(&self) -> Option<Self::Usage> {
        self.usage
    }
}

// OpenRouter shares OpenAI's Chat Completions message model. The request and
// response *message* types are the shared OpenAI ones; only OpenRouter's
// response envelope, provider routing preferences, and the conversion rules
// below are provider-specific.
pub use crate::providers::openai::completion::{
    FileData, ImageUrl, Message, ReasoningDetails, ResponseImage, UserContent, VideoUrl,
};

const OPENROUTER_RESPONSE_ONLY_KEY: &str = "response_only";
const OPENROUTER_RESPONSE_IMAGE_SOURCE_KEY: &str = "source";
const OPENROUTER_ASSISTANT_IMAGES_SOURCE: &str = "assistant.images";

/// Split a `data:<mime>;base64,<payload>` URI into `(mime, payload)`.
/// Returns `None` for plain URLs or non-base64 data URIs.
fn parse_data_uri(url: &str) -> Option<(&str, &str)> {
    url.strip_prefix("data:")?.split_once(";base64,")
}

fn openrouter_response_image_params() -> Option<message::AdditionalParams> {
    message::AdditionalParams::from_entries([(
        "openrouter",
        serde_json::json!({
            OPENROUTER_RESPONSE_ONLY_KEY: true,
            OPENROUTER_RESPONSE_IMAGE_SOURCE_KEY: OPENROUTER_ASSISTANT_IMAGES_SOURCE,
        }),
    )])
}

fn response_image_to_assistant_content(image: &ResponseImage) -> completion::AssistantContent {
    let url = &image.image_url.url;
    if let Some((mime, b64)) = parse_data_uri(url) {
        completion::AssistantContent::Image(message::Image {
            data: message::DocumentSourceKind::Base64(b64.to_string()),
            media_type: message::ImageMediaType::from_mime_type(mime),
            detail: None,
            additional_params: openrouter_response_image_params(),
        })
    } else {
        completion::AssistantContent::Image(message::Image {
            data: message::DocumentSourceKind::Url(url.clone()),
            media_type: None,
            detail: None,
            additional_params: openrouter_response_image_params(),
        })
    }
}

fn is_openrouter_response_image(image: &message::Image) -> bool {
    image
        .additional_params
        .as_ref()
        .and_then(|params| params.wire_extras("openrouter"))
        .is_some_and(|params| {
            params
                .get(OPENROUTER_RESPONSE_ONLY_KEY)
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
                && params
                    .get(OPENROUTER_RESPONSE_IMAGE_SOURCE_KEY)
                    .and_then(|value| value.as_str())
                    == Some(OPENROUTER_ASSISTANT_IMAGES_SOURCE)
        })
}

/// Convert rig user content into OpenRouter's OpenAI-compatible content parts.
///
/// OpenRouter shares OpenAI's content schema but keeps its own conversion
/// rules:
/// - image `detail` passes through unchanged, so an absent detail stays
///   absent on the wire,
/// - documents accept URLs and non-PDF media types via `file_data`, while
///   provider file IDs are rejected,
/// - audio requires an explicit media type instead of defaulting to MP3.
///
/// Text and video content use the shared OpenAI conversion.
fn user_content_to_openai(
    value: message::UserContent,
) -> Result<UserContent, message::MessageError> {
    match value {
        message::UserContent::Image(message::Image {
            data,
            detail,
            media_type,
            ..
        }) => {
            let url = match data {
                DocumentSourceKind::Url(url) => url,
                DocumentSourceKind::Base64(data) => {
                    let mime = media_type
                        .ok_or_else(|| {
                            message::MessageError::ConversionError(
                                "Image media type required for base64 encoding".into(),
                            )
                        })?
                        .to_mime_type();
                    format!("data:{mime};base64,{data}")
                }
                DocumentSourceKind::Raw(_) => {
                    return Err(message::MessageError::ConversionError(
                        "Raw bytes not supported, encode as base64 first".into(),
                    ));
                }
                DocumentSourceKind::FileId(_) => {
                    return Err(message::MessageError::ConversionError(
                        "File IDs are not supported for images".into(),
                    ));
                }
                DocumentSourceKind::String(_) => {
                    return Err(message::MessageError::ConversionError(
                        "String source not supported for images".into(),
                    ));
                }
                DocumentSourceKind::Unknown => {
                    return Err(message::MessageError::ConversionError(
                        "Image has no data".into(),
                    ));
                }
            };
            Ok(UserContent::Image {
                image_url: ImageUrl { url, detail },
            })
        }

        message::UserContent::Document(message::Document {
            data, media_type, ..
        }) => match data {
            DocumentSourceKind::FileId(_) => Err(message::MessageError::ConversionError(
                "Provider file IDs are not supported for OpenRouter document inputs".into(),
            )),
            DocumentSourceKind::Url(url) => Ok(UserContent::File {
                file: FileData {
                    file_data: Some(url),
                    file_id: None,
                    filename: document_filename(media_type.as_ref()),
                },
            }),
            DocumentSourceKind::Base64(data) => {
                let mime = media_type.as_ref().map_or(
                    "application/pdf",
                    crate::completion::message::MimeType::to_mime_type,
                );
                let data_uri = format!("data:{mime};base64,{data}");

                Ok(UserContent::File {
                    file: FileData {
                        file_data: Some(data_uri),
                        file_id: None,
                        filename: document_filename(media_type.as_ref()),
                    },
                })
            }
            DocumentSourceKind::String(text) => Ok(UserContent::Text { text }),
            DocumentSourceKind::Raw(_) => Err(message::MessageError::ConversionError(
                "Raw bytes not supported for documents, encode as base64 first".into(),
            )),
            DocumentSourceKind::Unknown => Err(message::MessageError::ConversionError(
                "Document has no data".into(),
            )),
        },

        message::UserContent::Audio(message::Audio {
            data, media_type, ..
        }) => match data {
            DocumentSourceKind::Base64(data) => {
                let format = media_type.ok_or_else(|| {
                    message::MessageError::ConversionError(
                        "Audio media type required for base64 encoding".into(),
                    )
                })?;
                Ok(UserContent::Audio {
                    input_audio: openai::InputAudio { data, format },
                })
            }
            DocumentSourceKind::Url(_) => Err(message::MessageError::ConversionError(
                "OpenRouter does not support audio URLs, encode as base64 first".into(),
            )),
            DocumentSourceKind::Raw(_) => Err(message::MessageError::ConversionError(
                "Raw bytes not supported for audio, encode as base64 first".into(),
            )),
            DocumentSourceKind::FileId(_) => Err(message::MessageError::ConversionError(
                "File IDs are not supported for audio".into(),
            )),
            DocumentSourceKind::String(_) => Err(message::MessageError::ConversionError(
                "String source not supported for audio".into(),
            )),
            DocumentSourceKind::Unknown => Err(message::MessageError::ConversionError(
                "Audio has no data".into(),
            )),
        },

        message::UserContent::ToolResult(_) => Err(message::MessageError::ConversionError(
            "Tool results should be handled as separate messages".into(),
        )),

        // Text and video conversions are identical to the shared OpenAI ones.
        value => UserContent::try_from(value),
    }
}

fn document_filename(media_type: Option<&DocumentMediaType>) -> Option<String> {
    media_type.map(|mt| {
        match mt {
            DocumentMediaType::PDF => "document.pdf",
            DocumentMediaType::TXT => "document.txt",
            DocumentMediaType::HTML => "document.html",
            DocumentMediaType::MARKDOWN => "document.md",
            DocumentMediaType::CSV => "document.csv",
            DocumentMediaType::XML => "document.xml",
            _ => "document",
        }
        .to_string()
    })
}

fn user_contents_to_messages(
    value: Vec<message::UserContent>,
) -> Result<Vec<Message>, message::MessageError> {
    fn flush_user_content(messages: &mut Vec<Message>, pending: &mut Vec<UserContent>) {
        // An empty flush is a legal no-op — it fires between consecutive
        // tool-result groups — not a conversion error. This early return is
        // the only emptiness decision here; the pushed content is non-empty
        // because of it.
        if pending.is_empty() {
            return;
        }

        messages.push(Message::User {
            content: std::mem::take(pending),
            name: None,
        });
    }

    let mut messages = Vec::new();
    let mut pending = Vec::new();

    for content in value {
        match content {
            message::UserContent::ToolResult(tool_result) => {
                flush_user_content(&mut messages, &mut pending);
                // Prefer the provider-issued call id, matching the
                // assistant echo (shared From<message::ToolCall>);
                // provider-less results fall back to rig's minted
                // handle — never empty.
                let tool_call_id = tool_result.wire_call_id().to_owned();
                let content = tool_result
                    .content
                    .into_iter()
                    .map(|content| match content {
                        message::ToolResultContent::Text(message::Text { text, .. }) => Ok(text),
                        message::ToolResultContent::Json { value } => Ok(value.to_string()),
                        message::ToolResultContent::Image(_) => {
                            Err(message::MessageError::ConversionError(
                                "OpenRouter does not support images in tool results".into(),
                            ))
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?
                    .join("\n");
                messages.push(Message::ToolResult {
                    tool_call_id,
                    content: openai::completion::ToolResultContentValue::String(content),
                });
            }
            content => pending.push(user_content_to_openai(content)?),
        }
    }

    flush_user_content(&mut messages, &mut pending);
    Ok(messages)
}

// ================================================================
// Response Types
// ================================================================

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Choice {
    pub index: usize,
    pub native_finish_reason: Option<String>,
    pub message: Message,
    pub finish_reason: Option<String>,
    /// Per-token probability metadata returned when `logprobs` is requested.
    ///
    /// Normalized completions intentionally omit provider-native
    /// probabilities; callers of `raw_completion` retain the complete object.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize, PartialEq, Clone)]
#[serde(untagged)]
enum ToolCallAdditionalParams {
    ReasoningDetails(ReasoningDetails),
    Minimal {
        id: Option<String>,
        format: Option<String>,
    },
}

/// Replay assistant history — including structured reasoning — as OpenRouter
/// request messages.
///
/// Maps rig [`message::Reasoning`] blocks back onto the `reasoning_details`
/// field of the shared assistant message, and recovers reasoning metadata
/// stored on tool calls (signature / `additional_params`) so providers that
/// require reasoning to be echoed back on tool-call turns keep working.
fn assistant_contents_to_messages(
    value: Vec<message::AssistantContent>,
) -> Result<Vec<Message>, message::MessageError> {
    let mut text_content = Vec::new();
    let mut tool_calls = Vec::new();
    let mut reasoning = None;
    let mut reasoning_details = Vec::new();

    for content in value.into_iter() {
        match content {
            message::AssistantContent::Text(text) => text_content.push(text),
            message::AssistantContent::ToolCall(tool_call) => {
                // We usually want to provide back the reasoning to OpenRouter since some
                // providers require it.
                // 1. Full reasoning details passed back the user
                // 2. The signature, an id and a format if present
                // 3. The signature and the call_id if present
                if let Some(additional_params) = &tool_call.additional_params
                    && let Ok(additional_params) = serde_json::from_value::<ToolCallAdditionalParams>(
                        additional_params.clone(),
                    )
                {
                    match additional_params {
                        ToolCallAdditionalParams::ReasoningDetails(full) => {
                            reasoning_details.push(full);
                        }
                        ToolCallAdditionalParams::Minimal { id, format } => {
                            // Correlate with the id the wire tool call will
                            // carry (provider call id when present, else
                            // rig's handle).
                            let id = id
                                .or_else(|| {
                                    tool_call
                                        .provider
                                        .as_ref()
                                        .map(|provider| provider.call_id.clone())
                                })
                                .unwrap_or_else(|| tool_call.id.as_str().to_owned());
                            if let Some(signature) = &tool_call.signature {
                                reasoning_details.push(ReasoningDetails::Encrypted {
                                    id: Some(id),
                                    format,
                                    index: None,
                                    data: signature.clone(),
                                });
                            }
                        }
                    }
                } else if let Some(signature) = &tool_call.signature {
                    reasoning_details.push(ReasoningDetails::Encrypted {
                        id: Some(tool_call.provider.as_ref().map_or_else(
                            || tool_call.id.as_str().to_owned(),
                            |provider| provider.call_id.clone(),
                        )),
                        format: None,
                        index: None,
                        data: signature.clone(),
                    });
                }
                tool_calls.push(tool_call.into());
            }
            message::AssistantContent::Reasoning(r) => {
                if r.content.is_empty() {
                    let display = r.display_text();
                    if !display.is_empty() {
                        reasoning = Some(display);
                    }
                } else {
                    // A block the stream aggregated without a wire id carries
                    // the accumulator's shared "" identity; send it back as a
                    // null id, the shape the non-streaming path produces.
                    let reasoning_id = r.id.clone().filter(|id| !id.is_empty());
                    for reasoning_block in &r.content {
                        let index = Some(reasoning_details.len());
                        match reasoning_block {
                            message::ReasoningContent::Text { text, signature } => {
                                reasoning_details.push(ReasoningDetails::Text {
                                    id: reasoning_id.clone(),
                                    format: None,
                                    index,
                                    text: Some(text.clone()),
                                    signature: signature.clone(),
                                });
                            }
                            message::ReasoningContent::Summary(summary) => {
                                reasoning_details.push(ReasoningDetails::Summary {
                                    id: reasoning_id.clone(),
                                    format: None,
                                    index,
                                    summary: summary.clone(),
                                });
                            }
                            message::ReasoningContent::Encrypted(data)
                            | message::ReasoningContent::Redacted { data } => {
                                reasoning_details.push(ReasoningDetails::Encrypted {
                                    id: reasoning_id.clone(),
                                    format: None,
                                    index,
                                    data: data.clone(),
                                });
                            }
                        }
                    }
                }
            }
            message::AssistantContent::Image(image) if is_openrouter_response_image(&image) => {
                // OpenRouter generated images are response artifacts. They remain
                // visible in Rig history, but OpenRouter does not define them as
                // replayable assistant request content.
            }
            message::AssistantContent::Image(_) => {
                return Err(message::MessageError::ConversionError(
                        "OpenRouter does not support assistant image content in request history; pass images as user image inputs instead".into(),
                    ));
            }
        }
    }

    if text_content.is_empty()
        && tool_calls.is_empty()
        && reasoning.is_none()
        && reasoning_details.is_empty()
    {
        return Ok(vec![]);
    }

    Ok(vec![Message::Assistant {
        content: text_content
            .into_iter()
            .map(|content| content.text.into())
            .collect::<Vec<_>>(),
        refusal: None,
        audio: None,
        name: None,
        tool_calls,
        reasoning,
        reasoning_details,
        images: Vec::new(),
    }])
}

/// Convert a rig message into OpenRouter request messages.
///
/// OpenRouter shares the OpenAI message model, but keeps its own conversion
/// rules for user content (see `user_content_to_openai`) and for replaying
/// assistant reasoning, so it does not use the shared
/// `TryFrom<message::Message> for Vec<openai::Message>` conversion.
pub fn messages_from_rig_message(
    message: message::Message,
) -> Result<Vec<Message>, message::MessageError> {
    match message {
        message::Message::System { content } => Ok(vec![Message::system(&content)]),
        message::Message::User { content } => user_contents_to_messages(content),
        message::Message::Assistant { content, .. } => assistant_contents_to_messages(content),
    }
}

/// Apply explicit prompt-caching markers to an already-serialized OpenRouter
/// request body.
///
/// Finds the first system message in `messages` and converts its `content`
/// to a structured text block with `cache_control: {"type": "ephemeral"}`.
/// This tells OpenRouter providers that support explicit `cache_control`
/// breakpoints to cache the system prompt so subsequent turns that share the
/// same prefix can be billed at the cache-hit rate.
///
/// This is intended for models and providers that support explicit
/// `cache_control` breakpoints.
pub(super) fn apply_prompt_caching(body: &mut serde_json::Value) {
    let Some(obj) = body.as_object_mut() else {
        return;
    };
    let Some(messages) = obj.get_mut("messages").and_then(|v| v.as_array_mut()) else {
        return;
    };

    let Some(system_msg) = messages
        .iter_mut()
        .find(|m| m.get("role").and_then(|v| v.as_str()) == Some("system"))
    else {
        return;
    };

    match system_msg.get("content").cloned() {
        Some(serde_json::Value::String(s)) => {
            if let Some(obj) = system_msg.as_object_mut() {
                obj.insert(
                    "content".to_string(),
                    serde_json::json!([{
                        "type": "text",
                        "text": s,
                        "cache_control": { "type": "ephemeral" }
                    }]),
                );
            }
        }
        Some(serde_json::Value::Array(mut arr)) => {
            // Mark the last block as the cache boundary; all other blocks (including
            // non-text blocks such as images) are preserved unchanged.
            if let Some(last) = arr.last_mut()
                && let Some(obj) = last.as_object_mut()
            {
                obj.insert(
                    "cache_control".to_string(),
                    serde_json::json!({ "type": "ephemeral" }),
                );
            }
            if let Some(obj) = system_msg.as_object_mut() {
                obj.insert("content".to_string(), serde_json::Value::Array(arr));
            }
        }
        _ => {}
    }
}

pub(super) fn finalize_openrouter_request_body(body: &mut serde_json::Value, prompt_caching: bool) {
    if prompt_caching {
        apply_prompt_caching(body);
    }

    // The shared assistant message serializes hidden reasoning under the
    // llama.cpp/DeepSeek key `reasoning_content`; OpenRouter's documented
    // assistant field is `reasoning`.
    if let Some(messages) = body
        .get_mut("messages")
        .and_then(serde_json::Value::as_array_mut)
    {
        for message in messages {
            if let Some(message) = message.as_object_mut()
                && message.get("role").and_then(serde_json::Value::as_str) == Some("assistant")
                && let Some(reasoning) = message.remove("reasoning_content")
            {
                message.insert("reasoning".to_string(), reasoning);
            }
        }
    }
}

#[cfg(test)]
pub(super) fn final_request_body(
    request: &OpenrouterCompletionRequest,
    prompt_caching: bool,
) -> Result<serde_json::Value, CompletionError> {
    let mut body = serde_json::to_value(request)?;
    finalize_openrouter_request_body(&mut body, prompt_caching);
    Ok(body)
}

pub(super) type OpenrouterCompletionRequest = openai::completion::CompletionRequest;

/// Parameters for building an OpenRouter CompletionRequest
pub struct OpenRouterRequestParams<'a> {
    pub model: &'a str,
    pub request: CompletionRequest,
    pub strict_tools: bool,
}

impl TryFrom<OpenRouterRequestParams<'_>> for OpenrouterCompletionRequest {
    type Error = CompletionError;

    fn try_from(params: OpenRouterRequestParams) -> Result<Self, Self::Error> {
        let OpenRouterRequestParams {
            model,
            request: req,
            strict_tools,
        } = params;
        let chat_history = req.chat_history_with_documents();
        let model = req.model.clone().unwrap_or_else(|| model.to_string());

        let mut full_history: Vec<Message> = vec![];

        let chat_history: Vec<Message> = chat_history
            .into_iter()
            .map(messages_from_rig_message)
            .collect::<Result<Vec<Vec<Message>>, _>>()?
            .into_iter()
            .flatten()
            .collect();

        full_history.extend(chat_history);

        let tool_choice = req
            .tool_choice
            .clone()
            .map(crate::providers::openai::completion::ToolChoice::try_from)
            .transpose()?;

        let tools: Vec<crate::providers::openai::completion::ToolDefinition> = req
            .tools
            .clone()
            .into_iter()
            .map(|tool| {
                let def = crate::providers::openai::completion::ToolDefinition::from(tool);
                if strict_tools { def.with_strict() } else { def }
            })
            .collect();

        let additional_params = if let Some(schema) = req.output_schema {
            let name = schema
                .as_object()
                .and_then(|o| o.get("title"))
                .and_then(|v| v.as_str())
                .unwrap_or("response_schema")
                .to_string();
            let mut schema_value = schema.to_value();
            openai::sanitize_schema(&mut schema_value);
            let response_format = serde_json::json!({
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": name,
                        "strict": true,
                        "schema": schema_value
                    }
                }
            });
            Some(match req.additional_params {
                Some(existing) => json_utils::merge(existing, response_format),
                None => response_format,
            })
        } else {
            req.additional_params
        };

        Ok(Self {
            model,
            messages: full_history,
            temperature: req.temperature,
            max_tokens: req.max_tokens,
            tools,
            tool_choice,
            additional_params,
        })
    }
}

impl TryFrom<(&str, CompletionRequest)> for OpenrouterCompletionRequest {
    type Error = CompletionError;

    fn try_from((model, req): (&str, CompletionRequest)) -> Result<Self, Self::Error> {
        let model = req.model.clone().unwrap_or_else(|| model.to_string());
        OpenrouterCompletionRequest::try_from(OpenRouterRequestParams {
            model: &model,
            request: req,
            strict_tools: false,
        })
    }
}

impl openai::completion::OpenAICompatibleProvider for OpenRouterExt {
    const PROVIDER_NAME: &'static str = self::PROVIDER_NAME;

    type StreamingUsage = Usage;
    type Response = CompletionResponse;

    const STREAM_INCLUDE_USAGE: bool = false;

    fn map_streaming_finish_reason(
        &self,
        finish_reason: Option<&str>,
        native_finish_reason: Option<&str>,
    ) -> Option<crate::completion::FinishReason> {
        if let Some(reason) = finish_reason.filter(|reason| !reason.is_empty()) {
            return Some(map_openai_finish_reason(reason));
        }

        native_finish_reason
            .filter(|reason| !reason.is_empty())
            .map(map_native_finish_reason)
    }

    fn build_completion_request(
        &self,
        model: String,
        request: CompletionRequest,
        options: openai::completion::CompletionModelOptions,
    ) -> Result<openai::completion::CompletionRequest, CompletionError> {
        OpenrouterCompletionRequest::try_from(OpenRouterRequestParams {
            model: &model,
            request,
            strict_tools: options.strict_tools,
        })
    }

    fn finalize_request_body_with_options(
        &self,
        body: &mut serde_json::Value,
        options: openai::completion::CompletionModelOptions,
    ) -> Result<(), CompletionError> {
        finalize_openrouter_request_body(body, options.prompt_caching);
        Ok(())
    }

    /// Encrypted reasoning (`{"type":"reasoning.encrypted"}`) is the turn's own
    /// output, not tool-call metadata: it arrives with `reasoning: null` and an
    /// `rs_*` id of its own, which never matches a `call_*` tool-call id, and it
    /// arrives before any tool call opens. Emitting it as a reasoning block
    /// matches the non-streaming path (which maps the same detail to
    /// [`message::ReasoningContent::Encrypted`]) and is what lets the blob reach
    /// the aggregated choice and be replayed on the next turn.
    fn streaming_detail_reasoning(
        &self,
        detail: &serde_json::Value,
    ) -> Option<(
        crate::streaming::StreamPartId,
        Option<crate::streaming::WireId>,
        message::ReasoningContent,
    )> {
        let Ok(ReasoningDetails::Encrypted { id, data, .. }) =
            serde_json::from_value::<ReasoningDetails>(detail.clone())
        else {
            return None;
        };

        // The durable handle exists only when the wire issued one; an
        // id-less detail keys accumulation by a minted key and replays with
        // the id absent — no fabricated empty "wire" id, and no
        // per-serializer empty-string filter downstream (84a43e9e #4).
        // The mint kind is `EncryptedReasoning`, NOT `Reasoning`: the shared
        // compat adapter accumulates `reasoning`/`reasoning_content` text
        // under `Minted { Reasoning, 0 }`, and a whole block under that same
        // key would restate — i.e. replace — the open text part. Distinct
        // content classes get distinct minted keys.
        let provider_id = id.and_then(crate::streaming::WireId::new);
        let key = provider_id.as_ref().map_or(
            crate::streaming::StreamPartId::minted(
                crate::streaming::MintKind::EncryptedReasoning,
                0,
            ),
            |id| crate::streaming::StreamPartId::wire(id.as_str()),
        );
        Some((key, provider_id, message::ReasoningContent::Encrypted(data)))
    }

    /// Anthropic routes stream the plaintext in `delta.reasoning`, then send
    /// its replay-required signature as a final signature-only
    /// `reasoning.text` detail immediately before the tool call. Feed that
    /// authoritative close into the shared lifecycle so the normalized
    /// reasoning block is signed just like the blocking response.
    fn streaming_reasoning_signature(&self, detail: &serde_json::Value) -> Option<String> {
        let Ok(ReasoningDetails::Text {
            signature: Some(signature),
            ..
        }) = serde_json::from_value::<ReasoningDetails>(detail.clone())
        else {
            return None;
        };
        (!signature.is_empty()).then_some(signature)
    }
}

/// OpenRouter completion model, driven by the shared OpenAI Chat Completions path.
///
/// The provider-native escape hatches come with it:
/// [`raw_completion`](openai::completion::GenericCompletionModel::raw_completion)
/// returns OpenRouter's own [`CompletionResponse`] and
/// [`raw_stream`](openai::completion::GenericCompletionModel::raw_stream) a
/// stream whose terminal record stays provider-native — both over the same
/// single request path as the normalized methods.
pub type CompletionModel<H = crate::http_client::BoxedHttpClient> =
    openai::completion::GenericCompletionModel<OpenRouterExt, H>;

/// Final streaming response, shared with the OpenAI Chat Completions path.
pub type StreamingCompletionResponse =
    openai::completion::streaming::StreamingCompletionResponse<Usage>;

impl<H> openai::completion::GenericCompletionModel<OpenRouterExt, H> {
    /// Enable explicit prompt caching for supported OpenRouter models.
    ///
    /// Adds `cache_control: {"type": "ephemeral"}` to the system-prompt
    /// block so subsequent turns that share the same system prefix can be
    /// billed at the cache-hit rate when the selected model/provider supports
    /// explicit cache breakpoints.
    pub fn with_prompt_caching(mut self) -> Self {
        self.prompt_caching = true;
        self
    }
}

#[cfg(test)]
mod tests;
