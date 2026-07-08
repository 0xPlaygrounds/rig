use super::{
    client::{ApiResponse, Client, Usage},
    streaming::StreamingCompletionResponse,
};
use crate::message::{self, DocumentMediaType, DocumentSourceKind, MimeType};
use crate::telemetry::SpanCombinator;
use crate::{
    OneOrMany,
    completion::{self, CompletionError, CompletionRequest},
    http_client::HttpClientExt,
    json_utils,
    providers::openai,
};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{Instrument, Level, enabled, info_span};

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
        self.order = Some(providers.into_iter().map(|p| p.into()).collect());
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
        self.only = Some(providers.into_iter().map(|p| p.into()).collect());
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
        self.ignore = Some(providers.into_iter().map(|p| p.into()).collect());
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

/// A openrouter completion object.
///
/// For more information, see this link: <https://docs.openrouter.xyz/reference/create_chat_completion_v1_chat_completions_post>
#[derive(Debug, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub system_fingerprint: Option<String>,
    pub usage: Option<Usage>,
}

impl TryFrom<CompletionResponse> for completion::CompletionResponse<CompletionResponse> {
    type Error = CompletionError;

    fn try_from(response: CompletionResponse) -> Result<Self, Self::Error> {
        let choice = response.choices.first().ok_or_else(|| {
            CompletionError::ResponseError("Response contained no choices".to_owned())
        })?;

        let content = match &choice.message {
            Message::Assistant {
                content,
                tool_calls,
                reasoning,
                reasoning_details,
                images,
                ..
            } => {
                let mut content = content
                    .iter()
                    .map(|c| match c {
                        openai::AssistantContent::Text { text, .. } => {
                            completion::AssistantContent::text(text)
                        }
                        openai::AssistantContent::Refusal { refusal } => {
                            completion::AssistantContent::text(refusal)
                        }
                    })
                    .collect::<Vec<_>>();

                content.extend(tool_calls.iter().map(|call| {
                    completion::AssistantContent::tool_call(
                        &call.id,
                        &call.function.name,
                        call.function.arguments.clone(),
                    )
                }));

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
                        content.push(completion::AssistantContent::reasoning(reasoning));
                    }
                } else {
                    for reasoning_id in reasoning_order {
                        let Some(mut blocks) = grouped_reasoning.remove(&reasoning_id) else {
                            continue;
                        };
                        blocks.sort_by_key(|(index, position, _)| (*index, *position));
                        content.push(completion::AssistantContent::Reasoning(
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

                content.extend(images.iter().map(response_image_to_assistant_content));

                Ok(content)
            }
            _ => Err(CompletionError::ResponseError(
                "Response did not contain a valid message or tool call".into(),
            )),
        }?;

        let choice = OneOrMany::many(content).map_err(|_| {
            CompletionError::ResponseError(
                "Response contained no message or tool call (empty)".to_owned(),
            )
        })?;

        let usage = response
            .usage
            .as_ref()
            .map(|usage| {
                let (cached_input, cache_creation) = usage
                    .prompt_tokens_details
                    .as_ref()
                    .map(|d| (d.cached_tokens as u64, d.cache_write_tokens as u64))
                    .unwrap_or((0, 0));
                completion::Usage {
                    input_tokens: usage.prompt_tokens as u64,
                    output_tokens: (usage.total_tokens - usage.prompt_tokens) as u64,
                    total_tokens: usage.total_tokens as u64,
                    cached_input_tokens: cached_input,
                    cache_creation_input_tokens: cache_creation,
                    tool_use_prompt_tokens: 0,
                    reasoning_tokens: 0,
                }
            })
            .unwrap_or_default();

        Ok(completion::CompletionResponse {
            choice,
            usage,
            raw_response: response,
            message_id: None,
        })
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

fn openrouter_response_image_params() -> serde_json::Value {
    serde_json::json!({
        "openrouter": {
            OPENROUTER_RESPONSE_ONLY_KEY: true,
            OPENROUTER_RESPONSE_IMAGE_SOURCE_KEY: OPENROUTER_ASSISTANT_IMAGES_SOURCE,
        }
    })
}

fn response_image_to_assistant_content(image: &ResponseImage) -> completion::AssistantContent {
    let url = &image.image_url.url;
    if let Some((mime, b64)) = parse_data_uri(url) {
        completion::AssistantContent::Image(message::Image {
            data: message::DocumentSourceKind::Base64(b64.to_string()),
            media_type: message::ImageMediaType::from_mime_type(mime),
            detail: None,
            additional_params: Some(openrouter_response_image_params()),
        })
    } else {
        completion::AssistantContent::Image(message::Image {
            data: message::DocumentSourceKind::Url(url.clone()),
            media_type: None,
            detail: None,
            additional_params: Some(openrouter_response_image_params()),
        })
    }
}

fn is_openrouter_response_image(image: &message::Image) -> bool {
    image
        .additional_params
        .as_ref()
        .and_then(|params| params.get("openrouter"))
        .is_some_and(|params| {
            params
                .get(OPENROUTER_RESPONSE_ONLY_KEY)
                .and_then(|value| value.as_bool())
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
                let mime = media_type
                    .as_ref()
                    .map(|m| m.to_mime_type())
                    .unwrap_or("application/pdf");
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
    value: OneOrMany<message::UserContent>,
) -> Result<Vec<Message>, message::MessageError> {
    let (tool_results, other_content): (Vec<_>, Vec<_>) = value
        .into_iter()
        .partition(|content| matches!(content, message::UserContent::ToolResult(_)));

    // If there are messages with both tool results and user content, we handle
    // tool results first. It's unlikely that there will be both.
    if !tool_results.is_empty() {
        tool_results
            .into_iter()
            .map(|content| match content {
                message::UserContent::ToolResult(tool_result) => Ok(Message::ToolResult {
                    // Prefer the provider-issued call id, matching the
                    // assistant echo (shared From<message::ToolCall>).
                    tool_call_id: tool_result.call_id.unwrap_or(tool_result.id),
                    content: openai::completion::ToolResultContentValue::String(
                        tool_result
                            .content
                            .into_iter()
                            .map(|c| match c {
                                message::ToolResultContent::Text(message::Text {
                                    text, ..
                                }) => text,
                                message::ToolResultContent::Image(_) => {
                                    "[Image content not supported in tool results]".to_string()
                                }
                            })
                            .collect::<Vec<_>>()
                            .join("\n"),
                    ),
                }),
                _ => Err(message::MessageError::ConversionError(
                    "expected tool result content while converting OpenRouter input".into(),
                )),
            })
            .collect::<Result<Vec<_>, _>>()
    } else {
        let user_content: Vec<UserContent> = other_content
            .into_iter()
            .map(user_content_to_openai)
            .collect::<Result<Vec<_>, _>>()?;

        let content = OneOrMany::many(user_content).map_err(|_| {
            message::MessageError::ConversionError(
                "OpenRouter user message did not contain any non-tool content".into(),
            )
        })?;

        Ok(vec![Message::User {
            content,
            name: None,
        }])
    }
}

// ================================================================
// Response Types
// ================================================================

#[derive(Debug, Deserialize, Serialize)]
pub struct Choice {
    pub index: usize,
    pub native_finish_reason: Option<String>,
    pub message: Message,
    pub finish_reason: Option<String>,
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
    value: OneOrMany<message::AssistantContent>,
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
                            let id = id.or_else(|| tool_call.call_id.clone());
                            if let Some(signature) = &tool_call.signature
                                && let Some(id) = id
                            {
                                reasoning_details.push(ReasoningDetails::Encrypted {
                                    id: Some(id),
                                    format,
                                    index: None,
                                    data: signature.clone(),
                                })
                            }
                        }
                    }
                } else if let Some(signature) = &tool_call.signature {
                    reasoning_details.push(ReasoningDetails::Encrypted {
                        id: tool_call.call_id.clone(),
                        format: None,
                        index: None,
                        data: signature.clone(),
                    });
                }
                tool_calls.push(tool_call.into())
            }
            message::AssistantContent::Reasoning(r) => {
                if r.content.is_empty() {
                    let display = r.display_text();
                    if !display.is_empty() {
                        reasoning = Some(display);
                    }
                } else {
                    for reasoning_block in &r.content {
                        let index = Some(reasoning_details.len());
                        match reasoning_block {
                            message::ReasoningContent::Text { text, signature } => {
                                reasoning_details.push(ReasoningDetails::Text {
                                    id: r.id.clone(),
                                    format: None,
                                    index,
                                    text: Some(text.clone()),
                                    signature: signature.clone(),
                                });
                            }
                            message::ReasoningContent::Summary(summary) => {
                                reasoning_details.push(ReasoningDetails::Summary {
                                    id: r.id.clone(),
                                    format: None,
                                    index,
                                    summary: summary.clone(),
                                });
                            }
                            message::ReasoningContent::Encrypted(data)
                            | message::ReasoningContent::Redacted { data } => {
                                reasoning_details.push(ReasoningDetails::Encrypted {
                                    id: r.id.clone(),
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

pub(super) fn final_request_body(
    request: &OpenrouterCompletionRequest,
    prompt_caching: bool,
) -> Result<serde_json::Value, CompletionError> {
    let mut body = serde_json::to_value(request)?;
    if prompt_caching {
        apply_prompt_caching(&mut body);
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

    Ok(body)
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct OpenrouterCompletionRequest {
    model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<crate::providers::openai::completion::ToolDefinition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<crate::providers::openai::completion::ToolChoice>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub additional_params: Option<serde_json::Value>,
}

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

        let mut full_history: Vec<Message> = match &req.preamble {
            Some(preamble) => vec![Message::system(preamble)],
            None => vec![],
        };

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

#[derive(Clone)]
pub struct CompletionModel<T = reqwest::Client> {
    pub(crate) client: Client<T>,
    pub model: String,
    /// Enable strict mode for tool schemas.
    /// When enabled, tool schemas are sanitized to meet OpenAI's strict mode requirements.
    pub strict_tools: bool,
    /// Enable explicit prompt caching via OpenRouter.
    ///
    /// When true, the outgoing JSON body is post-processed to attach
    /// `cache_control: {"type": "ephemeral"}` to the system prompt. This is
    /// intended for models and providers that support explicit cache
    /// breakpoints.
    pub prompt_caching: bool,
}

impl<T> CompletionModel<T> {
    pub fn new(client: Client<T>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
            strict_tools: false,
            prompt_caching: false,
        }
    }

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

    /// Enable strict mode for tool schemas.
    ///
    /// When enabled, tool schemas are automatically sanitized to meet OpenAI's strict mode requirements:
    /// - `additionalProperties: false` is added to all objects
    /// - All properties are marked as required
    /// - `strict: true` is set on each function definition
    ///
    /// Note: Not all models on OpenRouter support strict mode. This works best with OpenAI models.
    pub fn with_strict_tools(mut self) -> Self {
        self.strict_tools = true;
        self
    }
}

impl<T> completion::CompletionModel for CompletionModel<T>
where
    T: HttpClientExt + Clone + std::fmt::Debug + Default + 'static,
{
    type Response = CompletionResponse;
    type StreamingResponse = StreamingCompletionResponse;

    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model)
    }

    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
        let request_model = completion_request
            .model
            .clone()
            .unwrap_or_else(|| self.model.clone());
        let preamble = completion_request.preamble.clone();
        let request = OpenrouterCompletionRequest::try_from(OpenRouterRequestParams {
            model: request_model.as_ref(),
            request: completion_request,
            strict_tools: self.strict_tools,
        })?;

        let body = final_request_body(&request, self.prompt_caching)?;

        if enabled!(Level::TRACE) {
            tracing::trace!(
                target: "rig::completions",
                "OpenRouter completion request: {}",
                serde_json::to_string_pretty(&body)?
            );
        }

        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat",
                gen_ai.operation.name = "chat",
                gen_ai.provider.name = "openrouter",
                gen_ai.request.model = &request_model,
                gen_ai.system_instructions = preamble,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.usage.cache_read.input_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        let body = serde_json::to_vec(&body)?;

        let req = self
            .client
            .post("/chat/completions")?
            .body(body)
            .map_err(|x| CompletionError::HttpError(x.into()))?;

        async move {
            let response = self.client.send::<_, Bytes>(req).await?;
            let status = response.status();
            let response_body = response.into_body().into_future().await?.to_vec();

            if status.is_success() {
                let parsed: ApiResponse<CompletionResponse> =
                    serde_json::from_slice(&response_body).map_err(|e| {
                        CompletionError::ResponseError(format!(
                            "Failed to parse OpenRouter completion response: {}, response body: {}",
                            e,
                            String::from_utf8_lossy(&response_body)
                        ))
                    })?;
                match parsed {
                    ApiResponse::Ok(response) => {
                        let span = tracing::Span::current();
                        span.record_token_usage(&response.usage);
                        span.record("gen_ai.response.id", &response.id);
                        span.record("gen_ai.response.model", &response.model);

                        tracing::debug!(target: "rig::completions",
                            "OpenRouter response: {response:?}");
                        response.try_into()
                    }
                    ApiResponse::Err(err) => {
                        tracing::warn!(message = %err.message, "provider returned an error response");
                        Err(CompletionError::from_http_response(
                            status,
                            String::from_utf8_lossy(&response_body).into_owned(),
                        ))
                    }
                }
            } else {
                Err(CompletionError::from_http_response(
                    status,
                    String::from_utf8_lossy(&response_body).to_string(),
                ))
            }
        }
        .instrument(span)
        .await
    }

    async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<
        crate::streaming::StreamingCompletionResponse<Self::StreamingResponse>,
        CompletionError,
    > {
        CompletionModel::stream(self, completion_request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{AudioMediaType, ImageDetail, VideoMediaType};
    use serde_json::json;

    #[test]
    fn test_openrouter_request_uses_request_model_override() {
        let request = CompletionRequest {
            model: Some("google/gemini-2.5-flash".to_string()),
            preamble: None,
            chat_history: crate::OneOrMany::one("Hello".into()),
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
        };

        let openrouter_request =
            OpenrouterCompletionRequest::try_from(("openai/gpt-4o-mini", request))
                .expect("request conversion should succeed");
        let serialized =
            serde_json::to_value(openrouter_request).expect("serialization should succeed");

        assert_eq!(serialized["model"], "google/gemini-2.5-flash");
    }

    #[test]
    fn openrouter_params_include_direct_request_documents() {
        let request = CompletionRequest {
            model: None,
            preamble: None,
            chat_history: crate::OneOrMany::one(crate::message::Message::user(
                "What is glarb-glarb?",
            )),
            documents: vec![crate::completion::request::Document {
                id: "doc_1".to_string(),
                text: "Definition of glarb-glarb: an ancient tool.".to_string(),
                additional_props: Default::default(),
            }],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
        };

        let request = OpenrouterCompletionRequest::try_from(OpenRouterRequestParams {
            model: "openai/gpt-4o-mini",
            request,
            strict_tools: false,
        })
        .expect("request conversion should succeed");
        let serialized = serde_json::to_value(request).expect("serialization should succeed");

        assert!(
            serialized["messages"].to_string().contains("glarb-glarb"),
            "direct request documents should be normalized through public params"
        );
    }

    #[test]
    fn test_openrouter_request_uses_default_model_when_override_unset() {
        let request = CompletionRequest {
            model: None,
            preamble: None,
            chat_history: crate::OneOrMany::one("Hello".into()),
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
        };

        let openrouter_request =
            OpenrouterCompletionRequest::try_from(("openai/gpt-4o-mini", request))
                .expect("request conversion should succeed");
        let serialized =
            serde_json::to_value(openrouter_request).expect("serialization should succeed");

        assert_eq!(serialized["model"], "openai/gpt-4o-mini");
    }

    #[test]
    fn final_request_body_serializes_assistant_reasoning_under_openrouter_key() {
        // Reasoning replay normally flows through `reasoning_details`; the
        // plain string field must nevertheless hit the wire under
        // OpenRouter's `reasoning` key, not the shared `reasoning_content`.
        let request = OpenrouterCompletionRequest {
            model: "openai/gpt-4o".to_string(),
            messages: vec![Message::Assistant {
                content: vec![],
                reasoning: Some("thinking it through".to_string()),
                refusal: None,
                audio: None,
                name: None,
                tool_calls: vec![],
                reasoning_details: vec![],
                images: vec![],
            }],
            temperature: None,
            tools: vec![],
            tool_choice: None,
            additional_params: None,
        };

        let body = final_request_body(&request, false).expect("body should serialize");

        assert_eq!(
            body["messages"][0]["reasoning"],
            serde_json::json!("thinking it through")
        );
        assert!(
            body["messages"][0].get("reasoning_content").is_none(),
            "OpenRouter's assistant reasoning key is `reasoning`, not `reasoning_content`"
        );
    }

    #[test]
    fn test_openrouter_request_maps_output_schema_to_response_format() {
        let schema: schemars::Schema = serde_json::from_value(json!({
            "title": "WeatherResponse",
            "type": "object",
            "properties": {
                "city": { "type": "string" },
                "weather": { "type": "string" }
            }
        }))
        .expect("schema should deserialize");

        let request = CompletionRequest {
            model: None,
            preamble: None,
            chat_history: crate::OneOrMany::one("Hello".into()),
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: Some(schema),
        };

        let openrouter_request =
            OpenrouterCompletionRequest::try_from(("openai/gpt-4o-mini", request))
                .expect("request conversion should succeed");
        let serialized =
            serde_json::to_value(openrouter_request).expect("serialization should succeed");

        assert_eq!(
            serialized["response_format"],
            json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "WeatherResponse",
                    "strict": true,
                    "schema": {
                        "title": "WeatherResponse",
                        "type": "object",
                        "properties": {
                            "city": { "type": "string" },
                            "weather": { "type": "string" }
                        },
                        "additionalProperties": false,
                        "required": ["city", "weather"]
                    }
                }
            })
        );
    }

    #[test]
    fn test_openrouter_request_merges_output_schema_with_provider_preferences() {
        let schema: schemars::Schema = serde_json::from_value(json!({
            "type": "object",
            "properties": {
                "answer": { "type": "string" }
            }
        }))
        .expect("schema should deserialize");

        let request = CompletionRequest {
            model: None,
            preamble: None,
            chat_history: crate::OneOrMany::one("Hello".into()),
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: Some(
                ProviderPreferences::new()
                    .require_parameters(true)
                    .to_json(),
            ),
            output_schema: Some(schema),
        };

        let openrouter_request =
            OpenrouterCompletionRequest::try_from(("openai/gpt-4o-mini", request))
                .expect("request conversion should succeed");
        let serialized =
            serde_json::to_value(openrouter_request).expect("serialization should succeed");

        assert_eq!(serialized["provider"]["require_parameters"], true);
        assert_eq!(serialized["response_format"]["type"], "json_schema");
        assert_eq!(
            serialized["response_format"]["json_schema"]["name"],
            "response_schema"
        );
        assert_eq!(
            serialized["response_format"]["json_schema"]["schema"]["additionalProperties"],
            false
        );
    }

    #[test]
    fn test_completion_response_deserialization_gemini_flash() {
        // Real response from OpenRouter with google/gemini-2.5-flash
        let json = json!({
            "id": "gen-AAAAAAAAAA-AAAAAAAAAAAAAAAAAAAA",
            "provider": "Google",
            "model": "google/gemini-2.5-flash",
            "object": "chat.completion",
            "created": 1765971703u64,
            "choices": [{
                "logprobs": null,
                "finish_reason": "stop",
                "native_finish_reason": "STOP",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "CONTENT",
                    "refusal": null,
                    "reasoning": null
                }
            }],
            "usage": {
                "prompt_tokens": 669,
                "completion_tokens": 5,
                "total_tokens": 674
            }
        });

        let response: CompletionResponse = serde_json::from_value(json).unwrap();
        assert_eq!(response.id, "gen-AAAAAAAAAA-AAAAAAAAAAAAAAAAAAAA");
        assert_eq!(response.model, "google/gemini-2.5-flash");
        assert_eq!(response.choices.len(), 1);
        assert_eq!(response.choices[0].finish_reason, Some("stop".to_string()));
    }

    #[test]
    fn test_completion_response_maps_cache_token_accounting() {
        let json = json!({
            "id": "gen-cache-test",
            "object": "chat.completion",
            "created": 1,
            "model": "anthropic/claude-3.5-sonnet",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "Hi"
                }
            }],
            "usage": {
                "prompt_tokens": 500,
                "completion_tokens": 10,
                "total_tokens": 510,
                "prompt_tokens_details": {
                    "cached_tokens": 400,
                    "cache_write_tokens": 50
                }
            }
        });

        let response: CompletionResponse = serde_json::from_value(json).unwrap();
        let converted: completion::CompletionResponse<CompletionResponse> =
            response.try_into().unwrap();

        assert_eq!(converted.usage.input_tokens, 500);
        assert_eq!(converted.usage.output_tokens, 10);
        assert_eq!(converted.usage.cached_input_tokens, 400);
        assert_eq!(converted.usage.cache_creation_input_tokens, 50);
    }

    #[test]
    fn test_completion_response_cache_tokens_absent_defaults_to_zero() {
        let json = json!({
            "id": "gen-no-cache",
            "object": "chat.completion",
            "created": 1,
            "model": "openai/gpt-4o",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "Hi"
                }
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "total_tokens": 110
            }
        });

        let response: CompletionResponse = serde_json::from_value(json).unwrap();
        let converted: completion::CompletionResponse<CompletionResponse> =
            response.try_into().unwrap();

        assert_eq!(converted.usage.cached_input_tokens, 0);
        assert_eq!(converted.usage.cache_creation_input_tokens, 0);
    }

    #[test]
    fn test_completion_response_deserialization_gemini_model_role() {
        let json = json!({
            "id": "gen-BBBBBBBBBB-BBBBBBBBBBBBBBBBBBBB",
            "provider": "Google",
            "model": "google/gemini-2.5-pro-exp-03-25:free",
            "object": "chat.completion",
            "created": 1743780565u64,
            "choices": [{
                "logprobs": null,
                "finish_reason": "stop",
                "native_finish_reason": "STOP",
                "index": 0,
                "message": {
                    "role": "model",
                    "content": "CONTENT",
                    "refusal": null,
                    "reasoning": null
                }
            }],
            "usage": {
                "prompt_tokens": 669,
                "completion_tokens": 5,
                "total_tokens": 674
            }
        });

        let response: CompletionResponse = serde_json::from_value(json).unwrap();
        let converted: completion::CompletionResponse<CompletionResponse> =
            response.try_into().unwrap();

        assert_eq!(
            converted.raw_response.model,
            "google/gemini-2.5-pro-exp-03-25:free"
        );
        assert!(matches!(
            converted.choice.first(),
            completion::AssistantContent::Text(text) if text.text == "CONTENT"
        ));
    }

    #[test]
    fn test_message_assistant_without_reasoning_details() {
        // Verify that missing reasoning_details field doesn't cause deserialization failure
        let json = json!({
            "role": "assistant",
            "content": "Hello world",
            "refusal": null,
            "reasoning": null
        });

        let message: Message = serde_json::from_value(json).unwrap();
        match message {
            Message::Assistant {
                content,
                reasoning_details,
                ..
            } => {
                assert_eq!(content.len(), 1);
                assert!(reasoning_details.is_empty());
            }
            _ => panic!("Expected Assistant message"),
        }
    }

    #[test]
    fn test_data_collection_serialization() {
        assert_eq!(
            serde_json::to_string(&DataCollection::Allow).unwrap(),
            r#""allow""#
        );
        assert_eq!(
            serde_json::to_string(&DataCollection::Deny).unwrap(),
            r#""deny""#
        );
    }

    #[test]
    fn test_data_collection_default() {
        assert_eq!(DataCollection::default(), DataCollection::Allow);
    }

    #[test]
    fn test_quantization_serialization() {
        assert_eq!(
            serde_json::to_string(&Quantization::Int4).unwrap(),
            r#""int4""#
        );
        assert_eq!(
            serde_json::to_string(&Quantization::Int8).unwrap(),
            r#""int8""#
        );
        assert_eq!(
            serde_json::to_string(&Quantization::Fp16).unwrap(),
            r#""fp16""#
        );
        assert_eq!(
            serde_json::to_string(&Quantization::Bf16).unwrap(),
            r#""bf16""#
        );
        assert_eq!(
            serde_json::to_string(&Quantization::Fp32).unwrap(),
            r#""fp32""#
        );
        assert_eq!(
            serde_json::to_string(&Quantization::Fp8).unwrap(),
            r#""fp8""#
        );
        assert_eq!(
            serde_json::to_string(&Quantization::Unknown).unwrap(),
            r#""unknown""#
        );
    }

    #[test]
    fn test_provider_sort_strategy_serialization() {
        assert_eq!(
            serde_json::to_string(&ProviderSortStrategy::Price).unwrap(),
            r#""price""#
        );
        assert_eq!(
            serde_json::to_string(&ProviderSortStrategy::Throughput).unwrap(),
            r#""throughput""#
        );
        assert_eq!(
            serde_json::to_string(&ProviderSortStrategy::Latency).unwrap(),
            r#""latency""#
        );
    }

    #[test]
    fn test_sort_partition_serialization() {
        assert_eq!(
            serde_json::to_string(&SortPartition::Model).unwrap(),
            r#""model""#
        );
        assert_eq!(
            serde_json::to_string(&SortPartition::None).unwrap(),
            r#""none""#
        );
    }

    #[test]
    fn test_provider_sort_simple() {
        let sort = ProviderSort::Simple(ProviderSortStrategy::Latency);
        let json = serde_json::to_value(&sort).unwrap();
        assert_eq!(json, "latency");
    }

    #[test]
    fn test_provider_sort_complex() {
        let sort = ProviderSort::Complex(
            ProviderSortConfig::new(ProviderSortStrategy::Price).partition(SortPartition::None),
        );
        let json = serde_json::to_value(&sort).unwrap();
        assert_eq!(json["by"], "price");
        assert_eq!(json["partition"], "none");
    }

    #[test]
    fn test_provider_sort_complex_without_partition() {
        let sort = ProviderSort::Complex(ProviderSortConfig::new(ProviderSortStrategy::Throughput));
        let json = serde_json::to_value(&sort).unwrap();
        assert_eq!(json["by"], "throughput");
        assert!(json.get("partition").is_none());
    }

    #[test]
    fn test_provider_sort_from_strategy() {
        let sort: ProviderSort = ProviderSortStrategy::Price.into();
        assert_eq!(sort, ProviderSort::Simple(ProviderSortStrategy::Price));
    }

    #[test]
    fn test_provider_sort_from_config() {
        let config = ProviderSortConfig::new(ProviderSortStrategy::Latency);
        let sort: ProviderSort = config.into();
        match sort {
            ProviderSort::Complex(c) => assert_eq!(c.by, ProviderSortStrategy::Latency),
            _ => panic!("Expected Complex variant"),
        }
    }

    #[test]
    fn test_percentile_thresholds_builder() {
        let thresholds = PercentileThresholds::new()
            .p50(10.0)
            .p75(25.0)
            .p90(50.0)
            .p99(100.0);

        assert_eq!(thresholds.p50, Some(10.0));
        assert_eq!(thresholds.p75, Some(25.0));
        assert_eq!(thresholds.p90, Some(50.0));
        assert_eq!(thresholds.p99, Some(100.0));
    }

    #[test]
    fn test_percentile_thresholds_default() {
        let thresholds = PercentileThresholds::default();
        assert_eq!(thresholds.p50, None);
        assert_eq!(thresholds.p75, None);
        assert_eq!(thresholds.p90, None);
        assert_eq!(thresholds.p99, None);
    }

    #[test]
    fn test_throughput_threshold_simple() {
        let threshold = ThroughputThreshold::Simple(50.0);
        let json = serde_json::to_value(&threshold).unwrap();
        assert_eq!(json, 50.0);
    }

    #[test]
    fn test_throughput_threshold_percentile() {
        let threshold = ThroughputThreshold::Percentile(PercentileThresholds::new().p90(50.0));
        let json = serde_json::to_value(&threshold).unwrap();
        assert_eq!(json["p90"], 50.0);
    }

    #[test]
    fn test_latency_threshold_simple() {
        let threshold = LatencyThreshold::Simple(0.5);
        let json = serde_json::to_value(&threshold).unwrap();
        assert_eq!(json, 0.5);
    }

    #[test]
    fn test_latency_threshold_percentile() {
        let threshold = LatencyThreshold::Percentile(PercentileThresholds::new().p50(0.1).p99(1.0));
        let json = serde_json::to_value(&threshold).unwrap();
        assert_eq!(json["p50"], 0.1);
        assert_eq!(json["p99"], 1.0);
    }

    #[test]
    fn test_max_price_builder() {
        let price = MaxPrice::new().prompt(0.001).completion(0.002);

        assert_eq!(price.prompt, Some(0.001));
        assert_eq!(price.completion, Some(0.002));
        assert_eq!(price.request, None);
        assert_eq!(price.image, None);
    }

    #[test]
    fn test_max_price_all_fields() {
        let price = MaxPrice::new()
            .prompt(0.001)
            .completion(0.002)
            .request(0.01)
            .image(0.05);

        let json = serde_json::to_value(&price).unwrap();
        assert_eq!(json["prompt"], 0.001);
        assert_eq!(json["completion"], 0.002);
        assert_eq!(json["request"], 0.01);
        assert_eq!(json["image"], 0.05);
    }

    #[test]
    fn test_max_price_default() {
        let price = MaxPrice::default();
        assert_eq!(price.prompt, None);
        assert_eq!(price.completion, None);
        assert_eq!(price.request, None);
        assert_eq!(price.image, None);
    }

    #[test]
    fn test_provider_preferences_default() {
        let prefs = ProviderPreferences::default();
        assert!(prefs.order.is_none());
        assert!(prefs.only.is_none());
        assert!(prefs.ignore.is_none());
        assert!(prefs.allow_fallbacks.is_none());
        assert!(prefs.require_parameters.is_none());
        assert!(prefs.data_collection.is_none());
        assert!(prefs.zdr.is_none());
        assert!(prefs.sort.is_none());
        assert!(prefs.preferred_min_throughput.is_none());
        assert!(prefs.preferred_max_latency.is_none());
        assert!(prefs.max_price.is_none());
        assert!(prefs.quantizations.is_none());
    }

    #[test]
    fn test_provider_preferences_order_with_fallbacks() {
        let prefs = ProviderPreferences::new()
            .order(["anthropic", "openai"])
            .allow_fallbacks(true);

        let json = prefs.to_json();
        let provider = &json["provider"];

        assert_eq!(provider["order"], json!(["anthropic", "openai"]));
        assert_eq!(provider["allow_fallbacks"], true);
    }

    #[test]
    fn test_provider_preferences_only_allowlist() {
        let prefs = ProviderPreferences::new()
            .only(["azure", "together"])
            .allow_fallbacks(false);

        let json = prefs.to_json();
        let provider = &json["provider"];

        assert_eq!(provider["only"], json!(["azure", "together"]));
        assert_eq!(provider["allow_fallbacks"], false);
    }

    #[test]
    fn test_provider_preferences_ignore() {
        let prefs = ProviderPreferences::new().ignore(["deepinfra"]);

        let json = prefs.to_json();
        let provider = &json["provider"];

        assert_eq!(provider["ignore"], json!(["deepinfra"]));
    }

    #[test]
    fn test_provider_preferences_sort_latency() {
        let prefs = ProviderPreferences::new().sort(ProviderSortStrategy::Latency);

        let json = prefs.to_json();
        let provider = &json["provider"];

        assert_eq!(provider["sort"], "latency");
    }

    #[test]
    fn test_provider_preferences_price_with_throughput() {
        let prefs = ProviderPreferences::new()
            .sort(ProviderSortStrategy::Price)
            .preferred_min_throughput(ThroughputThreshold::Percentile(
                PercentileThresholds::new().p90(50.0),
            ));

        let json = prefs.to_json();
        let provider = &json["provider"];

        assert_eq!(provider["sort"], "price");
        assert_eq!(provider["preferred_min_throughput"]["p90"], 50.0);
    }

    #[test]
    fn test_provider_preferences_require_parameters() {
        let prefs = ProviderPreferences::new().require_parameters(true);

        let json = prefs.to_json();
        let provider = &json["provider"];

        assert_eq!(provider["require_parameters"], true);
    }

    #[test]
    fn test_provider_preferences_data_policy_and_zdr() {
        let prefs = ProviderPreferences::new()
            .data_collection(DataCollection::Deny)
            .zdr(true);

        let json = prefs.to_json();
        let provider = &json["provider"];

        assert_eq!(provider["data_collection"], "deny");
        assert_eq!(provider["zdr"], true);
    }

    #[test]
    fn test_provider_preferences_quantizations() {
        let prefs =
            ProviderPreferences::new().quantizations([Quantization::Int8, Quantization::Fp16]);

        let json = prefs.to_json();
        let provider = &json["provider"];

        assert_eq!(provider["quantizations"], json!(["int8", "fp16"]));
    }

    #[test]
    fn test_provider_preferences_convenience_methods() {
        let prefs = ProviderPreferences::new().zero_data_retention().fastest();

        assert_eq!(prefs.zdr, Some(true));
        assert_eq!(
            prefs.sort,
            Some(ProviderSort::Simple(ProviderSortStrategy::Throughput))
        );

        let prefs2 = ProviderPreferences::new().cheapest();
        assert_eq!(
            prefs2.sort,
            Some(ProviderSort::Simple(ProviderSortStrategy::Price))
        );

        let prefs3 = ProviderPreferences::new().lowest_latency();
        assert_eq!(
            prefs3.sort,
            Some(ProviderSort::Simple(ProviderSortStrategy::Latency))
        );
    }

    #[test]
    fn test_provider_preferences_serialization_skips_none() {
        let prefs = ProviderPreferences::new().sort(ProviderSortStrategy::Price);

        let json = serde_json::to_value(&prefs).unwrap();

        assert_eq!(json["sort"], "price");
        assert!(json.get("order").is_none());
        assert!(json.get("only").is_none());
        assert!(json.get("ignore").is_none());
        assert!(json.get("zdr").is_none());
    }

    #[test]
    fn test_provider_preferences_deserialization() {
        let json = json!({
            "order": ["anthropic", "openai"],
            "sort": "throughput",
            "data_collection": "deny",
            "zdr": true,
            "quantizations": ["int8", "fp16"]
        });

        let prefs: ProviderPreferences = serde_json::from_value(json).unwrap();

        assert_eq!(
            prefs.order,
            Some(vec!["anthropic".to_string(), "openai".to_string()])
        );
        assert_eq!(
            prefs.sort,
            Some(ProviderSort::Simple(ProviderSortStrategy::Throughput))
        );
        assert_eq!(prefs.data_collection, Some(DataCollection::Deny));
        assert_eq!(prefs.zdr, Some(true));
        assert_eq!(
            prefs.quantizations,
            Some(vec![Quantization::Int8, Quantization::Fp16])
        );
    }

    #[test]
    fn test_provider_preferences_deserialization_complex_sort() {
        let json = json!({
            "sort": {
                "by": "latency",
                "partition": "model"
            }
        });

        let prefs: ProviderPreferences = serde_json::from_value(json).unwrap();

        match prefs.sort {
            Some(ProviderSort::Complex(config)) => {
                assert_eq!(config.by, ProviderSortStrategy::Latency);
                assert_eq!(config.partition, Some(SortPartition::Model));
            }
            _ => panic!("Expected Complex sort variant"),
        }
    }

    #[test]
    fn test_provider_preferences_full_integration() {
        let prefs = ProviderPreferences::new()
            .order(["anthropic", "openai"])
            .only(["anthropic", "openai", "google"])
            .sort(ProviderSortStrategy::Throughput)
            .data_collection(DataCollection::Deny)
            .zdr(true)
            .quantizations([Quantization::Int8])
            .allow_fallbacks(false);

        let json = prefs.to_json();

        assert!(json.get("provider").is_some());
        let provider = &json["provider"];
        assert_eq!(provider["order"], json!(["anthropic", "openai"]));
        assert_eq!(provider["only"], json!(["anthropic", "openai", "google"]));
        assert_eq!(provider["sort"], "throughput");
        assert_eq!(provider["data_collection"], "deny");
        assert_eq!(provider["zdr"], true);
        assert_eq!(provider["quantizations"], json!(["int8"]));
        assert_eq!(provider["allow_fallbacks"], false);
    }

    #[test]
    fn test_provider_preferences_max_price() {
        let prefs =
            ProviderPreferences::new().max_price(MaxPrice::new().prompt(0.001).completion(0.002));

        let json = prefs.to_json();
        let provider = &json["provider"];

        assert_eq!(provider["max_price"]["prompt"], 0.001);
        assert_eq!(provider["max_price"]["completion"], 0.002);
    }

    #[test]
    fn test_provider_preferences_preferred_max_latency() {
        let prefs = ProviderPreferences::new().preferred_max_latency(LatencyThreshold::Simple(0.5));

        let json = prefs.to_json();
        let provider = &json["provider"];

        assert_eq!(provider["preferred_max_latency"], 0.5);
    }

    #[test]
    fn test_provider_preferences_empty_arrays() {
        let prefs = ProviderPreferences::new()
            .order(Vec::<String>::new())
            .quantizations(Vec::<Quantization>::new());

        let json = prefs.to_json();
        let provider = &json["provider"];

        assert_eq!(provider["order"], json!([]));
        assert_eq!(provider["quantizations"], json!([]));
    }

    // ================================================================
    // File Support Tests
    // ================================================================

    #[test]
    fn test_user_content_text_serialization() {
        let content = UserContent::Text {
            text: "Hello, world!".to_string(),
        };
        let json = serde_json::to_value(&content).unwrap();

        assert_eq!(json["type"], "text");
        assert_eq!(json["text"], "Hello, world!");
    }

    #[test]
    fn test_user_content_image_url_serialization() {
        let content = UserContent::Image {
            image_url: ImageUrl {
                url: "https://example.com/image.png".to_string(),
                detail: None,
            },
        };
        let json = serde_json::to_value(&content).unwrap();

        assert_eq!(json["type"], "image_url");
        assert_eq!(json["image_url"]["url"], "https://example.com/image.png");
        assert!(json["image_url"].get("detail").is_none());
    }

    #[test]
    fn test_user_content_image_url_with_detail_serialization() {
        let content = UserContent::Image {
            image_url: ImageUrl {
                url: "https://example.com/image.png".to_string(),
                detail: Some(ImageDetail::High),
            },
        };
        let json = serde_json::to_value(&content).unwrap();

        assert_eq!(json["type"], "image_url");
        assert_eq!(json["image_url"]["url"], "https://example.com/image.png");
        assert_eq!(json["image_url"]["detail"], "high");
    }

    #[test]
    fn test_user_content_image_base64_serialization() {
        let content = UserContent::Image {
            image_url: ImageUrl {
                url: "data:image/png;base64,SGVsbG8=".to_string(),
                detail: Some(ImageDetail::Low),
            },
        };
        let json = serde_json::to_value(&content).unwrap();

        assert_eq!(json["type"], "image_url");
        assert_eq!(json["image_url"]["url"], "data:image/png;base64,SGVsbG8=");
        assert_eq!(json["image_url"]["detail"], "low");
    }

    #[test]
    fn test_user_content_file_url_serialization() {
        let content = UserContent::File {
            file: FileData {
                file_data: Some("https://example.com/doc.pdf".to_string()),
                file_id: None,
                filename: Some("document.pdf".to_string()),
            },
        };
        let json = serde_json::to_value(&content).unwrap();

        assert_eq!(json["type"], "file");
        assert_eq!(json["file"]["file_data"], "https://example.com/doc.pdf");
        assert_eq!(json["file"]["filename"], "document.pdf");
    }

    #[test]
    fn test_user_content_file_base64_serialization() {
        let content = UserContent::File {
            file: FileData {
                file_data: Some("data:application/pdf;base64,JVBERi0xLjQ=".to_string()),
                file_id: None,
                filename: Some("report.pdf".to_string()),
            },
        };
        let json = serde_json::to_value(&content).unwrap();

        assert_eq!(json["type"], "file");
        assert_eq!(
            json["file"]["file_data"],
            "data:application/pdf;base64,JVBERi0xLjQ="
        );
        assert_eq!(json["file"]["filename"], "report.pdf");
    }

    #[test]
    fn test_user_content_text_deserialization() {
        let json = json!({
            "type": "text",
            "text": "Hello!"
        });

        let content: UserContent = serde_json::from_value(json).unwrap();
        assert_eq!(
            content,
            UserContent::Text {
                text: "Hello!".to_string()
            }
        );
    }

    #[test]
    fn test_user_content_image_url_deserialization() {
        let json = json!({
            "type": "image_url",
            "image_url": {
                "url": "https://example.com/img.jpg",
                "detail": "high"
            }
        });

        let content: UserContent = serde_json::from_value(json).unwrap();
        match content {
            UserContent::Image { image_url } => {
                assert_eq!(image_url.url, "https://example.com/img.jpg");
                assert_eq!(image_url.detail, Some(ImageDetail::High));
            }
            _ => panic!("Expected Image variant"),
        }
    }

    #[test]
    fn test_user_content_file_deserialization() {
        let json = json!({
            "type": "file",
            "file": {
                "filename": "doc.pdf",
                "file_data": "https://example.com/doc.pdf"
            }
        });

        let content: UserContent = serde_json::from_value(json).unwrap();
        match content {
            UserContent::File { file } => {
                assert_eq!(file.filename, Some("doc.pdf".to_string()));
                assert_eq!(
                    file.file_data,
                    Some("https://example.com/doc.pdf".to_string())
                );
            }
            _ => panic!("Expected File variant"),
        }
    }

    #[test]
    fn test_message_user_with_text_serialization() {
        let message = Message::User {
            content: OneOrMany::one(UserContent::Text {
                text: "Hello".to_string(),
            }),
            name: None,
        };
        let json = serde_json::to_value(&message).unwrap();

        // Single text content should be serialized as a plain string
        assert_eq!(json["role"], "user");
        assert_eq!(json["content"], "Hello");
    }

    #[test]
    fn test_message_user_with_mixed_content_serialization() {
        let message = Message::User {
            content: OneOrMany::many(vec![
                UserContent::Text {
                    text: "Check this image:".to_string(),
                },
                UserContent::Image {
                    image_url: ImageUrl {
                        url: "https://example.com/img.png".to_string(),
                        detail: None,
                    },
                },
            ])
            .unwrap(),
            name: None,
        };
        let json = serde_json::to_value(&message).unwrap();

        assert_eq!(json["role"], "user");
        let content = json["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[1]["type"], "image_url");
    }

    #[test]
    fn test_message_user_with_file_serialization() {
        let message = Message::User {
            content: OneOrMany::many(vec![
                UserContent::Text {
                    text: "Analyze this PDF:".to_string(),
                },
                UserContent::File {
                    file: FileData {
                        file_data: Some("https://example.com/doc.pdf".to_string()),
                        file_id: None,
                        filename: Some("document.pdf".to_string()),
                    },
                },
            ])
            .unwrap(),
            name: None,
        };
        let json = serde_json::to_value(&message).unwrap();

        assert_eq!(json["role"], "user");
        let content = json["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[1]["type"], "file");
        assert_eq!(
            content[1]["file"]["file_data"],
            "https://example.com/doc.pdf"
        );
    }

    #[test]
    fn test_user_content_from_rig_text() {
        let rig_content = message::UserContent::Text(message::Text::new("Hello".to_string()));
        let openrouter_content: UserContent = user_content_to_openai(rig_content).unwrap();

        assert_eq!(
            openrouter_content,
            UserContent::Text {
                text: "Hello".to_string()
            }
        );
    }

    #[test]
    fn test_user_content_from_rig_image_url() {
        let rig_content = message::UserContent::Image(message::Image {
            data: DocumentSourceKind::Url("https://example.com/img.png".to_string()),
            media_type: Some(message::ImageMediaType::PNG),
            detail: Some(ImageDetail::High),
            additional_params: None,
        });
        let openrouter_content: UserContent = user_content_to_openai(rig_content).unwrap();

        match openrouter_content {
            UserContent::Image { image_url } => {
                assert_eq!(image_url.url, "https://example.com/img.png");
                assert_eq!(image_url.detail, Some(ImageDetail::High));
            }
            _ => panic!("Expected Image variant"),
        }
    }

    #[test]
    fn test_user_content_from_rig_image_base64() {
        let rig_content = message::UserContent::Image(message::Image {
            data: DocumentSourceKind::Base64("SGVsbG8=".to_string()),
            media_type: Some(message::ImageMediaType::JPEG),
            detail: Some(ImageDetail::Low),
            additional_params: None,
        });
        let openrouter_content: UserContent = user_content_to_openai(rig_content).unwrap();

        match openrouter_content {
            UserContent::Image { image_url } => {
                assert_eq!(image_url.url, "data:image/jpeg;base64,SGVsbG8=");
                assert_eq!(image_url.detail, Some(ImageDetail::Low));
            }
            _ => panic!("Expected Image variant"),
        }
    }

    #[test]
    fn test_user_content_from_rig_document_url() {
        let rig_content = message::UserContent::Document(message::Document {
            data: DocumentSourceKind::Url("https://example.com/doc.pdf".to_string()),
            media_type: Some(DocumentMediaType::PDF),
            additional_params: None,
        });
        let openrouter_content: UserContent = user_content_to_openai(rig_content).unwrap();

        match openrouter_content {
            UserContent::File { file } => {
                assert_eq!(
                    file.file_data,
                    Some("https://example.com/doc.pdf".to_string())
                );
                assert_eq!(file.filename, Some("document.pdf".to_string()));
            }
            _ => panic!("Expected File variant"),
        }
    }

    #[test]
    fn test_user_content_from_rig_document_base64() {
        let rig_content = message::UserContent::Document(message::Document {
            data: DocumentSourceKind::Base64("JVBERi0xLjQ=".to_string()),
            media_type: Some(DocumentMediaType::PDF),
            additional_params: None,
        });
        let openrouter_content: UserContent = user_content_to_openai(rig_content).unwrap();

        match openrouter_content {
            UserContent::File { file } => {
                assert_eq!(
                    file.file_data,
                    Some("data:application/pdf;base64,JVBERi0xLjQ=".to_string())
                );
                assert_eq!(file.filename, Some("document.pdf".to_string()));
            }
            _ => panic!("Expected File variant"),
        }
    }

    #[test]
    fn test_user_content_from_rig_document_file_id() {
        let rig_content = message::UserContent::Document(message::Document {
            data: DocumentSourceKind::FileId("file_abc".to_string()),
            media_type: None,
            additional_params: None,
        });

        let result: Result<UserContent, _> = user_content_to_openai(rig_content);
        assert!(matches!(
            result,
            Err(message::MessageError::ConversionError(message))
                if message.contains("Provider file IDs are not supported")
        ));
    }

    #[test]
    fn test_openai_file_id_content_round_trips_through_rig_to_openrouter_error() {
        let openai_content = openai::UserContent::File {
            file: openai::FileData {
                file_data: None,
                file_id: Some("file_abc".to_string()),
                filename: None,
            },
        };
        let rig_content: message::UserContent = openai_content.into();

        let result: Result<UserContent, _> = user_content_to_openai(rig_content);
        assert!(matches!(
            result,
            Err(message::MessageError::ConversionError(message))
                if message.contains("Provider file IDs are not supported")
        ));
    }

    #[test]
    fn test_user_content_from_rig_document_string_becomes_text() {
        let rig_content = message::UserContent::Document(message::Document {
            data: DocumentSourceKind::String("Plain text document content".to_string()),
            media_type: Some(DocumentMediaType::TXT),
            additional_params: None,
        });
        let openrouter_content: UserContent = user_content_to_openai(rig_content).unwrap();

        assert_eq!(
            openrouter_content,
            UserContent::Text {
                text: "Plain text document content".to_string()
            }
        );
    }

    #[test]
    fn test_completion_response_with_reasoning_details_maps_to_typed_reasoning() {
        let json = json!({
            "id": "resp_123",
            "object": "chat.completion",
            "created": 1,
            "model": "openrouter/test-model",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "hello",
                    "reasoning": null,
                    "reasoning_details": [
                        {"type":"reasoning.summary","id":"rs_1","summary":"s1"},
                        {"type":"reasoning.text","id":"rs_1","text":"t1","signature":"sig_1"},
                        {"type":"reasoning.encrypted","id":"rs_1","data":"enc_1"}
                    ]
                }
            }]
        });

        let response: CompletionResponse = serde_json::from_value(json).unwrap();
        let converted: completion::CompletionResponse<CompletionResponse> =
            response.try_into().unwrap();
        let items: Vec<completion::AssistantContent> = converted.choice.into_iter().collect();

        assert!(items.iter().any(|item| matches!(
            item,
            completion::AssistantContent::Reasoning(message::Reasoning { id: Some(id), content })
                if id == "rs_1" && content.len() == 3
        )));
    }

    #[test]
    fn test_assistant_reasoning_emits_openrouter_reasoning_details() {
        let reasoning = message::Reasoning {
            id: Some("rs_2".to_string()),
            content: vec![
                message::ReasoningContent::Text {
                    text: "step".to_string(),
                    signature: Some("sig_step".to_string()),
                },
                message::ReasoningContent::Summary("summary".to_string()),
                message::ReasoningContent::Encrypted("enc_blob".to_string()),
            ],
        };

        let messages = assistant_contents_to_messages(OneOrMany::one(
            message::AssistantContent::Reasoning(reasoning),
        ))
        .unwrap();
        let Message::Assistant {
            reasoning,
            reasoning_details,
            ..
        } = messages.first().expect("assistant message")
        else {
            panic!("Expected assistant message");
        };

        assert!(reasoning.is_none());
        assert_eq!(reasoning_details.len(), 3);
        assert!(matches!(
            reasoning_details.first(),
            Some(ReasoningDetails::Text {
                id: Some(id),
                text: Some(text),
                signature: Some(signature),
                ..
            }) if id == "rs_2" && text == "step" && signature == "sig_step"
        ));
    }

    #[test]
    fn test_assistant_redacted_reasoning_emits_encrypted_detail_not_text() {
        let reasoning = message::Reasoning {
            id: Some("rs_redacted".to_string()),
            content: vec![message::ReasoningContent::Redacted {
                data: "opaque-redacted-data".to_string(),
            }],
        };

        let messages = assistant_contents_to_messages(OneOrMany::one(
            message::AssistantContent::Reasoning(reasoning),
        ))
        .unwrap();

        let Message::Assistant {
            reasoning_details,
            reasoning,
            ..
        } = messages.first().expect("assistant message")
        else {
            panic!("Expected assistant message");
        };

        assert!(reasoning.is_none());
        assert_eq!(reasoning_details.len(), 1);
        assert!(matches!(
            reasoning_details.first(),
            Some(ReasoningDetails::Encrypted {
                id: Some(id),
                data,
                ..
            }) if id == "rs_redacted" && data == "opaque-redacted-data"
        ));
    }

    #[test]
    fn test_completion_response_reasoning_details_respects_index_ordering() {
        let json = json!({
            "id": "resp_ordering",
            "object": "chat.completion",
            "created": 1,
            "model": "openrouter/test-model",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "hello",
                    "reasoning": null,
                    "reasoning_details": [
                        {"type":"reasoning.summary","id":"rs_order","index":1,"summary":"second"},
                        {"type":"reasoning.summary","id":"rs_order","index":0,"summary":"first"}
                    ]
                }
            }]
        });

        let response: CompletionResponse = serde_json::from_value(json).unwrap();
        let converted: completion::CompletionResponse<CompletionResponse> =
            response.try_into().unwrap();
        let items: Vec<completion::AssistantContent> = converted.choice.into_iter().collect();
        let reasoning_blocks: Vec<_> = items
            .into_iter()
            .filter_map(|item| match item {
                completion::AssistantContent::Reasoning(reasoning) => Some(reasoning),
                _ => None,
            })
            .collect();

        assert_eq!(reasoning_blocks.len(), 1);
        assert_eq!(reasoning_blocks[0].id.as_deref(), Some("rs_order"));
        assert_eq!(
            reasoning_blocks[0].content,
            vec![
                message::ReasoningContent::Summary("first".to_string()),
                message::ReasoningContent::Summary("second".to_string()),
            ]
        );
    }

    #[test]
    fn test_user_content_from_rig_image_missing_media_type_error() {
        let rig_content = message::UserContent::Image(message::Image {
            data: DocumentSourceKind::Base64("SGVsbG8=".to_string()),
            media_type: None, // Missing media type
            detail: None,
            additional_params: None,
        });
        let result: Result<UserContent, _> = user_content_to_openai(rig_content);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("media type required"));
    }

    #[test]
    fn test_user_content_from_rig_image_raw_bytes_error() {
        let rig_content = message::UserContent::Image(message::Image {
            data: DocumentSourceKind::Raw(vec![1, 2, 3]),
            media_type: Some(message::ImageMediaType::PNG),
            detail: None,
            additional_params: None,
        });
        let result: Result<UserContent, _> = user_content_to_openai(rig_content);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("base64"));
    }

    #[test]
    fn test_user_content_from_rig_video_url() {
        let rig_content = message::UserContent::Video(message::Video {
            data: DocumentSourceKind::Url("https://example.com/video.mp4".to_string()),
            media_type: Some(message::VideoMediaType::MP4),
            additional_params: None,
        });
        let openrouter_content: UserContent = user_content_to_openai(rig_content).unwrap();

        match openrouter_content {
            UserContent::Video { video_url } => {
                assert_eq!(video_url.url, "https://example.com/video.mp4");
            }
            _ => panic!("Expected Video variant"),
        }
    }

    #[test]
    fn test_user_content_from_rig_video_base64() {
        let rig_content = message::UserContent::Video(message::Video {
            data: DocumentSourceKind::Base64("SGVsbG8=".to_string()),
            media_type: Some(message::VideoMediaType::MP4),
            additional_params: None,
        });
        let openrouter_content: UserContent = user_content_to_openai(rig_content).unwrap();

        match openrouter_content {
            UserContent::Video { video_url } => {
                assert_eq!(video_url.url, "data:video/mp4;base64,SGVsbG8=");
            }
            _ => panic!("Expected Video variant"),
        }
    }

    #[test]
    fn test_user_content_from_rig_video_base64_missing_media_type_error() {
        let rig_content = message::UserContent::Video(message::Video {
            data: DocumentSourceKind::Base64("SGVsbG8=".to_string()),
            media_type: None,
            additional_params: None,
        });
        let result: Result<UserContent, _> = user_content_to_openai(rig_content);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("media type"));
    }

    #[test]
    fn test_user_content_from_rig_video_raw_bytes_error() {
        let rig_content = message::UserContent::Video(message::Video {
            data: DocumentSourceKind::Raw(vec![1, 2, 3]),
            media_type: Some(message::VideoMediaType::MP4),
            additional_params: None,
        });
        let result: Result<UserContent, _> = user_content_to_openai(rig_content);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("base64"));
    }

    #[test]
    fn test_user_content_from_rig_audio_base64() {
        let rig_content = message::UserContent::Audio(message::Audio {
            data: DocumentSourceKind::Base64("audiodata".to_string()),
            media_type: Some(message::AudioMediaType::MP3),
            additional_params: None,
        });
        let openrouter_content: UserContent = user_content_to_openai(rig_content).unwrap();

        match openrouter_content {
            UserContent::Audio { input_audio } => {
                assert_eq!(input_audio.data, "audiodata");
                assert_eq!(input_audio.format, message::AudioMediaType::MP3);
            }
            _ => panic!("Expected Audio variant"),
        }
    }

    #[test]
    fn test_user_content_from_rig_audio_missing_media_type_error() {
        let rig_content = message::UserContent::Audio(message::Audio {
            data: DocumentSourceKind::Base64("audiodata".to_string()),
            media_type: None, // missing media type
            additional_params: None,
        });
        let result: Result<UserContent, _> = user_content_to_openai(rig_content);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("media type required"));
    }

    #[test]
    fn test_user_content_from_rig_audio_url_error() {
        let rig_content = message::UserContent::Audio(message::Audio {
            data: DocumentSourceKind::Url("https://example.com/audio.wav".to_string()),
            media_type: Some(message::AudioMediaType::WAV),
            additional_params: None,
        });
        let result: Result<UserContent, _> = user_content_to_openai(rig_content);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("base64"));
    }

    #[test]
    fn test_user_content_from_rig_audio_raw_bytes_error() {
        let rig_content = message::UserContent::Audio(message::Audio {
            data: DocumentSourceKind::Raw(vec![1, 2, 3]),
            media_type: Some(message::AudioMediaType::WAV),
            additional_params: None,
        });
        let result: Result<UserContent, _> = user_content_to_openai(rig_content);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("base64"));
    }

    #[test]
    fn test_user_content_from_rig_video_file_id_error() {
        let rig_content = message::UserContent::Video(message::Video {
            data: DocumentSourceKind::FileId("file-123".to_string()),
            media_type: Some(message::VideoMediaType::MP4),
            additional_params: None,
        });
        let result: Result<UserContent, _> = user_content_to_openai(rig_content);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string()
                .contains("File IDs are not supported for video")
        );
    }

    #[test]
    fn test_user_content_from_rig_audio_file_id_error() {
        let rig_content = message::UserContent::Audio(message::Audio {
            data: DocumentSourceKind::FileId("file-123".to_string()),
            media_type: Some(message::AudioMediaType::MP3),
            additional_params: None,
        });
        let result: Result<UserContent, _> = user_content_to_openai(rig_content);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string()
                .contains("File IDs are not supported for audio")
        );
    }

    #[test]
    fn test_video_helper_converts_to_data_uri() {
        // `UserContent::video(..)` carries base64 data and should become a
        // `video_url` data URI.
        let rig_content =
            message::UserContent::video("SGVsbG8=", Some(message::VideoMediaType::MP4));
        let openrouter_content: UserContent = user_content_to_openai(rig_content).unwrap();

        match openrouter_content {
            UserContent::Video { video_url } => {
                assert_eq!(video_url.url, "data:video/mp4;base64,SGVsbG8=");
            }
            _ => panic!("Expected Video variant"),
        }
    }

    #[test]
    fn test_video_url_helper_passes_url_through() {
        // `UserContent::video_url(..)` passes the URL through unchanged and does
        // not require a media type.
        let rig_content = message::UserContent::video_url("https://example.com/video.mp4", None);
        let openrouter_content: UserContent = user_content_to_openai(rig_content).unwrap();

        match openrouter_content {
            UserContent::Video { video_url } => {
                assert_eq!(video_url.url, "https://example.com/video.mp4");
            }
            _ => panic!("Expected Video variant"),
        }
    }

    #[test]
    fn test_video_raw_helper_errors() {
        // `UserContent::video_raw(..)` carries raw bytes, which OpenRouter cannot
        // accept; the caller must base64-encode first.
        let rig_content =
            message::UserContent::video_raw(vec![1, 2, 3], Some(message::VideoMediaType::MP4));
        let result: Result<UserContent, _> = user_content_to_openai(rig_content);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("base64"));
    }

    #[test]
    fn test_message_conversion_with_pdf() {
        let rig_message = message::Message::User {
            content: OneOrMany::many(vec![
                message::UserContent::Text(message::Text::new(
                    "Summarize this document".to_string(),
                )),
                message::UserContent::Document(message::Document {
                    data: DocumentSourceKind::Url("https://example.com/paper.pdf".to_string()),
                    media_type: Some(DocumentMediaType::PDF),
                    additional_params: None,
                }),
            ])
            .unwrap(),
        };

        let openrouter_messages: Vec<Message> = messages_from_rig_message(rig_message).unwrap();
        assert_eq!(openrouter_messages.len(), 1);

        match &openrouter_messages[0] {
            Message::User { content, .. } => {
                assert_eq!(content.len(), 2);

                // First should be text
                match content.first_ref() {
                    UserContent::Text { text, .. } => assert_eq!(text, "Summarize this document"),
                    _ => panic!("Expected Text"),
                }
            }
            _ => panic!("Expected User message"),
        }
    }

    #[test]
    fn test_user_content_from_string() {
        let content: UserContent = "Hello".into();
        assert_eq!(
            content,
            UserContent::Text {
                text: "Hello".to_string()
            }
        );

        let content: UserContent = String::from("World").into();
        assert_eq!(
            content,
            UserContent::Text {
                text: "World".to_string()
            }
        );
    }

    #[test]
    fn test_completion_response_reasoning_details_with_multiple_ids_stay_separate() {
        let json = json!({
            "id": "resp_multi_id",
            "object": "chat.completion",
            "created": 1,
            "model": "openrouter/test-model",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "hello",
                    "reasoning": null,
                    "reasoning_details": [
                        {"type":"reasoning.summary","id":"rs_a","summary":"a1"},
                        {"type":"reasoning.summary","id":"rs_b","summary":"b1"},
                        {"type":"reasoning.summary","id":"rs_a","summary":"a2"}
                    ]
                }
            }]
        });

        let response: CompletionResponse = serde_json::from_value(json).unwrap();
        let converted: completion::CompletionResponse<CompletionResponse> =
            response.try_into().unwrap();
        let items: Vec<completion::AssistantContent> = converted.choice.into_iter().collect();
        let reasoning_blocks: Vec<_> = items
            .into_iter()
            .filter_map(|item| match item {
                completion::AssistantContent::Reasoning(reasoning) => Some(reasoning),
                _ => None,
            })
            .collect();

        assert_eq!(reasoning_blocks.len(), 2);
        assert_eq!(reasoning_blocks[0].id.as_deref(), Some("rs_a"));
        assert_eq!(
            reasoning_blocks[0].content,
            vec![
                message::ReasoningContent::Summary("a1".to_string()),
                message::ReasoningContent::Summary("a2".to_string()),
            ]
        );
        assert_eq!(reasoning_blocks[1].id.as_deref(), Some("rs_b"));
        assert_eq!(
            reasoning_blocks[1].content,
            vec![message::ReasoningContent::Summary("b1".to_string())]
        );
    }

    #[test]
    fn test_user_content_audio_serialization() {
        let content = UserContent::Audio {
            input_audio: openai::InputAudio {
                data: "SGVsbG8=".to_string(),
                format: AudioMediaType::WAV,
            },
        };
        let json = serde_json::to_value(&content).unwrap();

        assert_eq!(json["type"], "input_audio");
        assert_eq!(json["input_audio"]["data"], "SGVsbG8=");
        assert_eq!(json["input_audio"]["format"], "wav");
    }

    #[test]
    fn test_user_content_audio_deserialization() {
        let json = json!({
            "type": "input_audio",
            "input_audio": {
                "data": "SGVsbG8=",
                "format": "wav"
            }
        });

        let content: UserContent = serde_json::from_value(json).unwrap();
        match content {
            UserContent::Audio { input_audio } => {
                assert_eq!(input_audio.data, "SGVsbG8=");
                assert_eq!(input_audio.format, AudioMediaType::WAV);
            }
            _ => panic!("Expected Audio variant"),
        }
    }

    #[test]
    fn test_message_user_with_audio_serialization() {
        let msg = Message::User {
            content: OneOrMany::many(vec![
                UserContent::Text {
                    text: "Transcribe this audio:".to_string(),
                },
                UserContent::Audio {
                    input_audio: openai::InputAudio {
                        data: "SGVsbG8=".to_string(),
                        format: AudioMediaType::MP3,
                    },
                },
            ])
            .unwrap(),
            name: None,
        };
        let json = serde_json::to_value(&msg).unwrap();

        assert_eq!(json["role"], "user");
        let content = json["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[1]["type"], "input_audio");
        assert_eq!(content[1]["input_audio"]["data"], "SGVsbG8=");
        assert_eq!(content[1]["input_audio"]["format"], "mp3");
    }

    #[test]
    fn test_user_content_video_url_serialization() {
        let content = UserContent::Video {
            video_url: VideoUrl {
                url: "https://example.com/video.mp4".to_string(),
            },
        };
        let json = serde_json::to_value(&content).unwrap();

        assert_eq!(json["type"], "video_url");
        assert_eq!(json["video_url"]["url"], "https://example.com/video.mp4");
    }

    #[test]
    fn test_user_content_video_base64_serialization() {
        let content = UserContent::Video {
            video_url: VideoUrl {
                url: format!(
                    "data:{};base64,SGVsbG8=",
                    VideoMediaType::MP4.to_mime_type()
                ),
            },
        };
        let json = serde_json::to_value(&content).unwrap();

        assert_eq!(json["type"], "video_url");
        assert_eq!(json["video_url"]["url"], "data:video/mp4;base64,SGVsbG8=");
    }

    #[test]
    fn test_user_content_video_url_deserialization() {
        let json = json!({
            "type": "video_url",
            "video_url": {
                "url": "https://example.com/video.mp4"
            }
        });

        let content: UserContent = serde_json::from_value(json).unwrap();
        match content {
            UserContent::Video { video_url } => {
                assert_eq!(video_url.url, "https://example.com/video.mp4");
            }
            _ => panic!("Expected Video variant"),
        }
    }

    #[test]
    fn test_message_user_with_video_serialization() {
        let msg = Message::User {
            content: OneOrMany::many(vec![
                UserContent::Text {
                    text: "Describe this video:".to_string(),
                },
                UserContent::Video {
                    video_url: VideoUrl {
                        url: "https://example.com/video.mp4".to_string(),
                    },
                },
            ])
            .unwrap(),
            name: None,
        };
        let json = serde_json::to_value(&msg).unwrap();

        assert_eq!(json["role"], "user");
        let content = json["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[1]["type"], "video_url");
        assert_eq!(
            content[1]["video_url"]["url"],
            "https://example.com/video.mp4"
        );
    }

    #[test]
    fn test_user_content_video_url_no_media_type_needed() {
        let rig_content = message::UserContent::Video(message::Video {
            data: DocumentSourceKind::Url("https://example.com/video.mp4".to_string()),
            media_type: None,
            additional_params: None,
        });
        let openrouter_content: UserContent = user_content_to_openai(rig_content).unwrap();

        match openrouter_content {
            UserContent::Video { video_url } => {
                assert_eq!(video_url.url, "https://example.com/video.mp4");
            }
            _ => panic!("Expected Video variant"),
        }
    }

    fn prompt_caching_completion_request() -> CompletionRequest {
        CompletionRequest {
            model: None,
            preamble: Some("You are a helpful assistant.".to_string()),
            chat_history: crate::OneOrMany::one(crate::message::Message::user("Hello")),
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
        }
    }

    #[test]
    fn test_final_request_body_applies_prompt_caching_to_converted_completion_request() {
        let request = OpenrouterCompletionRequest::try_from(OpenRouterRequestParams {
            model: "anthropic/claude-3.5-sonnet",
            request: prompt_caching_completion_request(),
            strict_tools: false,
        })
        .expect("request conversion should succeed");

        let body = final_request_body(&request, true).expect("request body should serialize");
        let system_block = &body["messages"][0]["content"][0];

        assert_eq!(system_block["type"], "text");
        assert_eq!(system_block["text"], "You are a helpful assistant.");
        assert_eq!(system_block["cache_control"]["type"], "ephemeral");

        let body = final_request_body(&request, false).expect("request body should serialize");
        assert!(
            body["messages"][0]["content"][0]
                .get("cache_control")
                .is_none(),
            "prompt caching should be opt-in"
        );
    }

    #[test]
    fn test_final_request_body_preserves_stream_flag_when_prompt_caching_enabled() {
        let mut request = OpenrouterCompletionRequest::try_from(OpenRouterRequestParams {
            model: "anthropic/claude-3.5-sonnet",
            request: prompt_caching_completion_request(),
            strict_tools: false,
        })
        .expect("request conversion should succeed");
        request.additional_params = Some(json!({ "stream": true }));

        let body = final_request_body(&request, true).expect("request body should serialize");

        assert_eq!(body["stream"], true);
        assert_eq!(
            body["messages"][0]["content"][0]["cache_control"]["type"],
            "ephemeral"
        );
    }

    #[test]
    fn test_apply_prompt_caching_string_system_message() {
        let mut body = json!({
            "model": "anthropic/claude-3.5-sonnet",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"}
            ]
        });

        apply_prompt_caching(&mut body);

        let system_content = &body["messages"][0]["content"];
        assert!(
            system_content.is_array(),
            "system content should be an array after caching"
        );
        let block = &system_content[0];
        assert_eq!(block["type"], "text");
        assert_eq!(block["text"], "You are a helpful assistant.");
        assert_eq!(block["cache_control"]["type"], "ephemeral");

        // User message should be unchanged.
        assert_eq!(body["messages"][1]["content"], "Hello");
    }

    #[test]
    fn test_apply_prompt_caching_array_system_message_marks_last_block() {
        let mut body = json!({
            "model": "anthropic/claude-3.5-sonnet",
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "Part 1. "},
                        {"type": "text", "text": "Part 2."}
                    ]
                }
            ]
        });

        apply_prompt_caching(&mut body);

        let system_content = &body["messages"][0]["content"];
        assert!(system_content.is_array());
        // Both blocks are preserved; only the last one gets cache_control.
        assert_eq!(system_content.as_array().unwrap().len(), 2);
        assert_eq!(system_content[0]["text"], "Part 1. ");
        assert!(system_content[0].get("cache_control").is_none());
        assert_eq!(system_content[1]["text"], "Part 2.");
        assert_eq!(system_content[1]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn test_apply_prompt_caching_preserves_non_text_blocks() {
        let mut body = json!({
            "model": "anthropic/claude-3.5-sonnet",
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "image", "source": {"type": "url", "url": "https://example.com/img.png"}},
                        {"type": "text", "text": "Describe the image."}
                    ]
                }
            ]
        });

        apply_prompt_caching(&mut body);

        let system_content = &body["messages"][0]["content"];
        assert_eq!(system_content.as_array().unwrap().len(), 2);
        // Non-text block is preserved unchanged.
        assert_eq!(system_content[0]["type"], "image");
        assert!(system_content[0].get("cache_control").is_none());
        // Text block (last) receives the cache boundary.
        assert_eq!(system_content[1]["type"], "text");
        assert_eq!(system_content[1]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn test_apply_prompt_caching_no_system_message_is_noop() {
        let mut body = json!({
            "model": "openai/gpt-4o",
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        });

        let body_before = body.clone();
        apply_prompt_caching(&mut body);
        assert_eq!(
            body, body_before,
            "body should be unchanged when no system message exists"
        );
    }

    #[test]
    fn test_completion_response_extracts_generated_images() {
        let json = json!({
            "id": "resp_img",
            "object": "chat.completion",
            "created": 1,
            "model": "google/gemini-flash-image-preview",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "Here is your image.",
                    "images": [
                        {"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgo="}}
                    ]
                }
            }]
        });

        let response: CompletionResponse = serde_json::from_value(json).unwrap();
        let converted: completion::CompletionResponse<CompletionResponse> =
            response.try_into().unwrap();
        let items: Vec<completion::AssistantContent> = converted.choice.into_iter().collect();
        assert_eq!(items.len(), 2);

        assert!(items.iter().any(|item| matches!(
            item,
            completion::AssistantContent::Text(t) if t.text == "Here is your image."
        )));
        assert!(items.iter().any(|item| matches!(
            item,
            completion::AssistantContent::Image(message::Image {
                data: message::DocumentSourceKind::Base64(b64),
                media_type: Some(message::ImageMediaType::PNG),
                additional_params: Some(_),
                ..
            }) if b64 == "iVBORw0KGgo="
        )));
        assert!(
            items.iter().any(|item| matches!(
                item,
                completion::AssistantContent::Image(image)
                    if is_openrouter_response_image(image)
            )),
            "generated images should be marked as OpenRouter response-only artifacts"
        );
    }

    #[test]
    fn test_completion_response_extracts_generated_images_url() {
        let json = json!({
            "id": "resp_img_url",
            "object": "chat.completion",
            "created": 1,
            "model": "google/gemini-flash-image-preview",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "Here is your image.",
                    "images": [
                        {"type":"image_url","image_url":{"url":"https://example.com/generated.png"}}
                    ]
                }
            }]
        });

        let response: CompletionResponse = serde_json::from_value(json).unwrap();
        let converted: completion::CompletionResponse<CompletionResponse> =
            response.try_into().unwrap();
        let items: Vec<completion::AssistantContent> = converted.choice.into_iter().collect();
        assert_eq!(items.len(), 2);

        assert!(items.iter().any(|item| matches!(
            item,
            completion::AssistantContent::Image(message::Image {
                data: message::DocumentSourceKind::Url(url),
                media_type: None,
                additional_params: Some(_),
                ..
            }) if url == "https://example.com/generated.png"
        )));
        assert!(
            items.iter().any(|item| matches!(
                item,
                completion::AssistantContent::Image(image)
                    if is_openrouter_response_image(image)
            )),
            "generated URL images should be marked as OpenRouter response-only artifacts"
        );
    }

    #[test]
    fn test_generated_images_do_not_break_assistant_history_conversion() {
        let generated_image = response_image_to_assistant_content(&ResponseImage {
            image_url: ImageUrl {
                url: "data:image/png;base64,abc".to_string(),
                detail: None,
            },
        });

        let content = OneOrMany::many(vec![
            completion::AssistantContent::text("Here is your image."),
            generated_image,
        ])
        .unwrap();
        let messages = assistant_contents_to_messages(content).unwrap();

        assert_eq!(messages.len(), 1);
        assert!(matches!(
            &messages[0],
            Message::Assistant { content, .. }
                if content == &vec![openai::AssistantContent::Text {
                    text: "Here is your image.".to_string()
                }]
        ));
    }

    #[test]
    fn test_image_only_assistant_history_is_omitted_for_openrouter() {
        let generated_image = response_image_to_assistant_content(&ResponseImage {
            image_url: ImageUrl {
                url: "data:image/png;base64,abc".to_string(),
                detail: None,
            },
        });

        let messages = assistant_contents_to_messages(OneOrMany::one(generated_image)).unwrap();

        assert!(
            messages.is_empty(),
            "response-only generated image turns should not be replayed as assistant content"
        );
    }

    #[test]
    fn test_unmarked_assistant_image_history_errors_for_openrouter() {
        let image = completion::AssistantContent::image_base64(
            "abc",
            Some(message::ImageMediaType::PNG),
            None,
        );

        let err = assistant_contents_to_messages(OneOrMany::one(image)).unwrap_err();

        match err {
            message::MessageError::ConversionError(message) => assert!(
                message.contains("OpenRouter does not support assistant image content"),
                "unexpected error: {message}"
            ),
        }
    }

    #[test]
    fn test_mixed_text_and_generated_image_replays_text_only_for_openrouter() {
        let generated_image = response_image_to_assistant_content(&ResponseImage {
            image_url: ImageUrl {
                url: "https://example.com/generated.png".to_string(),
                detail: None,
            },
        });

        let messages = assistant_contents_to_messages(
            OneOrMany::many(vec![
                completion::AssistantContent::text("Keep this text."),
                generated_image,
            ])
            .unwrap(),
        )
        .unwrap();

        let serialized = serde_json::to_value(&messages).unwrap();
        assert_eq!(
            serialized,
            json!([{
                "role": "assistant",
                "content": [{"type": "text", "text": "Keep this text."}]
            }])
        );
    }

    #[test]
    fn test_assistant_images_not_serialized_in_request() {
        let msg = Message::Assistant {
            content: vec!["Hello".to_string().into()],
            refusal: None,
            audio: None,
            name: None,
            tool_calls: vec![],
            reasoning: None,
            reasoning_details: vec![],
            images: vec![ResponseImage {
                image_url: ImageUrl {
                    url: "data:image/png;base64,abc".to_string(),
                    detail: None,
                },
            }],
        };
        let serialized = serde_json::to_value(&msg).unwrap();
        assert!(
            serialized.get("images").is_none(),
            "images field must not appear in serialized assistant message"
        );
    }
}
