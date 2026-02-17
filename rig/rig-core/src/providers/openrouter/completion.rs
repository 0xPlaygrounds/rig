use super::{
    client::{ApiErrorResponse, ApiResponse, Client, Usage},
    streaming::StreamingCompletionResponse,
};
use crate::message::{self, DocumentMediaType, DocumentSourceKind, ImageDetail, MimeType};
use crate::telemetry::SpanCombinator;
use crate::{
    OneOrMany,
    completion::{self, CompletionError, CompletionRequest},
    http_client::HttpClientExt,
    json_utils,
    one_or_many::string_or_one_or_many,
    providers::openai,
};
use bytes::Bytes;
use serde::{Deserialize, Serialize, Serializer};
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
/// use rig::providers::openrouter::{ProviderPreferences, ProviderSortStrategy, Quantization};
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
    /// use rig::providers::openrouter::ProviderPreferences;
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
    /// use rig::providers::openrouter::ProviderPreferences;
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
    /// use rig::providers::openrouter::ProviderPreferences;
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
    /// use rig::providers::openrouter::ProviderPreferences;
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
    /// use rig::providers::openrouter::{ProviderPreferences, ProviderSortStrategy};
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
    /// use rig::providers::openrouter::{ProviderPreferences, ThroughputThreshold, PercentileThresholds};
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
    /// use rig::providers::openrouter::{ProviderPreferences, Quantization};
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

impl From<ApiErrorResponse> for CompletionError {
    fn from(err: ApiErrorResponse) -> Self {
        CompletionError::ProviderError(err.message)
    }
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
                ..
            } => {
                let mut content = content
                    .iter()
                    .map(|c| match c {
                        openai::AssistantContent::Text { text } => {
                            completion::AssistantContent::text(text)
                        }
                        openai::AssistantContent::Refusal { refusal } => {
                            completion::AssistantContent::text(refusal)
                        }
                    })
                    .collect::<Vec<_>>();

                content.extend(
                    tool_calls
                        .iter()
                        .map(|call| {
                            completion::AssistantContent::tool_call(
                                &call.id,
                                &call.function.name,
                                call.function.arguments.clone(),
                            )
                        })
                        .collect::<Vec<_>>(),
                );

                if let Some(reasoning) = reasoning {
                    content.push(completion::AssistantContent::reasoning(reasoning));
                }

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
            .map(|usage| completion::Usage {
                input_tokens: usage.prompt_tokens as u64,
                output_tokens: (usage.total_tokens - usage.prompt_tokens) as u64,
                total_tokens: usage.total_tokens as u64,
                cached_input_tokens: 0,
            })
            .unwrap_or_default();

        Ok(completion::CompletionResponse {
            choice,
            usage,
            raw_response: response,
        })
    }
}

/// User content types supported by OpenRouter.
///
/// OpenRouter uses different content type structures than OpenAI's Chat Completions API,
/// particularly for file/document content. This enum matches OpenRouter's API specification.
///
/// # Supported Content Types
///
/// - **Text**: Plain text content
/// - **ImageUrl**: Images via URL or base64 data URI
/// - **File**: PDF documents and other files via URL or base64 data URI
///
/// # Example
///
/// ```rust
/// use rig::providers::openrouter::UserContent;
///
/// // Text content
/// let text = UserContent::text("Hello, world!");
///
/// // Image from URL
/// let image = UserContent::image_url("https://example.com/image.png");
///
/// // PDF from URL
/// let pdf = UserContent::file_url("https://example.com/document.pdf", Some("document.pdf".to_string()));
/// ```
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UserContent {
    /// Plain text content
    Text { text: String },

    /// Image content (URL or base64 data URI)
    ///
    /// Supports: image/png, image/jpeg, image/webp, image/gif
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },

    /// File content (for PDFs and other documents)
    ///
    /// Uses `file_data` field which accepts either a publicly accessible URL
    /// or base64-encoded content as a data URI.
    File { file: FileContent },
}

impl UserContent {
    /// Create text content
    pub fn text(text: impl Into<String>) -> Self {
        UserContent::Text { text: text.into() }
    }

    /// Create image content from URL
    pub fn image_url(url: impl Into<String>) -> Self {
        UserContent::ImageUrl {
            image_url: ImageUrl {
                url: url.into(),
                detail: None,
            },
        }
    }

    /// Create image content from URL with detail level
    pub fn image_url_with_detail(url: impl Into<String>, detail: ImageDetail) -> Self {
        UserContent::ImageUrl {
            image_url: ImageUrl {
                url: url.into(),
                detail: Some(detail),
            },
        }
    }

    /// Create image content from base64 data
    ///
    /// # Arguments
    /// * `data` - Base64-encoded image data
    /// * `mime_type` - MIME type (e.g., "image/png", "image/jpeg")
    /// * `detail` - Optional detail level for image processing
    pub fn image_base64(
        data: impl Into<String>,
        mime_type: &str,
        detail: Option<ImageDetail>,
    ) -> Self {
        let data_uri = format!("data:{};base64,{}", mime_type, data.into());
        UserContent::ImageUrl {
            image_url: ImageUrl {
                url: data_uri,
                detail,
            },
        }
    }

    /// Create file content from URL
    ///
    /// # Arguments
    /// * `url` - URL to the file (must be publicly accessible)
    /// * `filename` - Optional filename for the document
    pub fn file_url(url: impl Into<String>, filename: Option<String>) -> Self {
        UserContent::File {
            file: FileContent {
                filename,
                file_data: Some(url.into()),
            },
        }
    }

    /// Create file content from base64 data
    ///
    /// # Arguments
    /// * `data` - Base64-encoded file data
    /// * `mime_type` - MIME type (e.g., "application/pdf")
    /// * `filename` - Optional filename for the document
    pub fn file_base64(data: impl Into<String>, mime_type: &str, filename: Option<String>) -> Self {
        let data_uri = format!("data:{};base64,{}", mime_type, data.into());
        UserContent::File {
            file: FileContent {
                filename,
                file_data: Some(data_uri),
            },
        }
    }
}

impl From<String> for UserContent {
    fn from(text: String) -> Self {
        UserContent::Text { text }
    }
}

impl From<&str> for UserContent {
    fn from(text: &str) -> Self {
        UserContent::Text {
            text: text.to_string(),
        }
    }
}

impl std::str::FromStr for UserContent {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(UserContent::Text {
            text: s.to_string(),
        })
    }
}

/// Image URL structure for OpenRouter
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ImageUrl {
    /// URL or data URI (data:image/png;base64,...)
    pub url: String,
    /// Image detail level (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<ImageDetail>,
}

/// File content structure for OpenRouter PDF/document support
///
/// OpenRouter supports sending files (particularly PDFs) to models via the `file_data` field,
/// which accepts either:
/// - A publicly accessible URL to the file
/// - A base64-encoded data URI (e.g., `data:application/pdf;base64,...`)
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct FileContent {
    /// Filename (e.g., "document.pdf")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    /// File data source - URL or base64-encoded data URI
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_data: Option<String>,
}

/// Serializes user content as a plain string when there's a single text item,
/// otherwise as an array of content parts.
fn serialize_user_content<S>(
    content: &OneOrMany<UserContent>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    if content.len() == 1
        && let UserContent::Text { text } = content.first_ref()
    {
        return serializer.serialize_str(text);
    }
    content.serialize(serializer)
}

impl TryFrom<message::UserContent> for UserContent {
    type Error = message::MessageError;

    fn try_from(value: message::UserContent) -> Result<Self, Self::Error> {
        match value {
            message::UserContent::Text(message::Text { text }) => Ok(UserContent::Text { text }),

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
                Ok(UserContent::ImageUrl {
                    image_url: ImageUrl { url, detail },
                })
            }

            message::UserContent::Document(message::Document {
                data, media_type, ..
            }) => match data {
                DocumentSourceKind::Url(url) => {
                    let filename = media_type.as_ref().map(|mt| match mt {
                        DocumentMediaType::PDF => "document.pdf",
                        DocumentMediaType::TXT => "document.txt",
                        DocumentMediaType::HTML => "document.html",
                        DocumentMediaType::MARKDOWN => "document.md",
                        DocumentMediaType::CSV => "document.csv",
                        DocumentMediaType::XML => "document.xml",
                        _ => "document",
                    });
                    Ok(UserContent::File {
                        file: FileContent {
                            filename: filename.map(String::from),
                            file_data: Some(url),
                        },
                    })
                }
                DocumentSourceKind::Base64(data) => {
                    let mime = media_type
                        .as_ref()
                        .map(|m| m.to_mime_type())
                        .unwrap_or("application/pdf");
                    let data_uri = format!("data:{mime};base64,{data}");

                    let filename = media_type.as_ref().map(|mt| match mt {
                        DocumentMediaType::PDF => "document.pdf",
                        DocumentMediaType::TXT => "document.txt",
                        DocumentMediaType::HTML => "document.html",
                        DocumentMediaType::MARKDOWN => "document.md",
                        DocumentMediaType::CSV => "document.csv",
                        DocumentMediaType::XML => "document.xml",
                        _ => "document",
                    });

                    Ok(UserContent::File {
                        file: FileContent {
                            filename: filename.map(String::from),
                            file_data: Some(data_uri),
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

            message::UserContent::Audio(_) => Err(message::MessageError::ConversionError(
                "Audio content not supported by OpenRouter file implementation. \
                 Use the OpenAI-compatible audio types for audio support."
                    .into(),
            )),

            message::UserContent::Video(_) => Err(message::MessageError::ConversionError(
                "Video content not supported by OpenRouter file implementation".into(),
            )),

            message::UserContent::ToolResult(_) => Err(message::MessageError::ConversionError(
                "Tool results should be handled as separate messages".into(),
            )),
        }
    }
}

impl TryFrom<OneOrMany<message::UserContent>> for Vec<Message> {
    type Error = message::MessageError;

    fn try_from(value: OneOrMany<message::UserContent>) -> Result<Self, Self::Error> {
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
                        tool_call_id: tool_result.id,
                        content: tool_result
                            .content
                            .into_iter()
                            .map(|c| match c {
                                message::ToolResultContent::Text(message::Text { text }) => text,
                                message::ToolResultContent::Image(_) => {
                                    "[Image content not supported in tool results]".to_string()
                                }
                            })
                            .collect::<Vec<_>>()
                            .join("\n"),
                    }),
                    _ => unreachable!(),
                })
                .collect::<Result<Vec<_>, _>>()
        } else {
            let user_content: Vec<UserContent> = other_content
                .into_iter()
                .map(|content| content.try_into())
                .collect::<Result<Vec<_>, _>>()?;

            let content = OneOrMany::many(user_content)
                .expect("There must be content here if there were no tool result content");

            Ok(vec![Message::User {
                content,
                name: None,
            }])
        }
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

/// OpenRouter message.
///
/// Almost identical to OpenAI's Message, but supports more parameters
/// for some providers like `reasoning`, and uses OpenRouter-specific
/// content types that support images, PDFs, and other file types.
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    #[serde(alias = "developer")]
    System {
        #[serde(deserialize_with = "string_or_one_or_many")]
        content: OneOrMany<openai::SystemContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    User {
        #[serde(
            deserialize_with = "string_or_one_or_many",
            serialize_with = "serialize_user_content"
        )]
        content: OneOrMany<UserContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    Assistant {
        #[serde(default, deserialize_with = "json_utils::string_or_vec")]
        content: Vec<openai::AssistantContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        refusal: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        audio: Option<openai::AudioAssistant>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(
            default,
            deserialize_with = "json_utils::null_or_vec",
            skip_serializing_if = "Vec::is_empty"
        )]
        tool_calls: Vec<openai::ToolCall>,
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        reasoning_details: Vec<ReasoningDetails>,
    },
    #[serde(rename = "tool")]
    ToolResult {
        tool_call_id: String,
        content: String,
    },
}

impl Message {
    pub fn system(content: &str) -> Self {
        Message::System {
            content: OneOrMany::one(content.to_owned().into()),
            name: None,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReasoningDetails {
    #[serde(rename = "reasoning.summary")]
    Summary {
        id: Option<String>,
        format: Option<String>,
        index: Option<usize>,
        summary: String,
    },
    #[serde(rename = "reasoning.encrypted")]
    Encrypted {
        id: Option<String>,
        format: Option<String>,
        index: Option<usize>,
        data: String,
    },
    #[serde(rename = "reasoning.text")]
    Text {
        id: Option<String>,
        format: Option<String>,
        index: Option<usize>,
        text: Option<String>,
        signature: Option<String>,
    },
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

/// Convert OpenAI's UserContent to OpenRouter's UserContent
impl From<openai::UserContent> for UserContent {
    fn from(value: openai::UserContent) -> Self {
        match value {
            openai::UserContent::Text { text } => UserContent::Text { text },
            openai::UserContent::Image { image_url } => UserContent::ImageUrl {
                image_url: ImageUrl {
                    url: image_url.url,
                    detail: Some(image_url.detail),
                },
            },
            openai::UserContent::Audio { input_audio } => {
                // Audio is not directly supported - convert to text placeholder
                // Users should use the native audio support if needed
                UserContent::Text {
                    text: format!("[Audio content: format={:?}]", input_audio.format),
                }
            }
        }
    }
}

impl From<openai::Message> for Message {
    fn from(value: openai::Message) -> Self {
        match value {
            openai::Message::System { content, name } => Self::System { content, name },
            openai::Message::User { content, name } => {
                // Convert OpenAI UserContent to OpenRouter UserContent
                let converted_content = content.map(UserContent::from);
                Self::User {
                    content: converted_content,
                    name,
                }
            }
            openai::Message::Assistant {
                content,
                refusal,
                audio,
                name,
                tool_calls,
            } => Self::Assistant {
                content,
                refusal,
                audio,
                name,
                tool_calls,
                reasoning: None,
                reasoning_details: Vec::new(),
            },
            openai::Message::ToolResult {
                tool_call_id,
                content,
            } => Self::ToolResult {
                tool_call_id,
                content: content.as_text(),
            },
        }
    }
}

impl TryFrom<OneOrMany<message::AssistantContent>> for Vec<Message> {
    type Error = message::MessageError;

    fn try_from(value: OneOrMany<message::AssistantContent>) -> Result<Self, Self::Error> {
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
                        && let Ok(additional_params) =
                            serde_json::from_value::<ToolCallAdditionalParams>(
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
                    let display = r.display_text();
                    if !display.is_empty() {
                        reasoning = Some(display);
                    }
                }
                message::AssistantContent::Image(_) => {
                    return Err(Self::Error::ConversionError(
                        "OpenRouter currently doesn't support images.".into(),
                    ));
                }
            }
        }

        // `OneOrMany` ensures at least one `AssistantContent::Text` or `ToolCall` exists,
        //  so either `content` or `tool_calls` will have some content.
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
        }])
    }
}

// OpenRouter uses its own content types for User messages to support
// images and PDFs. Assistant messages still use OpenAI-compatible types.
impl TryFrom<message::Message> for Vec<Message> {
    type Error = message::MessageError;

    fn try_from(message: message::Message) -> Result<Self, Self::Error> {
        match message {
            message::Message::User { content } => {
                // Use OpenRouter's own conversion for User content
                // This supports images and PDF files via the file content type
                content.try_into()
            }
            message::Message::Assistant { content, .. } => content.try_into(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged, rename_all = "snake_case")]
pub enum ToolChoice {
    None,
    Auto,
    Required,
    Function(Vec<ToolChoiceFunctionKind>),
}

impl TryFrom<crate::message::ToolChoice> for ToolChoice {
    type Error = CompletionError;

    fn try_from(value: crate::message::ToolChoice) -> Result<Self, Self::Error> {
        let res = match value {
            crate::message::ToolChoice::None => Self::None,
            crate::message::ToolChoice::Auto => Self::Auto,
            crate::message::ToolChoice::Required => Self::Required,
            crate::message::ToolChoice::Specific { function_names } => {
                let vec: Vec<ToolChoiceFunctionKind> = function_names
                    .into_iter()
                    .map(|name| ToolChoiceFunctionKind::Function { name })
                    .collect();

                Self::Function(vec)
            }
        };

        Ok(res)
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", content = "function")]
pub enum ToolChoiceFunctionKind {
    Function { name: String },
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
        let model = req.model.clone().unwrap_or_else(|| model.to_string());

        if req.output_schema.is_some() {
            tracing::warn!("Structured outputs currently not supported for OpenRouter");
        }

        let mut full_history: Vec<Message> = match &req.preamble {
            Some(preamble) => vec![Message::system(preamble)],
            None => vec![],
        };
        if let Some(docs) = req.normalized_documents() {
            let docs: Vec<Message> = docs.try_into()?;
            full_history.extend(docs);
        }

        let chat_history: Vec<Message> = req
            .chat_history
            .clone()
            .into_iter()
            .map(|message| message.try_into())
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

        Ok(Self {
            model,
            messages: full_history,
            temperature: req.temperature,
            tools,
            tool_choice,
            additional_params: req.additional_params,
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
}

impl<T> CompletionModel<T> {
    pub fn new(client: Client<T>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
            strict_tools: false,
        }
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

        if enabled!(Level::TRACE) {
            tracing::trace!(
                target: "rig::completions",
                "OpenRouter completion request: {}",
                serde_json::to_string_pretty(&request)?
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
            )
        } else {
            tracing::Span::current()
        };

        let body = serde_json::to_vec(&request)?;

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
                match serde_json::from_slice::<ApiResponse<CompletionResponse>>(&response_body)? {
                    ApiResponse::Ok(response) => {
                        let span = tracing::Span::current();
                        span.record_token_usage(&response.usage);
                        span.record("gen_ai.response.id", &response.id);
                        span.record("gen_ai.response.model_name", &response.model);

                        tracing::debug!(target: "rig::completions",
                            "OpenRouter response: {response:?}");
                        response.try_into()
                    }
                    ApiResponse::Err(err) => Err(CompletionError::ProviderError(err.message)),
                }
            } else {
                Err(CompletionError::ProviderError(
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
        let content = UserContent::text("Hello, world!");
        let json = serde_json::to_value(&content).unwrap();

        assert_eq!(json["type"], "text");
        assert_eq!(json["text"], "Hello, world!");
    }

    #[test]
    fn test_user_content_image_url_serialization() {
        let content = UserContent::image_url("https://example.com/image.png");
        let json = serde_json::to_value(&content).unwrap();

        assert_eq!(json["type"], "image_url");
        assert_eq!(json["image_url"]["url"], "https://example.com/image.png");
        assert!(json["image_url"].get("detail").is_none());
    }

    #[test]
    fn test_user_content_image_url_with_detail_serialization() {
        let content =
            UserContent::image_url_with_detail("https://example.com/image.png", ImageDetail::High);
        let json = serde_json::to_value(&content).unwrap();

        assert_eq!(json["type"], "image_url");
        assert_eq!(json["image_url"]["url"], "https://example.com/image.png");
        assert_eq!(json["image_url"]["detail"], "high");
    }

    #[test]
    fn test_user_content_image_base64_serialization() {
        let content = UserContent::image_base64("SGVsbG8=", "image/png", Some(ImageDetail::Low));
        let json = serde_json::to_value(&content).unwrap();

        assert_eq!(json["type"], "image_url");
        assert_eq!(json["image_url"]["url"], "data:image/png;base64,SGVsbG8=");
        assert_eq!(json["image_url"]["detail"], "low");
    }

    #[test]
    fn test_user_content_file_url_serialization() {
        let content = UserContent::file_url(
            "https://example.com/doc.pdf",
            Some("document.pdf".to_string()),
        );
        let json = serde_json::to_value(&content).unwrap();

        assert_eq!(json["type"], "file");
        assert_eq!(json["file"]["file_data"], "https://example.com/doc.pdf");
        assert_eq!(json["file"]["filename"], "document.pdf");
    }

    #[test]
    fn test_user_content_file_base64_serialization() {
        let content = UserContent::file_base64(
            "JVBERi0xLjQ=",
            "application/pdf",
            Some("report.pdf".to_string()),
        );
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
            UserContent::ImageUrl { image_url } => {
                assert_eq!(image_url.url, "https://example.com/img.jpg");
                assert_eq!(image_url.detail, Some(ImageDetail::High));
            }
            _ => panic!("Expected ImageUrl variant"),
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
            content: OneOrMany::one(UserContent::text("Hello")),
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
                UserContent::text("Check this image:"),
                UserContent::image_url("https://example.com/img.png"),
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
                UserContent::text("Analyze this PDF:"),
                UserContent::file_url(
                    "https://example.com/doc.pdf",
                    Some("document.pdf".to_string()),
                ),
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
        let rig_content = message::UserContent::Text(message::Text {
            text: "Hello".to_string(),
        });
        let openrouter_content: UserContent = rig_content.try_into().unwrap();

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
        let openrouter_content: UserContent = rig_content.try_into().unwrap();

        match openrouter_content {
            UserContent::ImageUrl { image_url } => {
                assert_eq!(image_url.url, "https://example.com/img.png");
                assert_eq!(image_url.detail, Some(ImageDetail::High));
            }
            _ => panic!("Expected ImageUrl variant"),
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
        let openrouter_content: UserContent = rig_content.try_into().unwrap();

        match openrouter_content {
            UserContent::ImageUrl { image_url } => {
                assert_eq!(image_url.url, "data:image/jpeg;base64,SGVsbG8=");
                assert_eq!(image_url.detail, Some(ImageDetail::Low));
            }
            _ => panic!("Expected ImageUrl variant"),
        }
    }

    #[test]
    fn test_user_content_from_rig_document_url() {
        let rig_content = message::UserContent::Document(message::Document {
            data: DocumentSourceKind::Url("https://example.com/doc.pdf".to_string()),
            media_type: Some(DocumentMediaType::PDF),
            additional_params: None,
        });
        let openrouter_content: UserContent = rig_content.try_into().unwrap();

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
        let openrouter_content: UserContent = rig_content.try_into().unwrap();

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
    fn test_user_content_from_rig_document_string_becomes_text() {
        let rig_content = message::UserContent::Document(message::Document {
            data: DocumentSourceKind::String("Plain text document content".to_string()),
            media_type: Some(DocumentMediaType::TXT),
            additional_params: None,
        });
        let openrouter_content: UserContent = rig_content.try_into().unwrap();

        assert_eq!(
            openrouter_content,
            UserContent::Text {
                text: "Plain text document content".to_string()
            }
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
        let result: Result<UserContent, _> = rig_content.try_into();

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
        let result: Result<UserContent, _> = rig_content.try_into();

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("base64"));
    }

    #[test]
    fn test_user_content_from_rig_video_not_supported() {
        let rig_content = message::UserContent::Video(message::Video {
            data: DocumentSourceKind::Url("https://example.com/video.mp4".to_string()),
            media_type: Some(message::VideoMediaType::MP4),
            additional_params: None,
        });
        let result: Result<UserContent, _> = rig_content.try_into();

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Video"));
    }

    #[test]
    fn test_user_content_from_rig_audio_not_supported() {
        let rig_content = message::UserContent::Audio(message::Audio {
            data: DocumentSourceKind::Base64("audiodata".to_string()),
            media_type: Some(message::AudioMediaType::MP3),
            additional_params: None,
        });
        let result: Result<UserContent, _> = rig_content.try_into();

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Audio"));
    }

    #[test]
    fn test_message_conversion_with_pdf() {
        let rig_message = message::Message::User {
            content: OneOrMany::many(vec![
                message::UserContent::Text(message::Text {
                    text: "Summarize this document".to_string(),
                }),
                message::UserContent::Document(message::Document {
                    data: DocumentSourceKind::Url("https://example.com/paper.pdf".to_string()),
                    media_type: Some(DocumentMediaType::PDF),
                    additional_params: None,
                }),
            ])
            .unwrap(),
        };

        let openrouter_messages: Vec<Message> = rig_message.try_into().unwrap();
        assert_eq!(openrouter_messages.len(), 1);

        match &openrouter_messages[0] {
            Message::User { content, .. } => {
                assert_eq!(content.len(), 2);

                // First should be text
                match content.first_ref() {
                    UserContent::Text { text } => assert_eq!(text, "Summarize this document"),
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
    fn test_openai_user_content_conversion() {
        // Test that OpenAI UserContent can be converted to OpenRouter UserContent
        let openai_text = openai::UserContent::Text {
            text: "Hello".to_string(),
        };
        let converted: UserContent = openai_text.into();
        assert_eq!(
            converted,
            UserContent::Text {
                text: "Hello".to_string()
            }
        );

        let openai_image = openai::UserContent::Image {
            image_url: openai::ImageUrl {
                url: "https://example.com/img.png".to_string(),
                detail: ImageDetail::Auto,
            },
        };
        let converted: UserContent = openai_image.into();
        match converted {
            UserContent::ImageUrl { image_url } => {
                assert_eq!(image_url.url, "https://example.com/img.png");
                assert_eq!(image_url.detail, Some(ImageDetail::Auto));
            }
            _ => panic!("Expected ImageUrl"),
        }
    }
}
