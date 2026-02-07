use super::{
    client::{ApiErrorResponse, ApiResponse, Client, Usage},
    streaming::StreamingCompletionResponse,
};
use crate::message;
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
use serde::{Deserialize, Serialize};
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

/// Data collection policy for providers.
///
/// Controls whether providers are allowed to collect and store request data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum DataCollection {
    /// Allow providers that may collect data
    #[default]
    Allow,
    /// Only use providers with zero data retention policies
    Deny,
}

/// Model quantization levels supported by OpenRouter.
///
/// Different quantization levels offer trade-offs between model quality and cost.
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

/// Ordering/sorting strategy for providers.
///
/// Determines how providers should be prioritized when multiple are available.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderSort {
    /// Sort by quality (default)
    Quality,
    /// Sort by price (cheapest first)
    Price,
    /// Sort by throughput (fastest first)
    Throughput,
    /// Sort by latency (lowest latency first)
    Latency,
}

/// Requirements that providers must satisfy.
///
/// These requirements filter providers based on their capabilities and policies.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ProviderRequire {
    /// Data collection policy requirement
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_collection: Option<DataCollection>,

    /// Required quantization levels (providers must support at least one)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantization: Option<Vec<Quantization>>,
}

impl ProviderRequire {
    /// Create a new empty requirements struct
    pub fn new() -> Self {
        Self::default()
    }

    /// Require providers with zero data retention
    pub fn deny_data_collection(mut self) -> Self {
        self.data_collection = Some(DataCollection::Deny);
        self
    }

    /// Allow providers that may collect data
    pub fn allow_data_collection(mut self) -> Self {
        self.data_collection = Some(DataCollection::Allow);
        self
    }

    /// Require specific quantization levels
    pub fn quantization(mut self, quantization: impl IntoIterator<Item = Quantization>) -> Self {
        self.quantization = Some(quantization.into_iter().collect());
        self
    }

    /// Require 8-bit integer quantization
    pub fn int8(mut self) -> Self {
        self.quantization = Some(vec![Quantization::Int8]);
        self
    }

    /// Require 4-bit integer quantization
    pub fn int4(mut self) -> Self {
        self.quantization = Some(vec![Quantization::Int4]);
        self
    }

    /// Require full precision (fp32)
    pub fn fp32(mut self) -> Self {
        self.quantization = Some(vec![Quantization::Fp32]);
        self
    }

    /// Require 16-bit floating point
    pub fn fp16(mut self) -> Self {
        self.quantization = Some(vec![Quantization::Fp16]);
        self
    }
}

/// Provider preferences for OpenRouter routing.
///
/// This struct allows you to control which providers are used and how they are prioritized
/// when making requests through OpenRouter.
///
/// # Example
///
/// ```rust
/// use rig::providers::openrouter::{ProviderPreferences, ProviderSort, ProviderRequire};
///
/// // Create preferences for zero data retention providers, sorted by throughput
/// let prefs = ProviderPreferences::new()
///     .sort(ProviderSort::Throughput)
///     .require(ProviderRequire::new().deny_data_collection().int8())
///     .allow(["Anthropic", "OpenAI"]);
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ProviderPreferences {
    /// Explicit ordering of providers by name (highest priority first)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub order: Option<Vec<String>>,

    /// Providers to allow (whitelist)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allow: Option<Vec<String>>,

    /// Providers to ignore/exclude (blacklist)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ignore: Option<Vec<String>>,

    /// Requirements that providers must satisfy
    #[serde(skip_serializing_if = "Option::is_none")]
    pub require: Option<ProviderRequire>,

    /// How to sort/prioritize providers
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sort: Option<ProviderSort>,
}

impl ProviderPreferences {
    /// Create a new empty provider preferences struct
    pub fn new() -> Self {
        Self::default()
    }

    /// Set explicit provider ordering (highest priority first)
    ///
    /// # Example
    ///
    /// ```rust
    /// use rig::providers::openrouter::ProviderPreferences;
    ///
    /// let prefs = ProviderPreferences::new()
    ///     .order(["Anthropic", "OpenAI", "Google"]);
    /// ```
    pub fn order(mut self, providers: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.order = Some(providers.into_iter().map(|p| p.into()).collect());
        self
    }

    /// Set allowed providers (whitelist)
    ///
    /// Only these providers will be used. Cannot be combined with `ignore`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use rig::providers::openrouter::ProviderPreferences;
    ///
    /// let prefs = ProviderPreferences::new()
    ///     .allow(["Anthropic", "OpenAI"]);
    /// ```
    pub fn allow(mut self, providers: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.allow = Some(providers.into_iter().map(|p| p.into()).collect());
        self
    }

    /// Set providers to ignore (blacklist)
    ///
    /// These providers will be excluded. Cannot be combined with `allow`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use rig::providers::openrouter::ProviderPreferences;
    ///
    /// let prefs = ProviderPreferences::new()
    ///     .ignore(["SomeProvider"]);
    /// ```
    pub fn ignore(mut self, providers: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.ignore = Some(providers.into_iter().map(|p| p.into()).collect());
        self
    }

    /// Set requirements that providers must satisfy
    ///
    /// # Example
    ///
    /// ```rust
    /// use rig::providers::openrouter::{ProviderPreferences, ProviderRequire};
    ///
    /// let prefs = ProviderPreferences::new()
    ///     .require(ProviderRequire::new().deny_data_collection());
    /// ```
    pub fn require(mut self, require: ProviderRequire) -> Self {
        self.require = Some(require);
        self
    }

    /// Set the sorting strategy for providers
    ///
    /// # Example
    ///
    /// ```rust
    /// use rig::providers::openrouter::{ProviderPreferences, ProviderSort};
    ///
    /// let prefs = ProviderPreferences::new()
    ///     .sort(ProviderSort::Throughput);
    /// ```
    pub fn sort(mut self, sort: ProviderSort) -> Self {
        self.sort = Some(sort);
        self
    }

    /// Convenience method: Only use providers with zero data retention
    pub fn zero_data_retention(self) -> Self {
        self.require(ProviderRequire::new().deny_data_collection())
    }

    /// Convenience method: Sort by throughput (fastest providers first)
    pub fn fastest(self) -> Self {
        self.sort(ProviderSort::Throughput)
    }

    /// Convenience method: Sort by price (cheapest providers first)
    pub fn cheapest(self) -> Self {
        self.sort(ProviderSort::Price)
    }

    /// Convenience method: Sort by latency (lowest latency first)
    pub fn lowest_latency(self) -> Self {
        self.sort(ProviderSort::Latency)
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
/// for some providers like `reasoning`.
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
        #[serde(deserialize_with = "string_or_one_or_many")]
        content: OneOrMany<openai::UserContent>,
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

impl From<openai::Message> for Message {
    fn from(value: openai::Message) -> Self {
        match value {
            openai::Message::System { content, name } => Self::System { content, name },
            openai::Message::User { content, name } => Self::User { content, name },
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
                    reasoning = r.reasoning.into_iter().next();
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

// We re-use most of the openai implementation when we can and we re-implement
// only the part that differentate for openrouter (like reasoning support).
impl TryFrom<message::Message> for Vec<Message> {
    type Error = message::MessageError;

    fn try_from(message: message::Message) -> Result<Self, Self::Error> {
        match message {
            message::Message::User { content } => {
                let messages: Vec<openai::Message> = content.try_into()?;
                Ok(messages.into_iter().map(Message::from).collect::<Vec<_>>())
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
            model: model.to_string(),
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
        OpenrouterCompletionRequest::try_from(OpenRouterRequestParams {
            model,
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
        let preamble = completion_request.preamble.clone();
        let request = OpenrouterCompletionRequest::try_from(OpenRouterRequestParams {
            model: self.model.as_ref(),
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
                gen_ai.request.model = self.model,
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

    // ================================================================
    // Provider Selection Tests
    // ================================================================

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
    }

    #[test]
    fn test_provider_sort_serialization() {
        assert_eq!(
            serde_json::to_string(&ProviderSort::Quality).unwrap(),
            r#""quality""#
        );
        assert_eq!(
            serde_json::to_string(&ProviderSort::Price).unwrap(),
            r#""price""#
        );
        assert_eq!(
            serde_json::to_string(&ProviderSort::Throughput).unwrap(),
            r#""throughput""#
        );
        assert_eq!(
            serde_json::to_string(&ProviderSort::Latency).unwrap(),
            r#""latency""#
        );
    }

    #[test]
    fn test_provider_require_builder() {
        let require = ProviderRequire::new()
            .deny_data_collection()
            .int8();

        assert_eq!(require.data_collection, Some(DataCollection::Deny));
        assert_eq!(require.quantization, Some(vec![Quantization::Int8]));
    }

    #[test]
    fn test_provider_require_multiple_quantizations() {
        let require = ProviderRequire::new()
            .quantization([Quantization::Int8, Quantization::Fp16]);

        assert_eq!(
            require.quantization,
            Some(vec![Quantization::Int8, Quantization::Fp16])
        );
    }

    #[test]
    fn test_provider_require_serialization() {
        let require = ProviderRequire::new()
            .deny_data_collection()
            .int8();

        let json = serde_json::to_value(&require).unwrap();
        assert_eq!(json["data_collection"], "deny");
        assert_eq!(json["quantization"], json!(["int8"]));
    }

    #[test]
    fn test_provider_preferences_builder() {
        let prefs = ProviderPreferences::new()
            .order(["Anthropic", "OpenAI"])
            .allow(["Anthropic", "OpenAI", "Google"])
            .sort(ProviderSort::Throughput)
            .require(ProviderRequire::new().deny_data_collection());

        assert_eq!(
            prefs.order,
            Some(vec!["Anthropic".to_string(), "OpenAI".to_string()])
        );
        assert_eq!(
            prefs.allow,
            Some(vec![
                "Anthropic".to_string(),
                "OpenAI".to_string(),
                "Google".to_string()
            ])
        );
        assert_eq!(prefs.sort, Some(ProviderSort::Throughput));
        assert!(prefs.require.is_some());
    }

    #[test]
    fn test_provider_preferences_ignore() {
        let prefs = ProviderPreferences::new()
            .ignore(["SomeProvider"]);

        assert_eq!(prefs.ignore, Some(vec!["SomeProvider".to_string()]));
    }

    #[test]
    fn test_provider_preferences_convenience_methods() {
        let prefs = ProviderPreferences::new()
            .zero_data_retention()
            .fastest();

        assert_eq!(prefs.sort, Some(ProviderSort::Throughput));
        assert!(prefs.require.is_some());
        let require = prefs.require.unwrap();
        assert_eq!(require.data_collection, Some(DataCollection::Deny));
    }

    #[test]
    fn test_provider_preferences_to_json() {
        let prefs = ProviderPreferences::new()
            .order(["Anthropic"])
            .sort(ProviderSort::Throughput)
            .require(ProviderRequire::new().deny_data_collection().int8());

        let json = prefs.to_json();
        let provider = &json["provider"];

        assert_eq!(provider["order"], json!(["Anthropic"]));
        assert_eq!(provider["sort"], "throughput");
        assert_eq!(provider["require"]["data_collection"], "deny");
        assert_eq!(provider["require"]["quantization"], json!(["int8"]));
    }

    #[test]
    fn test_provider_preferences_serialization_skips_none() {
        let prefs = ProviderPreferences::new()
            .sort(ProviderSort::Price);

        let json = serde_json::to_value(&prefs).unwrap();

        // Only sort should be present
        assert_eq!(json["sort"], "price");
        assert!(json.get("order").is_none());
        assert!(json.get("allow").is_none());
        assert!(json.get("ignore").is_none());
        assert!(json.get("require").is_none());
    }

    #[test]
    fn test_provider_preferences_deserialization() {
        let json = json!({
            "order": ["Anthropic", "OpenAI"],
            "sort": "throughput",
            "require": {
                "data_collection": "deny",
                "quantization": ["int8", "fp16"]
            }
        });

        let prefs: ProviderPreferences = serde_json::from_value(json).unwrap();

        assert_eq!(
            prefs.order,
            Some(vec!["Anthropic".to_string(), "OpenAI".to_string()])
        );
        assert_eq!(prefs.sort, Some(ProviderSort::Throughput));

        let require = prefs.require.unwrap();
        assert_eq!(require.data_collection, Some(DataCollection::Deny));
        assert_eq!(
            require.quantization,
            Some(vec![Quantization::Int8, Quantization::Fp16])
        );
    }

    #[test]
    fn test_provider_preferences_full_integration() {
        // Test a complete provider preferences object that would be sent to OpenRouter
        let prefs = ProviderPreferences::new()
            .order(["Anthropic", "OpenAI"])
            .allow(["Anthropic", "OpenAI", "Google"])
            .sort(ProviderSort::Throughput)
            .require(ProviderRequire::new().deny_data_collection().int8());

        let json = prefs.to_json();

        // Verify the structure matches OpenRouter's expected format
        assert!(json.get("provider").is_some());
        let provider = &json["provider"];
        assert_eq!(provider["order"], json!(["Anthropic", "OpenAI"]));
        assert_eq!(
            provider["allow"],
            json!(["Anthropic", "OpenAI", "Google"])
        );
        assert_eq!(provider["sort"], "throughput");
        assert_eq!(provider["require"]["data_collection"], "deny");
        assert_eq!(provider["require"]["quantization"], json!(["int8"]));
    }
}
