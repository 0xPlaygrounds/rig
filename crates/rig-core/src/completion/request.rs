//! Completion request and response definitions.
//!
//! Provider `functions` modules translate [`CompletionRequest`] into their
//! native HTTP request format and parse the reply back into
//! [`CompletionResponse`].
//!
//! # Low-level request example
//!
//! ```no_run
//! use rig_core::{
//!     completion::{AssistantContent, CompletionRequest},
//!     http_runtime::HttpRuntime,
//!     providers::openai,
//! };
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let cfg = openai::functions::Config::from_env(openai::GPT_5_2)?;
//! let rt = HttpRuntime::new();
//!
//! let request = CompletionRequest {
//!     temperature: Some(0.5),
//!     ..CompletionRequest::with_history(
//!         Some("You are a concise assistant."),
//!         Vec::new(),
//!         "Who are you?",
//!     )
//! };
//!
//! let response = openai::functions::complete(&cfg, &rt, request).await?;
//! for item in response.choice {
//!     if let AssistantContent::Text(text) = item {
//!         println!("{}", text.text);
//!     }
//! }
//! # Ok(())
//! # }
//! ```

use super::message::{AssistantContent, DocumentMediaType};
use crate::message::ToolChoice;
use crate::message::{Message, UserContent};
use crate::provider_response;
use crate::{OneOrMany, http_client};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ops::{Add, AddAssign};
use thiserror::Error;

// Errors
/// Errors returned by completion models.
///
/// Inspect provider failures with [`Self::provider_response_body`],
/// [`Self::provider_response_json`], and [`Self::provider_response_status`].
/// These recover the provider's raw HTTP status and response body so you can
/// branch on a provider error code or surface a precise diagnostic. The same
/// helpers are available on `EmbeddingError`, `ImageGenerationError`,
/// `AudioGenerationError`, `TranscriptionError`, and `RerankError`.
///
/// ```
/// use rig_core::completion::CompletionError;
///
/// /// Log the provider's raw error response when a completion fails.
/// fn report(error: &CompletionError) {
///     if let Some(status) = error.provider_response_status() {
///         // Note: this can be a 2xx status for providers that return an error
///         // envelope alongside a success status — the error itself means failure.
///         eprintln!("provider returned HTTP {status}");
///     }
///     match error.provider_response_json() {
///         Ok(Some(json)) => eprintln!("provider error payload: {json}"),
///         Ok(None) => eprintln!("no provider response body (e.g. a transport error)"),
///         Err(_) => eprintln!(
///             "provider response body was not valid JSON: {:?}",
///             error.provider_response_body(),
///         ),
///     }
/// }
/// ```
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CompletionError {
    /// Http error (e.g.: connection error, timeout, etc.)
    #[error("HttpError: {0}")]
    HttpError(#[from] http_client::Error),

    /// Json error (e.g.: serialization, deserialization)
    #[error("JsonError: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Url error (e.g.: invalid URL)
    #[error("UrlError: {0}")]
    UrlError(#[from] url::ParseError),

    #[cfg(not(target_family = "wasm"))]
    /// Error building the completion request
    #[error("RequestError: {0}")]
    RequestError(#[from] Box<dyn std::error::Error + Send + Sync + 'static>),

    #[cfg(target_family = "wasm")]
    /// Error building the completion request
    #[error("RequestError: {0}")]
    RequestError(#[from] Box<dyn std::error::Error + 'static>),

    /// Error parsing the completion response
    #[error("ResponseError: {0}")]
    ResponseError(String),

    /// Error returned by the completion model provider
    #[error("ProviderError: {0}")]
    ProviderError(String),

    /// Raw error response preserved from the completion model provider
    #[error("ProviderResponseError: {0}")]
    ProviderResponse(provider_response::ProviderResponseError),
}

crate::provider_response::impl_provider_response_helpers!(CompletionError);

impl CompletionError {
    /// Maps an SSE transport error into a completion error without flattening HTTP failures.
    ///
    /// Non-success HTTP responses remain [`CompletionError::HttpError`] so provider response
    /// helpers can read status and body. Other transport failures keep the existing
    /// [`CompletionError::ProviderError`] display string behavior.
    pub(crate) fn from_stream_transport(error: http_client::Error) -> Self {
        if error.non_success_status().is_some() {
            Self::HttpError(error)
        } else {
            Self::ProviderError(error.to_string())
        }
    }
}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct Document {
    /// Stable document identifier included in the serialized context block.
    pub id: String,
    /// Text content passed to the model as retrieval or static context.
    pub text: String,
    /// Additional string metadata rendered before the document text.
    #[serde(flatten)]
    pub additional_props: HashMap<String, String>,
}

impl std::fmt::Display for Document {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            concat!("<file id: {}>\n", "{}\n", "</file>\n"),
            self.id,
            if self.additional_props.is_empty() {
                self.text.clone()
            } else {
                let mut sorted_props = self.additional_props.iter().collect::<Vec<_>>();
                sorted_props.sort_by(|a, b| a.0.cmp(b.0));
                let metadata = sorted_props
                    .iter()
                    .map(|(k, v)| format!("{k}: {v:?}"))
                    .collect::<Vec<_>>()
                    .join(" ");
                format!("<metadata {} />\n{}", metadata, self.text)
            }
        )
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ToolDefinition {
    /// Tool name exposed to the model. It must match the registered tool name.
    pub name: String,
    /// Human-readable description sent to the model.
    pub description: String,
    /// JSON Schema describing tool arguments.
    pub parameters: serde_json::Value,
}

/// Provider-native tool definition.
///
/// Stored under `additional_params.tools` and forwarded by providers that support
/// provider-managed tools.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ProviderToolDefinition {
    /// Tool type/kind name as expected by the target provider (for example `web_search`).
    #[serde(rename = "type")]
    pub kind: String,
    /// Additional provider-specific configuration for this hosted tool.
    #[serde(flatten, default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub config: serde_json::Map<String, serde_json::Value>,
}

impl ProviderToolDefinition {
    /// Creates a provider-hosted tool definition by type.
    pub fn new(kind: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            config: serde_json::Map::new(),
        }
    }

    /// Adds a provider-specific configuration key/value.
    pub fn with_config(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.config.insert(key.into(), value);
        self
    }
}

/// Why the model stopped generating, normalized across providers.
///
/// Providers report this under different names and vocabularies
/// (`finish_reason`, `stop_reason`, `stopReason`, …); each provider's
/// response conversion maps its wire value onto these variants and preserves
/// anything unmapped verbatim in [`FinishReason::Other`]. Closes #2090/#1886.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// Natural end of the response.
    Stop,
    /// The response hit the output-token limit.
    Length,
    /// The model stopped to call one or more tools.
    ToolCalls,
    /// The provider filtered the content.
    ContentFilter,
    /// A provider-specific reason outside the normalized vocabulary,
    /// carried verbatim.
    Other(String),
}

/// General completion response struct: the completion choice plus normalized
/// response metadata. The completion choice contains one or more assistant
/// content items.
///
/// This type is concrete — it carries no provider-typed payload. Callers who
/// need a provider's raw typed response call that provider's own parse/
/// completion functions directly, on the provider's side of the boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CompletionResponse {
    /// The completion choice (represented by one or more assistant message content)
    /// returned by the completion model provider
    pub choice: OneOrMany<AssistantContent>,
    /// Tokens used during prompting and responding
    pub usage: Usage,
    /// Provider-assigned message ID (e.g. OpenAI Responses API `msg_` ID).
    /// Used to pair reasoning input items with their output items in multi-turn.
    pub message_id: Option<String>,
    /// Why the model stopped generating, when the provider reported it.
    pub finish_reason: Option<FinishReason>,
    /// Name of the provider that produced this response (descriptor name,
    /// e.g. `"openai"`).
    pub provider: String,
    /// Provider-reported model identifier for the response, when available.
    pub model: Option<String>,
}

impl CompletionResponse {
    /// Create a response from its required parts; optional metadata starts
    /// unset and is filled with the `with_*` helpers.
    pub fn new(
        choice: OneOrMany<AssistantContent>,
        usage: Usage,
        provider: impl Into<String>,
    ) -> Self {
        Self {
            choice,
            usage,
            message_id: None,
            finish_reason: None,
            provider: provider.into(),
            model: None,
        }
    }

    /// Attach the provider-assigned message ID.
    pub fn with_message_id(mut self, message_id: impl Into<String>) -> Self {
        self.message_id = Some(message_id.into());
        self
    }

    /// Attach the normalized finish reason.
    pub fn with_finish_reason(mut self, finish_reason: FinishReason) -> Self {
        self.finish_reason = Some(finish_reason);
        self
    }

    /// Attach the provider-reported model identifier.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }
}

/// Struct representing the token usage for a completion request.
/// If tokens used are `0`, then the provider failed to supply token usage metrics.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub struct Usage {
    /// The number of input ("prompt") tokens used in a given request.
    pub input_tokens: u64,
    /// The number of output ("completion") tokens used in a given request.
    pub output_tokens: u64,
    /// We store this separately as some providers may only report one number
    pub total_tokens: u64,
    /// The number of input tokens read from a provider-managed cache
    pub cached_input_tokens: u64,
    /// The number of input tokens written to a provider-managed cache
    pub cache_creation_input_tokens: u64,
    /// The number of tool-use prompt tokens used in a given request.
    #[serde(default)]
    pub tool_use_prompt_tokens: u64,
    /// The number of tokens spent on internal reasoning / "thoughts" by reasoning-capable
    /// models (e.g. Gemini thinking, Anthropic extended thinking, OpenAI o-series).
    pub reasoning_tokens: u64,
}

impl Usage {
    /// Creates a new instance of `Usage`.
    pub fn new() -> Self {
        Self {
            input_tokens: 0,
            output_tokens: 0,
            total_tokens: 0,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
            tool_use_prompt_tokens: 0,
            reasoning_tokens: 0,
        }
    }

    /// Whether any usage values are set and non-zero.
    ///
    /// Zero-valued usage is this type's documented sentinel for "the provider
    /// supplied no usage metrics", so `false` means usage was not reported.
    pub fn has_values(&self) -> bool {
        *self != Self::new()
    }
}

impl Default for Usage {
    fn default() -> Self {
        Self::new()
    }
}

impl Add for Usage {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Self {
            input_tokens: self.input_tokens + other.input_tokens,
            output_tokens: self.output_tokens + other.output_tokens,
            total_tokens: self.total_tokens + other.total_tokens,
            cached_input_tokens: self.cached_input_tokens + other.cached_input_tokens,
            cache_creation_input_tokens: self.cache_creation_input_tokens
                + other.cache_creation_input_tokens,
            tool_use_prompt_tokens: self.tool_use_prompt_tokens + other.tool_use_prompt_tokens,
            reasoning_tokens: self.reasoning_tokens + other.reasoning_tokens,
        }
    }
}

impl AddAssign for Usage {
    fn add_assign(&mut self, other: Self) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.total_tokens += other.total_tokens;
        self.cached_input_tokens += other.cached_input_tokens;
        self.cache_creation_input_tokens += other.cache_creation_input_tokens;
        self.tool_use_prompt_tokens += other.tool_use_prompt_tokens;
        self.reasoning_tokens += other.reasoning_tokens;
    }
}

/// Struct representing a general completion request that can be sent to a completion model provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    /// Optional model override for this request.
    pub model: Option<String>,
    /// Legacy preamble field preserved for backwards compatibility.
    ///
    /// New code should prefer a leading [`Message::System`]
    /// in `chat_history` as the canonical representation of system instructions.
    pub preamble: Option<String>,
    /// The chat history to be sent to the completion model provider.
    /// The very last message will always be the prompt (hence why there is *always* one)
    pub chat_history: OneOrMany<Message>,
    /// The documents to be sent to the completion model provider
    pub documents: Vec<Document>,
    /// The tools to be sent to the completion model provider
    pub tools: Vec<ToolDefinition>,
    /// The temperature to be sent to the completion model provider
    pub temperature: Option<f64>,
    /// The max tokens to be sent to the completion model provider
    pub max_tokens: Option<u64>,
    /// Whether tools are required to be used by the model provider or not before providing a response.
    pub tool_choice: Option<ToolChoice>,
    /// Additional provider-specific parameters to be sent to the completion model provider
    pub additional_params: Option<serde_json::Value>,
    /// Optional JSON Schema for structured output. When set, providers that support
    /// native structured outputs will constrain the model's response to match this schema.
    pub output_schema: Option<schemars::Schema>,
    /// Whether to record sensitive request, response, and tool content on GenAI
    /// telemetry spans.
    ///
    /// Defaults to `false`. Enabling this can expose prompts, retrieved context,
    /// tool results, model responses, and other sensitive or high-cardinality data
    /// through OpenTelemetry span attributes, which can increase observability
    /// backend storage and query costs. Only enable it when the caller has
    /// explicitly opted in to content telemetry.
    ///
    /// Higher-level agent drivers use this flag for portable input, output, and
    /// tool-content telemetry. Direct provider calls only forward the policy;
    /// the exact content fields available there are provider- and
    /// surface-dependent, especially for streaming responses that are consumed
    /// after the provider returns.
    ///
    /// This is local observability policy and is never serialized into provider
    /// request payloads.
    #[serde(skip)]
    pub record_telemetry_content: bool,
}

impl CompletionRequest {
    /// Starts a fluent [`CompletionRequestBuilder`] for `prompt`.
    ///
    /// The builder holds only owned request data — no model, transport, or
    /// provider — so [`CompletionRequestBuilder::build`] hands back a plain
    /// [`CompletionRequest`] that you pass to a provider's `functions::complete`
    /// or `functions::open_stream`.
    ///
    /// ```
    /// use rig_core::completion::CompletionRequest;
    ///
    /// let request = CompletionRequest::builder("What is Rust?")
    ///     .preamble("You are concise.")
    ///     .temperature(0.2)
    ///     .max_tokens(256)
    ///     .build();
    ///
    /// // The preamble is canonicalized into a leading system message.
    /// assert_eq!(request.chat_history.len(), 2);
    /// assert!(request.preamble.is_none());
    /// ```
    pub fn builder(prompt: impl Into<Message>) -> CompletionRequestBuilder {
        CompletionRequestBuilder::new(prompt)
    }

    /// A request for `prompt` with `preamble` and prior `history`: the
    /// canonical hand-built request shape (history first, prompt last, every
    /// other field defaulted for functional-update syntax).
    pub fn with_history(
        preamble: Option<&str>,
        history: Vec<Message>,
        prompt: impl Into<Message>,
    ) -> Self {
        let prompt = prompt.into();
        let chat_history = match OneOrMany::many(history) {
            Ok(mut messages) => {
                messages.push(prompt);
                messages
            }
            Err(_) => OneOrMany::one(prompt),
        };
        Self {
            model: None,
            preamble: preamble.map(str::to_string),
            chat_history,
            documents: Vec::new(),
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        }
    }

    /// A request for a bare `prompt` with no preamble or history.
    pub fn from_prompt(prompt: impl Into<Message>) -> Self {
        Self::with_history(None, Vec::new(), prompt)
    }

    /// Extracts a name from the output schema's `"title"` field, falling back to `"response_schema"`.
    /// Useful for providers that require a name alongside the JSON Schema (e.g., OpenAI).
    pub fn output_schema_name(&self) -> Option<String> {
        self.output_schema.as_ref().map(|schema| {
            schema
                .as_object()
                .and_then(|o| o.get("title"))
                .and_then(|v| v.as_str())
                .unwrap_or("response_schema")
                .to_string()
        })
    }

    /// Returns the normalized input messages used by runtime telemetry:
    /// the chat history (which already carries any system preamble and the
    /// prompt) with the request's documents inserted as a message after any
    /// leading system messages.
    pub fn messages_for_telemetry(&self) -> Vec<Message> {
        let mut chat_history: Vec<Message> = self.chat_history.clone().into_iter().collect();
        if let Some(documents) = Self::normalized_documents_from(&self.documents) {
            let insert_at = chat_history
                .iter()
                .position(|message| !matches!(message, Message::System { .. }))
                .unwrap_or(chat_history.len());
            chat_history.insert(insert_at, documents);
        }
        chat_history
    }

    /// Returns documents normalized into a message (if any).
    /// Most providers do not accept documents directly as input, so it needs to convert into a
    /// `Message` so that it can be incorporated into `chat_history`.
    pub fn normalized_documents(&self) -> Option<Message> {
        Self::normalized_documents_from(&self.documents)
    }

    fn normalized_documents_from(documents: &[Document]) -> Option<Message> {
        if documents.is_empty() {
            return None;
        }

        // Most providers will convert documents into a text unless it can handle document messages.
        // We use `UserContent::document` for those who handle it directly!
        let messages = documents
            .iter()
            .map(|doc| {
                UserContent::document(
                    doc.to_string(),
                    // In the future, we can customize `Document` to pass these extra types through.
                    // Most providers ditch these but they might want to use them.
                    Some(DocumentMediaType::TXT),
                )
            })
            .collect::<Vec<_>>();

        OneOrMany::from_iter_optional(messages).map(|content| Message::User { content })
    }

    pub(crate) fn chat_history_with_documents(&self) -> Vec<Message> {
        let mut chat_history = self.chat_history.iter().cloned().collect::<Vec<_>>();
        if let Some(documents) = self.normalized_documents() {
            let insert_at = chat_history
                .iter()
                .position(|message| !matches!(message, Message::System { .. }))
                .unwrap_or(chat_history.len());
            chat_history.insert(insert_at, documents);
        }
        chat_history
    }

    /// Adds a provider-hosted tool by storing it in `additional_params.tools`.
    pub fn with_provider_tool(mut self, tool: ProviderToolDefinition) -> Self {
        self.additional_params =
            merge_provider_tools_into_additional_params(self.additional_params, vec![tool]);
        self
    }

    /// Adds provider-hosted tools by storing them in `additional_params.tools`.
    pub fn with_provider_tools(mut self, tools: Vec<ProviderToolDefinition>) -> Self {
        self.additional_params =
            merge_provider_tools_into_additional_params(self.additional_params, tools);
        self
    }
}

fn merge_provider_tools_into_additional_params(
    additional_params: Option<serde_json::Value>,
    provider_tools: Vec<ProviderToolDefinition>,
) -> Option<serde_json::Value> {
    if provider_tools.is_empty() {
        return additional_params;
    }

    let mut provider_tools_json = provider_tools
        .into_iter()
        .map(|ProviderToolDefinition { kind, mut config }| {
            // Force the provider tool type from the strongly-typed field.
            config.insert("type".to_string(), serde_json::Value::String(kind));
            serde_json::Value::Object(config)
        })
        .collect::<Vec<_>>();

    let mut params_map = match additional_params {
        Some(serde_json::Value::Object(map)) => map,
        Some(serde_json::Value::Bool(stream)) => {
            let mut map = serde_json::Map::new();
            map.insert("stream".to_string(), serde_json::Value::Bool(stream));
            map
        }
        _ => serde_json::Map::new(),
    };

    let mut merged_tools = match params_map.remove("tools") {
        Some(serde_json::Value::Array(existing)) => existing,
        _ => Vec::new(),
    };
    merged_tools.append(&mut provider_tools_json);
    params_map.insert("tools".to_string(), serde_json::Value::Array(merged_tools));
    Some(serde_json::Value::Object(params_map))
}

/// A fluent builder for [`CompletionRequest`].
///
/// This is request *data* only: it stores no model, transport, provider, or
/// callback, and has no type parameter. [`Self::build`] returns a concrete
/// [`CompletionRequest`] which you hand to a provider's free functions —
/// there is deliberately no `send`/`stream` terminal, because that would put
/// a provider back inside the builder.
///
/// ```no_run
/// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
/// use rig_core::{completion::CompletionRequest, http_runtime::HttpRuntime, providers::openai};
///
/// let request = CompletionRequest::builder("Explain ownership.")
///     .preamble("You are a Rust tutor.")
///     .temperature(0.2)
///     .build();
///
/// let cfg = openai::functions::Config::from_env("gpt-4o")?;
/// let response = openai::functions::complete(&cfg, &HttpRuntime::new(), request).await?;
/// # let _ = response;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct CompletionRequestBuilder {
    prompt: Message,
    request_model: Option<String>,
    preamble: Option<String>,
    chat_history: Vec<Message>,
    documents: Vec<Document>,
    tools: Vec<ToolDefinition>,
    provider_tools: Vec<ProviderToolDefinition>,
    temperature: Option<f64>,
    max_tokens: Option<u64>,
    tool_choice: Option<ToolChoice>,
    additional_params: Option<serde_json::Value>,
    output_schema: Option<schemars::Schema>,
    record_telemetry_content: bool,
}

impl CompletionRequestBuilder {
    /// Starts a builder for `prompt`. Prefer [`CompletionRequest::builder`].
    pub fn new(prompt: impl Into<Message>) -> Self {
        Self {
            prompt: prompt.into(),
            request_model: None,
            preamble: None,
            chat_history: Vec::new(),
            documents: Vec::new(),
            tools: Vec::new(),
            provider_tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        }
    }

    /// Sets the preamble, canonicalized at [`Self::build`] into a leading
    /// [`Message::System`] rather than the legacy `preamble` field.
    pub fn preamble(mut self, preamble: impl Into<String>) -> Self {
        self.preamble = Some(preamble.into());
        self
    }

    /// Clears any preamble set so far.
    pub fn without_preamble(mut self) -> Self {
        self.preamble = None;
        self
    }

    /// Overrides the model used for this request.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.request_model = Some(model.into());
        self
    }

    /// Overrides the model used for this request, or clears the override.
    pub fn model_opt(mut self, model: Option<String>) -> Self {
        self.request_model = model;
        self
    }

    /// Appends a message to the chat history.
    pub fn message(mut self, message: Message) -> Self {
        self.chat_history.push(message);
        self
    }

    /// Appends messages to the chat history, in order.
    pub fn messages(mut self, messages: impl IntoIterator<Item = Message>) -> Self {
        self.chat_history.extend(messages);
        self
    }

    /// Appends a document to the request.
    pub fn document(mut self, document: Document) -> Self {
        self.documents.push(document);
        self
    }

    /// Appends documents to the request, in order.
    pub fn documents(mut self, documents: impl IntoIterator<Item = Document>) -> Self {
        self.documents.extend(documents);
        self
    }

    /// Appends a tool definition.
    pub fn tool(mut self, tool: ToolDefinition) -> Self {
        self.tools.push(tool);
        self
    }

    /// Appends tool definitions, in order.
    pub fn tools(mut self, tools: impl IntoIterator<Item = ToolDefinition>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Appends a provider-hosted tool, merged into `additional_params.tools`
    /// at [`Self::build`].
    pub fn provider_tool(mut self, tool: ProviderToolDefinition) -> Self {
        self.provider_tools.push(tool);
        self
    }

    /// Appends provider-hosted tools, in order.
    pub fn provider_tools(
        mut self,
        tools: impl IntoIterator<Item = ProviderToolDefinition>,
    ) -> Self {
        self.provider_tools.extend(tools);
        self
    }

    /// **Merges** provider-specific parameters into any already set.
    ///
    /// Use [`Self::additional_params_opt`] to replace them outright.
    pub fn additional_params(mut self, additional_params: serde_json::Value) -> Self {
        self.additional_params = match self.additional_params {
            Some(params) => Some(crate::json_utils::merge(params, additional_params)),
            None => Some(additional_params),
        };
        self
    }

    /// **Replaces** provider-specific parameters (or clears them with `None`).
    pub fn additional_params_opt(mut self, additional_params: Option<serde_json::Value>) -> Self {
        self.additional_params = additional_params;
        self
    }

    /// Sets the sampling temperature.
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets or clears the sampling temperature.
    pub fn temperature_opt(mut self, temperature: Option<f64>) -> Self {
        self.temperature = temperature;
        self
    }

    /// Sets the maximum tokens. Required by Anthropic.
    pub fn max_tokens(mut self, max_tokens: u64) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Sets or clears the maximum tokens.
    pub fn max_tokens_opt(mut self, max_tokens: Option<u64>) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Sets whether the model must call a tool.
    pub fn tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    /// Sets or clears the tool-choice policy.
    pub fn tool_choice_opt(mut self, tool_choice: Option<ToolChoice>) -> Self {
        self.tool_choice = tool_choice;
        self
    }

    /// Sets the JSON Schema constraining structured output.
    pub fn output_schema(mut self, schema: schemars::Schema) -> Self {
        self.output_schema = Some(schema);
        self
    }

    /// Sets or clears the structured-output schema.
    pub fn output_schema_opt(mut self, schema: Option<schemars::Schema>) -> Self {
        self.output_schema = schema;
        self
    }

    /// Opts in or out of recording sensitive content on GenAI telemetry spans.
    ///
    /// Defaults to `false`. Enabling this can expose prompts, retrieved
    /// context, tool results, and model responses through span attributes.
    pub fn record_content_telemetry(mut self, enabled: bool) -> Self {
        self.record_telemetry_content = enabled;
        self
    }

    /// The normalized input messages runtime telemetry would record, without
    /// consuming the builder.
    pub fn messages_for_telemetry(&self) -> Vec<Message> {
        let mut chat_history = self.chat_history.clone();
        if let Some(preamble) = &self.preamble {
            chat_history.insert(0, Message::system(preamble.clone()));
        }
        chat_history.push(self.prompt.clone());

        if let Some(documents) = CompletionRequest::normalized_documents_from(&self.documents) {
            let insert_at = chat_history
                .iter()
                .position(|message| !matches!(message, Message::System { .. }))
                .unwrap_or(chat_history.len());
            chat_history.insert(insert_at, documents);
        }

        chat_history
    }

    /// Builds the concrete [`CompletionRequest`].
    ///
    /// Any preamble becomes a leading [`Message::System`] and the request's
    /// legacy `preamble` field stays `None`; history keeps its order with the
    /// prompt last; provider tools merge into `additional_params.tools`.
    pub fn build(self) -> CompletionRequest {
        let mut chat_history = self.chat_history;
        let prompt = self.prompt;
        if let Some(preamble) = self.preamble {
            chat_history.insert(0, Message::system(preamble));
        }

        chat_history.push(prompt.clone());

        let chat_history =
            OneOrMany::from_iter_optional(chat_history).unwrap_or_else(|| OneOrMany::one(prompt));
        let additional_params = merge_provider_tools_into_additional_params(
            self.additional_params,
            self.provider_tools,
        );

        CompletionRequest {
            model: self.request_model,
            preamble: None,
            chat_history,
            documents: self.documents,
            tools: self.tools,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            tool_choice: self.tool_choice,
            additional_params,
            output_schema: self.output_schema,
            record_telemetry_content: self.record_telemetry_content,
        }
    }
}

#[cfg(test)]
mod builder_equivalence_tests {
    use super::*;

    fn tool(name: &str) -> ToolDefinition {
        ToolDefinition {
            name: name.to_string(),
            description: format!("{name} description"),
            parameters: serde_json::json!({ "type": "object" }),
        }
    }

    #[test]
    fn build_matches_explicit_construction_for_the_default_shape() {
        let built = CompletionRequest::builder("prompt").build();
        let explicit = CompletionRequest::from_prompt("prompt");

        // `from_prompt` leaves the legacy preamble field `None` too, so with no
        // preamble the two constructions agree field for field.
        assert_eq!(built.chat_history, explicit.chat_history);
        assert_eq!(built.preamble, explicit.preamble);
        assert_eq!(built.model, explicit.model);
        assert_eq!(built.documents, explicit.documents);
        assert_eq!(built.tools, explicit.tools);
        assert_eq!(built.temperature, explicit.temperature);
        assert_eq!(built.max_tokens, explicit.max_tokens);
        assert_eq!(built.tool_choice, explicit.tool_choice);
        assert_eq!(built.additional_params, explicit.additional_params);
        assert!(built.output_schema.is_none());
        assert!(!built.record_telemetry_content);
    }

    #[test]
    fn preamble_becomes_a_leading_system_message_not_the_legacy_field() {
        let request = CompletionRequest::builder("prompt")
            .preamble("be terse")
            .build();

        let messages: Vec<Message> = request.chat_history.clone().into_iter().collect();
        assert_eq!(messages.len(), 2);
        assert!(matches!(messages[0], Message::System { .. }));
        assert_eq!(messages[1], Message::from("prompt"));
        // The legacy field stays empty: the system message is canonical.
        assert!(request.preamble.is_none());
    }

    #[test]
    fn history_precedes_the_prompt_and_keeps_its_order() {
        let request = CompletionRequest::builder("prompt")
            .message(Message::from("first"))
            .messages([Message::from("second"), Message::from("third")])
            .preamble("system")
            .build();

        let messages: Vec<Message> = request.chat_history.into_iter().collect();
        assert!(matches!(messages[0], Message::System { .. }));
        assert_eq!(messages[1], Message::from("first"));
        assert_eq!(messages[2], Message::from("second"));
        assert_eq!(messages[3], Message::from("third"));
        assert_eq!(messages[4], Message::from("prompt"));
    }

    #[test]
    fn without_preamble_clears_a_previously_set_preamble() {
        let request = CompletionRequest::builder("prompt")
            .preamble("discarded")
            .without_preamble()
            .build();

        assert_eq!(request.chat_history.len(), 1);
        assert!(request.preamble.is_none());
    }

    #[test]
    fn additional_params_merges_while_the_opt_setter_replaces() {
        let merged = CompletionRequest::builder("p")
            .additional_params(serde_json::json!({ "a": 1 }))
            .additional_params(serde_json::json!({ "b": 2 }))
            .build();
        assert_eq!(
            merged.additional_params,
            Some(serde_json::json!({ "a": 1, "b": 2 }))
        );

        let replaced = CompletionRequest::builder("p")
            .additional_params(serde_json::json!({ "a": 1 }))
            .additional_params_opt(Some(serde_json::json!({ "b": 2 })))
            .build();
        assert_eq!(
            replaced.additional_params,
            Some(serde_json::json!({ "b": 2 }))
        );

        let cleared = CompletionRequest::builder("p")
            .additional_params(serde_json::json!({ "a": 1 }))
            .additional_params_opt(None)
            .build();
        assert_eq!(cleared.additional_params, None);
    }

    #[test]
    fn optional_setters_clear_their_field() {
        let request = CompletionRequest::builder("p")
            .temperature(0.5)
            .temperature_opt(None)
            .max_tokens(10)
            .max_tokens_opt(None)
            .model("m")
            .model_opt(None)
            .output_schema_opt(None)
            .build();

        assert_eq!(request.temperature, None);
        assert_eq!(request.max_tokens, None);
        assert_eq!(request.model, None);
        assert!(request.output_schema.is_none());
    }

    #[test]
    fn tools_and_documents_accumulate_in_order() {
        let request = CompletionRequest::builder("p")
            .tool(tool("one"))
            .tools([tool("two"), tool("three")])
            .document(Document {
                id: "a".into(),
                text: "first".into(),
                additional_props: HashMap::new(),
            })
            .documents([Document {
                id: "b".into(),
                text: "second".into(),
                additional_props: HashMap::new(),
            }])
            .build();

        let names: Vec<&str> = request.tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["one", "two", "three"]);
        let ids: Vec<&str> = request.documents.iter().map(|d| d.id.as_str()).collect();
        assert_eq!(ids, ["a", "b"]);
    }

    #[test]
    fn provider_tools_merge_into_additional_params_like_the_inherent_setters() {
        let via_builder = CompletionRequest::builder("p")
            .additional_params(serde_json::json!({ "keep": true }))
            .provider_tool(ProviderToolDefinition::new("web_search"))
            .provider_tools([ProviderToolDefinition::new("code_interpreter")])
            .build();

        let via_inherent = CompletionRequest {
            additional_params: Some(serde_json::json!({ "keep": true })),
            ..CompletionRequest::from_prompt("p")
        }
        .with_provider_tool(ProviderToolDefinition::new("web_search"))
        .with_provider_tools(vec![ProviderToolDefinition::new("code_interpreter")]);

        assert_eq!(
            via_builder.additional_params,
            via_inherent.additional_params
        );
        // And the merge really did land under `tools`, preserving `keep`.
        let params = via_builder.additional_params.expect("params present");
        assert_eq!(params["keep"], serde_json::json!(true));
        assert_eq!(params["tools"].as_array().expect("tools array").len(), 2);
    }

    #[test]
    fn messages_for_telemetry_matches_the_built_requests_view() {
        let builder = CompletionRequest::builder("prompt")
            .preamble("system")
            .message(Message::from("history"))
            .document(Document {
                id: "d".into(),
                text: "doc".into(),
                additional_props: HashMap::new(),
            });

        let from_builder = builder.clone().messages_for_telemetry();
        let from_request = builder.build().messages_for_telemetry();

        assert_eq!(from_builder, from_request);
        // Documents land after the system message, before the first non-system one.
        assert!(matches!(from_builder[0], Message::System { .. }));
    }

    #[test]
    fn scalar_setters_reach_the_built_request() {
        let schema = schemars::schema_for!(String);
        let request = CompletionRequest::builder("p")
            .model("gpt-4o")
            .temperature(0.25)
            .max_tokens(512)
            .tool_choice(ToolChoice::Auto)
            .output_schema(schema)
            .record_content_telemetry(true)
            .build();

        assert_eq!(request.model.as_deref(), Some("gpt-4o"));
        assert_eq!(request.temperature, Some(0.25));
        assert_eq!(request.max_tokens, Some(512));
        assert_eq!(request.tool_choice, Some(ToolChoice::Auto));
        assert!(request.output_schema.is_some());
        assert!(request.record_telemetry_content);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn usage_has_values_reflects_the_zero_sentinel() {
        use super::Usage;

        assert!(!Usage::new().has_values());

        let mut usage = Usage::new();
        usage.reasoning_tokens = 1;
        assert!(usage.has_values());
    }

    use super::*;

    #[test]
    fn completion_request_content_telemetry_is_opt_in_and_not_serialized() {
        let default_request = CompletionRequest::from_prompt("completion prompt");
        assert!(!default_request.record_telemetry_content);

        let default_json = serde_json::to_value(&default_request).expect("serialize request");
        assert!(
            default_json.get("record_telemetry_content").is_none(),
            "safe default should not serialize the telemetry opt-in field"
        );
        let default_roundtrip: CompletionRequest =
            serde_json::from_value(default_json).expect("deserialize default request");
        assert!(!default_roundtrip.record_telemetry_content);

        let opt_in_request = CompletionRequest {
            record_telemetry_content: true,
            ..CompletionRequest::from_prompt("completion prompt")
        };
        assert!(opt_in_request.record_telemetry_content);

        let opt_in_json = serde_json::to_value(&opt_in_request).expect("serialize opt-in request");
        assert!(
            opt_in_json.get("record_telemetry_content").is_none(),
            "local telemetry policy must not be serialized into provider requests"
        );
        let legacy_roundtrip: CompletionRequest =
            serde_json::from_value(opt_in_json).expect("deserialize legacy request");
        assert!(
            !legacy_roundtrip.record_telemetry_content,
            "missing field should deserialize to the safe default"
        );
    }

    fn test_document(id: &str, text: &str) -> Document {
        Document {
            id: id.to_string(),
            text: text.to_string(),
            additional_props: HashMap::new(),
        }
    }

    #[test]
    fn message_telemetry_includes_normalized_documents() {
        let request = CompletionRequest::builder("prompt")
            .messages(vec![Message::system("system"), Message::user("history")])
            .documents(vec![test_document("doc1", "static context secret")])
            .build();

        let messages = request.messages_for_telemetry();
        assert_eq!(messages.len(), 4);
        assert!(matches!(messages[0], Message::System { .. }));
        assert!(is_document_message(&messages[1], "doc1"));
        assert!(matches!(
            &messages[2],
            Message::User { content }
                if matches!(content.first(), UserContent::Text(text) if text.text == "history")
        ));
        assert!(matches!(
            &messages[3],
            Message::User { content }
                if matches!(content.first(), UserContent::Text(text) if text.text == "prompt")
        ));

        assert_eq!(messages, request.chat_history_with_documents());
    }

    fn is_document_message(message: &Message, expected_id: &str) -> bool {
        let Message::User { content } = message else {
            return false;
        };

        content.iter().any(|content| {
            matches!(
                content,
                UserContent::Document(document)
                    if document.data.to_string().contains(&format!("<file id: {expected_id}>"))
            )
        })
    }

    #[test]
    fn test_document_display_without_metadata() {
        let doc = Document {
            id: "123".to_string(),
            text: "This is a test document.".to_string(),
            additional_props: HashMap::new(),
        };

        let expected = "<file id: 123>\nThis is a test document.\n</file>\n";
        assert_eq!(format!("{doc}"), expected);
    }

    #[test]
    fn test_document_display_with_metadata() {
        let mut additional_props = HashMap::new();
        additional_props.insert("author".to_string(), "John Doe".to_string());
        additional_props.insert("length".to_string(), "42".to_string());

        let doc = Document {
            id: "123".to_string(),
            text: "This is a test document.".to_string(),
            additional_props,
        };

        let expected = concat!(
            "<file id: 123>\n",
            "<metadata author: \"John Doe\" length: \"42\" />\n",
            "This is a test document.\n",
            "</file>\n"
        );
        assert_eq!(format!("{doc}"), expected);
    }

    #[test]
    fn test_normalize_documents_with_documents() {
        let doc1 = Document {
            id: "doc1".to_string(),
            text: "Document 1 text.".to_string(),
            additional_props: HashMap::new(),
        };

        let doc2 = Document {
            id: "doc2".to_string(),
            text: "Document 2 text.".to_string(),
            additional_props: HashMap::new(),
        };

        let request = CompletionRequest {
            model: None,
            preamble: None,
            chat_history: OneOrMany::one("What is the capital of France?".into()),
            documents: vec![doc1, doc2],
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        };

        let expected = Message::User {
            content: OneOrMany::many(vec![
                UserContent::document(
                    "<file id: doc1>\nDocument 1 text.\n</file>\n".to_string(),
                    Some(DocumentMediaType::TXT),
                ),
                UserContent::document(
                    "<file id: doc2>\nDocument 2 text.\n</file>\n".to_string(),
                    Some(DocumentMediaType::TXT),
                ),
            ])
            .expect("There will be at least one document"),
        };

        assert_eq!(request.normalized_documents(), Some(expected));
    }

    #[test]
    fn test_normalize_documents_without_documents() {
        let request = CompletionRequest {
            model: None,
            preamble: None,
            chat_history: OneOrMany::one("What is the capital of France?".into()),
            documents: Vec::new(),
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        };

        assert_eq!(request.normalized_documents(), None);
    }

    #[test]
    fn chat_history_with_documents_places_documents_after_leading_system_messages() {
        let request = CompletionRequest {
            model: None,
            preamble: None,
            chat_history: OneOrMany::many(vec![
                Message::system("System prompt"),
                Message::assistant("Earlier assistant turn"),
                Message::user("Earlier user turn"),
                Message::user("Prompt"),
            ])
            .unwrap(),
            documents: vec![test_document("doc1", "Document text.")],
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        };

        assert_eq!(request.documents.len(), 1);

        let history = request.chat_history_with_documents();
        let history = history.iter().collect::<Vec<_>>();
        assert_eq!(history.len(), 5);
        assert!(matches!(history[0], Message::System { .. }));
        assert!(is_document_message(history[1], "doc1"));
        assert!(matches!(history[2], Message::Assistant { .. }));
        assert!(matches!(history[3], Message::User { .. }));
        assert!(matches!(history[4], Message::User { .. }));
    }

    #[test]
    fn chat_history_with_documents_places_documents_before_mid_conversation_system_messages() {
        let request = CompletionRequest {
            model: None,
            preamble: None,
            chat_history: OneOrMany::many(vec![
                Message::system("Leading system prompt"),
                Message::assistant("Earlier assistant turn"),
                Message::system("Mid-conversation instruction"),
                Message::user("Prompt"),
            ])
            .unwrap(),
            documents: vec![test_document("doc1", "Document text.")],
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        };

        let history = request.chat_history_with_documents();
        let history = history.iter().collect::<Vec<_>>();
        assert_eq!(history.len(), 5);
        assert!(matches!(
            history[0],
            Message::System { content } if content == "Leading system prompt"
        ));
        assert!(is_document_message(history[1], "doc1"));
        assert!(matches!(history[2], Message::Assistant { .. }));
        assert!(matches!(
            history[3],
            Message::System { content } if content == "Mid-conversation instruction"
        ));
        assert!(matches!(history[4], Message::User { .. }));
    }

    #[test]
    fn chat_history_with_documents_does_not_duplicate_documents() {
        let request = CompletionRequest {
            model: None,
            preamble: None,
            chat_history: OneOrMany::many(vec![
                Message::system("System prompt"),
                Message::user("Earlier user turn"),
                Message::assistant("Earlier assistant turn"),
                Message::user("Prompt"),
            ])
            .unwrap(),
            documents: vec![test_document("doc1", "Document text.")],
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        };

        let history = request.chat_history_with_documents();
        let document_messages = history
            .iter()
            .filter(|message| is_document_message(message, "doc1"))
            .count();
        assert_eq!(document_messages, 1);
    }

    #[test]
    fn completion_error_provider_response_helpers_with_preserved_json_body() {
        let body = r#"{"error":{"code":"rate_limit","message":"slow down"}}"#;
        let error = CompletionError::ProviderResponse(provider_response::ProviderResponseError {
            status: None,
            body: body.to_string(),
        });

        assert_eq!(error.provider_response_body(), Some(body));
        assert_eq!(error.provider_response_status(), None);
        assert_eq!(
            error
                .provider_response_json()
                .expect("fixture body should parse as valid JSON"),
            Some(serde_json::json!({
                "error": {
                    "code": "rate_limit",
                    "message": "slow down"
                }
            }))
        );
    }

    #[test]
    fn completion_error_provider_response_helpers_with_preserved_status() {
        let body = r#"{"error":{"message":"too many requests"}}"#;
        let error = CompletionError::ProviderResponse(provider_response::ProviderResponseError {
            status: Some(http::StatusCode::TOO_MANY_REQUESTS),
            body: body.to_string(),
        });

        assert_eq!(error.provider_response_body(), Some(body));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::TOO_MANY_REQUESTS)
        );
    }

    #[test]
    fn completion_error_provider_response_helpers_with_preserved_plain_text_body() {
        let error = CompletionError::ProviderResponse(provider_response::ProviderResponseError {
            status: None,
            body: "provider exploded".to_string(),
        });

        assert_eq!(error.provider_response_body(), Some("provider exploded"));
        assert_eq!(error.provider_response_status(), None);
        assert!(error.provider_response_json().is_err());
    }

    #[test]
    fn completion_error_provider_error_is_not_a_provider_response() {
        // `ProviderError` also carries Rig-generated diagnostics, so the helpers
        // must not report its string as a provider response body.
        let error = CompletionError::ProviderError("stream transport failed".to_string());

        assert_eq!(error.provider_response_body(), None);
        assert_eq!(error.provider_response_status(), None);
        assert_eq!(
            error
                .provider_response_json()
                .expect("no body is not an error"),
            None
        );
    }

    #[test]
    fn completion_error_provider_response_helpers_with_http_non_success_body_and_status() {
        let body = r#"{"error":{"type":"invalid_request","message":"bad request"}}"#;
        let error = CompletionError::HttpError(http_client::Error::InvalidStatusCodeWithMessage(
            http::StatusCode::BAD_REQUEST,
            body.to_string(),
        ));

        assert_eq!(error.provider_response_body(), Some(body));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::BAD_REQUEST)
        );
        assert_eq!(
            error.provider_response_json().expect("valid JSON body"),
            Some(serde_json::json!({
                "error": {
                    "type": "invalid_request",
                    "message": "bad request"
                }
            }))
        );
    }

    #[test]
    fn completion_error_provider_response_helpers_with_unrelated_variant() {
        let error = CompletionError::ResponseError("failed to parse provider response".to_string());

        assert_eq!(error.provider_response_body(), None);
        assert_eq!(error.provider_response_status(), None);
        assert_eq!(
            error
                .provider_response_json()
                .expect("no body is not an error"),
            None
        );
    }

    #[test]
    fn provider_response_json_returns_none_for_empty_preserved_body() {
        let error = CompletionError::ProviderResponse(provider_response::ProviderResponseError {
            status: None,
            body: String::new(),
        });

        assert_eq!(error.provider_response_body(), Some(""));
        assert_eq!(
            error
                .provider_response_json()
                .expect("empty body is not a JSON parse error"),
            None
        );
    }
}
