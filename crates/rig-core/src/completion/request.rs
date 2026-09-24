//! Completion requests, normalized responses, and provider model contracts.
//!
//! ```
//! use rig_core::completion::CompletionRequestBuilder;
//!
//! let request = CompletionRequestBuilder::unbound("Who are you?")
//!     .preamble("You are a concise assistant.".to_owned())
//!     .temperature(0.5)
//!     .build();
//! assert_eq!(request.temperature, Some(0.5));
//! ```

use super::message::{AssistantContent, DocumentMediaType};
use crate::error::ProviderError;
use crate::id::{MessageId, ModelName, ProviderName, RequestId, ResponseId};
use crate::message::ToolChoice;
use crate::non_empty::NonEmpty;
use crate::response::ResponseMeta;
use crate::streaming::StreamingCompletionResponse;
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use crate::{
    json_utils,
    message::{Message, UserContent},
};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ops::{Add, AddAssign};

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

/// Normalized generation ending. Unmapped provider values remain in [`Self::Other`].
/// Failure statuses may accompany parseable output; callers must decide whether
/// such output is usable rather than treating every response as successful.
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
    /// A provider-specific reason outside the normalized vocabulary, carried
    /// verbatim in the provider's own wire spelling.
    Other(String),
}

impl FinishReason {
    /// Changes [`Self::Stop`] to [`Self::ToolCalls`] when output contains a tool
    /// call. All other reasons remain unchanged. The fold that builds a
    /// response applies this reconciliation.
    pub fn reconcile_with_output(self, has_tool_call: bool) -> Self {
        if has_tool_call && matches!(self, Self::Stop) {
            Self::ToolCalls
        } else {
            self
        }
    }

    /// Returns whether the reason is [`Self::Length`] or [`Self::ContentFilter`].
    /// These reasons permit answerless turns without treating absent content as
    /// a malformed response. Unknown reasons are not classified as truncation.
    pub fn truncated_output(&self) -> bool {
        matches!(self, Self::Length | Self::ContentFilter)
    }

    /// Formats an answerless-turn diagnostic with budget or filtering advice
    /// for known truncation reasons, and a generic explanation otherwise.
    pub fn no_answer_message(&self) -> String {
        let remedy = match self {
            Self::Length => {
                "the turn ran out of output budget before producing one — \
                 raise max_tokens for this request"
            }
            Self::ContentFilter => {
                "the provider filtered the response — the content, not the \
                 budget, is what it objected to"
            }
            _ => "the turn ended before producing one",
        };
        format!(
            "the model produced no answer and stopped with \
             finish_reason={self:?}; {remedy}"
        )
    }
}

/// A completion: the assistant content and how the reply ended. The choice
/// may be empty, including for truncated or filtered turns.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletionResponse {
    /// Assistant content returned by the provider, possibly empty.
    pub choice: Vec<AssistantContent>,
    /// The terminal record: why the reply ended and the provider's metadata.
    pub end: CompletionEnd,
}

/// How a completion ended: the terminal record a stream closes with, and the
/// part of a [`CompletionResponse`] beyond its content. Content never rides
/// here; it arrives as block events or in [`CompletionResponse::choice`].
///
/// `meta.raw` has two meanings. For a buffered reply it is the whole reply
/// document; for a stream it is the terminal event's document, serialized
/// from the decoder's parsed terminal type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletionEnd {
    /// Why the model stopped, when the provider said. The fold that builds a
    /// response reconciles it with the tool calls the reply carried
    /// ([`FinishReason::reconcile_with_output`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    /// The assistant message id, suitable for replay in
    /// [`Message::Assistant`]. A message block's id takes precedence over
    /// the terminal record's in a folded response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message_id: Option<MessageId>,
    /// The service whose reasoning a stream carries, when it is not
    /// `meta.provider`: a transport or deployment of another provider's
    /// models. The consumer of the stream records it as its reasoning's
    /// issuer ([`crate::message::Reasoning::provider`]). A folded
    /// [`CompletionResponse`] has applied it to every reasoning part and
    /// leaves it `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_issuer: Option<ProviderName>,
    /// The provider's metadata.
    pub meta: ResponseMeta,
}

impl CompletionEnd {
    /// A record carrying the provider's metadata and nothing else.
    pub fn new(meta: ResponseMeta) -> Self {
        Self {
            finish_reason: None,
            message_id: None,
            reasoning_issuer: None,
            meta,
        }
    }

    /// The issuer this reply's reasoning records: [`Self::reasoning_issuer`]
    /// when set, otherwise the provider.
    pub fn issuer(&self) -> &ProviderName {
        self.reasoning_issuer
            .as_ref()
            .unwrap_or(&self.meta.provider)
    }

    /// What this record tells its operation's span.
    pub fn recorded(&self) -> crate::telemetry::Recorded<'_> {
        crate::telemetry::Recorded {
            meta: &self.meta,
            message_id: self.message_id.as_ref(),
        }
    }
}

/// What a decoder read off a completion reply's terminal record. The driver
/// completes it into a [`CompletionEnd`] with the provider and the transport
/// request id.
#[derive(Debug, Clone, PartialEq)]
pub struct ReportedEnd {
    /// Why the model stopped, when the reply said.
    pub finish_reason: Option<FinishReason>,
    /// The message id the terminal record names.
    pub message_id: Option<MessageId>,
    /// The issuer of the reply's reasoning, when it is not the provider.
    pub reasoning_issuer: Option<ProviderName>,
    /// The model the reply names.
    pub model: Option<ModelName>,
    /// The response id the reply names.
    pub response_id: Option<ResponseId>,
    /// The tokens the reply reports.
    pub usage: Usage,
    /// The terminal record's document, serialized from the decoder's parsed
    /// terminal type.
    pub raw: serde_json::Value,
}

impl ReportedEnd {
    /// A terminal record reporting `usage` and its document `raw`, and
    /// nothing else.
    pub fn new(usage: Usage, raw: serde_json::Value) -> Self {
        Self {
            finish_reason: None,
            message_id: None,
            reasoning_issuer: None,
            model: None,
            response_id: None,
            usage,
            raw,
        }
    }

    /// The terminal record, with the driver's provider and transport request
    /// id.
    pub fn complete(
        self,
        provider: ProviderName,
        provider_request_id: Option<RequestId>,
    ) -> CompletionEnd {
        CompletionEnd {
            finish_reason: self.finish_reason,
            message_id: self.message_id,
            reasoning_issuer: self.reasoning_issuer,
            meta: ResponseMeta {
                provider,
                model: self.model,
                response_id: self.response_id,
                provider_request_id,
                usage: self.usage,
                raw: self.raw,
            },
        }
    }
}

/// The token usage a provider reported for one completion.
///
/// A counter the provider did not send is `None`; a reported zero is
/// `Some(0)`. Serialized as the same keys, absent when `None`.
#[derive(Debug, Default, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub struct Usage {
    /// The number of input ("prompt") tokens used in a given request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    /// The number of output ("completion") tokens used in a given request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    /// We store this separately as some providers may only report one number
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u64>,
    /// The number of input tokens read from a provider-managed cache
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_input_tokens: Option<u64>,
    /// The number of input tokens written to a provider-managed cache
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u64>,
    /// The number of tool-use prompt tokens used in a given request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_use_prompt_tokens: Option<u64>,
    /// The number of tokens spent on internal reasoning / "thoughts" by reasoning-capable
    /// models (e.g. Gemini thinking, Anthropic extended thinking, OpenAI o-series).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u64>,
}

impl Usage {
    /// Whether the provider reported any counter at all.
    pub fn is_reported(&self) -> bool {
        *self != Self::default()
    }
}

/// Sum two counters where an unreported side does not turn a reported one
/// into "unreported".
fn add_counter(lhs: Option<u64>, rhs: Option<u64>) -> Option<u64> {
    match (lhs, rhs) {
        (None, None) => None,
        (lhs, rhs) => Some(lhs.unwrap_or(0) + rhs.unwrap_or(0)),
    }
}

impl Add for Usage {
    type Output = Self;

    fn add(mut self, other: Self) -> Self::Output {
        self += other;
        self
    }
}

impl AddAssign for Usage {
    fn add_assign(&mut self, other: Self) {
        self.input_tokens = add_counter(self.input_tokens, other.input_tokens);
        self.output_tokens = add_counter(self.output_tokens, other.output_tokens);
        self.total_tokens = add_counter(self.total_tokens, other.total_tokens);
        self.cached_input_tokens = add_counter(self.cached_input_tokens, other.cached_input_tokens);
        self.cache_creation_input_tokens = add_counter(
            self.cache_creation_input_tokens,
            other.cache_creation_input_tokens,
        );
        self.tool_use_prompt_tokens =
            add_counter(self.tool_use_prompt_tokens, other.tool_use_prompt_tokens);
        self.reasoning_tokens = add_counter(self.reasoning_tokens, other.reasoning_tokens);
    }
}

/// Model capabilities used by runtimes when preparing requests.
/// Defaults are conservative; construct through [`Self::new`] and setters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ProviderCapabilities {
    /// Whether native structured output can remain enabled with tool calls
    /// without suppressing them. Defaults to `false`.
    pub composes_native_output_with_tools: bool,
}

impl ProviderCapabilities {
    /// Create the conservative capability set used by default.
    pub const fn new() -> Self {
        Self {
            composes_native_output_with_tools: false,
        }
    }

    /// Declare whether native structured output composes with tool calls.
    pub const fn with_native_output_tool_composition(mut self, supported: bool) -> Self {
        self.composes_native_output_with_tools = supported;
        self
    }
}

/// Generates buffered or streamed normalized completions. Provider-specific
/// response data belongs in [`CompletionResponse::raw`]. Only
/// [`Self::completion_request`] requires cloning; `Arc<M>` can share a model.
pub trait CompletionModel: WasmCompatSend + WasmCompatSync {
    /// Generates a completion response for the given completion request.
    fn completion(
        &self,
        request: CompletionRequest,
    ) -> impl std::future::Future<Output = Result<CompletionResponse, ProviderError>> + WasmCompatSend;

    /// Streams a completion response for the given completion request.
    fn stream(
        &self,
        request: CompletionRequest,
    ) -> impl std::future::Future<Output = Result<StreamingCompletionResponse, ProviderError>>
    + WasmCompatSend;

    /// Generates a completion with optional execution-local observation.
    /// The default delegates without observations. Forwarding wrappers must
    /// preserve the context to retain per-call identity across retries and tasks.
    fn completion_with_context(
        &self,
        request: CompletionRequest,
        _context: Option<crate::observe::AdapterContext>,
    ) -> impl std::future::Future<Output = Result<CompletionResponse, ProviderError>> + WasmCompatSend
    {
        self.completion(request)
    }

    /// Optionally observe a stream, retaining context through lazy startup and drop.
    /// The default delegates to [`Self::stream`] without provider observations.
    fn stream_with_context(
        &self,
        request: CompletionRequest,
        _context: Option<crate::observe::AdapterContext>,
    ) -> impl std::future::Future<Output = Result<StreamingCompletionResponse, ProviderError>>
    + WasmCompatSend {
        self.stream(request)
    }

    /// Generates a completion request builder for the given `prompt`.
    fn completion_request(&self, prompt: impl Into<Message>) -> CompletionRequestBuilder<Self>
    where
        Self: Sized + Clone,
    {
        CompletionRequestBuilder::new(self.clone(), prompt)
    }

    /// Provider behavior a runtime should account for when preparing requests.
    ///
    /// The default is conservative; see [`ProviderCapabilities`]. Override
    /// this to declare the capabilities a provider actually supports.
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::default()
    }
}

/// Forwards model operations through shared ownership. Request builders clone
/// the `Arc`, not the underlying model.
impl<M: CompletionModel + ?Sized> CompletionModel for std::sync::Arc<M> {
    fn completion(
        &self,
        request: CompletionRequest,
    ) -> impl std::future::Future<Output = Result<CompletionResponse, ProviderError>> + WasmCompatSend
    {
        (**self).completion(request)
    }

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> impl std::future::Future<Output = Result<StreamingCompletionResponse, ProviderError>>
    + WasmCompatSend {
        (**self).stream(request)
    }

    fn completion_with_context(
        &self,
        request: CompletionRequest,
        context: Option<crate::observe::AdapterContext>,
    ) -> impl std::future::Future<Output = Result<CompletionResponse, ProviderError>> + WasmCompatSend
    {
        (**self).completion_with_context(request, context)
    }

    fn stream_with_context(
        &self,
        request: CompletionRequest,
        context: Option<crate::observe::AdapterContext>,
    ) -> impl std::future::Future<Output = Result<StreamingCompletionResponse, ProviderError>>
    + WasmCompatSend {
        (**self).stream_with_context(request, context)
    }

    fn capabilities(&self) -> ProviderCapabilities {
        (**self).capabilities()
    }
}

/// A completion request: the system prompt, the conversation, and what the
/// provider needs to answer it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    /// Optional model override for this request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// The system prompt. A wire that takes system messages renders it as the
    /// leading one ([`Self::history`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// The conversation, ending with the prompt. A [`Message::System`] here is
    /// a system message inside the conversation, not the system prompt.
    pub messages: NonEmpty<Message>,
    /// The documents to be sent to the completion model provider
    pub documents: Vec<Document>,
    /// The tools to be sent to the completion model provider
    pub tools: Vec<ToolDefinition>,
    /// The temperature to be sent to the completion model provider
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// The max tokens to be sent to the completion model provider
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    /// Whether tools are required to be used by the model provider or not before providing a response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    /// Additional provider-specific parameters to be sent to the completion model provider
    #[serde(skip_serializing_if = "Option::is_none")]
    pub additional_params: Option<serde_json::Value>,
    /// Optional JSON Schema for structured output. When set, providers that support
    /// native structured outputs will constrain the model's response to match this schema.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_schema: Option<schemars::Schema>,
    /// Opt-in for sensitive request, response, and tool-content telemetry.
    /// Defaults to `false` and is excluded from serialization. Enabling it can
    /// expose prompts, context, tool results, and model output in span attributes
    /// and increase telemetry storage costs. Requires explicit caller consent.
    /// Agent drivers record normalized content; direct provider coverage varies,
    /// especially for streams consumed after the provider returns.
    #[serde(skip)]
    pub record_telemetry_content: bool,
}

impl CompletionRequest {
    /// The system instructions of this request.
    pub fn system_instructions(&self) -> Option<&str> {
        self.system.as_deref()
    }

    /// Drop reasoning none of `issuers` issued from the conversation; see
    /// [`crate::message::retain_replayable_reasoning`]. A conversation that
    /// held nothing else is left as it is, since there is none without it.
    pub fn retain_replayable_reasoning(&mut self, issuers: &[&str]) {
        let mut messages = self.messages.to_vec();
        crate::message::retain_replayable_reasoning(&mut messages, issuers);
        if let Some(messages) = NonEmpty::from_vec(messages) {
            self.messages = messages;
        }
    }

    /// The conversation as a wire renders it: the system prompt as the
    /// leading [`Message::System`], then [`Self::messages`].
    pub fn history(&self) -> Vec<Message> {
        self.system
            .iter()
            .map(|system| Message::system(system.clone()))
            .chain(self.messages.iter().cloned())
            .collect()
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

        let messages = documents
            .iter()
            .map(|doc| UserContent::document(doc.to_string(), Some(DocumentMediaType::TXT)))
            .collect::<Vec<_>>();

        NonEmpty::from_vec(messages).map(|content| Message::User { content })
    }

    /// [`Self::history`] with the documents as one user message after the
    /// leading system messages: how a wire without native documents sends
    /// them.
    pub fn history_with_documents(&self) -> Vec<Message> {
        let mut chat_history = self.history();
        if let Some(documents) = self.normalized_documents() {
            insert_after_leading_system(&mut chat_history, documents);
        }
        chat_history
    }
}

/// Insert `message` at the first non-system position so document context lands
/// after any leading system messages; telemetry and the sent request must
/// agree on this placement.
fn insert_after_leading_system(chat_history: &mut Vec<Message>, message: Message) {
    let insert_at = chat_history
        .iter()
        .position(|message| !matches!(message, Message::System { .. }))
        .unwrap_or(chat_history.len());
    chat_history.insert(insert_at, message);
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

/// Builds completion requests, optionally retaining a model for dispatch.
///
/// ```no_run
/// use rig_core::completion::{CompletionModel, CompletionRequestBuilder};
///
/// # async fn run(model: impl CompletionModel) -> Result<(), Box<dyn std::error::Error>> {
/// let response = CompletionRequestBuilder::new(model, "Who are you?")
///     .temperature(0.5)
///     .send()
///     .await?;
/// # let _ = response;
/// # Ok(())
/// # }
/// ```
#[must_use = "a request builder does nothing until built or sent"]
pub struct CompletionRequestBuilder<M = Unbound> {
    model: M,
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

/// The model slot of a request under assembly that has no model attached:
/// the request is built with [`CompletionRequestBuilder::build`] and
/// dispatched elsewhere (an agent dispatches it through its bus).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Unbound;

impl CompletionRequestBuilder<Unbound> {
    /// A builder with no model attached; `build` produces the request.
    pub fn unbound(prompt: impl Into<Message>) -> Self {
        Self::new(Unbound, prompt)
    }
}

impl<M> CompletionRequestBuilder<M> {
    pub fn new(model: M, prompt: impl Into<Message>) -> Self {
        Self {
            model,
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

    /// Sets the preamble: the request's system prompt
    /// ([`CompletionRequest::system`]).
    pub fn preamble(mut self, preamble: String) -> Self {
        self.preamble = Some(preamble);
        self
    }

    /// Overrides the model used for this request.
    pub fn model<S: Into<String>>(mut self, model: impl Into<Option<S>>) -> Self {
        self.request_model = model.into().map(Into::into);
        self
    }

    /// Adds a message to the chat history for the completion request.
    pub fn message(mut self, message: Message) -> Self {
        self.chat_history.push(message);

        self
    }

    /// Adds a list of messages to the chat history for the completion request.
    pub fn messages(mut self, messages: impl IntoIterator<Item = Message>) -> Self {
        self.chat_history.extend(messages);

        self
    }

    /// Adds a document to the completion request.
    pub fn document(mut self, document: Document) -> Self {
        self.documents.push(document);
        self
    }

    /// Adds a list of documents to the completion request.
    pub fn documents(self, documents: impl IntoIterator<Item = Document>) -> Self {
        documents
            .into_iter()
            .fold(self, CompletionRequestBuilder::document)
    }

    /// Adds a tool to the completion request.
    pub fn tool(mut self, tool: ToolDefinition) -> Self {
        self.tools.push(tool);
        self
    }

    /// Adds a list of tools to the completion request.
    pub fn tools(self, tools: Vec<ToolDefinition>) -> Self {
        tools.into_iter().fold(self, CompletionRequestBuilder::tool)
    }

    /// Adds a provider-hosted tool to the completion request.
    pub fn provider_tool(mut self, tool: ProviderToolDefinition) -> Self {
        self.provider_tools.push(tool);
        self
    }

    /// Adds provider-hosted tools to the completion request.
    pub fn provider_tools(self, tools: Vec<ProviderToolDefinition>) -> Self {
        tools
            .into_iter()
            .fold(self, CompletionRequestBuilder::provider_tool)
    }

    /// Merges provider-specific parameters; `None` clears existing parameters.
    /// Provider conversion determines precedence over typed fields.
    /// [`Self::build`] warns about overlapping sampling, model, tool, and
    /// response-format keys without changing their values.
    pub fn additional_params(
        mut self,
        additional_params: impl Into<Option<serde_json::Value>>,
    ) -> Self {
        self.additional_params =
            json_utils::merge_params(self.additional_params.take(), additional_params.into());
        self
    }

    /// Sets (or, with `None`, clears) the temperature for the completion request.
    pub fn temperature(mut self, temperature: impl Into<Option<f64>>) -> Self {
        self.temperature = temperature.into();
        self
    }

    /// Sets the output-token limit, or clears it with `None`.
    /// Provider-specific defaults and requirements apply.
    pub fn max_tokens(mut self, max_tokens: impl Into<Option<u64>>) -> Self {
        self.max_tokens = max_tokens.into();
        self
    }

    /// Sets the tool-selection policy.
    pub fn tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    /// Sets a native structured-output schema for supporting providers.
    /// This does not deserialize the returned content. `None` clears the schema.
    pub fn output_schema(mut self, schema: impl Into<Option<schemars::Schema>>) -> Self {
        self.output_schema = schema.into();
        self
    }

    /// Sets the opt-in for sensitive content telemetry, disabled by default.
    /// See [`CompletionRequest::record_telemetry_content`] for exposure risks and
    /// provider coverage. Structural metadata and usage remain available when disabled.
    pub fn record_content_telemetry(mut self, enabled: bool) -> Self {
        self.record_telemetry_content = enabled;
        self
    }

    /// Returns the normalized input messages used by runtime telemetry.
    pub fn messages_for_telemetry(&self) -> Vec<Message> {
        let mut chat_history = self.chat_history.clone();
        if let Some(preamble) = &self.preamble {
            chat_history.insert(0, Message::system(preamble.clone()));
        }
        chat_history.push(self.prompt.clone());

        if let Some(documents) = CompletionRequest::normalized_documents_from(&self.documents) {
            insert_after_leading_system(&mut chat_history, documents);
        }

        chat_history
    }

    /// Builds the completion request.
    pub fn build(self) -> CompletionRequest {
        self.into_model_and_request().1
    }

    /// Moves out the model and constructs the request without cloning the model.
    fn into_model_and_request(self) -> (M, CompletionRequest) {
        let model = self.model;
        let (system, messages) = system_and_messages(self.preamble, self.chat_history, self.prompt);
        // Checked before provider tools are merged in: that merge writes a
        // `tools` key of its own, which is not a caller collision.
        for key in shadowed_typed_fields(
            self.additional_params.as_ref(),
            &[
                ("temperature", self.temperature.is_some()),
                ("max_tokens", self.max_tokens.is_some()),
                ("tool_choice", self.tool_choice.is_some()),
                ("model", self.request_model.is_some()),
                ("tools", !self.tools.is_empty()),
                ("response_format", self.output_schema.is_some()),
            ],
        ) {
            if matches!(key, "tools" | "response_format") {
                tracing::warn!(
                    key,
                    "additional_params also carries `{key}`; the provider decides how it combines with the typed field"
                );
            } else {
                tracing::warn!(
                    key,
                    "additional_params overrides the typed `{key}` field set on the same request"
                );
            }
        }
        let additional_params = merge_provider_tools_into_additional_params(
            self.additional_params,
            self.provider_tools,
        );

        let request = CompletionRequest {
            model: self.request_model,
            system,
            messages,
            documents: self.documents,
            tools: self.tools,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            tool_choice: self.tool_choice,
            additional_params,
            output_schema: self.output_schema,
            record_telemetry_content: self.record_telemetry_content,
        };
        (model, request)
    }
}

/// The system prompt and the conversation of a request built from a preamble,
/// a history and a prompt. Without a preamble, a leading system message of
/// the history is the system prompt, unless it is the conversation's only
/// message.
fn system_and_messages(
    preamble: Option<String>,
    history: Vec<Message>,
    prompt: Message,
) -> (Option<String>, NonEmpty<Message>) {
    let mut history = history.into_iter();
    let (system, first) = match (preamble, history.next()) {
        (Some(preamble), first) => (Some(preamble), first),
        (None, Some(Message::System { content })) => match history.next() {
            Some(next) => (Some(content), Some(next)),
            None => (Some(content), None),
        },
        (None, first) => (None, first),
    };
    (
        system,
        NonEmpty::with_last(first.into_iter().chain(history), prompt),
    )
}

/// The passthrough keys that will override a typed field the caller also set.
/// The override itself is the documented precedence (see
/// [`CompletionRequestBuilder::additional_params`]); naming the collisions
/// makes an accidental one visible instead of silent.
pub(crate) fn shadowed_typed_fields<'a>(
    additional_params: Option<&serde_json::Value>,
    typed: &[(&'a str, bool)],
) -> Vec<&'a str> {
    let Some(serde_json::Value::Object(params)) = additional_params else {
        return Vec::new();
    };
    typed
        .iter()
        .filter(|(key, set)| *set && params.contains_key(*key))
        .map(|(key, _)| *key)
        .collect()
}

impl<M: CompletionModel> CompletionRequestBuilder<M> {
    /// Sends the completion request to the completion model provider and returns the completion response.
    pub async fn send(self) -> Result<CompletionResponse, ProviderError> {
        let (model, request) = self.into_model_and_request();
        model.completion(request).await
    }

    /// Stream the completion request
    pub async fn stream(self) -> Result<StreamingCompletionResponse, ProviderError> {
        let (model, request) = self.into_model_and_request();
        model.stream(request).await
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod response_identity_tests;
