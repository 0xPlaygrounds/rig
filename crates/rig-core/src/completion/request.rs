//! Completion requests, normalized responses, and provider model contracts.
//!
//! ```
//! use rig_core::completion::CompletionRequest;
//!
//! let request = CompletionRequest::new("Who are you?")
//!     .preamble("You are a concise assistant.")
//!     .temperature(0.5);
//! assert_eq!(request.temperature, Some(0.5));
//! ```

use super::message::{AssistantContent, DocumentMediaType, Reasoning, ReasoningContent, ToolCall};
use crate::error::ProviderError;
use crate::message::ToolChoice;
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
    /// call. All other reasons remain unchanged. Response builders and streaming
    /// aggregation apply this reconciliation.
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

/// Assistant content and normalized completion metadata. The choice may be
/// empty, including for truncated or filtered turns. Provider-specific data is
/// available through [`Self::raw`] without retaining a concrete model type.
///
/// A response goes straight back into the conversation as the assistant
/// turn: `history.push(response.into())`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(from = "CompletionResponseRepr")]
pub struct CompletionResponse {
    /// Assistant content returned by the provider, possibly empty.
    pub choice: Vec<AssistantContent>,
    /// Tokens used during prompting and responding
    pub usage: Usage,
    /// Provider-issued assistant message ID suitable for replay in
    /// [`Message::Assistant`]. Response-wide IDs belong in [`Self::response_id`].
    #[serde(default)]
    pub message_id: Option<String>,
    /// Provider-issued response ID for telemetry and diagnostics.
    /// Must not be replayed as an assistant message ID.
    #[serde(default)]
    pub response_id: Option<String>,
    /// Request identifier from HTTP headers or SDK metadata, not the body's
    /// message or response ID. `None` when the provider reports none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_request_id: Option<String>,
    /// Reported finish reason, reconciled by the setters with tool-call output.
    /// Read through [`Self::finish_reason`].
    #[serde(default)]
    finish_reason: Option<FinishReason>,
    /// Stable descriptor name of the provider that produced this response, for
    /// example `"openai"`. Always populated, including for responses derived
    /// from a stream that ended before its terminal record.
    pub provider: String,
    /// Provider-reported model identifier for the response.
    ///
    /// This is the model named by the wire response, not the model that was
    /// requested; it is `None` when the provider reports no identifier.
    #[serde(default)]
    pub model: Option<String>,
    /// Provider response document for typed inspection through deserialization.
    /// Parsed wire types may omit unmodeled fields. This data does not override
    /// normalized fields; callers constructing responses must supply it.
    pub raw: serde_json::Value,
}

/// Distinct message, response, and transport identifiers for one model call.
/// Unreported identifiers remain `None`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponseIdentity {
    /// Provider-issued assistant message ID suitable for replay.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message_id: Option<String>,
    /// Response-wide ID, never replayed as a message ID.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    /// Transport request ID from HTTP headers or SDK metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_request_id: Option<String>,
}

impl CompletionResponse {
    /// Create a response from its required parts; optional metadata starts
    /// unset and is filled in with the `with_*` helpers. `raw` is the
    /// provider's own document for this response, serialized; see
    /// [`Self::raw`].
    pub fn new(
        choice: Vec<AssistantContent>,
        usage: Usage,
        provider: impl Into<String>,
        raw: serde_json::Value,
    ) -> Self {
        Self {
            choice,
            usage,
            message_id: None,
            response_id: None,
            provider_request_id: None,
            finish_reason: None,
            provider: provider.into(),
            model: None,
            raw,
        }
    }

    /// Why the model stopped generating, when the provider reported it.
    pub fn finish_reason(&self) -> Option<FinishReason> {
        self.finish_reason.clone()
    }

    /// This response's identity metadata as one [`ResponseIdentity`] carrier.
    pub fn identity(&self) -> ResponseIdentity {
        ResponseIdentity {
            message_id: self.message_id.clone(),
            response_id: self.response_id.clone(),
            provider_request_id: self.provider_request_id.clone(),
        }
    }

    /// Attach the normalized finish reason, reconciled against the choice via
    /// [`FinishReason::reconcile_with_output`].
    pub fn with_finish_reason(self, finish_reason: FinishReason) -> Self {
        self.with_optional_finish_reason(Some(finish_reason))
    }

    /// The text parts of [`Self::choice`], concatenated in order.
    pub fn text(&self) -> String {
        self.choice
            .iter()
            .filter_map(|part| match part {
                AssistantContent::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect()
    }

    /// The reasoning text of [`Self::choice`] (text and summaries), concatenated
    /// in order. Encrypted and redacted reasoning has no text.
    pub fn reasoning(&self) -> String {
        self.choice
            .iter()
            .filter_map(|part| match part {
                AssistantContent::Reasoning(reasoning) => Some(reasoning_text(reasoning)),
                _ => None,
            })
            .collect()
    }

    /// The tool calls in [`Self::choice`], in order.
    pub fn tool_calls(&self) -> impl Iterator<Item = &ToolCall> {
        self.choice.iter().filter_map(|part| match part {
            AssistantContent::ToolCall(call) => Some(call),
            _ => None,
        })
    }

    /// Sets or clears the finish reason, reconciling a present reason with the choice.
    pub fn with_optional_finish_reason(mut self, finish_reason: Option<FinishReason>) -> Self {
        let has_tool_call = self
            .choice
            .iter()
            .any(|content| matches!(content, AssistantContent::ToolCall(_)));
        self.finish_reason =
            finish_reason.map(|reason| reason.reconcile_with_output(has_tool_call));
        self
    }
}

crate::provider_response::response_metadata_setters!(CompletionResponse);

/// The assistant turn: [`CompletionResponse::choice`] in order, under the
/// provider's message id.
impl From<CompletionResponse> for Message {
    fn from(response: CompletionResponse) -> Self {
        Message::Assistant {
            id: response.message_id,
            content: response.choice,
        }
    }
}

/// The text and summaries of one reasoning part, concatenated in order.
pub(crate) fn reasoning_text(reasoning: &Reasoning) -> String {
    reasoning
        .content
        .iter()
        .filter_map(|content| match content {
            ReasoningContent::Text { text, .. } => Some(text.as_str()),
            ReasoningContent::Summary(summary) => Some(summary.as_str()),
            ReasoningContent::Encrypted(_) | ReasoningContent::Redacted { .. } => None,
        })
        .collect()
}

/// Deserialization shape routed through builders for finish-reason reconciliation
/// and empty-identifier normalization.
#[derive(Deserialize)]
struct CompletionResponseRepr {
    choice: Vec<AssistantContent>,
    usage: Usage,
    #[serde(default)]
    message_id: Option<String>,
    #[serde(default)]
    response_id: Option<String>,
    #[serde(default)]
    provider_request_id: Option<String>,
    #[serde(default)]
    finish_reason: Option<FinishReason>,
    provider: String,
    #[serde(default)]
    model: Option<String>,
    raw: serde_json::Value,
}

impl From<CompletionResponseRepr> for CompletionResponse {
    fn from(repr: CompletionResponseRepr) -> Self {
        let CompletionResponseRepr {
            choice,
            usage,
            message_id,
            response_id,
            provider_request_id,
            finish_reason,
            provider,
            model,
            raw,
        } = repr;
        Self::new(choice, usage, provider, raw)
            .with_optional_message_id(message_id)
            .with_optional_response_id(response_id)
            .with_optional_provider_request_id(provider_request_id)
            .with_optional_finish_reason(finish_reason)
            .with_optional_model(model)
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
/// Defaults describe the common case; construct through [`Self::new`] and setters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderCapabilities {
    /// Whether native structured output can remain enabled with tool calls
    /// without suppressing them. A wire that sets this also supports native
    /// structured output on its own. Defaults to `false`.
    pub composes_native_output_with_tools: bool,
    /// Whether the model accepts a forced tool choice
    /// ([`ToolChoice::Required`] or [`ToolChoice::Specific`]). A wire whose
    /// model rejects one fails to encode such a request, and runtimes that
    /// force the structured-output tool fall back to native output or `auto`.
    /// Defaults to `true`.
    #[serde(
        default = "crate::json_utils::default_true",
        skip_serializing_if = "crate::json_utils::is_true"
    )]
    pub accepts_forced_tool_choice: bool,
}

impl Default for ProviderCapabilities {
    fn default() -> Self {
        Self::new()
    }
}

impl ProviderCapabilities {
    /// Create the default capability set.
    pub const fn new() -> Self {
        Self {
            composes_native_output_with_tools: false,
            accepts_forced_tool_choice: true,
        }
    }

    /// Declare whether native structured output composes with tool calls.
    pub const fn with_native_output_tool_composition(mut self, supported: bool) -> Self {
        self.composes_native_output_with_tools = supported;
        self
    }

    /// Declare whether the model accepts a forced tool choice.
    pub const fn with_forced_tool_choice(mut self, accepted: bool) -> Self {
        self.accepts_forced_tool_choice = accepted;
        self
    }
}

/// Struct representing a general completion request that can be sent to a completion model provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    /// Optional model override for this request.
    pub model: Option<String>,
    /// Conversation ending with the prompt. Must contain at least one message;
    /// checked by [`Self::validate_message_content`].
    pub chat_history: Vec<Message>,
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
    /// The system instructions of this request: the content of the leading
    /// [`Message::System`] in `chat_history`, which is where
    /// [`Self::preamble`] places it.
    pub fn system_instructions(&self) -> Option<&str> {
        match self.chat_history.first() {
            Some(Message::System { content }) => Some(content.as_str()),
            _ => None,
        }
    }

    /// Returns a request error for empty history, empty user or assistant
    /// content lists, or tool results with no content blocks. Empty strings,
    /// including system messages, are allowed.
    ///
    /// The agent runtime validates the requests it prepares; call this
    /// before [`Model::call`](crate::Model::call) or `stream` when the
    /// history came from a caller.
    /// Response-content validation is provider-specific and is not performed
    /// here.
    pub fn validate_message_content(&self) -> Result<(), ProviderError> {
        if self.chat_history.is_empty() {
            return Err(ProviderError::Request(
                "request has an empty chat history; providers require at least one message"
                    .to_owned()
                    .into(),
            ));
        }

        let empty_message = |role: &str, index: usize| {
            ProviderError::Request(
                format!(
                    "{role} message at index {index} has no content; \
                     providers reject empty content blocks"
                )
                .into(),
            )
        };

        for (index, message) in self.chat_history.iter().enumerate() {
            match message {
                Message::System { .. } => {}
                Message::Assistant { content, .. } => {
                    if content.is_empty() {
                        return Err(empty_message("assistant", index));
                    }
                }
                Message::User { content } => {
                    if content.is_empty() {
                        return Err(empty_message("user", index));
                    }

                    for (position, item) in content.iter().enumerate() {
                        // Keep exhaustive so new content variants must choose a
                        // request-validation policy.
                        match item {
                            UserContent::ToolResult(result) if result.content.is_empty() => {
                                let name = &result.name;
                                return Err(ProviderError::Request(
                                    format!(
                                        "tool result for `{name}` at index {position} of the \
                                         user message at index {index} has no content; \
                                         providers reject empty content blocks"
                                    )
                                    .into(),
                                ));
                            }
                            UserContent::ToolResult(_)
                            | UserContent::Text(_)
                            | UserContent::Image(_)
                            | UserContent::Audio(_)
                            | UserContent::Video(_)
                            | UserContent::Document(_) => {}
                        }
                    }
                }
            }
        }

        Ok(())
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

        crate::message::non_empty(messages).map(|content| Message::User { content })
    }

    pub(crate) fn chat_history_with_documents(&self) -> Vec<Message> {
        let mut chat_history = self.chat_history.clone();
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

impl CompletionRequest {
    /// A request whose conversation is the one user message `prompt`, with
    /// no preamble, documents or tools. The setters below add to it; like
    /// [`Self::validate_message_content`], nothing here checks the content.
    ///
    /// Each setter changes the request's public fields as it is called, so
    /// order matters where two setters touch the same field: a second
    /// [`Self::preamble`] adds a second system message, and
    /// [`Self::additional_params`] with a `tools` key (or `None`) replaces
    /// provider tools added before it. Set `additional_params` first.
    ///
    /// ```
    /// use rig_core::completion::CompletionRequest;
    ///
    /// let request = CompletionRequest::new("Who are you?")
    ///     .preamble("You are a concise assistant.")
    ///     .temperature(0.5);
    /// assert_eq!(request.chat_history.len(), 2);
    /// assert_eq!(request.temperature, Some(0.5));
    /// ```
    pub fn new(prompt: impl Into<Message>) -> Self {
        Self::conversation(vec![prompt.into()])
    }

    /// A request for `chat_history` as given, with nothing else set.
    fn conversation(chat_history: Vec<Message>) -> Self {
        Self {
            model: None,
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

    /// Put `preamble` first in the conversation, as a [`Message::System`],
    /// ahead of any system message already there.
    pub fn preamble(mut self, preamble: impl Into<String>) -> Self {
        self.chat_history
            .insert(0, Message::system(preamble.into()));
        self
    }

    /// Override the model for this request.
    pub fn model<S: Into<String>>(mut self, model: impl Into<Option<S>>) -> Self {
        self.model = model.into().map(Into::into);
        self.warn_if_shadowed("model", self.model.is_some());
        self
    }

    /// Add `message` to the conversation, before the prompt (its last
    /// message).
    pub fn message(self, message: Message) -> Self {
        self.messages([message])
    }

    /// Add `messages` to the conversation in order, before the prompt (its
    /// last message).
    pub fn messages(mut self, messages: impl IntoIterator<Item = Message>) -> Self {
        let prompt = self.chat_history.pop();
        self.chat_history.extend(messages);
        self.chat_history.extend(prompt);
        self
    }

    /// Add a document.
    pub fn document(mut self, document: Document) -> Self {
        self.documents.push(document);
        self
    }

    /// Add documents in order.
    pub fn documents(mut self, documents: impl IntoIterator<Item = Document>) -> Self {
        self.documents.extend(documents);
        self
    }

    /// Add a tool.
    pub fn tool(self, tool: ToolDefinition) -> Self {
        self.tools(vec![tool])
    }

    /// Add tools in order.
    pub fn tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        let first = self.tools.is_empty();
        self.tools.extend(tools);
        self.warn_if_shadowed("tools", first && !self.tools.is_empty());
        self
    }

    /// Add a provider-hosted tool: appended to `additional_params.tools`,
    /// so a later [`Self::additional_params`] with a `tools` key replaces
    /// it.
    pub fn provider_tool(self, tool: ProviderToolDefinition) -> Self {
        self.provider_tools(vec![tool])
    }

    /// Add provider-hosted tools in order: appended to
    /// `additional_params.tools`.
    pub fn provider_tools(mut self, tools: Vec<ProviderToolDefinition>) -> Self {
        self.additional_params =
            merge_provider_tools_into_additional_params(self.additional_params.take(), tools);
        self
    }

    /// Merge provider-specific parameters into the request's, key by key;
    /// `None` clears them, provider tools included. Provider conversion determines precedence over typed fields,
    /// and a key that overrides a typed field this request sets is logged.
    pub fn additional_params(
        mut self,
        additional_params: impl Into<Option<serde_json::Value>>,
    ) -> Self {
        let additional_params = additional_params.into();
        for key in shadowed_typed_fields(
            additional_params.as_ref(),
            &[
                ("temperature", self.temperature.is_some()),
                ("max_tokens", self.max_tokens.is_some()),
                ("tool_choice", self.tool_choice.is_some()),
                ("model", self.model.is_some()),
                ("tools", !self.tools.is_empty()),
                ("response_format", self.output_schema.is_some()),
            ],
        ) {
            warn_shadowed(key);
        }
        self.additional_params =
            json_utils::merge_params(self.additional_params.take(), additional_params);
        self
    }

    /// Set, or with `None` clear, the temperature.
    pub fn temperature(mut self, temperature: impl Into<Option<f64>>) -> Self {
        self.temperature = temperature.into();
        self.warn_if_shadowed("temperature", self.temperature.is_some());
        self
    }

    /// Set, or with `None` clear, the output-token limit. Provider-specific
    /// defaults and requirements apply.
    pub fn max_tokens(mut self, max_tokens: impl Into<Option<u64>>) -> Self {
        self.max_tokens = max_tokens.into();
        self.warn_if_shadowed("max_tokens", self.max_tokens.is_some());
        self
    }

    /// Set the tool-selection policy.
    pub fn tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self.warn_if_shadowed("tool_choice", true);
        self
    }

    /// Set, or with `None` clear, a native structured-output schema for
    /// providers that support one. The returned content is not
    /// deserialized.
    pub fn output_schema(mut self, schema: impl Into<Option<schemars::Schema>>) -> Self {
        self.output_schema = schema.into();
        self.warn_if_shadowed("response_format", self.output_schema.is_some());
        self
    }

    /// Opt in to sensitive content telemetry, off by default. See
    /// [`Self::record_telemetry_content`] for what that exposes.
    pub fn record_content_telemetry(mut self, enabled: bool) -> Self {
        self.record_telemetry_content = enabled;
        self
    }

    /// The input messages telemetry records: the conversation with the
    /// documents inserted after any leading system messages.
    pub fn messages_for_telemetry(&self) -> Vec<Message> {
        self.chat_history_with_documents()
    }

    /// Log a typed field `key` that `additional_params` already overrides.
    fn warn_if_shadowed(&self, key: &'static str, set: bool) {
        if !shadowed_typed_fields(self.additional_params.as_ref(), &[(key, set)]).is_empty() {
            warn_shadowed(key);
        }
    }
}

fn warn_shadowed(key: &str) {
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

impl From<&str> for CompletionRequest {
    fn from(prompt: &str) -> Self {
        Self::new(prompt)
    }
}

impl From<String> for CompletionRequest {
    fn from(prompt: String) -> Self {
        Self::new(prompt)
    }
}

impl From<Message> for CompletionRequest {
    fn from(prompt: Message) -> Self {
        Self::new(prompt)
    }
}

/// The conversation as given, ending with the prompt. An empty one fails
/// [`CompletionRequest::validate_message_content`].
impl From<Vec<Message>> for CompletionRequest {
    fn from(chat_history: Vec<Message>) -> Self {
        Self::conversation(chat_history)
    }
}

/// The passthrough keys that will override a typed field the caller also set.
/// The override itself is the documented precedence (see
/// [`CompletionRequest::additional_params`]); naming the collisions
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

#[cfg(test)]
mod tests;

#[cfg(test)]
mod response_identity_tests;

#[cfg(test)]
mod plain_value_tests;
