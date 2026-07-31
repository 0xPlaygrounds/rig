//! Ollama API integration.
//!
//! # Example
//! ```no_run
//! use rig_core::providers::ollama;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let rt = rig_core::http_runtime::HttpRuntime::new();
//!
//! // Defaults to http://localhost:11434 with no auth; `with_base_url` /
//! // `with_api_key` point at a remote or proxied Ollama instead.
//! let cfg = ollama::functions::Config::from_env("qwen2.5:14b")?;
//! let request = rig_core::completion::CompletionRequest::builder("Entertain me!")
//!     .preamble("You are a comedian here to entertain the user using humour and jokes.")
//!     .build();
//! let response = ollama::functions::complete(&cfg, &rt, request).await?;
//! println!("{:?}", response.choice);
//!
//! // Embeddings use the sibling `EmbeddingConfig`.
//! let emb_cfg = ollama::functions::EmbeddingConfig::new(ollama::ALL_MINILM);
//! let embeddings = ollama::functions::embed(&emb_cfg, &rt, vec![
//!     "Why is the sky blue?".to_owned(),
//!     "Why is the grass green?".to_owned()
//! ]).await?;
//! println!("Embedding response: {:?}", embeddings);
//! # Ok(())
//! # }
//! ```
use crate::completion::Usage;
use crate::http_client;
use crate::message::DocumentSourceKind;
use crate::model::{Model, ModelList, ModelListingError};
use crate::streaming::RawStreamingChoice;
use crate::{
    OneOrMany,
    completion::{self, CompletionError, CompletionRequest},
    embeddings::{self, EmbeddingError},
    json_utils, message,
    message::{ImageDetail, Text},
};
use async_stream::try_stream;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{convert::TryFrom, str::FromStr};

const OLLAMA_API_BASE_URL: &str = "http://localhost:11434";

// ---------- Embedding API ----------

pub const ALL_MINILM: &str = "all-minilm";
pub const NOMIC_EMBED_TEXT: &str = "nomic-embed-text";

/// Known embedding dimensionality for a built-in Ollama embedding model.
///
/// Seeds [`functions::EmbeddingConfig::ndims`], and available directly to
/// callers that need the vector width (index creation, store schemas) without
/// building a config.
pub fn model_dimensions_from_identifier(identifier: &str) -> Option<usize> {
    match identifier {
        ALL_MINILM => Some(384),
        NOMIC_EMBED_TEXT => Some(768),
        _ => None,
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub model: String,
    pub embeddings: Vec<Vec<f64>>,
    #[serde(default)]
    pub total_duration: Option<u64>,
    #[serde(default)]
    pub load_duration: Option<u64>,
    #[serde(default)]
    pub prompt_eval_count: Option<u64>,
}

/// Build the serialized `/api/embed` request body. Pure; used by
/// [`functions::embed`].
pub(crate) fn build_embedding_body(
    model: &str,
    texts: &[String],
) -> Result<Vec<u8>, EmbeddingError> {
    Ok(serde_json::to_vec(&json!({
        "model": model,
        "input": texts
    }))?)
}

/// Parse an `/api/embed` response into the normalized
/// [`embeddings::EmbeddingResponse`], zipping vectors back onto
/// `documents`. Pure; used by [`functions::embed`].
/// Usage is taken from `prompt_eval_count` when present.
pub(crate) fn parse_embedding_response(
    status: http::StatusCode,
    body: &str,
    documents: Vec<String>,
) -> Result<embeddings::EmbeddingResponse, EmbeddingError> {
    if !status.is_success() {
        return Err(EmbeddingError::from_http_response(status, body.to_string()));
    }

    let api_resp: EmbeddingResponse = serde_json::from_str(body)?;

    if api_resp.embeddings.len() != documents.len() {
        return Err(EmbeddingError::ResponseError(
            "Number of returned embeddings does not match input".into(),
        ));
    }
    let mut usage = crate::completion::Usage::new();
    if let Some(prompt_eval_count) = api_resp.prompt_eval_count {
        usage.input_tokens = prompt_eval_count;
        usage.total_tokens = prompt_eval_count;
    }
    let embeddings = api_resp
        .embeddings
        .into_iter()
        .zip(documents)
        .map(|(vec, document)| embeddings::Embedding { document, vec })
        .collect();
    Ok(embeddings::EmbeddingResponse { embeddings, usage })
}

// ---------- Completion API ----------

pub const LLAMA3_2: &str = "llama3.2";
pub const LLAVA: &str = "llava";
pub const MISTRAL: &str = "mistral";

#[derive(Debug, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub model: String,
    pub created_at: String,
    pub message: Message,
    pub done: bool,
    #[serde(default)]
    pub done_reason: Option<String>,
    #[serde(default)]
    pub total_duration: Option<u64>,
    #[serde(default)]
    pub load_duration: Option<u64>,
    #[serde(default)]
    pub prompt_eval_count: Option<u64>,
    #[serde(default)]
    pub prompt_eval_duration: Option<u64>,
    #[serde(default)]
    pub eval_count: Option<u64>,
    #[serde(default)]
    pub eval_duration: Option<u64>,
}
/// Maps Ollama's `done_reason` onto the normalized
/// [`completion::FinishReason`] vocabulary, carrying unmapped values
/// verbatim in `Other`.
fn map_finish_reason(reason: &str) -> completion::FinishReason {
    match reason {
        "stop" => completion::FinishReason::Stop,
        "length" => completion::FinishReason::Length,
        other => completion::FinishReason::Other(other.to_string()),
    }
}

impl TryFrom<CompletionResponse> for completion::CompletionResponse {
    type Error = CompletionError;
    fn try_from(resp: CompletionResponse) -> Result<Self, Self::Error> {
        match resp.message {
            // Process only if an assistant message is present.
            Message::Assistant {
                content,
                thinking,
                tool_calls,
                ..
            } => {
                let mut assistant_contents = Vec::new();
                let permits_omitted_think_start = resp.model.to_ascii_lowercase().contains("qwen3");
                let (legacy_thinking, visible_content) =
                    if matches!(thinking.as_deref(), None | Some("")) {
                        split_legacy_thinking(&content, permits_omitted_think_start)
                    } else {
                        (None, content.as_str())
                    };
                // Preserve the model's reasoning so it round-trips into agent
                // history and is echoed back to Ollama on the next turn (issue
                // #1926). Without this, non-streaming `thinking` would be lost
                // from `choice`, unlike the streaming path (see
                // `RawStreamingChoice::ReasoningDelta` below).
                if let Some(thinking) = thinking.as_deref().filter(|t| !t.is_empty()) {
                    assistant_contents.push(completion::AssistantContent::reasoning(thinking));
                }
                if let Some(legacy_thinking) = legacy_thinking {
                    assistant_contents
                        .push(completion::AssistantContent::reasoning(legacy_thinking));
                }
                // Add the assistant's text content if any.
                if !visible_content.is_empty() {
                    assistant_contents.push(completion::AssistantContent::text(visible_content));
                }
                // Process tool_calls following Ollama's chat response definition.
                // Each ToolCall has an id, a type, and a function field.
                for tc in tool_calls.iter() {
                    assistant_contents.push(completion::AssistantContent::tool_call(
                        tc.function.name.clone(),
                        tc.function.name.clone(),
                        tc.function.arguments.clone(),
                    ));
                }
                let choice = OneOrMany::many(assistant_contents).map_err(|_| {
                    CompletionError::ResponseError("No content provided".to_owned())
                })?;
                let prompt_tokens = resp.prompt_eval_count.unwrap_or(0);
                let completion_tokens = resp.eval_count.unwrap_or(0);

                let usage = Usage {
                    input_tokens: prompt_tokens,
                    output_tokens: completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                    cached_input_tokens: 0,
                    cache_creation_input_tokens: 0,
                    tool_use_prompt_tokens: 0,
                    reasoning_tokens: 0,
                };

                let mut converted = completion::CompletionResponse::new(choice, usage, "ollama")
                    .with_model(resp.model);
                if let Some(done_reason) = resp.done_reason.as_deref() {
                    converted = converted.with_finish_reason(map_finish_reason(done_reason));
                }

                Ok(converted)
            }
            _ => Err(CompletionError::ResponseError(
                "Chat response does not include an assistant message".into(),
            )),
        }
    }
}

/// Older reasoning models served by Ollama sometimes returned their reasoning
/// in `content` instead of `thinking`. Qwen can also omit the opening marker
/// because its chat template prefills it. Only split a leading, terminated
/// reasoning block so ordinary mentions of the marker remain untouched.
fn split_legacy_thinking(content: &str, permits_omitted_start: bool) -> (Option<&str>, &str) {
    let trimmed = content.trim_start();
    let split = if let Some(reasoning_start) = trimmed.strip_prefix("<think>") {
        reasoning_start.split_once("</think>")
    } else if permits_omitted_start {
        // Qwen's prefilled opening marker produces this exact blank-line
        // boundary. Requiring the full boundary avoids hiding ordinary visible
        // text that merely demonstrates a closing XML-like tag on its own line.
        trimmed.split_once("\n</think>\n\n")
    } else {
        None
    };
    let Some((reasoning, visible)) = split else {
        return (None, content);
    };

    let reasoning = reasoning.trim();
    if reasoning.is_empty() {
        return (None, visible.trim_start());
    }

    (Some(reasoning), visible.trim_start())
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct OllamaCompletionRequest {
    model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ToolDefinition>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    think: Option<Think>,
    #[serde(skip_serializing_if = "Option::is_none")]
    keep_alive: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<schemars::Schema>,
    options: serde_json::Value,
}

impl TryFrom<(&str, CompletionRequest)> for OllamaCompletionRequest {
    type Error = CompletionError;

    fn try_from((model, req): (&str, CompletionRequest)) -> Result<Self, Self::Error> {
        let chat_history = req.chat_history_with_documents();
        let model = req.model.clone().unwrap_or_else(|| model.to_string());
        if req.tool_choice.is_some() {
            tracing::warn!("WARNING: `tool_choice` not supported for Ollama");
        }
        // Build up the order of messages.
        let mut partial_history = vec![];
        partial_history.extend(chat_history);

        // Add preamble to chat history (if available)
        let mut full_history: Vec<Message> = Vec::new();

        // Convert and extend the rest of the history
        full_history.extend(
            partial_history
                .into_iter()
                .map(message::Message::try_into)
                .collect::<Result<Vec<Vec<Message>>, _>>()?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>(),
        );

        let mut think: Option<Think> = None;
        let mut keep_alive: Option<String> = None;

        // The native API has no top-level `temperature` or `max_tokens`;
        // both are model parameters that belong in `options` (`max_tokens`
        // is called `num_predict` there).
        let mut base_options = serde_json::Map::new();
        if let Some(temperature) = req.temperature {
            base_options.insert("temperature".to_string(), json!(temperature));
        }
        if let Some(max_tokens) = req.max_tokens {
            base_options.insert("num_predict".to_string(), json!(max_tokens));
        }
        let base_options = Value::Object(base_options);

        let options = if let Some(mut extra) = req.additional_params {
            // Extract top-level parameters that should not be in `options`
            if let Some(obj) = extra.as_object_mut() {
                // Extract `think` parameter
                if let Some(think_val) = obj.remove("think") {
                    think = Some(match think_val {
                        Value::Bool(think) => Think::Bool(think),
                        Value::String(think) => Think::Level(match think.to_lowercase().as_str() {
                            "low" => Level::Low,
                            "medium" => Level::Medium,
                            "high" => Level::High,
                            "max" => Level::Max,
                            _ => {
                                return Err(CompletionError::RequestError(
                                    "`think` must be a 'low', 'medium', 'high', 'max' or bool"
                                        .into(),
                                ));
                            }
                        }),
                        _ => {
                            return Err(CompletionError::RequestError(
                                "`think` must be a 'low', 'medium', 'high', 'max' or bool".into(),
                            ));
                        }
                    });
                }

                // Extract `keep_alive` parameter
                if let Some(keep_alive_val) = obj.remove("keep_alive") {
                    keep_alive = Some(
                        keep_alive_val
                            .as_str()
                            .ok_or_else(|| {
                                CompletionError::RequestError(
                                    "`keep_alive` must be a string".into(),
                                )
                            })?
                            .to_string(),
                    );
                }
            }

            json_utils::merge(base_options, extra)
        } else {
            base_options
        };

        Ok(Self {
            model: model.to_string(),
            messages: full_history,
            stream: false,
            think,
            keep_alive,
            format: req.output_schema,
            tools: req
                .tools
                .clone()
                .into_iter()
                .map(ToolDefinition::from)
                .collect::<Vec<_>>(),
            options,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum Think {
    Bool(bool),
    Level(Level),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Level {
    Low,
    Medium,
    High,
    Max,
}

// ---------- Native NDJSON stream driver ----------

/// Reassembles newline-delimited JSON lines from a chunked HTTP byte stream.
///
/// `bytes_stream` makes no promises about chunk boundaries, so a single NDJSON
/// line can be split across multiple chunks. `NdjsonBuffer` holds the trailing
/// fragment between calls and yields only fully terminated lines.
#[derive(Default)]
struct NdjsonBuffer {
    buf: Vec<u8>,
}

impl NdjsonBuffer {
    fn new() -> Self {
        Self::default()
    }

    /// Appends `chunk` to the buffer and returns any newly completed lines.
    /// Empty lines are skipped; trailing partial data is retained for the next call.
    fn decode(&mut self, chunk: &[u8]) -> Vec<Vec<u8>> {
        self.buf.extend_from_slice(chunk);

        let mut lines = Vec::new();
        while let Some(pos) = self.buf.iter().position(|&b| b == b'\n') {
            let mut line: Vec<u8> = self.buf.drain(..=pos).collect();
            line.pop();
            if !line.is_empty() {
                lines.push(line);
            }
        }
        lines
    }
}

/// Consume a native `/api/chat` streaming response (NDJSON) as a raw
/// streaming-choice stream. Drives [`functions::open_stream`].
pub(crate) async fn consume_chat_streaming_response(
    response: http_client::StreamingResponse,
) -> Result<impl futures::Stream<Item = Result<RawStreamingChoice, CompletionError>>, CompletionError>
{
    let status = response.status();
    let mut byte_stream = response.into_body();

    if !status.is_success() {
        let mut body = Vec::new();
        while let Some(chunk) = byte_stream.next().await {
            match chunk {
                Ok(bytes) => body.extend_from_slice(&bytes),
                Err(e) => {
                    tracing::warn!(error = %e, "failed reading Ollama error-response body; preserving partial body");
                    break;
                }
            }
        }
        return Err(CompletionError::from_http_response(
            status,
            String::from_utf8_lossy(&body),
        ));
    }

    let stream = try_stream! {
        let span = tracing::Span::current();
        let mut line_buf = NdjsonBuffer::new();

        while let Some(chunk) = byte_stream.next().await {
            let bytes = chunk.map_err(|e| http_client::Error::Instance(e.into()))?;

            for line in line_buf.decode(&bytes) {
                tracing::debug!(target: "rig", "Received NDJSON line from Ollama: {}", String::from_utf8_lossy(&line));

                let response: CompletionResponse = serde_json::from_slice(&line)?;

                if response.done {
                    span.record("gen_ai.response.model", &response.model);
                }

                if let Message::Assistant { content, thinking, tool_calls, .. } = response.message {
                    if let Some(thinking_content) = thinking && !thinking_content.is_empty() {
                        yield RawStreamingChoice::ReasoningDelta {
                            id: None,
                            reasoning: thinking_content,
                        };
                    }

                    if !content.is_empty() {
                        yield RawStreamingChoice::Message(content);
                    }

                    for tool_call in tool_calls {
                        yield RawStreamingChoice::ToolCall(
                            crate::streaming::RawStreamingToolCall::new(String::new(), tool_call.function.name, tool_call.function.arguments)
                        );
                    }
                }

                if response.done {
                    span.record("gen_ai.usage.input_tokens", response.prompt_eval_count);
                    span.record("gen_ai.usage.output_tokens", response.eval_count);
                    let input_tokens = response.prompt_eval_count.unwrap_or_default();
                    let output_tokens = response.eval_count.unwrap_or_default();
                    let usage = Usage {
                        input_tokens,
                        output_tokens,
                        total_tokens: input_tokens + output_tokens,
                        ..Usage::new()
                    };
                    let mut final_response = crate::streaming::StreamFinal::new("ollama", usage)
                        .with_model(response.model.clone());
                    if let Some(done_reason) = response.done_reason.as_deref() {
                        final_response = final_response.with_finish_reason(map_finish_reason(done_reason));
                    }
                    yield RawStreamingChoice::FinalResponse(final_response);
                    break;
                }
            }
        }
    };

    Ok(stream)
}

// ---------- Model Listing  ----------

#[derive(Debug, Deserialize)]
struct ListModelsResponse {
    models: Vec<ListModelEntry>,
}

#[derive(Debug, Deserialize)]
struct ListModelEntry {
    name: String,
    model: String,
}

impl From<ListModelEntry> for Model {
    fn from(value: ListModelEntry) -> Self {
        Model::new(value.model, value.name)
    }
}

/// Path of the model-listing endpoint, relative to the API base URL.
pub(crate) const LIST_MODELS_PATH: &str = "/api/tags";

/// Parse a `GET /api/tags` response into a [`ModelList`]. Pure.
///
/// Used by [`functions::list_models`].
pub(crate) fn parse_list_models_response(
    status: http::StatusCode,
    body: &[u8],
) -> Result<ModelList, ModelListingError> {
    if !status.is_success() {
        return Err(ModelListingError::api_error_with_context(
            "Ollama",
            LIST_MODELS_PATH,
            status.as_u16(),
            body,
        ));
    }
    let api_resp: ListModelsResponse = serde_json::from_slice(body).map_err(|error| {
        ModelListingError::parse_error_with_context("Ollama", LIST_MODELS_PATH, &error, body)
    })?;
    let models = api_resp.models.into_iter().map(Model::from).collect();
    Ok(ModelList::new(models))
}

// ---------- Tool Definition Conversion ----------

/// Ollama-required tool definition format.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub type_field: String, // Fixed as "function"
    pub function: completion::ToolDefinition,
}

/// Convert internal ToolDefinition (from the completion module) into Ollama's tool definition.
impl From<crate::completion::ToolDefinition> for ToolDefinition {
    fn from(tool: crate::completion::ToolDefinition) -> Self {
        ToolDefinition {
            type_field: "function".to_owned(),
            function: completion::ToolDefinition {
                name: tool.name,
                description: tool.description,
                parameters: tool.parameters,
            },
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ToolCall {
    #[serde(default, rename = "type")]
    pub r#type: ToolType,
    pub function: Function,
}
#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub enum ToolType {
    #[default]
    Function,
}
#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct Function {
    pub name: String,
    pub arguments: Value,
}

// ---------- Provider Message Definition ----------

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    User {
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        images: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    Assistant {
        #[serde(default)]
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        thinking: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        images: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(default, deserialize_with = "json_utils::null_or_vec")]
        tool_calls: Vec<ToolCall>,
    },
    System {
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        images: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    #[serde(rename = "tool")]
    ToolResult {
        #[serde(rename = "tool_name")]
        name: String,
        content: String,
    },
}

/// -----------------------------
/// Provider Message Conversions
/// -----------------------------
fn user_message_from_content(
    content: Vec<crate::message::UserContent>,
) -> Result<Message, crate::message::MessageError> {
    let mut texts = Vec::new();
    let mut images = Vec::new();

    for content in content {
        match content {
            crate::message::UserContent::Text(crate::message::Text { text, .. }) => {
                texts.push(text);
            }
            crate::message::UserContent::Image(crate::message::Image {
                data: DocumentSourceKind::Base64(data),
                ..
            }) => images.push(data),
            crate::message::UserContent::Image(_) => {
                return Err(crate::message::MessageError::ConversionError(
                    "Ollama images must be base64 encoded data".into(),
                ));
            }
            crate::message::UserContent::Document(crate::message::Document {
                data: DocumentSourceKind::Base64(data) | DocumentSourceKind::String(data),
                ..
            }) => texts.push(data),
            crate::message::UserContent::Document(_) => {
                return Err(crate::message::MessageError::ConversionError(
                    "Ollama documents must be string or base64 encoded data".into(),
                ));
            }
            crate::message::UserContent::Audio(_) => {
                return Err(crate::message::MessageError::ConversionError(
                    "Ollama does not support audio user content".into(),
                ));
            }
            crate::message::UserContent::Video(_) => {
                return Err(crate::message::MessageError::ConversionError(
                    "Ollama does not support video user content".into(),
                ));
            }
            crate::message::UserContent::ToolResult(_) => {
                return Err(crate::message::MessageError::ConversionError(
                    "tool results must be converted to a separate Ollama message".into(),
                ));
            }
        }
    }

    Ok(Message::User {
        content: texts.join(" "),
        images: (!images.is_empty()).then_some(images),
        name: None,
    })
}

/// Conversion from an internal Rig message (crate::message::Message) to a provider Message.
/// (Only User and Assistant variants are supported.)
impl TryFrom<crate::message::Message> for Vec<Message> {
    type Error = crate::message::MessageError;
    fn try_from(internal_msg: crate::message::Message) -> Result<Self, Self::Error> {
        use crate::message::Message as InternalMessage;
        match internal_msg {
            InternalMessage::System { content } => Ok(vec![Message::System {
                content,
                images: None,
                name: None,
            }]),
            InternalMessage::User { content, .. } => {
                let mut messages = Vec::new();
                let mut pending_user_content = Vec::new();

                for content in content {
                    match content {
                        crate::message::UserContent::ToolResult(crate::message::ToolResult {
                            id,
                            content,
                            ..
                        }) => {
                            if !pending_user_content.is_empty() {
                                messages.push(user_message_from_content(std::mem::take(
                                    &mut pending_user_content,
                                ))?);
                            }

                            let content = content
                                .into_iter()
                                .map(|content| match content {
                                    crate::message::ToolResultContent::Text(text) => Ok(text.text),
                                    crate::message::ToolResultContent::Json { value } => {
                                        Ok(value.to_string())
                                    }
                                    crate::message::ToolResultContent::Image(_) => {
                                        Err(crate::message::MessageError::ConversionError(
                                            "Ollama does not support images in tool results".into(),
                                        ))
                                    }
                                })
                                .collect::<Result<Vec<_>, _>>()?
                                .join("\n");
                            messages.push(Message::ToolResult { name: id, content });
                        }
                        content => pending_user_content.push(content),
                    }
                }

                if !pending_user_content.is_empty() {
                    messages.push(user_message_from_content(pending_user_content)?);
                }

                Ok(messages)
            }
            InternalMessage::Assistant { content, .. } => {
                let mut thinking: Option<String> = None;
                let mut text_content = Vec::new();
                let mut tool_calls = Vec::new();

                for content in content.into_iter() {
                    match content {
                        crate::message::AssistantContent::Text(text) => {
                            text_content.push(text.text)
                        }
                        crate::message::AssistantContent::ToolCall(tool_call) => {
                            tool_calls.push(tool_call)
                        }
                        crate::message::AssistantContent::Reasoning(reasoning) => {
                            let display = reasoning.display_text();
                            if !display.is_empty() {
                                thinking = Some(display);
                            }
                        }
                        crate::message::AssistantContent::Image(_) => {
                            return Err(crate::message::MessageError::ConversionError(
                                "Ollama currently doesn't support images.".into(),
                            ));
                        }
                    }
                }

                // `OneOrMany` ensures at least one `AssistantContent::Text` or `ToolCall` exists,
                //  so either `content` or `tool_calls` will have some content.
                Ok(vec![Message::Assistant {
                    content: text_content.join(" "),
                    thinking,
                    images: None,
                    name: None,
                    tool_calls: tool_calls
                        .into_iter()
                        .map(|tool_call| tool_call.into())
                        .collect::<Vec<_>>(),
                }])
            }
        }
    }
}

/// Conversion from provider Message to a completion message.
/// This is needed so that responses can be converted back into chat history.
impl From<Message> for crate::completion::Message {
    fn from(msg: Message) -> Self {
        match msg {
            Message::User { content, .. } => crate::completion::Message::User {
                content: OneOrMany::one(crate::completion::message::UserContent::Text(Text::new(
                    content,
                ))),
            },
            Message::Assistant {
                content,
                thinking,
                tool_calls,
                ..
            } => {
                let mut assistant_contents = Vec::new();
                // Preserve reasoning so it survives the round-trip (issue #1926).
                if let Some(thinking) = thinking.filter(|t| !t.is_empty()) {
                    assistant_contents.push(
                        crate::completion::message::AssistantContent::reasoning(thinking),
                    );
                }
                assistant_contents.push(crate::completion::message::AssistantContent::Text(
                    Text::new(content),
                ));
                for tc in tool_calls {
                    assistant_contents.push(
                        crate::completion::message::AssistantContent::tool_call(
                            tc.function.name.clone(),
                            tc.function.name,
                            tc.function.arguments,
                        ),
                    );
                }
                let content =
                    OneOrMany::from_iter_optional(assistant_contents).unwrap_or_else(|| {
                        OneOrMany::one(crate::completion::message::AssistantContent::Text(
                            Text::new(String::new()),
                        ))
                    });

                crate::completion::Message::Assistant { id: None, content }
            }
            // System and ToolResult are converted to User message as needed.
            Message::System { content, .. } => crate::completion::Message::User {
                content: OneOrMany::one(crate::completion::message::UserContent::Text(Text::new(
                    content,
                ))),
            },
            Message::ToolResult { name, content } => crate::completion::Message::User {
                content: OneOrMany::one(message::UserContent::tool_result(
                    name,
                    OneOrMany::one(message::ToolResultContent::text(content)),
                )),
            },
        }
    }
}

impl Message {
    /// Constructs a system message.
    pub fn system(content: &str) -> Self {
        Message::System {
            content: content.to_owned(),
            images: None,
            name: None,
        }
    }
}

// ---------- Additional Message Types ----------

impl From<crate::message::ToolCall> for ToolCall {
    fn from(tool_call: crate::message::ToolCall) -> Self {
        Self {
            r#type: ToolType::Function,
            function: Function {
                name: tool_call.function.name,
                arguments: tool_call.function.arguments,
            },
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct SystemContent {
    #[serde(default)]
    r#type: SystemContentType,
    text: String,
}

#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub enum SystemContentType {
    #[default]
    Text,
}

impl From<String> for SystemContent {
    fn from(s: String) -> Self {
        SystemContent {
            r#type: SystemContentType::default(),
            text: s,
        }
    }
}

impl FromStr for SystemContent {
    type Err = std::convert::Infallible;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(SystemContent {
            r#type: SystemContentType::default(),
            text: s.to_string(),
        })
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct AssistantContent {
    pub text: String,
}

impl FromStr for AssistantContent {
    type Err = std::convert::Infallible;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(AssistantContent { text: s.to_owned() })
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum UserContent {
    Text { text: String },
    Image { image_url: ImageUrl },
    // Audio variant removed as Ollama API does not support audio input.
}

impl FromStr for UserContent {
    type Err = std::convert::Infallible;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(UserContent::Text { text: s.to_owned() })
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct ImageUrl {
    pub url: String,
    #[serde(default)]
    pub detail: ImageDetail,
}

// ---------- Data-oriented face ----------

/// Ollama native `/api/chat` as config + pure functions.
///
/// The data-oriented face of the Ollama provider: a serde `Config`, a
/// [`DESCRIPTOR`](functions::DESCRIPTOR) capability sheet, and free functions
/// — [`build_request`](functions::build_request) (data → HTTP request, no IO)
/// and [`parse_response`](functions::parse_response) (bytes → normalized
/// [`completion::CompletionResponse`], no IO) — plus the async
/// [`complete`](functions::complete) and [`open_stream`](functions::open_stream)
/// wrappers over [`HttpRuntime`](crate::http_runtime::HttpRuntime).
///
/// The pure functions delegate to the provider's typed conversions
/// (`OllamaCompletionRequest` / [`CompletionResponse`]`::try_into`), so the
/// non-streaming and streaming paths share one wire format.
pub mod functions {
    use http::header::{AUTHORIZATION, CONTENT_TYPE};
    use serde::{Deserialize, Serialize};

    use super::{CompletionResponse, OllamaCompletionRequest};
    use crate::completion::{self, CompletionError, CompletionRequest};
    use crate::http_runtime::HttpRuntime;
    use crate::providers::descriptor::{
        ApiKeyLocation, ConfigError, ProviderDescriptor, optional_env_var,
    };
    use crate::telemetry::{CompletionOperation, completion_span};

    /// Default Ollama API base URL (local instance).
    pub const DEFAULT_BASE_URL: &str = super::OLLAMA_API_BASE_URL;

    /// Ollama's capability sheet.
    pub const DESCRIPTOR: ProviderDescriptor = ProviderDescriptor {
        name: "ollama",
        supports_tools: true,
        // `output_schema` maps to the native `format` field.
        supports_response_format: true,
        // The native NDJSON stream always reports counts on the final chunk;
        // there is no OpenAI-style `stream_options` opt-in.
        stream_include_usage: false,
        // Ollama emits whole tool calls in a single NDJSON chunk.
        emits_complete_single_chunk_tool_calls: true,
        // Ollama does not compose `format` with `tools`, so agentic
        // structured-output runs use the output tool instead of native
        // `format` whenever tools are present. Keep the descriptor faithful
        // to that recorded behavior.
        composes_native_output_with_tools: false,
        max_embedding_documents: Some(1024),
        verify_path: Some("api/tags"),
    };

    /// Plain-data Ollama provider configuration.
    ///
    /// Ollama requires no authentication by default
    /// ([`ApiKeyLocation::None`]); proxied or secured deployments may carry a
    /// Bearer token via `Env`/`Inline`.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[non_exhaustive]
    pub struct Config {
        /// API base URL (defaults to [`DEFAULT_BASE_URL`]).
        pub base_url: String,
        /// Credential location ([`ApiKeyLocation::None`] by default).
        pub api_key: ApiKeyLocation,
        /// Model identifier requests are built for.
        pub model: String,
        /// Extra headers attached to every request.
        pub extra_headers: Vec<(String, String)>,
    }

    impl Config {
        /// Config for `model` against a local unauthenticated Ollama.
        pub fn new(model: impl Into<String>) -> Self {
            Self {
                base_url: DEFAULT_BASE_URL.to_string(),
                api_key: ApiKeyLocation::None,
                model: model.into(),
                extra_headers: Vec::new(),
            }
        }

        /// Config for `model` built from the process environment.
        ///
        /// Reads `OLLAMA_API_BASE_URL` (optional override of
        /// [`DEFAULT_BASE_URL`]) and `OLLAMA_API_KEY` (**optional** — Ollama
        /// serves unauthenticated by default; only proxied deployments need a
        /// Bearer token). These are the same variables the deleted
        /// `ollama::Client::from_env` read, where an absent key became
        /// `OllamaApiKey::default()`, i.e. no `Authorization` header at all —
        /// mirrored here as [`ApiKeyLocation::None`] (an empty value is likewise
        /// treated as no credential, as `OllamaApiKey::from` did). When the key is
        /// present and non-empty it is stored as [`ApiKeyLocation::Env`], so the
        /// secret is read at request time rather than held inside the config.
        ///
        /// # Errors
        /// [`ConfigError`] when a variable holds invalid Unicode.
        pub fn from_env(model: impl Into<String>) -> Result<Self, ConfigError> {
            let mut cfg = Self::new(model);
            if let Some(base_url) = optional_env_var("OLLAMA_API_BASE_URL")? {
                cfg.base_url = base_url;
            }
            cfg.api_key = match optional_env_var("OLLAMA_API_KEY")? {
                Some(key) if !key.is_empty() => ApiKeyLocation::Env("OLLAMA_API_KEY".to_string()),
                _ => ApiKeyLocation::None,
            };
            Ok(cfg)
        }

        /// Config for `model` with an explicit Bearer token (proxied
        /// deployments).
        pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
            self.api_key = ApiKeyLocation::Inline(key.into());
            self
        }

        /// Override the API base URL.
        pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
            self.base_url = base_url.into();
            self
        }
    }

    /// Build the serialized native `/api/chat` request body for `request`.
    ///
    /// Pure: the exact bytes the wire sees. `stream` sets the body's
    /// `stream` flag.
    pub fn build_request_body(
        cfg: &Config,
        request: &CompletionRequest,
        stream: bool,
    ) -> Result<Vec<u8>, CompletionError> {
        let mut typed = OllamaCompletionRequest::try_from((cfg.model.as_str(), request.clone()))?;
        typed.stream = stream;
        Ok(serde_json::to_vec(&typed)?)
    }

    /// Build the complete HTTP request (URL, headers, body) for `request`.
    ///
    /// Pure except for credential resolution (`ApiKeyLocation::Env` reads
    /// the environment). A resolved key becomes a Bearer `Authorization`
    /// header; [`ApiKeyLocation::None`] sends no credential.
    pub fn build_request(
        cfg: &Config,
        request: &CompletionRequest,
        stream: bool,
    ) -> Result<http::Request<Vec<u8>>, CompletionError> {
        let url = format!("{}/api/chat", cfg.base_url.trim_end_matches('/'));
        let body = build_request_body(cfg, request, stream)?;

        let mut builder = http::Request::post(url).header(CONTENT_TYPE, "application/json");
        if let Some(key) = cfg
            .api_key
            .resolve()
            .map_err(|e| CompletionError::RequestError(Box::new(e)))?
        {
            builder = builder.header(AUTHORIZATION, format!("Bearer {key}"));
        }
        for (name, value) in &cfg.extra_headers {
            builder = builder.header(name.as_str(), value.as_str());
        }
        builder
            .body(body)
            .map_err(|e| CompletionError::RequestError(Box::new(e)))
    }

    /// Parse a native `/api/chat` response body into the normalized
    /// [`completion::CompletionResponse`]. Pure.
    pub fn parse_response(
        status: http::StatusCode,
        body: &str,
    ) -> Result<completion::CompletionResponse, CompletionError> {
        if !status.is_success() {
            return Err(CompletionError::from_http_response(
                status,
                body.to_string(),
            ));
        }
        let response: CompletionResponse = serde_json::from_str(body)?;
        response.try_into()
    }

    /// Open a streaming completion for `request` over the native NDJSON
    /// `/api/chat` stream, reusing the provider's stream machinery.
    pub async fn open_stream(
        cfg: &Config,
        rt: &HttpRuntime,
        request: CompletionRequest,
    ) -> Result<crate::streaming::StreamingCompletionResponse, CompletionError> {
        use tracing_futures::Instrument;

        let model = request.model.clone().unwrap_or_else(|| cfg.model.clone());
        let span = completion_span(
            DESCRIPTOR.name,
            &model,
            CompletionOperation::ChatStreaming,
            &request,
        );
        let req = build_request(cfg, &request, true)?;
        // Ollama's native stream is NDJSON, not SSE, so this path takes the
        // raw byte-stream transport edge rather than `HttpRuntime::sse_events`.
        // `consume_chat_streaming_response` already consumes the type-erased
        // `http_client::StreamingResponse`, so no genericity leaks out.
        let response = rt.send_streaming(req).await?;
        // `consume_chat_streaming_response` records usage and response
        // metadata onto `tracing::Span::current()`, so the stream must run
        // inside the completion span — this is what the deleted
        // `CompletionModel::stream` did with `.instrument(span)`.
        let stream = super::consume_chat_streaming_response(response)
            .instrument(span.clone())
            .await?;
        Ok(crate::streaming::StreamingCompletionResponse::stream(
            Box::pin(stream.instrument(span)),
        ))
    }

    /// Send `request` to Ollama and return the normalized response.
    pub async fn complete(
        cfg: &Config,
        rt: &HttpRuntime,
        request: CompletionRequest,
    ) -> Result<completion::CompletionResponse, CompletionError> {
        let req = build_request(cfg, &request, false)?;
        let (status, body) = rt.send(req).await?;
        parse_response(status, &body)
    }

    // ================================================================
    // Embeddings
    // ================================================================

    /// Plain-data Ollama embeddings configuration.
    ///
    /// A sibling of [`Config`]: embeddings target their own model
    /// identifier.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[non_exhaustive]
    pub struct EmbeddingConfig {
        /// API base URL (defaults to [`DEFAULT_BASE_URL`]).
        pub base_url: String,
        /// Credential location ([`ApiKeyLocation::None`] by default).
        pub api_key: ApiKeyLocation,
        /// Embedding model identifier requests are built for.
        pub model: String,
        /// Dimensionality of the vectors this model returns.
        ///
        /// The data form of the deleted `EmbeddingModel::ndims()`, which the
        /// classic model took at construction
        /// (`Client::embedding_model_with_ndims`) and reported to callers
        /// sizing a vector-store index. Ollama's `/api/embed` request has no
        /// dimensionality parameter, so — exactly as before — this never
        /// reaches the wire; `build_embedding_body`
        /// sends only `model` and `input`.
        ///
        /// [`new`](Self::new) seeds it from
        /// [`model_dimensions_from_identifier`](super::model_dimensions_from_identifier),
        /// the same lookup the classic `make` used for a known model.
        pub ndims: Option<usize>,
        /// Extra headers attached to every request.
        pub extra_headers: Vec<(String, String)>,
    }

    impl EmbeddingConfig {
        /// Config for `model` against a local unauthenticated Ollama.
        pub fn new(model: impl Into<String>) -> Self {
            let model = model.into();
            let ndims = super::model_dimensions_from_identifier(&model);
            Self {
                base_url: DEFAULT_BASE_URL.to_string(),
                api_key: ApiKeyLocation::None,
                model,
                ndims,
                extra_headers: Vec::new(),
            }
        }

        /// Declare the dimensionality of the vectors this model returns.
        ///
        /// The replacement for `Client::embedding_model_with_ndims`, for
        /// models the built-in lookup does not know.
        pub fn with_ndims(mut self, ndims: usize) -> Self {
            self.ndims = Some(ndims);
            self
        }

        /// Config for `model` built from the process environment.
        ///
        /// Same variables as [`Config::from_env`]: `OLLAMA_API_BASE_URL`
        /// (optional base-URL override) and `OLLAMA_API_KEY` (**optional**; an
        /// absent or empty value leaves the config credential-less, matching the
        /// classic client's `OllamaApiKey::default()`, which emitted no
        /// `Authorization` header).
        ///
        /// # Errors
        /// [`ConfigError`] when a variable holds invalid Unicode.
        pub fn from_env(model: impl Into<String>) -> Result<Self, ConfigError> {
            let mut cfg = Self::new(model);
            if let Some(base_url) = optional_env_var("OLLAMA_API_BASE_URL")? {
                cfg.base_url = base_url;
            }
            cfg.api_key = match optional_env_var("OLLAMA_API_KEY")? {
                Some(key) if !key.is_empty() => ApiKeyLocation::Env("OLLAMA_API_KEY".to_string()),
                _ => ApiKeyLocation::None,
            };
            Ok(cfg)
        }

        /// Config for `model` with an explicit Bearer token (proxied
        /// deployments).
        pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
            self.api_key = ApiKeyLocation::Inline(key.into());
            self
        }

        /// Override the API base URL.
        pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
            self.base_url = base_url.into();
            self
        }
    }

    /// Build the complete HTTP `/api/embed` request for one chunk of
    /// `texts`.
    ///
    /// Pure except for credential resolution.
    pub fn build_embedding_request(
        cfg: &EmbeddingConfig,
        texts: &[String],
    ) -> Result<http::Request<Vec<u8>>, crate::embeddings::EmbeddingError> {
        use crate::embeddings::EmbeddingError;

        let body = super::build_embedding_body(&cfg.model, texts)?;
        let url = format!("{}/api/embed", cfg.base_url.trim_end_matches('/'));
        let mut builder = http::Request::post(url).header(CONTENT_TYPE, "application/json");
        if let Some(key) = cfg
            .api_key
            .resolve()
            .map_err(|e| EmbeddingError::ProviderError(e.to_string()))?
        {
            builder = builder.header(AUTHORIZATION, format!("Bearer {key}"));
        }
        for (name, value) in &cfg.extra_headers {
            builder = builder.header(name.as_str(), value.as_str());
        }
        builder
            .body(body)
            .map_err(|e| EmbeddingError::ProviderError(e.to_string()))
    }

    /// Parse an `/api/embed` response into the normalized
    /// [`crate::embeddings::EmbeddingResponse`]. Pure.
    pub fn parse_embedding_response(
        status: http::StatusCode,
        body: &str,
        documents: Vec<String>,
    ) -> Result<crate::embeddings::EmbeddingResponse, crate::embeddings::EmbeddingError> {
        super::parse_embedding_response(status, body, documents)
    }

    /// Embed `texts`, chunking to honor [`DESCRIPTOR`]'s
    /// `max_embedding_documents`; embeddings are returned in input order.
    pub async fn embed(
        cfg: &EmbeddingConfig,
        rt: &HttpRuntime,
        texts: Vec<String>,
    ) -> Result<crate::embeddings::EmbeddingResponse, crate::embeddings::EmbeddingError> {
        crate::embeddings::batching::embed_chunked(
            rt,
            texts,
            DESCRIPTOR.max_embedding_documents,
            |chunk| build_embedding_request(cfg, chunk),
            parse_embedding_response,
        )
        .await
    }

    /// Embed caller-defined batches, returning one order-aligned
    /// [`OneOrMany`](crate::OneOrMany) group per input batch plus summed
    /// usage.
    pub async fn embed_batches(
        cfg: &EmbeddingConfig,
        rt: &HttpRuntime,
        texts: Vec<Vec<String>>,
    ) -> Result<
        (
            Vec<crate::OneOrMany<crate::embeddings::Embedding>>,
            crate::completion::Usage,
        ),
        crate::embeddings::EmbeddingError,
    > {
        let (counts, flat) = crate::embeddings::batching::split_batches(texts);
        let response = embed(cfg, rt, flat).await?;
        let groups = crate::embeddings::batching::group_batches(&counts, response.embeddings)?;
        Ok((groups, response.usage))
    }

    /// Build the `GET /api/tags` request for [`list_models`].
    ///
    /// Pure except for credential resolution ([`ApiKeyLocation::None`] by
    /// default sends no auth header, matching a local Ollama).
    pub fn build_list_models_request(
        cfg: &Config,
    ) -> Result<http::Request<Vec<u8>>, crate::model::ModelListingError> {
        let url = format!(
            "{}{}",
            cfg.base_url.trim_end_matches('/'),
            super::LIST_MODELS_PATH
        );
        crate::providers::openai::functions::bearer_get(url, &cfg.api_key, &cfg.extra_headers)
    }

    /// List the locally available models.
    ///
    /// Parsing goes through the pure `super::parse_list_models_response`.
    pub async fn list_models(
        cfg: &Config,
        rt: &HttpRuntime,
    ) -> Result<crate::model::ModelList, crate::model::ModelListingError> {
        let req = build_list_models_request(cfg)?;
        let (status, body) = rt.send_bytes(req).await?;
        super::parse_list_models_response(status, &body)
    }
    /// Verify that `cfg`'s credential is accepted by the provider.
    ///
    /// The data-oriented replacement for the deleted `VerifyClient::verify`: the
    /// endpoint is [`DESCRIPTOR`]'s `verify_path` (`api/tags`, the value the
    /// deleted `Provider::VERIFY_PATH` carried) and the status mapping is the
    /// classic one — see [`crate::providers::verify`].
    ///
    /// # Errors
    /// [`VerifyError`](crate::providers::verify::VerifyError): invalid
    /// authentication on `401`/`403`, otherwise the preserved provider response
    /// or a transport failure.
    pub async fn verify(
        cfg: &Config,
        rt: &HttpRuntime,
    ) -> Result<(), crate::providers::verify::VerifyError> {
        crate::providers::verify::verify_bearer(
            &DESCRIPTOR,
            &cfg.base_url,
            &cfg.api_key,
            &cfg.extra_headers,
            rt,
        )
        .await
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::OneOrMany;
        use crate::message::Message;

        fn sample_request() -> CompletionRequest {
            CompletionRequest {
                model: None,
                chat_history: OneOrMany::one(Message::user("hello")),
                documents: Vec::new(),
                tools: Vec::new(),
                temperature: Some(0.5),
                max_tokens: Some(64),
                tool_choice: None,
                additional_params: None,
                output_schema: None,
                record_telemetry_content: false,
            }
        }

        #[test]
        fn build_request_sets_url_without_auth_by_default() {
            let cfg = Config::new("llama3.2");
            let req = build_request(&cfg, &sample_request(), false).expect("build");
            assert_eq!(req.uri(), "http://localhost:11434/api/chat");
            assert!(req.headers().get(http::header::AUTHORIZATION).is_none());
        }

        #[test]
        fn build_request_adds_bearer_when_key_configured() {
            let cfg = Config::new("llama3.2").with_api_key("secret");
            let req = build_request(&cfg, &sample_request(), false).expect("build");
            assert_eq!(
                req.headers()
                    .get(http::header::AUTHORIZATION)
                    .and_then(|v| v.to_str().ok()),
                Some("Bearer secret")
            );
        }

        #[test]
        fn build_request_body_injects_model_and_stream_flag() {
            let cfg = Config::new("llama3.2");
            let body = build_request_body(&cfg, &sample_request(), false).expect("build");
            let value: serde_json::Value = serde_json::from_slice(&body).expect("json");
            assert_eq!(value["model"], "llama3.2");
            assert_eq!(value["stream"], false);
            // temperature / max_tokens are model options in the native API.
            assert_eq!(value["options"]["temperature"], 0.5);
            assert_eq!(value["options"]["num_predict"], 64);

            let mut request = sample_request();
            request.model = Some("qwen3:8b".to_string());
            let body = build_request_body(&cfg, &request, true).expect("build");
            let value: serde_json::Value = serde_json::from_slice(&body).expect("json");
            assert_eq!(value["model"], "qwen3:8b");
            assert_eq!(value["stream"], true);
        }

        #[test]
        fn parse_response_normalizes() {
            let body = serde_json::json!({
                "model": "llama3.2",
                "created_at": "2024-01-01T00:00:00Z",
                "message": {"role": "assistant", "content": "hi"},
                "done": true,
                "done_reason": "stop",
                "prompt_eval_count": 3,
                "eval_count": 2
            })
            .to_string();
            let response = parse_response(http::StatusCode::OK, &body).expect("parse");
            assert_eq!(response.provider, "ollama");
            assert_eq!(response.model.as_deref(), Some("llama3.2"));
            assert_eq!(response.finish_reason, Some(completion::FinishReason::Stop));
            assert_eq!(response.usage.input_tokens, 3);
            assert_eq!(response.usage.total_tokens, 5);
        }

        #[test]
        fn parse_response_surfaces_http_errors() {
            let err = parse_response(http::StatusCode::NOT_FOUND, "model missing")
                .expect_err("non-success status must error");
            assert!(matches!(err, CompletionError::HttpError(_)));
        }
    }
}

// =================================================================
// Tests
// =================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn splits_legacy_reasoning_with_or_without_opening_marker() {
        assert_eq!(
            split_legacy_thinking("<think>private reasoning</think>\n\nvisible answer", false),
            (Some("private reasoning"), "visible answer")
        );
        assert_eq!(
            split_legacy_thinking("private reasoning\n</think>\n\nvisible answer", true),
            (Some("private reasoning"), "visible answer")
        );
    }

    #[test]
    fn leaves_unterminated_or_inline_reasoning_markers_visible() {
        assert_eq!(
            split_legacy_thinking("<think>unterminated", true),
            (None, "<think>unterminated")
        );
        assert_eq!(
            split_legacy_thinking("The literal marker is <think>.", true),
            (None, "The literal marker is <think>.")
        );
        assert_eq!(
            split_legacy_thinking("  visible indentation", true),
            (None, "  visible indentation")
        );
        assert_eq!(
            split_legacy_thinking("The closing token </think> is XML-like.", true),
            (None, "The closing token </think> is XML-like.")
        );
        assert_eq!(
            split_legacy_thinking("Example:\n</think>\nis a closing tag.", true),
            (None, "Example:\n</think>\nis a closing tag.")
        );
    }

    // Test deserialization and conversion for the /api/chat endpoint.
    #[tokio::test]
    async fn test_chat_completion() {
        // Sample JSON response from /api/chat (non-streaming) based on Ollama docs.
        let sample_chat_response = json!({
            "model": "llama3.2",
            "created_at": "2023-08-04T19:22:45.499127Z",
            "message": {
                "role": "assistant",
                "content": "The sky is blue because of Rayleigh scattering.",
                "images": null,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_current_weather",
                            "arguments": {
                                "location": "San Francisco, CA",
                                "format": "celsius"
                            }
                        }
                    }
                ]
            },
            "done": true,
            "total_duration": 8000000000u64,
            "load_duration": 6000000u64,
            "prompt_eval_count": 61u64,
            "prompt_eval_duration": 400000000u64,
            "eval_count": 468u64,
            "eval_duration": 7700000000u64
        });
        let sample_text = sample_chat_response.to_string();

        let chat_resp: CompletionResponse =
            serde_json::from_str(&sample_text).expect("Invalid JSON structure");
        let conv: completion::CompletionResponse = chat_resp.try_into().unwrap();
        assert!(
            !conv.choice.is_empty(),
            "Expected non-empty choice in chat response"
        );
    }

    // Test conversion from provider Message to completion Message.
    #[test]
    fn test_message_conversion() {
        // Construct a provider Message (User variant with String content).
        let provider_msg = Message::User {
            content: "Test message".to_owned(),
            images: None,
            name: None,
        };
        // Convert it into a completion::Message.
        let comp_msg: crate::completion::Message = provider_msg.into();
        match comp_msg {
            crate::completion::Message::User { content } => {
                // Assume OneOrMany<T> has a method first() to access the first element.
                let first_content = content.first();
                // The expected type is crate::completion::message::UserContent::Text wrapping a Text struct.
                match first_content {
                    crate::completion::message::UserContent::Text(text_struct) => {
                        assert_eq!(text_struct.text, "Test message");
                    }
                    _ => panic!("Expected text content in conversion"),
                }
            }
            _ => panic!("Conversion from provider Message to completion Message failed"),
        }
    }

    #[test]
    fn mixed_user_content_preserves_message_order() {
        use crate::OneOrMany;
        use crate::message::{Message as RigMessage, ToolResultContent, UserContent};

        let message = RigMessage::User {
            content: OneOrMany::many(vec![
                UserContent::text("before"),
                UserContent::tool_result(
                    "lookup",
                    OneOrMany::one(ToolResultContent::json(json!({ "ok": true }))),
                ),
                UserContent::text("after"),
            ])
            .expect("mixed content is non-empty"),
        };

        let messages = Vec::<Message>::try_from(message).expect("mixed content should convert");
        assert_eq!(messages.len(), 3);
        assert!(matches!(
            &messages[0],
            Message::User { content, .. } if content == "before"
        ));
        assert!(matches!(
            &messages[1],
            Message::ToolResult { name, content }
                if name == "lookup" && content == r#"{"ok":true}"#
        ));
        assert!(matches!(
            &messages[2],
            Message::User { content, .. } if content == "after"
        ));
    }

    #[test]
    fn unsupported_user_content_returns_a_conversion_error() {
        use crate::OneOrMany;
        use crate::message::{ImageMediaType, Message as RigMessage, UserContent};

        let message = RigMessage::User {
            content: OneOrMany::one(UserContent::image_url(
                "https://example.com/image.png",
                Some(ImageMediaType::PNG),
                None,
            )),
        };

        let error = Vec::<Message>::try_from(message).expect_err("URL image should be rejected");
        assert!(error.to_string().contains("base64"));
    }

    // Test conversion of internal tool definition to Ollama's ToolDefinition format.
    #[test]
    fn test_tool_definition_conversion() {
        // Internal tool definition from the completion module.
        let internal_tool = crate::completion::ToolDefinition {
            name: "get_current_weather".to_owned(),
            description: "Get the current weather for a location".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get the weather for, e.g. San Francisco, CA"
                    },
                    "format": {
                        "type": "string",
                        "description": "The format to return the weather in, e.g. 'celsius' or 'fahrenheit'",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location", "format"]
            }),
        };
        // Convert internal tool to Ollama's tool definition.
        let ollama_tool: ToolDefinition = internal_tool.into();
        assert_eq!(ollama_tool.type_field, "function");
        assert_eq!(ollama_tool.function.name, "get_current_weather");
        assert_eq!(
            ollama_tool.function.description,
            "Get the current weather for a location"
        );
        // Check JSON fields in parameters.
        let params = &ollama_tool.function.parameters;
        assert_eq!(params["properties"]["location"]["type"], "string");
    }

    // Test deserialization of chat response with thinking content
    #[tokio::test]
    async fn test_chat_completion_with_thinking() {
        let sample_response = json!({
            "model": "qwen-thinking",
            "created_at": "2023-08-04T19:22:45.499127Z",
            "message": {
                "role": "assistant",
                "content": "The answer is 42.",
                "thinking": "Let me think about this carefully. The question asks for the meaning of life...",
                "images": null,
                "tool_calls": []
            },
            "done": true,
            "total_duration": 8000000000u64,
            "load_duration": 6000000u64,
            "prompt_eval_count": 61u64,
            "prompt_eval_duration": 400000000u64,
            "eval_count": 468u64,
            "eval_duration": 7700000000u64
        });

        let chat_resp: CompletionResponse =
            serde_json::from_value(sample_response).expect("Failed to deserialize");

        // Verify thinking field is present
        if let Message::Assistant {
            thinking, content, ..
        } = &chat_resp.message
        {
            assert_eq!(
                thinking.as_ref().unwrap(),
                "Let me think about this carefully. The question asks for the meaning of life..."
            );
            assert_eq!(content, "The answer is 42.");
        } else {
            panic!("Expected Assistant message");
        }
    }

    // Test deserialization of chat response without thinking content
    #[tokio::test]
    async fn test_chat_completion_without_thinking() {
        let sample_response = json!({
            "model": "llama3.2",
            "created_at": "2023-08-04T19:22:45.499127Z",
            "message": {
                "role": "assistant",
                "content": "Hello!",
                "images": null,
                "tool_calls": []
            },
            "done": true,
            "total_duration": 8000000000u64,
            "load_duration": 6000000u64,
            "prompt_eval_count": 10u64,
            "prompt_eval_duration": 400000000u64,
            "eval_count": 5u64,
            "eval_duration": 7700000000u64
        });

        let chat_resp: CompletionResponse =
            serde_json::from_value(sample_response).expect("Failed to deserialize");

        // Verify thinking field is None when not provided
        if let Message::Assistant {
            thinking, content, ..
        } = &chat_resp.message
        {
            assert!(thinking.is_none());
            assert_eq!(content, "Hello!");
        } else {
            panic!("Expected Assistant message");
        }
    }

    // Test deserialization of streaming response with thinking content
    #[test]
    fn test_streaming_response_with_thinking() {
        let sample_chunk = json!({
            "model": "qwen-thinking",
            "created_at": "2023-08-04T19:22:45.499127Z",
            "message": {
                "role": "assistant",
                "content": "",
                "thinking": "Analyzing the problem...",
                "images": null,
                "tool_calls": []
            },
            "done": false
        });

        let chunk: CompletionResponse =
            serde_json::from_value(sample_chunk).expect("Failed to deserialize");

        if let Message::Assistant {
            thinking, content, ..
        } = &chunk.message
        {
            assert_eq!(thinking.as_ref().unwrap(), "Analyzing the problem...");
            assert_eq!(content, "");
        } else {
            panic!("Expected Assistant message");
        }
    }

    // Test message conversion with thinking content
    #[test]
    fn test_message_conversion_with_thinking() {
        // Create an internal message with reasoning content
        let reasoning_content = crate::message::Reasoning::new("Step 1: Consider the problem");

        let internal_msg = crate::message::Message::Assistant {
            id: None,
            content: crate::OneOrMany::many(vec![
                crate::message::AssistantContent::Reasoning(reasoning_content),
                crate::message::AssistantContent::Text(crate::message::Text::new(
                    "The answer is X".to_string(),
                )),
            ])
            .unwrap(),
        };

        // Convert to provider Message
        let provider_msgs: Vec<Message> = internal_msg.try_into().unwrap();
        assert_eq!(provider_msgs.len(), 1);

        if let Message::Assistant {
            thinking, content, ..
        } = &provider_msgs[0]
        {
            assert_eq!(thinking.as_ref().unwrap(), "Step 1: Consider the problem");
            assert_eq!(content, "The answer is X");
        } else {
            panic!("Expected Assistant message with thinking");
        }
    }

    /// Regression test for issue #1926: a non-streaming `/api/chat` response that
    /// carries `thinking` alongside `tool_calls` (the shape qwen3 thinking models
    /// emit on a tool-call turn) must surface the reasoning as an
    /// `AssistantContent::Reasoning` in `choice` — otherwise it never enters
    /// agent history and is never echoed back to Ollama, degrading multi-turn
    /// tool-call accuracy. Before the fix `choice` contained only the `ToolCall`.
    #[tokio::test]
    async fn nonstreaming_response_preserves_thinking_as_reasoning() {
        let sample_response = json!({
            "model": "qwen3:4b",
            "created_at": "2023-08-04T19:22:45.499127Z",
            "message": {
                "role": "assistant",
                "content": "",
                "thinking": "The user asked for the weather in Berlin. I should call get_weather with location=Berlin.",
                "images": null,
                "tool_calls": [
                    { "type": "function", "function": { "name": "get_weather", "arguments": { "location": "Berlin" } } }
                ]
            },
            "done": true,
            "done_reason": "stop",
            "total_duration": 8000000000u64,
            "load_duration": 6000000u64,
            "prompt_eval_count": 61u64,
            "prompt_eval_duration": 400000000u64,
            "eval_count": 468u64,
            "eval_duration": 7700000000u64
        });

        let raw: CompletionResponse =
            serde_json::from_value(sample_response).expect("deserialize ollama response");
        let completed: completion::CompletionResponse =
            raw.try_into().expect("convert to completion response");

        let reasoning = completed.choice.iter().find_map(|c| match c {
            completion::AssistantContent::Reasoning(r) => Some(r.clone()),
            _ => None,
        });
        let has_tool_call = completed
            .choice
            .iter()
            .any(|c| matches!(c, completion::AssistantContent::ToolCall(_)));

        assert!(has_tool_call, "tool call should survive the conversion");
        let reasoning = reasoning.expect(
            "non-streaming response must surface `thinking` as AssistantContent::Reasoning (issue #1926)",
        );
        assert_eq!(
            reasoning.display_text(),
            "The user asked for the weather in Berlin. I should call get_weather with location=Berlin.",
        );
    }

    // Test empty thinking content is handled correctly
    #[test]
    fn test_empty_thinking_content() {
        let sample_response = json!({
            "model": "llama3.2",
            "created_at": "2023-08-04T19:22:45.499127Z",
            "message": {
                "role": "assistant",
                "content": "Response",
                "thinking": "",
                "images": null,
                "tool_calls": []
            },
            "done": true,
            "total_duration": 8000000000u64,
            "load_duration": 6000000u64,
            "prompt_eval_count": 10u64,
            "prompt_eval_duration": 400000000u64,
            "eval_count": 5u64,
            "eval_duration": 7700000000u64
        });

        let chat_resp: CompletionResponse =
            serde_json::from_value(sample_response).expect("Failed to deserialize");

        if let Message::Assistant {
            thinking, content, ..
        } = &chat_resp.message
        {
            // Empty string should still deserialize as Some("")
            assert_eq!(thinking.as_ref().unwrap(), "");
            assert_eq!(content, "Response");
        } else {
            panic!("Expected Assistant message");
        }
    }

    // Test thinking with tool calls
    #[test]
    fn test_thinking_with_tool_calls() {
        let sample_response = json!({
            "model": "qwen-thinking",
            "created_at": "2023-08-04T19:22:45.499127Z",
            "message": {
                "role": "assistant",
                "content": "Let me check the weather.",
                "thinking": "User wants weather info, I should use the weather tool",
                "images": null,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {
                                "location": "San Francisco"
                            }
                        }
                    }
                ]
            },
            "done": true,
            "total_duration": 8000000000u64,
            "load_duration": 6000000u64,
            "prompt_eval_count": 30u64,
            "prompt_eval_duration": 400000000u64,
            "eval_count": 50u64,
            "eval_duration": 7700000000u64
        });

        let chat_resp: CompletionResponse =
            serde_json::from_value(sample_response).expect("Failed to deserialize");

        if let Message::Assistant {
            thinking,
            content,
            tool_calls,
            ..
        } = &chat_resp.message
        {
            assert_eq!(
                thinking.as_ref().unwrap(),
                "User wants weather info, I should use the weather tool"
            );
            assert_eq!(content, "Let me check the weather.");
            assert_eq!(tool_calls.len(), 1);
            assert_eq!(tool_calls[0].function.name, "get_weather");
        } else {
            panic!("Expected Assistant message with thinking and tool calls");
        }
    }

    // Test that `think` and `keep_alive` are extracted as top-level params, not in `options`
    #[test]
    fn test_completion_request_with_think_param() {
        use crate::OneOrMany;
        use crate::completion::Message as CompletionMessage;
        use crate::message::{Text, UserContent};

        // Create a CompletionRequest with "think": true, "keep_alive", and "num_ctx" in additional_params
        let completion_request = CompletionRequest {
            model: None,
            chat_history: OneOrMany::many(vec![
                crate::message::Message::system("You are a helpful assistant.".to_string()),
                CompletionMessage::User {
                    content: OneOrMany::one(UserContent::Text(Text::new(
                        "What is 2 + 2?".to_string(),
                    ))),
                },
            ])
            .expect("non-empty history"),
            documents: vec![],
            tools: vec![],
            temperature: Some(0.7),
            max_tokens: Some(1024),
            tool_choice: None,
            additional_params: Some(json!({
                "think": true,
                "keep_alive": "-1m",
                "num_ctx": 4096
            })),
            output_schema: None,
            record_telemetry_content: false,
        };

        // Convert to OllamaCompletionRequest
        let ollama_request = OllamaCompletionRequest::try_from(("qwen3:8b", completion_request))
            .expect("Failed to create Ollama request");

        // Serialize to JSON
        let serialized =
            serde_json::to_value(&ollama_request).expect("Failed to serialize request");

        // Assert equality with expected JSON
        // - "tools" is skipped when empty (skip_serializing_if)
        // - "think" should be a top-level boolean, NOT in options
        // - "keep_alive" should be a top-level string, NOT in options
        // - "num_ctx" should be in options (it's a model parameter)
        let expected = json!({
            "model": "qwen3:8b",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "What is 2 + 2?"
                }
            ],
            "stream": false,
            "think": true,
            "keep_alive": "-1m",
            "options": {
                "temperature": 0.7,
                "num_predict": 1024,
                "num_ctx": 4096
            }
        });

        assert_eq!(serialized, expected);
    }

    // Test that `think` and `keep_alive` are extracted as top-level params, not in `options`
    #[test]
    fn test_completion_request_with_level_low_think_param() {
        use crate::OneOrMany;
        use crate::completion::Message as CompletionMessage;
        use crate::message::{Text, UserContent};

        // Create a CompletionRequest with "think": true, "keep_alive", and "num_ctx" in additional_params
        let completion_request = CompletionRequest {
            model: None,
            chat_history: OneOrMany::many(vec![
                crate::message::Message::system("You are a helpful assistant.".to_string()),
                CompletionMessage::User {
                    content: OneOrMany::one(UserContent::Text(Text::new(
                        "What is 2 + 2?".to_string(),
                    ))),
                },
            ])
            .expect("non-empty history"),
            documents: vec![],
            tools: vec![],
            temperature: Some(0.7),
            max_tokens: Some(1024),
            tool_choice: None,
            additional_params: Some(json!({
                "think": "low",
                "keep_alive": "-1m",
                "num_ctx": 4096
            })),
            output_schema: None,
            record_telemetry_content: false,
        };

        // Convert to OllamaCompletionRequest
        let ollama_request = OllamaCompletionRequest::try_from(("qwen3:8b", completion_request))
            .expect("Failed to create Ollama request");

        // Serialize to JSON
        let serialized =
            serde_json::to_value(&ollama_request).expect("Failed to serialize request");

        // Assert equality with expected JSON
        // - "tools" is skipped when empty (skip_serializing_if)
        // - "think" should be a top-level boolean, NOT in options
        // - "keep_alive" should be a top-level string, NOT in options
        // - "num_ctx" should be in options (it's a model parameter)
        let expected = json!({
            "model": "qwen3:8b",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "What is 2 + 2?"
                }
            ],
            "stream": false,
            "think": "low",
            "keep_alive": "-1m",
            "options": {
                "temperature": 0.7,
                "num_predict": 1024,
                "num_ctx": 4096
            }
        });

        assert_eq!(serialized, expected);
    }

    // Test that `think` and `keep_alive` are extracted as top-level params, not in `options`
    #[test]
    fn test_completion_request_with_level_medium_think_param() {
        use crate::OneOrMany;
        use crate::completion::Message as CompletionMessage;
        use crate::message::{Text, UserContent};

        // Create a CompletionRequest with "think": true, "keep_alive", and "num_ctx" in additional_params
        let completion_request = CompletionRequest {
            model: None,
            chat_history: OneOrMany::many(vec![
                crate::message::Message::system("You are a helpful assistant.".to_string()),
                CompletionMessage::User {
                    content: OneOrMany::one(UserContent::Text(Text::new(
                        "What is 2 + 2?".to_string(),
                    ))),
                },
            ])
            .expect("non-empty history"),
            documents: vec![],
            tools: vec![],
            temperature: Some(0.7),
            max_tokens: Some(1024),
            tool_choice: None,
            additional_params: Some(json!({
                "think": "medium",
                "keep_alive": "-1m",
                "num_ctx": 4096
            })),
            output_schema: None,
            record_telemetry_content: false,
        };

        // Convert to OllamaCompletionRequest
        let ollama_request = OllamaCompletionRequest::try_from(("qwen3:8b", completion_request))
            .expect("Failed to create Ollama request");

        // Serialize to JSON
        let serialized =
            serde_json::to_value(&ollama_request).expect("Failed to serialize request");

        // Assert equality with expected JSON
        // - "tools" is skipped when empty (skip_serializing_if)
        // - "think" should be a top-level boolean, NOT in options
        // - "keep_alive" should be a top-level string, NOT in options
        // - "num_ctx" should be in options (it's a model parameter)
        let expected = json!({
            "model": "qwen3:8b",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "What is 2 + 2?"
                }
            ],
            "stream": false,
            "think": "medium",
            "keep_alive": "-1m",
            "options": {
                "temperature": 0.7,
                "num_predict": 1024,
                "num_ctx": 4096
            }
        });

        assert_eq!(serialized, expected);
    }

    // Test that `think` and `keep_alive` are extracted as top-level params, not in `options`
    #[test]
    fn test_completion_request_with_level_high_think_param() {
        use crate::OneOrMany;
        use crate::completion::Message as CompletionMessage;
        use crate::message::{Text, UserContent};

        // Create a CompletionRequest with "think": true, "keep_alive", and "num_ctx" in additional_params
        let completion_request = CompletionRequest {
            model: None,
            chat_history: OneOrMany::many(vec![
                crate::message::Message::system("You are a helpful assistant.".to_string()),
                CompletionMessage::User {
                    content: OneOrMany::one(UserContent::Text(Text::new(
                        "What is 2 + 2?".to_string(),
                    ))),
                },
            ])
            .expect("non-empty history"),
            documents: vec![],
            tools: vec![],
            temperature: Some(0.7),
            max_tokens: Some(1024),
            tool_choice: None,
            additional_params: Some(json!({
                "think": "high",
                "keep_alive": "-1m",
                "num_ctx": 4096
            })),
            output_schema: None,
            record_telemetry_content: false,
        };

        // Convert to OllamaCompletionRequest
        let ollama_request = OllamaCompletionRequest::try_from(("qwen3:8b", completion_request))
            .expect("Failed to create Ollama request");

        // Serialize to JSON
        let serialized =
            serde_json::to_value(&ollama_request).expect("Failed to serialize request");

        // Assert equality with expected JSON
        // - "tools" is skipped when empty (skip_serializing_if)
        // - "think" should be a top-level boolean, NOT in options
        // - "keep_alive" should be a top-level string, NOT in options
        // - "num_ctx" should be in options (it's a model parameter)
        let expected = json!({
            "model": "qwen3:8b",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "What is 2 + 2?"
                }
            ],
            "stream": false,
            "think": "high",
            "keep_alive": "-1m",
            "options": {
                "temperature": 0.7,
                "num_predict": 1024,
                "num_ctx": 4096
            }
        });

        assert_eq!(serialized, expected);
    }

    // Test that `think` and `keep_alive` are extracted as top-level params, not in `options`
    #[test]
    fn test_completion_request_with_level_invalid_think_param() {
        use crate::OneOrMany;
        use crate::completion::Message as CompletionMessage;
        use crate::message::{Text, UserContent};

        // Create a CompletionRequest with "think": true, "keep_alive", and "num_ctx" in additional_params
        let completion_request = CompletionRequest {
            model: None,
            chat_history: OneOrMany::many(vec![
                crate::message::Message::system("You are a helpful assistant.".to_string()),
                CompletionMessage::User {
                    content: OneOrMany::one(UserContent::Text(Text::new(
                        "What is 2 + 2?".to_string(),
                    ))),
                },
            ])
            .expect("non-empty history"),
            documents: vec![],
            tools: vec![],
            temperature: Some(0.7),
            max_tokens: Some(1024),
            tool_choice: None,
            additional_params: Some(json!({
                "think": "invalid",
                "keep_alive": "-1m",
                "num_ctx": 4096
            })),
            output_schema: None,
            record_telemetry_content: false,
        };

        // Convert to OllamaCompletionRequest
        let ollama_request = OllamaCompletionRequest::try_from(("qwen3:8b", completion_request));

        assert!(ollama_request.is_err())
    }

    // Test that `think` is omitted when not specified, so Ollama applies the
    // model's default thinking behavior (issue #1970)
    #[test]
    fn test_completion_request_with_think_omitted_by_default() {
        use crate::OneOrMany;
        use crate::completion::Message as CompletionMessage;
        use crate::message::{Text, UserContent};

        // Create a CompletionRequest WITHOUT "think" in additional_params
        let completion_request = CompletionRequest {
            model: None,
            chat_history: OneOrMany::many(vec![
                crate::message::Message::system("You are a helpful assistant.".to_string()),
                CompletionMessage::User {
                    content: OneOrMany::one(UserContent::Text(Text::new("Hello!".to_string()))),
                },
            ])
            .expect("non-empty history"),
            documents: vec![],
            tools: vec![],
            temperature: Some(0.5),
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        };

        // Convert to OllamaCompletionRequest
        let ollama_request = OllamaCompletionRequest::try_from(("llama3.2", completion_request))
            .expect("Failed to create Ollama request");

        // Serialize to JSON
        let serialized =
            serde_json::to_value(&ollama_request).expect("Failed to serialize request");

        // Assert that "think" is absent (so Ollama uses the model default) and
        // "keep_alive" is not present
        let expected = json!({
            "model": "llama3.2",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "Hello!"
                }
            ],
            "stream": false,
            "options": {
                "temperature": 0.5
            }
        });

        assert_eq!(serialized, expected);
    }

    // The native API takes the token limit as `options.num_predict`; an
    // explicit `num_predict` in `additional_params` wins over
    // `CompletionRequest::max_tokens`.
    #[test]
    fn test_completion_request_num_predict_from_additional_params_wins() {
        use crate::OneOrMany;
        use crate::completion::Message as CompletionMessage;
        use crate::message::{Text, UserContent};

        let completion_request = CompletionRequest {
            model: None,
            chat_history: OneOrMany::one(CompletionMessage::User {
                content: OneOrMany::one(UserContent::Text(Text::new("Hello!".to_string()))),
            }),
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: Some(1024),
            tool_choice: None,
            additional_params: Some(json!({ "num_predict": 42 })),
            output_schema: None,
            record_telemetry_content: false,
        };

        let ollama_request = OllamaCompletionRequest::try_from(("llama3.2", completion_request))
            .expect("Failed to create Ollama request");
        let serialized =
            serde_json::to_value(&ollama_request).expect("Failed to serialize request");

        assert_eq!(serialized["options"], json!({ "num_predict": 42 }));
        assert_eq!(serialized.get("max_tokens"), None);
    }

    // The plain path: `max_tokens` with no `additional_params` at all, which
    // skips the merge and serializes `base_options` directly. Every other
    // `max_tokens` test also sets `additional_params`, so without this one the
    // branch the fix exists for is never exercised.
    #[test]
    fn test_completion_request_num_predict_without_additional_params() {
        use crate::OneOrMany;
        use crate::completion::Message as CompletionMessage;
        use crate::message::{Text, UserContent};

        let completion_request = CompletionRequest {
            model: None,
            chat_history: OneOrMany::one(CompletionMessage::User {
                content: OneOrMany::one(UserContent::Text(Text::new("Hello!".to_string()))),
            }),
            documents: vec![],
            tools: vec![],
            temperature: Some(0.7),
            max_tokens: Some(1024),
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        };

        let ollama_request = OllamaCompletionRequest::try_from(("llama3.2", completion_request))
            .expect("Failed to create Ollama request");
        let serialized =
            serde_json::to_value(&ollama_request).expect("Failed to serialize request");

        assert_eq!(
            serialized["options"],
            json!({ "temperature": 0.7, "num_predict": 1024 })
        );
        // Neither belongs at the top level of a native `/api/chat` payload.
        assert_eq!(serialized.get("max_tokens"), None);
        assert_eq!(serialized.get("temperature"), None);
    }

    // With nothing to put in it, `options` is an empty object rather than
    // carrying `"temperature": null` as it did when temperature was seeded
    // unconditionally.
    #[test]
    fn test_completion_request_options_omit_unset_parameters() {
        use crate::OneOrMany;
        use crate::completion::Message as CompletionMessage;
        use crate::message::{Text, UserContent};

        let completion_request = CompletionRequest {
            model: None,
            chat_history: OneOrMany::one(CompletionMessage::User {
                content: OneOrMany::one(UserContent::Text(Text::new("Hello!".to_string()))),
            }),
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        };

        let ollama_request = OllamaCompletionRequest::try_from(("llama3.2", completion_request))
            .expect("Failed to create Ollama request");
        let serialized =
            serde_json::to_value(&ollama_request).expect("Failed to serialize request");

        assert_eq!(serialized["options"], json!({}));
    }

    #[test]
    fn test_completion_request_with_output_schema() {
        use crate::OneOrMany;
        use crate::completion::Message as CompletionMessage;
        use crate::message::{Text, UserContent};

        let schema: schemars::Schema = serde_json::from_value(json!({
            "type": "object",
            "properties": {
                "age": { "type": "integer" },
                "available": { "type": "boolean" }
            },
            "required": ["age", "available"]
        }))
        .expect("Failed to parse schema");

        let completion_request = CompletionRequest {
            model: Some("llama3.1".to_string()),
            chat_history: OneOrMany::one(CompletionMessage::User {
                content: OneOrMany::one(UserContent::Text(Text::new(
                    "How old is Ollama?".to_string(),
                ))),
            }),
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: Some(schema),
            record_telemetry_content: false,
        };

        let ollama_request = OllamaCompletionRequest::try_from(("llama3.1", completion_request))
            .expect("Failed to create Ollama request");

        let serialized =
            serde_json::to_value(&ollama_request).expect("Failed to serialize request");

        let format = serialized
            .get("format")
            .expect("format field should be present");
        assert_eq!(
            *format,
            json!({
                "type": "object",
                "properties": {
                    "age": { "type": "integer" },
                    "available": { "type": "boolean" }
                },
                "required": ["age", "available"]
            })
        );
    }

    #[test]
    fn test_completion_request_without_output_schema() {
        use crate::OneOrMany;
        use crate::completion::Message as CompletionMessage;
        use crate::message::{Text, UserContent};

        let completion_request = CompletionRequest {
            model: Some("llama3.1".to_string()),
            chat_history: OneOrMany::one(CompletionMessage::User {
                content: OneOrMany::one(UserContent::Text(Text::new("Hello!".to_string()))),
            }),
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        };

        let ollama_request = OllamaCompletionRequest::try_from(("llama3.1", completion_request))
            .expect("Failed to create Ollama request");

        let serialized =
            serde_json::to_value(&ollama_request).expect("Failed to serialize request");

        assert!(
            serialized.get("format").is_none(),
            "format field should be absent when output_schema is None"
        );
    }

    #[test]
    fn ndjson_buffer_returns_complete_lines_in_single_chunk() {
        let mut buf = NdjsonBuffer::new();
        let lines = buf.decode(b"{\"a\":1}\n{\"b\":2}\n");
        assert_eq!(lines, vec![b"{\"a\":1}".to_vec(), b"{\"b\":2}".to_vec()]);
    }

    #[test]
    fn ndjson_buffer_reassembles_line_split_across_chunks() {
        let mut buf = NdjsonBuffer::new();

        assert!(buf.decode(b"{\"model\":\"llama\",\"mes").is_empty());

        let lines = buf.decode(b"sage\":\"hi\"}\n{\"done\"");
        assert_eq!(
            lines,
            vec![b"{\"model\":\"llama\",\"message\":\"hi\"}".to_vec()]
        );

        let lines = buf.decode(b":true}\n");
        assert_eq!(lines, vec![b"{\"done\":true}".to_vec()]);
    }

    #[test]
    fn ndjson_buffer_skips_blank_lines() {
        let mut buf = NdjsonBuffer::new();
        let lines = buf.decode(b"\n{\"a\":1}\n\n");
        assert_eq!(lines, vec![b"{\"a\":1}".to_vec()]);
    }

    #[test]
    fn ndjson_buffer_retains_unterminated_trailing_data() {
        let mut buf = NdjsonBuffer::new();
        let lines = buf.decode(b"{\"a\":1}\n{\"b\":2");
        assert_eq!(lines, vec![b"{\"a\":1}".to_vec()]);
        let lines = buf.decode(b"}\n");
        assert_eq!(lines, vec![b"{\"b\":2}".to_vec()]);
    }

    #[test]
    fn ndjson_buffer_handles_empty_chunk() {
        let mut buf = NdjsonBuffer::new();
        assert!(buf.decode(b"").is_empty());

        buf.decode(b"{\"a\":1");
        assert!(buf.decode(b"").is_empty());

        let lines = buf.decode(b"}\n");
        assert_eq!(lines, vec![b"{\"a\":1}".to_vec()]);
    }

    #[test]
    fn ndjson_buffer_handles_multi_byte_utf8_split_across_chunks() {
        // `\n` (0x0A) cannot appear inside any UTF-8 continuation byte, so a
        // byte-wise newline scan is always safe — but verify explicitly that a
        // multi-byte sequence reassembles correctly when split across chunks.
        let mut buf = NdjsonBuffer::new();
        assert!(buf.decode(&[0xd0]).is_empty());
        assert!(buf.decode(&[0xb8, 0xd0, 0xb7, 0xd0]).is_empty());
        assert!(
            buf.decode(&[
                0xb2, 0xd0, 0xb5, 0xd1, 0x81, 0xd1, 0x82, 0xd0, 0xbd, 0xd0, 0xb8
            ])
            .is_empty()
        );

        let lines = buf.decode(b"\n");
        assert_eq!(lines.len(), 1);
        assert_eq!(std::str::from_utf8(&lines[0]).unwrap(), "известни");
    }

    #[test]
    fn ndjson_buffer_yields_parseable_chunks_when_split_arbitrarily() {
        let original = concat!(
            "{\"model\":\"llama3.2\",\"message\":{\"role\":\"assistant\",\"content\":\"hi\"},\"done\":false}\n",
            "{\"model\":\"llama3.2\",\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true}\n",
        );

        let mut buf = NdjsonBuffer::new();
        let mut received = Vec::new();
        for byte in original.as_bytes() {
            for line in buf.decode(std::slice::from_ref(byte)) {
                let parsed: serde_json::Value =
                    serde_json::from_slice(&line).expect("each drained line must be valid JSON");
                received.push(parsed);
            }
        }

        assert_eq!(received.len(), 2);
        assert_eq!(received[0]["message"]["content"], "hi");
        assert_eq!(received[1]["done"], true);
    }

    // Proves a non-success HTTP response from `/api/chat` preserves the
    // provider's status + body through the `provider_response_*` helpers
    // (issue #1931).
    #[tokio::test]
    async fn completion_non_success_preserves_status_and_body() {
        use crate::http_runtime::HttpRuntime;
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"error":"model not found"}"#;
        let rt = HttpRuntime::recording(RecordingHttpClient::with_error_response(
            http::StatusCode::SERVICE_UNAVAILABLE,
            body,
        ));
        let cfg = functions::Config::new(LLAMA3_2).with_api_key("test-key");
        let request = crate::completion::CompletionRequest::from_prompt("hello");

        let error = functions::complete(&cfg, &rt, request)
            .await
            .expect_err("should fail with non-success status");

        assert!(matches!(error, CompletionError::HttpError(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::SERVICE_UNAVAILABLE)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }

    // Proves a non-success HTTP response from `/api/embed` preserves the
    // provider's status + body through the `provider_response_*` helpers
    // (issue #1931).
    #[tokio::test]
    async fn embeddings_non_success_preserves_status_and_body() {
        use crate::http_runtime::HttpRuntime;
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"error":"model not found"}"#;
        let rt = HttpRuntime::recording(RecordingHttpClient::with_error_response(
            http::StatusCode::SERVICE_UNAVAILABLE,
            body,
        ));
        let cfg = functions::EmbeddingConfig::new(ALL_MINILM).with_api_key("test-key");

        let error = functions::embed(&cfg, &rt, vec!["hello".to_string()])
            .await
            .expect_err("should fail with non-success status");

        assert!(matches!(error, EmbeddingError::HttpError(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::SERVICE_UNAVAILABLE)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }
}

#[cfg(test)]
mod telemetry_tests {
    use super::functions;
    use crate::OneOrMany;
    use crate::completion::CompletionRequest;
    use crate::http_runtime::HttpRuntime;
    use crate::message::Message;
    use crate::test_utils::MockStreamingClient;
    use futures::StreamExt;
    use std::io;
    use std::sync::{Arc, Mutex};

    #[derive(Clone)]
    struct SharedWriter(Arc<Mutex<Vec<u8>>>);

    impl io::Write for SharedWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            if let Ok(mut sink) = self.0.lock() {
                sink.extend_from_slice(buf);
            }
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn sample_request() -> CompletionRequest {
        CompletionRequest {
            model: None,
            chat_history: OneOrMany::one(Message::user("hello")),
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

    /// The deleted `ollama::CompletionModel::stream` built a
    /// `CompletionSpanBuilder` span (`ollama`, `ChatStreaming`) and
    /// instrumented the NDJSON stream with it; `functions::open_stream` had
    /// lost it, so `record_token_usage` wrote to whatever ambient span
    /// existed.
    #[tokio::test]
    async fn open_stream_emits_the_chat_streaming_completion_span() {
        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard().await;
        let captured = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .with_ansi(false)
            .without_time()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::NEW)
            .with_writer({
                let captured = captured.clone();
                move || SharedWriter(captured.clone())
            })
            .finish();
        let _guard = tracing::subscriber::set_default(subscriber);

        // Ollama's native stream is NDJSON, one JSON object per line.
        let ndjson = format!(
            "{}\n{}\n",
            serde_json::json!({
                "model": "llama3.2",
                "created_at": "2024-01-01T00:00:00Z",
                "message": { "role": "assistant", "content": "hi" },
                "done": false
            }),
            serde_json::json!({
                "model": "llama3.2",
                "created_at": "2024-01-01T00:00:00Z",
                "message": { "role": "assistant", "content": "" },
                "done": true,
                "done_reason": "stop",
                "prompt_eval_count": 3,
                "eval_count": 2
            })
        );
        let rt = HttpRuntime::mock_streaming(MockStreamingClient {
            sse_bytes: bytes::Bytes::from(ndjson),
        });
        let cfg = functions::Config::new("llama3.2");

        let mut stream = functions::open_stream(&cfg, &rt, sample_request())
            .await
            .expect("stream opens");
        while stream.next().await.is_some() {}

        let logs = String::from_utf8(captured.lock().map(|sink| sink.clone()).unwrap_or_default())
            .expect("utf8 logs");
        assert!(
            logs.contains("chat_streaming"),
            "no chat_streaming span was created: {logs}"
        );
        assert!(
            logs.contains("ollama"),
            "span did not carry the provider name: {logs}"
        );
        assert!(
            logs.contains("llama3.2"),
            "span did not carry the request model: {logs}"
        );
    }
}

#[cfg(test)]
mod ndims_tests {
    use super::functions::{EmbeddingConfig, build_embedding_request};
    use super::{ALL_MINILM, NOMIC_EMBED_TEXT, model_dimensions_from_identifier};

    /// The deleted `EmbeddingModel` carried an `ndims` and reported it through
    /// `EmbeddingModel::ndims()`; `EmbeddingConfig` now carries the same
    /// value, seeded from the same lookup table the classic `make` used.
    #[test]
    fn embedding_config_carries_ndims_for_known_models() {
        assert_eq!(
            EmbeddingConfig::new(ALL_MINILM).ndims,
            model_dimensions_from_identifier(ALL_MINILM)
        );
        assert_eq!(EmbeddingConfig::new(ALL_MINILM).ndims, Some(384));
        assert_eq!(EmbeddingConfig::new(NOMIC_EMBED_TEXT).ndims, Some(768));
        assert_eq!(EmbeddingConfig::new("custom-local-model").ndims, None);
        assert_eq!(
            EmbeddingConfig::new("custom-local-model")
                .with_ndims(512)
                .ndims,
            Some(512)
        );
    }

    /// Ollama's `/api/embed` request has no dimensionality parameter, and the
    /// classic model never sent one either — `ndims` is carried, not
    /// serialized.
    #[test]
    fn ndims_does_not_reach_the_request_body() {
        let cfg = EmbeddingConfig::new(ALL_MINILM);
        let req = build_embedding_request(&cfg, &["hello".to_string()]).expect("build");
        let value: serde_json::Value = serde_json::from_slice(req.body()).expect("json");
        assert_eq!(value["model"], ALL_MINILM);
        assert!(value.get("ndims").is_none());
        assert!(value.get("dimensions").is_none());
    }
}
