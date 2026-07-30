//! GitHub Copilot provider.
//!
//! Supports Chat Completions, Responses, and Embeddings against
//! `https://api.githubcopilot.com`. Codex-class models route through
//! `/responses` and conversational models through `/chat/completions`;
//! [`functions::build_request`] picks the route from the configured model.
//!
//! # Example
//! ```no_run
//! use rig_core::providers::copilot;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! # let request = rig_core::completion::CompletionRequest::from_prompt("hello");
//! let cfg = copilot::functions::config_from_env(copilot::GPT_4O).await?;
//! let rt = rig_core::http_runtime::HttpRuntime::new();
//! let response = copilot::functions::complete(&cfg, &rt, request).await?;
//! # Ok(())
//! # }
//! ```

pub mod auth;
pub mod functions;

use crate::completion::{self, CompletionError};
use crate::embeddings::{self, EmbeddingError};
use crate::http_client;
use crate::model::{Model, ModelList, ModelListingError};
use crate::providers::internal::openai_chat_completions_compatible::{
    self, CompatibleChoice, CompatibleFinishReason, CompatibleToolCallChunk,
};
use crate::providers::openai;
use crate::providers::openai::responses_api::{self, CompletionRequest as ResponsesRequest};
use crate::streaming::{self, RawStreamingChoice, StreamingCompletionResponse};
use async_stream::stream;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::path::PathBuf;

/// Default GitHub Copilot API base URL (see [`functions::DEFAULT_BASE_URL`]).
pub const GITHUB_COPILOT_API_BASE_URL: &str = "https://api.githubcopilot.com";
pub(crate) const EDITOR_PLUGIN_VERSION: &str = "copilot-chat/0.35.0";
pub(crate) const USER_AGENT: &str = "GitHubCopilotChat/0.35.0";
pub(crate) const EDITOR_VERSION: &str = "vscode/1.107.0";
const API_VERSION: &str = "2025-04-01";

/// Copilot conversation intent sent in the `openai-intent` request header.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CopilotIntent {
    /// Generic chat panel conversation semantics.
    #[default]
    Panel,
    /// Edit-oriented conversation semantics.
    Edits,
}

impl CopilotIntent {
    fn as_header(self) -> &'static str {
        match self {
            Self::Panel => "conversation-panel",
            Self::Edits => "conversation-edits",
        }
    }
}

/// `gpt-4`
pub const GPT_4: &str = "gpt-4";
/// `gpt-4o`
pub const GPT_4O: &str = "gpt-4o";
/// `gpt-4o-mini`
pub const GPT_4O_MINI: &str = "gpt-4o-mini";
/// `gpt-4.1`
pub const GPT_4_1: &str = "gpt-4.1";
/// `gpt-4.1-mini`
pub const GPT_4_1_MINI: &str = "gpt-4.1-mini";
/// `gpt-4.1-nano`
pub const GPT_4_1_NANO: &str = "gpt-4.1-nano";
/// `gpt-5.3-codex`
pub const GPT_5_3_CODEX: &str = "gpt-5.3-codex";
/// `gpt-5.1-codex`
pub const GPT_5_1_CODEX: &str = "gpt-5.1-codex";
/// `gpt-5.5`
pub const GPT_5_5: &str = "gpt-5.5";
/// `gpt-5.4`
pub const GPT_5_4: &str = "gpt-5.4";
/// `claude-sonnet-4` completion model (Anthropic, via Copilot)
pub const CLAUDE_SONNET_4: &str = "claude-sonnet-4";
/// `claude-sonnet-4.6`
pub const CLAUDE_SONNET_4_6: &str = "claude-sonnet-4.6";
/// `claude-opus-4.6`
pub const CLAUDE_OPUS_4_6: &str = "claude-opus-4.6";
/// `claude-opus-4.7`
pub const CLAUDE_OPUS_4_7: &str = "claude-opus-4.7";
/// `claude-3.5-sonnet` completion model (Anthropic, via Copilot)
pub const CLAUDE_3_5_SONNET: &str = "claude-3.5-sonnet";
/// `gemini-3-flash-preview` completion model (Google, via Copilot)
pub const GEMINI_3_FLASH: &str = "gemini-3-flash-preview";
/// `gemini-3.1-pro-preview` completion model (Google, via Copilot)
pub const GEMINI_3_1_PRO_FLASH: &str = "gemini-3.1-pro-preview";
/// `gemini-2.0-flash-001` completion model (Google, via Copilot)
pub const GEMINI_2_0_FLASH: &str = "gemini-2.0-flash-001";
/// `o3-mini` reasoning model (OpenAI, via Copilot)
pub const O3_MINI: &str = "o3-mini";
/// `text-embedding-3-small`
pub const TEXT_EMBEDDING_3_SMALL: &str = "text-embedding-3-small";
/// `text-embedding-3-large`
pub const TEXT_EMBEDDING_3_LARGE: &str = "text-embedding-3-large";
/// `text-embedding-ada-002`
pub const TEXT_EMBEDDING_ADA_002: &str = "text-embedding-ada-002";

pub use openai::EncodingFormat;

pub(crate) fn default_headers(
    api_key: &str,
    initiator: &'static str,
    has_vision: bool,
    intent: CopilotIntent,
) -> Vec<(&'static str, String)> {
    let mut headers = vec![
        (
            http::header::AUTHORIZATION.as_str(),
            format!("Bearer {api_key}"),
        ),
        ("copilot-integration-id", "vscode-chat".to_string()),
        ("editor-version", EDITOR_VERSION.to_string()),
        ("editor-plugin-version", EDITOR_PLUGIN_VERSION.to_string()),
        ("user-agent", USER_AGENT.to_string()),
        ("openai-intent", intent.as_header().to_string()),
        ("x-github-api-version", API_VERSION.to_string()),
        ("x-request-id", crate::id::generate()),
        (
            "x-vscode-user-agent-library-version",
            "electron-fetch".to_string(),
        ),
        ("X-Initiator", initiator.to_string()),
    ];

    if has_vision {
        headers.push(("copilot-vision-request", "true".to_string()));
    }

    headers
}

pub(crate) fn apply_headers(
    builder: http_client::Builder,
    headers: &[(&'static str, String)],
) -> http_client::Builder {
    headers
        .iter()
        .fold(builder, |builder, (key, value)| builder.header(*key, value))
}

/// Derive the Copilot REST base URL from a chat token's `proxy-ep=` segment.
///
/// The endpoint is parsed from a credential string, not from explicit caller
/// configuration. For that reason, token-derived routing is limited to GitHub
/// Copilot service hosts and HTTPS. Callers that need a custom non-GitHub host
/// can still opt in explicitly with [`functions::Config::with_base_url`].
pub(crate) fn base_url_from_token(token: &str) -> Option<String> {
    let proxy_ep = token
        .split(';')
        .find_map(|part| part.trim().strip_prefix("proxy-ep="))?
        .trim();

    normalize_copilot_proxy_endpoint(proxy_ep)
}

fn normalize_copilot_proxy_endpoint(proxy_ep: &str) -> Option<String> {
    if proxy_ep.is_empty() {
        return None;
    }

    let candidate = if proxy_ep.starts_with("http://") || proxy_ep.starts_with("https://") {
        proxy_ep.to_string()
    } else {
        format!("https://{proxy_ep}")
    };

    let mut url = url::Url::parse(&candidate).ok()?;
    if url.scheme() != "https" || !url.username().is_empty() || url.password().is_some() {
        return None;
    }
    if url.path() != "/" || url.query().is_some() || url.fragment().is_some() {
        return None;
    }

    let host = url.host_str()?.to_ascii_lowercase();
    if !is_allowed_token_derived_copilot_host(&host) {
        return None;
    }

    let api_host = host
        .strip_prefix("proxy.")
        .map(|suffix| format!("api.{suffix}"))
        .unwrap_or(host);
    url.set_host(Some(&api_host)).ok()?;

    Some(url.to_string().trim_end_matches('/').to_string())
}

fn is_allowed_token_derived_copilot_host(host: &str) -> bool {
    host == "githubcopilot.com" || host.ends_with(".githubcopilot.com")
}

pub(crate) fn request_initiator(request: &completion::CompletionRequest) -> &'static str {
    for message in request.chat_history.iter() {
        match message {
            crate::completion::Message::Assistant { .. } => return "agent",
            crate::completion::Message::User { content } => {
                if content
                    .iter()
                    .any(|item| matches!(item, crate::message::UserContent::ToolResult(_)))
                {
                    return "agent";
                }
            }
            crate::completion::Message::System { .. } => {}
        }
    }

    "user"
}

pub(crate) fn request_has_vision(request: &completion::CompletionRequest) -> bool {
    request.chat_history.iter().any(|message| match message {
        crate::completion::Message::User { content } => content
            .iter()
            .any(|item| matches!(item, crate::message::UserContent::Image(_))),
        _ => false,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CompletionRoute {
    ChatCompletions,
    Responses,
}

/// Build the typed Copilot `/responses` request for `model`.
///
/// The single source of truth for the Responses route's request shape;
/// [`functions::build_request_body`] routes through it. Copilot's Responses endpoint expects strict function tool
/// schemas for reliable tool calls, so every tool is normalized to strict —
/// Chat Completions strict mode stays opt-in.
pub(crate) fn build_copilot_responses_request(
    model: String,
    completion_request: completion::CompletionRequest,
) -> Result<ResponsesRequest, CompletionError> {
    let mut request = ResponsesRequest::try_from(responses_api::ResponsesRequestParams {
        model,
        request: completion_request,
        system_instructions_placement:
            responses_api::SystemInstructionsPlacement::InputSystemMessages,
    })?;
    request.tools = request
        .tools
        .into_iter()
        .map(responses_api::ResponsesToolDefinition::with_strict)
        .collect();
    Ok(request)
}

pub(crate) fn route_for_model(model: &str) -> CompletionRoute {
    if model.to_ascii_lowercase().contains("codex") {
        CompletionRoute::Responses
    } else {
        CompletionRoute::ChatCompletions
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    #[serde(default)]
    pub object: Option<String>,
    #[serde(default)]
    pub created: Option<u64>,
    pub model: String,
    pub system_fingerprint: Option<String>,
    pub choices: Vec<ChatChoice>,
    pub usage: Option<openai::completion::Usage>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatChoice {
    #[serde(default)]
    pub index: usize,
    pub message: openai::completion::Message,
    pub logprobs: Option<serde_json::Value>,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

/// Maps an OpenAI-style `finish_reason` string onto the normalized
/// [`completion::FinishReason`] vocabulary, carrying unmapped values
/// verbatim in `Other`.
fn map_finish_reason(reason: &str) -> completion::FinishReason {
    match reason {
        "stop" => completion::FinishReason::Stop,
        "length" => completion::FinishReason::Length,
        "tool_calls" => completion::FinishReason::ToolCalls,
        "content_filter" => completion::FinishReason::ContentFilter,
        other => completion::FinishReason::Other(other.to_string()),
    }
}

impl TryFrom<ChatCompletionResponse> for completion::CompletionResponse {
    type Error = CompletionError;

    fn try_from(response: ChatCompletionResponse) -> Result<Self, Self::Error> {
        let first_choice = response.choices.first().ok_or_else(|| {
            CompletionError::ResponseError("Response contained no choices".to_owned())
        })?;

        let content = match &first_choice.message {
            openai::completion::Message::Assistant {
                content,
                tool_calls,
                ..
            } => {
                let mut content = content
                    .iter()
                    .filter_map(|c| {
                        let s = match c {
                            openai::completion::AssistantContent::Text { text } => text,
                            openai::completion::AssistantContent::Refusal { refusal } => refusal,
                        };
                        if s.is_empty() {
                            None
                        } else {
                            Some(completion::AssistantContent::text(s))
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
                Ok(content)
            }
            _ => Err(CompletionError::ResponseError(
                "Response did not contain a valid message or tool call".into(),
            )),
        }?;

        let choice = crate::OneOrMany::many(content).map_err(|_| {
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
                cached_input_tokens: usage
                    .prompt_tokens_details
                    .as_ref()
                    .map(|d| d.cached_tokens as u64)
                    .unwrap_or(0),
                cache_creation_input_tokens: 0,
                tool_use_prompt_tokens: 0,
                reasoning_tokens: 0,
            })
            .unwrap_or_default();

        let mut converted = completion::CompletionResponse::new(choice, usage, "copilot")
            .with_model(response.model.clone());
        if !response.id.is_empty() {
            converted = converted.with_message_id(response.id.clone());
        }
        if let Some(finish_reason) = first_choice.finish_reason.as_deref() {
            converted = converted.with_finish_reason(map_finish_reason(finish_reason));
        }

        Ok(converted)
    }
}

#[derive(Debug, Deserialize)]
pub struct ChatApiErrorResponse {
    #[serde(default)]
    pub message: Option<String>,
    #[serde(default)]
    pub error: Option<String>,
}

impl ChatApiErrorResponse {
    pub fn error_message(&self) -> &str {
        self.message
            .as_deref()
            .or(self.error.as_deref())
            .unwrap_or("unknown error")
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum ChatApiResponse<T> {
    Ok(T),
    Err(ChatApiErrorResponse),
}

/// Drive a Copilot `/responses` SSE connection into the normalized streaming
/// response.
///
/// The single source of truth for Copilot Responses streaming;
/// [`functions::open_stream`] routes through this function.
pub(crate) fn stream_copilot_responses_from_event_source(
    event_source: crate::http_client::sse::BoxedEventSource,
    span: tracing::Span,
) -> StreamingCompletionResponse {
    let mut event_source = event_source;
    let stream = tracing_futures::Instrument::instrument(
        stream! {
                let mut final_usage = responses_api::ResponsesUsage::new();
                let mut final_response_id: Option<String> = None;
                let mut final_model: Option<String> = None;
                let mut tool_calls: Vec<streaming::RawStreamingChoice> = Vec::new();
                let mut tool_call_internal_ids: HashMap<String, String> = HashMap::new();
                let span = tracing::Span::current();

                let mut terminated_with_error = false;

                while let Some(event_result) = event_source.next().await {
                    match event_result {
                        Ok(crate::http_client::sse::Event::Open) => continue,
                        Ok(crate::http_client::sse::Event::Message(evt)) => {
                            if evt.data.trim().is_empty() {
                                continue;
                            }

                            let Ok(data) = serde_json::from_str::<responses_api::streaming::StreamingCompletionChunk>(&evt.data) else {
                                continue;
                            };

                            if let responses_api::streaming::StreamingCompletionChunk::Delta(chunk) = &data {
                                use responses_api::streaming::{ItemChunkKind, StreamingItemDoneOutput};

                                match &chunk.data {
                                    ItemChunkKind::OutputItemAdded(message) => {
                                        if let StreamingItemDoneOutput { item: responses_api::Output::FunctionCall(func), .. } = message {
                                            let internal_call_id = tool_call_internal_ids
                                                .entry(func.id.clone())
                                                .or_insert_with(crate::id::generate)
                                                .clone();
                                            yield Ok(RawStreamingChoice::ToolCallDelta {
                                                id: func.id.clone(),
                                                internal_call_id,
                                                content: streaming::ToolCallDeltaContent::Name(func.name.clone()),
                                            });
                                        }
                                    }
                                    ItemChunkKind::OutputItemDone(message) => match message {
                                        StreamingItemDoneOutput { item: responses_api::Output::FunctionCall(func), .. } => {
                                            let internal_id = tool_call_internal_ids
                                                .entry(func.id.clone())
                                                .or_insert_with(crate::id::generate)
                                                .clone();
                                            let raw_tool_call = streaming::RawStreamingToolCall::new(
                                                func.id.clone(),
                                                func.name.clone(),
                                                func.arguments.clone(),
                                            )
                                            .with_internal_call_id(internal_id)
                                            .with_call_id(func.call_id.clone());
                                            tool_calls.push(RawStreamingChoice::ToolCall(raw_tool_call));
                                        }
                                        StreamingItemDoneOutput { item: responses_api::Output::Reasoning { summary, id, content, encrypted_content, .. }, .. } => {
                                            for reasoning_choice in responses_api::streaming::reasoning_choices_from_done_item(
                                                id,
                                                summary,
                                                content,
                                                encrypted_content.as_deref(),
                                            ) {
                                                match reasoning_choice {
                                                    RawStreamingChoice::Reasoning { id, content } => {
                                                        yield Ok(RawStreamingChoice::Reasoning { id, content });
                                                    }
                                                    RawStreamingChoice::ReasoningDelta { id, reasoning } => {
                                                        yield Ok(RawStreamingChoice::ReasoningDelta { id, reasoning });
                                                    }
                                                    _ => {}
                                                }
                                            }
                                        }
                                        StreamingItemDoneOutput { item: responses_api::Output::Message(msg), .. } => {
                                            yield Ok(RawStreamingChoice::MessageId(msg.id.clone()));
                                        }
                                        // Surface an unmodeled output item (e.g. a hosted-tool result) to the consumer verbatim.
                                        StreamingItemDoneOutput { item: responses_api::Output::Unknown(value), .. } => {
                                            yield Ok(RawStreamingChoice::Unknown(value.clone()));
                                        }
                                    },
                                    ItemChunkKind::OutputTextDelta(delta) => {
                                        yield Ok(RawStreamingChoice::Message(delta.delta.clone()))
                                    }
                                    ItemChunkKind::ReasoningSummaryTextDelta(delta) => {
                                        yield Ok(RawStreamingChoice::ReasoningDelta { id: None, reasoning: delta.delta.clone() })
                                    }
                                    ItemChunkKind::RefusalDelta(delta) => {
                                        yield Ok(RawStreamingChoice::Message(delta.delta.clone()))
                                    }
                                    ItemChunkKind::FunctionCallArgsDelta(delta) => {
                                        if let Some(item_id) = chunk.item_id.as_ref() {
                                            let internal_call_id = tool_call_internal_ids
                                                .entry(item_id.clone())
                                                .or_insert_with(crate::id::generate)
                                                .clone();
                                            yield Ok(RawStreamingChoice::ToolCallDelta {
                                                id: item_id.clone(),
                                                internal_call_id,
                                                content: streaming::ToolCallDeltaContent::Delta(delta.delta.clone())
                                            })
                                        }
                                    }
                                    _ => continue,
                                }
                            }

                            if let responses_api::streaming::StreamingCompletionChunk::Response(chunk) = data {
                                let responses_api::streaming::ResponseChunk { kind, response, .. } = *chunk;
                                match kind {
                                    responses_api::streaming::ResponseChunkKind::ResponseCompleted => {
                                        span.record("gen_ai.response.id", response.id.as_str());
                                        span.record("gen_ai.response.model", response.model.as_str());
                                        if !response.id.is_empty() {
                                            final_response_id = Some(response.id.clone());
                                        }
                                        if !response.model.is_empty() {
                                            final_model = Some(response.model.clone());
                                        }
                                        if let Some(usage) = response.usage {
                                            final_usage = usage;
                                        }
                                    }
                                    responses_api::streaming::ResponseChunkKind::ResponseFailed
                                    | responses_api::streaming::ResponseChunkKind::ResponseIncomplete => {
                                        terminated_with_error = true;
                                        // Deliberate two-tier behaviour matching the OpenAI Responses
                                        // SSE path: when the provider supplies an error object we
                                        // preserve the raw event JSON via `completion_error_from_body`
                                        // so the `provider_response_*` helpers surface the full payload
                                        // (code + message). The error arrives over an established 2xx
                                        // stream, so there is no HTTP status to attach (status: None).
                                        // When the object is absent we emit a Rig-authored
                                        // `ProviderError` diagnostic (provider_response_body() is None).
                                        match response.error.as_ref() {
                                            Some(_) => yield Err(
                                                crate::provider_response::completion_error_from_body(
                                                    evt.data.clone(),
                                                ),
                                            ),
                                            None => yield Err(CompletionError::ProviderError(
                                                "Copilot response stream failed".into(),
                                            )),
                                        }
                                        break;
                                    }
                                    _ => continue,
                                }
                            }
                        }
                        Err(crate::http_client::Error::StreamEnded) => {
                            break;
                        }
                        Err(error) => {
                            terminated_with_error = true;
                            yield Err(CompletionError::from_stream_transport(error));
                            break;
                        }
                    }
                }

                // Dropping the boxed event source is equivalent to closing it —
                // the state machine is finished with it.

                if terminated_with_error {
                    return;
                }

                for tool_call in &tool_calls {
                    yield Ok(tool_call.to_owned())
                }

                span.record("gen_ai.usage.input_tokens", final_usage.input_tokens);
                span.record("gen_ai.usage.output_tokens", final_usage.output_tokens);
                span.record(
                    "gen_ai.usage.cache_read.input_tokens",
                    final_usage
                        .input_tokens_details
                        .as_ref()
                        .map(|details| details.cached_tokens)
                        .unwrap_or(0),
                );

                let usage = completion::Usage {
                    input_tokens: final_usage.input_tokens,
                    output_tokens: final_usage.output_tokens,
                    total_tokens: final_usage.total_tokens,
                    cached_input_tokens: final_usage
                        .input_tokens_details
                        .as_ref()
                        .map(|details| details.cached_tokens)
                        .unwrap_or(0),
                    cache_creation_input_tokens: 0,
                    tool_use_prompt_tokens: 0,
                    reasoning_tokens: final_usage
                        .output_tokens_details
                        .as_ref()
                        .map(|details| details.reasoning_tokens)
                        .unwrap_or(0),
                };
                let mut final_response = streaming::StreamFinal::new("copilot", usage);
                if let Some(message_id) = final_response_id {
                    final_response = final_response.with_message_id(message_id);
                }
                if let Some(model) = final_model {
                    final_response = final_response.with_model(model);
                }
                yield Ok(RawStreamingChoice::FinalResponse(final_response));
        },
        span,
    );

    StreamingCompletionResponse::stream(Box::pin(stream))
}

#[derive(Deserialize)]
struct CopilotEmbeddingResponse {
    data: Vec<CopilotEmbeddingData>,
}

#[derive(Deserialize)]
struct CopilotEmbeddingData {
    embedding: Vec<serde_json::Number>,
}

/// Build the serialized `/embeddings` request body. Pure; used by
/// [`functions::embed`].
pub(crate) fn build_embedding_body(
    model: &str,
    texts: &[String],
    dimensions: Option<usize>,
    encoding_format: Option<&openai::EncodingFormat>,
    user: Option<&str>,
) -> Result<Vec<u8>, EmbeddingError> {
    let mut body = json!({
        "model": model,
        "input": texts,
    });

    let body_object = body.as_object_mut().ok_or_else(|| {
        EmbeddingError::ResponseError("embedding request body must be a JSON object".into())
    })?;

    if let Some(dimensions) = dimensions {
        body_object.insert("dimensions".to_owned(), json!(dimensions));
    }
    if let Some(encoding_format) = encoding_format {
        body_object.insert("encoding_format".to_owned(), json!(encoding_format));
    }
    if let Some(user) = user {
        body_object.insert("user".to_owned(), json!(user));
    }
    Ok(serde_json::to_vec(&body)?)
}

/// Parse an `/embeddings` response into the normalized
/// [`embeddings::EmbeddingResponse`], zipping vectors back onto
/// `documents`. Pure; used by [`functions::embed`].
/// Copilot reports no embedding usage.
pub(crate) fn parse_embedding_response(
    status: http::StatusCode,
    body: &str,
    documents: Vec<String>,
) -> Result<embeddings::EmbeddingResponse, EmbeddingError> {
    if !status.is_success() {
        return Err(EmbeddingError::from_http_response(status, body.to_string()));
    }

    #[derive(Deserialize)]
    struct NestedApiError {
        error: NestedApiErrorMessage,
    }

    #[derive(Deserialize)]
    struct NestedApiErrorMessage {
        message: String,
    }

    let parsed: CopilotEmbeddingResponse = match serde_json::from_str(body) {
        Ok(parsed) => parsed,
        Err(parse_error) => {
            if let Ok(err) = serde_json::from_str::<NestedApiError>(body) {
                tracing::warn!(message = %err.error.message, "provider returned an error response");
                return Err(EmbeddingError::from_http_response(status, body.to_string()));
            }

            let preview = if body.len() > 512 {
                let truncated: String = body.chars().take(512).collect();
                format!("{truncated}...")
            } else {
                body.to_string()
            };

            return Err(EmbeddingError::ProviderError(format!(
                "Failed to parse Copilot embeddings response: {parse_error}; body: {preview}"
            )));
        }
    };

    if parsed.data.len() != documents.len() {
        return Err(EmbeddingError::ResponseError(
            "Response data length does not match input length".into(),
        ));
    }

    let embeddings = parsed
        .data
        .into_iter()
        .zip(documents)
        .map(|(embedding, document)| embeddings::Embedding {
            document,
            vec: embedding
                .embedding
                .into_iter()
                .filter_map(|n| n.as_f64())
                .collect(),
        })
        .collect();
    Ok(embeddings::EmbeddingResponse {
        embeddings,
        usage: crate::completion::Usage::new(),
    })
}

const MODEL_LISTING_PATH: &str = "/models";
const MODEL_LISTING_PROVIDER: &str = "Copilot";

#[derive(Debug, Deserialize)]
struct ListModelsResponse {
    data: Vec<ListModelEntry>,
}

#[derive(Debug, Deserialize)]
struct ListModelEntry {
    id: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    vendor: Option<String>,
    #[serde(default)]
    capabilities: Option<ListModelEntryCapabilities>,
}

#[derive(Debug, Deserialize)]
struct ListModelEntryCapabilities {
    #[serde(default, rename = "type")]
    r#type: Option<String>,
}

impl From<ListModelEntry> for Model {
    fn from(value: ListModelEntry) -> Self {
        let mut model = Model::from_id(value.id);
        model.name = value.name;
        model.owned_by = value.vendor;
        if let Some(caps) = value.capabilities {
            model.r#type = caps.r#type;
        }
        model
    }
}

/// Parse a `GET /models` response into a [`ModelList`]. Pure.
///
/// Used by [`functions::list_models`].
pub(crate) fn parse_list_models_response(
    status: http::StatusCode,
    body: &[u8],
) -> Result<ModelList, ModelListingError> {
    if !status.is_success() {
        return Err(ModelListingError::api_error_with_context(
            MODEL_LISTING_PROVIDER,
            MODEL_LISTING_PATH,
            status.as_u16(),
            body,
        ));
    }
    let api_resp: ListModelsResponse = serde_json::from_slice(body).map_err(|error| {
        ModelListingError::parse_error_with_context(
            MODEL_LISTING_PROVIDER,
            MODEL_LISTING_PATH,
            &error,
            body,
        )
    })?;
    let models = api_resp.data.into_iter().map(Model::from).collect();
    Ok(ModelList::new(models))
}

#[derive(Deserialize, Debug)]
struct ChatStreamingFunction {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Deserialize, Debug)]
struct ChatStreamingToolCall {
    index: usize,
    id: Option<String>,
    function: ChatStreamingFunction,
}

impl From<&ChatStreamingToolCall> for CompatibleToolCallChunk {
    fn from(value: &ChatStreamingToolCall) -> Self {
        Self {
            index: value.index,
            id: value.id.clone(),
            name: value.function.name.clone(),
            arguments: value.function.arguments.clone(),
        }
    }
}

#[derive(Deserialize, Debug, Default)]
struct ChatStreamingDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default, deserialize_with = "crate::json_utils::null_or_vec")]
    tool_calls: Vec<ChatStreamingToolCall>,
}

#[derive(Deserialize, Debug, PartialEq)]
#[serde(rename_all = "snake_case")]
enum ChatFinishReason {
    ToolCalls,
    Stop,
    ContentFilter,
    Length,
    #[serde(untagged)]
    Other(String),
}

#[derive(Deserialize, Debug)]
struct ChatStreamingChoice {
    delta: ChatStreamingDelta,
    finish_reason: Option<ChatFinishReason>,
}

#[derive(Deserialize, Debug)]
struct ChatStreamingChunk {
    id: Option<String>,
    model: Option<String>,
    choices: Vec<ChatStreamingChoice>,
    usage: Option<openai::completion::Usage>,
}

/// Parse one GitHub Copilot chat-completions SSE `data` payload. Pure.
///
/// A narrower schema than the shared OpenAI dialect: `index` is required,
/// there is no `reasoning`/`reasoning_details` key, and the deprecated
/// `function_call` finish reason is not treated as a tool call.
pub(crate) fn normalize_copilot_chat_chunk(
    data: &str,
) -> crate::providers::internal::openai_chat_completions_compatible::NormalizedCompatibleChunk {
    let data = match serde_json::from_str::<ChatStreamingChunk>(data) {
        Ok(data) => data,
        Err(error) => {
            tracing::debug!(?error, "Couldn't parse Copilot chat SSE payload");
            return Ok(None);
        }
    };

    Ok(Some(
        openai_chat_completions_compatible::first_choice_chunk(
            data.id,
            data.model,
            data.usage.map(Into::into),
            &data.choices,
            |choice| CompatibleChoice {
                finish_reason: if choice.finish_reason == Some(ChatFinishReason::ToolCalls) {
                    CompatibleFinishReason::ToolCalls
                } else {
                    CompatibleFinishReason::Other
                },
                text: choice.delta.content.clone(),
                reasoning: choice.delta.reasoning_content.clone(),
                tool_calls: openai_chat_completions_compatible::tool_call_chunks(
                    &choice.delta.tool_calls,
                ),
                details: Vec::new(),
            },
        ),
    ))
}

pub(crate) fn send_copilot_chat_streaming_request(
    event_source: crate::http_client::sse::BoxedEventSource,
) -> StreamingCompletionResponse {
    openai_chat_completions_compatible::drive_compatible_stream(
        event_source,
        openai_chat_completions_compatible::ChunkNormalizer::CopilotChat,
    )
}

pub(crate) fn default_token_dir() -> Option<PathBuf> {
    config_dir().map(|dir| dir.join("github_copilot"))
}

fn config_dir() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        std::env::var_os("APPDATA").map(PathBuf::from)
    }

    #[cfg(not(target_os = "windows"))]
    {
        std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ChatApiErrorResponse, ChatCompletionResponse, CompletionRoute, CopilotIntent,
        TEXT_EMBEDDING_3_SMALL, base_url_from_token, default_headers, functions, route_for_model,
        send_copilot_chat_streaming_request, stream_copilot_responses_from_event_source,
    };
    use crate::http_client;
    use crate::http_runtime::HttpRuntime;
    use crate::providers::internal::openai_chat_completions_compatible::test_support::{
        sse_bytes_from_data_lines, sse_bytes_from_json_events,
    };
    use crate::streaming::{StreamedAssistantContent, StreamingCompletionResponse};
    use crate::test_utils::MockStreamingClient;
    use crate::test_utils::{RecordingHttpClient, SequencedStreamingHttpClient};
    use futures::StreamExt;

    /// Drive the surviving Responses SSE state machine over `http_client`,
    /// the way `functions::open_stream` does (building the event source
    /// directly keeps the test focused on the state machine).
    fn responses_stream<H>(http_client: H, model: &str) -> StreamingCompletionResponse
    where
        H: crate::http_client::Backend + Clone + 'static,
    {
        let cfg = functions::Config::new(model).with_api_key("copilot-token");
        let request = crate::completion::CompletionRequest::from_prompt("hello");
        let req = functions::build_request(&cfg, &request, true).expect("build request");
        stream_copilot_responses_from_event_source(
            crate::http_client::sse::boxed_event_source(http_client, req, false),
            tracing::Span::none(),
        )
    }

    /// Same, for the chat-completions SSE state machine.
    fn chat_stream<H>(http_client: H, model: &str) -> StreamingCompletionResponse
    where
        H: crate::http_client::Backend + Clone + 'static,
    {
        let cfg = functions::Config::new(model).with_api_key("copilot-token");
        let request = crate::completion::CompletionRequest::from_prompt("hello");
        let req = functions::build_request(&cfg, &request, true).expect("build request");
        send_copilot_chat_streaming_request(crate::http_client::sse::boxed_event_source(
            http_client,
            req,
            false,
        ))
    }

    fn minimal_chat_response() -> &'static str {
        r#"{
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "hello"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 4,
                "total_tokens": 7
            }
        }"#
    }

    fn minimal_responses_response() -> &'static str {
        r#"{
            "id": "resp_123",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "error": null,
            "incomplete_details": null,
            "instructions": null,
            "max_output_tokens": null,
            "model": "gpt-5.3-codex",
            "usage": {
                "input_tokens": 4,
                "input_tokens_details": {
                    "cached_tokens": 0
                },
                "output_tokens": 3,
                "output_tokens_details": {
                    "reasoning_tokens": 0
                },
                "total_tokens": 7
            },
            "output": [{
                "type": "message",
                "id": "msg_123",
                "role": "assistant",
                "status": "completed",
                "content": [{
                    "type": "output_text",
                    "text": "hello"
                }]
            }],
            "tools": []
        }"#
    }

    fn minimal_embeddings_response() -> &'static str {
        r#"{
            "data": [
                {
                    "embedding": [0.1, 0.2, 0.3]
                },
                {
                    "embedding": [0.4, 0.5, 0.6]
                }
            ]
        }"#
    }

    #[test]
    fn deserialize_standard_openai_response() {
        let json = r#"{
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello!"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }"#;

        let response: ChatCompletionResponse =
            serde_json::from_str(json).expect("standard OpenAI response should deserialize");
        assert_eq!(response.id, "chatcmpl-abc123");
        assert_eq!(response.object.as_deref(), Some("chat.completion"));
        assert_eq!(response.created, Some(1700000000));
        assert_eq!(response.model, "gpt-4o");
        assert_eq!(response.choices.len(), 1);
        assert_eq!(response.choices[0].finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn deserialize_copilot_response_without_object_and_created() {
        let response: ChatCompletionResponse = serde_json::from_str(minimal_chat_response())
            .expect("Copilot response should deserialize");

        assert_eq!(response.id, "chatcmpl-123");
        assert_eq!(response.object, None);
        assert_eq!(response.created, None);
        assert_eq!(response.model, "gpt-4o");
        assert_eq!(response.choices.len(), 1);
    }

    #[test]
    fn deserialize_copilot_response_without_finish_reason() {
        let json = r#"{
            "id": "chatcmpl-claude-001",
            "model": "claude-3.5-sonnet",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Here is my analysis."
                }
            }],
            "usage": {
                "prompt_tokens": 50,
                "total_tokens": 80
            }
        }"#;

        let response: ChatCompletionResponse =
            serde_json::from_str(json).expect("Claude-via-Copilot response should deserialize");

        assert_eq!(response.model, "claude-3.5-sonnet");
        assert_eq!(response.choices[0].finish_reason, None);
        assert_eq!(response.choices[0].index, 0);
    }

    #[test]
    fn error_response_with_message_field() {
        let json = r#"{"message": "rate limit exceeded"}"#;
        let err: ChatApiErrorResponse = serde_json::from_str(json).expect("message-shaped error");

        assert_eq!(err.error_message(), "rate limit exceeded");
    }

    #[test]
    fn error_response_with_error_field() {
        let json = r#"{"error": "model not found"}"#;
        let err: ChatApiErrorResponse = serde_json::from_str(json).expect("error-shaped error");

        assert_eq!(err.error_message(), "model not found");
    }

    #[test]
    fn routes_codex_models_to_responses() {
        assert_eq!(route_for_model("gpt-5.3-codex"), CompletionRoute::Responses);
        assert_eq!(
            route_for_model("gpt-5.1-CODEX-mini"),
            CompletionRoute::Responses
        );
        assert_eq!(route_for_model("gpt-5.2"), CompletionRoute::ChatCompletions);
        assert_eq!(
            route_for_model("claude-sonnet-4.5"),
            CompletionRoute::ChatCompletions
        );
    }

    #[test]
    fn copilot_intent_headers_use_panel_by_default_and_edits_when_requested() {
        let panel_headers = default_headers("token", "user", false, CopilotIntent::default());
        assert_eq!(
            panel_headers
                .iter()
                .find(|(name, _)| *name == "openai-intent")
                .map(|(_, value)| value.as_str()),
            Some("conversation-panel")
        );

        let edits_headers = default_headers("token", "user", false, CopilotIntent::Edits);
        assert_eq!(
            edits_headers
                .iter()
                .find(|(name, _)| *name == "openai-intent")
                .map(|(_, value)| value.as_str()),
            Some("conversation-edits")
        );
    }

    #[test]
    fn base_url_from_token_derives_api_endpoint() {
        assert_eq!(
            base_url_from_token("tid=1;proxy-ep=proxy.individual.githubcopilot.com;exp=2")
                .as_deref(),
            Some("https://api.individual.githubcopilot.com")
        );
        assert_eq!(
            base_url_from_token("tid=1;proxy-ep=https://proxy.individual.githubcopilot.com;exp=2")
                .as_deref(),
            Some("https://api.individual.githubcopilot.com")
        );
        assert_eq!(base_url_from_token("tid=1;exp=2"), None);
    }

    #[test]
    fn base_url_from_token_rejects_unsafe_or_non_copilot_endpoints() {
        assert_eq!(
            base_url_from_token("tid=1;proxy-ep=http://proxy.individual.githubcopilot.com;exp=2"),
            None
        );
        assert_eq!(
            base_url_from_token("tid=1;proxy-ep=https://evil.example.com;exp=2"),
            None
        );
        assert_eq!(base_url_from_token("tid=1;proxy-ep=://bad;exp=2"), None);
        assert_eq!(base_url_from_token("tid=1;proxy-ep=;exp=2"), None);
        assert_eq!(
            base_url_from_token(
                "tid=1;proxy-ep=https://proxy.individual.githubcopilot.com/base;exp=2"
            ),
            None
        );
    }

    #[tokio::test]
    async fn api_key_with_proxy_endpoint_overrides_base_url() {
        let http_client = RecordingHttpClient::new(minimal_chat_response());
        let rt = HttpRuntime::recording(http_client.clone());
        let cfg = functions::Config::new("gpt-4o")
            .with_api_key("tid=1;proxy-ep=proxy.individual.githubcopilot.com;exp=2");
        let request = crate::completion::CompletionRequest::from_prompt("hello");

        let _response = functions::complete(&cfg, &rt, request)
            .await
            .expect("chat completion");

        let requests = http_client.requests();
        assert_eq!(requests.len(), 1);
        let uri = requests.first().expect("one recorded request").uri.clone();
        assert!(
            uri.starts_with("https://api.individual.githubcopilot.com"),
            "expected proxy-derived base URL, got {uri}"
        );
    }

    #[tokio::test]
    async fn explicit_base_url_wins_over_token_proxy_endpoint() {
        let http_client = RecordingHttpClient::new(minimal_chat_response());
        let rt = HttpRuntime::recording(http_client.clone());
        let cfg = functions::Config::new("gpt-4o")
            .with_api_key("tid=1;proxy-ep=proxy.individual.githubcopilot.com;exp=2")
            .with_base_url("https://custom.example.com");
        let request = crate::completion::CompletionRequest::from_prompt("hello");

        let _response = functions::complete(&cfg, &rt, request)
            .await
            .expect("chat completion");

        let requests = http_client.requests();
        assert_eq!(requests.len(), 1);
        let uri = requests.first().expect("one recorded request").uri.clone();
        assert!(
            uri.starts_with("https://custom.example.com"),
            "expected explicit base URL, got {uri}"
        );
    }

    #[tokio::test]
    async fn completion_routes_chat_requests_to_chat_completions() {
        let http_client = RecordingHttpClient::new(minimal_chat_response());
        let rt = HttpRuntime::recording(http_client.clone());
        let cfg = functions::Config::new("gpt-4o").with_api_key("copilot-token");
        let request = crate::completion::CompletionRequest::from_prompt("hello");

        let _response = functions::complete(&cfg, &rt, request)
            .await
            .expect("chat completion");

        let requests = http_client.requests();
        assert_eq!(requests.len(), 1);
        let recorded = requests.first().expect("one recorded request");
        assert!(recorded.uri.ends_with("/chat/completions"));
        assert!(String::from_utf8_lossy(&recorded.body).contains("\"model\":\"gpt-4o\""));
    }

    #[tokio::test]
    async fn completion_routes_codex_requests_to_responses() {
        let http_client = RecordingHttpClient::new(minimal_responses_response());
        let rt = HttpRuntime::recording(http_client.clone());
        let cfg = functions::Config::new("gpt-5.3-codex").with_api_key("copilot-token");
        let request = crate::completion::CompletionRequest::from_prompt("hello");

        let _response = functions::complete(&cfg, &rt, request)
            .await
            .expect("responses completion");

        let requests = http_client.requests();
        assert_eq!(requests.len(), 1);
        let recorded = requests.first().expect("one recorded request");
        assert!(recorded.uri.ends_with("/responses"));
        assert!(String::from_utf8_lossy(&recorded.body).contains("\"model\":\"gpt-5.3-codex\""));
    }

    #[tokio::test]
    async fn embeddings_accept_minimal_copilot_response_shape() {
        let http_client = RecordingHttpClient::new(minimal_embeddings_response());
        let rt = HttpRuntime::recording(http_client.clone());
        let cfg = functions::EmbeddingConfig::new(TEXT_EMBEDDING_3_SMALL)
            .with_api_key("copilot-token")
            .with_dimensions(1536);

        let response = functions::embed(&cfg, &rt, vec!["one".to_string(), "two".to_string()])
            .await
            .expect("embeddings should deserialize");

        assert_eq!(response.embeddings.len(), 2);
        assert_eq!(
            response.embeddings.first().expect("first").vec,
            vec![0.1, 0.2, 0.3]
        );
        assert_eq!(
            response.embeddings.get(1).expect("second").vec,
            vec![0.4, 0.5, 0.6]
        );

        let requests = http_client.requests();
        assert_eq!(requests.len(), 1);
        let recorded = requests.first().expect("one recorded request");
        assert!(recorded.uri.ends_with("/embeddings"));
        assert!(
            String::from_utf8_lossy(&recorded.body)
                .contains("\"model\":\"text-embedding-3-small\"")
        );
    }

    #[tokio::test]
    async fn responses_stream_terminates_after_terminal_error() {
        let tool_call_done = serde_json::json!({
            "type": "response.output_item.done",
            "sequence_number": 1,
            "item": {
                "type": "function_call",
                "id": "fc_123",
                "arguments": "{}",
                "call_id": "call_123",
                "name": "example_tool",
                "status": "completed"
            }
        });
        let failed = serde_json::json!({
            "type": "response.failed",
            "sequence_number": 2,
            "response": {
                "id": "resp_123",
                "object": "response",
                "created_at": 1700000000,
                "status": "failed",
                "error": {
                    "code": "server_error",
                    "message": "Copilot response stream failed"
                },
                "incomplete_details": null,
                "instructions": null,
                "max_output_tokens": null,
                "model": "gpt-5.3-codex",
                "usage": null,
                "output": [],
                "tools": []
            }
        });
        let http_client = MockStreamingClient {
            sse_bytes: sse_bytes_from_json_events(&[tool_call_done, failed]),
        };
        let mut stream = responses_stream(http_client, "gpt-5.3-codex");

        let err = match stream.next().await.expect("stream should yield an item") {
            Ok(_) => panic!("stream should surface a provider error"),
            Err(err) => err,
        };
        // The terminal `response.failed` event carries the provider's error
        // payload, so the full raw event JSON is preserved for inspection
        // (status: None — the error arrived over an already-established stream),
        // matching the OpenAI Responses SSE path.
        assert!(matches!(
            err,
            crate::completion::CompletionError::ProviderResponse(_)
        ));
        assert_eq!(err.provider_response_status(), None);
        let json = err
            .provider_response_json()
            .expect("preserved body should parse as JSON")
            .expect("preserved body should not be empty");
        let response_error = json
            .get("response")
            .and_then(|response| response.get("error"))
            .expect("preserved body should retain the provider error object");
        assert_eq!(
            response_error.get("code").and_then(|value| value.as_str()),
            Some("server_error")
        );
        assert_eq!(
            response_error
                .get("message")
                .and_then(|value| value.as_str()),
            Some("Copilot response stream failed")
        );
        assert!(
            stream.next().await.is_none(),
            "responses stream should terminate immediately after a terminal error"
        );
    }

    #[tokio::test]
    async fn responses_stream_populates_final_response_metadata() {
        let metadata = serde_json::json!({
            "context": "all_turns",
            "effort": "ultra",
            "summary": null,
            "future_control": true
        });
        let completed = serde_json::json!({
            "type": "response.completed",
            "sequence_number": 1,
            "response": {
                "id": "resp_123",
                "object": "response",
                "created_at": 1700000000,
                "status": "completed",
                "error": null,
                "incomplete_details": null,
                "instructions": null,
                "max_output_tokens": null,
                "model": "gpt-5.3-codex",
                "reasoning": metadata.clone(),
                "usage": null,
                "output": [],
                "tools": []
            }
        });
        let http_client = MockStreamingClient {
            sse_bytes: sse_bytes_from_json_events(&[completed]),
        };
        let mut stream = responses_stream(http_client, "gpt-5.3-codex");

        while let Some(item) = stream.next().await {
            if let StreamedAssistantContent::Final(response) =
                item.expect("completed stream should not error")
            {
                assert_eq!(response.provider, "copilot");
                assert_eq!(response.message_id.as_deref(), Some("resp_123"));
                assert_eq!(response.model.as_deref(), Some("gpt-5.3-codex"));
                return;
            }
        }

        panic!("responses stream should yield a final response");
    }

    #[tokio::test]
    async fn chat_stream_terminates_after_transport_error() {
        let chunks = vec![
            Ok(sse_bytes_from_data_lines([
                "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_123\",\"function\":{\"name\":\"ping\",\"arguments\":\"\"}}]},\"finish_reason\":null}],\"usage\":null}",
            ])),
            Err(http_client::Error::InvalidStatusCode(
                http::StatusCode::BAD_GATEWAY,
            )),
        ];

        let http_client = SequencedStreamingHttpClient::new(chunks);
        let mut stream = chat_stream(http_client, "gpt-4o");

        let mut saw_error = false;
        while let Some(item) = stream.next().await {
            match item {
                Ok(StreamedAssistantContent::ToolCallDelta { .. }) => {}
                Err(err) => {
                    assert_eq!(
                        err.to_string(),
                        "HttpError: Invalid status code: 502 Bad Gateway"
                    );
                    assert_eq!(
                        err.provider_response_status(),
                        Some(http::StatusCode::BAD_GATEWAY)
                    );
                    assert_eq!(err.provider_response_body(), None);
                    saw_error = true;
                    break;
                }
                Ok(_) => panic!("unexpected non-error stream item before transport failure"),
            }
        }

        assert!(saw_error, "stream should surface the transport error");
        assert!(
            stream.next().await.is_none(),
            "chat stream should terminate immediately after a transport error"
        );
    }

    /// The four deleted `env_*` precedence tests, ported onto the
    /// `functions` module's variable lists and precedence helper.
    mod env_precedence {
        use super::functions::{
            ACCESS_TOKEN_VARS, API_KEY_VARS, BASE_URL_VARS, first_present_in, first_value_in,
        };
        use std::collections::HashMap;

        fn getter(
            entries: &'static [(&'static str, &'static str)],
        ) -> impl Fn(&'static str) -> Result<Option<String>, crate::providers::descriptor::ConfigError>
        {
            let env: HashMap<String, String> = entries
                .iter()
                .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
                .collect();
            move |name: &str| Ok(env.get(name).cloned())
        }

        #[test]
        fn env_api_key_prefers_github_prefixed_vars() {
            let get = getter(&[
                ("COPILOT_API_KEY", "copilot-key"),
                ("GITHUB_COPILOT_API_KEY", "github-key"),
                ("GITHUB_TOKEN", "bootstrap-token"),
            ]);
            assert_eq!(
                first_value_in(API_KEY_VARS, &get)
                    .expect("lookup")
                    .as_deref(),
                Some("github-key")
            );
        }

        #[test]
        fn env_github_access_token_prefers_explicit_bootstrap_var() {
            let get = getter(&[
                ("COPILOT_GITHUB_ACCESS_TOKEN", "explicit-bootstrap"),
                ("GITHUB_TOKEN", "fallback-bootstrap"),
            ]);
            assert_eq!(
                first_value_in(ACCESS_TOKEN_VARS, &get)
                    .expect("lookup")
                    .as_deref(),
                Some("explicit-bootstrap")
            );
        }

        #[test]
        fn env_base_url_prefers_github_prefixed_vars() {
            let get = getter(&[
                ("COPILOT_BASE_URL", "https://copilot.example"),
                ("GITHUB_COPILOT_API_BASE", "https://github.example"),
            ]);
            assert_eq!(
                first_value_in(BASE_URL_VARS, &get)
                    .expect("lookup")
                    .as_deref(),
                Some("https://github.example")
            );
        }

        #[test]
        fn env_without_api_key_falls_back_to_oauth() {
            // No API-key and no access-token variable => `config_from_env`
            // falls through to the OAuth arm.
            let get = getter(&[("COPILOT_BASE_URL", "https://copilot.example")]);
            assert_eq!(first_present_in(API_KEY_VARS, &get).expect("lookup"), None);
            assert_eq!(
                first_present_in(ACCESS_TOKEN_VARS, &get).expect("lookup"),
                None
            );
            assert_eq!(
                first_value_in(BASE_URL_VARS, &get)
                    .expect("lookup")
                    .as_deref(),
                Some("https://copilot.example")
            );
        }

        #[test]
        fn env_github_token_is_not_treated_as_copilot_api_key() {
            let get = getter(&[("GITHUB_TOKEN", "bootstrap-token")]);
            assert_eq!(first_present_in(API_KEY_VARS, &get).expect("lookup"), None);
            assert_eq!(
                first_value_in(ACCESS_TOKEN_VARS, &get)
                    .expect("lookup")
                    .as_deref(),
                Some("bootstrap-token")
            );
        }
    }
}
