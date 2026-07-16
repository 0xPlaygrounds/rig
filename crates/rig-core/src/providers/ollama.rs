//! Ollama API client and Rig integration
//!
//! # Example
//! ```rust,ignore
//! use rig_core::client::{Nothing, CompletionClient};
//! use rig_core::completion::Prompt;
//! use rig_core::providers::ollama;
//!
//! // Create a new Ollama client (defaults to http://localhost:11434, no auth)
//! let client = ollama::Client::new(Nothing).unwrap();
//!
//! // Or connect to a remote/proxied Ollama instance with authentication
//! let client = ollama::Client::builder()
//!     .api_key("my-secret-key")
//!     .base_url("http://remote-ollama:11434")
//!     .build()
//!     .unwrap();
//!
//! // Create an agent with a preamble
//! let comedian_agent = client
//!     .agent("qwen2.5:14b")
//!     .preamble("You are a comedian here to entertain the user using humour and jokes.")
//!     .build();
//!
//! // Prompt the agent and print the response
//! let response = comedian_agent.prompt("Entertain me!").await?;
//! println!("{response}");
//!
//! // Create an embedding model using the "all-minilm" model
//! let emb_model = client.embedding_model("all-minilm", 384);
//! let embeddings = emb_model.embed_texts(vec![
//!     "Why is the sky blue?".to_owned(),
//!     "Why is the grass green?".to_owned()
//! ]).await?;
//! println!("Embedding response: {:?}", embeddings);
//!
//! // Create an extractor if needed
//! let extractor = client.extractor::<serde_json::Value>("llama3.2").build();
//! ```
use crate::client::{
    self, ApiKey, Capabilities, Capable, DebugExt, ModelLister, Nothing, Provider, ProviderBuilder,
    ProviderClient,
};
use crate::completion::{GetTokenUsage, Usage};
use crate::http_client::{self, HttpClientExt};
use crate::message::DocumentSourceKind;
use crate::model::{Model, ModelList, ModelListingError};
use crate::streaming::RawStreamingChoice;
use crate::telemetry::{CompletionOperation, CompletionSpanBuilder};
use crate::{
    OneOrMany,
    completion::{self, CompletionError, CompletionRequest},
    embeddings::{self, EmbeddingError},
    json_utils, message,
    message::{ImageDetail, Text},
    streaming,
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};
use async_stream::try_stream;
use bytes::Bytes;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{convert::TryFrom, str::FromStr};
use tracing_futures::Instrument;
// ---------- Main Client ----------

const OLLAMA_API_BASE_URL: &str = "http://localhost:11434";

/// Optional API key for Ollama. By default Ollama requires no authentication,
/// but proxied or secured deployments may require a Bearer token.
#[derive(Debug, Default, Clone)]
pub struct OllamaApiKey(Option<String>);

impl ApiKey for OllamaApiKey {
    fn into_header(
        self,
    ) -> Option<http_client::Result<(http::header::HeaderName, http::header::HeaderValue)>> {
        self.0.map(http_client::make_auth_header)
    }
}

impl From<Nothing> for OllamaApiKey {
    fn from(_: Nothing) -> Self {
        Self(None)
    }
}

impl From<String> for OllamaApiKey {
    fn from(key: String) -> Self {
        if key.is_empty() {
            Self(None)
        } else {
            Self(Some(key))
        }
    }
}

impl From<&str> for OllamaApiKey {
    fn from(key: &str) -> Self {
        if key.is_empty() {
            Self(None)
        } else {
            Self(Some(key.to_owned()))
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct OllamaExt;

#[derive(Debug, Default, Clone, Copy)]
pub struct OllamaBuilder;

impl Provider for OllamaExt {
    type Builder = OllamaBuilder;
    const VERIFY_PATH: &'static str = "api/tags";
}

impl<H> Capabilities<H> for OllamaExt {
    type Completion = Capable<CompletionModel<H>>;
    type Transcription = Nothing;
    type Embeddings = Capable<EmbeddingModel<H>>;
    type ModelListing = Capable<OllamaModelLister<H>>;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;

    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
    type Rerank = Nothing;
}

impl DebugExt for OllamaExt {}

impl ProviderBuilder for OllamaBuilder {
    type Extension<H>
        = OllamaExt
    where
        H: HttpClientExt;
    type ApiKey = OllamaApiKey;

    const BASE_URL: &'static str = OLLAMA_API_BASE_URL;

    fn build<H>(
        _builder: &client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: HttpClientExt,
    {
        Ok(OllamaExt)
    }
}

pub type Client<H = reqwest::Client> = client::Client<OllamaExt, H>;
pub type ClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<OllamaBuilder, OllamaApiKey, H>;

impl ProviderClient for Client {
    type Input = OllamaApiKey;
    type Error = crate::client::ProviderClientError;

    fn from_env() -> Result<Self, Self::Error> {
        let api_base = crate::client::optional_env_var("OLLAMA_API_BASE_URL")?
            .unwrap_or_else(|| OLLAMA_API_BASE_URL.to_string());

        let api_key = crate::client::optional_env_var("OLLAMA_API_KEY")?
            .map(OllamaApiKey::from)
            .unwrap_or_default();

        Self::builder()
            .api_key(api_key)
            .base_url(&api_base)
            .build()
            .map_err(Into::into)
    }

    fn from_val(api_key: Self::Input) -> Result<Self, Self::Error> {
        Self::builder().api_key(api_key).build().map_err(Into::into)
    }
}

// ---------- Embedding API ----------

pub const ALL_MINILM: &str = "all-minilm";
pub const NOMIC_EMBED_TEXT: &str = "nomic-embed-text";

fn model_dimensions_from_identifier(identifier: &str) -> Option<usize> {
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

// ---------- Embedding Model ----------

#[derive(Clone)]
pub struct EmbeddingModel<T = reqwest::Client> {
    client: Client<T>,
    pub model: String,
    ndims: usize,
}

impl<T> EmbeddingModel<T> {
    pub fn new(client: Client<T>, model: impl Into<String>, ndims: usize) -> Self {
        Self {
            client,
            model: model.into(),
            ndims,
        }
    }

    pub fn with_model(client: Client<T>, model: &str, ndims: usize) -> Self {
        Self {
            client,
            model: model.into(),
            ndims,
        }
    }
}

impl<T> embeddings::EmbeddingModel for EmbeddingModel<T>
where
    T: HttpClientExt + Clone + 'static,
{
    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>, dims: Option<usize>) -> Self {
        let model = model.into();
        let dims = dims
            .or(model_dimensions_from_identifier(&model))
            .unwrap_or_default();
        Self::new(client.clone(), model, dims)
    }

    const MAX_DOCUMENTS: usize = 1024;
    fn ndims(&self) -> usize {
        self.ndims
    }

    async fn embed_texts(
        &self,
        documents: impl IntoIterator<Item = String>,
    ) -> Result<Vec<embeddings::Embedding>, EmbeddingError> {
        let docs: Vec<String> = documents.into_iter().collect();

        let body = serde_json::to_vec(&json!({
            "model": self.model,
            "input": docs
        }))?;

        let req = self
            .client
            .post("api/embed")?
            .body(body)
            .map_err(|e| EmbeddingError::HttpError(e.into()))?;

        let response = self.client.send::<_, Vec<u8>>(req).await?;

        let status = response.status();
        if !status.is_success() {
            let text = http_client::text(response).await?;
            return Err(EmbeddingError::from_http_response(status, text));
        }

        let bytes: Vec<u8> = response.into_body().await?;

        let api_resp: EmbeddingResponse = serde_json::from_slice(&bytes)?;

        if api_resp.embeddings.len() != docs.len() {
            return Err(EmbeddingError::ResponseError(
                "Number of returned embeddings does not match input".into(),
            ));
        }
        Ok(api_resp
            .embeddings
            .into_iter()
            .zip(docs.into_iter())
            .map(|(vec, document)| embeddings::Embedding { document, vec })
            .collect())
    }
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
impl TryFrom<CompletionResponse> for completion::CompletionResponse<CompletionResponse> {
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
                // #1926). Without this, non-streaming `thinking` is kept only in
                // `raw_response` and lost from `choice`, unlike the streaming path
                // (see `RawStreamingChoice::ReasoningDelta` below).
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

                let raw_response = CompletionResponse {
                    model: resp.model,
                    created_at: resp.created_at,
                    done: resp.done,
                    done_reason: resp.done_reason,
                    total_duration: resp.total_duration,
                    load_duration: resp.load_duration,
                    prompt_eval_count: resp.prompt_eval_count,
                    prompt_eval_duration: resp.prompt_eval_duration,
                    eval_count: resp.eval_count,
                    eval_duration: resp.eval_duration,
                    message: Message::Assistant {
                        content,
                        thinking,
                        images: None,
                        name: None,
                        tool_calls,
                    },
                };

                Ok(completion::CompletionResponse {
                    choice,
                    usage: Usage {
                        input_tokens: prompt_tokens,
                        output_tokens: completion_tokens,
                        total_tokens: prompt_tokens + completion_tokens,
                        cached_input_tokens: 0,
                        cache_creation_input_tokens: 0,
                        tool_use_prompt_tokens: 0,
                        reasoning_tokens: 0,
                    },
                    raw_response,
                    message_id: None,
                })
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
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ToolDefinition>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    think: Option<Think>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u64>,
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
        let mut full_history: Vec<Message> = match &req.preamble {
            Some(preamble) => vec![Message::system(preamble)],
            None => vec![],
        };

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

            json_utils::merge(json!({ "temperature": req.temperature }), extra)
        } else {
            json!({ "temperature": req.temperature })
        };

        Ok(Self {
            model: model.to_string(),
            messages: full_history,
            temperature: req.temperature,
            max_tokens: req.max_tokens,
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

#[derive(Clone)]
pub struct CompletionModel<T = reqwest::Client> {
    client: Client<T>,
    pub model: String,
}

impl<T> CompletionModel<T> {
    pub fn new(client: Client<T>, model: &str) -> Self {
        Self {
            client,
            model: model.to_owned(),
        }
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

// ---------- CompletionModel Implementation ----------

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct StreamingCompletionResponse {
    pub done_reason: Option<String>,
    pub total_duration: Option<u64>,
    pub load_duration: Option<u64>,
    pub prompt_eval_count: Option<u64>,
    pub prompt_eval_duration: Option<u64>,
    pub eval_count: Option<u64>,
    pub eval_duration: Option<u64>,
}

impl GetTokenUsage for StreamingCompletionResponse {
    fn token_usage(&self) -> crate::completion::Usage {
        let mut usage = crate::completion::Usage::new();
        let input_tokens = self.prompt_eval_count.unwrap_or_default();
        let output_tokens = self.eval_count.unwrap_or_default();
        usage.input_tokens = input_tokens;
        usage.output_tokens = output_tokens;
        usage.total_tokens = input_tokens + output_tokens;

        usage
    }
}

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

impl<T> completion::CompletionModel for CompletionModel<T>
where
    T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
{
    type Response = CompletionResponse;
    type StreamingResponse = StreamingCompletionResponse;

    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model.into().as_str())
    }

    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<Self::Response>, CompletionError> {
        let system_instructions = completion_request.preamble.clone();
        let record_telemetry_content = completion_request.record_telemetry_content;
        let request = OllamaCompletionRequest::try_from((self.model.as_ref(), completion_request))?;
        let span = CompletionSpanBuilder::new("ollama", &request.model, CompletionOperation::Chat)
            .system_instructions(system_instructions.as_deref(), record_telemetry_content)
            .build();

        if tracing::enabled!(tracing::Level::TRACE) {
            tracing::trace!(target: "rig::completions",
                "Ollama completion request: {}",
                serde_json::to_string_pretty(&request)?
            );
        }

        let body = serde_json::to_vec(&request)?;

        let req = self
            .client
            .post("api/chat")?
            .body(body)
            .map_err(http_client::Error::from)?;

        let async_block = async move {
            let response = self.client.send::<_, Bytes>(req).await?;
            let status = response.status();
            let response_body = response.into_body().into_future().await?.to_vec();

            if !status.is_success() {
                return Err(CompletionError::from_http_response(
                    status,
                    String::from_utf8_lossy(&response_body),
                ));
            }

            let response: CompletionResponse = serde_json::from_slice(&response_body)?;
            let span = tracing::Span::current();
            span.record("gen_ai.response.model", &response.model);
            span.record(
                "gen_ai.usage.input_tokens",
                response.prompt_eval_count.unwrap_or_default(),
            );
            span.record(
                "gen_ai.usage.output_tokens",
                response.eval_count.unwrap_or_default(),
            );

            if tracing::enabled!(tracing::Level::TRACE) {
                tracing::trace!(target: "rig::completions",
                    "Ollama completion response: {}",
                    serde_json::to_string_pretty(&response)?
                );
            }

            let response: completion::CompletionResponse<CompletionResponse> =
                response.try_into()?;

            Ok(response)
        };

        tracing::Instrument::instrument(async_block, span).await
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<streaming::StreamingCompletionResponse<Self::StreamingResponse>, CompletionError>
    {
        let system_instructions = request.preamble.clone();
        let record_telemetry_content = request.record_telemetry_content;
        let mut request = OllamaCompletionRequest::try_from((self.model.as_ref(), request))?;
        let span = CompletionSpanBuilder::new(
            "ollama",
            &request.model,
            CompletionOperation::ChatStreaming,
        )
        .system_instructions(system_instructions.as_deref(), record_telemetry_content)
        .build();
        request.stream = true;

        if tracing::enabled!(tracing::Level::TRACE) {
            tracing::trace!(target: "rig::completions",
                "Ollama streaming completion request: {}",
                serde_json::to_string_pretty(&request)?
            );
        }

        let body = serde_json::to_vec(&request)?;

        let req = self
            .client
            .post("api/chat")?
            .body(body)
            .map_err(http_client::Error::from)?;

        let response = self
            .client
            .send_streaming(req)
            .instrument(span.clone())
            .await?;
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
                        yield RawStreamingChoice::FinalResponse(
                            StreamingCompletionResponse {
                                total_duration: response.total_duration,
                                load_duration: response.load_duration,
                                prompt_eval_count: response.prompt_eval_count,
                                prompt_eval_duration: response.prompt_eval_duration,
                                eval_count: response.eval_count,
                                eval_duration: response.eval_duration,
                                done_reason: response.done_reason,
                            }
                        );
                        break;
                    }
                }
            }
        }.instrument(span);

        Ok(streaming::StreamingCompletionResponse::stream(Box::pin(
            stream,
        )))
    }
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

/// [`ModelLister`] implementation for the Ollama API (`GET /api/tags`).
#[derive(Clone)]
pub struct OllamaModelLister<H = reqwest::Client> {
    client: Client<H>,
}

impl<H> ModelLister<H> for OllamaModelLister<H>
where
    H: HttpClientExt + WasmCompatSend + WasmCompatSync + 'static,
{
    type Client = Client<H>;

    fn new(client: Self::Client) -> Self {
        Self { client }
    }

    async fn list_all(&self) -> Result<ModelList, ModelListingError> {
        let path = "/api/tags";
        let req = self.client.get(path)?.body(http_client::NoBody)?;
        let response = self.client.send::<_, Vec<u8>>(req).await?;

        if !response.status().is_success() {
            let status_code = response.status().as_u16();
            let body = response.into_body().await?;
            return Err(ModelListingError::api_error_with_context(
                "Ollama",
                path,
                status_code,
                &body,
            ));
        }

        let body = response.into_body().await?;
        let api_resp: ListModelsResponse = serde_json::from_slice(&body).map_err(|error| {
            ModelListingError::parse_error_with_context("Ollama", path, &error, &body)
        })?;
        let models = api_resp.models.into_iter().map(Model::from).collect();

        Ok(ModelList::new(models))
    }
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
        let conv: completion::CompletionResponse<CompletionResponse> =
            chat_resp.try_into().unwrap();
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
        let completed: completion::CompletionResponse<CompletionResponse> =
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
            preamble: Some("You are a helpful assistant.".to_string()),
            chat_history: OneOrMany::one(CompletionMessage::User {
                content: OneOrMany::one(UserContent::Text(Text::new("What is 2 + 2?".to_string()))),
            }),
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
            "temperature": 0.7,
            "stream": false,
            "think": true,
            "max_tokens": 1024,
            "keep_alive": "-1m",
            "options": {
                "temperature": 0.7,
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
            preamble: Some("You are a helpful assistant.".to_string()),
            chat_history: OneOrMany::one(CompletionMessage::User {
                content: OneOrMany::one(UserContent::Text(Text::new("What is 2 + 2?".to_string()))),
            }),
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
            "temperature": 0.7,
            "stream": false,
            "think": "low",
            "max_tokens": 1024,
            "keep_alive": "-1m",
            "options": {
                "temperature": 0.7,
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
            preamble: Some("You are a helpful assistant.".to_string()),
            chat_history: OneOrMany::one(CompletionMessage::User {
                content: OneOrMany::one(UserContent::Text(Text::new("What is 2 + 2?".to_string()))),
            }),
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
            "temperature": 0.7,
            "stream": false,
            "think": "medium",
            "max_tokens": 1024,
            "keep_alive": "-1m",
            "options": {
                "temperature": 0.7,
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
            preamble: Some("You are a helpful assistant.".to_string()),
            chat_history: OneOrMany::one(CompletionMessage::User {
                content: OneOrMany::one(UserContent::Text(Text::new("What is 2 + 2?".to_string()))),
            }),
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
            "temperature": 0.7,
            "stream": false,
            "think": "high",
            "max_tokens": 1024,
            "keep_alive": "-1m",
            "options": {
                "temperature": 0.7,
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
            preamble: Some("You are a helpful assistant.".to_string()),
            chat_history: OneOrMany::one(CompletionMessage::User {
                content: OneOrMany::one(UserContent::Text(Text::new("What is 2 + 2?".to_string()))),
            }),
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
            preamble: Some("You are a helpful assistant.".to_string()),
            chat_history: OneOrMany::one(CompletionMessage::User {
                content: OneOrMany::one(UserContent::Text(Text::new("Hello!".to_string()))),
            }),
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
            "temperature": 0.5,
            "stream": false,
            "options": {
                "temperature": 0.5
            }
        });

        assert_eq!(serialized, expected);
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
            preamble: None,
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
            preamble: None,
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
    fn test_client_initialization() {
        let _client = crate::providers::ollama::Client::new(Nothing).expect("Client::new() failed");
        let _client_from_builder = crate::providers::ollama::Client::builder()
            .api_key(Nothing)
            .build()
            .expect("Client::builder() failed");
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
        use crate::client::CompletionClient;
        use crate::completion::CompletionModel;
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"error":"model not found"}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
        let client = Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.completion_model(LLAMA3_2);
        let request = model.completion_request("hello").build();

        let error = model
            .completion(request)
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
        use crate::client::EmbeddingsClient;
        use crate::embeddings::EmbeddingModel;
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"error":"model not found"}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
        let client = Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.embedding_model(ALL_MINILM);

        let error = model
            .embed_texts(vec!["hello".to_string()])
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
