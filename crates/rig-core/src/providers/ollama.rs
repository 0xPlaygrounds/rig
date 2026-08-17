//! Ollama API client and Rig integration
//!
//! # Example
//! ```no_run
//! use rig_core::{
//!     client::{CompletionClient, EmbeddingsClient, Nothing},
//!     completion::CompletionModel,
//!     embeddings::EmbeddingModel,
//!     providers::ollama,
//! };
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a new Ollama client (defaults to http://localhost:11434, no auth)
//! let client = ollama::Client::new(Nothing)?;
//!
//! // Or connect to a remote/proxied Ollama instance with authentication
//! let client = ollama::Client::builder()
//!     .api_key("my-secret-key")
//!     .base_url("http://remote-ollama:11434")
//!     .build()?;
//!
//! // Send a completion request with a preamble.
//! let model = client.completion_model("qwen2.5:14b");
//! let request = model
//!     .completion_request("Entertain me!")
//!     .preamble("You are a comedian here to entertain the user using humour and jokes.".to_string())
//!     .build();
//! let response = model.completion(request).await?;
//! println!("{:?}", response.choice);
//!
//! // Create an embedding model using the "all-minilm" model
//! let emb_model = client.embedding_model_with_ndims("all-minilm", 384);
//! let embeddings = emb_model.embed_texts(vec![
//!     "Why is the sky blue?".to_owned(),
//!     "Why is the grass green?".to_owned()
//! ]).await?;
//! println!("Embedding response: {:?}", embeddings);
//! # Ok(())
//! # }
//! ```
use crate::client::{self, ApiKey, DebugExt, ModelLister, Nothing, Provider, ProviderClient};
use crate::completion::Usage;
use crate::http_client::{self, HttpClientExt};
use crate::message::DocumentSourceKind;
use crate::model::{Model, ModelList, ModelListingError};
use crate::providers::internal;
use crate::streaming::{RawStreamingChoice, RawStreamingResult, StreamFinal};
use crate::telemetry::{CompletionOperation, CompletionSpanBuilder, SpanCombinator};
use crate::{
    completion::{self, CompletionError, CompletionRequest},
    embeddings::{self, EmbeddingError},
    json_utils, message,
    message::Text,
    streaming,
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};
use async_stream::stream;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::convert::TryFrom;
use tracing_futures::Instrument;
// ---------- Main Client ----------

const OLLAMA_API_BASE_URL: &str = "http://localhost:11434";

/// Stable descriptor name recorded on normalized responses, streams, and
/// telemetry spans for this provider.
const PROVIDER_NAME: &str = "ollama";

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

client::impl_capabilities!(
    OllamaExt,
    completion = CompletionModel<H>,
    embeddings = EmbeddingModel<H>,
    model_listing = OllamaModelLister<H>,
);

impl DebugExt for OllamaExt {}

client::impl_default_provider_builder!(
    OllamaBuilder => OllamaExt,
    api_key = OllamaApiKey,
    base_url = OLLAMA_API_BASE_URL,
);

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
/// Map Ollama's `done_reason` onto rig's normalized vocabulary.
///
/// Ollama documents `stop` and `length`, but also emits operational reasons
/// such as `load`/`unload`; those are carried verbatim in Ollama's own spelling
/// rather than being flattened into a natural stop.
pub(crate) fn map_done_reason(reason: &str) -> completion::FinishReason {
    match reason {
        "stop" => completion::FinishReason::Stop,
        "length" => completion::FinishReason::Length,
        other => completion::FinishReason::Other(other.to_owned()),
    }
}

impl From<&CompletionResponse> for Usage {
    fn from(response: &CompletionResponse) -> Usage {
        let input_tokens = response.prompt_eval_count.unwrap_or(0);
        let output_tokens = response.eval_count.unwrap_or(0);
        crate::providers::internal::completion_usage(
            input_tokens,
            output_tokens,
            input_tokens + output_tokens,
            0,
        )
    }
}

impl crate::telemetry::ProviderResponseExt for CompletionResponse {
    type Usage = Usage;

    /// Ollama's chat API carries no response ID.
    fn get_response_id(&self) -> Option<String> {
        None
    }

    fn get_response_model_name(&self) -> Option<String> {
        Some(self.model.clone())
    }

    fn get_text_response(&self) -> Option<String> {
        match &self.message {
            Message::Assistant { content, .. } if !content.is_empty() => Some(content.clone()),
            _ => None,
        }
    }

    fn get_usage(&self) -> Option<Self::Usage> {
        Some(Usage::from(self))
    }
}

impl TryFrom<CompletionResponse> for completion::CompletionResponse {
    type Error = CompletionError;
    fn try_from(resp: CompletionResponse) -> Result<Self, Self::Error> {
        let usage = Usage::from(&resp);
        let finish_reason = resp.done_reason.as_deref().map(map_done_reason);
        let model = resp.model.clone();
        let permits_omitted_think_start = resp.model.to_ascii_lowercase().contains("qwen3");

        // Process only if an assistant message is present.
        let Message::Assistant {
            content,
            thinking,
            tool_calls,
            ..
        } = resp.message
        else {
            return Err(CompletionError::ResponseError(
                "Chat response does not include an assistant message".into(),
            ));
        };

        let mut assistant_contents = Vec::new();
        let (legacy_thinking, visible_content) = if matches!(thinking.as_deref(), None | Some("")) {
            split_legacy_thinking(&content, permits_omitted_think_start)
        } else {
            (None, content.as_str())
        };
        // Preserve the model's reasoning so it round-trips into agent history
        // and is echoed back to Ollama on the next turn (issue #1926). `choice`
        // is the only place it can live — the normalized response carries no
        // provider payload — so dropping it here would lose the reasoning
        // entirely, unlike the streaming path (see
        // `RawStreamingChoice::ReasoningDelta` below).
        if let Some(thinking) = thinking.as_deref().filter(|t| !t.is_empty()) {
            assistant_contents.push(completion::AssistantContent::reasoning(thinking));
        }
        if let Some(legacy_thinking) = legacy_thinking {
            assistant_contents.push(completion::AssistantContent::reasoning(legacy_thinking));
        }
        // Add the assistant's text content if any.
        if !visible_content.is_empty() {
            assistant_contents.push(completion::AssistantContent::text(visible_content));
        }
        // Process tool_calls following Ollama's chat response definition.
        // Modern daemons issue a call id (`"id":"call_..."`); it is read as
        // the provider id when present. An absent id mints the correlation
        // handle and records no provider id — never a name-as-id (which
        // would collide two same-tool calls) and never an empty sentinel.
        // Replay drops the id either way (Ollama tool messages correlate
        // by `tool_name`).
        for tc in tool_calls.iter() {
            assistant_contents.push(completion::AssistantContent::tool_call(
                tc.id.as_deref().unwrap_or(""),
                tc.function.name.clone(),
                tc.function.arguments.clone(),
            ));
        }
        let choice = crate::message::require_non_empty_response(assistant_contents)?;

        Ok(
            completion::CompletionResponse::new(choice, usage, PROVIDER_NAME)
                .with_model(model)
                .with_optional_finish_reason(finish_reason),
        )
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
        // Ollama tool messages are name-keyed: cross-provider ingested
        // results arrive with an empty name and their call carries it.
        crate::providers::internal::resolve_empty_tool_result_names(&mut partial_history);

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

#[derive(Clone)]
pub struct CompletionModel<T = reqwest::Client> {
    client: Client<T>,
    pub model: String,
}

impl<T> CompletionModel<T> {
    pub fn new(client: Client<T>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }
}

impl<T> crate::client::ConstructCompletionModel<Client<T>> for CompletionModel<T>
where
    Client<T>: Clone,
{
    fn construct(client: &Client<T>, model: String) -> Self {
        Self::new(client.clone(), model)
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

/// Ollama's terminal stream record, kept provider-native for
/// [`CompletionModel::raw_stream`].
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct StreamingCompletionResponse {
    /// Provider-reported model identifier from the terminating NDJSON line.
    pub model: String,
    pub done_reason: Option<String>,
    pub total_duration: Option<u64>,
    pub load_duration: Option<u64>,
    pub prompt_eval_count: Option<u64>,
    pub prompt_eval_duration: Option<u64>,
    pub eval_count: Option<u64>,
    pub eval_duration: Option<u64>,
}

impl From<&StreamingCompletionResponse> for Usage {
    fn from(response: &StreamingCompletionResponse) -> Usage {
        let input_tokens = response.prompt_eval_count.unwrap_or_default();
        let output_tokens = response.eval_count.unwrap_or_default();
        crate::providers::internal::completion_usage(
            input_tokens,
            output_tokens,
            input_tokens + output_tokens,
            0,
        )
    }
}

impl From<StreamingCompletionResponse> for StreamFinal {
    fn from(response: StreamingCompletionResponse) -> StreamFinal {
        // Ollama's `/api/chat` stream assigns no message identifier, so the
        // normalized `message_id` stays unset.
        StreamFinal::new(PROVIDER_NAME, Usage::from(&response))
            .with_optional_finish_reason(response.done_reason.as_deref().map(map_done_reason))
            .with_model(response.model)
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

impl<T> CompletionModel<T>
where
    T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
{
    /// Execute a completion and return Ollama's own wire response.
    ///
    /// This is the escape hatch for Ollama-specific fields rig does not
    /// normalize (the timing counters, `created_at`). It shares the request
    /// builder, transport, telemetry, and error handling with
    /// [`CompletionModel::completion`](completion::CompletionModel::completion),
    /// which calls it and then applies the provider-local mapping — one network
    /// request either way.
    pub async fn raw_completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<CompletionResponse, CompletionError> {
        let system_instructions = completion_request.preamble.clone();
        let record_telemetry_content = completion_request.record_telemetry_content;
        let request = OllamaCompletionRequest::try_from((self.model.as_ref(), completion_request))?;
        let span =
            CompletionSpanBuilder::new(PROVIDER_NAME, &request.model, CompletionOperation::Chat)
                .system_instructions(system_instructions.as_deref(), record_telemetry_content)
                .build();

        internal::trace_json(
            crate::providers::internal::LogTarget::Completions,
            "Ollama completion request",
            &request,
        );

        let body = serde_json::to_vec(&request)?;

        let req = self
            .client
            .post("api/chat")?
            .body(body)
            .map_err(http_client::Error::from)?;

        let async_block = internal::completion_send::send_completion::<
            _,
            internal::envelope::DirectPayload<CompletionResponse>,
            _,
        >(
            &self.client,
            req,
            "Ollama completion",
            // A local Ollama server reports no request-id response header.
            None,
            |response| {
                let span = tracing::Span::current();
                span.record_response_metadata(response);
                span.record_token_usage(&Usage::from(response));
            },
        );

        tracing::Instrument::instrument(async_block, span)
            .await
            .map(|(payload, _)| payload)
    }

    /// Open a stream whose terminal record stays Ollama-native.
    ///
    /// This is the escape hatch for Ollama's own terminal payload; it shares the
    /// request builder, transport, telemetry, and error handling with
    /// [`CompletionModel::stream`](completion::CompletionModel::stream), which
    /// calls it and normalizes the terminal record once through
    /// [`streaming::normalize_stream`] — one network request either way.
    pub async fn raw_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<RawStreamingResult<StreamingCompletionResponse>, CompletionError> {
        let system_instructions = request.preamble.clone();
        let record_telemetry_content = request.record_telemetry_content;
        let mut request = OllamaCompletionRequest::try_from((self.model.as_ref(), request))?;
        let span = CompletionSpanBuilder::new(
            PROVIDER_NAME,
            &request.model,
            CompletionOperation::ChatStreaming,
        )
        .system_instructions(system_instructions.as_deref(), record_telemetry_content)
        .build();
        request.stream = true;

        internal::trace_json(
            crate::providers::internal::LogTarget::Completions,
            "Ollama streaming completion request",
            &request,
        );

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

        // Transport layer: HTTP byte chunks → NDJSON-line `WireFrame`s. Byte
        // splitting and framing only — classification and policy live
        // downstream.
        let transport = stream! {
            let mut line_buf = NdjsonBuffer::new();
            while let Some(chunk) = byte_stream.next().await {
                let bytes = match chunk {
                    Ok(bytes) => bytes,
                    Err(e) => {
                        yield Err(CompletionError::from(http_client::Error::Instance(e.into())));
                        break;
                    }
                };

                for line in line_buf.decode(&bytes) {
                    tracing::debug!(target: "rig", "Received NDJSON line from Ollama: {}", String::from_utf8_lossy(&line));
                    yield Ok(internal::adapter::WireFrame::Bytes(line));
                }
            }
        };

        let stream: RawStreamingResult<StreamingCompletionResponse> = Box::pin(
            internal::adapter::run_wire_stream(transport, OllamaAdapter::default())
                .instrument(span),
        );

        Ok(stream)
    }
}

/// The Ollama NDJSON wire as a
/// [`WireAdapter`](internal::adapter::WireAdapter).
///
/// Stateless: every line is a whole response record. Frame-triage policy
/// (warn-skip `Unknown` — unpopulated on this undiscriminated wire — and
/// in-band `Err` on `Corrupt`, so a later genuine `done: true` record can
/// still complete the stream) lives in
/// [`run_wire_stream`](internal::adapter::run_wire_stream), not here.
struct OllamaAdapter {
    /// Owns the constant-key reasoning lifecycle: `thinking` deltas
    /// accumulate under the per-stream minted key, and the boundary end
    /// this wire never announces is derived, not hand-rolled here.
    reasoning: internal::chunk_lifecycle::MintedReasoningLifecycle,
    /// Per-stream minter for id-less tool-call keys. Counted across the
    /// whole stream, not per record — a per-record enumeration would hand
    /// two id-less calls in separate records the same `Minted(Tool, 0)`
    /// key, and one would silently swallow the other downstream.
    tool_ids: crate::streaming::SyntheticIds,
}

impl Default for OllamaAdapter {
    fn default() -> Self {
        Self {
            reasoning: internal::chunk_lifecycle::MintedReasoningLifecycle::new(
                crate::streaming::StreamPartId::minted(crate::streaming::MintKind::Reasoning, 0),
            ),
            tool_ids: crate::streaming::SyntheticIds::tool(),
        }
    }
}

impl internal::adapter::WireAdapter for OllamaAdapter {
    type Frame = internal::adapter::WireFrame;
    type Event = CompletionResponse;
    type Response = StreamingCompletionResponse;

    fn classify(&self, frame: Self::Frame) -> internal::wire::WireEvent<CompletionResponse> {
        match frame {
            internal::adapter::WireFrame::Bytes(line) => {
                internal::wire::classify_untyped_line(&line)
            }
            internal::adapter::WireFrame::Text(line) => {
                internal::wire::classify_untyped_line(line.as_bytes())
            }
        }
    }

    fn interpret(
        &mut self,
        response: CompletionResponse,
        out: &mut internal::adapter::AdapterOutput<Self::Response>,
    ) {
        let span = tracing::Span::current();
        if response.done {
            span.record("gen_ai.response.model", &response.model);
        }

        if let Message::Assistant {
            content,
            thinking,
            tool_calls,
            ..
        } = response.message
        {
            // A daemon-issued call id keys the stream and travels as the
            // durable id; an id-less call (older daemons) keys by a
            // distinct minted identity and its durable id stays absent —
            // never the tool name, which would collide two same-tool calls
            // in one turn.
            let mut tool_events = Vec::with_capacity(tool_calls.len());
            for tool_call in tool_calls {
                let key = match tool_call
                    .id
                    .as_deref()
                    .and_then(crate::streaming::WireId::new)
                {
                    Some(wire_id) => crate::streaming::StreamPartId::wire(wire_id.as_str()),
                    None => self.tool_ids.mint(),
                };
                tool_events.push(RawStreamingChoice::ToolCall(
                    crate::streaming::RawStreamingToolCall::new(
                        key,
                        tool_call.function.name,
                        tool_call.function.arguments,
                    ),
                ));
            }

            // Declare what the record carried; the shared lifecycle derives
            // the canonical sequence (boundary end included).
            self.reasoning.emit_chunk(
                internal::chunk_lifecycle::ChunkParts {
                    reasoning: thinking,
                    reasoning_signature: None,
                    text: Some(content),
                    tool_events,
                },
                out,
            );
        }

        // Only a `done: true` record counts as the provider completing the
        // turn; the driver stops consuming after the terminal record.
        if response.done {
            span.record("gen_ai.usage.input_tokens", response.prompt_eval_count);
            span.record("gen_ai.usage.output_tokens", response.eval_count);
            out.push(Ok(RawStreamingChoice::FinalResponse(
                StreamingCompletionResponse {
                    model: response.model,
                    total_duration: response.total_duration,
                    load_duration: response.load_duration,
                    prompt_eval_count: response.prompt_eval_count,
                    prompt_eval_duration: response.prompt_eval_duration,
                    eval_count: response.eval_count,
                    eval_duration: response.eval_duration,
                    done_reason: response.done_reason,
                },
            )));
        }
    }

    fn finish(&mut self, _out: &mut internal::adapter::AdapterOutput<Self::Response>) {
        // EOF without a `done: true` record is truncation: no terminal record
        // may be synthesized.
    }
}

impl<T> completion::CompletionModel for CompletionModel<T>
where
    T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
{
    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse, CompletionError> {
        // Capture before `try_into` consumes the raw value.
        let raw = self.raw_completion(completion_request).await?;
        let captured = serde_json::to_value(&raw)?;
        let response: completion::CompletionResponse = raw.try_into()?;
        Ok(response.with_raw(captured))
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<streaming::StreamingCompletionResponse, CompletionError> {
        let stream = self.raw_stream(request).await?;
        let normalized =
            streaming::normalize_stream(stream, |response: StreamingCompletionResponse| {
                Ok(response.into())
            });

        Ok(streaming::StreamingCompletionResponse::stream(
            PROVIDER_NAME,
            normalized,
        ))
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
        let api_resp: ListModelsResponse = crate::providers::internal::model_listing::get_json(
            &self.client,
            "Ollama",
            "/api/tags",
        )
        .await?;
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
    /// The daemon-issued call id (`"id":"call_..."`), present on modern
    /// Ollama daemons and absent on older ones. Read when present — it is
    /// the durable handle that distinguishes two same-tool calls in one
    /// turn — but never serialized back: Ollama's request schema correlates
    /// tool messages by `tool_name`, and replayed histories predate the id.
    #[serde(default, skip_serializing)]
    pub id: Option<String>,
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
        #[serde(default, deserialize_with = "json_utils::null_or_default")]
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
                            name,
                            content,
                            ..
                        }) => {
                            // The executed tool's name travels as required data.
                            let function_name = name;
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
                            messages.push(Message::ToolResult {
                                name: function_name,
                                content,
                            });
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

                // Both fields may be empty. This used to lean on the non-empty
                // content type to argue that at least one of them was populated;
                // content is a `Vec` now, so an assistant turn that carried
                // nothing renders as an Ollama message with empty text and no
                // tool calls, which is what such a turn actually was.
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
///
/// An assistant message with empty `content` and no thinking or tool calls
/// converts to **empty** message content — no fabricated empty-text block.
/// Such a message cannot be replayed through the request boundary
/// (`validate_message_content` rejects a content-less assistant message);
/// callers ingesting raw Ollama history should filter empty assistant
/// messages rather than expect rig to invent content for them. The agent
/// loop never produces this shape: it drops empty turns before history.
impl From<Message> for crate::completion::Message {
    fn from(msg: Message) -> Self {
        match msg {
            Message::User { content, .. } => crate::completion::Message::User {
                content: vec![crate::completion::message::UserContent::Text(Text::new(
                    content,
                ))],
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
                // Only a non-empty text body becomes a text block. Pushing
                // unconditionally would mint the legacy `vec![Text("")]`
                // sentinel for a content-less assistant message — the shape
                // `is_empty_assistant_turn` documents as produced by old
                // persisted histories only. Empty content is representable
                // now, and the agent layer handles it.
                if !content.is_empty() {
                    assistant_contents.push(crate::completion::message::AssistantContent::Text(
                        Text::new(content),
                    ));
                }
                // Same id policy as the unary decode above: a daemon-issued
                // id is preserved, an absent one mints (provider id: none).
                for tc in tool_calls {
                    assistant_contents.push(
                        crate::completion::message::AssistantContent::tool_call(
                            tc.id.as_deref().unwrap_or(""),
                            tc.function.name,
                            tc.function.arguments,
                        ),
                    );
                }
                crate::completion::Message::Assistant {
                    id: None,
                    content: assistant_contents,
                }
            }
            // System and ToolResult are converted to User message as needed.
            Message::System { content, .. } => crate::completion::Message::User {
                content: vec![crate::completion::message::UserContent::Text(Text::new(
                    content,
                ))],
            },
            Message::ToolResult { name, content } => crate::completion::Message::User {
                // Ollama tool messages carry no call id; the name is the
                // wire's correlator and the rig-level handle is minted.
                content: vec![message::UserContent::tool_result_from_wire(
                    "",
                    name,
                    vec![message::ToolResultContent::text(content)],
                )],
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
            // Never serialized (replay correlates by `tool_name`); the
            // request shape is id-less regardless of what history holds.
            id: None,
            r#type: ToolType::Function,
            function: Function {
                name: tool_call.function.name,
                arguments: tool_call.function.arguments,
            },
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

    // The NDJSON wire has no discriminator, so its classify has exactly two
    // outcomes: the response shape or corrupt.
    #[test]
    fn classify_ndjson_line_is_known_or_corrupt() {
        let line = json!({
            "model": "llama3.2",
            "created_at": "2024-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": "hi"},
            "done": false,
        })
        .to_string();
        assert!(matches!(
            internal::wire::classify_untyped_line::<CompletionResponse>(line.as_bytes()),
            internal::wire::WireEvent::Known(_)
        ));
        assert!(matches!(
            internal::wire::classify_untyped_line::<CompletionResponse>(b"{not json"),
            internal::wire::WireEvent::Corrupt(_)
        ));
        assert!(matches!(
            internal::wire::classify_untyped_line::<CompletionResponse>(br#"{"done": 42}"#),
            internal::wire::WireEvent::Corrupt(_)
        ));
    }

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

    #[test]
    fn done_reason_maps_documented_values_and_preserves_the_rest() {
        assert_eq!(map_done_reason("stop"), completion::FinishReason::Stop);
        assert_eq!(map_done_reason("length"), completion::FinishReason::Length);
        // Ollama's operational reasons have no normalized equivalent, so they
        // are carried through verbatim rather than read as a natural stop.
        assert_eq!(
            map_done_reason("load"),
            completion::FinishReason::Other("load".to_owned())
        );
        assert_eq!(
            map_done_reason("unload"),
            completion::FinishReason::Other("unload".to_owned())
        );
    }

    #[test]
    fn response_metadata_is_normalized() {
        let response: CompletionResponse = serde_json::from_value(json!({
            "model": "llama3.2",
            "created_at": "2023-08-04T19:22:45.499127Z",
            "message": {"role": "assistant", "content": "Hi!", "tool_calls": []},
            "done": true,
            "done_reason": "length",
            "prompt_eval_count": 12u64,
            "eval_count": 3u64
        }))
        .expect("fixture should deserialize");

        let normalized: completion::CompletionResponse =
            response.try_into().expect("normalization should succeed");

        assert_eq!(normalized.provider, PROVIDER_NAME);
        assert_eq!(normalized.model.as_deref(), Some("llama3.2"));
        assert_eq!(
            normalized.finish_reason(),
            Some(completion::FinishReason::Length)
        );
        // Ollama assigns no message identifier.
        assert_eq!(normalized.message_id, None);
        assert_eq!(normalized.usage.input_tokens, 12);
        assert_eq!(normalized.usage.output_tokens, 3);
        assert_eq!(normalized.usage.total_tokens, 15);
    }

    // A `done_reason` of `stop` on a turn that actually called a tool must be
    // upgraded by the response builder's reconciliation.
    #[test]
    fn tool_call_turn_upgrades_a_plain_stop_to_tool_calls() {
        let response: CompletionResponse = serde_json::from_value(json!({
            "model": "qwen3:4b",
            "created_at": "2023-08-04T19:22:45.499127Z",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"type": "function", "function": {"name": "get_weather", "arguments": {"location": "Berlin"}}}
                ]
            },
            "done": true,
            "done_reason": "stop"
        }))
        .expect("fixture should deserialize");

        let normalized: completion::CompletionResponse =
            response.try_into().expect("normalization should succeed");

        assert_eq!(
            normalized.finish_reason(),
            Some(completion::FinishReason::ToolCalls)
        );
    }

    #[test]
    fn streaming_terminal_record_is_normalized() {
        let terminal = StreamingCompletionResponse {
            model: "llama3.2".to_string(),
            done_reason: Some("dragons".to_string()),
            total_duration: None,
            load_duration: None,
            prompt_eval_count: Some(7),
            prompt_eval_duration: None,
            eval_count: Some(5),
            eval_duration: None,
        };

        let final_record = StreamFinal::from(terminal);
        assert_eq!(final_record.provider, PROVIDER_NAME);
        assert_eq!(final_record.model.as_deref(), Some("llama3.2"));
        assert_eq!(
            final_record.finish_reason,
            Some(completion::FinishReason::Other("dragons".to_owned()))
        );
        assert_eq!(final_record.usage.total_tokens, 12);
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
                let first_content = content.first();
                // The expected type is crate::completion::message::UserContent::Text wrapping a Text struct.
                match first_content {
                    Some(crate::completion::message::UserContent::Text(text_struct)) => {
                        assert_eq!(text_struct.text, "Test message");
                    }
                    _ => panic!("Expected text content in conversion"),
                }
            }
            _ => panic!("Conversion from provider Message to completion Message failed"),
        }
    }

    #[test]
    fn empty_assistant_history_converts_to_empty_content_not_a_sentinel() {
        // A content-less Ollama assistant message converts to genuinely empty
        // message content — no fabricated `Text("")` block. Pinned because the
        // consequence is deliberate: such a message cannot be replayed through
        // the request boundary, and callers ingesting raw Ollama history
        // filter it rather than rig inventing content (see the `From` doc).
        let provider_msg = Message::Assistant {
            content: String::new(),
            thinking: None,
            images: None,
            name: None,
            tool_calls: Vec::new(),
        };
        let comp_msg: crate::completion::Message = provider_msg.into();
        match comp_msg {
            crate::completion::Message::Assistant { content, .. } => {
                assert!(content.is_empty(), "expected empty content: {content:?}");
            }
            other => panic!("expected an assistant message, got {other:?}"),
        }

        // A non-empty body still converts to exactly one text block.
        let provider_msg = Message::Assistant {
            content: "hello".to_owned(),
            thinking: None,
            images: None,
            name: None,
            tool_calls: Vec::new(),
        };
        let comp_msg: crate::completion::Message = provider_msg.into();
        match comp_msg {
            crate::completion::Message::Assistant { content, .. } => {
                assert!(
                    matches!(
                        content.as_slice(),
                        [crate::completion::message::AssistantContent::Text(text)]
                            if text.text == "hello"
                    ),
                    "unexpected content: {content:?}"
                );
            }
            other => panic!("expected an assistant message, got {other:?}"),
        }
    }

    #[test]
    fn mixed_user_content_preserves_message_order() {
        use crate::message::{Message as RigMessage, ToolResultContent, UserContent};

        let message = RigMessage::User {
            content: vec![
                UserContent::text("before"),
                UserContent::tool_result(
                    "",
                    "lookup",
                    vec![ToolResultContent::json(json!({ "ok": true }))],
                ),
                UserContent::text("after"),
            ],
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
        use crate::message::{ImageMediaType, Message as RigMessage, UserContent};

        let message = RigMessage::User {
            content: vec![UserContent::image_url(
                "https://example.com/image.png",
                Some(ImageMediaType::PNG),
                None,
            )],
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
            content: vec![
                crate::message::AssistantContent::Reasoning(reasoning_content),
                crate::message::AssistantContent::Text(crate::message::Text::new(
                    "The answer is X".to_string(),
                )),
            ],
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

    /// A user-supplied ollama-format assistant message carrying a
    /// daemon-issued call id keeps it through conversion — the same id
    /// policy as the unary decode (preserve when present, absent mints).
    #[test]
    fn wire_message_conversion_preserves_the_daemon_tool_call_id() {
        let wire = Message::Assistant {
            content: String::new(),
            thinking: None,
            images: None,
            name: None,
            tool_calls: vec![ToolCall {
                id: Some("call_abc".to_owned()),
                r#type: ToolType::default(),
                function: Function {
                    name: "get_weather".to_owned(),
                    arguments: json!({}),
                },
            }],
        };

        let converted: crate::completion::Message = wire.into();
        let crate::completion::Message::Assistant { content, .. } = converted else {
            panic!("Expected Assistant message");
        };
        let ids: Vec<String> = content
            .iter()
            .filter_map(|item| match item {
                crate::message::AssistantContent::ToolCall(call) => {
                    Some(call.id.as_str().to_owned())
                }
                _ => None,
            })
            .collect();
        assert_eq!(ids, vec!["call_abc".to_owned()]);
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
        use crate::completion::Message as CompletionMessage;
        use crate::message::{Text, UserContent};

        // Create a CompletionRequest with "think": true, "keep_alive", and "num_ctx" in additional_params
        let completion_request = CompletionRequest {
            model: None,
            preamble: Some("You are a helpful assistant.".to_string()),
            chat_history: vec![CompletionMessage::User {
                content: vec![UserContent::Text(Text::new("What is 2 + 2?".to_string()))],
            }],
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
        use crate::completion::Message as CompletionMessage;
        use crate::message::{Text, UserContent};

        // Create a CompletionRequest with "think": true, "keep_alive", and "num_ctx" in additional_params
        let completion_request = CompletionRequest {
            model: None,
            preamble: Some("You are a helpful assistant.".to_string()),
            chat_history: vec![CompletionMessage::User {
                content: vec![UserContent::Text(Text::new("What is 2 + 2?".to_string()))],
            }],
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
        use crate::completion::Message as CompletionMessage;
        use crate::message::{Text, UserContent};

        // Create a CompletionRequest with "think": true, "keep_alive", and "num_ctx" in additional_params
        let completion_request = CompletionRequest {
            model: None,
            preamble: Some("You are a helpful assistant.".to_string()),
            chat_history: vec![CompletionMessage::User {
                content: vec![UserContent::Text(Text::new("What is 2 + 2?".to_string()))],
            }],
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
        use crate::completion::Message as CompletionMessage;
        use crate::message::{Text, UserContent};

        // Create a CompletionRequest with "think": true, "keep_alive", and "num_ctx" in additional_params
        let completion_request = CompletionRequest {
            model: None,
            preamble: Some("You are a helpful assistant.".to_string()),
            chat_history: vec![CompletionMessage::User {
                content: vec![UserContent::Text(Text::new("What is 2 + 2?".to_string()))],
            }],
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
        use crate::completion::Message as CompletionMessage;
        use crate::message::{Text, UserContent};

        // Create a CompletionRequest with "think": true, "keep_alive", and "num_ctx" in additional_params
        let completion_request = CompletionRequest {
            model: None,
            preamble: Some("You are a helpful assistant.".to_string()),
            chat_history: vec![CompletionMessage::User {
                content: vec![UserContent::Text(Text::new("What is 2 + 2?".to_string()))],
            }],
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
        use crate::completion::Message as CompletionMessage;
        use crate::message::{Text, UserContent};

        // Create a CompletionRequest WITHOUT "think" in additional_params
        let completion_request = CompletionRequest {
            model: None,
            preamble: Some("You are a helpful assistant.".to_string()),
            chat_history: vec![CompletionMessage::User {
                content: vec![UserContent::Text(Text::new("Hello!".to_string()))],
            }],
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
        use crate::completion::Message as CompletionMessage;
        use crate::message::{Text, UserContent};

        let completion_request = CompletionRequest {
            model: None,
            preamble: None,
            chat_history: vec![CompletionMessage::User {
                content: vec![UserContent::Text(Text::new("Hello!".to_string()))],
            }],
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
        use crate::completion::Message as CompletionMessage;
        use crate::message::{Text, UserContent};

        let completion_request = CompletionRequest {
            model: None,
            preamble: None,
            chat_history: vec![CompletionMessage::User {
                content: vec![UserContent::Text(Text::new("Hello!".to_string()))],
            }],
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
        use crate::completion::Message as CompletionMessage;
        use crate::message::{Text, UserContent};

        let completion_request = CompletionRequest {
            model: None,
            preamble: None,
            chat_history: vec![CompletionMessage::User {
                content: vec![UserContent::Text(Text::new("Hello!".to_string()))],
            }],
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
            chat_history: vec![CompletionMessage::User {
                content: vec![UserContent::Text(Text::new(
                    "How old is Ollama?".to_string(),
                ))],
            }],
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
        use crate::completion::Message as CompletionMessage;
        use crate::message::{Text, UserContent};

        let completion_request = CompletionRequest {
            model: Some("llama3.1".to_string()),
            preamble: None,
            chat_history: vec![CompletionMessage::User {
                content: vec![UserContent::Text(Text::new("Hello!".to_string()))],
            }],
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

    // Proves a truncated NDJSON stream — content chunks then EOF without a
    // `done: true` record — delivers its content but never a synthesized
    // terminal record.
    #[tokio::test]
    async fn truncated_stream_does_not_synthesize_a_terminal_record() {
        use crate::client::CompletionClient;
        use crate::completion::CompletionModel;
        use crate::streaming::StreamedAssistantContent;
        use crate::test_utils::MockStreamingClient;
        use futures::StreamExt;

        let ndjson = concat!(
            r#"{"model":"llama3.2","created_at":"2023-08-04T19:22:45.499127Z","message":{"role":"assistant","content":"hi"},"done":false}"#,
            "\n",
        );
        let client = Client::builder()
            .api_key("test-key")
            .http_client(MockStreamingClient {
                sse_bytes: bytes::Bytes::from(ndjson),
            })
            .build()
            .expect("build client");
        let model = client.completion_model(LLAMA3_2);
        let request = model.completion_request("hello").build();

        let mut stream = model.stream(request).await.expect("stream should open");

        let mut texts = Vec::new();
        let mut saw_terminal = false;
        while let Some(item) = stream.next().await {
            match item.expect("stream item should be Ok") {
                StreamedAssistantContent::Text(text) => texts.push(text.text),
                StreamedAssistantContent::Final(_) => saw_terminal = true,
                _ => {}
            }
        }

        assert_eq!(texts, ["hi"]);
        assert!(
            !saw_terminal,
            "EOF without a done record must not synthesize a terminal record"
        );
        assert!(stream.response.is_none());
    }

    // Proves a malformed NDJSON line between valid lines surfaces as an
    // `Err` item while the stream keeps consuming: the following content and
    // the `done: true` record still arrive.
    #[tokio::test]
    async fn malformed_line_is_surfaced_and_the_terminal_still_arrives() {
        use crate::client::CompletionClient;
        use crate::completion::CompletionModel;
        use crate::streaming::StreamedAssistantContent;
        use crate::test_utils::MockStreamingClient;
        use futures::StreamExt;

        let ndjson = concat!(
            r#"{"model":"llama3.2","created_at":"2023-08-04T19:22:45.499127Z","message":{"role":"assistant","content":"hi"},"done":false}"#,
            "\n",
            "{not json\n",
            r#"{"model":"llama3.2","created_at":"2023-08-04T19:22:46.499127Z","message":{"role":"assistant","content":" there"},"done":false}"#,
            "\n",
            r#"{"model":"llama3.2","created_at":"2023-08-04T19:22:47.499127Z","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":10,"eval_count":4}"#,
            "\n",
        );
        let client = Client::builder()
            .api_key("test-key")
            .http_client(MockStreamingClient {
                sse_bytes: bytes::Bytes::from(ndjson),
            })
            .build()
            .expect("build client");
        let model = client.completion_model(LLAMA3_2);
        let request = model.completion_request("hello").build();

        let mut stream = model.stream(request).await.expect("stream should open");

        let mut texts = Vec::new();
        let mut saw_error = false;
        let mut terminal = None;
        while let Some(item) = stream.next().await {
            match item {
                Ok(StreamedAssistantContent::Text(text)) => texts.push(text.text),
                Ok(StreamedAssistantContent::Final(final_response)) => {
                    terminal = Some(final_response)
                }
                Ok(_) => {}
                Err(_) => saw_error = true,
            }
        }

        assert_eq!(texts, ["hi", " there"]);
        assert!(saw_error, "the malformed line must reach the consumer");
        let terminal = terminal.expect("the genuine done record must still arrive");
        assert_eq!(terminal.usage.input_tokens, 10);
        assert_eq!(terminal.usage.output_tokens, 4);
    }

    // Proves the `done: true` record ends the stream: a content line that
    // arrives after it is never yielded — only the pre-done content and the
    // terminal record reach the consumer.
    #[tokio::test]
    async fn content_after_the_done_record_is_not_yielded() {
        use crate::client::CompletionClient;
        use crate::completion::CompletionModel;
        use crate::streaming::StreamedAssistantContent;
        use crate::test_utils::MockStreamingClient;
        use futures::StreamExt;

        let ndjson = concat!(
            r#"{"model":"llama3.2","created_at":"2023-08-04T19:22:45.499127Z","message":{"role":"assistant","content":"hi"},"done":false}"#,
            "\n",
            r#"{"model":"llama3.2","created_at":"2023-08-04T19:22:46.499127Z","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":10,"eval_count":4}"#,
            "\n",
            r#"{"model":"llama3.2","created_at":"2023-08-04T19:22:47.499127Z","message":{"role":"assistant","content":"stray"},"done":false}"#,
            "\n",
        );
        let client = Client::builder()
            .api_key("test-key")
            .http_client(MockStreamingClient {
                sse_bytes: bytes::Bytes::from(ndjson),
            })
            .build()
            .expect("build client");
        let model = client.completion_model(LLAMA3_2);
        let request = model.completion_request("hello").build();

        let mut stream = model.stream(request).await.expect("stream should open");

        let mut texts = Vec::new();
        let mut terminal = None;
        while let Some(item) = stream.next().await {
            match item.expect("stream item should be Ok") {
                StreamedAssistantContent::Text(text) => texts.push(text.text),
                StreamedAssistantContent::Final(final_response) => {
                    assert!(
                        terminal.is_none(),
                        "the terminal record must be yielded exactly once"
                    );
                    terminal = Some(final_response);
                }
                other => panic!("unexpected stream item: {other:?}"),
            }
        }

        assert_eq!(
            texts,
            ["hi"],
            "content after the done record must not be yielded"
        );
        let terminal = terminal.expect("the done record must yield the terminal record");
        assert_eq!(terminal.usage.input_tokens, 10);
        assert_eq!(terminal.usage.output_tokens, 4);
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

    /// Raw-capture tests: the `TryFrom` shape, driven end to end through
    /// `CompletionModel::completion` over the recording mock transport. Ollama
    /// has no request-id contract, so there is nothing transport-side to
    /// reattach; the capture is the `/api/chat` body exactly as `raw_completion`
    /// parses it. The body carries the timing fields (`total_duration`,
    /// `eval_duration`, ...) rig never normalizes, so the capture can be shown
    /// to answer more than the normalized response does.
    mod raw_capture {
        use super::*;
        use crate::client::CompletionClient;
        use crate::completion::CompletionModel as _;
        use crate::test_utils::RecordingHttpClient;

        const BODY: &str = r#"{
            "model": "llama3.2",
            "created_at": "2023-08-04T19:22:45.499127Z",
            "message": {"role": "assistant", "content": "hello"},
            "done": true,
            "done_reason": "stop",
            "total_duration": 5043500667,
            "load_duration": 5025959,
            "prompt_eval_count": 26,
            "prompt_eval_duration": 325953000,
            "eval_count": 5,
            "eval_duration": 4709213000
        }"#;

        fn model() -> CompletionModel<RecordingHttpClient> {
            let client = Client::builder()
                .api_key("test-key")
                .http_client(RecordingHttpClient::new(BODY))
                .build()
                .expect("build client");
            client.completion_model(LLAMA3_2)
        }

        /// The load-bearing capture property: `raw` is Ollama's
        /// `CompletionResponse` as rig parsed it — it deserializes back into
        /// that type and re-serializes to the identical value — and
        /// re-normalizing that capture through the same `TryFrom` reproduces
        /// every normalized field. Also reads `total_duration` and
        /// `eval_duration` off the capture, which the normalized response
        /// provably lacks.
        #[tokio::test]
        async fn completion_captures_raw_that_round_trips_into_the_wire_type() {
            let model = model();

            let response = model
                .completion(model.completion_request("hello").build())
                .await
                .expect("completion");

            let raw = &response.raw;
            let typed: CompletionResponse =
                serde_json::from_value(raw.clone()).expect("raw must deserialize");
            assert_eq!(
                serde_json::to_value(&typed).expect("re-serialize"),
                *raw,
                "the capture must be exactly what the wire type serializes to"
            );
            assert_eq!(typed.total_duration, Some(5_043_500_667));
            assert_eq!(typed.eval_duration, Some(4_709_213_000));
            assert_eq!(raw["total_duration"], 5_043_500_667_u64);
            assert_eq!(typed.done_reason.as_deref(), Some("stop"));

            let renormalized: completion::CompletionResponse =
                typed.try_into().expect("re-normalize the capture");
            assert_eq!(response.identity(), renormalized.identity());
            assert_eq!(response.finish_reason(), renormalized.finish_reason());
            assert_eq!(response.model, renormalized.model);
            assert_eq!(response.usage, renormalized.usage);
            assert_eq!(response.choice, renormalized.choice);
            assert_eq!(
                response.finish_reason(),
                Some(completion::FinishReason::Stop)
            );
            assert_eq!(response.model.as_deref(), Some("llama3.2"));
            assert_eq!(response.usage.total_tokens, 31);
        }
    }
}
