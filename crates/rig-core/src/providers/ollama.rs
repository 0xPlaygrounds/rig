//! Ollama API client and Rig integration
//!
//! # Example
//! ```ignore
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
//! println!("Embedding response: {embeddings:?}");
//! # Ok(())
//! # }
//! ```
use crate::client::{
    self, ApiKey, HasCompletion, HasEmbeddings, HasModelListing, ModelLister, ModelTransport,
    Nothing, Provider, ProviderClientResult,
};
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

    // Ollama needs no credential by default, so a builder without one is complete.
    fn absent() -> Option<Self> {
        Some(Self(None))
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

/// The Ollama provider.
#[derive(Debug, Default, Clone, Copy)]
pub struct Ollama;

pub type Client<H = crate::http_client::BoxedHttpClient> = client::Client<Ollama, H>;
pub type ClientBuilder<H = crate::markers::Missing> = client::ClientBuilder<Ollama, H>;

impl Provider for Ollama {
    const NAME: &'static str = PROVIDER_NAME;
    const BASE_URL: &'static str = OLLAMA_API_BASE_URL;
    const VERIFY_PATH: &'static str = "api/tags";
    type ApiKey = OllamaApiKey;
    type Config = ();
    type EnvInput = OllamaApiKey;

    fn build(_: (), _: &OllamaApiKey) -> http_client::Result<Self> {
        Ok(Ollama)
    }

    /// Read `OLLAMA_API_BASE_URL` (optional) and `OLLAMA_API_KEY` (optional).
    fn from_env<H: HttpClientExt>(http: H) -> ProviderClientResult<Client<H>> {
        let api_base = crate::client::optional_env_var("OLLAMA_API_BASE_URL")?
            .unwrap_or_else(|| OLLAMA_API_BASE_URL.to_string());

        let api_key = crate::client::optional_env_var("OLLAMA_API_KEY")?
            .map(OllamaApiKey::from)
            .unwrap_or_default();

        Client::builder()
            .api_key(api_key)
            .base_url(&api_base)
            .http_client(http)
            .build()
    }

    fn from_val<H: HttpClientExt>(
        api_key: OllamaApiKey,
        http: H,
    ) -> ProviderClientResult<Client<H>> {
        Client::new_with(api_key, http)
    }
}

impl HasCompletion for Ollama {
    type Model<H>
        = CompletionModel<H>
    where
        H: ModelTransport;

    fn completion_model<H: ModelTransport>(client: &Client<H>, model: String) -> Self::Model<H> {
        CompletionModel::new(client.clone(), model)
    }
}

impl HasEmbeddings for Ollama {
    type Model<H>
        = EmbeddingModel<H>
    where
        H: ModelTransport;

    fn embedding_model<H: ModelTransport>(
        client: &Client<H>,
        model: String,
        ndims: Option<usize>,
    ) -> Self::Model<H> {
        EmbeddingModel::make(client, model, ndims)
    }
}

impl HasModelListing for Ollama {
    type Lister<H>
        = OllamaModelLister<H>
    where
        H: ModelTransport;

    fn model_lister<H: ModelTransport>(client: &Client<H>) -> Self::Lister<H> {
        OllamaModelLister::new(client.clone())
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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

impl embeddings::NormalizeEmbeddingResponse for EmbeddingResponse {
    fn normalize(
        self,
        provider: &str,
        documents: Vec<String>,
    ) -> Result<embeddings::EmbeddingResponse, EmbeddingError> {
        if self.embeddings.len() != documents.len() {
            return Err(EmbeddingError::ResponseError(
                "Number of returned embeddings does not match input".into(),
            ));
        }
        let usage = crate::completion::Usage {
            input_tokens: self.prompt_eval_count.unwrap_or(0),
            total_tokens: self.prompt_eval_count.unwrap_or(0),
            ..crate::completion::Usage::new()
        };
        let embeddings = self
            .embeddings
            .into_iter()
            .zip(documents)
            .map(|(vec, document)| embeddings::Embedding { document, vec })
            .collect();
        Ok(embeddings::EmbeddingResponse::new(embeddings, provider)
            .with_model(self.model)
            .with_usage(usage))
    }
}

// ---------- Embedding Model ----------

#[derive(Clone)]
pub struct EmbeddingModel<T = crate::http_client::BoxedHttpClient> {
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

impl<T> EmbeddingModel<T>
where
    T: HttpClientExt + Clone + 'static,
{
    /// Perform the request and return Ollama's native `/api/embed` response
    /// instead of the normalized [`embeddings::EmbeddingResponse`]. Same
    /// request, transport, parser, and error path as
    /// [`embeddings::EmbeddingModel::embed_texts_response`].
    pub async fn raw_embed_texts(
        &self,
        documents: impl IntoIterator<Item = String>,
    ) -> Result<EmbeddingResponse, EmbeddingError> {
        let docs: Vec<String> = documents.into_iter().collect();
        self.raw_embed_texts_slice(&docs).await
    }

    /// Borrow-shaped twin of [`Self::raw_embed_texts`]: the batch is only
    /// serialized into the request body, so callers that keep their documents
    /// (the normalize path) can lend them instead of cloning the batch.
    async fn raw_embed_texts_slice(
        &self,
        docs: &[String],
    ) -> Result<EmbeddingResponse, EmbeddingError> {
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
        Ok(api_resp)
    }
}

impl<T> embeddings::EmbeddingModel for EmbeddingModel<T>
where
    T: HttpClientExt + Clone + 'static,
{
    fn max_documents(&self) -> usize {
        1024
    }
    fn ndims(&self) -> usize {
        self.ndims
    }

    async fn embed_texts_response(
        &self,
        documents: impl IntoIterator<Item = String>,
    ) -> Result<embeddings::EmbeddingResponse, EmbeddingError> {
        crate::telemetry::instrument_modality(
            PROVIDER_NAME,
            &self.model,
            crate::telemetry::ModalityOperation::Embeddings,
            async {
                use embeddings::NormalizeEmbeddingResponse as _;

                let docs: Vec<String> = documents.into_iter().collect();
                // Ollama reports no transport request-id header.
                let response = self.raw_embed_texts_slice(&docs).await?;
                let captured = serde_json::to_value(&response)?;
                Ok(response.normalize(PROVIDER_NAME, docs)?.with_raw(captured))
            },
        )
        .await
    }
}

impl<T> EmbeddingModel<T>
where
    T: HttpClientExt + Clone + 'static,
{
    /// Build the model, defaulting `ndims` from the model identifier when the
    /// caller gave none — the body behind `EmbeddingsClient::embedding_model`.
    pub fn make(client: &Client<T>, model: String, dims: Option<usize>) -> Self {
        let dims = dims
            .or(model_dimensions_from_identifier(&model))
            .unwrap_or_default();
        Self::new(client.clone(), model, dims)
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
    fn response_id(&self) -> Option<&str> {
        None
    }

    fn response_model_name(&self) -> Option<&str> {
        Some(self.model.as_str())
    }

    fn text_response(&self) -> Option<String> {
        match &self.message {
            Message::Assistant { content, .. } if !content.is_empty() => Some(content.clone()),
            _ => None,
        }
    }

    fn usage(&self) -> Option<Self::Usage> {
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

        let mut full_history: Vec<Message> = Vec::new();
        full_history.extend(
            partial_history
                .into_iter()
                .map(message::Message::try_into)
                .collect::<Result<Vec<Vec<Message>>, _>>()?
                .into_iter()
                .flatten(),
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
            model,
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
pub struct CompletionModel<T = crate::http_client::BoxedHttpClient> {
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
    T: HttpClientExt + Clone + WasmCompatSend + 'static,
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
        let system_instructions = completion_request.system_instructions().map(str::to_owned);
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
        let system_instructions = request.system_instructions().map(str::to_owned);
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
    T: HttpClientExt + Clone + WasmCompatSend + 'static,
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
pub struct OllamaModelLister<H = crate::http_client::BoxedHttpClient> {
    client: Client<H>,
}

impl<H> ModelLister<H> for OllamaModelLister<H>
where
    H: HttpClientExt + WasmCompatSend + WasmCompatSync + 'static,
{
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

impl<H> OllamaModelLister<H>
where
    H: HttpClientExt + WasmCompatSend + WasmCompatSync + 'static + Clone,
{
    /// Build the lister over `client`.
    pub fn new(client: Client<H>) -> Self {
        Self { client }
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
                            text_content.push(text.text);
                        }
                        crate::message::AssistantContent::ToolCall(tool_call) => {
                            tool_calls.push(tool_call);
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
                        .map(std::convert::Into::into)
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
mod tests;
