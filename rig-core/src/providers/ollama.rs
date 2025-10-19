//! Ollama API client and Rig integration
//!
//! # Example
//! ```rust
//! use rig::providers::ollama;
//!
//! // Create a new Ollama client (defaults to http://localhost:11434)
//! let client = ollama::Client::new();
//!
//! // Create a completion model interface using, for example, the "llama3.2" model
//! let comp_model = client.completion_model("llama3.2");
//!
//! let req = rig::completion::CompletionRequest {
//!     preamble: Some("You are now a humorous AI assistant.".to_owned()),
//!     chat_history: vec![],  // internal messages (if any)
//!     prompt: rig::message::Message::User {
//!         content: rig::one_or_many::OneOrMany::one(rig::message::UserContent::text("Please tell me why the sky is blue.")),
//!         name: None
//!     },
//!     temperature: 0.7,
//!     additional_params: None,
//!     tools: vec![],
//! };
//!
//! let response = comp_model.completion(req).await.unwrap();
//! println!("Ollama completion response: {:?}", response.choice);
//!
//! // Create an embedding interface using the "all-minilm" model
//! let emb_model = ollama::Client::new().embedding_model("all-minilm");
//! let docs = vec![
//!     "Why is the sky blue?".to_owned(),
//!     "Why is the grass green?".to_owned()
//! ];
//! let embeddings = emb_model.embed_texts(docs).await.unwrap();
//! println!("Embedding response: {:?}", embeddings);
//!
//! // Also create an agent and extractor if needed
//! let agent = client.agent("llama3.2");
//! let extractor = client.extractor::<serde_json::Value>("llama3.2");
//! ```
use crate::client::{
    CompletionClient, EmbeddingsClient, ProviderClient, VerifyClient, VerifyError,
};
use crate::completion::{GetTokenUsage, Usage};
use crate::http_client::{self, HttpClientExt};
use crate::json_utils::merge_inplace;
use crate::message::DocumentSourceKind;
use crate::streaming::RawStreamingChoice;
use crate::{
    Embed, OneOrMany,
    completion::{self, CompletionError, CompletionRequest},
    embeddings::{self, EmbeddingError, EmbeddingsBuilder},
    impl_conversion_traits, json_utils, message,
    message::{ImageDetail, Text},
    streaming,
};
use async_stream::try_stream;
use futures::StreamExt;
use reqwest;
// use reqwest_eventsource::{Event, RequestBuilderExt}; // (Not used currently as Ollama does not support SSE)
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{convert::TryFrom, str::FromStr};
use tracing::info_span;
use tracing_futures::Instrument;
// ---------- Main Client ----------

const OLLAMA_API_BASE_URL: &str = "http://localhost:11434";

pub struct ClientBuilder<'a, T = reqwest::Client> {
    base_url: &'a str,
    http_client: T,
}

impl<'a, T> ClientBuilder<'a, T>
where
    T: Default,
{
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            base_url: OLLAMA_API_BASE_URL,
            http_client: Default::default(),
        }
    }
}

impl<'a, T> ClientBuilder<'a, T> {
    pub fn base_url(mut self, base_url: &'a str) -> Self {
        self.base_url = base_url;
        self
    }

    pub fn with_client<U>(self, http_client: U) -> ClientBuilder<'a, U> {
        ClientBuilder {
            base_url: self.base_url,
            http_client,
        }
    }

    pub fn build(self) -> Client<T> {
        Client {
            base_url: self.base_url.into(),
            http_client: self.http_client,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Client<T = reqwest::Client> {
    base_url: String,
    http_client: T,
}

impl<T> Default for Client<T>
where
    T: Default,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Client<T>
where
    T: Default,
{
    /// Create a new Ollama client builder.
    ///
    /// # Example
    /// ```
    /// use rig::providers::ollama::{ClientBuilder, self};
    ///
    /// // Initialize the Ollama client
    /// let client = Client::builder()
    ///    .build()
    /// ```
    pub fn builder<'a>() -> ClientBuilder<'a, T> {
        ClientBuilder::new()
    }

    /// Create a new Ollama client. For more control, use the `builder` method.
    ///
    /// # Panics
    /// - If the reqwest client cannot be built (if the TLS backend cannot be initialized).
    pub fn new() -> Self {
        Self::builder().build()
    }
}

impl<T> Client<T> {
    fn req(&self, method: http_client::Method, path: &str) -> http_client::Builder {
        let url = format!("{}/{}", self.base_url, path.trim_start_matches('/'));
        http_client::Builder::new().method(method).uri(url)
    }

    pub(crate) fn post(&self, path: &str) -> http_client::Builder {
        self.req(http_client::Method::POST, path)
    }

    pub(crate) fn get(&self, path: &str) -> http_client::Builder {
        self.req(http_client::Method::GET, path)
    }
}

impl Client<reqwest::Client> {
    fn reqwest_post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/{}", self.base_url, path.trim_start_matches('/'));
        self.http_client.post(url)
    }
}

impl ProviderClient for Client<reqwest::Client> {
    fn from_env() -> Self {
        let api_base = std::env::var("OLLAMA_API_BASE_URL").expect("OLLAMA_API_BASE_URL not set");
        Self::builder().base_url(&api_base).build()
    }

    fn from_val(input: crate::client::ProviderValue) -> Self {
        let crate::client::ProviderValue::Simple(_) = input else {
            panic!("Incorrect provider value type")
        };

        Self::new()
    }
}

impl CompletionClient for Client<reqwest::Client> {
    type CompletionModel = CompletionModel<reqwest::Client>;

    fn completion_model(&self, model: &str) -> CompletionModel<reqwest::Client> {
        CompletionModel::new(self.clone(), model)
    }
}

impl EmbeddingsClient for Client<reqwest::Client> {
    type EmbeddingModel = EmbeddingModel<reqwest::Client>;
    fn embedding_model(&self, model: &str) -> EmbeddingModel<reqwest::Client> {
        EmbeddingModel::new(self.clone(), model, 0)
    }
    fn embedding_model_with_ndims(
        &self,
        model: &str,
        ndims: usize,
    ) -> EmbeddingModel<reqwest::Client> {
        EmbeddingModel::new(self.clone(), model, ndims)
    }
    fn embeddings<D: Embed>(&self, model: &str) -> EmbeddingsBuilder<Self::EmbeddingModel, D> {
        EmbeddingsBuilder::new(self.embedding_model(model))
    }
}

impl VerifyClient for Client<reqwest::Client> {
    #[cfg_attr(feature = "worker", worker::send)]
    async fn verify(&self) -> Result<(), VerifyError> {
        let req = self
            .get("api/tags")
            .body(http_client::NoBody)
            .map_err(http_client::Error::from)?;

        let response = HttpClientExt::send(&self.http_client, req).await?;

        match response.status() {
            reqwest::StatusCode::OK => Ok(()),
            reqwest::StatusCode::UNAUTHORIZED => Err(VerifyError::InvalidAuthentication),
            reqwest::StatusCode::INTERNAL_SERVER_ERROR
            | reqwest::StatusCode::SERVICE_UNAVAILABLE
            | reqwest::StatusCode::BAD_GATEWAY => {
                let text = http_client::text(response).await?;
                Err(VerifyError::ProviderError(text))
            }
            _ => {
                //response.error_for_status()?;
                Ok(())
            }
        }
    }
}

impl_conversion_traits!(
    AsTranscription,
    AsImageGeneration,
    AsAudioGeneration for Client<T>
);

// ---------- API Error and Response Structures ----------

#[derive(Debug, Deserialize)]
struct ApiErrorResponse {
    message: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
}

// ---------- Embedding API ----------

pub const ALL_MINILM: &str = "all-minilm";
pub const NOMIC_EMBED_TEXT: &str = "nomic-embed-text";

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

impl From<ApiErrorResponse> for EmbeddingError {
    fn from(err: ApiErrorResponse) -> Self {
        EmbeddingError::ProviderError(err.message)
    }
}

impl From<ApiResponse<EmbeddingResponse>> for Result<EmbeddingResponse, EmbeddingError> {
    fn from(value: ApiResponse<EmbeddingResponse>) -> Self {
        match value {
            ApiResponse::Ok(response) => Ok(response),
            ApiResponse::Err(err) => Err(EmbeddingError::ProviderError(err.message)),
        }
    }
}

// ---------- Embedding Model ----------

#[derive(Clone)]
pub struct EmbeddingModel<T> {
    client: Client<T>,
    pub model: String,
    ndims: usize,
}

impl<T> EmbeddingModel<T> {
    pub fn new(client: Client<T>, model: &str, ndims: usize) -> Self {
        Self {
            client,
            model: model.to_owned(),
            ndims,
        }
    }
}

impl embeddings::EmbeddingModel for EmbeddingModel<reqwest::Client> {
    const MAX_DOCUMENTS: usize = 1024;
    fn ndims(&self) -> usize {
        self.ndims
    }
    #[cfg_attr(feature = "worker", worker::send)]
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
            .post("api/embed")
            .header("Content-Type", "application/json")
            .body(body)
            .map_err(|e| EmbeddingError::HttpError(e.into()))?;

        let response = HttpClientExt::send(&self.client.http_client, req).await?;

        if !response.status().is_success() {
            let text = http_client::text(response).await?;
            return Err(EmbeddingError::ProviderError(text));
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
                // Add the assistant's text content if any.
                if !content.is_empty() {
                    assistant_contents.push(completion::AssistantContent::text(&content));
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
                    },
                    raw_response,
                })
            }
            _ => Err(CompletionError::ResponseError(
                "Chat response does not include an assistant message".into(),
            )),
        }
    }
}

// ---------- Completion Model ----------

#[derive(Clone)]
pub struct CompletionModel<T> {
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

    fn create_completion_request(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<Value, CompletionError> {
        if completion_request.tool_choice.is_some() {
            tracing::warn!("WARNING: `tool_choice` not supported for Ollama");
        }

        // Build up the order of messages (context, chat_history)
        let mut partial_history = vec![];
        if let Some(docs) = completion_request.normalized_documents() {
            partial_history.push(docs);
        }
        partial_history.extend(completion_request.chat_history);

        // Initialize full history with preamble (or empty if non-existent)
        let mut full_history: Vec<Message> = completion_request
            .preamble
            .map_or_else(Vec::new, |preamble| vec![Message::system(&preamble)]);

        // Convert and extend the rest of the history
        full_history.extend(
            partial_history
                .into_iter()
                .map(|msg| msg.try_into())
                .collect::<Result<Vec<Vec<Message>>, _>>()?
                .into_iter()
                .flatten()
                .collect::<Vec<Message>>(),
        );

        // Convert internal prompt into a provider Message
        let options = if let Some(extra) = completion_request.additional_params {
            json_utils::merge(
                json!({ "temperature": completion_request.temperature }),
                extra,
            )
        } else {
            json!({ "temperature": completion_request.temperature })
        };

        let mut request_payload = json!({
            "model": self.model,
            "messages": full_history,
            "options": options,
            "stream": false,
        });
        if !completion_request.tools.is_empty() {
            request_payload["tools"] = json!(
                completion_request
                    .tools
                    .into_iter()
                    .map(|tool| tool.into())
                    .collect::<Vec<ToolDefinition>>()
            );
        }

        tracing::debug!(target: "rig", "Chat mode payload: {}", request_payload);

        Ok(request_payload)
    }
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
    fn token_usage(&self) -> Option<crate::completion::Usage> {
        let mut usage = crate::completion::Usage::new();
        let input_tokens = self.prompt_eval_count.unwrap_or_default();
        let output_tokens = self.eval_count.unwrap_or_default();
        usage.input_tokens = input_tokens;
        usage.output_tokens = output_tokens;
        usage.total_tokens = input_tokens + output_tokens;

        Some(usage)
    }
}

impl completion::CompletionModel for CompletionModel<reqwest::Client> {
    type Response = CompletionResponse;
    type StreamingResponse = StreamingCompletionResponse;

    #[cfg_attr(feature = "worker", worker::send)]
    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<Self::Response>, CompletionError> {
        let preamble = completion_request.preamble.clone();
        let request = self.create_completion_request(completion_request)?;

        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat",
                gen_ai.operation.name = "chat",
                gen_ai.provider.name = "ollama",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = preamble,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.input.messages = serde_json::to_string(&request.get("messages").unwrap()).unwrap(),
                gen_ai.output.messages = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        let async_block = async move {
            let response = self
                .client
                .reqwest_post("api/chat")
                .json(&request)
                .send()
                .await
                .map_err(|e| http_client::Error::Instance(e.into()))?;

            if !response.status().is_success() {
                return Err(CompletionError::ProviderError(
                    response
                        .text()
                        .await
                        .map_err(|e| http_client::Error::Instance(e.into()))?,
                ));
            }

            let bytes = response
                .bytes()
                .await
                .map_err(|e| http_client::Error::Instance(e.into()))?;

            tracing::debug!(target: "rig", "Received response from Ollama: {}", String::from_utf8_lossy(&bytes));

            let response: CompletionResponse = serde_json::from_slice(&bytes)?;
            let span = tracing::Span::current();
            span.record("gen_ai.response.model_name", &response.model);
            span.record(
                "gen_ai.output.messages",
                serde_json::to_string(&vec![&response.message]).unwrap(),
            );
            span.record(
                "gen_ai.usage.input_tokens",
                response.prompt_eval_count.unwrap_or_default(),
            );
            span.record(
                "gen_ai.usage.output_tokens",
                response.eval_count.unwrap_or_default(),
            );

            let response: completion::CompletionResponse<CompletionResponse> =
                response.try_into()?;

            Ok(response)
        };

        tracing::Instrument::instrument(async_block, span).await
    }

    #[cfg_attr(feature = "worker", worker::send)]
    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<streaming::StreamingCompletionResponse<Self::StreamingResponse>, CompletionError>
    {
        let preamble = request.preamble.clone();
        let mut request = self.create_completion_request(request)?;
        merge_inplace(&mut request, json!({"stream": true}));

        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat_streaming",
                gen_ai.operation.name = "chat_streaming",
                gen_ai.provider.name = "ollama",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = preamble,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = self.model,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.input.messages = serde_json::to_string(&request.get("messages").unwrap()).unwrap(),
                gen_ai.output.messages = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        let response = self
            .client
            .reqwest_post("api/chat")
            .json(&request)
            .send()
            .await
            .map_err(|e| http_client::Error::Instance(e.into()))?;

        if !response.status().is_success() {
            return Err(CompletionError::ProviderError(
                response
                    .text()
                    .await
                    .map_err(|e| http_client::Error::Instance(e.into()))?,
            ));
        }

        let stream = try_stream! {
            let span = tracing::Span::current();
            let mut byte_stream = response.bytes_stream();
            let mut tool_calls_final = Vec::new();
            let mut text_response = String::new();
            let mut thinking_response = String::new();

            while let Some(chunk) = byte_stream.next().await {
                let bytes = chunk.map_err(|e| http_client::Error::Instance(e.into()))?;

                for line in bytes.split(|&b| b == b'\n') {
                    if line.is_empty() {
                        continue;
                    }

                    tracing::debug!(target: "rig", "Received NDJSON line from Ollama: {}", String::from_utf8_lossy(line));

                    let response: CompletionResponse = serde_json::from_slice(line)?;

                    if response.done {
                        span.record("gen_ai.usage.input_tokens", response.prompt_eval_count);
                        span.record("gen_ai.usage.output_tokens", response.eval_count);
                        let message = Message::Assistant {
                            content: text_response.clone(),
                            thinking: if thinking_response.is_empty() { None } else { Some(thinking_response.clone()) },
                            images: None,
                            name: None,
                            tool_calls: tool_calls_final.clone()
                        };
                        span.record("gen_ai.output.messages", serde_json::to_string(&vec![message]).unwrap());
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

                    if let Message::Assistant { content, thinking, tool_calls, .. } = response.message {
                        if let Some(thinking_content) = thinking
                            && !thinking_content.is_empty() {
                            thinking_response += &thinking_content;
                            yield RawStreamingChoice::Reasoning {
                                reasoning: thinking_content,
                                id: None,
                                signature: None,
                            };
                        }

                        if !content.is_empty() {
                            text_response += &content;
                            yield RawStreamingChoice::Message(content);
                        }

                        for tool_call in tool_calls {
                            tool_calls_final.push(tool_call.clone());
                            yield RawStreamingChoice::ToolCall {
                                id: String::new(),
                                name: tool_call.function.name,
                                arguments: tool_call.function.arguments,
                                call_id: None,
                            };
                        }
                    }
                }
            }
        }.instrument(span);

        Ok(streaming::StreamingCompletionResponse::stream(Box::pin(
            stream,
        )))
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
/// Conversion from an internal Rig message (crate::message::Message) to a provider Message.
/// (Only User and Assistant variants are supported.)
impl TryFrom<crate::message::Message> for Vec<Message> {
    type Error = crate::message::MessageError;
    fn try_from(internal_msg: crate::message::Message) -> Result<Self, Self::Error> {
        use crate::message::Message as InternalMessage;
        match internal_msg {
            InternalMessage::User { content, .. } => {
                let (tool_results, other_content): (Vec<_>, Vec<_>) =
                    content.into_iter().partition(|content| {
                        matches!(content, crate::message::UserContent::ToolResult(_))
                    });

                if !tool_results.is_empty() {
                    tool_results
                        .into_iter()
                        .map(|content| match content {
                            crate::message::UserContent::ToolResult(
                                crate::message::ToolResult { id, content, .. },
                            ) => {
                                // Ollama expects a single string for tool results, so we concatenate
                                let content_string = content
                                    .into_iter()
                                    .map(|content| match content {
                                        crate::message::ToolResultContent::Text(text) => text.text,
                                        _ => "[Non-text content]".to_string(),
                                    })
                                    .collect::<Vec<_>>()
                                    .join("\n");

                                Ok::<_, crate::message::MessageError>(Message::ToolResult {
                                    name: id,
                                    content: content_string,
                                })
                            }
                            _ => unreachable!(),
                        })
                        .collect::<Result<Vec<_>, _>>()
                } else {
                    // Ollama requires separate text content and images array
                    let (texts, images) = other_content.into_iter().fold(
                        (Vec::new(), Vec::new()),
                        |(mut texts, mut images), content| {
                            match content {
                                crate::message::UserContent::Text(crate::message::Text {
                                    text,
                                }) => texts.push(text),
                                crate::message::UserContent::Image(crate::message::Image {
                                    data: DocumentSourceKind::Base64(data),
                                    ..
                                }) => images.push(data),
                                crate::message::UserContent::Document(
                                    crate::message::Document {
                                        data:
                                            DocumentSourceKind::Base64(data)
                                            | DocumentSourceKind::String(data),
                                        ..
                                    },
                                ) => texts.push(data),
                                _ => {} // Audio not supported by Ollama
                            }
                            (texts, images)
                        },
                    );

                    Ok(vec![Message::User {
                        content: texts.join(" "),
                        images: if images.is_empty() {
                            None
                        } else {
                            Some(
                                images
                                    .into_iter()
                                    .map(|x| x.to_string())
                                    .collect::<Vec<String>>(),
                            )
                        },
                        name: None,
                    }])
                }
            }
            InternalMessage::Assistant { content, .. } => {
                let mut thinking: Option<String> = None;
                let (text_content, tool_calls) = content.into_iter().fold(
                    (Vec::new(), Vec::new()),
                    |(mut texts, mut tools), content| {
                        match content {
                            crate::message::AssistantContent::Text(text) => texts.push(text.text),
                            crate::message::AssistantContent::ToolCall(tool_call) => {
                                tools.push(tool_call)
                            }
                            crate::message::AssistantContent::Reasoning(
                                crate::message::Reasoning { reasoning, .. },
                            ) => {
                                thinking =
                                    Some(reasoning.first().cloned().unwrap_or(String::new()));
                            }
                        }
                        (texts, tools)
                    },
                );

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
                content: OneOrMany::one(crate::completion::message::UserContent::Text(Text {
                    text: content,
                })),
            },
            Message::Assistant {
                content,
                tool_calls,
                ..
            } => {
                let mut assistant_contents =
                    vec![crate::completion::message::AssistantContent::Text(Text {
                        text: content,
                    })];
                for tc in tool_calls {
                    assistant_contents.push(
                        crate::completion::message::AssistantContent::tool_call(
                            tc.function.name.clone(),
                            tc.function.name,
                            tc.function.arguments,
                        ),
                    );
                }
                crate::completion::Message::Assistant {
                    id: None,
                    content: OneOrMany::many(assistant_contents).unwrap(),
                }
            }
            // System and ToolResult are converted to User message as needed.
            Message::System { content, .. } => crate::completion::Message::User {
                content: OneOrMany::one(crate::completion::message::UserContent::Text(Text {
                    text: content,
                })),
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
        let reasoning_content = crate::message::Reasoning {
            id: None,
            reasoning: vec!["Step 1: Consider the problem".to_string()],
            signature: None,
        };

        let internal_msg = crate::message::Message::Assistant {
            id: None,
            content: crate::OneOrMany::many(vec![
                crate::message::AssistantContent::Reasoning(reasoning_content),
                crate::message::AssistantContent::Text(crate::message::Text {
                    text: "The answer is X".to_string(),
                }),
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
}
