//! Llamafile API client and Rig integration
//!
//! [Llamafile](https://github.com/Mozilla-Ocho/llamafile) is a Mozilla Builders project
//! that distributes LLMs as single-file executables. When started, it exposes an
//! OpenAI-compatible API at `http://localhost:8080/v1`.
//!
//! # Example
//! ```rust,ignore
//! use rig::providers::llamafile;
//! use rig::completion::Prompt;
//!
//! // Create a new Llamafile client (defaults to http://localhost:8080)
//! let client = llamafile::Client::from_url("http://localhost:8080");
//!
//! // Create an agent with a preamble
//! let agent = client
//!     .agent(llamafile::LLAMA_CPP)
//!     .preamble("You are a helpful assistant.")
//!     .build();
//!
//! // Prompt the agent and print the response
//! let response = agent.prompt("Hello!").await?;
//! println!("{response}");
//! ```

use crate::client::{
    self, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder, ProviderClient,
};
use crate::completion::GetTokenUsage;
use crate::http_client::sse::{Event, GenericEventSource};
use crate::http_client::{self, HttpClientExt};
use crate::json_utils::empty_or_none;
use crate::providers::openai::{self, StreamingToolCall};
use crate::{
    completion::{self, CompletionError, CompletionRequest},
    embeddings::{self, EmbeddingError},
    json_utils,
};
use async_stream::stream;
use bytes::Bytes;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{Level, info_span};
use tracing_futures::Instrument;

// ================================================================
// Main Llamafile Client
// ================================================================
const LLAMAFILE_API_BASE_URL: &str = "http://localhost:8080";

/// The default model identifier reported by llamafile.
pub const LLAMA_CPP: &str = "LLaMA_CPP";

#[derive(Debug, Default, Clone, Copy)]
pub struct LlamafileExt;

#[derive(Debug, Default, Clone, Copy)]
pub struct LlamafileBuilder;

impl Provider for LlamafileExt {
    type Builder = LlamafileBuilder;
    const VERIFY_PATH: &'static str = "v1/models";
}

impl<H> Capabilities<H> for LlamafileExt {
    type Completion = Capable<CompletionModel<H>>;
    type Embeddings = Capable<EmbeddingModel<H>>;
    type Transcription = Nothing;
    type ModelListing = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;
    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
}

impl DebugExt for LlamafileExt {}

impl ProviderBuilder for LlamafileBuilder {
    type Extension<H>
        = LlamafileExt
    where
        H: HttpClientExt;
    type ApiKey = Nothing;

    const BASE_URL: &'static str = LLAMAFILE_API_BASE_URL;

    fn build<H>(
        _builder: &client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: HttpClientExt,
    {
        Ok(LlamafileExt)
    }
}

pub type Client<H = reqwest::Client> = client::Client<LlamafileExt, H>;
pub type ClientBuilder<H = reqwest::Client> = client::ClientBuilder<LlamafileBuilder, Nothing, H>;

impl Client {
    /// Create a client pointing at the given llamafile base URL
    /// (e.g. `http://localhost:8080`).
    pub fn from_url(base_url: &str) -> Self {
        Self::builder()
            .api_key(Nothing)
            .base_url(base_url)
            .build()
            .expect("Failed to build llamafile client")
    }
}

impl ProviderClient for Client {
    type Input = Nothing;

    fn from_env() -> Self {
        let api_base =
            std::env::var("LLAMAFILE_API_BASE_URL").expect("LLAMAFILE_API_BASE_URL not set");
        Self::from_url(&api_base)
    }

    fn from_val(_: Self::Input) -> Self {
        Self::builder().api_key(Nothing).build().unwrap()
    }
}

// ================================================================
// API Error Handling
// ================================================================

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

// ================================================================
// Completion Request
// ================================================================

/// Llamafile uses the OpenAI chat completions format.
/// We reuse the OpenAI `Message` type for maximum compatibility.
#[derive(Debug, Serialize, Deserialize)]
struct LlamafileCompletionRequest {
    model: String,
    messages: Vec<openai::Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<openai::ToolDefinition>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    additional_params: Option<serde_json::Value>,
}

impl TryFrom<(&str, CompletionRequest)> for LlamafileCompletionRequest {
    type Error = CompletionError;

    fn try_from((model, req): (&str, CompletionRequest)) -> Result<Self, Self::Error> {
        if req.output_schema.is_some() {
            tracing::warn!("Structured outputs may not be supported by llamafile");
        }
        let model = req.model.clone().unwrap_or_else(|| model.to_string());

        // Build message history: preamble -> documents -> chat history
        let mut full_history: Vec<openai::Message> = match &req.preamble {
            Some(preamble) => vec![openai::Message::system(preamble)],
            None => vec![],
        };

        if let Some(docs) = req.normalized_documents() {
            let docs: Vec<openai::Message> = docs.try_into()?;
            full_history.extend(docs);
        }

        let chat_history: Vec<openai::Message> = req
            .chat_history
            .clone()
            .into_iter()
            .map(|msg| msg.try_into())
            .collect::<Result<Vec<Vec<openai::Message>>, _>>()?
            .into_iter()
            .flatten()
            .collect();

        full_history.extend(chat_history);

        Ok(Self {
            model,
            messages: full_history,
            temperature: req.temperature,
            max_tokens: req.max_tokens,
            tools: req
                .tools
                .into_iter()
                .map(openai::ToolDefinition::from)
                .collect(),
            additional_params: req.additional_params,
        })
    }
}

// ================================================================
// Completion Model
// ================================================================

/// Llamafile completion model.
#[derive(Clone)]
pub struct CompletionModel<T = reqwest::Client> {
    client: Client<T>,
    /// The model identifier (usually `LLaMA_CPP`).
    pub model: String,
}

impl<T> CompletionModel<T> {
    /// Create a new completion model for the given client and model name.
    pub fn new(client: Client<T>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }
}

impl<T> completion::CompletionModel for CompletionModel<T>
where
    T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
{
    type Response = openai::CompletionResponse;
    type StreamingResponse = StreamingCompletionResponse;
    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model)
    }

    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<openai::CompletionResponse>, CompletionError> {
        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat",
                gen_ai.operation.name = "chat",
                gen_ai.provider.name = "llamafile",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = completion_request.preamble,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        let request =
            LlamafileCompletionRequest::try_from((self.model.as_ref(), completion_request))?;

        if tracing::enabled!(Level::TRACE) {
            tracing::trace!(target: "rig::completions",
                "Llamafile completion request: {}",
                serde_json::to_string_pretty(&request)?
            );
        }

        let body = serde_json::to_vec(&request)?;
        let req = self
            .client
            .post("v1/chat/completions")?
            .body(body)
            .map_err(|e| CompletionError::HttpError(e.into()))?;

        async move {
            let response = self.client.send::<_, Bytes>(req).await?;
            let status = response.status();
            let response_body = response.into_body().into_future().await?.to_vec();

            if status.is_success() {
                match serde_json::from_slice::<ApiResponse<openai::CompletionResponse>>(
                    &response_body,
                )? {
                    ApiResponse::Ok(response) => {
                        let span = tracing::Span::current();
                        span.record("gen_ai.response.id", response.id.clone());
                        span.record("gen_ai.response.model_name", response.model.clone());
                        if let Some(ref usage) = response.usage {
                            span.record("gen_ai.usage.input_tokens", usage.prompt_tokens);
                            span.record(
                                "gen_ai.usage.output_tokens",
                                usage.total_tokens - usage.prompt_tokens,
                            );
                        }

                        if tracing::enabled!(Level::TRACE) {
                            tracing::trace!(target: "rig::completions",
                                "Llamafile completion response: {}",
                                serde_json::to_string_pretty(&response)?
                            );
                        }

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
        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat_streaming",
                gen_ai.operation.name = "chat_streaming",
                gen_ai.provider.name = "llamafile",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = completion_request.preamble,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        let mut request =
            LlamafileCompletionRequest::try_from((self.model.as_ref(), completion_request))?;

        let params = json_utils::merge(
            request.additional_params.unwrap_or(serde_json::json!({})),
            serde_json::json!({"stream": true}),
        );
        request.additional_params = Some(params);

        if tracing::enabled!(Level::TRACE) {
            tracing::trace!(target: "rig::completions",
                "Llamafile streaming completion request: {}",
                serde_json::to_string_pretty(&request)?
            );
        }

        let body = serde_json::to_vec(&request)?;
        let req = self
            .client
            .post("v1/chat/completions")?
            .body(body)
            .map_err(|e| CompletionError::HttpError(e.into()))?;

        send_streaming_request(self.client.clone(), req, span).await
    }
}

// ================================================================
// Streaming Support
// ================================================================

#[derive(Deserialize, Debug)]
struct StreamingDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default, deserialize_with = "json_utils::null_or_vec")]
    tool_calls: Vec<StreamingToolCall>,
}

#[derive(Deserialize, Debug)]
struct StreamingChoice {
    delta: StreamingDelta,
}

#[derive(Deserialize, Debug)]
struct StreamingCompletionChunk {
    choices: Vec<StreamingChoice>,
    usage: Option<openai::Usage>,
}

/// Final streaming response containing usage information.
#[derive(Clone, Deserialize, Serialize, Debug)]
pub struct StreamingCompletionResponse {
    /// Token usage from the streaming response.
    pub usage: openai::Usage,
}

impl GetTokenUsage for StreamingCompletionResponse {
    fn token_usage(&self) -> Option<crate::completion::Usage> {
        let mut usage = crate::completion::Usage::new();
        usage.input_tokens = self.usage.prompt_tokens as u64;
        usage.total_tokens = self.usage.total_tokens as u64;
        usage.output_tokens = self.usage.total_tokens as u64 - self.usage.prompt_tokens as u64;
        Some(usage)
    }
}

async fn send_streaming_request<T>(
    client: T,
    req: http::Request<Vec<u8>>,
    span: tracing::Span,
) -> Result<
    crate::streaming::StreamingCompletionResponse<StreamingCompletionResponse>,
    CompletionError,
>
where
    T: HttpClientExt + Clone + 'static,
{
    let mut event_source = GenericEventSource::new(client, req);

    let stream = stream! {
        let span = tracing::Span::current();
        let mut final_usage = openai::Usage {
            prompt_tokens: 0,
            total_tokens: 0,
            prompt_tokens_details: None,
        };
        let mut text_response = String::new();
        let mut calls: HashMap<usize, (String, String, String)> = HashMap::new();

        while let Some(event_result) = event_source.next().await {
            match event_result {
                Ok(Event::Open) => {
                    tracing::trace!("SSE connection opened");
                    continue;
                }
                Ok(Event::Message(message)) => {
                    let data_str = message.data.trim();
                    if data_str.is_empty() || data_str == "[DONE]" {
                        continue;
                    }

                    let parsed = serde_json::from_str::<StreamingCompletionChunk>(data_str);
                    let Ok(data) = parsed else {
                        let err = parsed.unwrap_err();
                        tracing::debug!("Couldn't parse SSE payload: {:?}", err);
                        continue;
                    };

                    if let Some(choice) = data.choices.first() {
                        let delta = &choice.delta;

                        // Handle tool calls
                        for tool_call in &delta.tool_calls {
                            let function = &tool_call.function;

                            // Start of tool call
                            if function.name.as_ref().map(|s| !s.is_empty()).unwrap_or(false)
                                && empty_or_none(&function.arguments)
                            {
                                let id = tool_call.id.clone().unwrap_or_default();
                                let name = function.name.clone().unwrap();
                                calls.insert(tool_call.index, (id, name, String::new()));
                            }
                            // Continuation
                            else if function.name.as_ref().map(|s| s.is_empty()).unwrap_or(true)
                                && let Some(arguments) = &function.arguments
                                && !arguments.is_empty()
                            {
                                if let Some((id, name, existing_args)) = calls.get(&tool_call.index) {
                                    let combined = format!("{}{}", existing_args, arguments);
                                    calls.insert(tool_call.index, (id.clone(), name.clone(), combined));
                                }
                            }
                            // Complete tool call in a single chunk
                            else {
                                let id = tool_call.id.clone().unwrap_or_default();
                                let name = function.name.clone().unwrap_or_default();
                                let arguments_str = function.arguments.clone().unwrap_or_default();

                                let Ok(arguments_json) = serde_json::from_str::<serde_json::Value>(&arguments_str) else {
                                    tracing::debug!("Couldn't parse tool call args '{}'", arguments_str);
                                    continue;
                                };

                                yield Ok(crate::streaming::RawStreamingChoice::ToolCall(
                                    crate::streaming::RawStreamingToolCall::new(id, name, arguments_json)
                                ));
                            }
                        }

                        // Streamed content
                        if let Some(content) = &delta.content {
                            text_response += content;
                            yield Ok(crate::streaming::RawStreamingChoice::Message(content.clone()));
                        }
                    }

                    if let Some(usage) = data.usage {
                        final_usage = usage;
                    }
                }
                Err(crate::http_client::Error::StreamEnded) => break,
                Err(err) => {
                    tracing::error!(?err, "SSE error");
                    yield Err(CompletionError::ResponseError(err.to_string()));
                    break;
                }
            }
        }

        event_source.close();

        // Flush accumulated tool calls
        for (_, (id, name, arguments)) in calls {
            let Ok(arguments_json) = serde_json::from_str::<serde_json::Value>(&arguments) else {
                continue;
            };
            yield Ok(crate::streaming::RawStreamingChoice::ToolCall(
                crate::streaming::RawStreamingToolCall::new(id, name, arguments_json)
            ));
        }

        span.record("gen_ai.usage.input_tokens", final_usage.prompt_tokens);
        span.record("gen_ai.usage.output_tokens", final_usage.total_tokens - final_usage.prompt_tokens);

        yield Ok(crate::streaming::RawStreamingChoice::FinalResponse(
            StreamingCompletionResponse { usage: final_usage }
        ));
    }.instrument(span);

    Ok(crate::streaming::StreamingCompletionResponse::stream(
        Box::pin(stream),
    ))
}

// ================================================================
// Embedding Model
// ================================================================

/// Llamafile embedding model.
///
/// Llamafile supports the OpenAI-compatible `/v1/embeddings` endpoint.
#[derive(Clone)]
pub struct EmbeddingModel<T = reqwest::Client> {
    client: Client<T>,
    /// The model identifier.
    pub model: String,
    ndims: usize,
}

impl<T> EmbeddingModel<T> {
    /// Create a new embedding model for the given client, model name, and dimensions.
    pub fn new(client: Client<T>, model: impl Into<String>, ndims: usize) -> Self {
        Self {
            client,
            model: model.into(),
            ndims,
        }
    }
}

impl<T> embeddings::EmbeddingModel for EmbeddingModel<T>
where
    T: HttpClientExt + Clone + std::fmt::Debug + Default + Send + 'static,
{
    const MAX_DOCUMENTS: usize = 1024;

    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>, ndims: Option<usize>) -> Self {
        Self::new(client.clone(), model, ndims.unwrap_or_default())
    }

    fn ndims(&self) -> usize {
        self.ndims
    }

    async fn embed_texts(
        &self,
        documents: impl IntoIterator<Item = String>,
    ) -> Result<Vec<embeddings::Embedding>, EmbeddingError> {
        let documents = documents.into_iter().collect::<Vec<_>>();

        let body = serde_json::json!({
            "model": self.model,
            "input": documents,
        });

        let body = serde_json::to_vec(&body)?;

        let req = self
            .client
            .post("v1/embeddings")?
            .body(body)
            .map_err(|e| EmbeddingError::HttpError(e.into()))?;

        let response = self.client.send(req).await?;

        if response.status().is_success() {
            let body: Vec<u8> = response.into_body().await?;
            let body: ApiResponse<openai::EmbeddingResponse> = serde_json::from_slice(&body)?;

            match body {
                ApiResponse::Ok(response) => {
                    tracing::info!(target: "rig",
                        "Llamafile embedding token usage: {:?}",
                        response.usage
                    );

                    if response.data.len() != documents.len() {
                        return Err(EmbeddingError::ResponseError(
                            "Response data length does not match input length".into(),
                        ));
                    }

                    Ok(response
                        .data
                        .into_iter()
                        .zip(documents.into_iter())
                        .map(|(embedding, document)| embeddings::Embedding {
                            document,
                            vec: embedding
                                .embedding
                                .into_iter()
                                .filter_map(|n| n.as_f64())
                                .collect(),
                        })
                        .collect())
                }
                ApiResponse::Err(err) => Err(EmbeddingError::ProviderError(err.message)),
            }
        } else {
            let text = http_client::text(response).await?;
            Err(EmbeddingError::ProviderError(text))
        }
    }
}

// ================================================================
// Tests
// ================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::Nothing;

    #[test]
    fn test_client_initialization() {
        let _client =
            crate::providers::llamafile::Client::new(Nothing).expect("Client::new() failed");
        let _client_from_builder = crate::providers::llamafile::Client::builder()
            .api_key(Nothing)
            .build()
            .expect("Client::builder() failed");
    }

    #[test]
    fn test_client_from_url() {
        let _client = crate::providers::llamafile::Client::from_url("http://localhost:8080");
    }

    #[test]
    fn test_completion_request_conversion() {
        use crate::OneOrMany;
        use crate::completion::Message as CompletionMessage;
        use crate::message::{Text, UserContent};

        let completion_request = CompletionRequest {
            model: None,
            preamble: Some("You are a helpful assistant.".to_string()),
            chat_history: OneOrMany::one(CompletionMessage::User {
                content: OneOrMany::one(UserContent::Text(Text {
                    text: "Hello!".to_string(),
                })),
            }),
            documents: vec![],
            tools: vec![],
            temperature: Some(0.7),
            max_tokens: Some(256),
            tool_choice: None,
            additional_params: None,
            output_schema: None,
        };

        let request = LlamafileCompletionRequest::try_from((LLAMA_CPP, completion_request))
            .expect("Failed to create request");

        assert_eq!(request.model, LLAMA_CPP);
        assert_eq!(request.messages.len(), 2); // system + user
        assert_eq!(request.temperature, Some(0.7));
        assert_eq!(request.max_tokens, Some(256));
    }
}
