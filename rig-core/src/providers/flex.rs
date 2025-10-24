//! A lightweight provider for OpenAI-like APIs that aren’t fully spec-compliant.
//!
//! Services like **Together AI**, **Chutes AI**, **vLLM**, **SGLang**, **mistral.rs**,
//! **candle-vllm**, and custom inference servers often mimic OpenAI’s API but add
//! custom fields, non-standard streaming, or extended behaviors that break strict
//! clients.
//!
//! The `Flex` provider handles these gracefully by letting you configure:
//! - **Base URL** (any endpoint),
//! - **API key** (per instance, no globals),
//! - **Model list** (enabling multi-model support and rotation across backends).
//!
//! It preserves OpenAI-style semantics while avoiding fragile workarounds in the core
//! client—ideal for dynamic, multi-provider LLM setups in Rust.
//!
//! Use `Flex` when your backend is *mostly* OpenAI-compatible but not perfect.
//! Use the standard `OpenAI` provider only for fully compliant APIs.
//!
//! # Example
//! ```no_run
//! use rig::providers::flex;
//!
//! // Configure using environment variables:
//! // FLEX_API_KEY=your_api_key
//! // FLEX_BASE_URL=https://your-api-endpoint.com/v1
//! // FLEX_MODELS=gpt-4o,gpt-4-turbo
//!
//! let client = flex::Client::from_env();
//! let models = flex::get_models_from_env();
//!
//! let model = client.completion_model("gpt-4o");
//! ```

use bytes::Bytes;
use http::{Method, Request};
use std::collections::HashMap;
use tracing::info_span;
use tracing_futures::Instrument;

use super::openai::{CompletionResponse, StreamingToolCall, TranscriptionResponse, Usage};
use crate::client::{CompletionClient, TranscriptionClient, VerifyClient, VerifyError};
use crate::completion::GetTokenUsage;
use crate::http_client::sse::{Event, GenericEventSource};
use crate::http_client::{self, HttpClientExt};
use crate::json_utils::merge;
use crate::providers::openai::{AssistantContent, Function, ToolType};
use async_stream::stream;
use futures::StreamExt;

use crate::{
    OneOrMany,
    completion::{self, CompletionError, CompletionRequest},
    json_utils,
    message::{self, MessageError},
    providers::openai::ToolDefinition,
    transcription::{self, TranscriptionError},
};
use reqwest::multipart::Part;
use rig::client::ProviderClient;
use rig::impl_conversion_traits;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

// ================================================================
// Main Flex Client
// ================================================================

pub struct ClientBuilder<'a, T = reqwest::Client> {
    api_key: &'a str,
    base_url: &'a str,
    http_client: T,
}

impl<'a, T> ClientBuilder<'a, T>
where
    T: Default,
{
    pub fn new(api_key: &'a str, base_url: &'a str) -> Self {
        Self {
            api_key,
            base_url,
            http_client: Default::default(),
        }
    }
}

impl<'a, T> ClientBuilder<'a, T> {
    pub fn new_with_client(api_key: &'a str, base_url: &'a str, http_client: T) -> Self {
        Self {
            api_key,
            base_url,
            http_client,
        }
    }

    pub fn base_url(mut self, base_url: &'a str) -> Self {
        self.base_url = base_url;
        self
    }

    pub fn with_client<U>(self, http_client: U) -> ClientBuilder<'a, U> {
        ClientBuilder {
            api_key: self.api_key,
            base_url: self.base_url,
            http_client,
        }
    }

    pub fn build(self) -> Client<T> {
        Client {
            base_url: self.base_url.to_string(),
            api_key: self.api_key.to_string(),
            http_client: self.http_client,
        }
    }
}

#[derive(Clone)]
pub struct Client<T = reqwest::Client> {
    base_url: String,
    api_key: String,
    http_client: T,
}

impl<T> std::fmt::Debug for Client<T>
where
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
            .field("base_url", &self.base_url)
            .field("http_client", &self.http_client)
            .field("api_key", &"<REDACTED>")
            .finish()
    }
}

impl<T> Client<T>
where
    T: HttpClientExt,
{
    fn req(
        &self,
        method: http_client::Method,
        path: &str,
    ) -> http_client::Result<http_client::Builder> {
        let url = format!("{}/{}", self.base_url, path.trim_start_matches('/'));

        http_client::with_bearer_auth(
            http_client::Builder::new().method(method).uri(url),
            &self.api_key,
        )
    }

    fn get(&self, path: &str) -> http_client::Result<http_client::Builder> {
        self.req(http_client::Method::GET, path)
    }
}

impl Client<reqwest::Client> {
    pub fn builder<'a>(api_key: &'a str, base_url: &'a str) -> ClientBuilder<'a, reqwest::Client> {
        ClientBuilder::new(api_key, base_url)
    }

    pub fn new(api_key: &str, base_url: &str) -> Self {
        ClientBuilder::new(api_key, base_url).build()
    }

    pub fn from_env() -> Self {
        <Self as ProviderClient>::from_env()
    }
}

impl<T> ProviderClient for Client<T>
where
    T: HttpClientExt + Clone + Send + std::fmt::Debug + Default + 'static,
{
    /// Create a new Flex client from the following environment variables:
    /// - `FLEX_API_KEY` (required): The API key for the flexible provider
    /// - `FLEX_BASE_URL` (required): The base URL for the API
    fn from_env() -> Self {
        let api_key = std::env::var("FLEX_API_KEY").expect("FLEX_API_KEY not set");
        let base_url = std::env::var("FLEX_BASE_URL").expect("FLEX_BASE_URL not set");

        ClientBuilder::<T>::new(&api_key, &base_url).build()
    }

    fn from_val(input: crate::client::ProviderValue) -> Self {
        // For now, we only support the simple case where the input is just the API key
        // The base URL would need to be provided via environment variable or some other config
        let crate::client::ProviderValue::Simple(api_key) = input else {
            panic!(
                "Incorrect provider value type - Flex provider only supports simple API key format for now"
            )
        };

        // In from_val, we still need the base URL. The user should set FLEX_BASE_URL in their environment
        let base_url = std::env::var("FLEX_BASE_URL")
            .expect("FLEX_BASE_URL must be set when using Flex provider from_val");

        ClientBuilder::<T>::new(&api_key, &base_url).build()
    }
}

impl<T> CompletionClient for Client<T>
where
    T: HttpClientExt + Clone + Send + std::fmt::Debug + Default + 'static,
{
    type CompletionModel = CompletionModel<T>;

    /// Create a completion model with the given name.
    fn completion_model(&self, model: &str) -> Self::CompletionModel {
        CompletionModel::new(self.clone(), model)
    }
}

impl<T> TranscriptionClient for Client<T>
where
    T: HttpClientExt + Clone + Send + std::fmt::Debug + Default + 'static,
{
    type TranscriptionModel = TranscriptionModel<T>;

    /// Create a transcription model with the given name.
    fn transcription_model(&self, model: &str) -> Self::TranscriptionModel {
        TranscriptionModel::new(self.clone(), model)
    }
}

impl<T> VerifyClient for Client<T>
where
    T: HttpClientExt + Clone + Send + std::fmt::Debug + Default + 'static,
{
    #[cfg_attr(feature = "worker", worker::send)]
    async fn verify(&self) -> Result<(), VerifyError> {
        let req = self
            .get("/models")?
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
            _ => Ok(()),
        }
    }
}

impl_conversion_traits!(
    AsEmbeddings,
    AsImageGeneration,
    AsAudioGeneration for Client<T>
);

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

#[derive(Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
}

impl TryFrom<Message> for message::Message {
    type Error = message::MessageError;

    fn try_from(message: Message) -> Result<Self, Self::Error> {
        match message.role.as_str() {
            "user" => Ok(Self::User {
                content: OneOrMany::one(
                    message
                        .content
                        .map(|content| message::UserContent::text(&content))
                        .ok_or_else(|| {
                            message::MessageError::ConversionError("Empty user message".to_string())
                        })?,
                ),
            }),
            "assistant" => Ok(Self::Assistant {
                id: None,
                content: OneOrMany::one(
                    message
                        .content
                        .map(|content| message::AssistantContent::text(&content))
                        .ok_or_else(|| {
                            message::MessageError::ConversionError(
                                "Empty assistant message".to_string(),
                            )
                        })?,
                ),
            }),
            _ => Err(message::MessageError::ConversionError(format!(
                "Unknown role: {}",
                message.role
            ))),
        }
    }
}

impl TryFrom<message::Message> for Message {
    type Error = message::MessageError;

    fn try_from(message: message::Message) -> Result<Self, Self::Error> {
        match message {
            message::Message::User { content } => Ok(Self {
                role: "user".to_string(),
                content: content.iter().find_map(|c| match c {
                    message::UserContent::Text(text) => Some(text.text.clone()),
                    _ => None,
                }),
                reasoning: None,
            }),
            message::Message::Assistant { content, .. } => {
                let mut text_content: Option<String> = None;
                let mut flex_reasoning: Option<String> = None;

                for c in content.iter() {
                    match c {
                        message::AssistantContent::Text(text) => {
                            text_content = Some(
                                text_content
                                    .map(|mut existing| {
                                        existing.push('\n');
                                        existing.push_str(&text.text);
                                        existing
                                    })
                                    .unwrap_or_else(|| text.text.clone()),
                            );
                        }
                        message::AssistantContent::ToolCall(_tool_call) => {
                            return Err(MessageError::ConversionError(
                                "Tool calls do not exist on this message".into(),
                            ));
                        }
                        message::AssistantContent::Reasoning(message::Reasoning {
                            reasoning,
                            ..
                        }) => {
                            flex_reasoning =
                                Some(reasoning.first().cloned().unwrap_or(String::new()));
                        }
                    }
                }

                Ok(Self {
                    role: "assistant".to_string(),
                    content: text_content,
                    reasoning: flex_reasoning,
                })
            }
        }
    }
}

// ================================================================
// Flex Completion API
// ================================================================

#[derive(Clone, Debug)]
pub struct CompletionModel<T> {
    client: Client<T>,
    /// Name of the model (e.g.: gpt-4o)
    pub model: String,
}

impl<T> CompletionModel<T> {
    pub fn new(client: Client<T>, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }

    fn create_completion_request(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<Value, CompletionError> {
        // Build up the order of messages (context, chat_history, prompt)
        let mut partial_history = vec![];
        if let Some(docs) = completion_request.normalized_documents() {
            partial_history.push(docs);
        }
        partial_history.extend(completion_request.chat_history);

        // Initialize full history with preamble (or empty if non-existent)
        let mut full_history: Vec<Message> =
            completion_request
                .preamble
                .map_or_else(Vec::new, |preamble| {
                    vec![Message {
                        role: "system".to_string(),
                        content: Some(preamble),
                        reasoning: None,
                    }]
                });

        // Convert and extend the rest of the history
        full_history.extend(
            partial_history
                .into_iter()
                .map(message::Message::try_into)
                .collect::<Result<Vec<Message>, _>>()?,
        );

        let tool_choice = completion_request
            .tool_choice
            .map(crate::providers::openai::ToolChoice::try_from)
            .transpose()?;

        let request = if completion_request.tools.is_empty() {
            json!({
                "model": self.model,
                "messages": full_history,
                "temperature": completion_request.temperature,
            })
        } else {
            json!({
                "model": self.model,
                "messages": full_history,
                "temperature": completion_request.temperature,
                "tools": completion_request.tools.into_iter().map(ToolDefinition::from).collect::<Vec<_>>(),
                "tool_choice": tool_choice,
                "reasoning_format": "parsed"
            })
        };

        let request = if let Some(params) = completion_request.additional_params {
            json_utils::merge(request, params)
        } else {
            request
        };

        Ok(request)
    }
}

impl<T> completion::CompletionModel for CompletionModel<T>
where
    T: HttpClientExt + Clone + Send + std::fmt::Debug + Default + 'static,
{
    type Response = CompletionResponse;
    type StreamingResponse = StreamingCompletionResponse;

    #[cfg_attr(feature = "worker", worker::send)]
    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
        let preamble = completion_request.preamble.clone();

        let request = self.create_completion_request(completion_request)?;
        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat",
                gen_ai.operation.name = "chat",
                gen_ai.provider.name = "flex",
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

        let body = serde_json::to_vec(&request)?;
        let req = self
            .client
            .req(Method::POST, "/chat/completions")?
            .header("Content-Type", "application/json")
            .body(body)
            .map_err(|e| http_client::Error::Instance(e.into()))?;

        let async_block = async move {
            let response = self.client.http_client.send::<_, Bytes>(req).await?;
            let status = response.status();
            let response_body = response.into_body().into_future().await?.to_vec();

            if status.is_success() {
                match serde_json::from_slice::<ApiResponse<CompletionResponse>>(&response_body)? {
                    ApiResponse::Ok(response) => {
                        let span = tracing::Span::current();
                        span.record("gen_ai.response.id", response.id.clone());
                        span.record("gen_ai.response.model_name", response.model.clone());
                        span.record(
                            "gen_ai.output.messages",
                            serde_json::to_string(&response.choices).unwrap(),
                        );
                        if let Some(ref usage) = response.usage {
                            span.record("gen_ai.usage.input_tokens", usage.prompt_tokens);
                            span.record(
                                "gen_ai.usage.output_tokens",
                                usage.total_tokens - usage.prompt_tokens,
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
        };

        tracing::Instrument::instrument(async_block, span).await
    }

    #[cfg_attr(feature = "worker", worker::send)]
    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<
        crate::streaming::StreamingCompletionResponse<Self::StreamingResponse>,
        CompletionError,
    > {
        let preamble = request.preamble.clone();
        let mut request = self.create_completion_request(request)?;

        request = merge(
            request,
            json!({"stream": true, "stream_options": {"include_usage": true}}),
        );

        let body = serde_json::to_vec(&request)?;
        let req = self
            .client
            .req(Method::POST, "/chat/completions")?
            .header("Content-Type", "application/json")
            .body(body)
            .map_err(|e| http_client::Error::Instance(e.into()))?;

        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat_streaming",
                gen_ai.operation.name = "chat_streaming",
                gen_ai.provider.name = "flex",
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

        tracing::Instrument::instrument(
            send_compatible_streaming_request(self.client.http_client.clone(), req),
            span,
        )
        .await
    }
}

// ================================================================
// Flex Transcription API
// ================================================================

#[derive(Clone)]
pub struct TranscriptionModel<T> {
    client: Client<T>,
    /// Name of the model (e.g.: whisper-1)
    pub model: String,
}

impl<T> TranscriptionModel<T> {
    pub fn new(client: Client<T>, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }
}
impl<T> transcription::TranscriptionModel for TranscriptionModel<T>
where
    T: HttpClientExt + Clone + Send + std::fmt::Debug + Default + 'static,
{
    type Response = TranscriptionResponse;

    #[cfg_attr(feature = "worker", worker::send)]
    async fn transcription(
        &self,
        request: transcription::TranscriptionRequest,
    ) -> Result<
        transcription::TranscriptionResponse<Self::Response>,
        transcription::TranscriptionError,
    > {
        let data = request.data;

        let mut body = reqwest::multipart::Form::new()
            .text("model", self.model.clone())
            .text("language", request.language)
            .part(
                "file",
                Part::bytes(data).file_name(request.filename.clone()),
            );

        if let Some(prompt) = request.prompt {
            body = body.text("prompt", prompt.clone());
        }

        if let Some(ref temperature) = request.temperature {
            body = body.text("temperature", temperature.to_string());
        }

        if let Some(ref additional_params) = request.additional_params {
            for (key, value) in additional_params
                .as_object()
                .expect("Additional Parameters to Flex Transcription should be a map")
            {
                body = body.text(key.to_owned(), value.to_string());
            }
        }

        let req = self
            .client
            .req(Method::POST, "/audio/transcriptions")?
            .body(body)
            .unwrap();

        let response = self
            .client
            .http_client
            .send_multipart::<Bytes>(req)
            .await
            .unwrap();

        let status = response.status();
        let response_body = response.into_body().into_future().await?.to_vec();

        if status.is_success() {
            match serde_json::from_slice::<ApiResponse<TranscriptionResponse>>(&response_body)? {
                ApiResponse::Ok(response) => response.try_into(),
                ApiResponse::Err(api_error_response) => Err(TranscriptionError::ProviderError(
                    api_error_response.message,
                )),
            }
        } else {
            Err(TranscriptionError::ProviderError(
                String::from_utf8_lossy(&response_body).to_string(),
            ))
        }
    }
}

#[derive(Deserialize, Debug)]
#[serde(untagged)]
pub enum StreamingDelta {
    Reasoning {
        reasoning: String,
    },
    MessageContent {
        #[serde(default)]
        content: Option<String>,
        #[serde(default, deserialize_with = "json_utils::null_or_vec")]
        tool_calls: Vec<StreamingToolCall>,
    },
}

#[derive(Deserialize, Debug)]
struct StreamingChoice {
    delta: StreamingDelta,
}

#[derive(Deserialize, Debug)]
struct StreamingCompletionChunk {
    choices: Vec<StreamingChoice>,
    usage: Option<Usage>,
}

#[derive(Clone, Deserialize, Serialize, Debug)]
pub struct StreamingCompletionResponse {
    pub usage: Usage,
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

pub async fn send_compatible_streaming_request<T>(
    client: T,
    req: Request<Vec<u8>>,
) -> Result<
    crate::streaming::StreamingCompletionResponse<StreamingCompletionResponse>,
    CompletionError,
>
where
    T: HttpClientExt + Clone + 'static,
{
    let span = tracing::Span::current();

    let mut event_source = GenericEventSource::new(client, req);

    let stream = stream! {
        let span = tracing::Span::current();
        let mut final_usage = Usage {
            prompt_tokens: 0,
            total_tokens: 0
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

                    let parsed = serde_json::from_str::<StreamingCompletionChunk>(data_str);
                    let Ok(data) = parsed else {
                        let err = parsed.unwrap_err();
                        tracing::debug!("Couldn't parse SSE payload as StreamingCompletionChunk: {:?}", err);
                        continue;
                    };

                    if let Some(choice) = data.choices.first() {
                        match &choice.delta {
                            StreamingDelta::Reasoning { reasoning } => {
                                yield Ok(crate::streaming::RawStreamingChoice::Reasoning {
                                    id: None,
                                    reasoning: reasoning.to_string(),
                                    signature: None,
                                });
                            }

                            StreamingDelta::MessageContent { content, tool_calls } => {
                                // Handle tool calls
                                for tool_call in tool_calls {
                                    let function = &tool_call.function;

                                    // Start of tool call
                                    if function.name.as_ref().map(|s| !s.is_empty()).unwrap_or(false)
                                        && function.arguments.is_empty()
                                    {
                                        let id = tool_call.id.clone().unwrap_or_default();
                                        let name = function.name.clone().unwrap();
                                        calls.insert(tool_call.index, (id, name, String::new()));
                                    }
                                    // Continuation
                                    else if function.name.as_ref().map(|s| s.is_empty()).unwrap_or(true)
                                        && !function.arguments.is_empty()
                                    {
                                        if let Some((id, name, existing_args)) = calls.get(&tool_call.index) {
                                            let combined = format!("{}{}", existing_args, function.arguments);
                                            calls.insert(tool_call.index, (id.clone(), name.clone(), combined));
                                        } else {
                                            tracing::debug!("Partial tool call received but tool call was never started.");
                                        }
                                    }
                                    // Complete tool call
                                    else {
                                        let id = tool_call.id.clone().unwrap_or_default();
                                        let name = function.name.clone().unwrap_or_default();
                                        let arguments_str = function.arguments.clone();

                                        let Ok(arguments_json) = serde_json::from_str::<serde_json::Value>(&arguments_str) else {
                                            tracing::debug!("Couldn't parse tool call args '{}'", arguments_str);
                                            continue;
                                        };

                                        yield Ok(crate::streaming::RawStreamingChoice::ToolCall {
                                            id,
                                            name,
                                            arguments: arguments_json,
                                            call_id: None
                                        });
                                    }
                                }

                                // Streamed content
                                if let Some(content) = content {
                                    text_response += content;
                                    yield Ok(crate::streaming::RawStreamingChoice::Message(content.clone()));
                                }
                            }
                        }
                    }

                    if let Some(usage) = data.usage {
                        final_usage = usage.clone();
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

        let mut tool_calls = Vec::new();
        // Flush accumulated tool calls
        for (_, (id, name, arguments)) in calls {
            let Ok(arguments_json) = serde_json::from_str::<serde_json::Value>(&arguments) else {
                continue;
            };

            tool_calls.push(rig::providers::openai::completion::ToolCall {
                id: id.clone(),
                r#type: ToolType::Function,
                function: Function {
                    name: name.clone(),
                    arguments: arguments_json.clone()
                }
            });
            yield Ok(crate::streaming::RawStreamingChoice::ToolCall {
                id,
                name,
                arguments: arguments_json,
                call_id: None,
            });
        }

        let response_message = crate::providers::openai::completion::Message::Assistant {
            content: vec![AssistantContent::Text { text: text_response }],
            refusal: None,
            audio: None,
            name: None,
            tool_calls
        };

        span.record("gen_ai.output.messages", serde_json::to_string(&vec![response_message]).unwrap());
        span.record("gen_ai.usage.input_tokens", final_usage.prompt_tokens);
        span.record("gen_ai.usage.output_tokens", final_usage.total_tokens - final_usage.prompt_tokens);

        // Final response
        yield Ok(crate::streaming::RawStreamingChoice::FinalResponse(
            StreamingCompletionResponse { usage: final_usage.clone() }
        ));
    }.instrument(span);

    Ok(crate::streaming::StreamingCompletionResponse::stream(
        Box::pin(stream),
    ))
}

/// Get model names from the `FLEX_MODELS` environment variable.
/// The models should be comma-separated.
///
/// # Example
/// ```
/// // Set environment variable: FLEX_MODELS=gpt-4o,gpt-4-turbo,llama-3
/// use rig::providers::flex;
///
/// let models = flex::get_models_from_env();
/// println!("{:?}", models); // ["gpt-4o", "gpt-4-turbo", "llama-3"]
/// ```
pub fn get_models_from_env() -> Vec<String> {
    std::env::var("FLEX_MODELS")
        .ok()
        .map(|models_str| {
            models_str
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default()
}
