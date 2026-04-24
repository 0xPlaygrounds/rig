//! Groq API client and Rig integration
//!
//! # Example
//! ```
//! use rig::providers::groq;
//!
//! let client = groq::Client::new("YOUR_API_KEY");
//!
//! let gpt4o = client.completion_model(groq::GPT_4O);
//! ```
use bytes::Bytes;
use http::Request;
use serde_json::{Map, Value};
use tracing::info_span;

use super::openai::{
    CompletionResponse, Message as OpenAIMessage, StreamingToolCall, TranscriptionResponse, Usage,
};
use crate::client::{
    self, BearerAuth, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder,
    ProviderClient,
};
use crate::completion::GetTokenUsage;
use crate::http_client::multipart::Part;
use crate::http_client::{self, HttpClientExt, MultipartForm};
use crate::providers::internal::openai_chat_completions_compatible::{
    self, CompatibleChoiceData, CompatibleChunk, CompatibleFinishReason, CompatibleStreamProfile,
};

use crate::{
    completion::{self, CompletionError, CompletionRequest},
    json_utils,
    message::{self},
    providers::openai::ToolDefinition,
    transcription::{self, TranscriptionError},
};
use serde::{Deserialize, Serialize};

// ================================================================
// Main Groq Client
// ================================================================
const GROQ_API_BASE_URL: &str = "https://api.groq.com/openai/v1";

#[derive(Debug, Default, Clone, Copy)]
pub struct GroqExt;
#[derive(Debug, Default, Clone, Copy)]
pub struct GroqBuilder;

type GroqApiKey = BearerAuth;

impl Provider for GroqExt {
    type Builder = GroqBuilder;
    const VERIFY_PATH: &'static str = "/models";
}

impl<H> Capabilities<H> for GroqExt {
    type Completion = Capable<CompletionModel<H>>;
    type Embeddings = Nothing;
    type Transcription = Capable<TranscriptionModel<H>>;
    type ModelListing = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;

    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
}

impl DebugExt for GroqExt {}

impl ProviderBuilder for GroqBuilder {
    type Extension<H>
        = GroqExt
    where
        H: HttpClientExt;
    type ApiKey = GroqApiKey;

    const BASE_URL: &'static str = GROQ_API_BASE_URL;

    fn build<H>(
        _builder: &client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: HttpClientExt,
    {
        Ok(GroqExt)
    }
}

pub type Client<H = reqwest::Client> = client::Client<GroqExt, H>;
pub type ClientBuilder<H = reqwest::Client> = client::ClientBuilder<GroqBuilder, String, H>;

impl ProviderClient for Client {
    type Input = String;
    type Error = crate::client::ProviderClientError;

    /// Create a new Groq client from the `GROQ_API_KEY` environment variable.
    fn from_env() -> Result<Self, Self::Error> {
        let api_key = crate::client::required_env_var("GROQ_API_KEY")?;
        Self::new(&api_key).map_err(Into::into)
    }

    fn from_val(input: Self::Input) -> Result<Self, Self::Error> {
        Self::new(&input).map_err(Into::into)
    }
}

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
// Groq Completion API
// ================================================================

/// The `deepseek-r1-distill-llama-70b` model. Used for chat completion.
pub const DEEPSEEK_R1_DISTILL_LLAMA_70B: &str = "deepseek-r1-distill-llama-70b";
/// The `gemma2-9b-it` model. Used for chat completion.
pub const GEMMA2_9B_IT: &str = "gemma2-9b-it";
/// The `llama-3.1-8b-instant` model. Used for chat completion.
pub const LLAMA_3_1_8B_INSTANT: &str = "llama-3.1-8b-instant";
/// The `llama-3.2-11b-vision-preview` model. Used for chat completion.
pub const LLAMA_3_2_11B_VISION_PREVIEW: &str = "llama-3.2-11b-vision-preview";
/// The `llama-3.2-1b-preview` model. Used for chat completion.
pub const LLAMA_3_2_1B_PREVIEW: &str = "llama-3.2-1b-preview";
/// The `llama-3.2-3b-preview` model. Used for chat completion.
pub const LLAMA_3_2_3B_PREVIEW: &str = "llama-3.2-3b-preview";
/// The `llama-3.2-90b-vision-preview` model. Used for chat completion.
pub const LLAMA_3_2_90B_VISION_PREVIEW: &str = "llama-3.2-90b-vision-preview";
/// The `llama-3.2-70b-specdec` model. Used for chat completion.
pub const LLAMA_3_2_70B_SPECDEC: &str = "llama-3.2-70b-specdec";
/// The `llama-3.2-70b-versatile` model. Used for chat completion.
pub const LLAMA_3_2_70B_VERSATILE: &str = "llama-3.2-70b-versatile";
/// The `llama-guard-3-8b` model. Used for chat completion.
pub const LLAMA_GUARD_3_8B: &str = "llama-guard-3-8b";
/// The `llama3-70b-8192` model. Used for chat completion.
pub const LLAMA_3_70B_8192: &str = "llama3-70b-8192";
/// The `llama3-8b-8192` model. Used for chat completion.
pub const LLAMA_3_8B_8192: &str = "llama3-8b-8192";
/// The `mixtral-8x7b-32768` model. Used for chat completion.
pub const MIXTRAL_8X7B_32768: &str = "mixtral-8x7b-32768";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningFormat {
    Parsed,
    Raw,
    Hidden,
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct GroqCompletionRequest {
    model: String,
    pub messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ToolDefinition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<crate::providers::openai::completion::ToolChoice>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub additional_params: Option<GroqAdditionalParameters>,
    pub(super) stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) stream_options: Option<StreamOptions>,
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub(super) struct StreamOptions {
    pub(super) include_usage: bool,
}

impl TryFrom<(&str, CompletionRequest)> for GroqCompletionRequest {
    type Error = CompletionError;

    fn try_from((model, mut req): (&str, CompletionRequest)) -> Result<Self, Self::Error> {
        if req.output_schema.is_some() {
            tracing::warn!("Structured outputs currently not supported for Groq");
        }
        let model = req.model.clone().unwrap_or_else(|| model.to_string());
        // Build up the order of messages (context, chat_history, prompt)
        let mut partial_history = vec![];
        if let Some(docs) = req.normalized_documents() {
            partial_history.push(docs);
        }
        partial_history.extend(req.chat_history);

        // Add preamble to chat history (if available)
        let mut full_history: Vec<OpenAIMessage> = match &req.preamble {
            Some(preamble) => vec![OpenAIMessage::system(preamble)],
            None => vec![],
        };

        // Convert and extend the rest of the history
        full_history.extend(
            partial_history
                .into_iter()
                .map(message::Message::try_into)
                .collect::<Result<Vec<Vec<OpenAIMessage>>, _>>()?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>(),
        );

        let tool_choice = req
            .tool_choice
            .clone()
            .map(crate::providers::openai::ToolChoice::try_from)
            .transpose()?;

        let mut additional_params_payload = req.additional_params.take().unwrap_or(Value::Null);
        let native_tools =
            extract_native_tools_from_additional_params(&mut additional_params_payload)?;

        let mut additional_params: Option<GroqAdditionalParameters> =
            if additional_params_payload.is_null() {
                None
            } else {
                Some(serde_json::from_value(additional_params_payload)?)
            };
        apply_native_tools_to_additional_params(&mut additional_params, native_tools);

        Ok(Self {
            model: model.to_string(),
            messages: full_history,
            temperature: req.temperature,
            tools: req
                .tools
                .clone()
                .into_iter()
                .map(ToolDefinition::from)
                .collect::<Vec<_>>(),
            tool_choice,
            additional_params,
            stream: false,
            stream_options: None,
        })
    }
}

fn extract_native_tools_from_additional_params(
    additional_params: &mut Value,
) -> Result<Vec<Value>, CompletionError> {
    if let Some(map) = additional_params.as_object_mut()
        && let Some(raw_tools) = map.remove("tools")
    {
        return serde_json::from_value::<Vec<Value>>(raw_tools).map_err(|err| {
            CompletionError::RequestError(
                format!("Invalid Groq `additional_params.tools` payload: {err}").into(),
            )
        });
    }

    Ok(Vec::new())
}

fn apply_native_tools_to_additional_params(
    additional_params: &mut Option<GroqAdditionalParameters>,
    native_tools: Vec<Value>,
) {
    if native_tools.is_empty() {
        return;
    }

    let params = additional_params.get_or_insert_with(GroqAdditionalParameters::default);
    let extra = params.extra.get_or_insert_with(Map::new);

    let mut compound_custom = match extra.remove("compound_custom") {
        Some(Value::Object(map)) => map,
        _ => Map::new(),
    };

    let mut enabled_tools = match compound_custom.remove("enabled_tools") {
        Some(Value::Array(values)) => values,
        _ => Vec::new(),
    };

    for native_tool in native_tools {
        let already_enabled = enabled_tools
            .iter()
            .any(|existing| native_tools_match(existing, &native_tool));
        if !already_enabled {
            enabled_tools.push(native_tool);
        }
    }

    compound_custom.insert("enabled_tools".to_string(), Value::Array(enabled_tools));
    extra.insert(
        "compound_custom".to_string(),
        Value::Object(compound_custom),
    );
}

fn native_tools_match(lhs: &Value, rhs: &Value) -> bool {
    if let (Some(lhs_type), Some(rhs_type)) = (native_tool_kind(lhs), native_tool_kind(rhs)) {
        return lhs_type == rhs_type;
    }

    lhs == rhs
}

fn native_tool_kind(value: &Value) -> Option<&str> {
    match value {
        Value::String(kind) => Some(kind),
        Value::Object(map) => map.get("type").and_then(Value::as_str),
        _ => None,
    }
}

/// Additional parameters to send to the Groq API
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GroqAdditionalParameters {
    /// The reasoning format. See Groq's API docs for more details.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_format: Option<ReasoningFormat>,
    /// Whether or not to include reasoning. See Groq's API docs for more details.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_reasoning: Option<bool>,
    /// Any other properties not included by default on this struct (that you want to send)
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub extra: Option<Map<String, serde_json::Value>>,
}

#[derive(Clone, Debug)]
pub struct CompletionModel<T = reqwest::Client> {
    client: Client<T>,
    /// Name of the model (e.g.: deepseek-r1-distill-llama-70b)
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

impl<T> completion::CompletionModel for CompletionModel<T>
where
    T: HttpClientExt + Clone + Send + std::fmt::Debug + Default + 'static,
{
    type Response = CompletionResponse;
    type StreamingResponse = StreamingCompletionResponse;

    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model)
    }

    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat",
                gen_ai.operation.name = "chat",
                gen_ai.provider.name = "groq",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = tracing::field::Empty,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.usage.cache_read.input_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        span.record("gen_ai.system_instructions", &completion_request.preamble);

        let request = GroqCompletionRequest::try_from((self.model.as_ref(), completion_request))?;

        if tracing::enabled!(tracing::Level::TRACE) {
            tracing::trace!(target: "rig::completions",
                "Groq completion request: {}",
                serde_json::to_string_pretty(&request)?
            );
        }

        let body = serde_json::to_vec(&request)?;
        let req = self
            .client
            .post("/chat/completions")?
            .body(body)
            .map_err(|e| http_client::Error::Instance(e.into()))?;

        let async_block = async move {
            let response = self.client.send::<_, Bytes>(req).await?;
            let status = response.status();
            let response_body = response.into_body().into_future().await?.to_vec();

            if status.is_success() {
                match serde_json::from_slice::<ApiResponse<CompletionResponse>>(&response_body)? {
                    ApiResponse::Ok(response) => {
                        let span = tracing::Span::current();
                        span.record("gen_ai.response.id", response.id.clone());
                        span.record("gen_ai.response.model", response.model.clone());
                        if let Some(ref usage) = response.usage {
                            span.record("gen_ai.usage.input_tokens", usage.prompt_tokens);
                            span.record(
                                "gen_ai.usage.output_tokens",
                                usage.total_tokens - usage.prompt_tokens,
                            );
                            span.record(
                                "gen_ai.usage.cache_read.input_tokens",
                                usage
                                    .prompt_tokens_details
                                    .as_ref()
                                    .map(|d| d.cached_tokens)
                                    .unwrap_or(0),
                            );
                        }

                        if tracing::enabled!(tracing::Level::TRACE) {
                            tracing::trace!(target: "rig::completions",
                                "Groq completion response: {}",
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
        };

        tracing::Instrument::instrument(async_block, span).await
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<
        crate::streaming::StreamingCompletionResponse<Self::StreamingResponse>,
        CompletionError,
    > {
        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat_streaming",
                gen_ai.operation.name = "chat_streaming",
                gen_ai.provider.name = "groq",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = tracing::field::Empty,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.usage.cache_read.input_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        span.record("gen_ai.system_instructions", &request.preamble);

        let mut request = GroqCompletionRequest::try_from((self.model.as_ref(), request))?;

        request.stream = true;
        request.stream_options = Some(StreamOptions {
            include_usage: true,
        });

        if tracing::enabled!(tracing::Level::TRACE) {
            tracing::trace!(target: "rig::completions",
                "Groq streaming completion request: {}",
                serde_json::to_string_pretty(&request)?
            );
        }

        let body = serde_json::to_vec(&request)?;
        let req = self
            .client
            .post("/chat/completions")?
            .body(body)
            .map_err(|e| http_client::Error::Instance(e.into()))?;

        tracing::Instrument::instrument(
            send_compatible_streaming_request(self.client.clone(), req),
            span,
        )
        .await
    }
}

// ================================================================
// Groq Transcription API
// ================================================================

pub const WHISPER_LARGE_V3: &str = "whisper-large-v3";
pub const WHISPER_LARGE_V3_TURBO: &str = "whisper-large-v3-turbo";
pub const DISTIL_WHISPER_LARGE_V3_EN: &str = "distil-whisper-large-v3-en";

#[derive(Clone)]
pub struct TranscriptionModel<T> {
    client: Client<T>,
    /// Name of the model (e.g.: gpt-3.5-turbo-1106)
    pub model: String,
}

impl<T> TranscriptionModel<T> {
    pub fn new(client: Client<T>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }
}
impl<T> transcription::TranscriptionModel for TranscriptionModel<T>
where
    T: HttpClientExt + Clone + Send + std::fmt::Debug + Default + 'static,
{
    type Response = TranscriptionResponse;

    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model)
    }

    async fn transcription(
        &self,
        request: transcription::TranscriptionRequest,
    ) -> Result<
        transcription::TranscriptionResponse<Self::Response>,
        transcription::TranscriptionError,
    > {
        let data = request.data;

        let mut body = MultipartForm::new()
            .text("model", self.model.clone())
            .part(Part::bytes("file", data).filename(request.filename.clone()));

        if let Some(language) = request.language {
            body = body.text("language", language);
        }

        if let Some(prompt) = request.prompt {
            body = body.text("prompt", prompt.clone());
        }

        if let Some(ref temperature) = request.temperature {
            body = body.text("temperature", temperature.to_string());
        }

        if let Some(ref additional_params) = request.additional_params {
            let params = additional_params.as_object().ok_or_else(|| {
                TranscriptionError::RequestError(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "additional transcription parameters must be a JSON object",
                )))
            })?;

            for (key, value) in params {
                body = body.text(key.to_owned(), value.to_string());
            }
        }

        let req = self
            .client
            .post("/audio/transcriptions")?
            .body(body)
            .map_err(|e| TranscriptionError::HttpError(e.into()))?;

        let response = self.client.send_multipart::<Bytes>(req).await?;

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
enum StreamingDelta {
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
    id: Option<String>,
    model: Option<String>,
    choices: Vec<StreamingChoice>,
    usage: Option<Usage>,
}

#[derive(Clone, Deserialize, Serialize, Debug)]
pub struct StreamingCompletionResponse {
    pub usage: Usage,
}

impl GetTokenUsage for StreamingCompletionResponse {
    fn token_usage(&self) -> Option<crate::completion::Usage> {
        self.usage.token_usage()
    }
}

#[derive(Clone, Copy)]
struct GroqCompatibleProfile;

impl CompatibleStreamProfile for GroqCompatibleProfile {
    type Usage = Usage;
    type Detail = ();
    type FinalResponse = StreamingCompletionResponse;

    fn normalize_chunk(
        &self,
        data: &str,
    ) -> Result<Option<CompatibleChunk<Self::Usage, Self::Detail>>, CompletionError> {
        let data = match serde_json::from_str::<StreamingCompletionChunk>(data) {
            Ok(data) => data,
            Err(error) => {
                tracing::debug!(
                    "Couldn't parse SSE payload as StreamingCompletionChunk: {:?}",
                    error
                );
                return Ok(None);
            }
        };

        Ok(Some(
            openai_chat_completions_compatible::normalize_first_choice_chunk(
                data.id,
                data.model,
                data.usage,
                &data.choices,
                |choice| match &choice.delta {
                    StreamingDelta::Reasoning { reasoning } => CompatibleChoiceData {
                        finish_reason: CompatibleFinishReason::Other,
                        text: None,
                        reasoning: Some(reasoning.clone()),
                        tool_calls: Vec::new(),
                        details: Vec::new(),
                    },
                    StreamingDelta::MessageContent {
                        content,
                        tool_calls,
                    } => CompatibleChoiceData {
                        finish_reason: CompatibleFinishReason::Other,
                        text: content.clone(),
                        reasoning: None,
                        tool_calls: openai_chat_completions_compatible::tool_call_chunks(
                            tool_calls,
                        ),
                        details: Vec::new(),
                    },
                },
            ),
        ))
    }

    fn build_final_response(&self, usage: Self::Usage) -> Self::FinalResponse {
        StreamingCompletionResponse { usage }
    }

    fn uses_distinct_tool_call_eviction(&self) -> bool {
        true
    }

    fn emits_complete_single_chunk_tool_calls(&self) -> bool {
        true
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
    openai_chat_completions_compatible::send_compatible_streaming_request(
        client,
        req,
        GroqCompatibleProfile,
    )
    .await
}

#[cfg(test)]
mod tests {
    use crate::{
        OneOrMany,
        providers::{
            groq::{GroqAdditionalParameters, GroqCompletionRequest},
            openai::{Message, UserContent},
        },
    };

    #[test]
    fn serialize_groq_request() {
        let additional_params = GroqAdditionalParameters {
            include_reasoning: Some(true),
            reasoning_format: Some(super::ReasoningFormat::Parsed),
            ..Default::default()
        };

        let groq = GroqCompletionRequest {
            model: "openai/gpt-120b-oss".to_string(),
            temperature: None,
            tool_choice: None,
            stream_options: None,
            tools: Vec::new(),
            messages: vec![Message::User {
                content: OneOrMany::one(UserContent::Text {
                    text: "Hello world!".to_string(),
                }),
                name: None,
            }],
            stream: false,
            additional_params: Some(additional_params),
        };

        let json = serde_json::to_value(&groq).unwrap();

        assert_eq!(
            json,
            serde_json::json!({
                "model": "openai/gpt-120b-oss",
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello world!"
                    }
                ],
                "stream": false,
                "include_reasoning": true,
                "reasoning_format": "parsed"
            })
        )
    }
    #[test]
    fn test_client_initialization() {
        let _client =
            crate::providers::groq::Client::new("dummy-key").expect("Client::new() failed");
        let _client_from_builder = crate::providers::groq::Client::builder()
            .api_key("dummy-key")
            .build()
            .expect("Client::builder() failed");
    }
}
