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
use super::openai::{CompletionResponse, TranscriptionResponse, send_compatible_streaming_request};
use crate::client::{CompletionClient, TranscriptionClient};
use crate::json_utils::merge;
use crate::providers::openai;
use crate::streaming::StreamingCompletionResponse;
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
// Main Groq Client
// ================================================================
const GROQ_API_BASE_URL: &str = "https://api.groq.com/openai/v1";

#[derive(Clone)]
pub struct Client {
    base_url: String,
    api_key: String,
    http_client: reqwest::Client,
}

impl std::fmt::Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
            .field("base_url", &self.base_url)
            .field("http_client", &self.http_client)
            .field("api_key", &"<REDACTED>")
            .finish()
    }
}

impl Client {
    /// Create a new Groq client with the given API key.
    pub fn new(api_key: &str) -> Self {
        Self::from_url(api_key, GROQ_API_BASE_URL)
    }

    /// Create a new Groq client with the given API key and base API URL.
    pub fn from_url(api_key: &str, base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            api_key: api_key.to_string(),
            http_client: reqwest::Client::builder()
                .build()
                .expect("Groq reqwest client should build"),
        }
    }

    /// Use your own `reqwest::Client`.
    /// The required headers will be automatically attached upon trying to make a request.
    pub fn with_custom_client(mut self, client: reqwest::Client) -> Self {
        self.http_client = client;

        self
    }

    fn post(&self, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/{}", self.base_url, path).replace("//", "/");
        self.http_client.post(url).bearer_auth(&self.api_key)
    }
}

impl ProviderClient for Client {
    /// Create a new Groq client from the `GROQ_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    fn from_env() -> Self {
        let api_key = std::env::var("GROQ_API_KEY").expect("GROQ_API_KEY not set");
        Self::new(&api_key)
    }
}

impl CompletionClient for Client {
    type CompletionModel = CompletionModel;

    /// Create a completion model with the given name.
    ///
    /// # Example
    /// ```
    /// use rig::providers::groq::{Client, self};
    ///
    /// // Initialize the Groq client
    /// let groq = Client::new("your-groq-api-key");
    ///
    /// let gpt4 = groq.completion_model(groq::GPT_4);
    /// ```
    fn completion_model(&self, model: &str) -> CompletionModel {
        CompletionModel::new(self.clone(), model)
    }
}

impl TranscriptionClient for Client {
    type TranscriptionModel = TranscriptionModel;

    /// Create a transcription model with the given name.
    ///
    /// # Example
    /// ```
    /// use rig::providers::groq::{Client, self};
    ///
    /// // Initialize the Groq client
    /// let groq = Client::new("your-groq-api-key");
    ///
    /// let gpt4 = groq.transcription_model(groq::WHISPER_LARGE_V3);
    /// ```
    fn transcription_model(&self, model: &str) -> TranscriptionModel {
        TranscriptionModel::new(self.clone(), model)
    }
}

impl_conversion_traits!(
    AsEmbeddings,
    AsImageGeneration,
    AsAudioGeneration for Client
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
            }),
            message::Message::Assistant { content, .. } => {
                let mut text_content: Option<String> = None;

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
                    }
                }

                Ok(Self {
                    role: "assistant".to_string(),
                    content: text_content,
                })
            }
        }
    }
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

#[derive(Clone, Debug)]
pub struct CompletionModel {
    client: Client,
    /// Name of the model (e.g.: deepseek-r1-distill-llama-70b)
    pub model: String,
}

impl CompletionModel {
    pub fn new(client: Client, model: &str) -> Self {
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
                    }]
                });

        // Convert and extend the rest of the history
        full_history.extend(
            partial_history
                .into_iter()
                .map(message::Message::try_into)
                .collect::<Result<Vec<Message>, _>>()?,
        );

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
                "tool_choice": "auto",
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

impl completion::CompletionModel for CompletionModel {
    type Response = CompletionResponse;
    type StreamingResponse = openai::StreamingCompletionResponse;

    #[cfg_attr(feature = "worker", worker::send)]
    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
        let request = self.create_completion_request(completion_request)?;

        let response = self
            .client
            .post("/chat/completions")
            .json(&request)
            .send()
            .await?;

        if response.status().is_success() {
            match response.json::<ApiResponse<CompletionResponse>>().await? {
                ApiResponse::Ok(response) => {
                    tracing::info!(target: "rig",
                        "groq completion token usage: {:?}",
                        response.usage.clone().map(|usage| format!("{usage}")).unwrap_or("N/A".to_string())
                    );
                    response.try_into()
                }
                ApiResponse::Err(err) => Err(CompletionError::ProviderError(err.message)),
            }
        } else {
            Err(CompletionError::ProviderError(response.text().await?))
        }
    }

    #[cfg_attr(feature = "worker", worker::send)]
    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        let mut request = self.create_completion_request(request)?;

        request = merge(
            request,
            json!({"stream": true, "stream_options": {"include_usage": true}}),
        );

        let builder = self.client.post("/chat/completions").json(&request);

        send_compatible_streaming_request(builder).await
    }
}

// ================================================================
// Groq Transcription API
// ================================================================
pub const WHISPER_LARGE_V3: &str = "whisper-large-v3";
pub const WHISPER_LARGE_V3_TURBO: &str = "whisper-large-v3-turbo";
pub const DISTIL_WHISPER_LARGE_V3: &str = "distil-whisper-large-v3-en";

#[derive(Clone)]
pub struct TranscriptionModel {
    client: Client,
    /// Name of the model (e.g.: gpt-3.5-turbo-1106)
    pub model: String,
}

impl TranscriptionModel {
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }
}
impl transcription::TranscriptionModel for TranscriptionModel {
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
                .expect("Additional Parameters to OpenAI Transcription should be a map")
            {
                body = body.text(key.to_owned(), value.to_string());
            }
        }

        let response = self
            .client
            .post("audio/transcriptions")
            .multipart(body)
            .send()
            .await?;

        if response.status().is_success() {
            match response
                .json::<ApiResponse<TranscriptionResponse>>()
                .await?
            {
                ApiResponse::Ok(response) => response.try_into(),
                ApiResponse::Err(api_error_response) => Err(TranscriptionError::ProviderError(
                    api_error_response.message,
                )),
            }
        } else {
            Err(TranscriptionError::ProviderError(response.text().await?))
        }
    }
}
