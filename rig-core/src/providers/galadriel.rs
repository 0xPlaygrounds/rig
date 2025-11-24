//! Galadriel API client and Rig integration
//!
//! # Example
//! ```
//! use rig::providers::galadriel;
//!
//! let client = galadriel::Client::new("YOUR_API_KEY", None);
//! // to use a fine-tuned model
//! // let client = galadriel::Client::new("YOUR_API_KEY", "FINE_TUNE_API_KEY");
//!
//! let gpt4o = client.completion_model(galadriel::GPT_4O);
//! ```
use super::openai;
use crate::client::{
    self, BearerAuth, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder,
    ProviderClient,
};
use crate::http_client::{self, HttpClientExt};
use crate::json_utils::merge;
use crate::message::MessageError;
use crate::providers::openai::send_compatible_streaming_request;
use crate::streaming::StreamingCompletionResponse;
use crate::{
    OneOrMany,
    completion::{self, CompletionError, CompletionRequest},
    json_utils, message,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tracing::{Instrument, info_span};

// ================================================================
// Main Galadriel Client
// ================================================================
const GALADRIEL_API_BASE_URL: &str = "https://api.galadriel.com/v1/verified";

#[derive(Debug, Default, Clone)]
pub struct GaladrielExt {
    fine_tune_api_key: Option<String>,
}

#[derive(Debug, Default, Clone)]
pub struct GaladrielBuilder {
    fine_tune_api_key: Option<String>,
}

type GaladrielApiKey = BearerAuth;

impl Provider for GaladrielExt {
    type Builder = GaladrielBuilder;

    /// There is currently no way to verify a Galadriel api key without consuming tokens
    const VERIFY_PATH: &'static str = "";

    fn build<H>(
        builder: &crate::client::ClientBuilder<
            Self::Builder,
            <Self::Builder as crate::client::ProviderBuilder>::ApiKey,
            H,
        >,
    ) -> http_client::Result<Self> {
        let GaladrielBuilder { fine_tune_api_key } = builder.ext().clone();

        Ok(Self { fine_tune_api_key })
    }
}

impl<H> Capabilities<H> for GaladrielExt {
    type Completion = Capable<CompletionModel<H>>;
    type Embeddings = Nothing;
    type Transcription = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;
    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
}

impl DebugExt for GaladrielExt {
    fn fields(&self) -> impl Iterator<Item = (&'static str, &dyn std::fmt::Debug)> {
        std::iter::once((
            "fine_tune_api_key",
            (&self.fine_tune_api_key as &dyn std::fmt::Debug),
        ))
    }
}

impl ProviderBuilder for GaladrielBuilder {
    type Output = GaladrielExt;
    type ApiKey = GaladrielApiKey;

    const BASE_URL: &'static str = GALADRIEL_API_BASE_URL;
}

pub type Client<H = reqwest::Client> = client::Client<GaladrielExt, H>;
pub type ClientBuilder<H = reqwest::Client> =
    client::ClientBuilder<GaladrielBuilder, GaladrielApiKey, H>;

impl<T> ClientBuilder<T> {
    pub fn fine_tune_api_key<S>(mut self, fine_tune_api_key: S) -> Self
    where
        S: AsRef<str>,
    {
        *self.ext_mut() = GaladrielBuilder {
            fine_tune_api_key: Some(fine_tune_api_key.as_ref().into()),
        };

        self
    }
}

impl ProviderClient for Client {
    type Input = (String, Option<String>);

    /// Create a new Galadriel client from the `GALADRIEL_API_KEY` environment variable,
    /// and optionally from the `GALADRIEL_FINE_TUNE_API_KEY` environment variable.
    /// Panics if the `GALADRIEL_API_KEY` environment variable is not set.
    fn from_env() -> Self {
        let api_key = std::env::var("GALADRIEL_API_KEY").expect("GALADRIEL_API_KEY not set");
        let fine_tune_api_key = std::env::var("GALADRIEL_FINE_TUNE_API_KEY").ok();

        let mut builder = Self::builder().api_key(api_key);

        if let Some(fine_tune_api_key) = fine_tune_api_key.as_deref() {
            builder = builder.fine_tune_api_key(fine_tune_api_key);
        }

        builder.build().unwrap()
    }

    fn from_val((api_key, fine_tune_api_key): Self::Input) -> Self {
        let mut builder = Self::builder().api_key(api_key);

        if let Some(fine_tune_key) = fine_tune_api_key {
            builder = builder.fine_tune_api_key(fine_tune_key)
        }

        builder.build().unwrap()
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

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

impl std::fmt::Display for Usage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Prompt tokens: {} Total tokens: {}",
            self.prompt_tokens, self.total_tokens
        )
    }
}

// ================================================================
// Galadriel Completion API
// ================================================================

/// `o1-preview` completion model
pub const O1_PREVIEW: &str = "o1-preview";
/// `o1-preview-2024-09-12` completion model
pub const O1_PREVIEW_2024_09_12: &str = "o1-preview-2024-09-12";
/// `o1-mini completion model
pub const O1_MINI: &str = "o1-mini";
/// `o1-mini-2024-09-12` completion model
pub const O1_MINI_2024_09_12: &str = "o1-mini-2024-09-12";
/// `gpt-4o` completion model
pub const GPT_4O: &str = "gpt-4o";
/// `gpt-4o-2024-05-13` completion model
pub const GPT_4O_2024_05_13: &str = "gpt-4o-2024-05-13";
/// `gpt-4-turbo` completion model
pub const GPT_4_TURBO: &str = "gpt-4-turbo";
/// `gpt-4-turbo-2024-04-09` completion model
pub const GPT_4_TURBO_2024_04_09: &str = "gpt-4-turbo-2024-04-09";
/// `gpt-4-turbo-preview` completion model
pub const GPT_4_TURBO_PREVIEW: &str = "gpt-4-turbo-preview";
/// `gpt-4-0125-preview` completion model
pub const GPT_4_0125_PREVIEW: &str = "gpt-4-0125-preview";
/// `gpt-4-1106-preview` completion model
pub const GPT_4_1106_PREVIEW: &str = "gpt-4-1106-preview";
/// `gpt-4-vision-preview` completion model
pub const GPT_4_VISION_PREVIEW: &str = "gpt-4-vision-preview";
/// `gpt-4-1106-vision-preview` completion model
pub const GPT_4_1106_VISION_PREVIEW: &str = "gpt-4-1106-vision-preview";
/// `gpt-4` completion model
pub const GPT_4: &str = "gpt-4";
/// `gpt-4-0613` completion model
pub const GPT_4_0613: &str = "gpt-4-0613";
/// `gpt-4-32k` completion model
pub const GPT_4_32K: &str = "gpt-4-32k";
/// `gpt-4-32k-0613` completion model
pub const GPT_4_32K_0613: &str = "gpt-4-32k-0613";
/// `gpt-3.5-turbo` completion model
pub const GPT_35_TURBO: &str = "gpt-3.5-turbo";
/// `gpt-3.5-turbo-0125` completion model
pub const GPT_35_TURBO_0125: &str = "gpt-3.5-turbo-0125";
/// `gpt-3.5-turbo-1106` completion model
pub const GPT_35_TURBO_1106: &str = "gpt-3.5-turbo-1106";
/// `gpt-3.5-turbo-instruct` completion model
pub const GPT_35_TURBO_INSTRUCT: &str = "gpt-3.5-turbo-instruct";

#[derive(Debug, Deserialize, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: Option<String>,
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
}

impl From<ApiErrorResponse> for CompletionError {
    fn from(err: ApiErrorResponse) -> Self {
        CompletionError::ProviderError(err.message)
    }
}

impl TryFrom<CompletionResponse> for completion::CompletionResponse<CompletionResponse> {
    type Error = CompletionError;

    fn try_from(response: CompletionResponse) -> Result<Self, Self::Error> {
        let Choice { message, .. } = response.choices.first().ok_or_else(|| {
            CompletionError::ResponseError("Response contained no choices".to_owned())
        })?;

        let mut content = message
            .content
            .as_ref()
            .map(|c| vec![completion::AssistantContent::text(c)])
            .unwrap_or_default();

        content.extend(message.tool_calls.iter().map(|call| {
            completion::AssistantContent::tool_call(
                &call.function.name,
                &call.function.name,
                call.function.arguments.clone(),
            )
        }));

        let choice = OneOrMany::many(content).map_err(|_| {
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
            })
            .unwrap_or_default();

        Ok(completion::CompletionResponse {
            choice,
            usage,
            raw_response: response,
        })
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: Option<String>,
    #[serde(default, deserialize_with = "json_utils::null_or_vec")]
    pub tool_calls: Vec<openai::ToolCall>,
}

impl TryFrom<Message> for message::Message {
    type Error = message::MessageError;

    fn try_from(message: Message) -> Result<Self, Self::Error> {
        let tool_calls: Vec<message::ToolCall> = message
            .tool_calls
            .into_iter()
            .map(|tool_call| tool_call.into())
            .collect();

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
                content: OneOrMany::many(
                    tool_calls
                        .into_iter()
                        .map(message::AssistantContent::ToolCall)
                        .chain(
                            message
                                .content
                                .map(|content| message::AssistantContent::text(&content))
                                .into_iter(),
                        ),
                )
                .map_err(|_| {
                    message::MessageError::ConversionError("Empty assistant message".to_string())
                })?,
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
                tool_calls: vec![],
            }),
            message::Message::Assistant { content, .. } => {
                let mut text_content: Option<String> = None;
                let mut tool_calls = vec![];

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
                        message::AssistantContent::ToolCall(tool_call) => {
                            tool_calls.push(tool_call.clone().into());
                        }
                        message::AssistantContent::Reasoning(_) => {
                            return Err(MessageError::ConversionError(
                                "Galadriel currently doesn't support reasoning.".into(),
                            ));
                        }
                    }
                }

                Ok(Self {
                    role: "assistant".to_string(),
                    content: text_content,
                    tool_calls,
                })
            }
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ToolDefinition {
    pub r#type: String,
    pub function: completion::ToolDefinition,
}

impl From<completion::ToolDefinition> for ToolDefinition {
    fn from(tool: completion::ToolDefinition) -> Self {
        Self {
            r#type: "function".into(),
            function: tool,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct Function {
    pub name: String,
    pub arguments: String,
}

#[derive(Clone)]
pub struct CompletionModel<T = reqwest::Client> {
    client: Client<T>,
    /// Name of the model (e.g.: gpt-3.5-turbo-1106)
    pub model: String,
}

impl<T> CompletionModel<T>
where
    T: HttpClientExt,
{
    pub fn new(client: Client<T>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }

    pub fn with_model(client: Client<T>, model: &str) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }

    pub(crate) fn create_completion_request(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<Value, CompletionError> {
        // Build up the order of messages (context, chat_history, prompt)
        let mut partial_history = vec![];
        if let Some(docs) = completion_request.normalized_documents() {
            partial_history.push(docs);
        }
        partial_history.extend(completion_request.chat_history);

        // Add preamble to chat history (if available)
        let mut full_history: Vec<Message> = match &completion_request.preamble {
            Some(preamble) => vec![Message {
                role: "system".to_string(),
                content: Some(preamble.to_string()),
                tool_calls: vec![],
            }],
            None => vec![],
        };

        // Convert and extend the rest of the history
        full_history.extend(
            partial_history
                .into_iter()
                .map(message::Message::try_into)
                .collect::<Result<Vec<Message>, _>>()?,
        );

        let tool_choice = completion_request
            .tool_choice
            .clone()
            .map(crate::providers::openai::completion::ToolChoice::try_from)
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
    T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
{
    type Response = CompletionResponse;
    type StreamingResponse = openai::StreamingCompletionResponse;

    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model.into())
    }

    #[cfg_attr(feature = "worker", worker::send)]
    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
        let preamble = completion_request.preamble.clone();
        let request = self.create_completion_request(completion_request)?;
        let body = serde_json::to_vec(&request)?;

        let req = self
            .client
            .post("/chat/completions")?
            .header("Content-Type", "application/json")
            .body(body)
            .map_err(http_client::Error::from)?;

        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat",
                gen_ai.operation.name = "chat",
                gen_ai.provider.name = "galadriel",
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

        async move {
            let response = self.client.http_client().send(req).await?;

            if response.status().is_success() {
                let t = http_client::text(response).await?;
                tracing::debug!(target: "rig::completions", "Galadriel completion response: {t}");

                match serde_json::from_str::<ApiResponse<CompletionResponse>>(&t)? {
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
                let text = http_client::text(response).await?;

                Err(CompletionError::ProviderError(text))
            }
        }
        .instrument(span)
        .await
    }

    #[cfg_attr(feature = "worker", worker::send)]
    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        let preamble = request.preamble.clone();
        let mut request = self.create_completion_request(request)?;

        request = merge(
            request,
            json!({"stream": true, "stream_options": {"include_usage": true}}),
        );

        let body = serde_json::to_vec(&request)?;

        let req = self
            .client
            .post("/chat/completions")?
            .header("Content-Type", "application/json")
            .body(body)
            .map_err(http_client::Error::from)?;

        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat_streaming",
                gen_ai.operation.name = "chat_streaming",
                gen_ai.provider.name = "galadriel",
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

        send_compatible_streaming_request(self.client.http_client().clone(), req)
            .instrument(span)
            .await
    }
}
