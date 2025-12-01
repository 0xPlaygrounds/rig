//! Moonshot API client and Rig integration
//!
//! # Example
//! ```
//! use rig::providers::moonshot;
//!
//! let client = moonshot::Client::new("YOUR_API_KEY");
//!
//! let moonshot_model = client.completion_model(moonshot::MOONSHOT_CHAT);
//! ```
use crate::client::{
    self, BearerAuth, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder,
    ProviderClient,
};
use crate::http_client::HttpClientExt;
use crate::providers::openai::send_compatible_streaming_request;
use crate::streaming::StreamingCompletionResponse;
use crate::{
    completion::{self, CompletionError, CompletionRequest},
    json_utils,
    providers::openai,
};
use crate::{http_client, message};
use serde::{Deserialize, Serialize};
use tracing::{Instrument, info_span};

// ================================================================
// Main Moonshot Client
// ================================================================
const MOONSHOT_API_BASE_URL: &str = "https://api.moonshot.cn/v1";

#[derive(Debug, Default, Clone, Copy)]
pub struct MoonshotExt;
#[derive(Debug, Default, Clone, Copy)]
pub struct MoonshotBuilder;

type MoonshotApiKey = BearerAuth;

impl Provider for MoonshotExt {
    type Builder = MoonshotBuilder;

    const VERIFY_PATH: &'static str = "/models";

    fn build<H>(
        _: &crate::client::ClientBuilder<
            Self::Builder,
            <Self::Builder as crate::client::ProviderBuilder>::ApiKey,
            H,
        >,
    ) -> http_client::Result<Self> {
        Ok(Self)
    }
}

impl DebugExt for MoonshotExt {}

impl ProviderBuilder for MoonshotBuilder {
    type Output = MoonshotExt;
    type ApiKey = MoonshotApiKey;

    const BASE_URL: &'static str = MOONSHOT_API_BASE_URL;
}

impl<H> Capabilities<H> for MoonshotExt {
    type Completion = Capable<CompletionModel<H>>;
    type Embeddings = Nothing;
    type Transcription = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;
    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
}

pub type Client<H = reqwest::Client> = client::Client<MoonshotExt, H>;
pub type ClientBuilder<H = reqwest::Client> =
    client::ClientBuilder<MoonshotBuilder, MoonshotApiKey, H>;

impl ProviderClient for Client {
    type Input = String;

    /// Create a new Moonshot client from the `MOONSHOT_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    fn from_env() -> Self {
        let api_key = std::env::var("MOONSHOT_API_KEY").expect("MOONSHOT_API_KEY not set");
        Self::new(&api_key).unwrap()
    }

    fn from_val(input: Self::Input) -> Self {
        Self::new(&input).unwrap()
    }
}

#[derive(Debug, Deserialize)]
struct ApiErrorResponse {
    error: MoonshotError,
}

#[derive(Debug, Deserialize)]
struct MoonshotError {
    message: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
}

// ================================================================
// Moonshot Completion API
// ================================================================

pub const MOONSHOT_CHAT: &str = "moonshot-v1-128k";

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct MoonshotCompletionRequest {
    model: String,
    pub messages: Vec<openai::Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<openai::ToolDefinition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<crate::providers::openai::completion::ToolChoice>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub additional_params: Option<serde_json::Value>,
}

impl TryFrom<(&str, CompletionRequest)> for MoonshotCompletionRequest {
    type Error = CompletionError;

    fn try_from((model, req): (&str, CompletionRequest)) -> Result<Self, Self::Error> {
        // Build up the order of messages (context, chat_history, prompt)
        let mut partial_history = vec![];
        if let Some(docs) = req.normalized_documents() {
            partial_history.push(docs);
        }
        partial_history.extend(req.chat_history);

        // Add preamble to chat history (if available)
        let mut full_history: Vec<openai::Message> = match &req.preamble {
            Some(preamble) => vec![openai::Message::system(preamble)],
            None => vec![],
        };

        // Convert and extend the rest of the history
        full_history.extend(
            partial_history
                .into_iter()
                .map(message::Message::try_into)
                .collect::<Result<Vec<Vec<openai::Message>>, _>>()?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>(),
        );

        let tool_choice = req
            .tool_choice
            .clone()
            .map(crate::providers::openai::ToolChoice::try_from)
            .transpose()?;

        Ok(Self {
            model: model.to_string(),
            messages: full_history,
            temperature: req.temperature,
            max_tokens: req.max_tokens,
            tools: req
                .tools
                .clone()
                .into_iter()
                .map(openai::ToolDefinition::from)
                .collect::<Vec<_>>(),
            tool_choice,
            additional_params: req.additional_params,
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

impl<T> completion::CompletionModel for CompletionModel<T>
where
    T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
{
    type Response = openai::CompletionResponse;
    type StreamingResponse = openai::StreamingCompletionResponse;

    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model)
    }

    #[cfg_attr(feature = "worker", worker::send)]
    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<openai::CompletionResponse>, CompletionError> {
        let preamble = completion_request.preamble.clone();
        let request =
            MoonshotCompletionRequest::try_from((self.model.as_ref(), completion_request))?;

        tracing::trace!(
            "Moonshot API input: {request}",
            request = serde_json::to_string_pretty(&request).unwrap()
        );

        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat",
                gen_ai.operation.name = "chat",
                gen_ai.provider.name = "moonshot",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = preamble,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.input.messages = serde_json::to_string(&request.messages)?,
                gen_ai.output.messages = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        let body = serde_json::to_vec(&request)?;
        let req = self
            .client
            .post("/chat/completions")?
            .body(body)
            .map_err(http_client::Error::from)?;

        let async_block = async move {
            let response = self.client.send::<_, bytes::Bytes>(req).await?;

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
                        tracing::trace!(
                            target: "rig::completions",
                            "MoonShot completion response: {}",
                            serde_json::to_string_pretty(&response)?
                        );
                        response.try_into()
                    }
                    ApiResponse::Err(err) => Err(CompletionError::ProviderError(err.error.message)),
                }
            } else {
                Err(CompletionError::ProviderError(
                    String::from_utf8_lossy(&response_body).to_string(),
                ))
            }
        };

        async_block.instrument(span).await
    }

    #[cfg_attr(feature = "worker", worker::send)]
    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        let preamble = request.preamble.clone();
        let mut request = MoonshotCompletionRequest::try_from((self.model.as_ref(), request))?;

        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat_streaming",
                gen_ai.operation.name = "chat_streaming",
                gen_ai.provider.name = "moonshot",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = preamble,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.input.messages = serde_json::to_string(&request.messages)?,
                gen_ai.output.messages = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        let params = json_utils::merge(
            request.additional_params.unwrap_or(serde_json::json!({})),
            serde_json::json!({"stream": true, "stream_options": {"include_usage": true} }),
        );

        request.additional_params = Some(params);

        let body = serde_json::to_vec(&request)?;
        let req = self
            .client
            .post("/chat/completions")?
            .body(body)
            .map_err(http_client::Error::from)?;

        send_compatible_streaming_request(self.client.http_client().clone(), req)
            .instrument(span)
            .await
    }
}

#[derive(Default, Debug, Deserialize, Serialize)]
pub enum ToolChoice {
    None,
    #[default]
    Auto,
}

impl TryFrom<message::ToolChoice> for ToolChoice {
    type Error = CompletionError;

    fn try_from(value: message::ToolChoice) -> Result<Self, Self::Error> {
        let res = match value {
            message::ToolChoice::None => Self::None,
            message::ToolChoice::Auto => Self::Auto,
            choice => {
                return Err(CompletionError::ProviderError(format!(
                    "Unsupported tool choice type: {choice:?}"
                )));
            }
        };

        Ok(res)
    }
}
