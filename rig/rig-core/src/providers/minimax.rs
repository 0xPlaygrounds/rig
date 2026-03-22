//! MiniMax API client and Rig integration
//!
//! MiniMax provides an OpenAI-compatible Chat Completions API.
//! For more information, see <https://platform.minimax.io/docs/api-reference/text-openai-api>.
//!
//! # Example
//! ```
//! use rig::providers::minimax;
//!
//! let client = minimax::Client::new("YOUR_MINIMAX_API_KEY");
//!
//! let m27 = client.completion_model(minimax::MINIMAX_M2_7);
//! ```

use crate::client::BearerAuth;
use crate::completion::CompletionRequest;
use crate::providers::openai;
use crate::providers::openai::send_compatible_streaming_request;
use crate::streaming::StreamingCompletionResponse;
use crate::{
    client::{
        self, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder, ProviderClient,
    },
    completion::{self, CompletionError},
    http_client::{self, HttpClientExt},
    providers::openai::completion::{Message as OpenAIMessage, ToolChoice, ToolDefinition},
};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use tracing::{Instrument, info_span};

// ================================================================
// Main MiniMax Client
// ================================================================
const MINIMAX_API_BASE_URL: &str = "https://api.minimax.io/v1";

/// MiniMax-M2.7 — latest flagship model with enhanced reasoning and coding.
pub const MINIMAX_M2_7: &str = "MiniMax-M2.7";

/// MiniMax-M2.7-highspeed — high-speed version of M2.7 for low-latency scenarios.
pub const MINIMAX_M2_7_HIGHSPEED: &str = "MiniMax-M2.7-highspeed";

#[derive(Debug, Default, Clone, Copy)]
pub struct MiniMaxExt;

#[derive(Debug, Default, Clone, Copy)]
pub struct MiniMaxBuilder;

type MiniMaxApiKey = BearerAuth;

impl Provider for MiniMaxExt {
    type Builder = MiniMaxBuilder;
    const VERIFY_PATH: &'static str = "/models";
}

impl<H> Capabilities<H> for MiniMaxExt {
    type Completion = Capable<CompletionModel<H>>;
    type Embeddings = Nothing;
    type Transcription = Nothing;
    type ModelListing = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;
    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
}

impl DebugExt for MiniMaxExt {}

impl ProviderBuilder for MiniMaxBuilder {
    type Extension<H>
        = MiniMaxExt
    where
        H: HttpClientExt;
    type ApiKey = MiniMaxApiKey;

    const BASE_URL: &'static str = MINIMAX_API_BASE_URL;

    fn build<H>(
        _builder: &client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: HttpClientExt,
    {
        Ok(MiniMaxExt)
    }
}

pub type Client<H = reqwest::Client> = client::Client<MiniMaxExt, H>;
pub type ClientBuilder<H = reqwest::Client> =
    client::ClientBuilder<MiniMaxBuilder, MiniMaxApiKey, H>;

impl ProviderClient for Client {
    type Input = String;

    /// Create a new MiniMax client from the `MINIMAX_API_KEY` environment variable.
    ///
    /// # Panics
    /// Panics if the environment variable is not set.
    fn from_env() -> Self {
        let api_key = std::env::var("MINIMAX_API_KEY").expect("MINIMAX_API_KEY not set");
        Self::new(&api_key).unwrap()
    }

    fn from_val(input: Self::Input) -> Self {
        Self::new(&input).unwrap()
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
// MiniMax Completion API
// ================================================================

/// Clamp temperature to MiniMax's valid range (0.0, 1.0].
/// MiniMax rejects temperature = 0. Values outside the range are clamped.
fn clamp_temperature(temperature: Option<f64>) -> Option<f64> {
    temperature.map(|t| {
        if t <= 0.0 {
            tracing::warn!("MiniMax does not support temperature <= 0; clamping to 0.01");
            0.01
        } else if t > 1.0 {
            tracing::warn!("MiniMax does not support temperature > 1.0; clamping to 1.0");
            1.0
        } else {
            t
        }
    })
}

#[derive(Debug, Serialize, Deserialize)]
struct MiniMaxCompletionRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ToolDefinition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoice>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    additional_params: Option<serde_json::Value>,
    stream: bool,
}

impl TryFrom<(&str, CompletionRequest)> for MiniMaxCompletionRequest {
    type Error = CompletionError;

    fn try_from((model, req): (&str, CompletionRequest)) -> Result<Self, Self::Error> {
        // MiniMax does not support structured output / response_format
        if req.output_schema.is_some() {
            tracing::warn!(
                "MiniMax does not support structured outputs (response_format); ignoring output_schema"
            );
        }

        let model = req.model.clone().unwrap_or_else(|| model.to_string());

        let mut partial_history = vec![];
        if let Some(docs) = req.normalized_documents() {
            partial_history.push(docs);
        }
        partial_history.extend(req.chat_history);

        let mut full_history: Vec<OpenAIMessage> = match &req.preamble {
            Some(preamble) => vec![OpenAIMessage::system(preamble)],
            None => vec![],
        };

        full_history.extend(
            partial_history
                .into_iter()
                .map(crate::message::Message::try_into)
                .collect::<Result<Vec<Vec<OpenAIMessage>>, _>>()?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>(),
        );

        let tool_choice = req
            .tool_choice
            .clone()
            .map(ToolChoice::try_from)
            .transpose()?;

        Ok(Self {
            model,
            messages: full_history,
            temperature: clamp_temperature(req.temperature),
            tools: req
                .tools
                .clone()
                .into_iter()
                .map(ToolDefinition::from)
                .collect(),
            tool_choice,
            additional_params: req.additional_params,
            stream: false,
        })
    }
}

/// MiniMax completion model.
#[derive(Clone)]
pub struct CompletionModel<T = reqwest::Client> {
    client: Client<T>,
    /// Name of the model (e.g.: MiniMax-M2.7)
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

    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<openai::CompletionResponse>, CompletionError> {
        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat",
                gen_ai.operation.name = "chat",
                gen_ai.provider.name = "minimax",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = tracing::field::Empty,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.usage.cached_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        span.record("gen_ai.system_instructions", &completion_request.preamble);

        let request =
            MiniMaxCompletionRequest::try_from((self.model.as_ref(), completion_request))?;

        if tracing::enabled!(tracing::Level::TRACE) {
            tracing::trace!(target: "rig::completions",
                "MiniMax completion request: {}",
                serde_json::to_string_pretty(&request)?
            );
        }

        let body = serde_json::to_vec(&request)?;
        let req = self
            .client
            .post("/chat/completions")?
            .body(body)
            .map_err(|e| http_client::Error::from(e))?;

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
                            span.record(
                                "gen_ai.usage.cached_tokens",
                                usage
                                    .prompt_tokens_details
                                    .as_ref()
                                    .map(|d| d.cached_tokens)
                                    .unwrap_or(0),
                            );
                        }
                        if tracing::enabled!(tracing::Level::TRACE) {
                            tracing::trace!(target: "rig::completions",
                                "MiniMax completion response: {}",
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
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat_streaming",
                gen_ai.operation.name = "chat_streaming",
                gen_ai.provider.name = "minimax",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = tracing::field::Empty,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.usage.cached_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        span.record("gen_ai.system_instructions", &completion_request.preamble);

        let mut request =
            MiniMaxCompletionRequest::try_from((self.model.as_ref(), completion_request))?;
        request.stream = true;

        if tracing::enabled!(tracing::Level::TRACE) {
            tracing::trace!(target: "rig::completions",
                "MiniMax streaming completion request: {}",
                serde_json::to_string_pretty(&request)?
            );
        }

        let body = serde_json::to_vec(&request)?;
        let req = self
            .client
            .post("/chat/completions")?
            .body(body)
            .map_err(|e| http_client::Error::from(e))?;

        send_compatible_streaming_request(self.client.clone(), req)
            .instrument(span)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clamp_temperature_zero() {
        assert_eq!(clamp_temperature(Some(0.0)), Some(0.01));
    }

    #[test]
    fn test_clamp_temperature_negative() {
        assert_eq!(clamp_temperature(Some(-0.5)), Some(0.01));
    }

    #[test]
    fn test_clamp_temperature_above_max() {
        assert_eq!(clamp_temperature(Some(1.5)), Some(1.0));
    }

    #[test]
    fn test_clamp_temperature_valid() {
        assert_eq!(clamp_temperature(Some(0.7)), Some(0.7));
    }

    #[test]
    fn test_clamp_temperature_max() {
        assert_eq!(clamp_temperature(Some(1.0)), Some(1.0));
    }

    #[test]
    fn test_clamp_temperature_none() {
        assert_eq!(clamp_temperature(None), None);
    }

    #[test]
    fn test_client_initialization() {
        let _client =
            crate::providers::minimax::Client::new("dummy-key").expect("Client::new() failed");
        let _client_from_builder = crate::providers::minimax::Client::builder()
            .api_key("dummy-key")
            .build()
            .expect("Client::builder() failed");
    }

    #[test]
    fn test_model_constants() {
        assert_eq!(MINIMAX_M2_7, "MiniMax-M2.7");
        assert_eq!(MINIMAX_M2_7_HIGHSPEED, "MiniMax-M2.7-highspeed");
    }
}
