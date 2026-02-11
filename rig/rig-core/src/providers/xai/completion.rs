//! xAI Completion Integration
//!
//! Uses the xAI Responses API: <https://docs.x.ai/docs/guides/chat>

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use tracing::{Instrument, Level, enabled, info_span};

use super::api::{ApiResponse, Message, ToolDefinition};
use super::client::Client;
use crate::OneOrMany;
use crate::completion::{self, CompletionError, CompletionRequest};
use crate::http_client::HttpClientExt;
use crate::providers::openai::completion::ToolChoice;
use crate::providers::openai::responses_api::streaming::StreamingCompletionResponse;
use crate::providers::openai::responses_api::{Output, ResponsesUsage};
use crate::streaming::StreamingCompletionResponse as BaseStreamingCompletionResponse;

/// xAI completion models as of 2025-06-04
pub const GROK_2_1212: &str = "grok-2-1212";
pub const GROK_2_VISION_1212: &str = "grok-2-vision-1212";
pub const GROK_3: &str = "grok-3";
pub const GROK_3_FAST: &str = "grok-3-fast";
pub const GROK_3_MINI: &str = "grok-3-mini";
pub const GROK_3_MINI_FAST: &str = "grok-3-mini-fast";
pub const GROK_2_IMAGE_1212: &str = "grok-2-image-1212";
pub const GROK_4: &str = "grok-4-0709";

// ================================================================
// Request Types
// ================================================================

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct XAICompletionRequest {
    model: String,
    pub input: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ToolDefinition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoice>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub additional_params: Option<serde_json::Value>,
}

impl TryFrom<(&str, CompletionRequest)> for XAICompletionRequest {
    type Error = CompletionError;

    fn try_from((model, req): (&str, CompletionRequest)) -> Result<Self, Self::Error> {
        let model = req.model.clone().unwrap_or_else(|| model.to_string());
        let mut input: Vec<Message> = req
            .preamble
            .as_ref()
            .map_or_else(Vec::new, |p| vec![Message::system(p)]);

        for msg in req.chat_history {
            let msg: Vec<Message> = msg.try_into()?;
            input.extend(msg);
        }

        let tool_choice = req.tool_choice.map(ToolChoice::try_from).transpose()?;
        let tools = req.tools.into_iter().map(ToolDefinition::from).collect();

        Ok(Self {
            model: model.to_string(),
            input,
            temperature: req.temperature,
            max_output_tokens: req.max_tokens,
            tools,
            tool_choice,
            additional_params: req.additional_params,
        })
    }
}

// ================================================================
// Response Types
// ================================================================

#[derive(Debug, Deserialize, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub model: String,
    pub output: Vec<Output>,
    #[serde(default)]
    pub created: i64,
    #[serde(default)]
    pub object: String,
    #[serde(default)]
    pub status: Option<String>,
    pub usage: Option<ResponsesUsage>,
}

impl TryFrom<CompletionResponse> for completion::CompletionResponse<CompletionResponse> {
    type Error = CompletionError;

    fn try_from(response: CompletionResponse) -> Result<Self, Self::Error> {
        let content: Vec<completion::AssistantContent> = response
            .output
            .iter()
            .cloned()
            .flat_map(<Vec<completion::AssistantContent>>::from)
            .collect();

        let choice = OneOrMany::many(content).map_err(|_| {
            CompletionError::ResponseError("Response contained no output".to_owned())
        })?;

        let usage = response
            .usage
            .as_ref()
            .map(|u| completion::Usage {
                input_tokens: u.input_tokens,
                output_tokens: u.output_tokens,
                total_tokens: u.total_tokens,
                cached_input_tokens: u
                    .input_tokens_details
                    .clone()
                    .map(|x| x.cached_tokens)
                    .unwrap_or_default(),
            })
            .unwrap_or_default();

        Ok(completion::CompletionResponse {
            choice,
            usage,
            raw_response: response,
        })
    }
}

// ================================================================
// Completion Model
// ================================================================

#[derive(Clone)]
pub struct CompletionModel<T = reqwest::Client> {
    pub(crate) client: Client<T>,
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
    type Response = CompletionResponse;
    type StreamingResponse = StreamingCompletionResponse;

    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model)
    }

    async fn completion(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat",
                gen_ai.operation.name = "chat",
                gen_ai.provider.name = "xai",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = tracing::field::Empty,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        span.record("gen_ai.system_instructions", &completion_request.preamble);

        let request =
            XAICompletionRequest::try_from((self.model.to_string().as_ref(), completion_request))?;

        if enabled!(Level::TRACE) {
            tracing::trace!(target: "rig::completions",
                "xAI completion request: {}",
                serde_json::to_string_pretty(&request)?
            );
        }

        let body = serde_json::to_vec(&request)?;
        let req = self
            .client
            .post("/v1/responses")?
            .body(body)
            .map_err(|e| CompletionError::HttpError(e.into()))?;

        async move {
            let response = self.client.send::<_, Bytes>(req).await?;
            let status = response.status();
            let response_body = response.into_body().into_future().await?.to_vec();

            if status.is_success() {
                match serde_json::from_slice::<ApiResponse<CompletionResponse>>(&response_body)? {
                    ApiResponse::Ok(response) => {
                        if enabled!(Level::TRACE) {
                            tracing::trace!(target: "rig::completions",
                                "xAI completion response: {}",
                                serde_json::to_string_pretty(&response)?
                            );
                        }

                        response.try_into()
                    }
                    ApiResponse::Error(error) => {
                        Err(CompletionError::ProviderError(error.message()))
                    }
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
        request: CompletionRequest,
    ) -> Result<BaseStreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        self.stream(request).await
    }
}
