// ================================================================
// Copilot Completion API — OpenAI-derived with relaxed response types
// ================================================================

use super::client::Client;
use crate::completion::{CompletionError, CompletionRequest as CoreCompletionRequest};
use crate::http_client::{self, HttpClientExt};
use crate::providers::openai;
use crate::telemetry::{ProviderResponseExt, SpanCombinator};
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use crate::{OneOrMany, completion};
use serde::{Deserialize, Serialize};
use tracing::{Instrument, Level, enabled, info_span};

// Re-export OpenAI message types — the request wire format is identical.
pub use openai::completion::streaming::StreamingCompletionResponse;
pub use openai::completion::{
    AssistantContent, CompletionRequest, Message, OpenAIRequestParams, ToolCall, ToolDefinition,
    ToolType, Usage,
};

// Re-export a selection of common model names from OpenAI — most are also
// available on the Copilot API.  Because the available set varies by GitHub
// Enterprise instance, users should query `/models` or pass the model name
// as a plain string rather than relying on these constants alone.
pub use openai::completion::{GPT_4_1, GPT_4_1_MINI, GPT_4_1_NANO, GPT_4O, GPT_4O_MINI};

// Well-known non-OpenAI models available through the Copilot API.
// This list is intentionally non-exhaustive — new models are added
// regularly and availability depends on your Copilot plan / GHE config.

/// `claude-sonnet-4` completion model (Anthropic, via Copilot)
pub const CLAUDE_SONNET_4: &str = "claude-sonnet-4";
/// `claude-3.5-sonnet` completion model (Anthropic, via Copilot)
pub const CLAUDE_3_5_SONNET: &str = "claude-3.5-sonnet";
/// `gemini-2.0-flash-001` completion model (Google, via Copilot)
pub const GEMINI_2_0_FLASH: &str = "gemini-2.0-flash-001";
/// `o3-mini` reasoning model (OpenAI, via Copilot)
pub const O3_MINI: &str = "o3-mini";

// ================================================================
// Relaxed response types
// ================================================================

/// Copilot API completion response.
///
/// Identical to the standard OpenAI `CompletionResponse` except that the
/// `object` and `created` fields are `Option` — the Copilot API and some
/// Azure-flavored endpoints omit them.
#[derive(Debug, Deserialize, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    /// Omitted by the Copilot API; always `"chat.completion"` on standard OpenAI.
    #[serde(default)]
    pub object: Option<String>,
    /// Omitted by the Copilot API.
    #[serde(default)]
    pub created: Option<u64>,
    pub model: String,
    pub system_fingerprint: Option<String>,
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
}

/// A single completion choice.
///
/// The `finish_reason` field is `Option<String>` because some Copilot-hosted
/// models (notably Claude) may omit it or use a different naming convention.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Choice {
    #[serde(default)]
    pub index: usize,
    pub message: Message,
    pub logprobs: Option<serde_json::Value>,
    /// May be absent for Claude models routed through Copilot.
    #[serde(default)]
    pub finish_reason: Option<String>,
}

// ================================================================
// Trait implementations for the rig completion pipeline
// ================================================================

impl TryFrom<CompletionResponse> for completion::CompletionResponse<CompletionResponse> {
    type Error = CompletionError;

    fn try_from(response: CompletionResponse) -> Result<Self, Self::Error> {
        let choice = response.choices.first().ok_or_else(|| {
            CompletionError::ResponseError("Response contained no choices".to_owned())
        })?;

        let content = match &choice.message {
            Message::Assistant {
                content,
                tool_calls,
                ..
            } => {
                let mut content = content
                    .iter()
                    .filter_map(|c| {
                        let s = match c {
                            AssistantContent::Text { text } => text,
                            AssistantContent::Refusal { refusal } => refusal,
                        };
                        if s.is_empty() {
                            None
                        } else {
                            Some(completion::AssistantContent::text(s))
                        }
                    })
                    .collect::<Vec<_>>();

                content.extend(
                    tool_calls
                        .iter()
                        .map(|call| {
                            completion::AssistantContent::tool_call(
                                &call.id,
                                &call.function.name,
                                call.function.arguments.clone(),
                            )
                        })
                        .collect::<Vec<_>>(),
                );
                Ok(content)
            }
            _ => Err(CompletionError::ResponseError(
                "Response did not contain a valid message or tool call".into(),
            )),
        }?;

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
                cached_input_tokens: usage
                    .prompt_tokens_details
                    .as_ref()
                    .map(|d| d.cached_tokens as u64)
                    .unwrap_or(0),
                cache_creation_input_tokens: 0,
            })
            .unwrap_or_default();

        Ok(completion::CompletionResponse {
            choice,
            usage,
            raw_response: response,
            message_id: None,
        })
    }
}

impl ProviderResponseExt for CompletionResponse {
    type OutputMessage = Choice;
    type Usage = Usage;

    fn get_response_id(&self) -> Option<String> {
        Some(self.id.to_owned())
    }

    fn get_response_model_name(&self) -> Option<String> {
        Some(self.model.to_owned())
    }

    fn get_output_messages(&self) -> Vec<Self::OutputMessage> {
        self.choices.clone()
    }

    fn get_text_response(&self) -> Option<String> {
        let Message::User { ref content, .. } = self.choices.last()?.message.clone() else {
            return None;
        };

        let openai::UserContent::Text { text } = content.first() else {
            return None;
        };

        Some(text)
    }

    fn get_usage(&self) -> Option<Self::Usage> {
        self.usage.clone()
    }
}

// ================================================================
// Error response — Copilot may return a different shape
// ================================================================

#[derive(Debug, Deserialize)]
pub struct ApiErrorResponse {
    /// Standard `"message"` field.
    #[serde(default)]
    pub message: Option<String>,
    /// Some Copilot errors surface the text under `"error"` instead.
    #[serde(default)]
    pub error: Option<String>,
}

impl ApiErrorResponse {
    pub fn error_message(&self) -> &str {
        self.message
            .as_deref()
            .or(self.error.as_deref())
            .unwrap_or("unknown error")
    }
}

impl From<ApiErrorResponse> for CompletionError {
    fn from(err: ApiErrorResponse) -> Self {
        CompletionError::ProviderError(err.error_message().to_string())
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
}

// ================================================================
// Completion model
// ================================================================

#[derive(Clone)]
pub struct CompletionModel<T = reqwest::Client> {
    pub(crate) client: Client<T>,
    pub model: String,
}

impl<T> CompletionModel<T>
where
    T: Default + std::fmt::Debug + Clone + 'static,
{
    pub fn new(client: Client<T>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }
}

impl<T> completion::CompletionModel for CompletionModel<T>
where
    T: HttpClientExt
        + Default
        + std::fmt::Debug
        + Clone
        + WasmCompatSend
        + WasmCompatSync
        + 'static,
{
    type Response = CompletionResponse;
    type StreamingResponse = StreamingCompletionResponse;

    type Client = super::Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model)
    }

    async fn completion(
        &self,
        completion_request: CoreCompletionRequest,
    ) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat",
                gen_ai.operation.name = "chat",
                gen_ai.provider.name = "copilot",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = &completion_request.preamble,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        let request = CompletionRequest::try_from(OpenAIRequestParams {
            model: self.model.to_owned(),
            request: completion_request,
            strict_tools: false,
            tool_result_array_content: false,
        })?;

        if enabled!(Level::TRACE) {
            tracing::trace!(
                target: "rig::completions",
                "Copilot completion request: {}",
                serde_json::to_string_pretty(&request)?
            );
        }

        let body = serde_json::to_vec(&request)?;

        let req = self
            .client
            .post("/chat/completions")?
            .body(body)
            .map_err(|e| CompletionError::HttpError(e.into()))?;

        async move {
            let response = self.client.send(req).await?;

            if response.status().is_success() {
                let text = http_client::text(response).await?;

                match serde_json::from_str::<ApiResponse<CompletionResponse>>(&text)? {
                    ApiResponse::Ok(response) => {
                        let span = tracing::Span::current();
                        span.record_response_metadata(&response);
                        span.record_token_usage(&response.usage);

                        if enabled!(Level::TRACE) {
                            tracing::trace!(
                                target: "rig::completions",
                                "Copilot completion response: {}",
                                serde_json::to_string_pretty(&response)?
                            );
                        }

                        response.try_into()
                    }
                    ApiResponse::Err(err) => Err(CompletionError::ProviderError(
                        err.error_message().to_string(),
                    )),
                }
            } else {
                let text = http_client::text(response).await?;
                Err(CompletionError::ProviderError(text))
            }
        }
        .instrument(span)
        .await
    }

    async fn stream(
        &self,
        _completion_request: CoreCompletionRequest,
    ) -> Result<
        crate::streaming::StreamingCompletionResponse<StreamingCompletionResponse>,
        CompletionError,
    > {
        // Streaming reuses the OpenAI streaming implementation since the SSE
        // format is identical on the Copilot API.  For now we delegate to the
        // underlying OpenAI streaming path — the relaxed response fields only
        // affect the non-streaming endpoint.
        Err(CompletionError::ResponseError(
            "Copilot streaming not yet implemented — use non-streaming completion".to_string(),
        ))
    }
}

// ================================================================
// Tests
// ================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialize_standard_openai_response() {
        let json = r#"{
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello!"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }"#;

        let response: CompletionResponse =
            serde_json::from_str(json).expect("standard OpenAI response should deserialize");
        assert_eq!(response.id, "chatcmpl-abc123");
        assert_eq!(response.object, Some("chat.completion".to_string()));
        assert_eq!(response.created, Some(1700000000));
        assert_eq!(response.model, "gpt-4o");
        assert_eq!(response.choices.len(), 1);
        assert_eq!(response.choices[0].finish_reason, Some("stop".to_string()));
    }

    #[test]
    fn deserialize_copilot_response_without_object_and_created() {
        let json = r#"{
            "id": "chatcmpl-xyz789",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Review complete."
                },
                "finish_reason": "stop",
                "content_filter_results": {}
            }],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30
            },
            "prompt_filter_results": [{"prompt_index": 0}]
        }"#;

        let response: CompletionResponse =
            serde_json::from_str(json).expect("Copilot response should deserialize");
        assert_eq!(response.id, "chatcmpl-xyz789");
        assert_eq!(response.object, None);
        assert_eq!(response.created, None);
        assert_eq!(response.model, "gpt-4o");
        assert_eq!(response.choices.len(), 1);
    }

    #[test]
    fn deserialize_copilot_response_without_finish_reason() {
        let json = r#"{
            "id": "chatcmpl-claude-001",
            "model": "claude-3.5-sonnet",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Here is my analysis."
                }
            }],
            "usage": {
                "prompt_tokens": 50,
                "total_tokens": 80
            }
        }"#;

        let response: CompletionResponse =
            serde_json::from_str(json).expect("Claude-via-Copilot response should deserialize");
        assert_eq!(response.model, "claude-3.5-sonnet");
        assert_eq!(response.choices[0].finish_reason, None);
        assert_eq!(response.choices[0].index, 0); // default
    }

    #[test]
    fn error_response_with_message_field() {
        let json = r#"{"message": "rate limit exceeded"}"#;
        let err: ApiErrorResponse = serde_json::from_str(json).unwrap();
        assert_eq!(err.error_message(), "rate limit exceeded");
    }

    #[test]
    fn error_response_with_error_field() {
        let json = r#"{"error": "model not found"}"#;
        let err: ApiErrorResponse = serde_json::from_str(json).unwrap();
        assert_eq!(err.error_message(), "model not found");
    }
}
