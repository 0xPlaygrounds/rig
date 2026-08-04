//! xAI Completion Integration
//!
//! Uses the xAI Responses API: <https://docs.x.ai/docs/guides/chat>

use crate::telemetry::{CompletionOperation, CompletionSpanBuilder, SpanCombinator};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{Instrument, Level, enabled};

use super::api::{ApiResponse, Message, ToolDefinition};
use super::client::Client;
use crate::OneOrMany;
use crate::completion::{self, CompletionError, CompletionRequest};
use crate::http_client::HttpClientExt;
use crate::providers::openai::responses_api::ToolChoice;
use crate::providers::openai::responses_api::{IncompleteDetailsReason, Output, ResponsesUsage};

/// Stable descriptor name recorded on telemetry spans and on every normalized
/// response produced by this provider.
pub(crate) const PROVIDER_NAME: &str = "xai";

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
    pub(super) model: String,
    pub input: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoice>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub additional_params: Option<serde_json::Value>,
}

impl TryFrom<(&str, CompletionRequest)> for XAICompletionRequest {
    type Error = CompletionError;

    fn try_from((model, req): (&str, CompletionRequest)) -> Result<Self, Self::Error> {
        let chat_history = req.chat_history_with_documents();
        if req.output_schema.is_some() {
            tracing::warn!("Structured outputs currently not supported for xAI");
        }
        let model = req.model.clone().unwrap_or_else(|| model.to_string());
        let mut input: Vec<Message> = req
            .preamble
            .as_ref()
            .map_or_else(Vec::new, |p| vec![Message::system(p)]);

        let mut additional_params_payload = req.additional_params.unwrap_or(Value::Null);

        for msg in chat_history {
            let msg: Vec<Message> = msg.try_into()?;
            input.extend(msg);
        }

        let tool_choice = req.tool_choice.map(ToolChoice::try_from).transpose()?;
        let mut additional_tools =
            extract_tools_from_additional_params(&mut additional_params_payload)?;
        let mut tools = req
            .tools
            .into_iter()
            .map(ToolDefinition::from)
            .map(serde_json::to_value)
            .collect::<Result<Vec<_>, _>>()?;
        tools.append(&mut additional_tools);
        let additional_params = if additional_params_payload.is_null() {
            None
        } else {
            Some(additional_params_payload)
        };

        Ok(Self {
            model: model.to_string(),
            input,
            temperature: req.temperature,
            max_output_tokens: req.max_tokens,
            tools,
            tool_choice,
            additional_params,
        })
    }
}

fn extract_tools_from_additional_params(
    additional_params: &mut Value,
) -> Result<Vec<Value>, CompletionError> {
    if let Some(map) = additional_params.as_object_mut()
        && let Some(raw_tools) = map.remove("tools")
    {
        return serde_json::from_value::<Vec<Value>>(raw_tools).map_err(|err| {
            CompletionError::RequestError(
                format!("Invalid xAI `additional_params.tools` payload: {err}").into(),
            )
        });
    }

    Ok(Vec::new())
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
    /// Why an `incomplete` response stopped early. Only the Responses API's
    /// `incomplete` status distinguishes a token-limit truncation from a
    /// filtered one, and it does so here rather than on `status`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub incomplete_details: Option<IncompleteDetailsReason>,
    pub usage: Option<ResponsesUsage>,
}

/// Map an xAI Responses API terminal state onto the normalized vocabulary.
///
/// Mirrors
/// [`responses_api::map_finish_reason`](crate::providers::openai::responses_api)
/// decision for decision — the two APIs share this vocabulary — but reads the
/// status as a string, because xAI's `status` is kept untyped here so a status
/// xAI adds later still deserializes (and still surfaces) instead of failing
/// the whole response.
///
/// The API splits "why did generation end" across two fields: `status` says
/// whether the turn ran to completion, and `incomplete_details.reason` says why
/// it did not. Anything unrecognized is carried verbatim in
/// [`FinishReason::Other`](completion::FinishReason::Other) so it never reads
/// as a natural stop; in-flight statuses report no reason at all.
///
/// Tool-call termination has no distinct status: xAI reports `completed` on a
/// turn whose output carries `function_call` items. That upgrade to
/// [`FinishReason::ToolCalls`](completion::FinishReason::ToolCalls) is applied
/// once, centrally, by
/// [`completion::CompletionResponse::with_optional_finish_reason`].
pub(crate) fn map_finish_reason(
    status: &str,
    incomplete_details: Option<&IncompleteDetailsReason>,
) -> Option<completion::FinishReason> {
    match status {
        "completed" => Some(completion::FinishReason::Stop),
        "incomplete" => Some(
            match incomplete_details
                .map(|details| details.reason.as_str())
                .filter(|reason| !reason.is_empty())
            {
                Some("max_output_tokens") => completion::FinishReason::Length,
                Some("content_filter") => completion::FinishReason::ContentFilter,
                Some(other) => completion::FinishReason::Other(other.to_owned()),
                // Incomplete without a stated reason: the status itself is all
                // the provider told us.
                None => completion::FinishReason::Other(status.to_owned()),
            },
        ),
        // The turn has not terminated, so there is genuinely no reason yet.
        "in_progress" | "queued" => None,
        other => Some(completion::FinishReason::Other(other.to_owned())),
    }
}

impl TryFrom<CompletionResponse> for completion::CompletionResponse {
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
            .map(crate::completion::Usage::from)
            .unwrap_or_default();
        let message_id = response.output.iter().find_map(|item| match item {
            Output::Message(message) => Some(message.id.clone()),
            _ => None,
        });
        let finish_reason = response
            .status
            .as_deref()
            .filter(|status| !status.is_empty())
            .and_then(|status| map_finish_reason(status, response.incomplete_details.as_ref()));

        Ok(
            completion::CompletionResponse::new(choice, usage, PROVIDER_NAME)
                .with_optional_message_id(message_id)
                .with_optional_response_id(Some(response.id.as_str()).filter(|id| !id.is_empty()))
                .with_model(response.model)
                .with_optional_finish_reason(finish_reason),
        )
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

impl<T> CompletionModel<T>
where
    T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
{
    /// Execute a completion and return xAI's own wire response.
    ///
    /// This is the escape hatch for provider-specific fields rig does not
    /// normalize. It shares the request builder, transport, telemetry, and
    /// error handling with
    /// [`CompletionModel::completion`](completion::CompletionModel::completion),
    /// which calls it and then applies the provider-local mapping — one
    /// network request either way.
    pub async fn raw_completion(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<CompletionResponse, CompletionError> {
        let system_instructions = completion_request.preamble.clone();
        let record_telemetry_content = completion_request.record_telemetry_content;
        let request =
            XAICompletionRequest::try_from((self.model.to_string().as_ref(), completion_request))?;
        let span =
            CompletionSpanBuilder::new(PROVIDER_NAME, &request.model, CompletionOperation::Chat)
                .system_instructions(system_instructions.as_deref(), record_telemetry_content)
                .build();

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
                        let span = tracing::Span::current();
                        span.record("gen_ai.response.id", response.id.as_str());
                        span.record("gen_ai.response.model", response.model.as_str());
                        if let Some(usage) = &response.usage {
                            span.record_token_usage(&crate::completion::Usage::from(usage));
                        }

                        if enabled!(Level::TRACE) {
                            tracing::trace!(target: "rig::completions",
                                "xAI completion response: {}",
                                serde_json::to_string_pretty(&response)?
                            );
                        }

                        Ok(response)
                    }
                    ApiResponse::Error(error) => {
                        tracing::warn!(message = %error.message(), "provider returned an error response");
                        Err(CompletionError::from_http_response(
                            status,
                            String::from_utf8_lossy(&response_body),
                        ))
                    }
                }
            } else {
                Err(CompletionError::from_http_response(
                    status,
                    String::from_utf8_lossy(&response_body),
                ))
            }
        }
        .instrument(span)
        .await
    }
}

impl<T> completion::CompletionModel for CompletionModel<T>
where
    T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
{
    async fn completion(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<completion::CompletionResponse, CompletionError> {
        self.raw_completion(completion_request).await?.try_into()
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<crate::streaming::StreamingCompletionResponse, CompletionError> {
        CompletionModel::stream(self, request).await
    }
}

impl<H> crate::client::ConstructCompletionModel<Client<H>> for CompletionModel<H>
where
    Client<H>: Clone,
{
    fn construct(client: &Client<H>, model: String) -> Self {
        Self::new(client.clone(), model)
    }
}

#[cfg(test)]
mod tests {
    use super::XAICompletionRequest;
    use crate::OneOrMany;
    use crate::completion::request::Document;
    use crate::completion::{CompletionRequest, CompletionRequestBuilder, Message, ToolDefinition};
    use crate::message::ToolChoice;
    use crate::test_utils::MockCompletionModel;

    #[test]
    fn xai_request_includes_normalized_documents() {
        let request =
            CompletionRequestBuilder::new(MockCompletionModel::default(), "What is glarb-glarb?")
                .message(Message::system("Use the provided context."))
                .document(Document {
                    id: "doc_1".to_string(),
                    text: "Definition of glarb-glarb: an ancient tool.".to_string(),
                    additional_props: Default::default(),
                })
                .build();

        let xai_request = XAICompletionRequest::try_from(("grok-4-0709", request))
            .expect("request conversion should succeed");
        let serialized = serde_json::to_value(xai_request).expect("serialization should succeed");
        let input = serialized["input"]
            .as_array()
            .expect("xAI request input should be an array");

        assert!(
            input
                .iter()
                .any(|message| message.to_string().contains("glarb-glarb")),
            "normalized documents should be forwarded into xAI input"
        );
    }

    #[test]
    fn xai_direct_request_keeps_documents_after_system_messages() {
        let request = CompletionRequest {
            model: None,
            preamble: None,
            chat_history: OneOrMany::many(vec![
                Message::system("System prompt"),
                Message::assistant("Earlier assistant turn"),
                Message::system("Mid-conversation instruction"),
                Message::user("What is glarb-glarb?"),
            ])
            .unwrap(),
            documents: vec![Document {
                id: "doc_1".to_string(),
                text: "Definition of glarb-glarb: an ancient tool.".to_string(),
                additional_props: Default::default(),
            }],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        };

        let xai_request = XAICompletionRequest::try_from(("grok-4-0709", request))
            .expect("request conversion should succeed");
        let serialized = serde_json::to_value(xai_request).expect("serialization should succeed");
        let input = serialized["input"]
            .as_array()
            .expect("xAI request input should be an array");

        assert_eq!(input.len(), 5);
        assert_eq!(input[0]["role"], "system");
        assert_eq!(input[1]["role"], "user");
        assert!(
            input[1].to_string().contains("<file id: doc_1>"),
            "document input should follow leading system input: {input:?}"
        );
        assert_eq!(input[2]["role"], "assistant");
        assert_eq!(input[3]["role"], "system");
        assert_eq!(input[4]["role"], "user");
        assert_eq!(
            input
                .iter()
                .filter(|message| message.to_string().contains("<file id: doc_1>"))
                .count(),
            1,
            "document input should appear exactly once: {input:?}"
        );
    }

    #[test]
    fn xai_request_uses_responses_tool_choice_for_specific_tool() {
        let request = CompletionRequestBuilder::new(MockCompletionModel::default(), "Use a tool.")
            .tool(ToolDefinition {
                name: "alpha".to_string(),
                description: "Alpha tool".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                }),
            })
            .tool(ToolDefinition {
                name: "beta".to_string(),
                description: "Beta tool".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                }),
            })
            .tool_choice(ToolChoice::Specific {
                function_names: vec!["beta".to_string()],
            })
            .build();

        let xai_request = XAICompletionRequest::try_from(("grok-4.3", request))
            .expect("xAI Responses API should support specific tool choice");
        let serialized = serde_json::to_value(xai_request).expect("serialization should succeed");

        assert_eq!(
            serialized["tool_choice"],
            serde_json::json!({"type": "function", "name": "beta"})
        );
    }

    #[test]
    fn xai_response_preserves_message_id_and_reasoning_token_usage() {
        let raw: super::CompletionResponse = serde_json::from_value(serde_json::json!({
            "id": "resp_123",
            "model": "grok-4.3",
            "output": [
                {
                    "type": "reasoning",
                    "id": "rs_123",
                    "summary": [{ "type": "summary_text", "text": "thinking" }],
                    "status": "completed"
                },
                {
                    "type": "message",
                    "id": "msg_123",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        { "type": "output_text", "text": "done", "annotations": [] }
                    ]
                }
            ],
            "usage": {
                "input_tokens": 10,
                "input_tokens_details": { "cached_tokens": 3 },
                "output_tokens": 8,
                "output_tokens_details": { "reasoning_tokens": 5 },
                "total_tokens": 18
            }
        }))
        .expect("fixture should deserialize");

        let converted = crate::completion::CompletionResponse::try_from(raw)
            .expect("xAI response should convert");

        assert_eq!(converted.message_id.as_deref(), Some("msg_123"));
        assert_eq!(converted.usage.input_tokens, 10);
        assert_eq!(converted.usage.cached_input_tokens, 3);
        assert_eq!(converted.usage.output_tokens, 8);
        assert_eq!(converted.usage.reasoning_tokens, 5);
        assert_eq!(converted.provider, "xai");
        // The provider-reported model, not the one the handle was created with.
        assert_eq!(converted.model.as_deref(), Some("grok-4.3"));
        // The fixture reports no top-level status, so no reason is invented.
        assert_eq!(converted.finish_reason(), None);
    }

    #[test]
    fn xai_finish_reasons_map_and_preserve_unknown_values() {
        use crate::completion::FinishReason;
        use crate::providers::openai::responses_api::IncompleteDetailsReason;

        let incomplete = |reason: &str| IncompleteDetailsReason {
            reason: reason.to_string(),
        };

        assert_eq!(
            super::map_finish_reason("completed", None),
            Some(FinishReason::Stop)
        );
        assert_eq!(
            super::map_finish_reason("incomplete", Some(&incomplete("max_output_tokens"))),
            Some(FinishReason::Length)
        );
        assert_eq!(
            super::map_finish_reason("incomplete", Some(&incomplete("content_filter"))),
            Some(FinishReason::ContentFilter)
        );
        // An unknown incomplete reason survives verbatim rather than being
        // flattened into a natural stop.
        assert_eq!(
            super::map_finish_reason("incomplete", Some(&incomplete("moderation_hold"))),
            Some(FinishReason::Other("moderation_hold".to_string()))
        );
        assert_eq!(
            super::map_finish_reason("incomplete", None),
            Some(FinishReason::Other("incomplete".to_string()))
        );
        assert_eq!(
            super::map_finish_reason("failed", None),
            Some(FinishReason::Other("failed".to_string()))
        );
        // A terminal status xAI has not documented is reported, not smoothed
        // into a natural stop.
        assert_eq!(
            super::map_finish_reason("throttled", None),
            Some(FinishReason::Other("throttled".to_string()))
        );
        // In-flight statuses genuinely have no terminal reason yet.
        assert_eq!(super::map_finish_reason("in_progress", None), None);
        assert_eq!(super::map_finish_reason("queued", None), None);
    }

    #[test]
    fn xai_response_reports_length_for_truncated_output() {
        let raw: super::CompletionResponse = serde_json::from_value(serde_json::json!({
            "id": "resp_trunc",
            "model": "grok-4-0709",
            "status": "incomplete",
            "incomplete_details": { "reason": "max_output_tokens" },
            "output": [
                {
                    "type": "message",
                    "id": "msg_trunc",
                    "role": "assistant",
                    "status": "incomplete",
                    "content": [
                        { "type": "output_text", "text": "partial", "annotations": [] }
                    ]
                }
            ]
        }))
        .expect("fixture should deserialize");

        let converted = crate::completion::CompletionResponse::try_from(raw)
            .expect("xAI response should convert");

        assert_eq!(
            converted.finish_reason(),
            Some(crate::completion::FinishReason::Length)
        );
    }

    #[test]
    fn xai_completed_response_with_tool_call_reports_tool_calls() {
        // xAI reports `completed` even when the turn ended to call a tool;
        // the reconciliation on the normalized response upgrades it.
        let raw: super::CompletionResponse = serde_json::from_value(serde_json::json!({
            "id": "resp_tool",
            "model": "grok-4-0709",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "example_tool",
                    "arguments": "{}",
                    "status": "completed"
                }
            ]
        }))
        .expect("fixture should deserialize");

        let converted = crate::completion::CompletionResponse::try_from(raw)
            .expect("xAI response should convert");

        assert_eq!(
            converted.finish_reason(),
            Some(crate::completion::FinishReason::ToolCalls)
        );
    }

    #[tokio::test]
    async fn completion_non_success_preserves_status_and_body() {
        use crate::client::CompletionClient;
        use crate::completion::{CompletionError, CompletionModel as _};
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"error":"boom","code":"503"}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
        let client = crate::providers::xai::Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.completion_model(crate::providers::xai::completion::GROK_4);
        let request = model.completion_request("hello").build();

        let error = model
            .completion(request)
            .await
            .expect_err("should fail with non-success status");

        assert!(matches!(error, CompletionError::HttpError(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::SERVICE_UNAVAILABLE)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }

    #[tokio::test]
    async fn completion_2xx_error_envelope_preserves_status_and_body() {
        use crate::client::CompletionClient;
        use crate::completion::{CompletionError, CompletionModel as _};
        use crate::test_utils::RecordingHttpClient;

        // Deserializes to `ApiResponse::Error(ApiError { error, code })` on a 200 OK.
        let body = r#"{"error":"boom","code":"503"}"#;
        let http_client = RecordingHttpClient::new(body);
        let client = crate::providers::xai::Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.completion_model(crate::providers::xai::completion::GROK_4);
        let request = model.completion_request("hello").build();

        let error = model
            .completion(request)
            .await
            .expect_err("should fail with provider error envelope");

        match &error {
            CompletionError::ProviderResponse(stored) => {
                assert_eq!(stored.body, body);
                assert_eq!(stored.status, Some(http::StatusCode::OK));
            }
            other => panic!("expected ProviderResponse, got {other:?}"),
        }
    }
}
