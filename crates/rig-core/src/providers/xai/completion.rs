//! xAI Completion Integration
//!
//! Uses the xAI Responses API: <https://docs.x.ai/docs/guides/chat>

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::api::{Message, ToolDefinition};
use crate::OneOrMany;
use crate::completion::{self, CompletionError, CompletionRequest};
use crate::providers::openai::responses_api::ToolChoice;
use crate::providers::openai::responses_api::{Output, ResponsesUsage};

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
        let mut input: Vec<Message> = Vec::new();

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

/// Merge the top-level `stream: true` flag into `request`'s
/// `additional_params` — the single place the xAI streaming flag is applied,
/// shared by the trait streaming path and the data-oriented
/// [`super::functions`] face.
pub(super) fn apply_stream_flag(request: &mut XAICompletionRequest) {
    let params = crate::json_utils::merge(
        request
            .additional_params
            .take()
            .unwrap_or(serde_json::json!({})),
        serde_json::json!({"stream": true}),
    );
    request.additional_params = Some(params);
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

/// Convert an xAI Responses usage payload into the normalized [`completion::Usage`].
pub(super) fn usage_from_responses(usage: &ResponsesUsage) -> completion::Usage {
    completion::Usage {
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        total_tokens: usage.total_tokens,
        cached_input_tokens: usage
            .input_tokens_details
            .as_ref()
            .map(|details| details.cached_tokens)
            .unwrap_or(0),
        cache_creation_input_tokens: 0,
        tool_use_prompt_tokens: 0,
        reasoning_tokens: usage
            .output_tokens_details
            .as_ref()
            .map(|details| details.reasoning_tokens)
            .unwrap_or(0),
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
            .map(usage_from_responses)
            .unwrap_or_default();
        let message_id = response.output.iter().find_map(|item| match item {
            Output::Message(message) => Some(message.id.clone()),
            _ => None,
        });

        let mut converted = completion::CompletionResponse::new(choice, usage, "xai")
            .with_model(response.model.clone());
        if let Some(message_id) = message_id {
            converted = converted.with_message_id(message_id);
        }
        Ok(converted)
    }
}

#[cfg(test)]
mod tests {
    use super::XAICompletionRequest;
    use crate::OneOrMany;
    use crate::completion::request::Document;
    use crate::completion::{CompletionRequest, Message, ToolDefinition};
    use crate::message::ToolChoice;

    #[test]
    fn xai_request_includes_normalized_documents() {
        let request = CompletionRequest::builder("What is glarb-glarb?")
            .messages(vec![Message::system("Use the provided context.")])
            .documents(vec![Document {
                id: "doc_1".to_string(),
                text: "Definition of glarb-glarb: an ancient tool.".to_string(),
                additional_props: Default::default(),
            }])
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
        let request = CompletionRequest::builder("Use a tool.")
            .tools(vec![
                ToolDefinition {
                    name: "alpha".to_string(),
                    description: "Alpha tool".to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {},
                        "required": []
                    }),
                },
                ToolDefinition {
                    name: "beta".to_string(),
                    description: "Beta tool".to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {},
                        "required": []
                    }),
                },
            ])
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
    }

    #[tokio::test]
    async fn completion_non_success_preserves_status_and_body() {
        use crate::completion::CompletionError;
        use crate::providers::xai::functions;
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"error":"boom","code":"503"}"#;
        let rt = crate::http_runtime::HttpRuntime::recording(
            RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body),
        );
        let cfg = functions::Config::new(super::GROK_4).with_api_key("test-key");

        let error = functions::complete(&cfg, &rt, CompletionRequest::from_prompt("hello"))
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
        use crate::completion::CompletionError;
        use crate::providers::xai::functions;
        use crate::test_utils::RecordingHttpClient;

        // Deserializes to `ApiResponse::Error(ApiError { error, code })` on a 200 OK.
        let body = r#"{"error":"boom","code":"503"}"#;
        let rt = crate::http_runtime::HttpRuntime::recording(RecordingHttpClient::new(body));
        let cfg = functions::Config::new(super::GROK_4).with_api_key("test-key");

        let error = functions::complete(&cfg, &rt, CompletionRequest::from_prompt("hello"))
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
