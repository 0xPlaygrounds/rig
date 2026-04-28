use http::Request;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::info_span;

use crate::completion::{CompletionError, CompletionRequest, GetTokenUsage};
use crate::http_client::HttpClientExt;
use crate::json_utils;
use crate::providers::internal::openai_chat_completions_compatible::{
    self, CompatibleChoiceData, CompatibleChunk, CompatibleFinishReason, CompatibleStreamProfile,
    CompatibleToolCallChunk,
};
use crate::providers::openrouter::{
    OpenRouterRequestParams, OpenrouterCompletionRequest, ReasoningDetails,
};
use crate::streaming;

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct StreamingCompletionResponse {
    pub usage: Usage,
}

impl GetTokenUsage for StreamingCompletionResponse {
    fn token_usage(&self) -> Option<crate::completion::Usage> {
        self.usage.token_usage()
    }
}

#[derive(Deserialize, Debug, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    ToolCalls,
    Stop,
    Error,
    ContentFilter,
    Length,
    #[serde(untagged)]
    Other(String),
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct StreamingChoice {
    pub finish_reason: Option<FinishReason>,
    pub native_finish_reason: Option<String>,
    pub logprobs: Option<Value>,
    pub index: usize,
    pub delta: StreamingDelta,
}

#[derive(Deserialize, Debug)]
struct StreamingFunction {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct StreamingToolCall {
    pub index: usize,
    pub id: Option<String>,
    pub r#type: Option<String>,
    pub function: StreamingFunction,
}

impl From<&StreamingToolCall> for CompatibleToolCallChunk {
    fn from(value: &StreamingToolCall) -> Self {
        Self {
            index: value.index,
            id: value.id.clone(),
            name: value.function.name.clone(),
            arguments: value.function.arguments.clone(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl GetTokenUsage for Usage {
    fn token_usage(&self) -> Option<crate::completion::Usage> {
        Some(crate::providers::internal::completion_usage(
            self.prompt_tokens as u64,
            self.completion_tokens as u64,
            self.total_tokens as u64,
            0,
        ))
    }
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct ErrorResponse {
    pub code: i32,
    pub message: String,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct StreamingDelta {
    pub role: Option<String>,
    pub content: Option<String>,
    #[serde(default, deserialize_with = "json_utils::null_or_vec")]
    pub tool_calls: Vec<StreamingToolCall>,
    pub reasoning: Option<String>,
    #[serde(default, deserialize_with = "json_utils::null_or_vec")]
    pub reasoning_details: Vec<ReasoningDetails>,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct StreamingCompletionChunk {
    id: String,
    model: String,
    choices: Vec<StreamingChoice>,
    usage: Option<Usage>,
    error: Option<ErrorResponse>,
}

impl<T> super::CompletionModel<T>
where
    T: HttpClientExt + Clone + std::fmt::Debug + Default + 'static,
{
    pub(crate) async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<streaming::StreamingCompletionResponse<StreamingCompletionResponse>, CompletionError>
    {
        let request_model = completion_request
            .model
            .clone()
            .unwrap_or_else(|| self.model.clone());
        let preamble = completion_request.preamble.clone();
        let mut request = OpenrouterCompletionRequest::try_from(OpenRouterRequestParams {
            model: request_model.as_ref(),
            request: completion_request,
            strict_tools: self.strict_tools,
        })?;

        let params = json_utils::merge(
            request.additional_params.unwrap_or(serde_json::json!({})),
            serde_json::json!({"stream": true }),
        );

        request.additional_params = Some(params);

        let body = serde_json::to_vec(&request)?;

        let req = self
            .client
            .post("/chat/completions")?
            .body(body)
            .map_err(|x| CompletionError::HttpError(x.into()))?;

        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat_streaming",
                gen_ai.operation.name = "chat_streaming",
                gen_ai.provider.name = "openrouter",
                gen_ai.request.model = &request_model,
                gen_ai.system_instructions = preamble,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.usage.cache_read.input_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        tracing::Instrument::instrument(
            send_compatible_streaming_request(self.client.clone(), req),
            span,
        )
        .await
    }
}

#[derive(Clone, Copy)]
struct OpenRouterCompatibleProfile;

impl CompatibleStreamProfile for OpenRouterCompatibleProfile {
    type Usage = Usage;
    type Detail = ReasoningDetails;
    type FinalResponse = StreamingCompletionResponse;

    fn normalize_chunk(
        &self,
        data: &str,
    ) -> Result<Option<CompatibleChunk<Self::Usage, Self::Detail>>, CompletionError> {
        let data = match serde_json::from_str::<StreamingCompletionChunk>(data) {
            Ok(data) => data,
            Err(error) => {
                tracing::error!(?error, message = data, "Failed to parse SSE message");
                return Ok(None);
            }
        };

        Ok(Some(
            openai_chat_completions_compatible::normalize_first_choice_chunk(
                Some(data.id),
                Some(data.model),
                data.usage,
                &data.choices,
                |choice| CompatibleChoiceData {
                    finish_reason: if choice.finish_reason == Some(FinishReason::ToolCalls) {
                        CompatibleFinishReason::ToolCalls
                    } else {
                        CompatibleFinishReason::Other
                    },
                    text: choice.delta.content.clone(),
                    reasoning: choice.delta.reasoning.clone(),
                    tool_calls: openai_chat_completions_compatible::tool_call_chunks(
                        &choice.delta.tool_calls,
                    ),
                    details: choice.delta.reasoning_details.clone(),
                },
            ),
        ))
    }

    fn build_final_response(&self, usage: Self::Usage) -> Self::FinalResponse {
        StreamingCompletionResponse { usage }
    }

    fn decorate_tool_call(
        &self,
        detail: &Self::Detail,
        tool_calls: &mut std::collections::HashMap<usize, crate::streaming::RawStreamingToolCall>,
    ) {
        if let ReasoningDetails::Encrypted { id, data, .. } = detail
            && let Some(id) = id
            && let Some(tool_call) = tool_calls
                .values_mut()
                .find(|tool_call| tool_call.id.eq(id))
            && let Ok(additional_params) = serde_json::to_value(detail)
        {
            tool_call.signature = Some(data.clone());
            tool_call.additional_params = Some(additional_params);
        }
    }
}

pub async fn send_compatible_streaming_request<T>(
    http_client: T,
    req: Request<Vec<u8>>,
) -> Result<streaming::StreamingCompletionResponse<StreamingCompletionResponse>, CompletionError>
where
    T: HttpClientExt + Clone + 'static,
{
    openai_chat_completions_compatible::send_compatible_streaming_request(
        http_client,
        req,
        OpenRouterCompatibleProfile,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http_client::mock::MockStreamingClient;
    use crate::providers::internal::openai_chat_completions_compatible::test_support::sse_bytes_from_data_lines;
    use crate::streaming::StreamedAssistantContent;
    use futures::StreamExt;
    use serde_json::json;

    #[test]
    fn test_streaming_completion_response_deserialization() {
        let json = json!({
            "id": "gen-abc123",
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": "Hello"
                }
            }],
            "created": 1234567890u64,
            "model": "gpt-3.5-turbo",
            "object": "chat.completion.chunk"
        });

        let response: StreamingCompletionChunk = serde_json::from_value(json).unwrap();
        assert_eq!(response.id, "gen-abc123");
        assert_eq!(response.model, "gpt-3.5-turbo");
        assert_eq!(response.choices.len(), 1);
    }

    #[test]
    fn test_delta_with_content() {
        let json = json!({
            "role": "assistant",
            "content": "Hello, world!"
        });

        let delta: StreamingDelta = serde_json::from_value(json).unwrap();
        assert_eq!(delta.role, Some("assistant".to_string()));
        assert_eq!(delta.content, Some("Hello, world!".to_string()));
    }

    #[test]
    fn test_delta_with_tool_call() {
        let json = json!({
            "role": "assistant",
            "tool_calls": [{
                "index": 0,
                "id": "call_abc",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": "{\"location\":"
                }
            }]
        });

        let delta: StreamingDelta = serde_json::from_value(json).unwrap();
        assert_eq!(delta.tool_calls.len(), 1);
        assert_eq!(delta.tool_calls[0].index, 0);
        assert_eq!(delta.tool_calls[0].id, Some("call_abc".to_string()));
    }

    #[test]
    fn test_tool_call_with_partial_arguments() {
        let json = json!({
            "index": 0,
            "id": null,
            "type": null,
            "function": {
                "name": null,
                "arguments": "Paris"
            }
        });

        let tool_call: StreamingToolCall = serde_json::from_value(json).unwrap();
        assert_eq!(tool_call.index, 0);
        assert!(tool_call.id.is_none());
        assert_eq!(tool_call.function.arguments, Some("Paris".to_string()));
    }

    #[test]
    fn test_streaming_with_usage() {
        let json = json!({
            "id": "gen-xyz",
            "choices": [{
                "index": 0,
                "delta": {
                    "content": null
                }
            }],
            "created": 1234567890u64,
            "model": "gpt-4",
            "object": "chat.completion.chunk",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        });

        let response: StreamingCompletionChunk = serde_json::from_value(json).unwrap();
        assert!(response.usage.is_some());
        let usage = response.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.completion_tokens, 50);
        assert_eq!(usage.total_tokens, 150);
    }

    #[test]
    fn test_multiple_tool_call_deltas() {
        // Simulates the sequence of deltas for a tool call with arguments
        let start_json = json!({
            "id": "gen-1",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": ""
                        }
                    }]
                }
            }],
            "created": 1234567890u64,
            "model": "gpt-4",
            "object": "chat.completion.chunk"
        });

        let delta1_json = json!({
            "id": "gen-2",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {
                            "arguments": "{\"query\":"
                        }
                    }]
                }
            }],
            "created": 1234567890u64,
            "model": "gpt-4",
            "object": "chat.completion.chunk"
        });

        let delta2_json = json!({
            "id": "gen-3",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {
                            "arguments": "\"Rust programming\"}"
                        }
                    }]
                }
            }],
            "created": 1234567890u64,
            "model": "gpt-4",
            "object": "chat.completion.chunk"
        });

        // Verify all chunks deserialize
        let start: StreamingCompletionChunk = serde_json::from_value(start_json).unwrap();
        assert_eq!(
            start.choices[0].delta.tool_calls[0].id,
            Some("call_123".to_string())
        );

        let delta1: StreamingCompletionChunk = serde_json::from_value(delta1_json).unwrap();
        assert_eq!(
            delta1.choices[0].delta.tool_calls[0].function.arguments,
            Some("{\"query\":".to_string())
        );

        let delta2: StreamingCompletionChunk = serde_json::from_value(delta2_json).unwrap();
        assert_eq!(
            delta2.choices[0].delta.tool_calls[0].function.arguments,
            Some("\"Rust programming\"}".to_string())
        );
    }

    #[test]
    fn test_response_with_error() {
        let json = json!({
            "id": "cmpl-abc123",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "gpt-3.5-turbo",
            "provider": "openai",
            "error": { "code": 500, "message": "Provider disconnected" },
            "choices": [
                { "index": 0, "delta": { "content": "" }, "finish_reason": "error" }
            ]
        });

        let response: StreamingCompletionChunk = serde_json::from_value(json).unwrap();
        assert!(response.error.is_some());
        let error = response.error.as_ref().unwrap();
        assert_eq!(error.code, 500);
        assert_eq!(error.message, "Provider disconnected");
    }

    #[tokio::test]
    async fn encrypted_reasoning_details_attach_to_emitted_tool_calls() {
        let client = MockStreamingClient {
            sse_bytes: sse_bytes_from_data_lines([
                "{\"id\":\"gen-1\",\"model\":\"openai/gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_123\",\"type\":\"function\",\"function\":{\"name\":\"search\",\"arguments\":\"\"}}],\"reasoning_details\":[]},\"finish_reason\":null}],\"usage\":null}",
                "{\"id\":\"gen-2\",\"model\":\"openai/gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[],\"reasoning_details\":[{\"type\":\"reasoning.encrypted\",\"id\":\"call_123\",\"format\":\"opaque\",\"index\":0,\"data\":\"enc_blob\"}]},\"finish_reason\":null}],\"usage\":null}",
                "{\"id\":\"gen-3\",\"model\":\"openai/gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[],\"reasoning_details\":[]},\"finish_reason\":\"tool_calls\"}],\"usage\":null}",
                "[DONE]",
            ]),
        };

        let req = Request::builder()
            .method("POST")
            .uri("http://localhost/v1/chat/completions")
            .body(Vec::new())
            .expect("request should build");

        let mut stream = send_compatible_streaming_request(client, req)
            .await
            .expect("stream should start");

        let tool_call = loop {
            match stream.next().await.expect("stream should yield an item") {
                Ok(StreamedAssistantContent::ToolCall { tool_call, .. }) => break tool_call,
                Ok(_) => continue,
                Err(err) => panic!("stream should not error: {err}"),
            }
        };

        assert_eq!(tool_call.id, "call_123");
        assert_eq!(tool_call.function.name, "search");
        assert_eq!(tool_call.function.arguments, serde_json::json!({}));
        assert_eq!(tool_call.signature.as_deref(), Some("enc_blob"));
        assert_eq!(
            tool_call.additional_params,
            Some(json!({
                "type": "reasoning.encrypted",
                "id": "call_123",
                "format": "opaque",
                "index": 0,
                "data": "enc_blob"
            }))
        );
    }
}
