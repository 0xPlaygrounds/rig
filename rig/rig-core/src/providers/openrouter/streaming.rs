use std::collections::HashMap;

use async_stream::stream;
use futures::StreamExt;
use http::Request;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::info_span;
use tracing_futures::Instrument;

use crate::completion::{CompletionError, CompletionRequest, GetTokenUsage};
use crate::http_client::HttpClientExt;
use crate::http_client::sse::{Event, GenericEventSource};
use crate::json_utils;
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
        let mut usage = crate::completion::Usage::new();

        usage.input_tokens = self.usage.prompt_tokens as u64;
        usage.output_tokens = self.usage.completion_tokens as u64;
        usage.total_tokens = self.usage.total_tokens as u64;

        Some(usage)
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

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
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
        let preamble = completion_request.preamble.clone();
        let mut request = OpenrouterCompletionRequest::try_from(OpenRouterRequestParams {
            model: self.model.as_ref(),
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
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = preamble,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
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

pub async fn send_compatible_streaming_request<T>(
    http_client: T,
    req: Request<Vec<u8>>,
) -> Result<streaming::StreamingCompletionResponse<StreamingCompletionResponse>, CompletionError>
where
    T: HttpClientExt + Clone + 'static,
{
    let span = tracing::Span::current();
    // Build the request with proper headers for SSE
    let mut event_source = GenericEventSource::new(http_client, req);

    let stream = stream! {
        // Accumulate tool calls by index while streaming
        let mut tool_calls: HashMap<usize, streaming::RawStreamingToolCall> = HashMap::new();
        let mut final_usage = None;

        while let Some(event_result) = event_source.next().await {
            match event_result {
                Ok(Event::Open) => {
                    tracing::trace!("SSE connection opened");
                    continue;
                }

                Ok(Event::Message(message)) => {
                    if message.data.trim().is_empty() || message.data == "[DONE]" {
                        continue;
                    }

                    let data = match serde_json::from_str::<StreamingCompletionChunk>(&message.data) {
                        Ok(data) => data,
                        Err(error) => {
                            tracing::error!(?error, message = message.data, "Failed to parse SSE message");
                            continue;
                        }
                    };

                    // Expect at least one choice
                     let Some(choice) = data.choices.first() else {
                        tracing::debug!("There is no choice");
                        continue;
                    };
                    let delta = &choice.delta;

                    if !delta.tool_calls.is_empty() {
                        for tool_call in &delta.tool_calls {
                            let index = tool_call.index;

                            // Get or create tool call entry
                            let existing_tool_call = tool_calls.entry(index).or_insert_with(streaming::RawStreamingToolCall::empty);

                            // Update fields if present
                            if let Some(id) = &tool_call.id && !id.is_empty() {
                                    existing_tool_call.id = id.clone();
                            }

                            if let Some(name) = &tool_call.function.name && !name.is_empty() {
                                    existing_tool_call.name = name.clone();
                                    yield Ok(streaming::RawStreamingChoice::ToolCallDelta {
                                        id: existing_tool_call.id.clone(),
                                        content: streaming::ToolCallDeltaContent::Name(name.clone()),
                                    });
                            }

                                // Convert current arguments to string if needed
                            if let Some(chunk) = &tool_call.function.arguments && !chunk.is_empty() {
                                let current_args = match &existing_tool_call.arguments {
                                    serde_json::Value::Null => String::new(),
                                    serde_json::Value::String(s) => s.clone(),
                                    v => v.to_string(),
                                };

                                // Concatenate the new chunk
                                let combined = format!("{current_args}{chunk}");

                                // Try to parse as JSON if it looks complete
                                if combined.trim_start().starts_with('{') && combined.trim_end().ends_with('}') {
                                    match serde_json::from_str(&combined) {
                                        Ok(parsed) => existing_tool_call.arguments = parsed,
                                        Err(_) => existing_tool_call.arguments = serde_json::Value::String(combined),
                                    }
                                } else {
                                    existing_tool_call.arguments = serde_json::Value::String(combined);
                                }

                                // Emit the delta so UI can show progress
                                yield Ok(streaming::RawStreamingChoice::ToolCallDelta {
                                    id: existing_tool_call.id.clone(),
                                    content: streaming::ToolCallDeltaContent::Delta(chunk.clone()),
                                });
                            }
                        }

                        // Update the signature and the additional params of the tool call if present
                        for reasoning_detail in &delta.reasoning_details {
                            if let ReasoningDetails::Encrypted { id, data, .. } = reasoning_detail
                                && let Some(id) = id
                                && let Some(tool_call) = tool_calls.values_mut().find(|tool_call| tool_call.id.eq(id))
                                && let Ok(additional_params) = serde_json::to_value(reasoning_detail) {
                                tool_call.signature = Some(data.clone());
                                tool_call.additional_params = Some(additional_params);
                            }
                        }
                    }

                    // Streamed reasoning content
                    if let Some(reasoning) = &delta.reasoning && !reasoning.is_empty() {
                        yield Ok(streaming::RawStreamingChoice::ReasoningDelta {
                            reasoning: reasoning.clone(),
                            id: None,
                        });
                    }

                    // Streamed text content
                    if let Some(content) = &delta.content && !content.is_empty() {
                        yield Ok(streaming::RawStreamingChoice::Message(content.clone()));
                    }

                    // Usage updates
                    if let Some(usage) = data.usage {
                        final_usage = Some(usage);
                    }

                    // Finish reason
                    if let Some(finish_reason) = &choice.finish_reason && *finish_reason == FinishReason::ToolCalls {
                        for (_idx, tool_call) in tool_calls.into_iter() {
                            yield Ok(streaming::RawStreamingChoice::ToolCall(tool_call));
                        }
                        tool_calls = HashMap::new();
                    }
                }
                Err(crate::http_client::Error::StreamEnded) => {
                    break;
                }
                Err(error) => {
                    tracing::error!(?error, "SSE error");
                    yield Err(CompletionError::ProviderError(error.to_string()));
                    break;
                }
            }
        }

        // Ensure event source is closed when stream ends
        event_source.close();

        // Flush any accumulated tool calls (that weren't emitted as ToolCall earlier)
        for (_idx, tool_call) in tool_calls.into_iter() {
            yield Ok(streaming::RawStreamingChoice::ToolCall(tool_call));
        }

        // Final response with usage
        yield Ok(streaming::RawStreamingChoice::FinalResponse(StreamingCompletionResponse {
            usage: final_usage.unwrap_or_default(),
        }));
    }.instrument(span);

    Ok(streaming::StreamingCompletionResponse::stream(Box::pin(
        stream,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
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
}
