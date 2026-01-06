use std::collections::HashMap;

use async_stream::stream;
use futures::StreamExt;
use http::Request;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::{Level, enabled, info_span};
use tracing_futures::Instrument;

use crate::completion::{CompletionError, CompletionRequest, GetTokenUsage};
use crate::http_client::HttpClientExt;
use crate::http_client::sse::{Event, GenericEventSource};
use crate::json_utils::{self, merge};
use crate::message::{ToolCall, ToolFunction};
use crate::providers::openai::completion::{self, CompletionModel, OpenAIRequestParams, Usage};
use crate::streaming::{self, RawStreamingChoice};

// ================================================================
// OpenAI Completion Streaming API
// ================================================================
#[derive(Deserialize, Debug)]
pub(crate) struct StreamingFunction {
    pub(crate) name: Option<String>,
    pub(crate) arguments: Option<String>,
}

#[derive(Deserialize, Debug)]
pub(crate) struct StreamingToolCall {
    pub(crate) index: usize,
    pub(crate) id: Option<String>,
    pub(crate) function: StreamingFunction,
}

#[derive(Deserialize, Debug)]
struct StreamingDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default, deserialize_with = "json_utils::null_or_vec")]
    tool_calls: Vec<StreamingToolCall>,
}

#[derive(Deserialize, Debug, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    ToolCalls,
    Stop,
    ContentFilter,
    Length,
    #[serde(untagged)]
    Other(String), // This will handle the deprecated function_call
}

#[derive(Deserialize, Debug)]
struct StreamingChoice {
    delta: StreamingDelta,
    finish_reason: Option<FinishReason>,
}

#[derive(Deserialize, Debug)]
struct StreamingCompletionChunk {
    choices: Vec<StreamingChoice>,
    usage: Option<Usage>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct StreamingCompletionResponse {
    pub usage: Usage,
}

impl GetTokenUsage for StreamingCompletionResponse {
    fn token_usage(&self) -> Option<crate::completion::Usage> {
        let mut usage = crate::completion::Usage::new();
        usage.input_tokens = self.usage.prompt_tokens as u64;
        usage.output_tokens = self.usage.total_tokens as u64 - self.usage.prompt_tokens as u64;
        usage.total_tokens = self.usage.total_tokens as u64;
        Some(usage)
    }
}

impl<T> CompletionModel<T>
where
    T: HttpClientExt + Clone + 'static,
{
    pub(crate) async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<streaming::StreamingCompletionResponse<StreamingCompletionResponse>, CompletionError>
    {
        let request = super::CompletionRequest::try_from(OpenAIRequestParams {
            model: self.model.clone(),
            request: completion_request,
            strict_tools: self.strict_tools,
            tool_result_array_content: self.tool_result_array_content,
        })?;
        let request_messages = serde_json::to_string(&request.messages)
            .expect("Converting to JSON from a Rust struct shouldn't fail");
        let mut request_as_json = serde_json::to_value(request).expect("this should never fail");

        request_as_json = merge(
            request_as_json,
            json!({"stream": true, "stream_options": {"include_usage": true}}),
        );

        if enabled!(Level::TRACE) {
            tracing::trace!(
                target: "rig::completions",
                "OpenAI Chat Completions streaming completion request: {}",
                serde_json::to_string_pretty(&request_as_json)?
            );
        }

        let req_body = serde_json::to_vec(&request_as_json)?;

        let req = self
            .client
            .post("/chat/completions")?
            .body(req_body)
            .map_err(|e| CompletionError::HttpError(e.into()))?;

        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat",
                gen_ai.operation.name = "chat",
                gen_ai.provider.name = "openai",
                gen_ai.request.model = self.model,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = self.model,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.input.messages = request_messages,
                gen_ai.output.messages = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        let client = self.client.clone();

        tracing::Instrument::instrument(send_compatible_streaming_request(client, req), span).await
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
        let span = tracing::Span::current();

        // Accumulate tool calls by index while streaming
        let mut tool_calls: HashMap<usize, ToolCall> = HashMap::new();
        let mut text_content = String::new();
        let mut final_tool_calls: Vec<completion::ToolCall> = Vec::new();
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

                    // Usage updates (some providers send a final "usage-only" chunk with empty choices)
                    if let Some(usage) = data.usage {
                        final_usage = Some(usage);
                    }

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
                            let existing_tool_call = tool_calls.entry(index).or_insert_with(|| ToolCall {
                                id: String::new(),
                                call_id: None,
                                function: ToolFunction {
                                    name: String::new(),
                                    arguments: serde_json::Value::Null,
                                },
                                signature: None,
                                additional_params: None,
                            });

                            // Update fields if present
                            if let Some(id) = &tool_call.id && !id.is_empty() {
                                    existing_tool_call.id = id.clone();
                            }

                            if let Some(name) = &tool_call.function.name && !name.is_empty() {
                                    existing_tool_call.function.name = name.clone();
                                    yield Ok(streaming::RawStreamingChoice::ToolCallDelta {
                                        id: existing_tool_call.id.clone(),
                                        content: streaming::ToolCallDeltaContent::Name(name.clone()),
                                    });
                            }

                                // Convert current arguments to string if needed
                            if let Some(chunk) = &tool_call.function.arguments && !chunk.is_empty() {
                                let current_args = match &existing_tool_call.function.arguments {
                                    serde_json::Value::Null => String::new(),
                                    serde_json::Value::String(s) => s.clone(),
                                    v => v.to_string(),
                                };

                                // Concatenate the new chunk
                                let combined = format!("{current_args}{chunk}");

                                // Try to parse as JSON if it looks complete
                                if combined.trim_start().starts_with('{') && combined.trim_end().ends_with('}') {
                                    match serde_json::from_str(&combined) {
                                        Ok(parsed) => existing_tool_call.function.arguments = parsed,
                                        Err(_) => existing_tool_call.function.arguments = serde_json::Value::String(combined),
                                    }
                                } else {
                                    existing_tool_call.function.arguments = serde_json::Value::String(combined);
                                }

                                // Emit the delta so UI can show progress
                                yield Ok(streaming::RawStreamingChoice::ToolCallDelta {
                                    id: existing_tool_call.id.clone(),
                                    content: streaming::ToolCallDeltaContent::Delta(chunk.clone()),
                                });
                            }
                        }
                    }

                    // Streamed text content
                    if let Some(content) = &delta.content && !content.is_empty() {
                        text_content += content;
                        yield Ok(streaming::RawStreamingChoice::Message(content.clone()));
                    }

                    // Finish reason
                    if let Some(finish_reason) = &choice.finish_reason && *finish_reason == FinishReason::ToolCalls {
                        for (_idx, tool_call) in tool_calls.into_iter() {
                            final_tool_calls.push(completion::ToolCall {
                                id: tool_call.id.clone(),
                                r#type: completion::ToolType::Function,
                                function: completion::Function {
                                    name: tool_call.function.name.clone(),
                                    arguments: tool_call.function.arguments.clone(),
                                },
                            });
                            yield Ok(streaming::RawStreamingChoice::ToolCall(
                                streaming::RawStreamingToolCall::new(
                                    tool_call.id,
                                    tool_call.function.name,
                                    tool_call.function.arguments,
                                )
                            ));
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
            yield Ok(streaming::RawStreamingChoice::ToolCall(
                streaming::RawStreamingToolCall::new(
                    tool_call.id,
                    tool_call.function.name,
                    tool_call.function.arguments,
                )
            ));
        }

        let final_usage = final_usage.unwrap_or_default();
        if !span.is_disabled() {
            span.record("gen_ai.usage.input_tokens", final_usage.prompt_tokens);
            span.record("gen_ai.usage.output_tokens", final_usage.total_tokens - final_usage.prompt_tokens);
        }

        yield Ok(RawStreamingChoice::FinalResponse(StreamingCompletionResponse {
            usage: final_usage
        }));
    }.instrument(span);

    Ok(streaming::StreamingCompletionResponse::stream(Box::pin(
        stream,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_function_deserialization() {
        let json = r#"{"name": "get_weather", "arguments": "{\"location\":\"Paris\"}"}"#;
        let function: StreamingFunction = serde_json::from_str(json).unwrap();
        assert_eq!(function.name, Some("get_weather".to_string()));
        assert_eq!(
            function.arguments.as_ref().unwrap(),
            r#"{"location":"Paris"}"#
        );
    }

    #[test]
    fn test_streaming_tool_call_deserialization() {
        let json = r#"{
            "index": 0,
            "id": "call_abc123",
            "function": {
                "name": "get_weather",
                "arguments": "{\"city\":\"London\"}"
            }
        }"#;
        let tool_call: StreamingToolCall = serde_json::from_str(json).unwrap();
        assert_eq!(tool_call.index, 0);
        assert_eq!(tool_call.id, Some("call_abc123".to_string()));
        assert_eq!(tool_call.function.name, Some("get_weather".to_string()));
    }

    #[test]
    fn test_streaming_tool_call_partial_deserialization() {
        // Partial tool calls have no name and partial arguments
        let json = r#"{
            "index": 0,
            "id": null,
            "function": {
                "name": null,
                "arguments": "Paris"
            }
        }"#;
        let tool_call: StreamingToolCall = serde_json::from_str(json).unwrap();
        assert_eq!(tool_call.index, 0);
        assert!(tool_call.id.is_none());
        assert!(tool_call.function.name.is_none());
        assert_eq!(tool_call.function.arguments.as_ref().unwrap(), "Paris");
    }

    #[test]
    fn test_streaming_delta_with_tool_calls() {
        let json = r#"{
            "content": null,
            "tool_calls": [{
                "index": 0,
                "id": "call_xyz",
                "function": {
                    "name": "search",
                    "arguments": ""
                }
            }]
        }"#;
        let delta: StreamingDelta = serde_json::from_str(json).unwrap();
        assert!(delta.content.is_none());
        assert_eq!(delta.tool_calls.len(), 1);
        assert_eq!(delta.tool_calls[0].id, Some("call_xyz".to_string()));
    }

    #[test]
    fn test_streaming_chunk_deserialization() {
        let json = r#"{
            "choices": [{
                "delta": {
                    "content": "Hello",
                    "tool_calls": []
                }
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }"#;
        let chunk: StreamingCompletionChunk = serde_json::from_str(json).unwrap();
        assert_eq!(chunk.choices.len(), 1);
        assert_eq!(chunk.choices[0].delta.content, Some("Hello".to_string()));
        assert!(chunk.usage.is_some());
    }

    #[test]
    fn test_streaming_chunk_with_multiple_tool_call_deltas() {
        // Simulates multiple partial tool call chunks arriving
        let json_start = r#"{
            "choices": [{
                "delta": {
                    "content": null,
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_123",
                        "function": {
                            "name": "get_weather",
                            "arguments": ""
                        }
                    }]
                }
            }],
            "usage": null
        }"#;

        let json_chunk1 = r#"{
            "choices": [{
                "delta": {
                    "content": null,
                    "tool_calls": [{
                        "index": 0,
                        "id": null,
                        "function": {
                            "name": null,
                            "arguments": "{\"loc"
                        }
                    }]
                }
            }],
            "usage": null
        }"#;

        let json_chunk2 = r#"{
            "choices": [{
                "delta": {
                    "content": null,
                    "tool_calls": [{
                        "index": 0,
                        "id": null,
                        "function": {
                            "name": null,
                            "arguments": "ation\":\"NYC\"}"
                        }
                    }]
                }
            }],
            "usage": null
        }"#;

        // Verify each chunk deserializes correctly
        let start_chunk: StreamingCompletionChunk = serde_json::from_str(json_start).unwrap();
        assert_eq!(start_chunk.choices[0].delta.tool_calls.len(), 1);
        assert_eq!(
            start_chunk.choices[0].delta.tool_calls[0]
                .function
                .name
                .as_ref()
                .unwrap(),
            "get_weather"
        );

        let chunk1: StreamingCompletionChunk = serde_json::from_str(json_chunk1).unwrap();
        assert_eq!(chunk1.choices[0].delta.tool_calls.len(), 1);
        assert_eq!(
            chunk1.choices[0].delta.tool_calls[0]
                .function
                .arguments
                .as_ref()
                .unwrap(),
            "{\"loc"
        );

        let chunk2: StreamingCompletionChunk = serde_json::from_str(json_chunk2).unwrap();
        assert_eq!(chunk2.choices[0].delta.tool_calls.len(), 1);
        assert_eq!(
            chunk2.choices[0].delta.tool_calls[0]
                .function
                .arguments
                .as_ref()
                .unwrap(),
            "ation\":\"NYC\"}"
        );
    }

    #[tokio::test]
    async fn test_streaming_usage_only_chunk_is_not_ignored() {
        use bytes::Bytes;
        use futures::StreamExt;

        #[derive(Clone)]
        struct MockHttpClient {
            sse_bytes: Bytes,
        }

        impl crate::http_client::HttpClientExt for MockHttpClient {
            fn send<T, U>(
                &self,
                _req: http::Request<T>,
            ) -> impl std::future::Future<
                Output = crate::http_client::Result<
                    http::Response<crate::http_client::LazyBody<U>>,
                >,
            > + crate::wasm_compat::WasmCompatSend
            + 'static
            where
                T: Into<Bytes>,
                T: crate::wasm_compat::WasmCompatSend,
                U: From<Bytes>,
                U: crate::wasm_compat::WasmCompatSend + 'static,
            {
                std::future::ready(Err(crate::http_client::Error::InvalidStatusCode(
                    http::StatusCode::NOT_IMPLEMENTED,
                )))
            }

            fn send_multipart<U>(
                &self,
                _req: http::Request<crate::http_client::MultipartForm>,
            ) -> impl std::future::Future<
                Output = crate::http_client::Result<
                    http::Response<crate::http_client::LazyBody<U>>,
                >,
            > + crate::wasm_compat::WasmCompatSend
            + 'static
            where
                U: From<Bytes>,
                U: crate::wasm_compat::WasmCompatSend + 'static,
            {
                std::future::ready(Err(crate::http_client::Error::InvalidStatusCode(
                    http::StatusCode::NOT_IMPLEMENTED,
                )))
            }

            fn send_streaming<T>(
                &self,
                _req: http::Request<T>,
            ) -> impl std::future::Future<
                Output = crate::http_client::Result<crate::http_client::StreamingResponse>,
            > + crate::wasm_compat::WasmCompatSend
            where
                T: Into<Bytes>,
            {
                let sse_bytes = self.sse_bytes.clone();
                async move {
                    let byte_stream = futures::stream::iter(vec![Ok::<
                        Bytes,
                        crate::http_client::Error,
                    >(sse_bytes)]);
                    let boxed_stream: crate::http_client::sse::BoxedStream = Box::pin(byte_stream);

                    http::Response::builder()
                        .status(http::StatusCode::OK)
                        .header(reqwest::header::CONTENT_TYPE, "text/event-stream")
                        .body(boxed_stream)
                        .map_err(crate::http_client::Error::Protocol)
                }
            }
        }

        // Some providers emit a final "usage-only" chunk where `choices` is empty.
        let sse = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\",\"tool_calls\":[]}}],\"usage\":null}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15}}\n\n",
            "data: [DONE]\n\n",
        );

        let client = MockHttpClient {
            sse_bytes: Bytes::from(sse),
        };

        let req = http::Request::builder()
            .method("POST")
            .uri("http://localhost/v1/chat/completions")
            .body(Vec::new())
            .unwrap();

        let mut stream = send_compatible_streaming_request(client, req)
            .await
            .unwrap();

        let mut final_usage = None;
        while let Some(chunk) = stream.next().await {
            if let streaming::StreamedAssistantContent::Final(res) = chunk.unwrap() {
                final_usage = Some(res.usage);
                break;
            }
        }

        let usage = final_usage.expect("expected a final response with usage");
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.total_tokens, 15);
    }
}
