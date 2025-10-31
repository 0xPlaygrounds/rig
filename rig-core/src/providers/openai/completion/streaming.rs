use crate::completion::{CompletionError, CompletionRequest, GetTokenUsage};
use crate::http_client::HttpClientExt;
use crate::http_client::sse::{Event, GenericEventSource};
use crate::json_utils;
use crate::json_utils::merge;
use crate::providers::openai::completion::{CompletionModel, Usage};
use crate::streaming;
use crate::streaming::RawStreamingChoice;
use async_stream::stream;
use futures::StreamExt;
use http::Request;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use tracing::{debug, info_span};
use tracing_futures::Instrument;

// ================================================================
// OpenAI Completion Streaming API
// ================================================================
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StreamingFunction {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StreamingToolCall {
    pub index: usize,
    pub id: Option<String>,
    pub function: StreamingFunction,
}

#[derive(Deserialize, Debug)]
struct StreamingDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default, deserialize_with = "json_utils::null_or_vec")]
    tool_calls: Vec<StreamingToolCall>,
}

#[derive(Deserialize, Debug)]
struct StreamingChoice {
    delta: StreamingDelta,
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

impl CompletionModel<reqwest::Client> {
    pub(crate) async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<streaming::StreamingCompletionResponse<StreamingCompletionResponse>, CompletionError>
    {
        let request = super::CompletionRequest::try_from((self.model.clone(), completion_request))?;
        let request_messages = serde_json::to_string(&request.messages)
            .expect("Converting to JSON from a Rust struct shouldn't fail");
        let mut request_as_json = serde_json::to_value(request).expect("this should never fail");

        request_as_json = merge(
            request_as_json,
            json!({"stream": true, "stream_options": {"include_usage": true}}),
        );

        let req_body = serde_json::to_vec(&request_as_json)?;

        let req = self
            .client
            .post("/chat/completions")?
            .header("Content-Type", "application/json")
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

        tracing::Instrument::instrument(
            send_compatible_streaming_request(self.client.http_client.clone(), req),
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
        let span = tracing::Span::current();
        let mut final_usage = Usage::new();

        // Track in-progress tool calls
        let mut tool_calls: HashMap<usize, (String, String, String)> = HashMap::new();

        let mut text_content = String::new();

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

                    let data = serde_json::from_str::<StreamingCompletionChunk>(&message.data);
                    let Ok(data) = data else {
                        let err = data.unwrap_err();
                        debug!("Couldn't serialize data as StreamingCompletionChunk: {:?}", err);
                        continue;
                    };

                    if let Some(choice) = data.choices.first() {
                        let delta = &choice.delta;

                        // Tool calls
                        if !delta.tool_calls.is_empty() {
                            for tool_call in &delta.tool_calls {
                                let function = tool_call.function.clone();

                                // Start of tool call
                                if function.name.is_some() && function.arguments.is_empty() {
                                    let id = tool_call.id.clone().unwrap_or_default();
                                    tool_calls.insert(
                                        tool_call.index,
                                        (id, function.name.clone().unwrap(), "".to_string()),
                                    );
                                }
                                // tool call partial (ie, a continuation of a previously received tool call)
                                // name: None or Empty String
                                // arguments: Some(String)
                                else if function.name.clone().is_none_or(|s| s.is_empty())
                                    && !function.arguments.is_empty()
                                {
                                    if let Some((id, name, arguments)) =
                                        tool_calls.get(&tool_call.index).cloned()
                                    {
                                        let new_arguments = &tool_call.function.arguments;
                                        let combined_arguments = format!("{arguments}{new_arguments}");
                                        tool_calls.insert(
                                            tool_call.index,
                                            (id.clone(), name.clone(), combined_arguments),
                                        );

                                        // Emit the delta so UI can show progress
                                        yield Ok(streaming::RawStreamingChoice::ToolCallDelta {
                                            id: id.clone(),
                                            delta: new_arguments.clone(),
                                        });
                                    } else {
                                        debug!("Partial tool call received but tool call was never started.");
                                    }
                                }
                                // Complete tool call
                                else {
                                    let id = tool_call.id.clone().unwrap_or_default();
                                    let name = function.name.expect("tool call should have a name");
                                    let arguments = function.arguments;
                                    let Ok(arguments) = serde_json::from_str(&arguments) else {
                                        debug!("Couldn't serialize '{arguments}' as JSON");
                                        continue;
                                    };

                                    yield Ok(streaming::RawStreamingChoice::ToolCall {
                                        id,
                                        name,
                                        arguments,
                                        call_id: None,
                                    });
                                }
                            }
                        }

                        // Message content
                        if let Some(content) = &choice.delta.content {
                            text_content += content;
                            yield Ok(streaming::RawStreamingChoice::Message(content.clone()))
                        }
                    }

                    // Usage updates
                    if let Some(usage) = data.usage {
                        final_usage = usage.clone();
                    }
                }
                Err(crate::http_client::Error::StreamEnded) => {
                    break;
                }
                Err(error) => {
                    tracing::error!(?error, "SSE error");
                    yield Err(CompletionError::ResponseError(error.to_string()));
                    break;
                }
            }
        }

        // Ensure event source is closed when stream ends
        event_source.close();

        let mut vec_toolcalls = vec![];

        // Flush any tool calls that weren’t fully yielded
        for (_, (id, name, arguments)) in tool_calls {
            let Ok(arguments) = serde_json::from_str::<serde_json::Value>(&arguments) else {
                continue;
            };

            vec_toolcalls.push(super::ToolCall {
                r#type: super::ToolType::Function,
                id: id.clone(),
                function: super::Function {
                    name: name.clone(), arguments: arguments.clone()
                },
            });

            yield Ok(RawStreamingChoice::ToolCall {
                id,
                name,
                arguments,
                call_id: None,
            });
        }

        let message_output = super::Message::Assistant {
            content: vec![super::AssistantContent::Text { text: text_content }],
            refusal: None,
            audio: None,
            name: None,
            tool_calls: vec_toolcalls
        };

        span.record("gen_ai.usage.input_tokens", final_usage.prompt_tokens);
        span.record("gen_ai.usage.output_tokens", final_usage.total_tokens - final_usage.prompt_tokens);
        span.record("gen_ai.output.messages", serde_json::to_string(&vec![message_output]).expect("Converting from a Rust struct should always convert to JSON without failing"));

        yield Ok(RawStreamingChoice::FinalResponse(StreamingCompletionResponse {
            usage: final_usage.clone()
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
        assert_eq!(function.arguments, r#"{"location":"Paris"}"#.to_string());
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
        assert_eq!(tool_call.function.arguments, "Paris");
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
            chunk1.choices[0].delta.tool_calls[0].function.arguments,
            "{\"loc"
        );

        let chunk2: StreamingCompletionChunk = serde_json::from_str(json_chunk2).unwrap();
        assert_eq!(chunk2.choices[0].delta.tool_calls.len(), 1);
        assert_eq!(
            chunk2.choices[0].delta.tool_calls[0].function.arguments,
            "ation\":\"NYC\"}"
        );
    }
}
