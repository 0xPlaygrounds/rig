use async_stream::stream;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::info_span;
use tracing_futures::Instrument;

use super::completion::{CompletionModel, Content, Message, ToolChoice, ToolDefinition, Usage};
use super::decoders::sse::from_response as sse_from_response;
use crate::OneOrMany;
use crate::completion::{CompletionError, CompletionRequest, GetTokenUsage};
use crate::http_client::{self, HttpClientExt};
use crate::json_utils::merge_inplace;
use crate::streaming::{self, RawStreamingChoice, StreamingResult};
use crate::telemetry::SpanCombinator;

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamingEvent {
    MessageStart {
        message: MessageStart,
    },
    ContentBlockStart {
        index: usize,
        content_block: Content,
    },
    ContentBlockDelta {
        index: usize,
        delta: ContentDelta,
    },
    ContentBlockStop {
        index: usize,
    },
    MessageDelta {
        delta: MessageDelta,
        usage: PartialUsage,
    },
    MessageStop,
    Ping,
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
pub struct MessageStart {
    pub id: String,
    pub role: String,
    pub content: Vec<Content>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentDelta {
    TextDelta { text: String },
    InputJsonDelta { partial_json: String },
    ThinkingDelta { thinking: String },
    SignatureDelta { signature: String },
}

#[derive(Debug, Deserialize)]
pub struct MessageDelta {
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
}

#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct PartialUsage {
    pub output_tokens: usize,
    #[serde(default)]
    pub input_tokens: Option<usize>,
}

impl GetTokenUsage for PartialUsage {
    fn token_usage(&self) -> Option<crate::completion::Usage> {
        let mut usage = crate::completion::Usage::new();

        usage.input_tokens = self.input_tokens.unwrap_or_default() as u64;
        usage.output_tokens = self.output_tokens as u64;
        usage.total_tokens = usage.input_tokens + usage.output_tokens;
        Some(usage)
    }
}

#[derive(Default)]
struct ToolCallState {
    name: String,
    id: String,
    input_json: String,
}

#[derive(Default)]
struct ThinkingState {
    thinking: String,
    signature: String,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct StreamingCompletionResponse {
    pub usage: PartialUsage,
}

impl GetTokenUsage for StreamingCompletionResponse {
    fn token_usage(&self) -> Option<crate::completion::Usage> {
        let mut usage = crate::completion::Usage::new();
        usage.input_tokens = self.usage.input_tokens.unwrap_or(0) as u64;
        usage.output_tokens = self.usage.output_tokens as u64;
        usage.total_tokens =
            self.usage.input_tokens.unwrap_or(0) as u64 + self.usage.output_tokens as u64;

        Some(usage)
    }
}

impl<T> CompletionModel<T>
where
    T: HttpClientExt + Clone + Default,
{
    pub(crate) async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<streaming::StreamingCompletionResponse<StreamingCompletionResponse>, CompletionError>
    {
        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat_streaming",
                gen_ai.operation.name = "chat_streaming",
                gen_ai.provider.name = "anthropic",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = &completion_request.preamble,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = self.model,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.input.messages = tracing::field::Empty,
                gen_ai.output.messages = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };
        let max_tokens = if let Some(tokens) = completion_request.max_tokens {
            tokens
        } else if let Some(tokens) = self.default_max_tokens {
            tokens
        } else {
            return Err(CompletionError::RequestError(
                "`max_tokens` must be set for Anthropic".into(),
            ));
        };

        let mut full_history = vec![];
        if let Some(docs) = completion_request.normalized_documents() {
            full_history.push(docs);
        }
        full_history.extend(completion_request.chat_history);
        span.record_model_input(&full_history);

        let full_history = full_history
            .into_iter()
            .map(Message::try_from)
            .collect::<Result<Vec<Message>, _>>()?;

        let mut body = json!({
            "model": self.model,
            "messages": full_history,
            "max_tokens": max_tokens,
            "system": completion_request.preamble.unwrap_or("".to_string()),
            "stream": true,
        });

        if let Some(temperature) = completion_request.temperature {
            merge_inplace(&mut body, json!({ "temperature": temperature }));
        }

        if !completion_request.tools.is_empty() {
            merge_inplace(
                &mut body,
                json!({
                    "tools": completion_request
                        .tools
                        .into_iter()
                        .map(|tool| ToolDefinition {
                            name: tool.name,
                            description: Some(tool.description),
                            input_schema: tool.parameters,
                        })
                        .collect::<Vec<_>>(),
                    "tool_choice": ToolChoice::Auto,
                }),
            );
        }

        if let Some(ref params) = completion_request.additional_params {
            merge_inplace(&mut body, params.clone())
        }

        let body: Vec<u8> = serde_json::to_vec(&body)?;

        let req = self
            .client
            .post("/v1/messages")
            .header("Content-Type", "application/json")
            .body(body)
            .map_err(http_client::Error::Protocol)?;

        let response = self.client.send_streaming(req).await?;

        if !response.status().is_success() {
            let mut stream = response.into_body();
            let mut text = String::with_capacity(1024);
            loop {
                let Some(chunk) = stream.next().await else {
                    break;
                };

                let chunk: Vec<u8> = chunk?.into();

                let str = String::from_utf8_lossy(&chunk);

                text.push_str(&str)
            }
            return Err(CompletionError::ProviderError(text));
        }

        let stream = sse_from_response(response.into_body());

        // Use our SSE decoder to directly handle Server-Sent Events format
        let stream: StreamingResult<StreamingCompletionResponse> = Box::pin(stream! {
            let mut current_tool_call: Option<ToolCallState> = None;
            let mut current_thinking: Option<ThinkingState> = None;
            let mut sse_stream = Box::pin(stream);
            let mut input_tokens = 0;

            let mut text_content = String::new();

            while let Some(sse_result) = sse_stream.next().await {
                match sse_result {
                    Ok(sse) => {
                        // Parse the SSE data as a StreamingEvent
                        match serde_json::from_str::<StreamingEvent>(&sse.data) {
                            Ok(event) => {
                                match &event {
                                    StreamingEvent::MessageStart { message } => {
                                        input_tokens = message.usage.input_tokens;

                                        let span = tracing::Span::current();
                                        span.record("gen_ai.response.id", &message.id);
                                        span.record("gen_ai.response.model_name", &message.model);
                                    },
                                    StreamingEvent::MessageDelta { delta, usage } => {
                                        if delta.stop_reason.is_some() {
                                            let usage = PartialUsage {
                                                 output_tokens: usage.output_tokens,
                                                 input_tokens: Some(input_tokens.try_into().expect("Failed to convert input_tokens to usize")),
                                            };

                                            let span = tracing::Span::current();
                                            span.record_token_usage(&usage);
                                            span.record_model_output(&Message {
                                                role: super::completion::Role::Assistant,
                                                content: OneOrMany::one(Content::Text { text: text_content.clone() })}
                                            );

                                            yield Ok(RawStreamingChoice::FinalResponse(StreamingCompletionResponse {
                                                usage
                                            }))
                                        }
                                    }
                                    _ => {}
                                }

                                if let Some(result) = handle_event(&event, &mut current_tool_call, &mut current_thinking) {
                                    if let Ok(RawStreamingChoice::Message(ref text)) = result {
                                        text_content += text;
                                    }
                                    yield result;
                                }
                            },
                            Err(e) => {
                                if !sse.data.trim().is_empty() {
                                    yield Err(CompletionError::ResponseError(
                                        format!("Failed to parse JSON: {} (Data: {})", e, sse.data)
                                    ));
                                }
                            }
                        }
                    },
                    Err(e) => {
                        yield Err(CompletionError::ResponseError(format!("SSE Error: {e}")));
                        break;
                    }
                }
            }
        }.instrument(span));

        Ok(streaming::StreamingCompletionResponse::stream(stream))
    }
}

fn handle_event(
    event: &StreamingEvent,
    current_tool_call: &mut Option<ToolCallState>,
    current_thinking: &mut Option<ThinkingState>,
) -> Option<Result<RawStreamingChoice<StreamingCompletionResponse>, CompletionError>> {
    match event {
        StreamingEvent::ContentBlockDelta { delta, .. } => match delta {
            ContentDelta::TextDelta { text } => {
                if current_tool_call.is_none() {
                    return Some(Ok(RawStreamingChoice::Message(text.clone())));
                }
                None
            }
            ContentDelta::InputJsonDelta { partial_json } => {
                if let Some(tool_call) = current_tool_call {
                    tool_call.input_json.push_str(partial_json);
                }
                None
            }
            ContentDelta::ThinkingDelta { thinking } => {
                if current_thinking.is_none() {
                    *current_thinking = Some(ThinkingState::default());
                }

                if let Some(state) = current_thinking {
                    state.thinking.push_str(thinking);
                }

                Some(Ok(RawStreamingChoice::Reasoning {
                    id: None,
                    reasoning: thinking.clone(),
                    signature: None,
                }))
            }
            ContentDelta::SignatureDelta { signature } => {
                if current_thinking.is_none() {
                    *current_thinking = Some(ThinkingState::default());
                }

                if let Some(state) = current_thinking {
                    state.signature.push_str(signature);
                }

                // Don't yield signature chunks, they will be included in the final Reasoning
                None
            }
        },
        StreamingEvent::ContentBlockStart { content_block, .. } => match content_block {
            Content::ToolUse { id, name, .. } => {
                *current_tool_call = Some(ToolCallState {
                    name: name.clone(),
                    id: id.clone(),
                    input_json: String::new(),
                });
                None
            }
            Content::Thinking { .. } => {
                *current_thinking = Some(ThinkingState::default());
                None
            }
            // Handle other content types - they don't need special handling
            _ => None,
        },
        StreamingEvent::ContentBlockStop { .. } => {
            if let Some(thinking_state) = Option::take(current_thinking)
                && !thinking_state.thinking.is_empty()
            {
                let signature = if thinking_state.signature.is_empty() {
                    None
                } else {
                    Some(thinking_state.signature)
                };

                return Some(Ok(RawStreamingChoice::Reasoning {
                    id: None,
                    reasoning: thinking_state.thinking,
                    signature,
                }));
            }

            if let Some(tool_call) = Option::take(current_tool_call) {
                let json_str = if tool_call.input_json.is_empty() {
                    "{}"
                } else {
                    &tool_call.input_json
                };
                match serde_json::from_str(json_str) {
                    Ok(json_value) => Some(Ok(RawStreamingChoice::ToolCall {
                        name: tool_call.name,
                        id: tool_call.id,
                        arguments: json_value,
                        call_id: None,
                    })),
                    Err(e) => Some(Err(CompletionError::from(e))),
                }
            } else {
                None
            }
        }
        // Ignore other event types or handle as needed
        StreamingEvent::MessageStart { .. }
        | StreamingEvent::MessageDelta { .. }
        | StreamingEvent::MessageStop
        | StreamingEvent::Ping
        | StreamingEvent::Unknown => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thinking_delta_deserialization() {
        let json = r#"{"type": "thinking_delta", "thinking": "Let me think about this..."}"#;
        let delta: ContentDelta = serde_json::from_str(json).unwrap();

        match delta {
            ContentDelta::ThinkingDelta { thinking } => {
                assert_eq!(thinking, "Let me think about this...");
            }
            _ => panic!("Expected ThinkingDelta variant"),
        }
    }

    #[test]
    fn test_signature_delta_deserialization() {
        let json = r#"{"type": "signature_delta", "signature": "abc123def456"}"#;
        let delta: ContentDelta = serde_json::from_str(json).unwrap();

        match delta {
            ContentDelta::SignatureDelta { signature } => {
                assert_eq!(signature, "abc123def456");
            }
            _ => panic!("Expected SignatureDelta variant"),
        }
    }

    #[test]
    fn test_thinking_delta_streaming_event_deserialization() {
        let json = r#"{
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "thinking_delta",
                "thinking": "First, I need to understand the problem."
            }
        }"#;

        let event: StreamingEvent = serde_json::from_str(json).unwrap();

        match event {
            StreamingEvent::ContentBlockDelta { index, delta } => {
                assert_eq!(index, 0);
                match delta {
                    ContentDelta::ThinkingDelta { thinking } => {
                        assert_eq!(thinking, "First, I need to understand the problem.");
                    }
                    _ => panic!("Expected ThinkingDelta"),
                }
            }
            _ => panic!("Expected ContentBlockDelta event"),
        }
    }

    #[test]
    fn test_signature_delta_streaming_event_deserialization() {
        let json = r#"{
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "signature_delta",
                "signature": "ErUBCkYICBgCIkCaGbqC85F4"
            }
        }"#;

        let event: StreamingEvent = serde_json::from_str(json).unwrap();

        match event {
            StreamingEvent::ContentBlockDelta { index, delta } => {
                assert_eq!(index, 0);
                match delta {
                    ContentDelta::SignatureDelta { signature } => {
                        assert_eq!(signature, "ErUBCkYICBgCIkCaGbqC85F4");
                    }
                    _ => panic!("Expected SignatureDelta"),
                }
            }
            _ => panic!("Expected ContentBlockDelta event"),
        }
    }

    #[test]
    fn test_handle_thinking_delta_event() {
        let event = StreamingEvent::ContentBlockDelta {
            index: 0,
            delta: ContentDelta::ThinkingDelta {
                thinking: "Analyzing the request...".to_string(),
            },
        };

        let mut tool_call_state = None;
        let mut thinking_state = None;
        let result = handle_event(&event, &mut tool_call_state, &mut thinking_state);

        assert!(result.is_some());
        let choice = result.unwrap().unwrap();

        match choice {
            RawStreamingChoice::Reasoning { id, reasoning, .. } => {
                assert_eq!(id, None);
                assert_eq!(reasoning, "Analyzing the request...");
            }
            _ => panic!("Expected Reasoning choice"),
        }

        // Verify thinking state was updated
        assert!(thinking_state.is_some());
        assert_eq!(thinking_state.unwrap().thinking, "Analyzing the request...");
    }

    #[test]
    fn test_handle_signature_delta_event() {
        let event = StreamingEvent::ContentBlockDelta {
            index: 0,
            delta: ContentDelta::SignatureDelta {
                signature: "test_signature".to_string(),
            },
        };

        let mut tool_call_state = None;
        let mut thinking_state = None;
        let result = handle_event(&event, &mut tool_call_state, &mut thinking_state);

        // SignatureDelta should not yield anything (returns None)
        assert!(result.is_none());

        // But signature should be captured in thinking state
        assert!(thinking_state.is_some());
        assert_eq!(thinking_state.unwrap().signature, "test_signature");
    }

    #[test]
    fn test_handle_text_delta_event() {
        let event = StreamingEvent::ContentBlockDelta {
            index: 0,
            delta: ContentDelta::TextDelta {
                text: "Hello, world!".to_string(),
            },
        };

        let mut tool_call_state = None;
        let mut thinking_state = None;
        let result = handle_event(&event, &mut tool_call_state, &mut thinking_state);

        assert!(result.is_some());
        let choice = result.unwrap().unwrap();

        match choice {
            RawStreamingChoice::Message(text) => {
                assert_eq!(text, "Hello, world!");
            }
            _ => panic!("Expected Message choice"),
        }
    }

    #[test]
    fn test_thinking_delta_does_not_interfere_with_tool_calls() {
        // Thinking deltas should still be processed even if a tool call is in progress
        let event = StreamingEvent::ContentBlockDelta {
            index: 0,
            delta: ContentDelta::ThinkingDelta {
                thinking: "Thinking while tool is active...".to_string(),
            },
        };

        let mut tool_call_state = Some(ToolCallState {
            name: "test_tool".to_string(),
            id: "tool_123".to_string(),
            input_json: String::new(),
        });
        let mut thinking_state = None;

        let result = handle_event(&event, &mut tool_call_state, &mut thinking_state);

        assert!(result.is_some());
        let choice = result.unwrap().unwrap();

        match choice {
            RawStreamingChoice::Reasoning { reasoning, .. } => {
                assert_eq!(reasoning, "Thinking while tool is active...");
            }
            _ => panic!("Expected Reasoning choice"),
        }

        // Tool call state should remain unchanged
        assert!(tool_call_state.is_some());
    }
}
