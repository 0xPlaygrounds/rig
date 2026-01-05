use crate::completion::{CompletionError, CompletionRequest, GetTokenUsage};
use crate::http_client::HttpClientExt;
use crate::http_client::sse::{Event, GenericEventSource};
use crate::providers::cohere::CompletionModel;
use crate::providers::cohere::completion::{
    AssistantContent, CohereCompletionRequest, Message, ToolCall, ToolCallFunction, ToolType, Usage,
};
use crate::streaming::{RawStreamingChoice, RawStreamingToolCall, ToolCallDeltaContent};
use crate::telemetry::SpanCombinator;
use crate::{json_utils, streaming};
use async_stream::stream;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tracing::{Level, enabled, info_span};
use tracing_futures::Instrument;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "kebab-case", tag = "type")]
enum StreamingEvent {
    MessageStart,
    ContentStart,
    ContentDelta { delta: Option<Delta> },
    ContentEnd,
    ToolPlan,
    ToolCallStart { delta: Option<Delta> },
    ToolCallDelta { delta: Option<Delta> },
    ToolCallEnd,
    MessageEnd { delta: Option<MessageEndDelta> },
}

#[derive(Debug, Deserialize)]
struct MessageContentDelta {
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MessageToolFunctionDelta {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MessageToolCallDelta {
    id: Option<String>,
    function: Option<MessageToolFunctionDelta>,
}

#[derive(Debug, Deserialize)]
struct MessageDelta {
    content: Option<MessageContentDelta>,
    tool_calls: Option<MessageToolCallDelta>,
}

#[derive(Debug, Deserialize)]
struct Delta {
    message: Option<MessageDelta>,
}

#[derive(Debug, Deserialize)]
struct MessageEndDelta {
    usage: Option<Usage>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct StreamingCompletionResponse {
    pub usage: Option<Usage>,
}

impl GetTokenUsage for StreamingCompletionResponse {
    fn token_usage(&self) -> Option<crate::completion::Usage> {
        let tokens = self
            .usage
            .clone()
            .and_then(|response| response.tokens)
            .map(|tokens| {
                (
                    tokens.input_tokens.map(|x| x as u64),
                    tokens.output_tokens.map(|y| y as u64),
                )
            });
        let Some((Some(input), Some(output))) = tokens else {
            return None;
        };
        let mut usage = crate::completion::Usage::new();
        usage.input_tokens = input;
        usage.output_tokens = output;
        usage.total_tokens = input + output;

        Some(usage)
    }
}

impl<T> CompletionModel<T>
where
    T: HttpClientExt + Clone + 'static,
{
    pub(crate) async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<streaming::StreamingCompletionResponse<StreamingCompletionResponse>, CompletionError>
    {
        let mut request = CohereCompletionRequest::try_from((self.model.as_ref(), request))?;
        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat_streaming",
                gen_ai.operation.name = "chat_streaming",
                gen_ai.provider.name = "cohere",
                gen_ai.request.model = self.model,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = self.model,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        let params = json_utils::merge(
            request.additional_params.unwrap_or(serde_json::json!({})),
            serde_json::json!({"stream": true}),
        );

        request.additional_params = Some(params);

        if enabled!(Level::TRACE) {
            tracing::trace!(
                target: "rig::streaming",
                "Cohere streaming completion input: {}",
                serde_json::to_string_pretty(&request)?
            );
        }

        let body = serde_json::to_vec(&request)?;

        let req = self.client.post("/v2/chat")?.body(body).unwrap();

        let mut event_source = GenericEventSource::new(self.client.clone(), req);

        let stream = stream! {
            let mut current_tool_call: Option<(String, String, String)> = None;
            let mut text_response = String::new();
            let mut tool_calls = Vec::new();
            let mut final_usage = None;

            while let Some(event_result) = event_source.next().await {
                match event_result {
                    Ok(Event::Open) => {
                        tracing::trace!("SSE connection opened");
                        continue;
                    }

                    Ok(Event::Message(message)) => {
                        let data_str = message.data.trim();
                        if data_str.is_empty() || data_str == "[DONE]" {
                            continue;
                        }

                        let event: StreamingEvent = match serde_json::from_str(data_str) {
                            Ok(ev) => ev,
                            Err(_) => {
                                tracing::debug!("Couldn't parse SSE payload as StreamingEvent");
                                continue;
                            }
                        };

                        match event {
                            StreamingEvent::ContentDelta { delta: Some(delta) } => {
                                let Some(message) = &delta.message else { continue; };
                                let Some(content) = &message.content else { continue; };
                                let Some(text) = &content.text else { continue; };

                                text_response += text;

                                yield Ok(RawStreamingChoice::Message(text.clone()));
                            },

                            StreamingEvent::MessageEnd { delta: Some(delta) } => {
                                let message = Message::Assistant {
                                    tool_calls: tool_calls.clone(),
                                    content: vec![AssistantContent::Text { text: text_response.clone() }],
                                    tool_plan: None,
                                    citations: vec![]
                                };

                                let span = tracing::Span::current();
                                span.record_token_usage(&delta.usage);
                                span.record_model_output(&vec![message]);

                                final_usage = Some(delta.usage.clone());
                                break;
                            },

                            StreamingEvent::ToolCallStart { delta: Some(delta) } => {
                                let Some(message) = &delta.message else { continue; };
                                let Some(tool_calls) = &message.tool_calls else { continue; };
                                let Some(id) = tool_calls.id.clone() else { continue; };
                                let Some(function) = &tool_calls.function else { continue; };
                                let Some(name) = function.name.clone() else { continue; };
                                let Some(arguments) = function.arguments.clone() else { continue; };

                                current_tool_call = Some((id.clone(), name.clone(), arguments));

                                yield Ok(RawStreamingChoice::ToolCallDelta {
                                    id,
                                    content: ToolCallDeltaContent::Name(name),
                                });
                            },

                            StreamingEvent::ToolCallDelta { delta: Some(delta) } => {
                                let Some(message) = &delta.message else { continue; };
                                let Some(tool_calls) = &message.tool_calls else { continue; };
                                let Some(function) = &tool_calls.function else { continue; };
                                let Some(arguments) = function.arguments.clone() else { continue; };

                                let Some(tc) = current_tool_call.clone() else { continue; };
                                current_tool_call = Some((tc.0.clone(), tc.1, format!("{}{}", tc.2, arguments)));

                                // Emit the delta so UI can show progress
                                yield Ok(RawStreamingChoice::ToolCallDelta {
                                    id: tc.0,
                                    content: ToolCallDeltaContent::Delta(arguments),
                                });
                            },

                            StreamingEvent::ToolCallEnd => {
                                let Some(tc) = current_tool_call.clone() else { continue; };
                                let Ok(args) = serde_json::from_str::<serde_json::Value>(&tc.2) else { continue; };

                                tool_calls.push(ToolCall {
                                    id: Some(tc.0.clone()),
                                    r#type: Some(ToolType::Function),
                                    function: Some(ToolCallFunction {
                                        name: tc.1.clone(),
                                        arguments: args.clone()
                                    })
                                });

                                yield Ok(RawStreamingChoice::ToolCall(
                                    RawStreamingToolCall::new(tc.0, tc.1, args)
                                ));

                                current_tool_call = None;
                            },

                            _ => {}
                        }
                    },
                    Err(crate::http_client::Error::StreamEnded) => {
                        break;
                    }
                    Err(err) => {
                        tracing::error!(?err, "SSE error");
                        yield Err(CompletionError::ProviderError(err.to_string()));
                        break;
                    }
                }
            }

            // Ensure event source is closed when stream ends
            event_source.close();

            yield Ok(RawStreamingChoice::FinalResponse(StreamingCompletionResponse {
                usage: final_usage.unwrap_or_default()
            }))
        }.instrument(span);

        Ok(streaming::StreamingCompletionResponse::stream(Box::pin(
            stream,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_message_content_delta_deserialization() {
        let json = json!({
            "type": "content-delta",
            "delta": {
                "message": {
                    "content": {
                        "text": "Hello world"
                    }
                }
            }
        });

        let event: StreamingEvent = serde_json::from_value(json).unwrap();
        match event {
            StreamingEvent::ContentDelta { delta } => {
                assert!(delta.is_some());
                let message = delta.unwrap().message.unwrap();
                let content = message.content.unwrap();
                assert_eq!(content.text, Some("Hello world".to_string()));
            }
            _ => panic!("Expected ContentDelta"),
        }
    }

    #[test]
    fn test_tool_call_start_deserialization() {
        let json = json!({
            "type": "tool-call-start",
            "delta": {
                "message": {
                    "tool_calls": {
                        "id": "call_123",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{"
                        }
                    }
                }
            }
        });

        let event: StreamingEvent = serde_json::from_value(json).unwrap();
        match event {
            StreamingEvent::ToolCallStart { delta } => {
                assert!(delta.is_some());
                let tool_call = delta.unwrap().message.unwrap().tool_calls.unwrap();
                assert_eq!(tool_call.id, Some("call_123".to_string()));
                assert_eq!(
                    tool_call.function.unwrap().name,
                    Some("get_weather".to_string())
                );
            }
            _ => panic!("Expected ToolCallStart"),
        }
    }

    #[test]
    fn test_tool_call_delta_deserialization() {
        let json = json!({
            "type": "tool-call-delta",
            "delta": {
                "message": {
                    "tool_calls": {
                        "function": {
                            "arguments": "\"location\""
                        }
                    }
                }
            }
        });

        let event: StreamingEvent = serde_json::from_value(json).unwrap();
        match event {
            StreamingEvent::ToolCallDelta { delta } => {
                assert!(delta.is_some());
                let tool_call = delta.unwrap().message.unwrap().tool_calls.unwrap();
                let function = tool_call.function.unwrap();
                assert_eq!(function.arguments, Some("\"location\"".to_string()));
            }
            _ => panic!("Expected ToolCallDelta"),
        }
    }

    #[test]
    fn test_tool_call_end_deserialization() {
        let json = json!({
            "type": "tool-call-end"
        });

        let event: StreamingEvent = serde_json::from_value(json).unwrap();
        match event {
            StreamingEvent::ToolCallEnd => {
                // Success
            }
            _ => panic!("Expected ToolCallEnd"),
        }
    }

    #[test]
    fn test_message_end_with_usage_deserialization() {
        let json = json!({
            "type": "message-end",
            "delta": {
                "usage": {
                    "tokens": {
                        "input_tokens": 100,
                        "output_tokens": 50
                    }
                }
            }
        });

        let event: StreamingEvent = serde_json::from_value(json).unwrap();
        match event {
            StreamingEvent::MessageEnd { delta } => {
                assert!(delta.is_some());
                let usage = delta.unwrap().usage.unwrap();
                let tokens = usage.tokens.unwrap();
                assert_eq!(tokens.input_tokens, Some(100.0));
                assert_eq!(tokens.output_tokens, Some(50.0));
            }
            _ => panic!("Expected MessageEnd"),
        }
    }

    #[test]
    fn test_streaming_event_order() {
        // Test that a typical sequence of events deserializes correctly
        let events = vec![
            json!({"type": "message-start"}),
            json!({"type": "content-start"}),
            json!({
                "type": "content-delta",
                "delta": {
                    "message": {
                        "content": {
                            "text": "Sure, "
                        }
                    }
                }
            }),
            json!({
                "type": "content-delta",
                "delta": {
                    "message": {
                        "content": {
                            "text": "I can help with that."
                        }
                    }
                }
            }),
            json!({"type": "content-end"}),
            json!({"type": "tool-plan"}),
            json!({
                "type": "tool-call-start",
                "delta": {
                    "message": {
                        "tool_calls": {
                            "id": "call_abc",
                            "function": {
                                "name": "search",
                                "arguments": ""
                            }
                        }
                    }
                }
            }),
            json!({
                "type": "tool-call-delta",
                "delta": {
                    "message": {
                        "tool_calls": {
                            "function": {
                                "arguments": "{\"query\":"
                            }
                        }
                    }
                }
            }),
            json!({
                "type": "tool-call-delta",
                "delta": {
                    "message": {
                        "tool_calls": {
                            "function": {
                                "arguments": "\"Rust\"}"
                            }
                        }
                    }
                }
            }),
            json!({"type": "tool-call-end"}),
            json!({
                "type": "message-end",
                "delta": {
                    "usage": {
                        "tokens": {
                            "input_tokens": 50,
                            "output_tokens": 25
                        }
                    }
                }
            }),
        ];

        for (i, event_json) in events.iter().enumerate() {
            let result = serde_json::from_value::<StreamingEvent>(event_json.clone());
            assert!(
                result.is_ok(),
                "Failed to deserialize event at index {}: {:?}",
                i,
                result.err()
            );
        }
    }
}
