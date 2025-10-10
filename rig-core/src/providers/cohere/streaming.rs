use crate::completion::{CompletionError, CompletionRequest, GetTokenUsage};
use crate::providers::cohere::CompletionModel;
use crate::providers::cohere::completion::{
    AssistantContent, Message, ToolCall, ToolCallFunction, ToolType, Usage,
};
use crate::streaming::RawStreamingChoice;
use crate::telemetry::SpanCombinator;
use crate::{json_utils, streaming};
use async_stream::stream;
use futures::StreamExt;
use reqwest_eventsource::Event;
use serde::{Deserialize, Serialize};
use tracing::info_span;
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

impl CompletionModel<reqwest::Client> {
    pub(crate) async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<streaming::StreamingCompletionResponse<StreamingCompletionResponse>, CompletionError>
    {
        let request = self.create_completion_request(request)?;
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
                gen_ai.input.messages = serde_json::to_string(&request.get("messages").unwrap()).unwrap(),
                gen_ai.output.messages = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        let request = json_utils::merge(request, serde_json::json!({"stream": true}));

        tracing::debug!(
            "Cohere streaming completion input: {}",
            serde_json::to_string_pretty(&request)?
        );

        let req = self.client.client().post("/v2/chat").json(&request);

        let mut event_source = self
            .client
            .eventsource(req)
            .await
            .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

        let stream = stream! {
            let mut current_tool_call: Option<(String, String, String)> = None;
            let mut text_response = String::new();
            let mut tool_calls = Vec::new();

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

                                yield Ok(RawStreamingChoice::FinalResponse(StreamingCompletionResponse {
                                    usage: delta.usage.clone()
                                }));
                            },

                            StreamingEvent::ToolCallStart { delta: Some(delta) } => {
                                let Some(message) = &delta.message else { continue; };
                                let Some(tool_calls) = &message.tool_calls else { continue; };
                                let Some(id) = tool_calls.id.clone() else { continue; };
                                let Some(function) = &tool_calls.function else { continue; };
                                let Some(name) = function.name.clone() else { continue; };
                                let Some(arguments) = function.arguments.clone() else { continue; };

                                current_tool_call = Some((id, name, arguments));
                            },

                            StreamingEvent::ToolCallDelta { delta: Some(delta) } => {
                                let Some(message) = &delta.message else { continue; };
                                let Some(tool_calls) = &message.tool_calls else { continue; };
                                let Some(function) = &tool_calls.function else { continue; };
                                let Some(arguments) = function.arguments.clone() else { continue; };

                                let Some(tc) = current_tool_call.clone() else { continue; };
                                current_tool_call = Some((tc.0, tc.1, format!("{}{}", tc.2, arguments)));
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

                                yield Ok(RawStreamingChoice::ToolCall {
                                    id: tc.0,
                                    name: tc.1,
                                    arguments: args,
                                    call_id: None
                                });

                                current_tool_call = None;
                            },

                            _ => {}
                        }
                    },

                    Err(reqwest_eventsource::Error::StreamEnded) => break,

                    Err(err) => {
                        tracing::error!(?err, "SSE error");
                        yield Err(CompletionError::ResponseError(err.to_string()));
                        break;
                    }
                }
            }

            event_source.close();
        }.instrument(span);

        Ok(streaming::StreamingCompletionResponse::stream(Box::pin(
            stream,
        )))
    }
}
