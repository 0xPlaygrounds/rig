use crate::completion::{CompletionError, CompletionRequest, GetTokenUsage};
use crate::providers::cohere::CompletionModel;
use crate::providers::cohere::completion::Usage;
use crate::streaming::RawStreamingChoice;
use crate::{json_utils, streaming};
use async_stream::stream;
use futures::StreamExt;
use reqwest_eventsource::Event;
use serde::{Deserialize, Serialize};

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
        let request = json_utils::merge(request, serde_json::json!({"stream": true}));

        tracing::debug!(
            "Cohere request: {}",
            serde_json::to_string_pretty(&request)?
        );

        let req = self.client.client().post("/v2/chat").json(&request);

        let mut event_source = self
            .client
            .eventsource(req)
            .await
            .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

        let stream = Box::pin(stream! {
            let mut current_tool_call: Option<(String, String, String)> = None;

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

                                yield Ok(RawStreamingChoice::Message(text.clone()));
                            },

                            StreamingEvent::MessageEnd { delta: Some(delta) } => {
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
                                let Ok(args) = serde_json::from_str(&tc.2) else { continue; };

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
        });

        Ok(streaming::StreamingCompletionResponse::stream(stream))
    }
}
