use crate::completion::{CompletionError, CompletionRequest, GetTokenUsage};
use crate::providers::cohere::CompletionModel;
use crate::providers::cohere::completion::Usage;
use crate::streaming::RawStreamingChoice;
use crate::{json_utils, streaming};
use async_stream::stream;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::json;

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

impl CompletionModel {
    pub(crate) async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<streaming::StreamingCompletionResponse<StreamingCompletionResponse>, CompletionError>
    {
        let request = self.create_completion_request(request)?;
        let request = json_utils::merge(request, json!({"stream": true}));

        tracing::debug!(
            "Cohere request: {}",
            serde_json::to_string_pretty(&request)?
        );

        let response = self.client.post("/v2/chat").json(&request).send().await?;

        if !response.status().is_success() {
            return Err(CompletionError::ProviderError(format!(
                "{}: {}",
                response.status(),
                response.text().await?
            )));
        }

        let stream = Box::pin(stream! {
            let mut stream = response.bytes_stream();
            let mut current_tool_call: Option<(String, String, String)> = None;

            while let Some(chunk_result) = stream.next().await {
               let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => {
                        yield Err(CompletionError::from(e));
                        break;
                    }
                };

               let text = match String::from_utf8(chunk.to_vec()) {
                    Ok(t) => t,
                    Err(e) => {
                        yield Err(CompletionError::ResponseError(e.to_string()));
                        break;
                    }
               };

                for line in text.lines() {

                    let Some(line) = line.strip_prefix("data: ") else {
                        continue;
                    };

                    let event = {
                       let result = serde_json::from_str::<StreamingEvent>(line);

                       let Ok(event) = result else {
                           continue;
                       };

                        event
                    };

                    match event {
                        StreamingEvent::ContentDelta { delta: Some(delta) } => {
                            let Some(message) = &delta.message else { continue; };
                            let Some(content) = &message.content else { continue; };
                            let Some(text) = &content.text else { continue; };

                            yield Ok(RawStreamingChoice::Message(text.clone()));
                        },
                        StreamingEvent::MessageEnd {delta: Some(delta)} => {
                            yield Ok(RawStreamingChoice::FinalResponse(StreamingCompletionResponse {
                                usage: delta.usage.clone()
                            }));
                        },
                        StreamingEvent::ToolCallStart { delta: Some(delta)} => {
                            // Skip the delta if there's any missing information,
                            // though this *should* all be present
                            let Some(message) = &delta.message else { continue; };
                            let Some(tool_calls) = &message.tool_calls else { continue; };
                            let Some(id) = tool_calls.id.clone() else { continue; };
                            let Some(function) = &tool_calls.function else { continue; };
                            let Some(name) = function.name.clone() else { continue; };
                            let Some(arguments) = function.arguments.clone() else { continue; };

                            current_tool_call = Some((id, name, arguments));
                        },
                        StreamingEvent::ToolCallDelta { delta: Some(delta)} => {
                            // Skip the delta if there's any missing information,
                            // though this *should* all be present
                            let Some(message) = &delta.message else { continue; };
                            let Some(tool_calls) = &message.tool_calls else { continue; };
                            let Some(function) = &tool_calls.function else { continue; };
                            let Some(arguments) = function.arguments.clone() else { continue; };

                            if let Some(tc) = current_tool_call.clone() {
                                current_tool_call = Some((
                                    tc.0,
                                    tc.1,
                                    format!("{}{}", tc.2, arguments)
                                ));
                            };
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
                    };
                }
            }
        });

        Ok(streaming::StreamingCompletionResponse::stream(stream))
    }
}
