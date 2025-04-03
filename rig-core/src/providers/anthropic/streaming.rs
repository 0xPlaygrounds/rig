use async_stream::stream;
use futures::StreamExt;
use serde::Deserialize;
use serde_json::json;

use super::completion::{CompletionModel, Content, Message, ToolChoice, ToolDefinition, Usage};
use super::decoders::sse::from_response as sse_from_response;
use crate::completion::{CompletionError, CompletionRequest};
use crate::json_utils::merge_inplace;
use crate::streaming::{StreamingChoice, StreamingCompletionModel, StreamingResult};

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
}

#[derive(Debug, Deserialize)]
pub struct MessageDelta {
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct PartialUsage {
    pub output_tokens: usize,
    #[serde(default)]
    pub input_tokens: Option<usize>,
}

#[derive(Default)]
struct ToolCallState {
    name: String,
    id: String,
    input_json: String,
}

impl StreamingCompletionModel for CompletionModel {
    async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<StreamingResult, CompletionError> {
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

        let full_history = full_history
            .into_iter()
            .map(Message::try_from)
            .collect::<Result<Vec<Message>, _>>()?;

        let mut request = json!({
            "model": self.model,
            "messages": full_history,
            "max_tokens": max_tokens,
            "system": completion_request.preamble.unwrap_or("".to_string()),
            "stream": true,
        });

        if let Some(temperature) = completion_request.temperature {
            merge_inplace(&mut request, json!({ "temperature": temperature }));
        }

        if !completion_request.tools.is_empty() {
            merge_inplace(
                &mut request,
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
            merge_inplace(&mut request, params.clone())
        }

        let response = self
            .client
            .post("/v1/messages")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(CompletionError::ProviderError(response.text().await?));
        }

        // Use our SSE decoder to directly handle Server-Sent Events format
        let sse_stream = sse_from_response(response);

        Ok(Box::pin(stream! {
            let mut current_tool_call: Option<ToolCallState> = None;
            let mut sse_stream = Box::pin(sse_stream);

            while let Some(sse_result) = sse_stream.next().await {
                match sse_result {
                    Ok(sse) => {
                        // Parse the SSE data as a StreamingEvent
                        match serde_json::from_str::<StreamingEvent>(&sse.data) {
                            Ok(event) => {
                                if let Some(result) = handle_event(&event, &mut current_tool_call) {
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
                        yield Err(CompletionError::ResponseError(format!("SSE Error: {}", e)));
                        break;
                    }
                }
            }
        }))
    }
}

fn handle_event(
    event: &StreamingEvent,
    current_tool_call: &mut Option<ToolCallState>,
) -> Option<Result<StreamingChoice, CompletionError>> {
    match event {
        StreamingEvent::ContentBlockDelta { delta, .. } => match delta {
            ContentDelta::TextDelta { text } => {
                if current_tool_call.is_none() {
                    return Some(Ok(StreamingChoice::Message(text.clone())));
                }
                None
            }
            ContentDelta::InputJsonDelta { partial_json } => {
                if let Some(ref mut tool_call) = current_tool_call {
                    tool_call.input_json.push_str(partial_json);
                }
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
            // Handle other content types - they don't need special handling
            _ => None,
        },
        StreamingEvent::ContentBlockStop { .. } => {
            if let Some(tool_call) = current_tool_call.take() {
                let json_str = if tool_call.input_json.is_empty() {
                    "{}"
                } else {
                    &tool_call.input_json
                };
                match serde_json::from_str(json_str) {
                    Ok(json_value) => Some(Ok(StreamingChoice::ToolCall(
                        tool_call.name,
                        tool_call.id,
                        json_value,
                    ))),
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
