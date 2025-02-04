use async_stream::stream;
use futures::StreamExt;
use serde::Deserialize;
use serde_json::json;
use std::iter;

use super::completion::{CompletionModel, Content, Message, ToolChoice, ToolDefinition, Usage};
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
        usage: Usage,
    },
    MessageStop,
    Ping,
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
        let prompt_with_context = completion_request.prompt_with_context();

        let max_tokens = if let Some(tokens) = completion_request.max_tokens {
            tokens
        } else if let Some(tokens) = self.default_max_tokens {
            tokens
        } else {
            return Err(CompletionError::RequestError(
                "`max_tokens` must be set for Anthropic".into(),
            ));
        };

        let mut request = json!({
            "model": self.model,
            "messages": completion_request
                .chat_history
                .into_iter()
                .map(Message::from)
                .chain(iter::once(Message {
                    role: "user".to_owned(),
                    content: prompt_with_context,
                }))
                .collect::<Vec<_>>(),
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

        Ok(Box::pin(stream! {
            let mut current_tool_call: Option<ToolCallState> = None;
            let mut stream = response.bytes_stream();

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
                    if let Some(data) = line.strip_prefix("data: ") {
                        if let Ok(event) = serde_json::from_str::<StreamingEvent>(data) {
                            match event {
                                StreamingEvent::ContentBlockDelta { delta, .. } => {
                                    match delta {
                                        ContentDelta::TextDelta { text } => {
                                            if current_tool_call.is_none() {
                                                yield Ok(StreamingChoice::Message(text));
                                            }
                                        }
                                        ContentDelta::InputJsonDelta { partial_json } => {
                                            if let Some(ref mut tool_call) = current_tool_call {
                                                tool_call.input_json.push_str(&partial_json);
                                            }
                                        }
                                    }
                                }
                                StreamingEvent::ContentBlockStart {
                                    content_block: Content::ToolUse { id, name, .. },
                                    ..
                                } => {
                                    current_tool_call = Some(ToolCallState {
                                        name,
                                        id,
                                        input_json: String::new(),
                                    });
                                }
                                StreamingEvent::ContentBlockStop { .. } => {
                                    if let Some(tool_call) = current_tool_call.take() {
                                        let json_str = if tool_call.input_json.is_empty() {
                                            "{}"
                                        } else {
                                            &tool_call.input_json
                                        };
                                        match serde_json::from_str(json_str) {
                                            Ok(json_value) => {
                                                yield Ok(StreamingChoice::ToolCall(
                                                    tool_call.name,
                                                    tool_call.id,
                                                    json_value,
                                                ));
                                            }
                                            Err(e) => {
                                                yield Err(CompletionError::from(e));
                                            }
                                        }
                                    }
                                },
                                _ => {}
                            }
                        }
                    }
                }
            }
        }))
    }
}
