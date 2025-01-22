use futures::{Stream, StreamExt};
use serde::Deserialize;
use serde_json::json;
use std::iter;
use std::pin::Pin;
use tokio_stream::wrappers::ReceiverStream;

use super::completion::{CompletionModel, Content, Message, ToolChoice, ToolDefinition, Usage};
use crate::completion::{CompletionError, CompletionRequest};
use crate::json_utils;
use crate::streaming::{StreamingChoice, StreamingCompletionModel};

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamingEvent {
    MessageStart {
        message: MessageStart,
    },
    ContentBlockStart {
        index: usize,
        content_block: ContentBlock,
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
pub enum ContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
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

impl StreamingCompletionModel for CompletionModel {
    async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<StreamingChoice, CompletionError>> + Send>>,
        CompletionError,
    > {
        // Similar setup to the completion implementation
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
            json_utils::merge_inplace(&mut request, json!({ "temperature": temperature }));
        }

        // Add tools configuration similar to completion implementation
        if !completion_request.tools.is_empty() {
            json_utils::merge_inplace(
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
            json_utils::merge_inplace(&mut request, params.clone())
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

        let mut stream = response.bytes_stream();
        let (tx, rx) = tokio::sync::mpsc::channel(100);

        // Spawn a task to process the SSE stream
        tokio::spawn(async move {
            let mut current_tool_call: Option<(String, String, String)> = None;

            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        if let Ok(text) = String::from_utf8(chunk.to_vec()) {
                            for line in text.lines() {
                                if let Some(data) = line.strip_prefix("data: ") {
                                    if data.trim() == "[DONE]" {
                                        break;
                                    }

                                    if let Ok(event) = serde_json::from_str::<StreamingEvent>(data)
                                    {
                                        match event {
                                            StreamingEvent::ContentBlockDelta { delta, .. } => {
                                                match delta {
                                                    ContentDelta::TextDelta { text } => {
                                                        let _ = tx
                                                            .send(Ok(StreamingChoice::Message(
                                                                text,
                                                            )))
                                                            .await;
                                                    }
                                                    ContentDelta::InputJsonDelta {
                                                        partial_json,
                                                    } => {
                                                        if let Some((name, id, mut input)) =
                                                            current_tool_call.clone()
                                                        {
                                                            input.push_str(&partial_json);
                                                            let _ = tx
                                                                .send(Ok(
                                                                    StreamingChoice::ToolCall(
                                                                        name,
                                                                        id,
                                                                        serde_json::from_str(
                                                                            &input,
                                                                        )
                                                                        .unwrap_or(json!({})),
                                                                    ),
                                                                ))
                                                                .await;
                                                        }
                                                    }
                                                }
                                            }
                                            StreamingEvent::ContentBlockStart {
                                                content_block,
                                                ..
                                            } => {
                                                if let ContentBlock::ToolUse { name, id, .. } =
                                                    content_block
                                                {
                                                    current_tool_call =
                                                        Some((name, id, String::new()));
                                                }
                                            }
                                            _ => continue,
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(CompletionError::from(e))).await;
                        break;
                    }
                }
            }
        });

        Ok(Box::pin(ReceiverStream::new(rx)))
    }
}
