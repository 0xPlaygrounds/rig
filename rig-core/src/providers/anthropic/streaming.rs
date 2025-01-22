use futures::{Stream, StreamExt};
use serde::Deserialize;
use std::pin::Pin;
use tokio_stream::wrappers::ReceiverStream;

use super::completion::{CompletionModel, Content, Usage};
use crate::completion::{CompletionError, CompletionRequest};
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
        request: CompletionRequest,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<StreamingChoice, CompletionError>> + Send>>,
        CompletionError,
    > {
        let prompt_with_context = request.prompt_with_context();

        // Similar to the completion implementation, but with stream: true
        let max_tokens = request.max_tokens.or(Some(2048)).ok_or_else(|| {
            CompletionError::RequestError("`max_tokens` must be set for Anthropic".into())
        })?;

        let mut body = serde_json::json!({
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": prompt_with_context,
            }],
            "max_tokens": max_tokens,
            "stream": true,
        });

        if let Some(temperature) = request.temperature {
            body["temperature"] = serde_json::json!(temperature);
        }

        if !request.tools.is_empty() {
            body["tools"] = serde_json::json!(request.tools);
            body["tool_choice"] = serde_json::json!({"type": "any"});
        }

        let response = self.client.post("/v1/messages").json(&body).send().await?;

        if !response.status().is_success() {
            return Err(CompletionError::ProviderError(response.text().await?));
        }

        let mut stream = response.bytes_stream();
        let (tx, rx) = tokio::sync::mpsc::channel(100);

        // Spawn a task to process the SSE stream
        tokio::spawn(async move {
            let mut current_text = String::new();

            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        if let Ok(text) = String::from_utf8(chunk.to_vec()) {
                            for line in text.lines() {
                                if let Some(data) = line.strip_prefix("data: ") {
                                    if let Ok(event) = serde_json::from_str::<StreamingEvent>(data)
                                    {
                                        match event {
                                            StreamingEvent::ContentBlockDelta { delta, .. } => {
                                                if let ContentDelta::TextDelta { text } = delta {
                                                    current_text.push_str(&text);
                                                    let _ = tx
                                                        .send(Ok(StreamingChoice::Message(text)))
                                                        .await;
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
