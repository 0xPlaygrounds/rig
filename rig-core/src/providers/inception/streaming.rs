use async_stream::stream;
use futures::StreamExt;
use serde::Deserialize;
use serde_json::json;

use super::completion::{CompletionModel, Message};
use crate::completion::{CompletionError, CompletionRequest};
use crate::json_utils::merge_inplace;
use crate::message::MessageError;
use crate::providers::anthropic::decoders::sse::from_response as sse_from_response;
use crate::streaming::{self, StreamingCompletionModel, StreamingResult};

#[derive(Debug, Deserialize)]
pub struct StreamingResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<StreamingChoice>,
}

#[derive(Debug, Deserialize)]
pub struct StreamingChoice {
    pub index: usize,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Delta {
    pub content: Option<String>,
    pub role: Option<String>,
}

impl StreamingCompletionModel for CompletionModel {
    async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<StreamingResult, CompletionError> {
        let prompt_message: Message = completion_request
            .prompt_with_context()
            .try_into()
            .map_err(|e: MessageError| CompletionError::RequestError(e.into()))?;

        let mut messages = completion_request
            .chat_history
            .into_iter()
            .map(|message| {
                message
                    .try_into()
                    .map_err(|e: MessageError| CompletionError::RequestError(e.into()))
            })
            .collect::<Result<Vec<Message>, _>>()?;

        messages.push(prompt_message);

        let mut request = json!({
            "model": self.model,
            "messages": messages,
            "max_tokens": completion_request.max_tokens.unwrap_or(8192),
            "stream": true,
        });

        if let Some(temperature) = completion_request.temperature {
            merge_inplace(&mut request, json!({ "temperature": temperature }));
        }

        if let Some(ref params) = completion_request.additional_params {
            merge_inplace(&mut request, params.clone())
        }

        let response = self
            .client
            .post("chat/completions")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(CompletionError::ProviderError(response.text().await?));
        }

        // Use our SSE decoder to directly handle Server-Sent Events format
        let sse_stream = sse_from_response(response);

        Ok(Box::pin(stream! {
            let mut sse_stream = Box::pin(sse_stream);

            while let Some(sse_result) = sse_stream.next().await {
                match sse_result {
                    Ok(sse) => {
                        // Parse the SSE data as a StreamingResponse
                        match serde_json::from_str::<StreamingResponse>(&sse.data) {
                            Ok(response) => {
                                if let Some(choice) = response.choices.first() {
                                    if let Some(content) = &choice.delta.content {
                                        yield Ok(streaming::StreamingChoice::Message(content.clone()));
                                    }
                                    if choice.finish_reason.as_deref() == Some("stop") {
                                        break;
                                    }
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
