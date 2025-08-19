use async_stream::stream;
use futures::StreamExt;
use reqwest_eventsource::{Event, RequestBuilderExt};
use serde::{Deserialize, Serialize};

use super::completion::{
    CompletionModel, create_request_body,
    gemini_api_types::{ContentCandidate, Part, PartKind},
};
use crate::{
    completion::{CompletionError, CompletionRequest, GetTokenUsage},
    streaming::{self},
};

#[derive(Debug, Deserialize, Serialize, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub struct PartialUsage {
    pub total_token_count: i32,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StreamGenerateContentResponse {
    /// Candidate responses from the model.
    pub candidates: Vec<ContentCandidate>,
    pub model_version: Option<String>,
    pub usage_metadata: Option<PartialUsage>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamingCompletionResponse {
    pub usage_metadata: PartialUsage,
}

impl GetTokenUsage for StreamingCompletionResponse {
    fn token_usage(&self) -> Option<crate::completion::Usage> {
        let mut usage = crate::completion::Usage::new();
        usage.total_tokens = self.usage_metadata.total_token_count as u64;
        Some(usage)
    }
}

impl CompletionModel {
    pub(crate) async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<streaming::StreamingCompletionResponse<StreamingCompletionResponse>, CompletionError>
    {
        let request = create_request_body(completion_request)?;

        tracing::debug!(
            "Sending completion request to Gemini API {}",
            serde_json::to_string_pretty(&request)?
        );

        // Build the request with proper headers for SSE
        let mut event_source = self
            .client
            .post_sse(&format!(
                "/v1beta/models/{}:streamGenerateContent",
                self.model
            ))
            .json(&request)
            .eventsource()
            .expect("Cloning request must always succeed");

        let stream = Box::pin(stream! {
            while let Some(event_result) = event_source.next().await {
                match event_result {
                    Ok(Event::Open) => {
                        tracing::trace!("SSE connection opened");
                        continue;
                    }
                    Ok(Event::Message(message)) => {
                        // Skip heartbeat messages or empty data
                        if message.data.trim().is_empty() {
                            continue;
                        }

                        let data = match serde_json::from_str::<StreamGenerateContentResponse>(&message.data) {
                            Ok(d) => d,
                            Err(error) => {
                                tracing::error!(?error, message = message.data, "Failed to parse SSE message");
                                continue;
                            }
                        };

                        // Process the response data
                        let Some(choice) = data.candidates.first() else {
                            tracing::debug!("There is no content candidate");
                            continue;
                        };

                        match choice.content.parts.first() {
                            Some(Part {
                                part: PartKind::Text(text),
                                thought: Some(true),
                                ..
                            }) => {
                                yield Ok(streaming::RawStreamingChoice::Reasoning { reasoning: text.clone(), id: None });
                            },
                            Some(Part {
                                part: PartKind::Text(text),
                                ..
                            }) => {
                                yield Ok(streaming::RawStreamingChoice::Message(text.clone()));
                            },
                            Some(Part {
                                part: PartKind::FunctionCall(function_call),
                                ..
                            }) => {
                                yield Ok(streaming::RawStreamingChoice::ToolCall {
                                    name: function_call.name.clone(),
                                    id: function_call.name.clone(),
                                    arguments: function_call.args.clone(),
                                    call_id: None
                                });
                            },
                            Some(part) => {
                                tracing::warn!(?part, "Unsupported response type with streaming");
                            }
                            None => tracing::trace!(reason = ?choice.finish_reason, "There is no part in the streaming content"),
                        }

                        // Check if this is the final response
                        if choice.finish_reason.is_some() {
                            let usage = data.usage_metadata
                                            .map(|u| u.total_token_count)
                                            .unwrap_or(0);

                            yield Ok(streaming::RawStreamingChoice::FinalResponse(StreamingCompletionResponse {
                                usage_metadata: PartialUsage {
                                    total_token_count: usage,
                                }
                            }));
                            break;
                        }
                    }
                    Err(reqwest_eventsource::Error::StreamEnded) => {
                        break;
                    }
                    Err(error) => {
                        tracing::error!(?error, "SSE error");
                        yield Err(CompletionError::ResponseError(error.to_string()));
                        break;
                    }
                }
            }

            // Ensure event source is closed when stream ends
            event_source.close();
        });

        Ok(streaming::StreamingCompletionResponse::stream(stream))
    }
}
