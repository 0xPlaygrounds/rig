use async_stream::stream;
use futures::StreamExt;
use reqwest_eventsource::{Event, RequestBuilderExt};
use serde::{Deserialize, Serialize};

use super::completion::{
    CompletionModel, create_request_body,
    gemini_api_types::{ContentCandidate, PartKind},
};
use crate::{
    completion::{CompletionError, CompletionRequest},
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
                                tracing::warn!(?error, "Failed to parse SSE message, data: {}", message.data);
                                continue;
                            }
                        };

                        // Process the response data
                        if let Some(choice) = data.candidates.first() {
                            match choice.content.parts.first() {
                                super::completion::gemini_api_types::Part {
                                    part: PartKind::Text(text),
                                    thought: Some(true),
                                    ..
                                } => {
                                    yield Ok(streaming::RawStreamingChoice::Reasoning { reasoning: text.clone(), id: None });
                                },
                                super::completion::gemini_api_types::Part {
                                    part: PartKind::Text(text),
                                    thought,
                                    ..
                                } => {
                                    if thought != Some(true) {
                                        yield Ok(streaming::RawStreamingChoice::Message(text.clone()));
                                    }
                                },
                                super::completion::gemini_api_types::Part {
                                    part: PartKind::FunctionCall(function_call),
                                    ..
                                } => {
                                    yield Ok(streaming::RawStreamingChoice::ToolCall {
                                        name: function_call.name.clone(),
                                        id: function_call.name.clone(),
                                        arguments: function_call.args.clone(),
                                        call_id: None
                                    });
                                },
                                part => {
                                    tracing::warn!(?part, "Unsupported response type with streaming");
                                    continue;
                                }
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
                                // Close the event source after final response
                                event_source.close();
                                break;
                            }
                        }
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
