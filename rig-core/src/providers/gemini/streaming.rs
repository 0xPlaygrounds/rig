use async_stream::stream;
use futures::StreamExt;
use serde::Deserialize;

use super::completion::{CompletionModel, create_request_body, gemini_api_types::ContentCandidate};
use crate::{
    completion::{CompletionError, CompletionRequest},
    streaming::{self},
};

#[derive(Debug, Deserialize, Default, Clone)]
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

#[derive(Clone, Debug)]
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

        let response = self
            .client
            .post_sse(&format!(
                "/v1beta/models/{}:streamGenerateContent",
                self.model
            ))
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(CompletionError::ProviderError(format!(
                "{}: {}",
                response.status(),
                response.text().await?
            )));
        }

        let stream = Box::pin(stream! {
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
                    let Some(line) = line.strip_prefix("data: ") else { continue; };

                    let Ok(data) = serde_json::from_str::<StreamGenerateContentResponse>(line) else {
                        continue;
                    };

                    let choice = data.candidates.first().expect("Should have at least one choice");

                    match choice.content.parts.first() {
                        super::completion::gemini_api_types::Part::Text(text)
                            => yield Ok(streaming::RawStreamingChoice::Message(text)),
                        super::completion::gemini_api_types::Part::FunctionCall(function_call)
                            => yield Ok(streaming::RawStreamingChoice::ToolCall {
                                    name: function_call.name,
                                    id: "".to_string(),
                                    arguments: function_call.args,
                                    call_id: None
                                }),
                        _ => panic!("Unsupported response type with streaming.")
                    };

                    if choice.finish_reason.is_some() {
                        yield Ok(streaming::RawStreamingChoice::FinalResponse(StreamingCompletionResponse {
                            usage_metadata: PartialUsage {
                                total_token_count: data.usage_metadata.unwrap().total_token_count,
                            }
                        }))
                    }
                }
            }
        });

        Ok(streaming::StreamingCompletionResponse::stream(stream))
    }
}
