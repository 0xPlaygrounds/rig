use async_stream::stream;
use futures::StreamExt;
use serde::Deserialize;

use crate::{
    completion::{CompletionError, CompletionRequest},
    streaming::{self, StreamingCompletionModel, StreamingResult},
};

use super::completion::{create_request_body, gemini_api_types::ContentCandidate, CompletionModel};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StreamGenerateContentResponse {
    /// Candidate responses from the model.
    pub candidates: Vec<ContentCandidate>,
    pub model_version: Option<String>,
}

impl StreamingCompletionModel for CompletionModel {
    async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<StreamingResult, CompletionError> {
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

        Ok(Box::pin(stream! {
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
                            => yield Ok(streaming::StreamingChoice::Message(text)),
                        super::completion::gemini_api_types::Part::FunctionCall(function_call)
                            => yield Ok(streaming::StreamingChoice::ToolCall(function_call.name, "".to_string(), function_call.args)),
                        _ => panic!("Unsupported response type with streaming.")
                    };
                }
            }
        }))
    }
}
