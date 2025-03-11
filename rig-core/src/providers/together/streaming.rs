use serde_json::json;

use super::completion::CompletionModel;
use crate::providers::openai::handle_sse_stream;
use crate::{
    completion::{CompletionError, CompletionRequest},
    json_utils::merge,
    streaming::{StreamingCompletionModel, StreamingResult},
};

impl StreamingCompletionModel for CompletionModel {
    async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<StreamingResult, CompletionError> {
        let mut request = self.create_completion_request(completion_request)?;

        request = merge(request, json!({"stream_tokens": true}));

        let response = self
            .client
            .post("/v1/chat/completions")
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

        handle_sse_stream(response)
    }
}
