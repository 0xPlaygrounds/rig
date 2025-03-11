use crate::completion::{CompletionError, CompletionRequest};
use crate::json_utils::merge;
use crate::providers::openai::handle_sse_stream;
use crate::providers::xai::completion::CompletionModel;
use crate::streaming::{StreamingCompletionModel, StreamingResult};
use serde_json::json;

impl StreamingCompletionModel for CompletionModel {
    async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<StreamingResult, CompletionError> {
        let mut request = self.create_completion_request(completion_request)?;

        request = merge(request, json!({"stream": true}));

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
