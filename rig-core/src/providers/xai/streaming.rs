use crate::completion::{CompletionError, CompletionRequest};
use crate::json_utils::merge;
use crate::providers::openai::send_compatible_streaming_request;
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

        let builder = self.client.post("/v1/chat/completions").json(&request);

        send_compatible_streaming_request(builder).await
    }
}
