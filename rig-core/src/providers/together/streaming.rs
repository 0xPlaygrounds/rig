use serde_json::json;

use super::completion::CompletionModel;
use crate::providers::openai::send_compatible_streaming_request;
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

        let builder = self.client.post("/v1/chat/completions").json(&request);

        send_compatible_streaming_request(builder).await
    }
}
