use super::completion::CompletionModel;
use crate::completion::{CompletionError, CompletionRequest};
use crate::json_utils::merge_inplace;
use crate::providers::openai::{StreamingCompletionResponse, send_compatible_streaming_request};
use crate::streaming;
use serde_json::json;

impl CompletionModel<reqwest::Client> {
    pub(crate) async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<streaming::StreamingCompletionResponse<StreamingCompletionResponse>, CompletionError>
    {
        let mut request = self.create_request_body(&completion_request)?;

        // Enable streaming
        merge_inplace(
            &mut request,
            json!({"stream": true, "stream_options": {"include_usage": true}}),
        );

        if let Some(ref params) = completion_request.additional_params {
            merge_inplace(&mut request, params.clone());
        }

        // HF Inference API uses the model in the path even though its specified in the request body
        let path = self.client.sub_provider.completion_endpoint(&self.model);

        let request = serde_json::to_vec(&request)?;

        let builder = self.client.post_reqwest(&path).body(request);

        send_compatible_streaming_request(builder).await
    }
}
