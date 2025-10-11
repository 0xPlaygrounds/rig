use super::completion::CompletionModel;
use crate::completion::{CompletionError, CompletionRequest};
use crate::json_utils::merge_inplace;
use crate::providers::openai::{StreamingCompletionResponse, send_compatible_streaming_request};
use crate::streaming;
use serde_json::json;
use tracing::{Instrument, info_span};

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

        let body = serde_json::to_vec(&request)?;

        let builder = self
            .client
            .post_reqwest(&path)
            .header("Content-Type", "application/json")
            .body(body);

        let span = if tracing::Span::current().is_disabled() {
            info_span!(
            target: "rig::completions",
            "chat",
            gen_ai.operation.name = "chat",
            gen_ai.provider.name = "huggingface",
            gen_ai.request.model = self.model,
            gen_ai.response.id = tracing::field::Empty,
            gen_ai.response.model = self.model,
            gen_ai.usage.output_tokens = tracing::field::Empty,
            gen_ai.usage.input_tokens = tracing::field::Empty,
            gen_ai.input.messages = serde_json::to_string(&request["messages"]).unwrap(),
            gen_ai.output.messages = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        send_compatible_streaming_request(builder)
            .instrument(span)
            .await
    }
}
