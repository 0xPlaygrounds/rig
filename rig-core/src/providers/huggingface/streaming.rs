use super::completion::CompletionModel;
use crate::completion::{CompletionError, CompletionRequest};
use crate::http_client::HttpClientExt;
use crate::json_utils::{self};
use crate::providers::huggingface::completion::HuggingfaceCompletionRequest;
use crate::providers::openai::{StreamingCompletionResponse, send_compatible_streaming_request};
use crate::streaming;
use tracing::{Instrument, info_span};

impl<T> CompletionModel<T>
where
    T: HttpClientExt + Clone + 'static,
{
    pub(crate) async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<streaming::StreamingCompletionResponse<StreamingCompletionResponse>, CompletionError>
    {
        let model = self.client.subprovider().model_identifier(&self.model);
        let mut request =
            HuggingfaceCompletionRequest::try_from((model.as_ref(), completion_request))?;

        let params = json_utils::merge(
            request.additional_params.unwrap_or(serde_json::json!({})),
            serde_json::json!({"stream": true, "stream_options": {"include_usage": true }}),
        );

        request.additional_params = Some(params);

        if tracing::enabled!(tracing::Level::TRACE) {
            tracing::trace!(
                target: "rig::streaming",
                "Huggingface streaming completion request: {}",
                serde_json::to_string_pretty(&request)?
            );
        }

        // HF Inference API uses the model in the path even though its specified in the request body
        let path = self.client.subprovider().completion_endpoint(&self.model);

        let body = serde_json::to_vec(&request)?;

        let req = self
            .client
            .post(&path)?
            .header("Content-Type", "application/json")
            .body(body)
            .map_err(|e| CompletionError::HttpError(e.into()))?;

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
            )
        } else {
            tracing::Span::current()
        };

        send_compatible_streaming_request(self.client.clone(), req)
            .instrument(span)
            .await
    }
}
