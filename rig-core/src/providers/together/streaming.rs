use super::completion::CompletionModel;
use crate::completion::{CompletionError, CompletionRequest};
use crate::http_client::HttpClientExt;
use crate::json_utils;
use crate::providers::openai;
use crate::providers::openai::send_compatible_streaming_request;
use crate::providers::together::completion::TogetherAICompletionRequest;
use crate::streaming::StreamingCompletionResponse;

use tracing::{Instrument, info_span};

impl<T> CompletionModel<T>
where
    T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
{
    pub(crate) async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<openai::StreamingCompletionResponse>, CompletionError>
    {
        let preamble = completion_request.preamble.clone();
        let mut request = TogetherAICompletionRequest::try_from((
            self.model.to_string().as_ref(),
            completion_request,
        ))?;

        let params = json_utils::merge(
            request.additional_params.unwrap_or(serde_json::json!({})),
            serde_json::json!({"stream_tokens": true }),
        );

        request.additional_params = Some(params);

        let body = serde_json::to_vec(&request)?;

        let req = self
            .client
            .post("/v1/chat/completions")?
            .body(body)
            .map_err(|x| CompletionError::HttpError(x.into()))?;

        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat_streaming",
                gen_ai.operation.name = "chat_streaming",
                gen_ai.provider.name = "together",
                gen_ai.request.model = self.model.to_string(),
                gen_ai.system_instructions = preamble,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.input.messages = serde_json::to_string(&request.messages)?,
                gen_ai.output.messages = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        send_compatible_streaming_request(self.client.http_client().clone(), req)
            .instrument(span)
            .await
    }
}
