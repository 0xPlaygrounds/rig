use crate::completion::{CompletionError, CompletionRequest};
use crate::http_client::HttpClientExt;
use crate::json_utils::{self};
use crate::providers::openai;
use crate::providers::openai::send_compatible_streaming_request;
use crate::providers::xai::completion::{CompletionModel, XAICompletionRequest};
use crate::streaming::StreamingCompletionResponse;
use crate::telemetry::SpanCombinator;
use tracing::{Instrument, info_span};

impl<T> CompletionModel<T>
where
    T: HttpClientExt + Clone + 'static,
{
    pub(crate) async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<openai::StreamingCompletionResponse>, CompletionError>
    {
        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat_streaming",
                gen_ai.operation.name = "chat_streaming",
                gen_ai.provider.name = "xai",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = tracing::field::Empty,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.input.messages = tracing::field::Empty,
                gen_ai.output.messages = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        if self.telemetry_config.include_preamble {
            span.record_preamble(&completion_request.preamble);
        }

        let mut request =
            XAICompletionRequest::try_from((self.model.to_string().as_ref(), completion_request))?;

        let params = json_utils::merge(
            request.additional_params.unwrap_or(serde_json::json!({})),
            serde_json::json!({"stream": true }),
        );

        request.additional_params = Some(params);

        if self.telemetry_config.debug_logging {
            let request_messages_json_str = serde_json::to_string(&request.messages).unwrap();
            tracing::trace!("xAI completion request: {request_messages_json_str}");
        }

        let body = serde_json::to_vec(&request)?;
        let req = self
            .client
            .post("/v1/chat/completions")?
            .header("Content-Type", "application/json")
            .body(body)
            .map_err(|e| CompletionError::HttpError(e.into()))?;

        send_compatible_streaming_request(
            self.client.http_client().clone(),
            req,
            self.telemetry_config.include_message_contents,
        )
        .instrument(span)
        .await
    }
}
