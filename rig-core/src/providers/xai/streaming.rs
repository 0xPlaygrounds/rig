use crate::completion::{CompletionError, CompletionRequest};
use crate::json_utils::merge;
use crate::providers::openai;
use crate::providers::openai::send_compatible_streaming_request;
use crate::providers::xai::completion::CompletionModel;
use crate::streaming::StreamingCompletionResponse;
use serde_json::json;
use tracing::{Instrument, info_span};

impl CompletionModel<reqwest::Client> {
    pub(crate) async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<openai::StreamingCompletionResponse>, CompletionError>
    {
        let preamble = completion_request.preamble.clone();
        let mut request = self.create_completion_request(completion_request)?;

        request = merge(request, json!({"stream": true}));

        let builder = self
            .client
            .reqwest_post("/v1/chat/completions")
            .json(&request);

        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat_streaming",
                gen_ai.operation.name = "chat_streaming",
                gen_ai.provider.name = "xai",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = preamble,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.input.messages = serde_json::to_string(request.get("messages").unwrap()).unwrap(),
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
