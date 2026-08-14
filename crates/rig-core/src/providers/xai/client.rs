use crate::client::{self, BearerAuth, DebugExt, Provider};
use crate::providers::openai::responses_api::{
    ResponsesProviderExt, ResponsesToolDefinition, SystemInstructionsPlacement,
};

#[derive(Debug, Default, Clone, Copy)]
pub struct XAiExt;
#[derive(Debug, Default, Clone, Copy)]
pub struct XAiExtBuilder;

type XAiApiKey = BearerAuth;

pub type Client<H = reqwest::Client> = client::Client<XAiExt, H>;
pub type ClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<XAiExtBuilder, XAiApiKey, H>;

const XAI_BASE_URL: &str = "https://api.x.ai";

impl Provider for XAiExt {
    type Builder = XAiExtBuilder;

    const VERIFY_PATH: &'static str = "/v1/api-key";
}

impl ResponsesProviderExt for XAiExt {
    const PROVIDER_NAME: &'static str = "xai";
    const RESPONSES_PATH: &'static str = "/v1/responses";
    const EMITS_COMPLETE_TOOL_CALLS_IMMEDIATELY: bool = true;
    const USES_2XX_ERROR_ENVELOPE: bool = true;
    const COMPOSES_NATIVE_OUTPUT_WITH_TOOLS: bool = false;

    fn system_instructions_placement(&self) -> SystemInstructionsPlacement {
        SystemInstructionsPlacement::InputSystemMessages
    }

    fn create_responses_request(
        &self,
        model: String,
        request: crate::completion::CompletionRequest,
        default_tools: &[ResponsesToolDefinition],
        strict_tools: bool,
        _system_instructions_placement: SystemInstructionsPlacement,
        stream: bool,
    ) -> Result<(String, serde_json::Value), crate::completion::CompletionError> {
        super::api::create_completion_request(model, request, default_tools, strict_tools, stream)
    }
}

client::impl_capabilities!(
    XAiExt,
    completion = super::completion::CompletionModel<H>,
    image_generation = super::image_generation::ImageGenerationModel<H>,
    audio_generation = super::audio_generation::AudioGenerationModel<H>,
);

impl DebugExt for XAiExt {}

client::impl_default_provider_builder!(
    XAiExtBuilder => XAiExt,
    api_key = XAiApiKey,
    base_url = XAI_BASE_URL,
);

client::impl_provider_client!(Client, input = String, api_key_env = "XAI_API_KEY");
#[cfg(test)]
mod tests {
    #[test]
    fn test_client_initialization() {
        let _client_from_builder = crate::providers::xai::Client::builder()
            .api_key("dummy-key")
            .build()
            .expect("Client::builder() failed");
    }
}
