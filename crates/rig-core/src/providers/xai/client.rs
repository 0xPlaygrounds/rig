#[cfg(feature = "audio")]
use crate::client::HasAudioGeneration;
#[cfg(feature = "image")]
use crate::client::HasImageGeneration;
use crate::client::{
    self, BearerAuth, HasCompletion, ModelTransport, Provider, ProviderClientResult,
};
use crate::http_client::{self, HttpClientExt};
use crate::providers::openai::responses_api::{
    ResponsesProviderExt, ResponsesToolDefinition, SystemInstructionsPlacement,
};

#[derive(Debug, Default, Clone, Copy)]
pub struct XAi;
type XAiApiKey = BearerAuth;

pub type Client<H = crate::http_client::BoxedHttpClient> = client::Client<XAi, H>;
pub type ClientBuilder<H = crate::markers::Missing> = client::ClientBuilder<XAi, H>;

const XAI_BASE_URL: &str = "https://api.x.ai";

impl Provider for XAi {
    const NAME: &'static str = "xai";
    const BASE_URL: &'static str = XAI_BASE_URL;
    const VERIFY_PATH: &'static str = "/v1/api-key";
    type ApiKey = XAiApiKey;
    type Config = ();
    type EnvInput = String;

    fn build(_: (), _: &XAiApiKey) -> http_client::Result<Self> {
        Ok(XAi)
    }

    fn from_env<H: HttpClientExt>(http: H) -> ProviderClientResult<Client<H>> {
        Client::from_env_api_key("XAI_API_KEY", None, http)
    }

    fn from_val<H: HttpClientExt>(input: String, http: H) -> ProviderClientResult<Client<H>> {
        Client::new_with(input, http)
    }
}

impl HasCompletion for XAi {
    type Model<H>
        = super::completion::CompletionModel<H>
    where
        H: ModelTransport;

    fn completion_model<H: ModelTransport>(client: &Client<H>, model: String) -> Self::Model<H> {
        super::completion::CompletionModel::new(client.clone(), model)
    }
}

#[cfg(feature = "image")]
impl HasImageGeneration for XAi {
    type Model<H>
        = super::image_generation::ImageGenerationModel<H>
    where
        H: ModelTransport;

    fn image_generation_model<H: ModelTransport>(
        client: &Client<H>,
        model: String,
    ) -> Self::Model<H> {
        super::image_generation::ImageGenerationModel::new(client.clone(), model)
    }
}

#[cfg(feature = "audio")]
impl HasAudioGeneration for XAi {
    type Model<H>
        = super::audio_generation::AudioGenerationModel<H>
    where
        H: ModelTransport;

    fn audio_generation_model<H: ModelTransport>(
        client: &Client<H>,
        model: String,
    ) -> Self::Model<H> {
        super::audio_generation::AudioGenerationModel::new(client.clone(), model)
    }
}

impl ResponsesProviderExt for XAi {
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

#[cfg(test)]
mod tests;
