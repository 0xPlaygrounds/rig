use crate::client::{self, BearerAuth, DebugExt, Provider};

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
