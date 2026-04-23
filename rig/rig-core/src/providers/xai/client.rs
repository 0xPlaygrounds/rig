use crate::{
    client::{
        self, BearerAuth, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder,
        ProviderClient,
    },
    http_client,
};

#[derive(Debug, Default, Clone, Copy)]
pub struct XAiExt;
#[derive(Debug, Default, Clone, Copy)]
pub struct XAiExtBuilder;

type XAiApiKey = BearerAuth;

pub type Client<H = reqwest::Client> = client::Client<XAiExt, H>;
pub type ClientBuilder<H = reqwest::Client> = client::ClientBuilder<XAiExtBuilder, XAiApiKey, H>;

const XAI_BASE_URL: &str = "https://api.x.ai";

impl Provider for XAiExt {
    type Builder = XAiExtBuilder;

    const VERIFY_PATH: &'static str = "/v1/api-key";
}

impl<H> Capabilities<H> for XAiExt {
    type Completion = Capable<super::completion::CompletionModel<H>>;

    type Embeddings = Nothing;
    type Transcription = Nothing;
    type ModelListing = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Capable<super::image_generation::ImageGenerationModel<H>>;
    #[cfg(feature = "audio")]
    type AudioGeneration = Capable<super::audio_generation::AudioGenerationModel<H>>;
}

impl DebugExt for XAiExt {}

impl ProviderBuilder for XAiExtBuilder {
    type Extension<H>
        = XAiExt
    where
        H: http_client::HttpClientExt;
    type ApiKey = XAiApiKey;

    const BASE_URL: &'static str = XAI_BASE_URL;

    fn build<H>(
        _builder: &client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: http_client::HttpClientExt,
    {
        Ok(XAiExt)
    }
}

impl ProviderClient for Client {
    type Input = String;
    type Error = crate::client::ProviderClientError;

    /// Create a new xAI client from the `XAI_API_KEY` environment variable.
    fn from_env() -> Result<Self, Self::Error> {
        let api_key = crate::client::required_env_var("XAI_API_KEY")?;
        Self::new(&api_key).map_err(Into::into)
    }

    fn from_val(input: Self::Input) -> Result<Self, Self::Error> {
        Self::new(&input).map_err(Into::into)
    }
}
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
