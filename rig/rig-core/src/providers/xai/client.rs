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

    fn build<H>(
        _: &client::ClientBuilder<Self::Builder, XAiApiKey, H>,
    ) -> http_client::Result<Self> {
        Ok(Self)
    }
}

impl<H> Capabilities<H> for XAiExt {
    type Completion = Capable<super::completion::CompletionModel<H>>;

    type Embeddings = Nothing;
    type Transcription = Nothing;
    type ModelListing = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;
    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
}

impl DebugExt for XAiExt {}

impl ProviderBuilder for XAiExtBuilder {
    type Output = XAiExt;
    type ApiKey = XAiApiKey;

    const BASE_URL: &'static str = XAI_BASE_URL;
}

impl ProviderClient for Client {
    type Input = String;

    /// Create a new xAI client from the `XAI_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    fn from_env() -> Self {
        let api_key = std::env::var("XAI_API_KEY").expect("XAI_API_KEY not set");
        Self::new(&api_key).unwrap()
    }

    fn from_val(input: Self::Input) -> Self {
        Self::new(&input).unwrap()
    }
}
#[cfg(test)]
mod tests {
    #[test]
    fn test_client_initialization() {
        let _client_from_builder: crate::providers::xai::Client =
            crate::providers::xai::Client::builder()
                .api_key("dummy-key")
                .build()
                .expect("Client::builder() failed");
    }
}
