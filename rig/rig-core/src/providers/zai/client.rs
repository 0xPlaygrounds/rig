use crate::{
    client::{
        self, BearerAuth, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder,
        ProviderClient,
    },
    http_client,
};

// ================================================================
// z.ai Client
// ================================================================

#[derive(Debug, Default, Clone, Copy)]
pub struct Zai;
#[derive(Debug, Default, Clone, Copy)]
pub struct ZaiBuilder;

type ZaiApiKey = BearerAuth;

pub type Client<H = reqwest::Client> = client::Client<Zai, H>;
pub type ClientBuilder<H = reqwest::Client> = client::ClientBuilder<ZaiBuilder, ZaiApiKey, H>;

const ZAI_BASE_URL: &str = "https://api.z.ai/api/paas/v4";

impl Provider for Zai {
    type Builder = ZaiBuilder;

    const VERIFY_PATH: &'static str = "/v1/api-key";

    fn build<H>(
        _: &client::ClientBuilder<Self::Builder, ZaiApiKey, H>,
    ) -> http_client::Result<Self> {
        Ok(Self)
    }
}

impl<H> Capabilities<H> for Zai {
    type Completion = Capable<super::completion::CompletionModel<H>>;

    type Embeddings = Nothing;
    type Transcription = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;
    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
}

impl DebugExt for Zai {}

impl ProviderBuilder for ZaiBuilder {
    type Output = Zai;
    type ApiKey = ZaiApiKey;

    const BASE_URL: &'static str = ZAI_BASE_URL;
}

impl ProviderClient for Client {
    type Input = String;

    /// Create a new z.ai client from the `ZAI_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    fn from_env() -> Self {
        let api_key = std::env::var("ZAI_API_KEY").expect("ZAI_API_KEY not set");
        Self::new(&api_key).unwrap()
    }

    fn from_val(input: Self::Input) -> Self {
        Self::new(&input).unwrap()
    }
}

pub mod zai_api_types {
    use serde::Deserialize;

    impl ApiErrorResponse {
        pub fn message(&self) -> String {
            format!("Code `{}`: {}", self.code, self.error)
        }
    }

    #[derive(Debug, Deserialize)]
    pub struct ApiErrorResponse {
        pub error: String,
        pub code: String,
    }

    #[derive(Debug, Deserialize)]
    #[serde(untagged)]
    pub enum ApiResponse<T> {
        Ok(T),
        Error(ApiErrorResponse),
    }
}
