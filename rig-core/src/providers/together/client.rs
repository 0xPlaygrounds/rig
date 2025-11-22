use crate::{
    client::{
        self, BearerAuth, Capabilities, Capable, Nothing, Provider, ProviderBuilder, ProviderClient,
    },
    http_client,
};

// ================================================================
// Together AI Client
// ================================================================
const TOGETHER_AI_BASE_URL: &str = "https://api.together.xyz";

#[derive(Debug, Default, Clone, Copy)]
pub struct TogetherExt;
#[derive(Debug, Default, Clone, Copy)]
pub struct TogetherExtBuilder;

type TogetherApiKey = BearerAuth;

pub type Client<H = reqwest::Client> = client::Client<TogetherExt, H>;
pub type ClientBuilder<H = reqwest::Client> =
    client::ClientBuilder<TogetherExtBuilder, TogetherApiKey, H>;

impl Provider for TogetherExt {
    type Builder = TogetherExtBuilder;

    const VERIFY_PATH: &'static str = "/models";

    fn build<H>(
        _: &client::ClientBuilder<Self::Builder, TogetherApiKey, H>,
    ) -> http_client::Result<Self> {
        Ok(Self)
    }
}

impl<H> Capabilities<H> for TogetherExt {
    type Completion = Capable<super::CompletionModel<H>>;
    type Embeddings = Capable<super::EmbeddingModel<H>>;

    type Transcription = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;
    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
}

impl ProviderBuilder for TogetherExtBuilder {
    type Output = TogetherExt;
    type ApiKey = TogetherApiKey;

    const BASE_URL: &'static str = TOGETHER_AI_BASE_URL;
}

impl ProviderClient for Client {
    type Input = String;

    /// Create a new Together AI client from the `TOGETHER_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    fn from_env() -> Self {
        let api_key = std::env::var("TOGETHER_API_KEY").expect("TOGETHER_API_KEY not set");
        Self::new(&api_key).unwrap()
    }

    fn from_val(input: Self::Input) -> Self {
        Self::new(&input).unwrap()
    }
}

pub mod together_ai_api_types {
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
