use crate::{
    client::{
        self, BearerAuth, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder,
        ProviderClient,
    },
    http_client,
};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

const MISTRAL_API_BASE_URL: &str = "https://api.mistral.ai";

#[derive(Debug, Default, Clone, Copy)]
pub struct MistralExt;
#[derive(Debug, Default, Clone, Copy)]
pub struct MistralBuilder;

type MistralApiKey = BearerAuth;

pub type Client<H = reqwest::Client> = client::Client<MistralExt, H>;
pub type ClientBuilder<H = reqwest::Client> = client::ClientBuilder<MistralBuilder, String, H>;

impl Provider for MistralExt {
    type Builder = MistralBuilder;

    const VERIFY_PATH: &'static str = "/models";

    fn build<H>(
        _: &client::ClientBuilder<Self::Builder, MistralApiKey, H>,
    ) -> http_client::Result<Self> {
        Ok(Self)
    }
}

impl<H> Capabilities<H> for MistralExt {
    type Completion = Capable<super::CompletionModel<H>>;
    type Embeddings = Capable<super::EmbeddingModel<H>>;

    type Transcription = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;

    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
}

impl DebugExt for MistralExt {}

impl ProviderBuilder for MistralBuilder {
    type Output = MistralExt;
    type ApiKey = MistralApiKey;

    const BASE_URL: &'static str = MISTRAL_API_BASE_URL;
}

impl ProviderClient for Client {
    type Input = String;

    /// Create a new Mistral client from the `MISTRAL_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    fn from_env() -> Self
    where
        Self: Sized,
    {
        let api_key = std::env::var("MISTRAL_API_KEY").expect("MISTRAL_API_KEY not set");
        Self::new(&api_key).unwrap()
    }

    fn from_val(input: Self::Input) -> Self {
        Self::new(&input).unwrap()
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Usage {
    pub completion_tokens: usize,
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

impl std::fmt::Display for Usage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Prompt tokens: {} Total tokens: {}",
            self.prompt_tokens, self.total_tokens
        )
    }
}

#[derive(Debug, Deserialize)]
pub struct ApiErrorResponse {
    pub(crate) message: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
}
