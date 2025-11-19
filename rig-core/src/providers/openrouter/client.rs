use crate::{
    client::{
        self, BearerAuth, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder,
        ProviderClient,
    },
    completion::GetTokenUsage,
    http_client,
};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

// ================================================================
// Main openrouter Client
// ================================================================
const OPENROUTER_API_BASE_URL: &str = "https://openrouter.ai/api/v1";

#[derive(Debug, Default, Clone, Copy)]
pub struct OpenRouterExt;
#[derive(Debug, Default, Clone, Copy)]
pub struct OpenRouterExtBuilder;

type OpenRouterApiKey = BearerAuth;

pub type Client<H = reqwest::Client> = client::Client<OpenRouterExt, H>;
pub type ClientBuilder<H = reqwest::Client> =
    client::ClientBuilder<OpenRouterExtBuilder, OpenRouterApiKey, H>;

impl Provider for OpenRouterExt {
    type Builder = OpenRouterExtBuilder;

    const VERIFY_PATH: &'static str = "/key";

    fn build<H>(
        _: &crate::client::ClientBuilder<Self::Builder, OpenRouterApiKey, H>,
    ) -> http_client::Result<Self> {
        Ok(Self)
    }
}

impl<H> Capabilities<H> for OpenRouterExt {
    type Completion = Capable<super::CompletionModel<H>>;
    type Embeddings = Nothing;
    type Transcription = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;

    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
}

impl DebugExt for OpenRouterExt {}

impl ProviderBuilder for OpenRouterExtBuilder {
    type Output = OpenRouterExt;
    type ApiKey = OpenRouterApiKey;

    const BASE_URL: &'static str = OPENROUTER_API_BASE_URL;
}

impl ProviderClient for Client {
    type Input = OpenRouterApiKey;

    /// Create a new openrouter client from the `OPENROUTER_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    fn from_env() -> Self {
        let api_key = std::env::var("OPENROUTER_API_KEY").expect("OPENROUTER_API_KEY not set");

        Self::new(&api_key).unwrap()
    }

    fn from_val(input: Self::Input) -> Self {
        Self::new(input).unwrap()
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct ApiErrorResponse {
    pub message: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
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

impl GetTokenUsage for Usage {
    fn token_usage(&self) -> Option<crate::completion::Usage> {
        let mut usage = crate::completion::Usage::new();

        usage.input_tokens = self.prompt_tokens as u64;
        usage.output_tokens = self.completion_tokens as u64;
        usage.total_tokens = self.total_tokens as u64;

        Some(usage)
    }
}
