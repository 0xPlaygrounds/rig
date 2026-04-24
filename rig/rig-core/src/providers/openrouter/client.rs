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
}

impl<H> Capabilities<H> for OpenRouterExt {
    type Completion = Capable<super::CompletionModel<H>>;
    type Embeddings = Capable<super::EmbeddingModel<H>>;
    type Transcription = Nothing;
    type ModelListing = Capable<super::OpenRouterModelLister<H>>;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;

    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
}

impl DebugExt for OpenRouterExt {}

impl ProviderBuilder for OpenRouterExtBuilder {
    type Extension<H>
        = OpenRouterExt
    where
        H: http_client::HttpClientExt;
    type ApiKey = OpenRouterApiKey;

    const BASE_URL: &'static str = OPENROUTER_API_BASE_URL;

    fn build<H>(
        _builder: &crate::client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: http_client::HttpClientExt,
    {
        Ok(OpenRouterExt)
    }
}

impl ProviderClient for Client {
    type Input = OpenRouterApiKey;
    type Error = crate::client::ProviderClientError;

    /// Create a new openrouter client from the `OPENROUTER_API_KEY` environment variable.
    fn from_env() -> Result<Self, Self::Error> {
        let api_key = crate::client::required_env_var("OPENROUTER_API_KEY")?;

        Self::new(&api_key).map_err(Into::into)
    }

    fn from_val(input: Self::Input) -> Result<Self, Self::Error> {
        Self::new(input).map_err(Into::into)
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
    #[serde(default)]
    pub completion_tokens: usize,
    pub total_tokens: usize,
    #[serde(default)]
    pub cost: f64,
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
        Some(crate::providers::internal::completion_usage(
            self.prompt_tokens as u64,
            self.completion_tokens as u64,
            self.total_tokens as u64,
            0,
        ))
    }
}
#[cfg(test)]
mod tests {
    #[test]
    fn test_client_initialization() {
        let _client =
            crate::providers::openrouter::Client::new("dummy-key").expect("Client::new() failed");
        let _client_from_builder = crate::providers::openrouter::Client::builder()
            .api_key("dummy-key")
            .build()
            .expect("Client::builder() failed");
    }
}
