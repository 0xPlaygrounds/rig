#[cfg(any(feature = "image", feature = "audio"))]
use crate::client::Nothing;
use crate::{
    client::{
        self, BearerAuth, Capabilities, Capable, DebugExt, Provider, ProviderBuilder,
        ProviderClient,
    },
    http_client,
    providers::mistral::MistralModelLister,
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
pub type ClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<MistralBuilder, String, H>;

impl Provider for MistralExt {
    type Builder = MistralBuilder;
    const VERIFY_PATH: &'static str = "/models";
}

impl<H> Capabilities<H> for MistralExt {
    type Completion = Capable<super::CompletionModel<H>>;
    type Embeddings = Capable<super::EmbeddingModel<H>>;

    type Transcription = Capable<super::TranscriptionModel<H>>;
    type ModelListing = Capable<MistralModelLister<H>>;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;

    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
}

impl DebugExt for MistralExt {}

impl ProviderBuilder for MistralBuilder {
    type Extension<H>
        = MistralExt
    where
        H: http_client::HttpClientExt;
    type ApiKey = MistralApiKey;

    const BASE_URL: &'static str = MISTRAL_API_BASE_URL;

    fn build<H>(
        _builder: &client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: http_client::HttpClientExt,
    {
        Ok(MistralExt)
    }
}

impl ProviderClient for Client {
    type Input = String;
    type Error = crate::client::ProviderClientError;

    /// Create a new Mistral client from the `MISTRAL_API_KEY` environment variable.
    fn from_env() -> Result<Self, Self::Error>
    where
        Self: Sized,
    {
        let api_key = crate::client::required_env_var("MISTRAL_API_KEY")?;
        Self::new(&api_key).map_err(Into::into)
    }

    fn from_val(input: Self::Input) -> Result<Self, Self::Error> {
        Self::new(&input).map_err(Into::into)
    }
}

/// In-depth details on prompt tokens.
///
/// Mirrors Mistral's `PromptTokensDetails` schema. The Mistral API also exposes
/// the same shape under the singular field name `prompt_token_details`; the
/// `Usage` field accepts either form via `serde(alias = ...)`.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct PromptTokensDetails {
    /// Number of tokens served from the prompt cache.
    #[serde(default)]
    pub cached_tokens: u64,
}

/// Token usage returned by Mistral's chat completions and embeddings endpoints.
///
/// See <https://docs.mistral.ai/api/> (`UsageInfo` schema). The three counts are
/// always present; the remaining fields are populated by Mistral on a best-effort
/// basis (e.g. cached-token information appears once a prompt is large enough to
/// be cached).
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Usage {
    pub completion_tokens: usize,
    pub prompt_tokens: usize,
    pub total_tokens: usize,
    /// Duration in seconds of audio tokens in the prompt (audio-input models only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_audio_seconds: Option<u64>,
    /// Total cached prompt tokens reported at the top level. Some Mistral
    /// responses populate this in addition to (or instead of)
    /// `prompt_tokens_details.cached_tokens`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_cached_tokens: Option<u64>,
    /// In-depth breakdown of prompt token usage (currently only cached tokens).
    #[serde(
        default,
        alias = "prompt_token_details",
        skip_serializing_if = "Option::is_none"
    )]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
}

impl Usage {
    /// Returns the number of cached prompt tokens, preferring the structured
    /// `prompt_tokens_details.cached_tokens` field and falling back to the
    /// top-level `num_cached_tokens`. Returns 0 when neither is present.
    pub fn cached_tokens(&self) -> u64 {
        self.prompt_tokens_details
            .as_ref()
            .map(|d| d.cached_tokens)
            .or(self.num_cached_tokens)
            .unwrap_or(0)
    }
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

#[cfg(test)]
mod tests {
    #[test]
    fn test_client_initialization() {
        let _client =
            crate::providers::mistral::Client::new("dummy-key").expect("Client::new() failed");
        let _client_from_builder = crate::providers::mistral::Client::builder()
            .api_key("dummy-key")
            .build()
            .expect("Client::builder() failed");
    }
}
