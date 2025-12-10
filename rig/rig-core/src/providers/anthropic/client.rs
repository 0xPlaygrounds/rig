//! Anthropic client api implementation
use http::{HeaderName, HeaderValue};

use super::completion::{ANTHROPIC_VERSION_LATEST, CompletionModel};
use crate::{
    client::{
        self, ApiKey, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder,
        ProviderClient,
    },
    http_client,
};

// ================================================================
// Main Anthropic Client
// ================================================================
#[derive(Debug, Default, Clone)]
pub struct AnthropicExt;

impl Provider for AnthropicExt {
    type Builder = AnthropicBuilder;

    const VERIFY_PATH: &'static str = "/v1/models";

    fn build<H>(
        _builder: &client::ClientBuilder<Self::Builder, AnthropicKey, H>,
    ) -> http_client::Result<Self> {
        Ok(Self)
    }
}

impl<H> Capabilities<H> for AnthropicExt {
    type Completion = Capable<CompletionModel<H>>;

    type Embeddings = Nothing;
    type Transcription = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;
    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
}

#[derive(Debug, Clone)]
pub struct AnthropicBuilder {
    anthropic_version: String,
    anthropic_betas: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct AnthropicKey(String);

impl<S> From<S> for AnthropicKey
where
    S: Into<String>,
{
    fn from(value: S) -> Self {
        Self(value.into())
    }
}

impl ApiKey for AnthropicKey {
    fn into_header(self) -> Option<http_client::Result<(http::HeaderName, HeaderValue)>> {
        Some(
            HeaderValue::from_str(&self.0)
                .map(|val| (HeaderName::from_static("x-api-key"), val))
                .map_err(Into::into),
        )
    }
}

pub type Client<H = reqwest::Client> = client::Client<AnthropicExt, H>;
pub type ClientBuilder<H = reqwest::Client> =
    client::ClientBuilder<AnthropicBuilder, AnthropicKey, H>;

impl Default for AnthropicBuilder {
    fn default() -> Self {
        Self {
            anthropic_version: ANTHROPIC_VERSION_LATEST.into(),
            anthropic_betas: Vec::new(),
        }
    }
}

impl ProviderBuilder for AnthropicBuilder {
    type Output = AnthropicExt;
    type ApiKey = AnthropicKey;

    const BASE_URL: &'static str = "https://api.anthropic.com";

    fn finish<H>(
        &self,
        mut builder: client::ClientBuilder<Self, AnthropicKey, H>,
    ) -> http_client::Result<client::ClientBuilder<Self, AnthropicKey, H>> {
        builder.headers_mut().insert(
            "anthropic-version",
            HeaderValue::from_str(&self.anthropic_version)?,
        );

        if !self.anthropic_betas.is_empty() {
            builder.headers_mut().insert(
                "anthropic-beta",
                HeaderValue::from_str(&self.anthropic_betas.join(","))?,
            );
        }

        Ok(builder)
    }
}

impl DebugExt for AnthropicExt {}

impl ProviderClient for Client {
    type Input = String;

    fn from_env() -> Self
    where
        Self: Sized,
    {
        let key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY not set");

        Self::builder().api_key(key).build().unwrap()
    }

    fn from_val(input: Self::Input) -> Self
    where
        Self: Sized,
    {
        Self::builder().api_key(input).build().unwrap()
    }
}

/// Create a new anthropic client using the builder
///
/// # Example
/// ```
/// use rig::providers::anthropic::{ClientBuilder, self};
///
/// // Initialize the Anthropic client
/// let anthropic_client = ClientBuilder::new("your-claude-api-key")
///    .anthropic_version(ANTHROPIC_VERSION_LATEST)
///    .anthropic_beta("prompt-caching-2024-07-31")
///    .build()
/// ```
impl<H> ClientBuilder<H> {
    pub fn anthropic_version(self, anthropic_version: &str) -> Self {
        self.over_ext(|ext| AnthropicBuilder {
            anthropic_version: anthropic_version.into(),
            ..ext
        })
    }

    pub fn anthropic_betas(self, anthropic_betas: &[&str]) -> Self {
        self.over_ext(|mut ext| {
            ext.anthropic_betas
                .extend(anthropic_betas.iter().copied().map(String::from));

            ext
        })
    }

    pub fn anthropic_beta(self, anthropic_beta: &str) -> Self {
        self.over_ext(|mut ext| {
            ext.anthropic_betas.push(anthropic_beta.into());

            ext
        })
    }
}
