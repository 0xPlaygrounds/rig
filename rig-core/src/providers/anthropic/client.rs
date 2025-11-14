//! Anthropic client api implementation
use http::HeaderValue;

use super::completion::{ANTHROPIC_VERSION_LATEST, CompletionModel};
use crate::{
    client::{
        self, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder, ProviderClient,
    },
    http_client, models,
};

pub type Client<H = reqwest::Client> = client::Client<AnthropicExt, H>;
pub type ClientBuilder<H = reqwest::Client> = client::ClientBuilder<AnthropicBuilder, String, H>;

// ================================================================
// Main Anthropic Client
// ================================================================
#[derive(Debug, Default, Clone)]
pub struct AnthropicExt;

models! {
    pub enum AnthropicModels {
        /// `claude-opus-4-0` completion model
        Claude4Opus => "claude-opus-4-0",
        /// `claude-sonnet-4-0` completion model
        Claude4Sonnet => "claude-sonnet-4-0",
        /// `claude-3-7-sonnet-latest` completion model
        Claude37Sonnet => "claude-3-7-sonnet-latest",
        /// `claude-3-5-sonnet-latest` completion model
        Claude35Sonnet => "claude-3-5-sonnet-latest",
        /// `claude-3-5-haiku-latest` completion model
        Claude35Haiku => "claude-3-5-haiku-latest",
        /// `claude-3-5-haiku-latest` completion model
        Claude3Opus => "claude-3-opus-latest",
        /// `claude-3-sonnet-20240229` completion model
        Claude3Sonnet => "claude-3-sonnet-20240229",
        /// `claude-3-haiku-20240307` completion model
        Claude3Haiku => "claude-3-haiku-20240307",
    }
}
pub use AnthropicModels::*;

impl Provider for AnthropicExt {
    type Builder = AnthropicBuilder;

    const VERIFY_PATH: &'static str = "/v1/models";

    fn build<H>(_builder: &client::ClientBuilder<Self::Builder, String, H>) -> Self {
        Self
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

impl ProviderBuilder for AnthropicBuilder {
    type Output = AnthropicExt;
    type ApiKey = String;

    const BASE_URL: &'static str = "https://api.anthropic.com";

    fn finish<H>(
        &self,
        mut builder: client::ClientBuilder<Self, String, H>,
    ) -> http_client::Result<client::ClientBuilder<Self, String, H>> {
        let api_key = builder.get_api_key().to_string();

        builder
            .headers_mut()
            .insert("x-api-key", HeaderValue::from_str(&api_key)?);

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

pub struct AnthropicBuilder {
    anthropic_version: String,
    anthropic_betas: Vec<String>,
}

impl Default for AnthropicBuilder {
    fn default() -> Self {
        Self {
            anthropic_version: ANTHROPIC_VERSION_LATEST.into(),
            anthropic_betas: Vec::new(),
        }
    }
}

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
