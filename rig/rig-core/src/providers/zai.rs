//! Z.AI API clients and Rig integrations.
//!
//! Z.AI exposes OpenAI-compatible APIs for both its general platform and
//! coding-focused platform, plus an Anthropic-compatible endpoint for tools
//! like Claude Code.
//!
//! # OpenAI-compatible example
//! ```no_run
//! use rig::client::CompletionClient;
//! use rig::providers::zai;
//!
//! let client = zai::Client::new("YOUR_API_KEY").expect("Failed to build client");
//! let glm_4_6 = client.completion_model(zai::GLM_4_6);
//! ```
//!
//! # Anthropic-compatible example
//! ```no_run
//! use rig::client::CompletionClient;
//! use rig::providers::zai;
//!
//! let client = zai::AnthropicClient::new("YOUR_API_KEY").expect("Failed to build client");
//! let glm_4_6 = client.completion_model(zai::GLM_4_6);
//! ```

use crate::client::{
    self, BearerAuth, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder,
    ProviderClient,
};
use crate::http_client::{self, HttpClientExt};
use crate::providers::anthropic::client::{
    AnthropicBuilder as AnthropicCompatBuilder, AnthropicKey, finish_anthropic_builder,
};

/// General-purpose OpenAI-compatible base URL.
pub const GENERAL_API_BASE_URL: &str = "https://api.z.ai/api/paas/v4";
/// Coding-focused OpenAI-compatible base URL.
pub const CODING_API_BASE_URL: &str = "https://api.z.ai/api/coding/paas/v4";
/// Anthropic-compatible base URL.
pub const ANTHROPIC_API_BASE_URL: &str = "https://api.z.ai/api/anthropic";

/// `glm-4.6`
pub const GLM_4_6: &str = "glm-4.6";
/// `glm-4.6-air`
pub const GLM_4_6_AIR: &str = "glm-4.6-air";
/// `glm-4.6-x`
pub const GLM_4_6_X: &str = "glm-4.6-x";
/// `glm-4.5`
pub const GLM_4_5: &str = "glm-4.5";
/// `glm-4.5-air`
pub const GLM_4_5_AIR: &str = "glm-4.5-air";
/// `glm-4.5v`
pub const GLM_4_5V: &str = "glm-4.5v";
/// `glm-4.5-airx`
pub const GLM_4_5_AIRX: &str = "glm-4.5-airx";

#[derive(Debug, Default, Clone, Copy)]
pub struct ZAiExt;

#[derive(Debug, Default, Clone, Copy)]
pub struct ZAiBuilder;

#[derive(Debug, Default, Clone)]
pub struct ZAiAnthropicBuilder {
    anthropic: AnthropicCompatBuilder,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ZAiAnthropicExt;

type ZAiApiKey = BearerAuth;

pub type Client<H = reqwest::Client> = client::Client<ZAiExt, H>;
pub type ClientBuilder<H = reqwest::Client> = client::ClientBuilder<ZAiBuilder, ZAiApiKey, H>;

pub type AnthropicClient<H = reqwest::Client> = client::Client<ZAiAnthropicExt, H>;
pub type AnthropicClientBuilder<H = reqwest::Client> =
    client::ClientBuilder<ZAiAnthropicBuilder, AnthropicKey, H>;

impl Provider for ZAiExt {
    type Builder = ZAiBuilder;

    const VERIFY_PATH: &'static str = "/models";
}

impl Provider for ZAiAnthropicExt {
    type Builder = ZAiAnthropicBuilder;

    const VERIFY_PATH: &'static str = "/v1/models";
}

impl<H> Capabilities<H> for ZAiExt {
    type Completion = Capable<super::openai::completion::CompletionModel<ZAiExt, H>>;
    type Embeddings = Nothing;
    type Transcription = Nothing;
    type ModelListing = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;
    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
}

impl<H> Capabilities<H> for ZAiAnthropicExt {
    type Completion =
        Capable<super::anthropic::completion::GenericCompletionModel<ZAiAnthropicExt, H>>;
    type Embeddings = Nothing;
    type Transcription = Nothing;
    type ModelListing = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;
    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
}

impl DebugExt for ZAiExt {}
impl DebugExt for ZAiAnthropicExt {}

impl ProviderBuilder for ZAiBuilder {
    type Extension<H>
        = ZAiExt
    where
        H: HttpClientExt;
    type ApiKey = ZAiApiKey;

    const BASE_URL: &'static str = GENERAL_API_BASE_URL;

    fn build<H>(
        _builder: &client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: HttpClientExt,
    {
        Ok(ZAiExt)
    }
}

impl ProviderBuilder for ZAiAnthropicBuilder {
    type Extension<H>
        = ZAiAnthropicExt
    where
        H: HttpClientExt;
    type ApiKey = AnthropicKey;

    const BASE_URL: &'static str = ANTHROPIC_API_BASE_URL;

    fn build<H>(
        _builder: &client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: HttpClientExt,
    {
        Ok(ZAiAnthropicExt)
    }

    fn finish<H>(
        &self,
        builder: client::ClientBuilder<Self, AnthropicKey, H>,
    ) -> http_client::Result<client::ClientBuilder<Self, AnthropicKey, H>> {
        finish_anthropic_builder(&self.anthropic, builder)
    }
}

impl super::anthropic::completion::AnthropicCompatibleProvider for ZAiAnthropicExt {
    const PROVIDER_NAME: &'static str = "z.ai";

    fn default_max_tokens(_model: &str) -> Option<u64> {
        Some(4096)
    }
}

impl ProviderClient for Client {
    type Input = ZAiApiKey;

    fn from_env() -> Self {
        let api_key = std::env::var("ZAI_API_KEY").expect("ZAI_API_KEY not set");
        let mut builder = Self::builder().api_key(api_key);

        if let Ok(base_url) = std::env::var("ZAI_API_BASE") {
            builder = builder.base_url(base_url);
        }

        builder.build().unwrap()
    }

    fn from_val(input: Self::Input) -> Self {
        Self::new(input).unwrap()
    }
}

impl ProviderClient for AnthropicClient {
    type Input = String;

    fn from_env() -> Self {
        let api_key = std::env::var("ZAI_API_KEY").expect("ZAI_API_KEY not set");
        let mut builder = Self::builder().api_key(api_key);

        if let Some(base_url) = anthropic_base_override("ZAI_ANTHROPIC_API_BASE", "ZAI_API_BASE") {
            builder = builder.base_url(base_url);
        }

        builder.build().unwrap()
    }

    fn from_val(input: Self::Input) -> Self {
        Self::builder().api_key(input).build().unwrap()
    }
}

fn anthropic_base_override(primary_env: &str, fallback_env: &str) -> Option<String> {
    if let Ok(base_url) = std::env::var(primary_env) {
        return Some(base_url);
    }

    std::env::var(fallback_env)
        .ok()
        .filter(|base_url| base_url.contains("/anthropic"))
}

impl<H> ClientBuilder<H> {
    pub fn general(self) -> Self {
        self.base_url(GENERAL_API_BASE_URL)
    }

    pub fn coding(self) -> Self {
        self.base_url(CODING_API_BASE_URL)
    }
}

impl<H> AnthropicClientBuilder<H> {
    pub fn general(self) -> Self {
        self.base_url(ANTHROPIC_API_BASE_URL)
    }

    pub fn anthropic_version(self, anthropic_version: &str) -> Self {
        self.over_ext(|mut ext| {
            ext.anthropic.anthropic_version = anthropic_version.into();
            ext
        })
    }

    pub fn anthropic_betas(self, anthropic_betas: &[&str]) -> Self {
        self.over_ext(|mut ext| {
            ext.anthropic
                .anthropic_betas
                .extend(anthropic_betas.iter().copied().map(String::from));
            ext
        })
    }

    pub fn anthropic_beta(self, anthropic_beta: &str) -> Self {
        self.over_ext(|mut ext| {
            ext.anthropic.anthropic_betas.push(anthropic_beta.into());
            ext
        })
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_client_initialization() {
        let _client = crate::providers::zai::Client::new("dummy-key").expect("Client::new()");
        let _client_from_builder = crate::providers::zai::Client::builder()
            .api_key("dummy-key")
            .build()
            .expect("Client::builder()");
        let _anthropic_client = crate::providers::zai::AnthropicClient::new("dummy-key")
            .expect("AnthropicClient::new()");
        let _anthropic_client_from_builder = crate::providers::zai::AnthropicClient::builder()
            .api_key("dummy-key")
            .build()
            .expect("AnthropicClient::builder()");
    }
}
