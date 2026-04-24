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
    type Completion = Capable<super::openai::completion::GenericCompletionModel<ZAiExt, H>>;
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
    type Error = crate::client::ProviderClientError;

    fn from_env() -> Result<Self, Self::Error> {
        let api_key = crate::client::required_env_var("ZAI_API_KEY")?;
        let mut builder = Self::builder().api_key(api_key);

        if let Some(base_url) = crate::client::optional_env_var("ZAI_API_BASE")? {
            builder = builder.base_url(base_url);
        }

        builder.build().map_err(Into::into)
    }

    fn from_val(input: Self::Input) -> Result<Self, Self::Error> {
        Self::new(input).map_err(Into::into)
    }
}

impl ProviderClient for AnthropicClient {
    type Input = String;
    type Error = crate::client::ProviderClientError;

    fn from_env() -> Result<Self, Self::Error> {
        let api_key = crate::client::required_env_var("ZAI_API_KEY")?;
        let mut builder = Self::builder().api_key(api_key);

        if let Some(base_url) = anthropic_base_override("ZAI_ANTHROPIC_API_BASE", "ZAI_API_BASE")? {
            builder = builder.base_url(base_url);
        }

        builder.build().map_err(Into::into)
    }

    fn from_val(input: Self::Input) -> Result<Self, Self::Error> {
        Self::builder().api_key(input).build().map_err(Into::into)
    }
}

fn anthropic_base_override(
    primary_env: &'static str,
    fallback_env: &'static str,
) -> crate::client::ProviderClientResult<Option<String>> {
    let primary = crate::client::optional_env_var(primary_env)?;
    let fallback = crate::client::optional_env_var(fallback_env)?;

    Ok(resolve_anthropic_base_override(
        primary.as_deref(),
        fallback.as_deref(),
    ))
}

fn resolve_anthropic_base_override(
    primary: Option<&str>,
    fallback: Option<&str>,
) -> Option<String> {
    primary
        .map(str::to_owned)
        .or_else(|| fallback.and_then(normalize_anthropic_base_url))
}

fn normalize_anthropic_base_url(base_url: &str) -> Option<String> {
    if base_url.contains("/anthropic") {
        return Some(base_url.to_owned());
    }

    match base_url.trim_end_matches('/') {
        GENERAL_API_BASE_URL | CODING_API_BASE_URL => Some(ANTHROPIC_API_BASE_URL.to_owned()),
        _ => {
            let mut url = url::Url::parse(base_url).ok()?;
            if !matches!(
                url.path(),
                "/api/paas/v4" | "/api/paas/v4/" | "/api/coding/paas/v4" | "/api/coding/paas/v4/"
            ) {
                return None;
            }
            url.set_path("/api/anthropic");
            Some(url.to_string())
        }
    }
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
    use super::{
        ANTHROPIC_API_BASE_URL, CODING_API_BASE_URL, GENERAL_API_BASE_URL,
        normalize_anthropic_base_url, resolve_anthropic_base_override,
    };

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

    #[test]
    fn normalize_openai_style_bases_to_anthropic_base() {
        assert_eq!(
            normalize_anthropic_base_url(GENERAL_API_BASE_URL).as_deref(),
            Some(ANTHROPIC_API_BASE_URL)
        );
        assert_eq!(
            normalize_anthropic_base_url(CODING_API_BASE_URL).as_deref(),
            Some(ANTHROPIC_API_BASE_URL)
        );
        assert_eq!(
            normalize_anthropic_base_url("https://proxy.example.com/api/paas/v4").as_deref(),
            Some("https://proxy.example.com/api/anthropic")
        );
        assert_eq!(
            normalize_anthropic_base_url("https://proxy.example.com/api/coding/paas/v4").as_deref(),
            Some("https://proxy.example.com/api/anthropic")
        );
    }

    #[test]
    fn normalize_preserves_existing_anthropic_base() {
        assert_eq!(
            normalize_anthropic_base_url("https://proxy.example.com/api/anthropic").as_deref(),
            Some("https://proxy.example.com/api/anthropic")
        );
    }

    #[test]
    fn anthropic_primary_override_wins() {
        let override_url = resolve_anthropic_base_override(
            Some("https://primary.example.com/api/anthropic"),
            Some(GENERAL_API_BASE_URL),
        );

        assert_eq!(
            override_url.as_deref(),
            Some("https://primary.example.com/api/anthropic")
        );
    }
}
