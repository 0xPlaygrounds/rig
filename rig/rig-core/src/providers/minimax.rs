//! MiniMax API clients and Rig integrations.
//!
//! MiniMax exposes both OpenAI-compatible and Anthropic-compatible chat APIs,
//! with distinct global and China entrypoints.
//!
//! # OpenAI-compatible example
//! ```no_run
//! use rig::client::CompletionClient;
//! use rig::providers::minimax;
//!
//! let client = minimax::Client::new("YOUR_API_KEY").expect("Failed to build client");
//! let model = client.completion_model(minimax::MINIMAX_M2_7);
//! ```
//!
//! # Anthropic-compatible example
//! ```no_run
//! use rig::client::CompletionClient;
//! use rig::providers::minimax;
//!
//! let client = minimax::AnthropicClient::new("YOUR_API_KEY").expect("Failed to build client");
//! let model = client.completion_model(minimax::MINIMAX_M2);
//! ```

use crate::client::{
    self, BearerAuth, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder,
    ProviderClient,
};
use crate::http_client::{self, HttpClientExt};
use crate::providers::anthropic::client::{
    AnthropicBuilder as AnthropicCompatBuilder, AnthropicKey, finish_anthropic_builder,
};

/// Global OpenAI-compatible base URL.
pub const GLOBAL_API_BASE_URL: &str = "https://api.minimax.io/v1";
/// China OpenAI-compatible base URL.
pub const CHINA_API_BASE_URL: &str = "https://api.minimaxi.com/v1";
/// Global Anthropic-compatible base URL.
pub const GLOBAL_ANTHROPIC_API_BASE_URL: &str = "https://api.minimax.io/anthropic";
/// China Anthropic-compatible base URL.
pub const CHINA_ANTHROPIC_API_BASE_URL: &str = "https://api.minimaxi.com/anthropic";

/// `MiniMax-M2.7`
pub const MINIMAX_M2_7: &str = "MiniMax-M2.7";
/// `MiniMax-M2.7-highspeed`
pub const MINIMAX_M2_7_HIGHSPEED: &str = "MiniMax-M2.7-highspeed";
/// `MiniMax-M2.5`
pub const MINIMAX_M2_5: &str = "MiniMax-M2.5";
/// `MiniMax-M2.5-highspeed`
pub const MINIMAX_M2_5_HIGHSPEED: &str = "MiniMax-M2.5-highspeed";
/// `MiniMax-M2.1`
pub const MINIMAX_M2_1: &str = "MiniMax-M2.1";
/// `MiniMax-M2.1-highspeed`
pub const MINIMAX_M2_1_HIGHSPEED: &str = "MiniMax-M2.1-highspeed";
/// `MiniMax-M2`
pub const MINIMAX_M2: &str = "MiniMax-M2";

#[derive(Debug, Default, Clone, Copy)]
pub struct MiniMaxExt;

#[derive(Debug, Default, Clone, Copy)]
pub struct MiniMaxBuilder;

#[derive(Debug, Default, Clone)]
pub struct MiniMaxAnthropicBuilder {
    anthropic: AnthropicCompatBuilder,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct MiniMaxAnthropicExt;

type MiniMaxApiKey = BearerAuth;

pub type Client<H = reqwest::Client> = client::Client<MiniMaxExt, H>;
pub type ClientBuilder<H = reqwest::Client> =
    client::ClientBuilder<MiniMaxBuilder, MiniMaxApiKey, H>;

pub type AnthropicClient<H = reqwest::Client> = client::Client<MiniMaxAnthropicExt, H>;
pub type AnthropicClientBuilder<H = reqwest::Client> =
    client::ClientBuilder<MiniMaxAnthropicBuilder, AnthropicKey, H>;

impl Provider for MiniMaxExt {
    type Builder = MiniMaxBuilder;

    const VERIFY_PATH: &'static str = "/models";
}

impl Provider for MiniMaxAnthropicExt {
    type Builder = MiniMaxAnthropicBuilder;

    const VERIFY_PATH: &'static str = "/v1/models";
}

impl<H> Capabilities<H> for MiniMaxExt {
    type Completion = Capable<super::openai::completion::GenericCompletionModel<MiniMaxExt, H>>;
    type Embeddings = Nothing;
    type Transcription = Nothing;
    type ModelListing = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;
    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
}

impl<H> Capabilities<H> for MiniMaxAnthropicExt {
    type Completion =
        Capable<super::anthropic::completion::GenericCompletionModel<MiniMaxAnthropicExt, H>>;
    type Embeddings = Nothing;
    type Transcription = Nothing;
    type ModelListing = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;
    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
}

impl DebugExt for MiniMaxExt {}
impl DebugExt for MiniMaxAnthropicExt {}

impl ProviderBuilder for MiniMaxBuilder {
    type Extension<H>
        = MiniMaxExt
    where
        H: HttpClientExt;
    type ApiKey = MiniMaxApiKey;

    const BASE_URL: &'static str = GLOBAL_API_BASE_URL;

    fn build<H>(
        _builder: &client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: HttpClientExt,
    {
        Ok(MiniMaxExt)
    }
}

impl ProviderBuilder for MiniMaxAnthropicBuilder {
    type Extension<H>
        = MiniMaxAnthropicExt
    where
        H: HttpClientExt;
    type ApiKey = AnthropicKey;

    const BASE_URL: &'static str = GLOBAL_ANTHROPIC_API_BASE_URL;

    fn build<H>(
        _builder: &client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: HttpClientExt,
    {
        Ok(MiniMaxAnthropicExt)
    }

    fn finish<H>(
        &self,
        builder: client::ClientBuilder<Self, AnthropicKey, H>,
    ) -> http_client::Result<client::ClientBuilder<Self, AnthropicKey, H>> {
        finish_anthropic_builder(&self.anthropic, builder)
    }
}

impl super::anthropic::completion::AnthropicCompatibleProvider for MiniMaxAnthropicExt {
    const PROVIDER_NAME: &'static str = "minimax";

    fn default_max_tokens(_model: &str) -> Option<u64> {
        Some(4096)
    }
}

impl ProviderClient for Client {
    type Input = MiniMaxApiKey;
    type Error = crate::client::ProviderClientError;

    fn from_env() -> Result<Self, Self::Error> {
        let api_key = crate::client::required_env_var("MINIMAX_API_KEY")?;
        let mut builder = Self::builder().api_key(api_key);

        if let Some(base_url) = crate::client::optional_env_var("MINIMAX_API_BASE")? {
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
        let api_key = crate::client::required_env_var("MINIMAX_API_KEY")?;
        let mut builder = Self::builder().api_key(api_key);

        if let Some(base_url) =
            anthropic_base_override("MINIMAX_ANTHROPIC_API_BASE", "MINIMAX_API_BASE")?
        {
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
        GLOBAL_API_BASE_URL => Some(GLOBAL_ANTHROPIC_API_BASE_URL.to_owned()),
        CHINA_API_BASE_URL => Some(CHINA_ANTHROPIC_API_BASE_URL.to_owned()),
        _ => {
            let mut url = url::Url::parse(base_url).ok()?;
            if !matches!(url.path(), "/v1" | "/v1/") {
                return None;
            }
            url.set_path("/anthropic");
            Some(url.to_string())
        }
    }
}

impl<H> ClientBuilder<H> {
    pub fn global(self) -> Self {
        self.base_url(GLOBAL_API_BASE_URL)
    }

    pub fn china(self) -> Self {
        self.base_url(CHINA_API_BASE_URL)
    }
}

impl<H> AnthropicClientBuilder<H> {
    pub fn global(self) -> Self {
        self.base_url(GLOBAL_ANTHROPIC_API_BASE_URL)
    }

    pub fn china(self) -> Self {
        self.base_url(CHINA_ANTHROPIC_API_BASE_URL)
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
        CHINA_ANTHROPIC_API_BASE_URL, CHINA_API_BASE_URL, GLOBAL_ANTHROPIC_API_BASE_URL,
        GLOBAL_API_BASE_URL, normalize_anthropic_base_url, resolve_anthropic_base_override,
    };

    #[test]
    fn test_client_initialization() {
        let _client = crate::providers::minimax::Client::new("dummy-key").expect("Client::new()");
        let _client_from_builder = crate::providers::minimax::Client::builder()
            .api_key("dummy-key")
            .build()
            .expect("Client::builder()");
        let _anthropic_client = crate::providers::minimax::AnthropicClient::new("dummy-key")
            .expect("AnthropicClient::new()");
        let _anthropic_client_from_builder = crate::providers::minimax::AnthropicClient::builder()
            .api_key("dummy-key")
            .build()
            .expect("AnthropicClient::builder()");
    }

    #[test]
    fn normalize_openai_bases_to_anthropic_bases() {
        assert_eq!(
            normalize_anthropic_base_url(GLOBAL_API_BASE_URL).as_deref(),
            Some(GLOBAL_ANTHROPIC_API_BASE_URL)
        );
        assert_eq!(
            normalize_anthropic_base_url(CHINA_API_BASE_URL).as_deref(),
            Some(CHINA_ANTHROPIC_API_BASE_URL)
        );
        assert_eq!(
            normalize_anthropic_base_url("https://proxy.example.com/v1").as_deref(),
            Some("https://proxy.example.com/anthropic")
        );
    }

    #[test]
    fn normalize_preserves_existing_anthropic_base() {
        assert_eq!(
            normalize_anthropic_base_url(CHINA_ANTHROPIC_API_BASE_URL).as_deref(),
            Some(CHINA_ANTHROPIC_API_BASE_URL)
        );
    }

    #[test]
    fn anthropic_primary_override_wins() {
        let override_url = resolve_anthropic_base_override(
            Some("https://primary.example.com/anthropic"),
            Some(CHINA_API_BASE_URL),
        );

        assert_eq!(
            override_url.as_deref(),
            Some("https://primary.example.com/anthropic")
        );
    }
}
