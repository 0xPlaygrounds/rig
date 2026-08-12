//! MiniMax API clients and Rig integrations.
//!
//! MiniMax exposes both OpenAI-compatible and Anthropic-compatible chat APIs,
//! with distinct global and China entrypoints.
//!
//! # OpenAI-compatible example
//! ```no_run
//! use rig_core::client::CompletionClient;
//! use rig_core::providers::minimax;
//!
//! let client = minimax::Client::new("YOUR_API_KEY").expect("Failed to build client");
//! let model = client.completion_model(minimax::MINIMAX_M2_7);
//! ```
//!
//! # Anthropic-compatible example
//! ```no_run
//! use rig_core::client::CompletionClient;
//! use rig_core::providers::minimax;
//!
//! let client = minimax::AnthropicClient::new("YOUR_API_KEY").expect("Failed to build client");
//! let model = client.completion_model(minimax::MINIMAX_M2);
//! ```

use crate::client::{self, BearerAuth, DebugExt, Provider};
use crate::providers::anthropic::client::{
    AnthropicBuilder as AnthropicCompatBuilder, AnthropicKey, impl_anthropic_compatible_builder,
};
use crate::providers::internal::anthropic_compatible::AnthropicBaseUrl;

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
pub type ClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<MiniMaxBuilder, MiniMaxApiKey, H>;

pub type AnthropicClient<H = reqwest::Client> = client::Client<MiniMaxAnthropicExt, H>;
pub type AnthropicClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<MiniMaxAnthropicBuilder, AnthropicKey, H>;

impl Provider for MiniMaxExt {
    type Builder = MiniMaxBuilder;

    const VERIFY_PATH: &'static str = "/models";
}

impl Provider for MiniMaxAnthropicExt {
    type Builder = MiniMaxAnthropicBuilder;

    const VERIFY_PATH: &'static str = "/v1/models";
}

client::impl_capabilities!(
    MiniMaxExt,
    completion = super::openai::completion::GenericCompletionModel<MiniMaxExt, H>,
);

client::impl_capabilities!(
    MiniMaxAnthropicExt,
    completion = super::anthropic::completion::GenericCompletionModel<MiniMaxAnthropicExt, H>,
);

impl DebugExt for MiniMaxExt {}
impl DebugExt for MiniMaxAnthropicExt {}

impl super::openai::completion::OpenAICompatibleProvider for MiniMaxExt {
    const PROVIDER_NAME: &'static str = "minimax";

    type StreamingUsage = super::openai::Usage;

    type Response = super::openai::CompletionResponse;
}

client::impl_default_provider_builder!(
    MiniMaxBuilder => MiniMaxExt,
    api_key = MiniMaxApiKey,
    base_url = GLOBAL_API_BASE_URL,
);
impl_anthropic_compatible_builder!(
    MiniMaxAnthropicBuilder => MiniMaxAnthropicExt,
    base_url = GLOBAL_ANTHROPIC_API_BASE_URL,
);

impl super::anthropic::completion::AnthropicCompatibleProvider for MiniMaxAnthropicExt {
    const PROVIDER_NAME: &'static str = "minimax";

    fn default_max_tokens(_model: &str) -> Option<u64> {
        Some(4096)
    }
}

client::impl_provider_client!(
    Client,
    input = MiniMaxApiKey,
    api_key_env = "MINIMAX_API_KEY",
    base_url_env = "MINIMAX_API_BASE",
);

client::impl_provider_client!(
    AnthropicClient,
    input = String,
    api_key_env = "MINIMAX_API_KEY",
    base_url =
        ANTHROPIC_BASE_URLS.resolve_from_env("MINIMAX_ANTHROPIC_API_BASE", "MINIMAX_API_BASE")?,
);

const ANTHROPIC_BASE_URLS: AnthropicBaseUrl = AnthropicBaseUrl::new(
    &[
        (GLOBAL_API_BASE_URL, GLOBAL_ANTHROPIC_API_BASE_URL),
        (CHINA_API_BASE_URL, CHINA_ANTHROPIC_API_BASE_URL),
    ],
    &["/v1", "/v1/"],
    "/anthropic",
);

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
}

#[cfg(test)]
mod tests {
    use super::{
        ANTHROPIC_BASE_URLS, CHINA_ANTHROPIC_API_BASE_URL, CHINA_API_BASE_URL,
        GLOBAL_ANTHROPIC_API_BASE_URL, GLOBAL_API_BASE_URL,
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
            ANTHROPIC_BASE_URLS
                .normalize(GLOBAL_API_BASE_URL)
                .as_deref(),
            Some(GLOBAL_ANTHROPIC_API_BASE_URL)
        );
        assert_eq!(
            ANTHROPIC_BASE_URLS.normalize(CHINA_API_BASE_URL).as_deref(),
            Some(CHINA_ANTHROPIC_API_BASE_URL)
        );
        assert_eq!(
            ANTHROPIC_BASE_URLS
                .normalize("https://proxy.example.com/v1")
                .as_deref(),
            Some("https://proxy.example.com/anthropic")
        );
    }

    #[test]
    fn normalize_preserves_existing_anthropic_base() {
        assert_eq!(
            ANTHROPIC_BASE_URLS
                .normalize(CHINA_ANTHROPIC_API_BASE_URL)
                .as_deref(),
            Some(CHINA_ANTHROPIC_API_BASE_URL)
        );
    }

    #[test]
    fn anthropic_primary_override_wins() {
        let override_url = ANTHROPIC_BASE_URLS.resolve(
            Some("https://primary.example.com/anthropic"),
            Some(CHINA_API_BASE_URL),
        );

        assert_eq!(
            override_url.as_deref(),
            Some("https://primary.example.com/anthropic")
        );
    }
}
