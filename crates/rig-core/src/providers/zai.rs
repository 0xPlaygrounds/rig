//! Z.AI API clients and Rig integrations.
//!
//! Z.AI exposes OpenAI-compatible APIs for both its general platform and
//! coding-focused platform, plus an Anthropic-compatible endpoint for tools
//! like Claude Code.
//!
//! # OpenAI-compatible example
//! ```no_run
//! use rig_core::client::CompletionClient;
//! use rig_core::providers::zai;
//!
//! let client = zai::Client::new("YOUR_API_KEY").expect("Failed to build client");
//! let glm_4_6 = client.completion_model(zai::GLM_4_6);
//! ```
//!
//! # Anthropic-compatible example
//! ```no_run
//! use rig_core::client::CompletionClient;
//! use rig_core::providers::zai;
//!
//! let client = zai::AnthropicClient::new("YOUR_API_KEY").expect("Failed to build client");
//! let glm_4_6 = client.completion_model(zai::GLM_4_6);
//! ```

use crate::client;
use crate::providers::internal::anthropic_compatible::{
    AnthropicBaseUrl, impl_dual_dialect_provider,
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

impl_dual_dialect_provider!(
    ext = ZAiExt,
    builder = ZAiBuilder,
    anthropic_ext = ZAiAnthropicExt,
    anthropic_builder = ZAiAnthropicBuilder,
    client_input = client::BearerAuth,
    api_key_env = "ZAI_API_KEY",
    base_url = GENERAL_API_BASE_URL,
    base_url_env = "ZAI_API_BASE",
    anthropic_provider_name = "z.ai",
    anthropic_base_url = ANTHROPIC_API_BASE_URL,
    anthropic_base_url_env = "ZAI_ANTHROPIC_API_BASE",
);

client::impl_capabilities!(
    ZAiExt,
    completion = super::openai::completion::GenericCompletionModel<ZAiExt, H>,
);

impl super::openai::completion::OpenAICompatibleProvider for ZAiExt {
    const PROVIDER_NAME: &'static str = "zai";

    type StreamingUsage = super::openai::Usage;

    type Response = super::openai::CompletionResponse;
}

const ANTHROPIC_BASE_URLS: AnthropicBaseUrl = AnthropicBaseUrl::new(
    &[
        (GENERAL_API_BASE_URL, ANTHROPIC_API_BASE_URL),
        (CODING_API_BASE_URL, ANTHROPIC_API_BASE_URL),
    ],
    &[
        "/api/paas/v4",
        "/api/paas/v4/",
        "/api/coding/paas/v4",
        "/api/coding/paas/v4/",
    ],
    "/api/anthropic",
);

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
}

#[cfg(test)]
mod tests {
    use super::{
        ANTHROPIC_API_BASE_URL, ANTHROPIC_BASE_URLS, CODING_API_BASE_URL, GENERAL_API_BASE_URL,
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
            ANTHROPIC_BASE_URLS
                .normalize(GENERAL_API_BASE_URL)
                .as_deref(),
            Some(ANTHROPIC_API_BASE_URL)
        );
        assert_eq!(
            ANTHROPIC_BASE_URLS
                .normalize(CODING_API_BASE_URL)
                .as_deref(),
            Some(ANTHROPIC_API_BASE_URL)
        );
        assert_eq!(
            ANTHROPIC_BASE_URLS
                .normalize("https://proxy.example.com/api/paas/v4")
                .as_deref(),
            Some("https://proxy.example.com/api/anthropic")
        );
        assert_eq!(
            ANTHROPIC_BASE_URLS
                .normalize("https://proxy.example.com/api/coding/paas/v4")
                .as_deref(),
            Some("https://proxy.example.com/api/anthropic")
        );
    }

    #[test]
    fn normalize_preserves_existing_anthropic_base() {
        assert_eq!(
            ANTHROPIC_BASE_URLS
                .normalize("https://proxy.example.com/api/anthropic")
                .as_deref(),
            Some("https://proxy.example.com/api/anthropic")
        );
    }

    #[test]
    fn anthropic_primary_override_wins() {
        let override_url = ANTHROPIC_BASE_URLS.resolve(
            Some("https://primary.example.com/api/anthropic"),
            Some(GENERAL_API_BASE_URL),
        );

        assert_eq!(
            override_url.as_deref(),
            Some("https://primary.example.com/api/anthropic")
        );
    }
}
