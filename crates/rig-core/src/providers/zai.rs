//! Z.AI API clients and Rig integrations.
//!
//! Z.AI exposes OpenAI-compatible APIs for both its general platform and
//! coding-focused platform, plus an Anthropic-compatible endpoint for tools
//! like Claude Code.
//!
//! # OpenAI-compatible example
//! ```ignore
//! use rig_core::client::CompletionClient;
//! use rig_core::providers::zai;
//!
//! let client = zai::Client::new("YOUR_API_KEY").expect("Failed to build client");
//! let glm_4_6 = client.completion_model(zai::GLM_4_6);
//! ```
//!
//! # Anthropic-compatible example
//! ```ignore
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
    provider = ZAi,
    anthropic_provider = ZAiAnthropic,
    client_input = client::BearerAuth,
    name = "zai",
    api_key_env = "ZAI_API_KEY",
    base_url = GENERAL_API_BASE_URL,
    base_url_env = "ZAI_API_BASE",
    anthropic_provider_name = "z.ai",
    anthropic_base_url = ANTHROPIC_API_BASE_URL,
    anthropic_base_url_env = "ZAI_ANTHROPIC_API_BASE",
);

impl client::HasCompletion for ZAi {
    type Model<H>
        = super::openai::completion::GenericCompletionModel<ZAi, H>
    where
        H: client::ModelTransport;

    fn completion_model<H: client::ModelTransport>(
        client: &Client<H>,
        model: String,
    ) -> Self::Model<H> {
        super::openai::completion::GenericCompletionModel::new(client.clone(), model)
    }
}

impl super::openai::completion::OpenAICompatibleProvider for ZAi {
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
mod tests;
