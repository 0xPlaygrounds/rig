//! Xiaomi MiMo API clients and Rig integrations.
//!
//! Xiaomi exposes both OpenAI-compatible and Anthropic-compatible chat APIs
//! under a single global host.
//!
//! # OpenAI-compatible example
//! ```no_run
//! use rig_core::client::CompletionClient;
//! use rig_core::providers::xiaomimimo;
//!
//! let client = xiaomimimo::Client::new("YOUR_API_KEY").expect("Failed to build client");
//! let model = client.completion_model(xiaomimimo::MIMO_V2_5_PRO);
//! ```
//!
//! # Anthropic-compatible example
//! ```no_run
//! use rig_core::client::CompletionClient;
//! use rig_core::providers::xiaomimimo;
//!
//! let client = xiaomimimo::AnthropicClient::new("YOUR_API_KEY").expect("Failed to build client");
//! let model = client.completion_model(xiaomimimo::MIMO_V2_5_PRO);
//! ```

use crate::client;
use crate::providers::internal::anthropic_compatible::{
    AnthropicBaseUrl, impl_dual_dialect_provider,
};
use crate::providers::internal::model_listing::{ListModelEntry, impl_model_lister};

/// OpenAI-compatible base URL.
pub const API_BASE_URL: &str = "https://api.xiaomimimo.com/v1";
/// Anthropic-compatible base URL.
pub const ANTHROPIC_API_BASE_URL: &str = "https://api.xiaomimimo.com/anthropic/v1";

/// `mimo-v2-flash`
pub const MIMO_V2_FLASH: &str = "mimo-v2-flash";
/// `mimo-v2-omni`
pub const MIMO_V2_OMNI: &str = "mimo-v2-omni";
/// `mimo-v2-pro`
pub const MIMO_V2_PRO: &str = "mimo-v2-pro";
/// `mimo-v2.5`
pub const MIMO_V2_5: &str = "mimo-v2.5";
/// `mimo-v2.5-pro`
pub const MIMO_V2_5_PRO: &str = "mimo-v2.5-pro";

impl_dual_dialect_provider!(
    ext = XiaomiMimoExt,
    builder = XiaomiMimoBuilder,
    anthropic_ext = XiaomiMimoAnthropicExt,
    anthropic_builder = XiaomiMimoAnthropicBuilder,
    client_input = client::BearerAuth,
    api_key_env = "XIAOMI_MIMO_API_KEY",
    base_url = API_BASE_URL,
    base_url_env = "XIAOMI_MIMO_API_BASE",
    anthropic_provider_name = "xiaomimimo",
    anthropic_base_url = ANTHROPIC_API_BASE_URL,
    anthropic_base_url_env = "XIAOMI_MIMO_ANTHROPIC_API_BASE",
);

client::impl_capabilities!(
    XiaomiMimoExt,
    completion = super::openai::completion::GenericCompletionModel<XiaomiMimoExt, H>,
    model_listing = XiaomiMimoModelLister<H>,
);

impl super::openai::completion::OpenAICompatibleProvider for XiaomiMimoExt {
    const PROVIDER_NAME: &'static str = "xiaomimimo";

    type StreamingUsage = super::openai::Usage;

    type Response = super::openai::CompletionResponse;
}

const ANTHROPIC_BASE_URLS: AnthropicBaseUrl = AnthropicBaseUrl::new(
    &[(API_BASE_URL, ANTHROPIC_API_BASE_URL)],
    &["/v1", "/v1/"],
    "/anthropic/v1",
);

impl_model_lister!(
    /// [`ModelLister`](crate::client::ModelLister) implementation for the
    /// Xiaomi MiMo API (`GET /models`).
    XiaomiMimoModelLister,
    Client<H>,
    ListModelEntry,
    "Xiaomi MiMo",
    "/models"
);

#[cfg(test)]
mod tests {
    use super::{ANTHROPIC_API_BASE_URL, ANTHROPIC_BASE_URLS, API_BASE_URL};

    #[test]
    fn test_client_initialization() {
        let _client =
            crate::providers::xiaomimimo::Client::new("dummy-key").expect("Client::new()");
        let _client_from_builder = crate::providers::xiaomimimo::Client::builder()
            .api_key("dummy-key")
            .build()
            .expect("Client::builder()");
        let _anthropic_client = crate::providers::xiaomimimo::AnthropicClient::new("dummy-key")
            .expect("AnthropicClient::new()");
        let _anthropic_client_from_builder =
            crate::providers::xiaomimimo::AnthropicClient::builder()
                .api_key("dummy-key")
                .build()
                .expect("AnthropicClient::builder()");
    }

    #[test]
    fn normalize_openai_bases_to_anthropic_bases() {
        assert_eq!(
            ANTHROPIC_BASE_URLS.normalize(API_BASE_URL).as_deref(),
            Some(ANTHROPIC_API_BASE_URL)
        );
        assert_eq!(
            ANTHROPIC_BASE_URLS
                .normalize("https://proxy.example.com/v1")
                .as_deref(),
            Some("https://proxy.example.com/anthropic/v1")
        );
    }

    #[test]
    fn normalize_preserves_existing_anthropic_base() {
        assert_eq!(
            ANTHROPIC_BASE_URLS
                .normalize(ANTHROPIC_API_BASE_URL)
                .as_deref(),
            Some(ANTHROPIC_API_BASE_URL)
        );
    }

    #[test]
    fn anthropic_primary_override_wins() {
        let override_url = ANTHROPIC_BASE_URLS.resolve(
            Some("https://primary.example.com/anthropic/v1"),
            Some(API_BASE_URL),
        );

        assert_eq!(
            override_url.as_deref(),
            Some("https://primary.example.com/anthropic/v1")
        );
    }
}
