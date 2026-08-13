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

use crate::client;
use crate::providers::internal::anthropic_compatible::{
    AnthropicBaseUrl, impl_dual_dialect_provider,
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

impl_dual_dialect_provider!(
    ext = MiniMaxExt,
    builder = MiniMaxBuilder,
    anthropic_ext = MiniMaxAnthropicExt,
    anthropic_builder = MiniMaxAnthropicBuilder,
    client_input = client::BearerAuth,
    api_key_env = "MINIMAX_API_KEY",
    base_url = GLOBAL_API_BASE_URL,
    base_url_env = "MINIMAX_API_BASE",
    anthropic_provider_name = "minimax",
    anthropic_base_url = GLOBAL_ANTHROPIC_API_BASE_URL,
    anthropic_base_url_env = "MINIMAX_ANTHROPIC_API_BASE",
);

client::impl_capabilities!(
    MiniMaxExt,
    completion = super::openai::completion::GenericCompletionModel<MiniMaxExt, H>,
    model_listing = MiniMaxModelLister<H>,
);

crate::providers::internal::model_listing::impl_model_lister!(
    /// [`ModelLister`](crate::client::ModelLister) implementation for
    /// MiniMax's OpenAI-compatible API (`GET /models`). Only this dialect is
    /// listed: MiniMax's Anthropic-compatible `/anthropic/v1/models` returns a
    /// differently shaped payload, so [`MiniMaxAnthropicExt`] keeps no lister.
    MiniMaxModelLister,
    Client<H>,
    crate::providers::internal::model_listing::ListModelEntry,
    "MiniMax",
    "/models"
);

impl super::openai::completion::OpenAICompatibleProvider for MiniMaxExt {
    const PROVIDER_NAME: &'static str = "minimax";

    type StreamingUsage = super::openai::Usage;

    type Response = super::openai::CompletionResponse;
}

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

    /// The lister must hit `<base>/models` and map MiniMax's OpenAI-style
    /// envelope onto `Model`s.
    #[tokio::test]
    async fn list_models_uses_the_models_endpoint() {
        use crate::client::ModelListingClient;
        use crate::test_utils::RecordingHttpClient;

        let http_client = RecordingHttpClient::new(
            r#"{
                "object": "list",
                "data": [
                    {"id": "MiniMax-M2.7", "object": "model", "created": 1780272000, "owned_by": "minimax"},
                    {"id": "MiniMax-M2", "object": "model", "created": 1761782400, "owned_by": "minimax"}
                ]
            }"#,
        );
        let client = super::Client::builder()
            .api_key("test-key")
            .http_client(http_client.clone())
            .build()
            .expect("build client");

        let models = client
            .list_models()
            .await
            .expect("list_models should succeed");

        assert_eq!(models.len(), 2);
        assert_eq!(models.data[0].id, "MiniMax-M2.7");
        assert_eq!(models.data[0].created_at, Some(1780272000));
        assert_eq!(models.data[0].owned_by.as_deref(), Some("minimax"));
        assert_eq!(models.data[1].id, "MiniMax-M2");

        let requests = http_client.requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].uri, "https://api.minimax.io/v1/models");
    }

    /// The listing path is relative to the configured base URL, so selecting
    /// the China host moves the listing with it.
    #[tokio::test]
    async fn list_models_follows_the_configured_base_url() {
        use crate::client::ModelListingClient;
        use crate::test_utils::RecordingHttpClient;

        let http_client = RecordingHttpClient::new(r#"{"object":"list","data":[]}"#);
        let client = super::Client::builder()
            .api_key("test-key")
            .china()
            .http_client(http_client.clone())
            .build()
            .expect("build client");

        let models = client
            .list_models()
            .await
            .expect("list_models should succeed");

        assert!(models.data.is_empty());
        assert_eq!(
            http_client.requests()[0].uri,
            "https://api.minimaxi.com/v1/models"
        );
    }
}
