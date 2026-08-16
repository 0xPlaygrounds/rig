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
///
/// **Not in Z.AI's catalog.** The 4.6 generation shipped only as `glm-4.6`;
/// the `-air`/`-x`/`-airx` variants belong to 4.5 and 4.7. This identifier
/// appears neither in the `model` enum of Z.AI's chat-completion reference nor
/// in the pricing table, so naming it can only fail.
#[deprecated(
    note = "Z.AI does not serve this model. `glm-4.6-air` is not in Z.AI's documented model list; \
    use `GLM_4_6`, or `GLM_4_5_AIR` for an Air-class model"
)]
pub const GLM_4_6_AIR: &str = "glm-4.6-air";
/// `glm-4.6-x`
///
/// **Not in Z.AI's catalog.** See [`GLM_4_6_AIR`]: the 4.6 generation shipped
/// only as `glm-4.6`.
#[deprecated(
    note = "Z.AI does not serve this model. `glm-4.6-x` is not in Z.AI's documented model list; \
    use `GLM_4_6`, or `GLM_4_5_AIRX` for an X-class model"
)]
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
        GLM_4_5, GLM_4_5_AIR, GLM_4_5_AIRX, GLM_4_5V, GLM_4_6,
    };

    /// Z.AI's documented `model` enum, transcribed from the chat-completion
    /// API reference and cross-checked against the pricing table (both read
    /// 2026-08-16):
    ///
    /// * <https://docs.z.ai/api-reference/llm/chat-completion>
    /// * <https://docs.z.ai/guides/overview/pricing>
    const DOCUMENTED_MODELS: &[&str] = &[
        "glm-5.2",
        "glm-5.1",
        "glm-5-turbo",
        "glm-5",
        "glm-4.7",
        "glm-4.7-flash",
        "glm-4.7-flashx",
        "glm-4.6",
        "glm-4.5",
        "glm-4.5-air",
        "glm-4.5-x",
        "glm-4.5-airx",
        "glm-4.5-flash",
        "glm-4-32b-0414-128k",
        "glm-5v-turbo",
        "glm-4.6v",
        "glm-4.6v-flash",
        "glm-4.6v-flashx",
        "glm-4.5v",
    ];

    /// Every model constant this module exports must either name a model Z.AI
    /// documents or be marked deprecated, and the deprecated ones must stay
    /// absent from the catalog.
    ///
    /// A model handle that the API cannot resolve fails every call that names
    /// it, so a public constant is a promise the provider has to keep; this
    /// pins both halves of that promise against a transcription of Z.AI's own
    /// enum, which is the only place the set is defined.
    ///
    /// **Unit test rather than a cassette because no `ZAI_API_KEY` was
    /// available in the environment where this was written**, so the 400 that
    /// `glm-4.6-air` produces could not be recorded (`tests/README.md` asks a
    /// unit test of provider-facing behavior to say why it is not a cassette
    /// test). The wire half is already written and `#[ignore]`d as
    /// `general/unknown_model_constant_400` and its `_x` sibling in
    /// `tests/providers/zai/cassette/models.rs`; recording those turns this
    /// documentation-based claim into an observed one.
    #[test]
    // The retired constants are the subject of the second assertion.
    #[allow(deprecated)]
    fn model_constants_match_zais_documented_catalog() {
        for model in [GLM_4_6, GLM_4_5, GLM_4_5_AIR, GLM_4_5V, GLM_4_5_AIRX] {
            assert!(
                DOCUMENTED_MODELS.contains(&model),
                "{model} is exported as a usable Z.AI model handle but is not in Z.AI's \
                 documented model enum; correct the constant or deprecate it"
            );
        }

        for model in [super::GLM_4_6_AIR, super::GLM_4_6_X] {
            assert!(
                !DOCUMENTED_MODELS.contains(&model),
                "{model} is back in Z.AI's documented model enum; drop its #[deprecated] \
                 attribute rather than leaving a usable model marked retired"
            );
        }
    }


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
