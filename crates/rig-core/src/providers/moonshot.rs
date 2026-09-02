//! Moonshot AI (Kimi) API client and Rig integration
//!
//! # Example
//! ```ignore
//! use rig_core::providers::moonshot;
//! use rig_core::client::CompletionClient;
//!
//! let client = moonshot::Client::new("YOUR_API_KEY").expect("Failed to build client");
//!
//! let kimi_model = client.completion_model(moonshot::KIMI_K2_5);
//! ```
//!
//! # Custom base URL
//! The default base URL is `https://api.moonshot.ai/v1`. For China access,
//! use `https://api.moonshot.cn/v1`:
//! ```ignore
//! use rig_core::providers::moonshot;
//!
//! let client = moonshot::Client::builder()
//!     .api_key("YOUR_API_KEY")
//!     .base_url("https://api.moonshot.ai/v1")
//!     .build()
//!     .expect("Failed to build Moonshot client");
//! ```
use crate::client;
use crate::providers::internal::anthropic_compatible::{
    AnthropicBaseUrl, impl_dual_dialect_provider,
};
use crate::{completion::CompletionError, providers::openai};

// ================================================================
// Main Moonshot Client
// ================================================================
/// Global OpenAI-compatible base URL.
pub const GLOBAL_API_BASE_URL: &str = "https://api.moonshot.ai/v1";
/// China OpenAI-compatible base URL.
pub const CHINA_API_BASE_URL: &str = "https://api.moonshot.cn/v1";
/// Anthropic-compatible base URL.
pub const ANTHROPIC_API_BASE_URL: &str = "https://api.moonshot.ai/anthropic";

impl_dual_dialect_provider!(
    ext = MoonshotExt,
    builder = MoonshotBuilder,
    anthropic_ext = MoonshotAnthropicExt,
    anthropic_builder = MoonshotAnthropicBuilder,
    client_input = String,
    api_key_env = "MOONSHOT_API_KEY",
    base_url = GLOBAL_API_BASE_URL,
    base_url_env = "MOONSHOT_API_BASE",
    anthropic_provider_name = "moonshot",
    anthropic_base_url = ANTHROPIC_API_BASE_URL,
    anthropic_base_url_env = "MOONSHOT_ANTHROPIC_API_BASE",
);

client::impl_capabilities!(
    MoonshotExt,
    completion = CompletionModel<H>,
    model_listing = MoonshotModelLister<H>,
);

crate::providers::internal::model_listing::impl_model_lister!(
    /// [`ModelLister`](crate::client::ModelLister) implementation for the
    /// Moonshot API (`GET /models`).
    ///
    /// Moonshot documents the OpenAI-style `{"object":"list","data":[…]}`
    /// envelope; its entries carry extra fields (`context_length`,
    /// `supports_image_in`, …) that the shared entry ignores.
    MoonshotModelLister,
    Client<H>,
    crate::providers::internal::model_listing::ListModelEntry,
    "Moonshot",
    "/models"
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
        self.base_url(ANTHROPIC_API_BASE_URL)
    }
}

const ANTHROPIC_BASE_URLS: AnthropicBaseUrl = AnthropicBaseUrl::new(
    &[
        (GLOBAL_API_BASE_URL, ANTHROPIC_API_BASE_URL),
        (CHINA_API_BASE_URL, "https://api.moonshot.cn/anthropic"),
    ],
    &["/v1", "/v1/"],
    "/anthropic",
);

// ================================================================
// Moonshot Completion API
// ================================================================

/// Moonshot v1 128K context model (legacy)
pub const MOONSHOT_CHAT: &str = "moonshot-v1-128k";

/// Kimi K2 — Mixture-of-Experts model (1T total params, 32B active)
pub const KIMI_K2: &str = "kimi-k2";

/// Kimi K2.5 — Native multimodal agentic model with 256K context
pub const KIMI_K2_5: &str = "kimi-k2.5";

/// Moonshot completion model, driven by the shared OpenAI Chat Completions path.
pub type CompletionModel<H = crate::http_client::BoxedHttpClient> =
    openai::completion::GenericCompletionModel<MoonshotExt, H>;

impl openai::completion::OpenAICompatibleProvider for MoonshotExt {
    const PROVIDER_NAME: &'static str = "moonshot";

    type StreamingUsage = openai::Usage;

    // Moonshot's API rejects the `json_schema` response format; keep the
    // pre-migration behavior of dropping `output_schema` with a warning.
    const SUPPORTS_RESPONSE_FORMAT: bool = false;

    type Response = openai::CompletionResponse;

    fn prepare_request(
        &self,
        request: &mut openai::completion::CompletionRequest,
    ) -> Result<(), CompletionError> {
        // Moonshot only supports `auto`/`none` tool choices. Forcing one
        // specific tool has no workaround; fail fast like the pre-migration
        // conversion did (on main, `openai::ToolChoice::try_from` returned
        // "Provider doesn't support only using specific tools" for every
        // `ToolChoice::Specific`, single- or multi-name).
        if matches!(
            request.tool_choice,
            Some(openai::completion::ToolChoice::Function { .. })
        ) {
            return Err(CompletionError::ProviderError(
                "Moonshot does not support forcing a specific tool".to_string(),
            ));
        }

        // Moonshot does not support `tool_choice: "required"`; coerce it to
        // `auto` and steer the model with an extra user message instead.
        if matches!(
            request.tool_choice,
            Some(openai::completion::ToolChoice::Required)
        ) {
            tracing::warn!(
                "Moonshot does not support tool_choice=required; coercing to auto with an additional steering message"
            );
            request.tool_choice = Some(openai::completion::ToolChoice::Auto);
            request.messages.push(openai::Message::User {
                content: vec![openai::UserContent::Text {
                    text: "Please select a tool to handle the current issue.".to_string(),
                }],
                name: None,
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests;
