//! MiniMax API clients and Rig integrations.
//!
//! MiniMax exposes both OpenAI-compatible and Anthropic-compatible chat APIs,
//! with distinct global and China entrypoints.
//!
//! # OpenAI-compatible example
//! ```ignore
//! use rig_core::client::CompletionClient;
//! use rig_core::providers::minimax;
//!
//! let client = minimax::Client::new("YOUR_API_KEY").expect("Failed to build client");
//! let model = client.completion_model(minimax::MINIMAX_M2_7);
//! ```
//!
//! # Anthropic-compatible example
//! ```ignore
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
    provider = MiniMax,
    anthropic_provider = MiniMaxAnthropic,
    client_input = client::BearerAuth,
    name = "minimax",
    api_key_env = "MINIMAX_API_KEY",
    base_url = GLOBAL_API_BASE_URL,
    base_url_env = "MINIMAX_API_BASE",
    anthropic_provider_name = "minimax",
    anthropic_base_url = GLOBAL_ANTHROPIC_API_BASE_URL,
    anthropic_base_url_env = "MINIMAX_ANTHROPIC_API_BASE",
);

impl client::HasCompletion for MiniMax {
    type Model<H>
        = super::openai::completion::GenericCompletionModel<MiniMax, H>
    where
        H: client::ModelTransport;

    fn completion_model<H: client::ModelTransport>(
        client: &Client<H>,
        model: String,
    ) -> Self::Model<H> {
        super::openai::completion::GenericCompletionModel::new(client.clone(), model)
    }
}

impl client::HasModelListing for MiniMax {
    type Lister<H>
        = MiniMaxModelLister<H>
    where
        H: client::ModelTransport;

    fn model_lister<H: client::ModelTransport>(client: &Client<H>) -> Self::Lister<H> {
        MiniMaxModelLister::new(client.clone())
    }
}

crate::providers::internal::model_listing::impl_model_lister!(
    /// [`ModelLister`](crate::client::ModelLister) implementation for the
    /// MiniMax API (`GET /models`).
    ///
    /// MiniMax documents the OpenAI-style `{"object":"list","data":[…]}`
    /// envelope with `id`, `created` and `owned_by` on each entry.
    MiniMaxModelLister,
    Client<H>,
    crate::providers::internal::model_listing::ListModelEntry,
    "MiniMax",
    "/models"
);

impl super::openai::completion::OpenAICompatibleProvider for MiniMax {
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
mod tests;
