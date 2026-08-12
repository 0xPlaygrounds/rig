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

use crate::client::{
    self, BearerAuth, Capabilities, Capable, DebugExt, ModelLister, Nothing, Provider,
};
use crate::http_client::HttpClientExt;
use crate::model::{Model, ModelList, ModelListingError};
use crate::providers::anthropic::client::{
    AnthropicBuilder as AnthropicCompatBuilder, AnthropicKey, impl_anthropic_compatible_builder,
};
use crate::providers::internal::anthropic_compatible::AnthropicBaseUrl;
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};

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

#[derive(Debug, Default, Clone, Copy)]
pub struct XiaomiMimoExt;

#[derive(Debug, Default, Clone, Copy)]
pub struct XiaomiMimoBuilder;

#[derive(Debug, Default, Clone)]
pub struct XiaomiMimoAnthropicBuilder {
    anthropic: AnthropicCompatBuilder,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct XiaomiMimoAnthropicExt;

type XiaomiMimoApiKey = BearerAuth;

pub type Client<H = reqwest::Client> = client::Client<XiaomiMimoExt, H>;
pub type ClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<XiaomiMimoBuilder, XiaomiMimoApiKey, H>;

pub type AnthropicClient<H = reqwest::Client> = client::Client<XiaomiMimoAnthropicExt, H>;
pub type AnthropicClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<XiaomiMimoAnthropicBuilder, AnthropicKey, H>;

impl Provider for XiaomiMimoExt {
    type Builder = XiaomiMimoBuilder;

    const VERIFY_PATH: &'static str = "/models";
}

impl Provider for XiaomiMimoAnthropicExt {
    type Builder = XiaomiMimoAnthropicBuilder;

    const VERIFY_PATH: &'static str = "/v1/models";
}

impl<H> Capabilities<H> for XiaomiMimoExt {
    type Completion = Capable<super::openai::completion::GenericCompletionModel<XiaomiMimoExt, H>>;
    type Embeddings = Nothing;
    type Transcription = Nothing;
    type ModelListing = Capable<XiaomiMimoModelLister<H>>;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;
    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
    type Rerank = Nothing;
}

impl<H> Capabilities<H> for XiaomiMimoAnthropicExt {
    type Completion =
        Capable<super::anthropic::completion::GenericCompletionModel<XiaomiMimoAnthropicExt, H>>;
    type Embeddings = Nothing;
    type Transcription = Nothing;
    type ModelListing = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;
    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
    type Rerank = Nothing;
}

impl DebugExt for XiaomiMimoExt {}
impl DebugExt for XiaomiMimoAnthropicExt {}

impl super::openai::completion::OpenAICompatibleProvider for XiaomiMimoExt {
    const PROVIDER_NAME: &'static str = "xiaomimimo";

    type StreamingUsage = super::openai::Usage;

    type Response = super::openai::CompletionResponse;
}

client::impl_default_provider_builder!(
    XiaomiMimoBuilder => XiaomiMimoExt,
    api_key = XiaomiMimoApiKey,
    base_url = API_BASE_URL,
);
impl_anthropic_compatible_builder!(
    XiaomiMimoAnthropicBuilder => XiaomiMimoAnthropicExt,
    base_url = ANTHROPIC_API_BASE_URL,
);

impl super::anthropic::completion::AnthropicCompatibleProvider for XiaomiMimoAnthropicExt {
    const PROVIDER_NAME: &'static str = "xiaomimimo";

    fn default_max_tokens(_model: &str) -> Option<u64> {
        Some(4096)
    }
}

client::impl_provider_client!(
    Client,
    input = XiaomiMimoApiKey,
    api_key_env = "XIAOMI_MIMO_API_KEY",
    base_url_env = "XIAOMI_MIMO_API_BASE",
);

client::impl_provider_client!(
    AnthropicClient,
    input = String,
    api_key_env = "XIAOMI_MIMO_API_KEY",
    base_url = ANTHROPIC_BASE_URLS
        .resolve_from_env("XIAOMI_MIMO_ANTHROPIC_API_BASE", "XIAOMI_MIMO_API_BASE")?,
);

const ANTHROPIC_BASE_URLS: AnthropicBaseUrl = AnthropicBaseUrl::new(
    &[(API_BASE_URL, ANTHROPIC_API_BASE_URL)],
    &["/v1", "/v1/"],
    "/anthropic/v1",
);

#[derive(Debug, serde::Deserialize)]
struct ListModelEntry {
    id: String,
    owned_by: String,
}

impl From<ListModelEntry> for Model {
    fn from(value: ListModelEntry) -> Self {
        let mut model = Model::from_id(value.id);
        model.owned_by = Some(value.owned_by);
        model
    }
}

/// [`ModelLister`] implementation for the Xiaomi MiMo API (`GET /models`).
#[derive(Clone)]
pub struct XiaomiMimoModelLister<H = reqwest::Client> {
    client: Client<H>,
}

impl<H> ModelLister<H> for XiaomiMimoModelLister<H>
where
    H: HttpClientExt + WasmCompatSend + WasmCompatSync + 'static,
{
    type Client = Client<H>;

    fn new(client: Self::Client) -> Self {
        Self { client }
    }

    async fn list_all(&self) -> Result<ModelList, ModelListingError> {
        crate::providers::internal::model_listing::list_models::<ListModelEntry, _, _>(
            &self.client,
            "Xiaomi MiMo",
            "/models",
        )
        .await
    }
}

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
