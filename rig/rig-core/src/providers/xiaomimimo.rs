//! Xiaomi MiMo API clients and Rig integrations.
//!
//! Xiaomi exposes both OpenAI-compatible and Anthropic-compatible chat APIs
//! under a single global host.
//!
//! # OpenAI-compatible example
//! ```no_run
//! use rig::client::CompletionClient;
//! use rig::providers::xiaomimimo;
//!
//! let client = xiaomimimo::Client::new("YOUR_API_KEY").expect("Failed to build client");
//! let model = client.completion_model(xiaomimimo::MIMO_V2_5_PRO);
//! ```
//!
//! # Anthropic-compatible example
//! ```no_run
//! use rig::client::CompletionClient;
//! use rig::providers::xiaomimimo;
//!
//! let client = xiaomimimo::AnthropicClient::new("YOUR_API_KEY").expect("Failed to build client");
//! let model = client.completion_model(xiaomimimo::MIMO_V2_5_PRO);
//! ```

use crate::client::{
    self, BearerAuth, Capabilities, Capable, DebugExt, ModelLister, Nothing, Provider,
    ProviderBuilder, ProviderClient,
};
use crate::http_client::{self, HttpClientExt};
use crate::model::{Model, ModelList, ModelListingError};
use crate::providers::anthropic::client::{
    AnthropicBuilder as AnthropicCompatBuilder, AnthropicKey, finish_anthropic_builder,
};
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
pub type ClientBuilder<H = reqwest::Client> =
    client::ClientBuilder<XiaomiMimoBuilder, XiaomiMimoApiKey, H>;

pub type AnthropicClient<H = reqwest::Client> = client::Client<XiaomiMimoAnthropicExt, H>;
pub type AnthropicClientBuilder<H = reqwest::Client> =
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
}

impl DebugExt for XiaomiMimoExt {}
impl DebugExt for XiaomiMimoAnthropicExt {}

impl ProviderBuilder for XiaomiMimoBuilder {
    type Extension<H>
        = XiaomiMimoExt
    where
        H: HttpClientExt;
    type ApiKey = XiaomiMimoApiKey;

    const BASE_URL: &'static str = API_BASE_URL;

    fn build<H>(
        _builder: &client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: HttpClientExt,
    {
        Ok(XiaomiMimoExt)
    }
}

impl ProviderBuilder for XiaomiMimoAnthropicBuilder {
    type Extension<H>
        = XiaomiMimoAnthropicExt
    where
        H: HttpClientExt;
    type ApiKey = AnthropicKey;

    const BASE_URL: &'static str = ANTHROPIC_API_BASE_URL;

    fn build<H>(
        _builder: &client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: HttpClientExt,
    {
        Ok(XiaomiMimoAnthropicExt)
    }

    fn finish<H>(
        &self,
        builder: client::ClientBuilder<Self, AnthropicKey, H>,
    ) -> http_client::Result<client::ClientBuilder<Self, AnthropicKey, H>> {
        finish_anthropic_builder(&self.anthropic, builder)
    }
}

impl super::anthropic::completion::AnthropicCompatibleProvider for XiaomiMimoAnthropicExt {
    const PROVIDER_NAME: &'static str = "xiaomimimo";

    fn default_max_tokens(_model: &str) -> Option<u64> {
        Some(4096)
    }
}

impl ProviderClient for Client {
    type Input = XiaomiMimoApiKey;
    type Error = crate::client::ProviderClientError;

    fn from_env() -> Result<Self, Self::Error> {
        let api_key = crate::client::required_env_var("XIAOMI_MIMO_API_KEY")?;
        let mut builder = Self::builder().api_key(api_key);

        if let Some(base_url) = crate::client::optional_env_var("XIAOMI_MIMO_API_BASE")? {
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
        let api_key = crate::client::required_env_var("XIAOMI_MIMO_API_KEY")?;
        let mut builder = Self::builder().api_key(api_key);

        if let Some(base_url) =
            anthropic_base_override("XIAOMI_MIMO_ANTHROPIC_API_BASE", "XIAOMI_MIMO_API_BASE")?
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

    if base_url.trim_end_matches('/') == API_BASE_URL {
        return Some(ANTHROPIC_API_BASE_URL.to_owned());
    }

    let mut url = url::Url::parse(base_url).ok()?;
    if !matches!(url.path(), "/v1" | "/v1/") {
        return None;
    }
    url.set_path("/anthropic/v1");
    Some(url.to_string())
}

impl<H> AnthropicClientBuilder<H> {
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

#[derive(Debug, serde::Deserialize)]
struct ListModelsResponse {
    data: Vec<ListModelEntry>,
}

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
        let path = "/models";
        let req = self.client.get(path)?.body(http_client::NoBody)?;
        let response = self
            .client
            .send::<_, Vec<u8>>(req)
            .await
            .map_err(|error| match error {
                http_client::Error::InvalidStatusCodeWithMessage(status, message) => {
                    ModelListingError::api_error_with_context(
                        "Xiaomi MiMo",
                        path,
                        status.as_u16(),
                        message.as_bytes(),
                    )
                }
                other => ModelListingError::from(other),
            })?;

        if !response.status().is_success() {
            let status_code = response.status().as_u16();
            let body = response.into_body().await?;
            return Err(ModelListingError::api_error_with_context(
                "Xiaomi MiMo",
                path,
                status_code,
                &body,
            ));
        }

        let body = response.into_body().await?;
        let api_resp: ListModelsResponse = serde_json::from_slice(&body).map_err(|error| {
            ModelListingError::parse_error_with_context("Xiaomi MiMo", path, &error, &body)
        })?;

        let models = api_resp.data.into_iter().map(Model::from).collect();

        Ok(ModelList::new(models))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ANTHROPIC_API_BASE_URL, API_BASE_URL, normalize_anthropic_base_url,
        resolve_anthropic_base_override,
    };

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
            normalize_anthropic_base_url(API_BASE_URL).as_deref(),
            Some(ANTHROPIC_API_BASE_URL)
        );
        assert_eq!(
            normalize_anthropic_base_url("https://proxy.example.com/v1").as_deref(),
            Some("https://proxy.example.com/anthropic/v1")
        );
    }

    #[test]
    fn normalize_preserves_existing_anthropic_base() {
        assert_eq!(
            normalize_anthropic_base_url(ANTHROPIC_API_BASE_URL).as_deref(),
            Some(ANTHROPIC_API_BASE_URL)
        );
    }

    #[test]
    fn anthropic_primary_override_wins() {
        let override_url = resolve_anthropic_base_override(
            Some("https://primary.example.com/anthropic/v1"),
            Some(API_BASE_URL),
        );

        assert_eq!(
            override_url.as_deref(),
            Some("https://primary.example.com/anthropic/v1")
        );
    }
}
