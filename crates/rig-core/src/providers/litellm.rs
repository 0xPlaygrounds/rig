//! LiteLLM AI gateway client and Rig integration.
//!
//! [LiteLLM](https://github.com/BerriAI/litellm) is an AI gateway that
//! provides a unified OpenAI-compatible API to 100+ LLM providers (OpenAI,
//! Anthropic, Google, Azure, AWS Bedrock, and more).
//!
//! This provider connects to a running LiteLLM proxy and reuses Rig's
//! existing OpenAI-compatible completion and streaming machinery.
//!
//! # Example
//! ```no_run
//! use rig_core::{client::CompletionClient, providers::litellm};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! // Connect to a LiteLLM proxy running at the default address.
//! let client = litellm::Client::from_env()?;
//!
//! // Use any model configured on the proxy.
//! let model = client.completion_model("anthropic/claude-sonnet-4-20250514");
//! # Ok(())
//! # }
//! ```
//!
//! # Custom proxy address
//! ```no_run
//! use rig_core::providers::litellm;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let client = litellm::Client::builder()
//!     .api_key("sk-your-proxy-key")
//!     .base_url("https://litellm.example.com/v1")
//!     .build()?;
//! # Ok(())
//! # }
//! ```

use serde::Deserialize;

use crate::client::{
    self, BearerAuth, Capabilities, Capable, DebugExt, ModelLister, Nothing, Provider,
    ProviderBuilder, ProviderClient,
};
use crate::http_client::{self, HttpClientExt};
use crate::model::{Model, ModelList, ModelListingError};
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};

/// Default LiteLLM proxy base URL.
pub const LITELLM_API_BASE_URL: &str = "http://localhost:4000/v1";

#[derive(Debug, Default, Clone, Copy)]
pub struct LiteLLMExt;

#[derive(Debug, Default, Clone, Copy)]
pub struct LiteLLMBuilder;

type LiteLLMApiKey = BearerAuth;

pub type Client<H = reqwest::Client> = client::Client<LiteLLMExt, H>;
pub type ClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<LiteLLMBuilder, LiteLLMApiKey, H>;

impl Provider for LiteLLMExt {
    type Builder = LiteLLMBuilder;

    const VERIFY_PATH: &'static str = "/models";
}

impl<H> Capabilities<H> for LiteLLMExt {
    type Completion = Capable<super::openai::completion::GenericCompletionModel<LiteLLMExt, H>>;
    type Embeddings = Nothing;
    type Transcription = Nothing;
    type ModelListing = Capable<LiteLLMModelLister<H>>;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;
    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
    type Rerank = Nothing;
}

impl DebugExt for LiteLLMExt {}

impl ProviderBuilder for LiteLLMBuilder {
    type Extension<H>
        = LiteLLMExt
    where
        H: HttpClientExt;
    type ApiKey = LiteLLMApiKey;

    const BASE_URL: &'static str = LITELLM_API_BASE_URL;

    fn build<H>(
        _builder: &client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: HttpClientExt,
    {
        Ok(LiteLLMExt)
    }
}

impl ProviderClient for Client {
    type Input = LiteLLMApiKey;
    type Error = crate::client::ProviderClientError;

    fn from_env() -> Result<Self, Self::Error> {
        let api_key = crate::client::required_env_var("LITELLM_API_KEY")?;
        let mut builder = Self::builder().api_key(api_key);

        if let Some(base_url) = crate::client::optional_env_var("LITELLM_API_BASE")? {
            builder = builder.base_url(base_url);
        }

        builder.build().map_err(Into::into)
    }

    fn from_val(input: Self::Input) -> Result<Self, Self::Error> {
        Self::new(input).map_err(Into::into)
    }
}

// ================================================================
// Model Listing
// ================================================================

#[derive(Debug, Deserialize)]
struct ListModelsResponse {
    data: Vec<ListModelEntry>,
}

#[derive(Debug, Deserialize)]
struct ListModelEntry {
    id: String,
    #[serde(default)]
    created: u64,
    #[serde(default)]
    owned_by: String,
}

impl From<ListModelEntry> for Model {
    fn from(value: ListModelEntry) -> Self {
        let mut model = Model::from_id(value.id);
        model.created_at = Some(value.created);
        model.owned_by = Some(value.owned_by);
        model
    }
}

#[derive(Clone)]
pub struct LiteLLMModelLister<H = reqwest::Client> {
    client: Client<H>,
}

impl<H> ModelLister<H> for LiteLLMModelLister<H>
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
                        "LiteLLM",
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
                "LiteLLM",
                path,
                status_code,
                &body,
            ));
        }

        let body = response.into_body().await?;
        let api_resp: ListModelsResponse = serde_json::from_slice(&body).map_err(|error| {
            ModelListingError::parse_error_with_context("LiteLLM", path, &error, &body)
        })?;

        let models = api_resp.data.into_iter().map(Model::from).collect();

        Ok(ModelList::new(models))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_initialization() {
        let _client = crate::providers::litellm::Client::new("dummy-key").expect("Client::new()");
        let _client_from_builder = crate::providers::litellm::Client::builder()
            .api_key("dummy-key")
            .build()
            .expect("Client::builder()");
    }

    #[test]
    fn test_client_with_custom_base_url() {
        let _client = crate::providers::litellm::Client::builder()
            .api_key("dummy-key")
            .base_url("https://litellm.example.com/v1")
            .build()
            .expect("Client::builder() with custom base_url");
    }

    #[test]
    fn test_model_listing_response_parsing() {
        let data = r#"{
            "data": [
                {"id": "gpt-4o", "created": 1700000000, "owned_by": "openai"},
                {"id": "claude-sonnet-4-20250514", "created": 1700000001, "owned_by": "anthropic"}
            ]
        }"#;

        let response: ListModelsResponse = serde_json::from_str(data).unwrap();
        assert_eq!(response.data.len(), 2);
        assert_eq!(response.data[0].id, "gpt-4o");
        assert_eq!(response.data[1].id, "claude-sonnet-4-20250514");
    }

    #[test]
    fn test_model_listing_response_with_missing_fields() {
        let data = r#"{
            "data": [
                {"id": "custom-model"}
            ]
        }"#;

        let response: ListModelsResponse = serde_json::from_str(data).unwrap();
        assert_eq!(response.data.len(), 1);
        assert_eq!(response.data[0].id, "custom-model");
        assert_eq!(response.data[0].created, 0);
        assert_eq!(response.data[0].owned_by, "");
    }
}
