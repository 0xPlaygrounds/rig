//! Anthropic provider construction and capabilities.
use super::completion::CompletionModel;
use super::model_listing::AnthropicModelLister;
use crate::client::{
    self, HasCompletion, HasModelListing, ModelTransport, Provider, ProviderClientResult,
};
use crate::http_client::{self, HttpClientExt};
pub use crate::providers::anthropic_compatible::client::*;

/// The Anthropic Messages API provider.
#[derive(Debug, Default, Clone, Copy)]
pub struct Anthropic;

impl Provider for Anthropic {
    const NAME: &'static str = "anthropic";
    const BASE_URL: &'static str = "https://api.anthropic.com";
    const VERIFY_PATH: &'static str = "/v1/models";
    type ApiKey = AnthropicKey;
    type Config = AnthropicConfig;
    type EnvInput = String;

    fn build(_: AnthropicConfig, _: &AnthropicKey) -> http_client::Result<Self> {
        Ok(Anthropic)
    }

    fn finish<H>(
        &self,
        builder: client::ClientBuilder<Self, H>,
    ) -> http_client::Result<client::ClientBuilder<Self, H>> {
        finish_anthropic_builder(builder)
    }

    fn from_env<H: HttpClientExt>(http: H) -> ProviderClientResult<Client<H>> {
        Client::from_env_api_key("ANTHROPIC_API_KEY", Some("ANTHROPIC_BASE_URL"), http)
    }

    fn from_val<H: HttpClientExt>(input: String, http: H) -> ProviderClientResult<Client<H>> {
        Client::new_with(input, http)
    }
}

impl HasCompletion for Anthropic {
    type Model<H>
        = CompletionModel<H>
    where
        H: ModelTransport;

    fn completion_model<H: ModelTransport>(client: &Client<H>, model: String) -> Self::Model<H> {
        CompletionModel::new(client.clone(), model)
    }
}

impl HasModelListing for Anthropic {
    type Lister<H>
        = AnthropicModelLister<H>
    where
        H: ModelTransport;

    fn model_lister<H: ModelTransport>(client: &Client<H>) -> Self::Lister<H> {
        AnthropicModelLister::new(client.clone())
    }
}

pub type Client<H = crate::http_client::BoxedHttpClient> = client::Client<Anthropic, H>;
pub type ClientBuilder<H = crate::markers::Missing> = client::ClientBuilder<Anthropic, H>;

#[cfg(test)]
mod tests;
