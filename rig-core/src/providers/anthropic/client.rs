//! Anthropic client api implementation
use std::marker::PhantomData;

use http::HeaderValue;

use super::completion::{ANTHROPIC_VERSION_LATEST, CompletionModel};
use crate::{
    client::{
        CompletionClient,
        client::{self, ClientExtBuilder, ClientSpecific, IntoHeader},
        impl_conversion_traits,
    },
    http_client::{self, HttpClientExt},
    wasm_compat::WasmCompatSend,
};

type Client<'a, H = reqwest::Client> = client::Client<'a, AnthropicExt<'a>, H>;
type ClientBuilder<'a, H = reqwest::Client> =
    client::ClientBuilder<AnthropicBuilder<'a>, AnthropicKey<'a>, H>;

// ================================================================
// Main Anthropic Client
// ================================================================
#[derive(Debug, Clone)]
pub struct AnthropicExt<'a>(PhantomData<&'a ()>);

impl<'a> ClientSpecific<'a> for AnthropicExt<'a> {
    type ApiKey = AnthropicKey<'a>;
    const BASE_URL: &'static str = "https://api.anthropic.com";
    const VERIFY_PATH: &'static str = "/v1/models";
}

impl<'a> Default for AnthropicExt<'a> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

pub struct AnthropicKey<'a>(&'a str);

impl IntoHeader for AnthropicKey<'_> {
    fn make_header(self) -> Option<http_client::Result<(http::HeaderName, HeaderValue)>> {
        Some(
            HeaderValue::from_str(self.0)
                .map(|val| (http::HeaderName::from_static("X-Api-Key"), val))
                .map_err(|e| http_client::Error::from(http::Error::from(e))),
        )
    }
}

impl<'a> From<&'a str> for AnthropicKey<'a> {
    fn from(value: &'a str) -> Self {
        Self(value)
    }
}

pub struct AnthropicBuilder<'a> {
    anthropic_version: &'static str,
    anthropic_betas: Vec<&'static str>,
    _lifetime: PhantomData<&'a ()>,
}

impl Default for AnthropicBuilder<'_> {
    fn default() -> Self {
        Self {
            anthropic_version: ANTHROPIC_VERSION_LATEST,
            anthropic_betas: Vec::new(),
            _lifetime: PhantomData,
        }
    }
}

impl<'a> ClientExtBuilder<'a> for AnthropicBuilder<'a> {
    type Extension = AnthropicExt<'a>;

    fn customize(&self, mut headers: http::HeaderMap) -> http_client::Result<http::HeaderMap> {
        headers.insert(
            "anthropic-version",
            HeaderValue::from_str(self.anthropic_version)?,
        );

        if !self.anthropic_betas.is_empty() {
            headers.insert(
                "anthropic-beta",
                HeaderValue::from_str(&self.anthropic_betas.join(","))?,
            );
        }

        Ok(headers)
    }

    fn build(self) -> Self::Extension {
        AnthropicExt(PhantomData)
    }
}

/// Create a new anthropic client using the builder
///
/// # Example
/// ```
/// use rig::providers::anthropic::{ClientBuilder, self};
///
/// // Initialize the Anthropic client
/// let anthropic_client = ClientBuilder::new("your-claude-api-key")
///    .anthropic_version(ANTHROPIC_VERSION_LATEST)
///    .anthropic_beta("prompt-caching-2024-07-31")
///    .build()
/// ```
impl<'a, H> ClientBuilder<'a, H> {
    pub fn anthropic_version(self, anthropic_version: &'static str) -> Self {
        Self {
            ext: AnthropicBuilder {
                anthropic_version,
                ..self.ext
            },
            ..self
        }
    }

    pub fn anthropic_betas(mut self, anthropic_betas: &[&'static str]) -> Self {
        self.ext.anthropic_betas.extend(anthropic_betas);

        self
    }

    pub fn anthropic_beta(mut self, anthropic_beta: &'static str) -> Self {
        self.ext.anthropic_betas.push(anthropic_beta);

        self
    }
}

impl<'a, H> CompletionClient for Client<'a, H>
where
    H: HttpClientExt + Clone + std::fmt::Debug,
{
    type CompletionModel = CompletionModel;

    fn completion_model(&self, model: &str) -> CompletionModel<T> {
        CompletionModel::new(self.clone(), model)
    }
}

impl_conversion_traits!(
    AsTranscription,
    AsEmbeddings,
    AsImageGeneration,
    AsAudioGeneration
    for Client<T>
);
