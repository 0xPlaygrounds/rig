//! Anthropic client api implementation
use std::marker::PhantomData;

use http::HeaderValue;

use super::completion::{ANTHROPIC_VERSION_LATEST, CompletionModel};
use crate::{
    client::{
        CompletionClient,
        client::{self, DebugExt, IntoHeader, Provider, ProviderBuilder},
        impl_conversion_traits,
    },
    http_client::{self, HttpClientExt},
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};

pub type Client<H = reqwest::Client> = client::Client<AnthropicExt, H>;
pub type ClientBuilder<H = reqwest::Client> =
    client::ClientBuilder<AnthropicBuilder, AnthropicKey, H>;

// ================================================================
// Main Anthropic Client
// ================================================================
#[derive(Debug, Clone)]
pub struct AnthropicExt;

impl Provider for AnthropicExt {
    type ApiKey = AnthropicKey;
    type Builder = AnthropicBuilder;

    const VERIFY_PATH: &'static str = "/v1/models";

    fn build(_: Self::Builder) -> Self {
        Self
    }
}

impl ProviderBuilder for AnthropicBuilder {
    const BASE_URL: &'static str = "https://api.anthropic.com";

    fn finish<ApiKey, H>(
        &self,
        mut builder: client::ClientBuilder<Self, ApiKey, H>,
    ) -> client::ClientBuilder<Self, ApiKey, H> {
        todo!()
    }
}

impl DebugExt for AnthropicExt {
    fn with_fields<'a, 'b>(
        &'a self,
        f: &'b mut std::fmt::DebugStruct,
    ) -> &'b mut std::fmt::DebugStruct
    where
        'a: 'b,
    {
        f
    }
}

impl<'a> Default for AnthropicExt {
    fn default() -> Self {
        Self
    }
}

pub struct AnthropicKey(String);

impl From<String> for AnthropicKey {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl IntoHeader for AnthropicKey {
    fn make_header(self) -> Option<http_client::Result<(http::HeaderName, HeaderValue)>> {
        let header = HeaderValue::from_str(&self.0)
            .map(|val| (http::HeaderName::from_static("X-Api-Key"), val))
            .map_err(|e| http_client::Error::from(http::Error::from(e)));

        Some(header)
    }
}

impl<'a> From<&'a str> for AnthropicKey {
    fn from(value: &'a str) -> Self {
        Self(value.into())
    }
}

pub struct AnthropicBuilder {
    anthropic_version: String,
    anthropic_betas: Vec<String>,
}

impl Default for AnthropicBuilder {
    fn default() -> Self {
        Self {
            anthropic_version: ANTHROPIC_VERSION_LATEST.into(),
            anthropic_betas: Vec::new(),
        }
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
impl<H> ClientBuilder<H> {
    pub fn anthropic_version(self, anthropic_version: &str) -> Self {
        self.over_ext(|ext| AnthropicBuilder {
            anthropic_version: anthropic_version.into(),
            ..ext
        })
    }

    pub fn anthropic_betas(self, anthropic_betas: &[&str]) -> Self {
        self.over_ext(|mut ext| {
            ext.anthropic_betas
                .extend(anthropic_betas.iter().copied().map(String::from));

            ext
        })
    }

    pub fn anthropic_beta(self, anthropic_beta: &str) -> Self {
        self.over_ext(|mut ext| {
            ext.anthropic_betas.push(anthropic_beta.into());

            ext
        })
    }
}

// impl_conversion_traits!(
//     AsTranscription,
//     AsEmbeddings,
//     AsImageGeneration,
//     AsAudioGeneration
//     for Client<T>
// );
