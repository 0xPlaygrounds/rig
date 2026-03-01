//! Zai client api implementation
use http::{HeaderName, HeaderValue};

use super::completion::{CompletionModel, ZAI_VERSION_LATEST};
use crate::{
    client::{
        self, ApiKey, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder,
        ProviderClient,
    },
    http_client,
};

// ================================================================
// Main Zai Client
// ================================================================
#[derive(Debug, Default, Clone)]
pub struct ZaiExt;

impl Provider for ZaiExt {
    type Builder = ZaiBuilder;

    const VERIFY_PATH: &'static str = "/v1/models";

    fn build<H>(
        _builder: &client::ClientBuilder<Self::Builder, ZaiKey, H>,
    ) -> http_client::Result<Self> {
        Ok(Self)
    }
}

impl<H> Capabilities<H> for ZaiExt {
    type Completion = Capable<CompletionModel<H>>;

    type Embeddings = Nothing;
    type Transcription = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;
    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
}

#[derive(Debug, Clone)]
pub struct ZaiBuilder {
    zai_version: String,
    zai_betas: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ZaiKey(String);

impl<S> From<S> for ZaiKey
where
    S: Into<String>,
{
    fn from(value: S) -> Self {
        Self(value.into())
    }
}

impl ApiKey for ZaiKey {
    fn into_header(self) -> Option<http_client::Result<(http::HeaderName, HeaderValue)>> {
        Some(
            HeaderValue::from_str(&self.0)
                .map(|val| (HeaderName::from_static("x-api-key"), val))
                .map_err(Into::into),
        )
    }
}

pub type Client<H = reqwest::Client> = client::Client<ZaiExt, H>;
pub type ClientBuilder<H = reqwest::Client> = client::ClientBuilder<ZaiBuilder, ZaiKey, H>;

impl Default for ZaiBuilder {
    fn default() -> Self {
        Self {
            zai_version: ZAI_VERSION_LATEST.into(),
            zai_betas: Vec::new(),
        }
    }
}

impl ProviderBuilder for ZaiBuilder {
    type Output = ZaiExt;
    type ApiKey = ZaiKey;

    const BASE_URL: &'static str = "https://api.z.ai/api/anthropic";

    fn finish<H>(
        &self,
        mut builder: client::ClientBuilder<Self, ZaiKey, H>,
    ) -> http_client::Result<client::ClientBuilder<Self, ZaiKey, H>> {
        builder.headers_mut().insert(
            "anthropic-version",
            HeaderValue::from_str(&self.zai_version)?,
        );

        if !self.zai_betas.is_empty() {
            builder.headers_mut().insert(
                "anthropic-beta",
                HeaderValue::from_str(&self.zai_betas.join(","))?,
            );
        }

        Ok(builder)
    }
}

impl DebugExt for ZaiExt {}

impl ProviderClient for Client {
    type Input = String;

    fn from_env() -> Self
    where
        Self: Sized,
    {
        let key = std::env::var("ZAI_API_KEY").expect("ZAI_API_KEY not set");

        Self::builder().api_key(key).build().unwrap()
    }

    fn from_val(input: Self::Input) -> Self
    where
        Self: Sized,
    {
        Self::builder().api_key(input).build().unwrap()
    }
}

/// Create a new zai client using the builder
///
/// # Example
/// ```
/// use rig::providers::zai::{ClientBuilder, self};
///
/// // Initialize the Zai client
/// let zai_client = ClientBuilder::new("your-zai-api-key")
///    .zai_version(ZAI_VERSION_LATEST)
///    .zai_beta("prompt-caching-2024-07-31")
///    .build()
/// ```
impl<H> ClientBuilder<H> {
    pub fn zai_version(self, zai_version: &str) -> Self {
        self.over_ext(|ext| ZaiBuilder {
            zai_version: zai_version.into(),
            ..ext
        })
    }

    pub fn zai_betas(self, zai_betas: &[&str]) -> Self {
        self.over_ext(|mut ext| {
            ext.zai_betas
                .extend(zai_betas.iter().copied().map(String::from));

            ext
        })
    }

    pub fn zai_beta(self, zai_beta: &str) -> Self {
        self.over_ext(|mut ext| {
            ext.zai_betas.push(zai_beta.into());

            ext
        })
    }
}
