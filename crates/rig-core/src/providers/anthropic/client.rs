//! Anthropic client api implementation
use http::{HeaderName, HeaderValue};

use super::completion::{ANTHROPIC_VERSION_LATEST, CompletionModel};
use crate::{
    client::{
        self, ApiKey, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder,
        ProviderClient,
    },
    http_client::{self, HttpClientExt},
    providers::anthropic::model_listing::AnthropicModelLister,
};

// ================================================================
// Main Anthropic Client
// ================================================================
#[derive(Debug, Default, Clone)]
pub struct AnthropicExt;

impl Provider for AnthropicExt {
    type Builder = AnthropicBuilder;
    const VERIFY_PATH: &'static str = "/v1/models";
}

impl<H> Capabilities<H> for AnthropicExt {
    type Completion = Capable<CompletionModel<H>>;

    type Embeddings = Nothing;
    type Transcription = Nothing;
    type ModelListing = Capable<AnthropicModelLister<H>>;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;
    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
    type Rerank = Nothing;
}

#[derive(Debug, Clone)]
pub struct AnthropicBuilder {
    pub(crate) anthropic_version: String,
    pub(crate) anthropic_betas: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct AnthropicKey(String);

impl<S> From<S> for AnthropicKey
where
    S: Into<String>,
{
    fn from(value: S) -> Self {
        Self(value.into())
    }
}

impl ApiKey for AnthropicKey {
    fn into_header(self) -> Option<http_client::Result<(http::HeaderName, HeaderValue)>> {
        Some(
            HeaderValue::from_str(&self.0)
                .map(|val| (HeaderName::from_static("x-api-key"), val))
                .map_err(Into::into),
        )
    }
}

pub type Client<H = reqwest::Client> = client::Client<AnthropicExt, H>;
pub type ClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<AnthropicBuilder, AnthropicKey, H>;

impl Default for AnthropicBuilder {
    fn default() -> Self {
        Self {
            anthropic_version: ANTHROPIC_VERSION_LATEST.into(),
            anthropic_betas: Vec::new(),
        }
    }
}

impl ProviderBuilder for AnthropicBuilder {
    type Extension<H>
        = AnthropicExt
    where
        H: HttpClientExt;
    type ApiKey = AnthropicKey;

    const BASE_URL: &'static str = "https://api.anthropic.com";

    fn build<H>(
        _builder: &client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: HttpClientExt,
    {
        Ok(AnthropicExt)
    }

    fn finish<H>(
        &self,
        builder: client::ClientBuilder<Self, AnthropicKey, H>,
    ) -> http_client::Result<client::ClientBuilder<Self, AnthropicKey, H>> {
        finish_anthropic_builder(self, builder)
    }
}

impl DebugExt for AnthropicExt {}

impl ProviderClient for Client {
    type Input = String;
    type Error = crate::client::ProviderClientError;

    fn from_env() -> Result<Self, Self::Error>
    where
        Self: Sized,
    {
        let base_url = crate::client::optional_env_var("ANTHROPIC_BASE_URL")?;
        let key = crate::client::required_env_var("ANTHROPIC_API_KEY")?;

        let mut builder = Self::builder().api_key(key);

        if let Some(base) = base_url {
            builder = builder.base_url(&base);
        }

        builder.build().map_err(Into::into)
    }

    fn from_val(input: Self::Input) -> Result<Self, Self::Error>
    where
        Self: Sized,
    {
        Self::builder().api_key(input).build().map_err(Into::into)
    }
}

/// Create a new anthropic client using the builder
///
/// # Example
/// ```no_run
/// use rig_core::providers::anthropic::{Client, self};
/// use rig_core::providers::anthropic::completion::ANTHROPIC_VERSION_LATEST;
///
/// # fn run() -> Result<(), Box<dyn std::error::Error>> {
/// // Initialize the Anthropic client
/// let anthropic_client = Client::builder()
///    .api_key("your-claude-api-key")
///    .anthropic_version(ANTHROPIC_VERSION_LATEST)
///    .anthropic_beta("prompt-caching-2024-07-31")
///    .build()?;
/// # Ok(())
/// # }
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

pub fn normalize_anthropic_base_url(base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');

    if let Some(stripped) = trimmed.strip_suffix("/v1/messages") {
        stripped.to_string()
    } else if let Some(stripped) = trimmed.strip_suffix("/messages") {
        stripped.to_string()
    } else if let Some(stripped) = trimmed.strip_suffix("/v1") {
        stripped.to_string()
    } else {
        trimmed.to_string()
    }
}

pub fn finish_anthropic_builder<ExtBuilder, H>(
    ext: &AnthropicBuilder,
    mut builder: client::ClientBuilder<ExtBuilder, AnthropicKey, H>,
) -> http_client::Result<client::ClientBuilder<ExtBuilder, AnthropicKey, H>>
where
    ExtBuilder: Clone,
{
    let normalized_base_url = normalize_anthropic_base_url(builder.get_base_url());
    builder = builder.base_url(normalized_base_url);

    builder.headers_mut().insert(
        "anthropic-version",
        HeaderValue::from_str(&ext.anthropic_version)?,
    );

    if !ext.anthropic_betas.is_empty() {
        builder.headers_mut().insert(
            "anthropic-beta",
            HeaderValue::from_str(&ext.anthropic_betas.join(","))?,
        );
    }

    Ok(builder)
}
#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use crate::client::ProviderClient;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct EnvVarGuard {
        key: &'static str,
        original: Option<String>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let original = std::env::var(key).ok();
            // SAFETY: Tests in this module hold ENV_LOCK while mutating process
            // environment and restore the original value before releasing it.
            unsafe { std::env::set_var(key, value) };

            Self { key, original }
        }

        fn remove(key: &'static str) -> Self {
            let original = std::env::var(key).ok();
            // SAFETY: Tests in this module hold ENV_LOCK while mutating process
            // environment and restore the original value before releasing it.
            unsafe { std::env::remove_var(key) };

            Self { key, original }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            // SAFETY: Tests in this module hold ENV_LOCK while mutating process
            // environment and restore the original value before releasing it.
            unsafe {
                match &self.original {
                    Some(value) => std::env::set_var(self.key, value),
                    None => std::env::remove_var(self.key),
                }
            }
        }
    }

    #[test]
    fn test_client_initialization() {
        let _client =
            crate::providers::anthropic::Client::new("dummy-key").expect("Client::new() failed");
        let _client_from_builder = crate::providers::anthropic::Client::builder()
            .api_key("dummy-key")
            .build()
            .expect("Client::builder() failed");
    }

    #[test]
    fn from_env_uses_anthropic_base_url() {
        let _guard = ENV_LOCK.lock().expect("env lock should not be poisoned");
        let _api_key = EnvVarGuard::set("ANTHROPIC_API_KEY", "dummy-key");
        let _base_url = EnvVarGuard::set(
            "ANTHROPIC_BASE_URL",
            "https://anthropic-compatible.example/v1/messages",
        );

        let client = crate::providers::anthropic::Client::from_env()
            .expect("Client::from_env should build with ANTHROPIC_BASE_URL");

        assert_eq!(
            client.base_url(),
            "https://anthropic-compatible.example",
            "from_env should apply ANTHROPIC_BASE_URL and the existing Anthropic base URL normalization"
        );
    }

    #[test]
    fn from_env_uses_default_base_url_when_anthropic_base_url_is_unset() {
        let _guard = ENV_LOCK.lock().expect("env lock should not be poisoned");
        let _api_key = EnvVarGuard::set("ANTHROPIC_API_KEY", "dummy-key");
        let _base_url = EnvVarGuard::remove("ANTHROPIC_BASE_URL");

        let client = crate::providers::anthropic::Client::from_env()
            .expect("Client::from_env should build without ANTHROPIC_BASE_URL");

        assert_eq!(client.base_url(), "https://api.anthropic.com");
    }
}
