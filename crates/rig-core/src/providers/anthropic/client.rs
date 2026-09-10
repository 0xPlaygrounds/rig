//! Anthropic client api implementation
use http::{HeaderName, HeaderValue};

use super::completion::{ANTHROPIC_VERSION_LATEST, CompletionModel};
use crate::{
    client::{
        self, ApiKey, HasCompletion, HasModelListing, ModelTransport, Provider,
        ProviderClientResult,
    },
    http_client::{self, HttpClientExt},
    providers::anthropic::model_listing::AnthropicModelLister,
};

// ================================================================
// Main Anthropic Client
// ================================================================

/// The Anthropic Messages API provider.
#[derive(Debug, Default, Clone)]
pub struct Anthropic {
    /// Set when the client was built with [`AnthropicKey::sigv4`]. Carries the AWS region to sign
    /// for. `None` means header-based auth (`x-api-key`), which is the default and unchanged.
    ///
    /// This lives on the provider value rather than being inferred from the base URL so that the
    /// choice is explicit: a blank API key against an AWS host would otherwise silently become a
    /// signed request, and the resulting 403 would look nothing like a missing credential.
    pub(crate) sigv4_region: Option<String>,
}

/// Builder settings for [`Anthropic`] and every Anthropic-compatible
/// provider: the `anthropic-version` header and the `anthropic-beta` flags.
#[derive(Debug, Clone)]
pub struct AnthropicConfig {
    pub(crate) anthropic_version: String,
    pub(crate) anthropic_betas: Vec<String>,
}

impl Default for AnthropicConfig {
    fn default() -> Self {
        Self {
            anthropic_version: ANTHROPIC_VERSION_LATEST.into(),
            anthropic_betas: Vec::new(),
        }
    }
}

/// Anthropic API key, sent as the `x-api-key` header.
#[derive(Debug, Clone)]
pub enum AnthropicKey {
    /// Header-based auth. Sent as `x-api-key`. The default, and what `From<S: Into<String>>` builds,
    /// so existing callers are unaffected.
    ApiKey(String),
    /// AWS SigV4 request signing, for Anthropic-compatible endpoints fronted by AWS (the
    /// `bedrock-mantle` endpoint accepts either a Bedrock API key or SigV4 credentials).
    ///
    /// No static header is produced; signing is per-request because the signature covers the body
    /// and the current time. Credentials come from the standard AWS provider chain.
    SigV4 { region: String },
}

impl AnthropicKey {
    /// Sign requests with AWS SigV4 for `region` instead of sending an API key.
    ///
    /// Requires the `sigv4` cargo feature. Without it the client still builds, and the first request
    /// fails with a message naming the missing feature rather than sending an unsigned request.
    pub fn sigv4(region: impl Into<String>) -> Self {
        Self::SigV4 {
            region: region.into(),
        }
    }
}

impl<S> From<S> for AnthropicKey
where
    S: Into<String>,
{
    fn from(value: S) -> Self {
        Self::ApiKey(value.into())
    }
}

impl ApiKey for AnthropicKey {
    fn into_header(self) -> Option<http_client::Result<(http::HeaderName, HeaderValue)>> {
        match self {
            Self::ApiKey(key) => Some(
                HeaderValue::from_str(&key)
                    .map(|val| (HeaderName::from_static("x-api-key"), val))
                    .map_err(Into::into),
            ),
            // Deliberately no header: a SigV4 request must not also carry x-api-key, and the
            // signature cannot be computed here because the body does not exist yet.
            Self::SigV4 { .. } => None,
        }
    }
}

impl Provider for Anthropic {
    const NAME: &'static str = "anthropic";
    const BASE_URL: &'static str = "https://api.anthropic.com";
    const VERIFY_PATH: &'static str = "/v1/models";
    type ApiKey = AnthropicKey;
    type Config = AnthropicConfig;
    type EnvInput = String;

    fn build(_: AnthropicConfig, api_key: &AnthropicKey) -> http_client::Result<Self> {
        // Carry the signing region from the key onto the provider value, which is the only part of
        // the builder the request path can still see. `into_header()` discards the key itself.
        let sigv4_region = match api_key {
            AnthropicKey::SigV4 { region } => Some(region.clone()),
            AnthropicKey::ApiKey(_) => None,
        };
        Ok(Anthropic { sigv4_region })
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

/// Anthropic header settings, available on the builder of every provider
/// whose settings are [`AnthropicConfig`] (Anthropic itself and the
/// Anthropic-compatible dialects).
///
/// # Example
/// ```ignore
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
impl<P, H> client::ClientBuilder<P, H>
where
    P: Provider<Config = AnthropicConfig>,
{
    pub fn anthropic_version(self, anthropic_version: impl Into<String>) -> Self {
        self.map_config(|config| AnthropicConfig {
            anthropic_version: anthropic_version.into(),
            ..config
        })
    }

    pub fn anthropic_betas(self, anthropic_betas: &[&str]) -> Self {
        self.map_config(|mut config| {
            config
                .anthropic_betas
                .extend(anthropic_betas.iter().copied().map(String::from));

            config
        })
    }

    pub fn anthropic_beta(self, anthropic_beta: impl Into<String>) -> Self {
        self.map_config(|mut config| {
            config.anthropic_betas.push(anthropic_beta.into());

            config
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

/// The [`Provider::finish`] body shared by every provider configured through
/// [`AnthropicConfig`]: normalise the base URL and set the version/beta headers.
pub fn finish_anthropic_builder<P, H>(
    mut builder: client::ClientBuilder<P, H>,
) -> http_client::Result<client::ClientBuilder<P, H>>
where
    P: Provider<Config = AnthropicConfig>,
{
    let normalized_base_url = normalize_anthropic_base_url(builder.get_base_url());
    builder = builder.base_url(normalized_base_url);

    let config = builder.config().clone();
    builder.headers_mut().insert(
        "anthropic-version",
        HeaderValue::from_str(&config.anthropic_version)?,
    );

    if !config.anthropic_betas.is_empty() {
        builder.headers_mut().insert(
            "anthropic-beta",
            HeaderValue::from_str(&config.anthropic_betas.join(","))?,
        );
    }

    Ok(builder)
}

#[cfg(test)]
mod tests;
