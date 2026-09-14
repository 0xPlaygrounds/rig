//! Shared Anthropic-compatible authentication and version header settings.
use super::completion::ANTHROPIC_VERSION_LATEST;
use crate::client::{self, ApiKey, Provider};
use crate::http_client;
use http::{HeaderName, HeaderValue};

/// Builder settings for Anthropic and every Anthropic-compatible
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
#[derive(Clone)]
pub struct AnthropicKey(String);

impl std::fmt::Debug for AnthropicKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("AnthropicKey(<redacted>)")
    }
}

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

#[cfg(test)]
mod tests;

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
