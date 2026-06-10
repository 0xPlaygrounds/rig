//! Anthropic client api implementation
use std::{
    fmt,
    path::{Path, PathBuf},
};

use http::{HeaderName, HeaderValue};

use super::{
    auth,
    completion::{ANTHROPIC_VERSION_LATEST, CompletionModel, OAuthRequestHook},
};
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
#[derive(Clone)]
pub struct AnthropicExt {
    pub(crate) authenticator: auth::Authenticator,
    pub(crate) oauth_request_hook: Option<OAuthRequestHook>,
}

impl fmt::Debug for AnthropicExt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnthropicExt")
            .field("authenticator", &self.authenticator)
            .field("oauth_request_hook", &self.oauth_request_hook.is_some())
            .finish()
    }
}

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
}

#[derive(Clone)]
pub struct AnthropicBuilder {
    pub(crate) anthropic_version: String,
    pub(crate) anthropic_betas: Vec<String>,
    pub(crate) auth_file: Option<PathBuf>,
    pub(crate) oauth_prompt_handler: auth::OAuthPromptHandler,
    pub(crate) manual_code_handler: auth::ManualCodeHandler,
    pub(crate) oauth_request_hook: Option<OAuthRequestHook>,
}

impl fmt::Debug for AnthropicBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnthropicBuilder")
            .field("anthropic_version", &self.anthropic_version)
            .field("anthropic_betas", &self.anthropic_betas)
            .field("auth_file", &self.auth_file)
            .field("oauth_prompt_handler", &self.oauth_prompt_handler)
            .field("manual_code_handler", &self.manual_code_handler)
            .field("oauth_request_hook", &self.oauth_request_hook.is_some())
            .finish()
    }
}

#[derive(Debug, Clone)]
pub enum AnthropicKey {
    ApiKey(String),
    OAuth,
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
            Self::OAuth => None,
        }
    }
}

pub type Client<H = reqwest::Client> = client::Client<AnthropicExt, H>;
pub type ClientBuilder<H = crate::markers::Missing> =
    client::ClientBuilder<AnthropicBuilder, AnthropicKey, H>;

impl<H> Client<H>
where
    H: HttpClientExt
        + Clone
        + std::fmt::Debug
        + Default
        + crate::wasm_compat::WasmCompatSend
        + crate::wasm_compat::WasmCompatSync
        + 'static,
{
    /// Force the Anthropic OAuth flow, caching tokens without sending a request.
    ///
    /// # Returns
    ///
    /// Returns an error if the OAuth login, refresh, or token-cache write fails.
    pub async fn authorize(&self) -> Result<(), auth::AuthError> {
        self.ext().authenticator.auth_context().await.map(|_| ())
    }
}

impl Default for AnthropicBuilder {
    fn default() -> Self {
        Self {
            anthropic_version: ANTHROPIC_VERSION_LATEST.into(),
            anthropic_betas: Vec::new(),
            auth_file: default_auth_file(),
            oauth_prompt_handler: auth::OAuthPromptHandler::default(),
            manual_code_handler: auth::ManualCodeHandler::default(),
            oauth_request_hook: None,
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
        builder: &client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: HttpClientExt,
    {
        let ext = builder.ext();
        let source = match builder.get_api_key() {
            AnthropicKey::ApiKey(key) => auth::AuthSource::ApiKey(key.clone()),
            AnthropicKey::OAuth => auth::AuthSource::OAuth,
        };
        let authenticator = auth::Authenticator::new(
            source,
            ext.auth_file.clone(),
            ext.oauth_prompt_handler.clone(),
            ext.manual_code_handler.clone(),
        );
        Ok(AnthropicExt {
            authenticator,
            oauth_request_hook: ext.oauth_request_hook.clone(),
        })
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
        let key = crate::client::required_env_var("ANTHROPIC_API_KEY")?;

        Self::builder().api_key(key).build().map_err(Into::into)
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
impl<H> client::ClientBuilder<AnthropicBuilder, crate::markers::Missing, H> {
    /// Select Anthropic Claude subscription OAuth managed by rig.
    pub fn oauth(self) -> client::ClientBuilder<AnthropicBuilder, AnthropicKey, H> {
        self.api_key(AnthropicKey::OAuth)
    }
}

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

    /// Register a callback for the Anthropic OAuth authorization URL.
    pub fn on_oauth_prompt<F>(self, handler: F) -> Self
    where
        F: Fn(auth::OAuthPrompt) + Send + Sync + 'static,
    {
        self.over_ext(|mut ext| {
            ext.oauth_prompt_handler = auth::OAuthPromptHandler::new(handler);
            ext
        })
    }

    /// Register a callback that customizes Anthropic OAuth requests before they are sent.
    pub fn on_oauth_request<F>(self, hook: F) -> Self
    where
        F: Fn(&mut serde_json::Value, &mut http::HeaderMap, &str) -> super::completion::ToolNameMap
            + crate::wasm_compat::WasmCompatSend
            + crate::wasm_compat::WasmCompatSync
            + 'static,
    {
        self.over_ext(|mut ext| {
            ext.oauth_request_hook = Some(std::sync::Arc::new(hook));
            ext
        })
    }

    /// Register a callback that returns a manually pasted OAuth redirect URL or `code#state`.
    pub fn on_manual_code<F>(self, handler: F) -> Self
    where
        F: Fn() -> Option<String> + Send + Sync + 'static,
    {
        self.over_ext(|mut ext| {
            ext.manual_code_handler = auth::ManualCodeHandler::new(handler);
            ext
        })
    }

    /// Store Anthropic OAuth tokens in the provided file.
    pub fn auth_file(self, path: impl AsRef<Path>) -> Self {
        let auth_file = path.as_ref().to_path_buf();
        self.over_ext(|mut ext| {
            ext.auth_file = Some(auth_file);
            ext
        })
    }
}

fn default_auth_file() -> Option<PathBuf> {
    std::env::home_dir().map(|home| home.join(".rig").join("anthropic-auth.json"))
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
    #[test]
    fn test_client_initialization() {
        let _client =
            crate::providers::anthropic::Client::new("dummy-key").expect("Client::new() failed");
        let _client_from_builder = crate::providers::anthropic::Client::builder()
            .api_key("dummy-key")
            .build()
            .expect("Client::builder() failed");
    }
}
