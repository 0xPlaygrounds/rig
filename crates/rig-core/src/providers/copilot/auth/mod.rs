use crate::http_client::HttpClientExt;
use crate::providers::internal::auth::{request, send_json};
use crate::wire::Secret;
use futures::lock::Mutex;
use http::Method;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

pub use crate::providers::internal::auth::{DeviceCodeHandler, DeviceCodePrompt};

#[cfg(not(target_family = "wasm"))]
mod native;

const GITHUB_API_KEY_URL: &str = "https://api.github.com/copilot_internal/v2/token";

/// Return `{config_dir}/github_copilot`, or `None` without a platform config directory.
/// The directory conventionally contains `access-token` and `api-key.json`.
pub fn default_token_dir() -> Option<PathBuf> {
    crate::providers::internal::auth::config_dir().map(|dir| dir.join("github_copilot"))
}

#[derive(Clone)]
pub enum AuthSource {
    ApiKey(String),
    GitHubAccessToken(String),
    OAuth,
}

impl fmt::Debug for AuthSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ApiKey(_) => f.write_str("ApiKey(<redacted>)"),
            Self::GitHubAccessToken(_) => f.write_str("GitHubAccessToken(<redacted>)"),
            Self::OAuth => f.write_str("OAuth"),
        }
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(target_family = "wasm", allow(dead_code))]
pub struct Authenticator {
    source: AuthSource,
    access_token_file: Option<PathBuf>,
    api_key_file: Option<PathBuf>,
    device_code_handler: DeviceCodeHandler,
    allow_device_flow: bool,
    /// Held across a cache refresh to prevent concurrent updates.
    refresh_lock: Arc<Mutex<()>>,
}

pub use crate::providers::internal::auth::AuthError;

#[derive(Debug, Clone)]
pub struct AuthContext {
    /// Resolved credential. Use [`Secret::expose`] only when raw bytes are required.
    pub api_key: Secret,
    pub api_base: Option<String>,
}

impl Authenticator {
    pub fn new(
        source: AuthSource,
        access_token_file: Option<PathBuf>,
        api_key_file: Option<PathBuf>,
        device_code_handler: DeviceCodeHandler,
        allow_device_flow: bool,
    ) -> Self {
        Self {
            source,
            access_token_file,
            api_key_file,
            device_code_handler,
            allow_device_flow,
            refresh_lock: Arc::default(),
        }
    }

    /// Resolve the API key and optional API base, refreshing through `http` as needed.
    /// Return cache, transport, or authorization errors. OAuth device login is
    /// unsupported on WASM; access-token exchange remains available.
    pub async fn auth_context<H>(&self, http: &H) -> Result<AuthContext, AuthError>
    where
        H: HttpClientExt,
    {
        match &self.source {
            AuthSource::ApiKey(api_key) => Ok(AuthContext {
                api_key: api_key.clone().into(),
                api_base: None,
            }),
            #[cfg(not(target_family = "wasm"))]
            AuthSource::GitHubAccessToken(access_token) => {
                self.auth_context_with_github_access_token(http, access_token)
                    .await
            }
            #[cfg(target_family = "wasm")]
            AuthSource::GitHubAccessToken(access_token) => {
                Ok(refresh_api_key(http, access_token).await?.into_context())
            }
            #[cfg(not(target_family = "wasm"))]
            AuthSource::OAuth => self.auth_context_oauth(http).await,
            #[cfg(target_family = "wasm")]
            AuthSource::OAuth => Err(AuthError::Message(
                "GitHub Copilot OAuth is not supported on wasm targets".into(),
            )),
        }
    }
}

/// Copilot API key exchanged for a GitHub access token, and its native cache record.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
struct ApiKeyRecord {
    token: Option<String>,
    expires_at: Option<i64>,
    endpoints: Option<ApiKeyEndpoints>,
    bootstrap_token_fingerprint: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
struct ApiKeyEndpoints {
    api: Option<String>,
}

impl ApiKeyRecord {
    fn api_base(&self) -> Option<String> {
        self.endpoints
            .as_ref()
            .and_then(|endpoints| endpoints.api.as_ref())
            .cloned()
    }

    fn into_context(self) -> AuthContext {
        AuthContext {
            api_base: self.api_base(),
            api_key: self.token.unwrap_or_default().into(),
        }
    }
}

/// Exchange a GitHub access token for a Copilot API key.
/// Return transport errors, or an error when the response has no non-blank token.
async fn refresh_api_key<H>(http: &H, access_token: &str) -> Result<ApiKeyRecord, AuthError>
where
    H: HttpClientExt,
{
    let response: ApiKeyRecord = send_json(
        http,
        request(Method::GET, GITHUB_API_KEY_URL)
            .header(http::header::ACCEPT, "application/json")
            .header("editor-version", super::EDITOR_VERSION)
            .header("editor-plugin-version", super::EDITOR_PLUGIN_VERSION)
            .header("user-agent", super::USER_AGENT)
            .header(http::header::AUTHORIZATION, format!("token {access_token}"))
            .body(bytes::Bytes::new()),
    )
    .await?;

    if response
        .token
        .as_ref()
        .is_none_or(|token| token.trim().is_empty())
    {
        return Err(AuthError::Message(
            "GitHub Copilot API key response did not include a token".into(),
        ));
    }

    Ok(response)
}
