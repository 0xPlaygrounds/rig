use crate::http_client::HttpClientExt;
use crate::wire::Secret;
use futures::lock::Mutex;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

pub use crate::providers::internal::auth::{DeviceCodeHandler, DeviceCodePrompt};

#[cfg(not(target_family = "wasm"))]
mod native;
#[cfg(target_family = "wasm")]
mod wasm;

#[cfg(not(target_family = "wasm"))]
use native as platform;
#[cfg(target_family = "wasm")]
use wasm as platform;

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

#[derive(Clone)]
pub struct Authenticator {
    source: AuthSource,
    /// Shared cache access, locked across refresh to prevent concurrent updates.
    platform: Arc<Mutex<platform::PlatformAuthenticator>>,
}

impl fmt::Debug for Authenticator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Authenticator")
            .field("source", &self.source)
            .field("platform", &"<serialized>")
            .finish()
    }
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
            platform: Arc::new(Mutex::new(platform::PlatformAuthenticator::new(
                access_token_file,
                api_key_file,
                device_code_handler,
                allow_device_flow,
            ))),
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
            AuthSource::GitHubAccessToken(access_token) => {
                self.platform
                    .lock()
                    .await
                    .auth_context_with_github_access_token(http, access_token)
                    .await
            }
            AuthSource::OAuth => self.platform.lock().await.auth_context_oauth(http).await,
        }
    }
}
