use crate::http_client::HttpClientExt;
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
    /// The platform half owns the token/key caches (files plus their parsed
    /// state); serializing access to it — rather than to a detached unit
    /// lock — is what prevents concurrent refreshes from racing the cache.
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
    pub api_key: String,
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

    /// Resolve the API key (and optional API base), refreshing or signing in
    /// through `http` — the client's own transport — when the cache is stale.
    pub async fn auth_context<H>(&self, http: &H) -> Result<AuthContext, AuthError>
    where
        H: HttpClientExt,
    {
        match &self.source {
            AuthSource::ApiKey(api_key) => Ok(AuthContext {
                api_key: api_key.clone(),
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
