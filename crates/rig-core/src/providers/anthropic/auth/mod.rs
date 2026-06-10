//! Anthropic OAuth authentication types and target-specific dispatch.

use std::{fmt, path::PathBuf, sync::Arc};
use tokio::sync::Mutex;

#[cfg(not(target_family = "wasm"))]
mod native;
#[cfg(target_family = "wasm")]
mod wasm;

#[cfg(not(target_family = "wasm"))]
use native as platform;
#[cfg(target_family = "wasm")]
use wasm as platform;

/// Prompt emitted when interactive browser OAuth needs user action.
#[derive(Debug, Clone)]
pub struct OAuthPrompt {
    /// Authorization URL the user should open in a browser.
    pub authorization_url: String,
}

/// Callback invoked with the Anthropic OAuth authorization URL.
#[derive(Clone, Default)]
pub struct OAuthPromptHandler(pub(super) Option<Arc<dyn Fn(OAuthPrompt) + Send + Sync>>);

impl OAuthPromptHandler {
    /// Create a new prompt callback.
    pub fn new<F>(handler: F) -> Self
    where
        F: Fn(OAuthPrompt) + Send + Sync + 'static,
    {
        Self(Some(Arc::new(handler)))
    }
}

impl fmt::Debug for OAuthPromptHandler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_some() {
            f.write_str("OAuthPromptHandler(<callback>)")
        } else {
            f.write_str("OAuthPromptHandler(None)")
        }
    }
}

/// Callback invoked when the OAuth redirect cannot be captured automatically.
#[derive(Clone, Default)]
pub struct ManualCodeHandler(pub(super) Option<Arc<dyn Fn() -> Option<String> + Send + Sync>>);

impl ManualCodeHandler {
    /// Create a new manual code callback.
    pub fn new<F>(handler: F) -> Self
    where
        F: Fn() -> Option<String> + Send + Sync + 'static,
    {
        Self(Some(Arc::new(handler)))
    }
}

impl fmt::Debug for ManualCodeHandler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_some() {
            f.write_str("ManualCodeHandler(<callback>)")
        } else {
            f.write_str("ManualCodeHandler(None)")
        }
    }
}

/// Anthropic authentication provenance.
#[derive(Clone)]
pub enum AuthSource {
    /// Static Anthropic API key using `x-api-key`.
    ApiKey(String),
    /// Anthropic Claude subscription OAuth managed by rig.
    OAuth,
}

impl fmt::Debug for AuthSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ApiKey(_) => f.write_str("ApiKey(<redacted>)"),
            Self::OAuth => f.write_str("OAuth"),
        }
    }
}

/// Per-request Anthropic authentication context.
#[derive(Debug, Clone)]
pub struct AuthContext {
    /// Access token or API key to put on the outbound request.
    pub access_token: String,
    /// Source used to produce the token.
    pub source: AuthSource,
}

/// Anthropic authenticator with serialized OAuth refresh/login state.
#[derive(Clone)]
pub struct Authenticator {
    source: AuthSource,
    platform: platform::PlatformAuthenticator,
    state_lock: Arc<Mutex<()>>,
}

impl fmt::Debug for Authenticator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Authenticator")
            .field("source", &self.source)
            .field("platform", &self.platform)
            .finish()
    }
}

/// Anthropic authentication errors.
#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    /// Human-readable authentication failure.
    #[error("{0}")]
    Message(String),
    /// File-system failure while reading or writing the token cache.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// JSON serialization failure.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// HTTP failure during token exchange or refresh.
    #[error(transparent)]
    Http(#[from] reqwest::Error),
}

impl Authenticator {
    /// Create an Anthropic authenticator.
    pub fn new(
        source: AuthSource,
        auth_file: Option<PathBuf>,
        oauth_prompt_handler: OAuthPromptHandler,
        manual_code_handler: ManualCodeHandler,
    ) -> Self {
        Self {
            source,
            platform: platform::PlatformAuthenticator::new(
                auth_file,
                oauth_prompt_handler,
                manual_code_handler,
            ),
            state_lock: Arc::new(Mutex::new(())),
        }
    }

    /// Resolve the current request authentication context, refreshing OAuth tokens if required.
    pub async fn auth_context(&self) -> Result<AuthContext, AuthError> {
        match &self.source {
            AuthSource::ApiKey(key) => {
                if key.is_empty() {
                    return Err(AuthError::Message(
                        "Anthropic API key is empty; configure an API key or use OAuth".into(),
                    ));
                }
                Ok(AuthContext {
                    access_token: key.clone(),
                    source: AuthSource::ApiKey(key.clone()),
                })
            }
            AuthSource::OAuth => {
                let _guard = self.state_lock.lock().await;
                self.platform.auth_context_oauth().await
            }
        }
    }
}
