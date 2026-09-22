//! ChatGPT access-token configuration and native OAuth authentication.
//!
//! ```no_run
//! use rig_core::providers::chatgpt::auth::{AuthSource, Authenticator, DeviceCodeHandler};
//!
//! let auth = Authenticator::new(AuthSource::OAuth, None, DeviceCodeHandler::default(), true);
//! ```

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

#[derive(Clone)]
pub enum AuthSource {
    AccessToken {
        access_token: String,
        account_id: Option<String>,
    },
    OAuth,
}

impl fmt::Debug for AuthSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AccessToken { .. } => f.write_str("AccessToken(<redacted>)"),
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
    pub access_token: Secret,
    pub account_id: Option<String>,
}

impl Authenticator {
    pub fn new(
        source: AuthSource,
        auth_file: Option<PathBuf>,
        device_code_handler: DeviceCodeHandler,
        allow_device_flow: bool,
    ) -> Self {
        Self {
            source,
            platform: Arc::new(Mutex::new(platform::PlatformAuthenticator::new(
                auth_file,
                device_code_handler,
                allow_device_flow,
            ))),
        }
    }

    /// Resolve the access token and account id, refreshing through `http` as needed.
    /// Return cache, transport, or authorization errors. OAuth is unsupported
    /// on WASM; explicit access tokens remain available.
    pub async fn auth_context<H>(&self, http: &H) -> Result<AuthContext, AuthError>
    where
        H: HttpClientExt,
    {
        match &self.source {
            AuthSource::AccessToken {
                access_token,
                account_id,
            } => Ok(AuthContext {
                access_token: access_token.clone().into(),
                account_id: account_id.clone(),
            }),
            AuthSource::OAuth => self.platform.lock().await.auth_context_oauth(http).await,
        }
    }
}
