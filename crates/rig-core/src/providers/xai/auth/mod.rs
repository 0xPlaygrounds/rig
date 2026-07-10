//! Shared xAI OAuth token-cache authentication.

use std::fmt;
use std::path::PathBuf;
use std::sync::OnceLock;
use tokio::sync::Mutex;

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
    ApiKey,
    OAuth,
}

impl fmt::Debug for AuthSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ApiKey => f.write_str("ApiKey"),
            Self::OAuth => f.write_str("OAuth"),
        }
    }
}

#[derive(Clone)]
pub struct Authenticator {
    source: AuthSource,
    platform: platform::PlatformAuthenticator,
}

impl fmt::Debug for Authenticator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Authenticator")
            .field("source", &self.source)
            .field("platform", &self.platform)
            .finish()
    }
}

pub use crate::providers::internal::auth::AuthError;

impl Authenticator {
    pub fn new(source: AuthSource, auth_file: Option<PathBuf>) -> Self {
        Self {
            source,
            platform: platform::PlatformAuthenticator::new(auth_file),
        }
    }

    pub async fn access_token(&self) -> Result<Option<String>, AuthError> {
        match self.source {
            AuthSource::ApiKey => Ok(None),
            AuthSource::OAuth => {
                // xAI refresh tokens can rotate. Serialize refreshes across
                // independently-built clients, not only clones of one client,
                // so a stale concurrent refresh cannot overwrite fresh state.
                let _guard = refresh_lock().lock().await;
                self.platform.access_token_oauth().await.map(Some)
            }
        }
    }
}

fn refresh_lock() -> &'static Mutex<()> {
    static REFRESH_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    REFRESH_LOCK.get_or_init(|| Mutex::new(()))
}

pub(super) fn into_http_error(error: AuthError) -> crate::http_client::Error {
    crate::http_client::Error::Instance(Box::new(error))
}
