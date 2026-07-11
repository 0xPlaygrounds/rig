//! Shared xAI OAuth token-cache authentication.

use std::fmt;
use std::path::PathBuf;

#[cfg(not(target_family = "wasm"))]
mod native;
#[cfg(target_family = "wasm")]
mod wasm;

#[cfg(not(target_family = "wasm"))]
use native as platform;
#[cfg(target_family = "wasm")]
use wasm as platform;

#[derive(Clone)]
pub struct Authenticator {
    platform: platform::PlatformAuthenticator,
}

impl fmt::Debug for Authenticator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Authenticator")
            .field("platform", &self.platform)
            .finish()
    }
}

pub use crate::providers::internal::auth::AuthError;

impl Authenticator {
    pub fn new(auth_file: Option<PathBuf>) -> Self {
        Self {
            platform: platform::PlatformAuthenticator::new(auth_file),
        }
    }

    pub async fn access_token(&self) -> Result<Option<String>, AuthError> {
        self.platform.access_token_oauth().await.map(Some)
    }
}

pub(super) fn into_http_error(error: AuthError) -> crate::http_client::Error {
    crate::http_client::Error::Instance(Box::new(error))
}
