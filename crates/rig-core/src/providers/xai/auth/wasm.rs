use super::AuthError;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub(super) struct PlatformAuthenticator {
    _auth_file: Option<PathBuf>,
}

impl PlatformAuthenticator {
    pub(super) fn new(auth_file: Option<PathBuf>) -> Self {
        Self {
            _auth_file: auth_file,
        }
    }

    pub(super) async fn access_token_oauth(&self) -> Result<String, AuthError> {
        Err(AuthError::Message(
            "xAI OAuth token-cache auth is not supported on wasm targets".into(),
        ))
    }
}
