use super::{AuthContext, AuthError, DeviceCodeHandler};
use std::path::PathBuf;

#[derive(Debug, Clone, Default)]
pub(super) struct PlatformAuthenticator;

impl PlatformAuthenticator {
    pub(super) fn new(
        _access_token_file: Option<PathBuf>,
        _api_key_file: Option<PathBuf>,
        _device_code_handler: DeviceCodeHandler,
    ) -> Self {
        Self
    }

    pub(super) async fn auth_context_oauth(&self) -> Result<AuthContext, AuthError> {
        Err(AuthError::Message(
            "GitHub Copilot OAuth is not supported on wasm targets".into(),
        ))
    }

    pub(super) async fn auth_context_with_github_access_token(
        &self,
        _access_token: &str,
    ) -> Result<AuthContext, AuthError> {
        Err(AuthError::Message(
            "GitHub Copilot OAuth is not supported on wasm targets".into(),
        ))
    }
}
