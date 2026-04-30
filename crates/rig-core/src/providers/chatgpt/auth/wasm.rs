//! WASM ChatGPT auth implementation.

use super::{AuthContext, AuthError, DeviceCodeHandler};
use std::path::PathBuf;

#[derive(Debug, Clone, Default)]
pub(super) struct PlatformAuthenticator;

impl PlatformAuthenticator {
    pub(super) fn new(
        _auth_file: Option<PathBuf>,
        _device_code_handler: DeviceCodeHandler,
    ) -> Self {
        Self
    }

    pub(super) async fn auth_context_oauth(&self) -> Result<AuthContext, AuthError> {
        Err(AuthError::Message(
            "ChatGPT OAuth is not supported on wasm targets".into(),
        ))
    }
}
