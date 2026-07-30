//! WASM ChatGPT auth implementation.

use super::{AuthContext, AuthError, DeviceCodePrompter};
use std::path::PathBuf;

#[derive(Debug, Clone, Default)]
pub(super) struct PlatformAuthenticator;

impl PlatformAuthenticator {
    pub(super) fn new(
        _auth_file: Option<PathBuf>,
        _device_code_prompter: DeviceCodePrompter,
        _allow_device_flow: bool,
    ) -> Self {
        Self
    }

    pub(super) async fn auth_context_oauth(&self) -> Result<AuthContext, AuthError> {
        Err(AuthError::Message(
            "ChatGPT OAuth is not supported on wasm targets".into(),
        ))
    }

    /// OAuth is unsupported on wasm, so there is never a cached credential.
    pub(super) fn cached_auth_context(&self) -> Option<AuthContext> {
        None
    }
}
