//! Wasm Anthropic OAuth placeholder.

use super::{AuthContext, AuthError, ManualCodeHandler, OAuthPromptHandler};
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub(super) struct PlatformAuthenticator;

impl PlatformAuthenticator {
    pub(super) fn new(
        _auth_file: Option<PathBuf>,
        _oauth_prompt_handler: OAuthPromptHandler,
        _manual_code_handler: ManualCodeHandler,
    ) -> Self {
        Self
    }

    pub(super) async fn auth_context_oauth(&self) -> Result<AuthContext, AuthError> {
        Err(AuthError::Message(
            "Anthropic OAuth is not supported on wasm targets".into(),
        ))
    }
}
