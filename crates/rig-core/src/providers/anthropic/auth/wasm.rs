//! Wasm Anthropic OAuth placeholder.

use super::{AuthContext, AuthError, ManualCodeHandler, OAuthPromptHandler};
use crate::http_client::HttpClientExt;
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

    pub(super) async fn auth_context_oauth<H>(
        &self,
        _http_client: &H,
    ) -> Result<AuthContext, AuthError>
    where
        H: HttpClientExt,
    {
        Err(AuthError::Message(
            "Anthropic OAuth is not supported on wasm targets".into(),
        ))
    }
}
