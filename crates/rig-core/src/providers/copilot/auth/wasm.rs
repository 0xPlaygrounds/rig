use super::{AuthContext, AuthError, DeviceCodeHandler};
use serde::Deserialize;
use std::path::PathBuf;

const GITHUB_API_KEY_URL: &str = "https://api.github.com/copilot_internal/v2/token";

#[derive(Debug, Clone, Default)]
pub(super) struct PlatformAuthenticator;

#[derive(Debug, Deserialize)]
struct ApiKeyRecord {
    token: Option<String>,
    endpoints: Option<ApiKeyEndpoints>,
}

#[derive(Debug, Deserialize)]
struct ApiKeyEndpoints {
    api: Option<String>,
}

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
        access_token: &str,
    ) -> Result<AuthContext, AuthError> {
        let response = reqwest::Client::new()
            .get(GITHUB_API_KEY_URL)
            .header(reqwest::header::ACCEPT, "application/json")
            .header("editor-version", "vscode/1.95.0")
            .header("editor-plugin-version", "copilot-chat/0.26.7")
            .header("user-agent", "GitHubCopilotChat/0.26.7")
            .header(
                reqwest::header::AUTHORIZATION,
                format!("token {access_token}"),
            )
            .send()
            .await?
            .error_for_status()?
            .json::<ApiKeyRecord>()
            .await?;

        let Some(api_key) = response.token.filter(|token| !token.trim().is_empty()) else {
            return Err(AuthError::Message(
                "GitHub Copilot API key response did not include a token".into(),
            ));
        };

        Ok(AuthContext {
            api_key,
            api_base: response.endpoints.and_then(|endpoints| endpoints.api),
        })
    }
}
