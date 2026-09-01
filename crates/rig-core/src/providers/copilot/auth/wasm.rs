use super::{AuthContext, AuthError, DeviceCodeHandler};
use crate::http_client::HttpClientExt;
use crate::providers::internal::auth::{request, send_json};
use http::Method;
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
        _allow_device_flow: bool,
    ) -> Self {
        Self
    }

    pub(super) async fn auth_context_oauth<H>(&self, _http: &H) -> Result<AuthContext, AuthError>
    where
        H: HttpClientExt,
    {
        Err(AuthError::Message(
            "GitHub Copilot OAuth is not supported on wasm targets".into(),
        ))
    }

    pub(super) async fn auth_context_with_github_access_token<H>(
        &self,
        http: &H,
        access_token: &str,
    ) -> Result<AuthContext, AuthError>
    where
        H: HttpClientExt,
    {
        let response: ApiKeyRecord = send_json(
            http,
            request(Method::GET, GITHUB_API_KEY_URL)
                .header(http::header::ACCEPT, "application/json")
                .header("editor-version", super::super::EDITOR_VERSION)
                .header("editor-plugin-version", super::super::EDITOR_PLUGIN_VERSION)
                .header("user-agent", super::super::USER_AGENT)
                .header(http::header::AUTHORIZATION, format!("token {access_token}"))
                .body(bytes::Bytes::new()),
        )
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
