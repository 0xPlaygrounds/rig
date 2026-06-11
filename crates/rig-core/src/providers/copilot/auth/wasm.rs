use super::{AuthContext, AuthError, DeviceCodeHandler};
use crate::http_client::{self, HttpClientExt};
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
            "GitHub Copilot OAuth is not supported on wasm targets".into(),
        ))
    }

    pub(super) async fn auth_context_with_github_access_token<H>(
        &self,
        http_client: &H,
        access_token: &str,
    ) -> Result<AuthContext, AuthError>
    where
        H: HttpClientExt,
    {
        let authorization = format!("token {access_token}");
        let req = http::Request::builder()
            .method(Method::GET)
            .uri(GITHUB_API_KEY_URL)
            .header(http::header::ACCEPT, "application/json")
            .header("editor-version", super::super::EDITOR_VERSION)
            .header("editor-plugin-version", super::super::EDITOR_PLUGIN_VERSION)
            .header("user-agent", super::super::USER_AGENT)
            .header(http::header::AUTHORIZATION, authorization)
            .body(Vec::new())
            .map_err(http_client::Error::Protocol)?;
        let response = http_client.send::<_, Vec<u8>>(req).await?;
        let body = response.into_body().await?;
        let response = serde_json::from_slice::<ApiKeyRecord>(&body).map_err(|error| {
            AuthError::Message(format!(
                "GitHub Copilot auth response could not be parsed: {error}; body: {}",
                String::from_utf8_lossy(&body)
            ))
        })?;

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
