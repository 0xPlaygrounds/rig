//! Native ChatGPT OAuth and token cache implementation.

use super::{AuthContext, AuthError, DeviceCodeHandler, DeviceCodePrompt};
use base64::Engine;
use base64::prelude::BASE64_URL_SAFE_NO_PAD;
use serde::{Deserialize, Deserializer, Serialize};
use std::path::{Path, PathBuf};

const CHATGPT_AUTH_BASE: &str = "https://auth.openai.com";
const CHATGPT_DEVICE_CODE_URL: &str = "https://auth.openai.com/api/accounts/deviceauth/usercode";
const CHATGPT_DEVICE_TOKEN_URL: &str = "https://auth.openai.com/api/accounts/deviceauth/token";
const CHATGPT_OAUTH_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
const CHATGPT_DEVICE_VERIFY_URL: &str = "https://auth.openai.com/codex/device";
const CHATGPT_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const TOKEN_EXPIRY_SKEW_SECONDS: i64 = 60;
const DEVICE_CODE_TIMEOUT_SECONDS: i64 = 15 * 60;
const DEVICE_CODE_POLL_SLEEP_SECONDS: u64 = 5;

#[derive(Debug, Clone)]
pub(super) struct PlatformAuthenticator {
    auth_file: Option<PathBuf>,
    device_code_handler: DeviceCodeHandler,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
struct AuthRecord {
    access_token: Option<String>,
    refresh_token: Option<String>,
    id_token: Option<String>,
    expires_at: Option<i64>,
    account_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DeviceCodeResponse {
    device_auth_id: String,
    #[serde(alias = "usercode")]
    user_code: String,
    #[serde(default, deserialize_with = "deserialize_optional_u64")]
    interval: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct DeviceTokenResponse {
    authorization_code: String,
    code_verifier: String,
}

#[derive(Debug, Deserialize)]
struct OAuthTokenResponse {
    access_token: String,
    refresh_token: Option<String>,
    id_token: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OAuthErrorResponse {
    error: Option<String>,
    error_description: Option<String>,
}

enum RefreshTokensError {
    Reauthenticate,
    Auth(AuthError),
}

impl PlatformAuthenticator {
    pub(super) fn new(auth_file: Option<PathBuf>, device_code_handler: DeviceCodeHandler) -> Self {
        Self {
            auth_file,
            device_code_handler,
        }
    }

    pub(super) async fn auth_context_oauth(&self) -> Result<AuthContext, AuthError> {
        let mut record = self.read_auth_record()?;

        if let Some(access_token) = record.access_token.clone()
            && !token_expired(record.expires_at)
        {
            let account_id = record
                .account_id
                .clone()
                .or_else(|| extract_account_id(record.id_token.as_deref()))
                .or_else(|| extract_account_id(Some(&access_token)));
            if account_id != record.account_id {
                record.account_id = account_id.clone();
                self.write_auth_record(&record)?;
            }
            return Ok(AuthContext {
                access_token,
                account_id,
            });
        }

        if let Some(refresh_token) = record.refresh_token.clone() {
            match self.refresh_tokens(&refresh_token).await {
                Ok(refreshed) => {
                    self.write_auth_record(&refreshed)?;
                    return Ok(AuthContext {
                        access_token: refreshed.access_token.unwrap_or_default(),
                        account_id: refreshed.account_id,
                    });
                }
                Err(RefreshTokensError::Reauthenticate) => {}
                Err(RefreshTokensError::Auth(err)) => return Err(err),
            }
        }

        let fresh = self.login_device_flow().await?;
        self.write_auth_record(&fresh)?;
        Ok(AuthContext {
            access_token: fresh.access_token.unwrap_or_default(),
            account_id: fresh.account_id,
        })
    }

    fn read_auth_record(&self) -> Result<AuthRecord, AuthError> {
        let Some(path) = &self.auth_file else {
            return Ok(AuthRecord::default());
        };

        match std::fs::read(path) {
            Ok(bytes) => Ok(serde_json::from_slice(&bytes)?),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(AuthRecord::default()),
            Err(err) => Err(err.into()),
        }
    }

    fn write_auth_record(&self, record: &AuthRecord) -> Result<(), AuthError> {
        let Some(path) = &self.auth_file else {
            return Ok(());
        };

        ensure_parent_dir(path)?;
        std::fs::write(path, serde_json::to_vec_pretty(record)?)?;
        Ok(())
    }

    async fn login_device_flow(&self) -> Result<AuthRecord, AuthError> {
        let client = reqwest::Client::new();
        let device = client
            .post(CHATGPT_DEVICE_CODE_URL)
            .json(&serde_json::json!({ "client_id": CHATGPT_CLIENT_ID }))
            .send()
            .await?
            .error_for_status()?
            .json::<DeviceCodeResponse>()
            .await?;

        emit_device_code_prompt(
            &self.device_code_handler,
            DeviceCodePrompt {
                verification_uri: CHATGPT_DEVICE_VERIFY_URL.to_string(),
                user_code: device.user_code.clone(),
            },
        );

        let interval = device.interval.unwrap_or(DEVICE_CODE_POLL_SLEEP_SECONDS);
        let start = std::time::Instant::now();
        let code = loop {
            if start.elapsed().as_secs() as i64 >= DEVICE_CODE_TIMEOUT_SECONDS {
                return Err(AuthError::Message(
                    "Timed out waiting for ChatGPT device authorization".into(),
                ));
            }

            let response = client
                .post(CHATGPT_DEVICE_TOKEN_URL)
                .json(&serde_json::json!({
                    "device_auth_id": device.device_auth_id,
                    "user_code": device.user_code,
                }))
                .send()
                .await?;

            if response.status().is_success() {
                let token_response = response.json::<DeviceTokenResponse>().await?;
                break token_response;
            }

            let status = response.status();
            if status.as_u16() == 403 || status.as_u16() == 404 {
                tokio::time::sleep(std::time::Duration::from_secs(interval)).await;
                continue;
            }

            let text = response.text().await.unwrap_or_default();
            return Err(AuthError::Message(format!(
                "ChatGPT device authorization failed: {status} {text}"
            )));
        };

        let redirect_uri = format!("{CHATGPT_AUTH_BASE}/deviceauth/callback");
        let form = [
            ("grant_type", "authorization_code"),
            ("code", code.authorization_code.as_str()),
            ("redirect_uri", redirect_uri.as_str()),
            ("client_id", CHATGPT_CLIENT_ID),
            ("code_verifier", code.code_verifier.as_str()),
        ];
        let body = url::form_urlencoded::Serializer::new(String::new())
            .extend_pairs(form)
            .finish();

        let tokens = client
            .post(CHATGPT_OAUTH_TOKEN_URL)
            .header(
                reqwest::header::CONTENT_TYPE,
                "application/x-www-form-urlencoded",
            )
            .body(body)
            .send()
            .await?
            .error_for_status()?
            .json::<OAuthTokenResponse>()
            .await?;

        Ok(build_auth_record(tokens, None))
    }

    async fn refresh_tokens(&self, refresh_token: &str) -> Result<AuthRecord, RefreshTokensError> {
        let client = reqwest::Client::new();
        let form = [
            ("client_id", CHATGPT_CLIENT_ID),
            ("grant_type", "refresh_token"),
            ("refresh_token", refresh_token),
            ("scope", "openid profile email"),
        ];

        let body = url::form_urlencoded::Serializer::new(String::new())
            .extend_pairs(form)
            .finish();

        let response = client
            .post(CHATGPT_OAUTH_TOKEN_URL)
            .header(
                reqwest::header::CONTENT_TYPE,
                "application/x-www-form-urlencoded",
            )
            .body(body)
            .send()
            .await
            .map_err(AuthError::from)
            .map_err(RefreshTokensError::Auth)?;

        let status = response.status();
        if status.is_success() {
            let tokens = response
                .json::<OAuthTokenResponse>()
                .await
                .map_err(AuthError::from)
                .map_err(RefreshTokensError::Auth)?;
            return Ok(build_auth_record(tokens, Some(refresh_token.to_owned())));
        }

        let body = response.text().await.unwrap_or_default();
        let oauth_error = serde_json::from_str::<OAuthErrorResponse>(&body).ok();
        if should_reauthenticate_after_refresh(
            status,
            oauth_error
                .as_ref()
                .and_then(|error| error.error.as_deref()),
        ) {
            return Err(RefreshTokensError::Reauthenticate);
        }

        Err(RefreshTokensError::Auth(AuthError::Message(
            format_refresh_error(status, oauth_error.as_ref(), &body),
        )))
    }
}

fn emit_device_code_prompt(handler: &DeviceCodeHandler, prompt: DeviceCodePrompt) {
    if let Some(callback) = &handler.0 {
        callback(prompt);
    } else {
        println!(
            "Sign in with ChatGPT:\n1) Visit {}\n2) Enter code: {}\nDo not share this device code.",
            prompt.verification_uri, prompt.user_code
        );
    }
}

fn ensure_parent_dir(path: &Path) -> Result<(), std::io::Error> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn build_auth_record(
    tokens: OAuthTokenResponse,
    previous_refresh_token: Option<String>,
) -> AuthRecord {
    let access_token = Some(tokens.access_token);
    let id_token = tokens.id_token;
    AuthRecord {
        expires_at: access_token
            .as_deref()
            .and_then(extract_expiration_timestamp),
        account_id: extract_account_id(id_token.as_deref()).or_else(|| {
            access_token
                .as_deref()
                .and_then(|token| extract_account_id(Some(token)))
        }),
        access_token,
        refresh_token: tokens.refresh_token.or(previous_refresh_token),
        id_token,
    }
}

fn extract_expiration_timestamp(token: &str) -> Option<i64> {
    decode_jwt_claims(token)
        .get("exp")
        .and_then(|value| value.as_i64().or_else(|| value.as_u64().map(|v| v as i64)))
}

fn extract_account_id(token: Option<&str>) -> Option<String> {
    let claims = decode_jwt_claims(token?);
    claims
        .get("https://api.openai.com/auth")
        .and_then(|value| value.as_object())
        .and_then(|map| map.get("chatgpt_account_id"))
        .and_then(|value| value.as_str())
        .map(ToOwned::to_owned)
}

fn decode_jwt_claims(token: &str) -> serde_json::Value {
    let payload = token.split('.').nth(1).unwrap_or_default();
    let decoded = BASE64_URL_SAFE_NO_PAD.decode(payload.as_bytes());
    decoded
        .ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .unwrap_or(serde_json::Value::Null)
}

fn should_reauthenticate_after_refresh(
    status: reqwest::StatusCode,
    error_code: Option<&str>,
) -> bool {
    matches!(
        status,
        reqwest::StatusCode::BAD_REQUEST | reqwest::StatusCode::UNAUTHORIZED
    ) && matches!(error_code, Some("invalid_grant"))
}

fn format_refresh_error(
    status: reqwest::StatusCode,
    oauth_error: Option<&OAuthErrorResponse>,
    body: &str,
) -> String {
    let error_code = oauth_error.and_then(|error| error.error.as_deref());
    let description = oauth_error.and_then(|error| error.error_description.as_deref());

    if let Some(description) = description
        .map(str::trim)
        .filter(|description| !description.is_empty())
    {
        return format!(
            "ChatGPT token refresh failed: {status} {} ({description})",
            error_code.unwrap_or("unknown_error")
        );
    }

    if let Some(error_code) = error_code {
        return format!("ChatGPT token refresh failed: {status} {error_code}");
    }

    if !body.trim().is_empty() {
        return format!("ChatGPT token refresh failed: {status} {body}");
    }

    format!("ChatGPT token refresh failed: {status}")
}

fn token_expired(expires_at: Option<i64>) -> bool {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or_default();

    match expires_at {
        Some(exp) => now >= exp - TOKEN_EXPIRY_SKEW_SECONDS,
        None => true,
    }
}

fn deserialize_optional_u64<'de, D>(deserializer: D) -> Result<Option<u64>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum U64OrString {
        U64(u64),
        String(String),
    }

    let value = Option::<U64OrString>::deserialize(deserializer)?;
    match value {
        None => Ok(None),
        Some(U64OrString::U64(value)) => Ok(Some(value)),
        Some(U64OrString::String(value)) => {
            let value = value.trim();
            if value.is_empty() {
                Ok(None)
            } else {
                value
                    .parse::<u64>()
                    .map(Some)
                    .map_err(serde::de::Error::custom)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DeviceCodeResponse, OAuthErrorResponse, OAuthTokenResponse, build_auth_record,
        format_refresh_error, should_reauthenticate_after_refresh,
    };
    use reqwest::StatusCode;

    #[test]
    fn device_code_response_accepts_numeric_interval() {
        let response: DeviceCodeResponse = serde_json::from_str(
            r#"{
                "device_auth_id": "deviceauth_123",
                "user_code": "ABCD-EFGH",
                "interval": 5
            }"#,
        )
        .expect("device code response");

        assert_eq!(response.interval, Some(5));
    }

    #[test]
    fn device_code_response_accepts_string_interval() {
        let response: DeviceCodeResponse = serde_json::from_str(
            r#"{
                "device_auth_id": "deviceauth_123",
                "user_code": "ABCD-EFGH",
                "interval": "5"
            }"#,
        )
        .expect("device code response");

        assert_eq!(response.interval, Some(5));
    }

    #[test]
    fn refresh_reauth_only_on_invalid_grant() {
        assert!(should_reauthenticate_after_refresh(
            StatusCode::BAD_REQUEST,
            Some("invalid_grant")
        ));
        assert!(should_reauthenticate_after_refresh(
            StatusCode::UNAUTHORIZED,
            Some("invalid_grant")
        ));
        assert!(!should_reauthenticate_after_refresh(
            StatusCode::BAD_GATEWAY,
            Some("invalid_grant")
        ));
        assert!(!should_reauthenticate_after_refresh(
            StatusCode::BAD_REQUEST,
            Some("invalid_request")
        ));
        assert!(!should_reauthenticate_after_refresh(
            StatusCode::UNAUTHORIZED,
            None
        ));
    }

    #[test]
    fn refresh_error_uses_oauth_description_when_present() {
        let oauth_error = OAuthErrorResponse {
            error: Some("temporarily_unavailable".into()),
            error_description: Some("please retry".into()),
        };

        assert_eq!(
            format_refresh_error(StatusCode::BAD_GATEWAY, Some(&oauth_error), ""),
            "ChatGPT token refresh failed: 502 Bad Gateway temporarily_unavailable (please retry)"
        );
    }

    #[test]
    fn build_auth_record_preserves_existing_refresh_token_when_refresh_omits_one() {
        let record = build_auth_record(
            OAuthTokenResponse {
                access_token: "access-token".into(),
                refresh_token: None,
                id_token: None,
            },
            Some("cached-refresh-token".into()),
        );

        assert_eq!(
            record.refresh_token.as_deref(),
            Some("cached-refresh-token")
        );
    }
}
