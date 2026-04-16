#[cfg(not(target_family = "wasm"))]
use base64::Engine;
#[cfg(not(target_family = "wasm"))]
use base64::prelude::BASE64_URL_SAFE_NO_PAD;
#[cfg(not(target_family = "wasm"))]
use serde::{Deserialize, Deserializer, Serialize};
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

#[cfg(not(target_family = "wasm"))]
const CHATGPT_AUTH_BASE: &str = "https://auth.openai.com";
#[cfg(not(target_family = "wasm"))]
const CHATGPT_DEVICE_CODE_URL: &str = "https://auth.openai.com/api/accounts/deviceauth/usercode";
#[cfg(not(target_family = "wasm"))]
const CHATGPT_DEVICE_TOKEN_URL: &str = "https://auth.openai.com/api/accounts/deviceauth/token";
#[cfg(not(target_family = "wasm"))]
const CHATGPT_OAUTH_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
#[cfg(not(target_family = "wasm"))]
const CHATGPT_DEVICE_VERIFY_URL: &str = "https://auth.openai.com/codex/device";
#[cfg(not(target_family = "wasm"))]
const CHATGPT_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
#[cfg(not(target_family = "wasm"))]
const TOKEN_EXPIRY_SKEW_SECONDS: i64 = 60;
#[cfg(not(target_family = "wasm"))]
const DEVICE_CODE_TIMEOUT_SECONDS: i64 = 15 * 60;
#[cfg(not(target_family = "wasm"))]
const DEVICE_CODE_POLL_SLEEP_SECONDS: u64 = 5;

#[derive(Debug, Clone)]
pub struct DeviceCodePrompt {
    pub verification_uri: String,
    pub user_code: String,
}

#[derive(Clone, Default)]
pub struct DeviceCodeHandler(Option<Arc<dyn Fn(DeviceCodePrompt) + Send + Sync>>);

impl DeviceCodeHandler {
    pub fn new<F>(handler: F) -> Self
    where
        F: Fn(DeviceCodePrompt) + Send + Sync + 'static,
    {
        Self(Some(Arc::new(handler)))
    }

    #[cfg(not(target_family = "wasm"))]
    fn emit(&self, prompt: DeviceCodePrompt) {
        if let Some(handler) = &self.0 {
            handler(prompt);
        } else {
            println!(
                "Sign in with ChatGPT:\n1) Visit {}\n2) Enter code: {}\nDo not share this device code.",
                prompt.verification_uri, prompt.user_code
            );
        }
    }
}

impl fmt::Debug for DeviceCodeHandler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_some() {
            f.write_str("DeviceCodeHandler(<callback>)")
        } else {
            f.write_str("DeviceCodeHandler(None)")
        }
    }
}

#[derive(Clone)]
pub enum AuthSource {
    AccessToken {
        access_token: String,
        account_id: Option<String>,
    },
    OAuth,
}

#[derive(Clone)]
pub struct Authenticator {
    source: AuthSource,
    auth_file: Option<PathBuf>,
    device_code_handler: DeviceCodeHandler,
    state_lock: Arc<Mutex<()>>,
}

impl fmt::Debug for Authenticator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Authenticator")
            .field("source", &self.source)
            .field("auth_file", &self.auth_file)
            .field("device_code_handler", &self.device_code_handler)
            .finish()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    #[error("{0}")]
    Message(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Http(#[from] reqwest::Error),
}

#[derive(Debug, Clone)]
pub struct AuthContext {
    pub access_token: String,
    pub account_id: Option<String>,
}

#[cfg(not(target_family = "wasm"))]
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
struct AuthRecord {
    access_token: Option<String>,
    refresh_token: Option<String>,
    id_token: Option<String>,
    expires_at: Option<i64>,
    account_id: Option<String>,
}

#[cfg(not(target_family = "wasm"))]
#[derive(Debug, Deserialize)]
struct DeviceCodeResponse {
    device_auth_id: String,
    #[serde(alias = "usercode")]
    user_code: String,
    #[serde(default, deserialize_with = "deserialize_optional_u64")]
    interval: Option<u64>,
}

#[cfg(not(target_family = "wasm"))]
#[derive(Debug, Deserialize)]
struct DeviceTokenResponse {
    authorization_code: String,
    code_verifier: String,
}

#[cfg(not(target_family = "wasm"))]
#[derive(Debug, Deserialize)]
struct OAuthTokenResponse {
    access_token: String,
    refresh_token: Option<String>,
    id_token: Option<String>,
}

impl Authenticator {
    pub fn new(
        source: AuthSource,
        auth_file: Option<PathBuf>,
        device_code_handler: DeviceCodeHandler,
    ) -> Self {
        Self {
            source,
            auth_file,
            device_code_handler,
            state_lock: Arc::new(Mutex::new(())),
        }
    }

    pub async fn auth_context(&self) -> Result<AuthContext, AuthError> {
        match &self.source {
            AuthSource::AccessToken {
                access_token,
                account_id,
            } => Ok(AuthContext {
                access_token: access_token.clone(),
                account_id: account_id.clone(),
            }),
            AuthSource::OAuth => {
                let _guard = self.state_lock.lock().await;
                self.auth_context_locked().await
            }
        }
    }

    async fn auth_context_locked(&self) -> Result<AuthContext, AuthError> {
        #[cfg(target_family = "wasm")]
        {
            Err(AuthError::Message(
                "ChatGPT OAuth is not supported on wasm targets".into(),
            ))
        }

        #[cfg(not(target_family = "wasm"))]
        {
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

            if let Some(refresh_token) = record.refresh_token.clone()
                && let Ok(refreshed) = self.refresh_tokens(&refresh_token).await
            {
                self.write_auth_record(&refreshed)?;
                return Ok(AuthContext {
                    access_token: refreshed.access_token.unwrap_or_default(),
                    account_id: refreshed.account_id,
                });
            }

            let fresh = self.login_device_flow().await?;
            self.write_auth_record(&fresh)?;
            Ok(AuthContext {
                access_token: fresh.access_token.unwrap_or_default(),
                account_id: fresh.account_id,
            })
        }
    }

    #[cfg(not(target_family = "wasm"))]
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

    #[cfg(not(target_family = "wasm"))]
    fn write_auth_record(&self, record: &AuthRecord) -> Result<(), AuthError> {
        let Some(path) = &self.auth_file else {
            return Ok(());
        };

        ensure_parent_dir(path)?;
        std::fs::write(path, serde_json::to_vec_pretty(record)?)?;
        Ok(())
    }

    #[cfg(not(target_family = "wasm"))]
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

        self.device_code_handler.emit(DeviceCodePrompt {
            verification_uri: CHATGPT_DEVICE_VERIFY_URL.to_string(),
            user_code: device.user_code.clone(),
        });

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

        Ok(build_auth_record(tokens))
    }

    #[cfg(not(target_family = "wasm"))]
    async fn refresh_tokens(&self, refresh_token: &str) -> Result<AuthRecord, AuthError> {
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

        Ok(build_auth_record(tokens))
    }
}

impl fmt::Debug for AuthSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AccessToken { .. } => f.write_str("AccessToken(<redacted>)"),
            Self::OAuth => f.write_str("OAuth"),
        }
    }
}

#[cfg(not(target_family = "wasm"))]
fn ensure_parent_dir(path: &std::path::Path) -> Result<(), std::io::Error> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}

#[cfg(not(target_family = "wasm"))]
fn build_auth_record(tokens: OAuthTokenResponse) -> AuthRecord {
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
        refresh_token: tokens.refresh_token,
        id_token,
    }
}

#[cfg(not(target_family = "wasm"))]
fn extract_expiration_timestamp(token: &str) -> Option<i64> {
    decode_jwt_claims(token)
        .get("exp")
        .and_then(|value| value.as_i64().or_else(|| value.as_u64().map(|v| v as i64)))
}

#[cfg(not(target_family = "wasm"))]
fn extract_account_id(token: Option<&str>) -> Option<String> {
    let claims = decode_jwt_claims(token?);
    claims
        .get("https://api.openai.com/auth")
        .and_then(|value| value.as_object())
        .and_then(|map| map.get("chatgpt_account_id"))
        .and_then(|value| value.as_str())
        .map(ToOwned::to_owned)
}

#[cfg(not(target_family = "wasm"))]
fn decode_jwt_claims(token: &str) -> serde_json::Value {
    let payload = token.split('.').nth(1).unwrap_or_default();
    let decoded = BASE64_URL_SAFE_NO_PAD.decode(payload.as_bytes());
    decoded
        .ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .unwrap_or(serde_json::Value::Null)
}

#[cfg(not(target_family = "wasm"))]
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

#[cfg(not(target_family = "wasm"))]
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
    use super::DeviceCodeResponse;

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
}
