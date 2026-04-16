use super::{AuthContext, AuthError, DeviceCodeHandler, DeviceCodePrompt};
use serde::{Deserialize, Serialize};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};

const GITHUB_CLIENT_ID: &str = "Iv1.b507a08c87ecfe98";
const GITHUB_DEVICE_CODE_URL: &str = "https://github.com/login/device/code";
const GITHUB_ACCESS_TOKEN_URL: &str = "https://github.com/login/oauth/access_token";
const GITHUB_API_KEY_URL: &str = "https://api.github.com/copilot_internal/v2/token";
const DEVICE_CODE_POLL_SLEEP_SECONDS: u64 = 5;
const DEVICE_CODE_TIMEOUT_SECONDS: u64 = 15 * 60;
const DEVICE_CODE_SLOW_DOWN_SECONDS: u64 = 5;

#[derive(Debug, Clone)]
pub(super) struct PlatformAuthenticator {
    access_token_file: Option<PathBuf>,
    api_key_file: Option<PathBuf>,
    device_code_handler: DeviceCodeHandler,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
struct ApiKeyRecord {
    token: Option<String>,
    expires_at: Option<i64>,
    endpoints: Option<ApiKeyEndpoints>,
    bootstrap_token_fingerprint: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
struct ApiKeyEndpoints {
    api: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DeviceCodeResponse {
    device_code: String,
    user_code: String,
    verification_uri: String,
    interval: Option<u64>,
    expires_in: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct AccessTokenResponse {
    access_token: Option<String>,
    error: Option<String>,
    error_description: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AccessTokenState {
    token: String,
    from_cache: bool,
}

impl PlatformAuthenticator {
    pub(super) fn new(
        access_token_file: Option<PathBuf>,
        api_key_file: Option<PathBuf>,
        device_code_handler: DeviceCodeHandler,
    ) -> Self {
        Self {
            access_token_file,
            api_key_file,
            device_code_handler,
        }
    }

    pub(super) async fn auth_context_oauth(&self) -> Result<AuthContext, AuthError> {
        let record = self.read_api_key_record()?;
        let cached_access_token = self.read_access_token().ok().flatten();
        let api_base = record.api_base();
        if record.can_reuse_for_oauth(cached_access_token.as_deref())
            && let Some(token) = record.token
        {
            return Ok(AuthContext {
                api_key: token,
                api_base,
            });
        }

        let access_token = if let Some(token) = cached_access_token {
            AccessTokenState {
                token,
                from_cache: true,
            }
        } else {
            self.access_token().await?
        };
        let record = match self.refresh_api_key(&access_token.token).await {
            Ok(record) => record.bind_to_bootstrap_token(&access_token.token),
            Err(err) if access_token.from_cache && should_retry_with_fresh_access_token(&err) => {
                self.clear_access_token()?;
                let fresh_access_token = self.reauthenticate_access_token().await?;
                self.refresh_api_key(&fresh_access_token)
                    .await?
                    .bind_to_bootstrap_token(&fresh_access_token)
            }
            Err(err) => return Err(err),
        };
        let api_base = record.api_base();
        self.write_api_key_record(&record)?;
        Ok(AuthContext {
            api_key: record.token.unwrap_or_default(),
            api_base,
        })
    }

    pub(super) async fn auth_context_with_github_access_token(
        &self,
        access_token: &str,
    ) -> Result<AuthContext, AuthError> {
        let record = self.read_api_key_record()?;
        let api_base = record.api_base();
        if record.can_reuse_for_bootstrap_token(access_token)
            && let Some(token) = record.token
        {
            return Ok(AuthContext {
                api_key: token,
                api_base,
            });
        }

        let record = self
            .refresh_api_key(access_token)
            .await?
            .bind_to_bootstrap_token(access_token);
        let api_base = record.api_base();
        self.write_api_key_record(&record)?;
        Ok(AuthContext {
            api_key: record.token.unwrap_or_default(),
            api_base,
        })
    }

    async fn access_token(&self) -> Result<AccessTokenState, AuthError> {
        if let Some(token) = self.read_access_token()? {
            return Ok(AccessTokenState {
                token,
                from_cache: true,
            });
        }

        self.reauthenticate_access_token()
            .await
            .map(|token| AccessTokenState {
                token,
                from_cache: false,
            })
    }

    async fn login_device_flow(&self) -> Result<String, AuthError> {
        let client = reqwest::Client::new();
        let body = url::form_urlencoded::Serializer::new(String::new())
            .append_pair("client_id", GITHUB_CLIENT_ID)
            .append_pair("scope", "read:user")
            .finish();

        let device = client
            .post(GITHUB_DEVICE_CODE_URL)
            .header(reqwest::header::ACCEPT, "application/json")
            .header(
                reqwest::header::CONTENT_TYPE,
                "application/x-www-form-urlencoded",
            )
            .body(body)
            .send()
            .await?
            .error_for_status()?
            .json::<DeviceCodeResponse>()
            .await?;

        emit_device_code_prompt(
            &self.device_code_handler,
            DeviceCodePrompt {
                verification_uri: device.verification_uri.clone(),
                user_code: device.user_code.clone(),
            },
        );

        let deadline = std::time::Instant::now()
            + std::time::Duration::from_secs(
                device.expires_in.unwrap_or(DEVICE_CODE_TIMEOUT_SECONDS),
            );
        let mut interval = normalize_poll_interval_seconds(device.interval);

        while std::time::Instant::now() < deadline {
            let body = url::form_urlencoded::Serializer::new(String::new())
                .append_pair("client_id", GITHUB_CLIENT_ID)
                .append_pair("device_code", &device.device_code)
                .append_pair("grant_type", "urn:ietf:params:oauth:grant-type:device_code")
                .finish();

            let response = client
                .post(GITHUB_ACCESS_TOKEN_URL)
                .header(reqwest::header::ACCEPT, "application/json")
                .header(
                    reqwest::header::CONTENT_TYPE,
                    "application/x-www-form-urlencoded",
                )
                .body(body)
                .send()
                .await?
                .error_for_status()?
                .json::<AccessTokenResponse>()
                .await?;

            if let Some(access_token) = response.access_token {
                return Ok(access_token);
            }

            interval = next_poll_interval_seconds(
                interval,
                response.error.as_deref(),
                response.error_description.as_deref(),
            )?;
            tokio::time::sleep(std::time::Duration::from_secs(interval)).await;
        }

        Err(AuthError::Message(
            "Timed out waiting for GitHub Copilot device authorization".into(),
        ))
    }

    async fn refresh_api_key(&self, access_token: &str) -> Result<ApiKeyRecord, AuthError> {
        let client = reqwest::Client::new();
        let response = client
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

        if response.token.is_none() {
            return Err(AuthError::Message(
                "GitHub Copilot API key response did not include a token".into(),
            ));
        }

        Ok(response)
    }

    fn read_access_token(&self) -> Result<Option<String>, AuthError> {
        let Some(path) = &self.access_token_file else {
            return Ok(None);
        };

        match std::fs::read_to_string(path) {
            Ok(token) => {
                let token = token.trim();
                if token.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(token.to_owned()))
                }
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(err) => Err(err.into()),
        }
    }

    fn write_access_token(&self, token: &str) -> Result<(), AuthError> {
        let Some(path) = &self.access_token_file else {
            return Ok(());
        };

        ensure_parent_dir(path)?;
        std::fs::write(path, token.as_bytes())?;
        Ok(())
    }

    fn clear_access_token(&self) -> Result<(), AuthError> {
        let Some(path) = &self.access_token_file else {
            return Ok(());
        };

        match std::fs::remove_file(path) {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(err) => Err(err.into()),
        }
    }

    async fn reauthenticate_access_token(&self) -> Result<String, AuthError> {
        let token = self.login_device_flow().await?;
        self.write_access_token(&token)?;
        Ok(token)
    }

    fn read_api_key_record(&self) -> Result<ApiKeyRecord, AuthError> {
        let Some(path) = &self.api_key_file else {
            return Ok(ApiKeyRecord::default());
        };

        match std::fs::read(path) {
            Ok(bytes) => Ok(serde_json::from_slice(&bytes)?),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(ApiKeyRecord::default()),
            Err(err) => Err(err.into()),
        }
    }

    fn write_api_key_record(&self, record: &ApiKeyRecord) -> Result<(), AuthError> {
        let Some(path) = &self.api_key_file else {
            return Ok(());
        };

        ensure_parent_dir(path)?;
        std::fs::write(path, serde_json::to_vec_pretty(record)?)?;
        Ok(())
    }
}

impl ApiKeyRecord {
    fn api_base(&self) -> Option<String> {
        self.endpoints
            .as_ref()
            .and_then(|endpoints| endpoints.api.as_ref())
            .cloned()
    }

    fn can_reuse_for_oauth(&self, bootstrap_token: Option<&str>) -> bool {
        if !self.has_live_api_key() {
            return false;
        }

        match bootstrap_token {
            Some(bootstrap_token) => self.matches_bootstrap_token(bootstrap_token),
            None => true,
        }
    }

    fn can_reuse_for_bootstrap_token(&self, bootstrap_token: &str) -> bool {
        self.has_live_api_key() && self.matches_bootstrap_token(bootstrap_token)
    }

    fn bind_to_bootstrap_token(mut self, bootstrap_token: &str) -> Self {
        self.bootstrap_token_fingerprint = Some(bootstrap_token_fingerprint(bootstrap_token));
        self
    }

    fn has_live_api_key(&self) -> bool {
        self.token
            .as_ref()
            .is_some_and(|token| !token.trim().is_empty())
            && !token_expired(self.expires_at)
    }

    fn matches_bootstrap_token(&self, bootstrap_token: &str) -> bool {
        self.bootstrap_token_fingerprint.as_deref()
            == Some(bootstrap_token_fingerprint(bootstrap_token).as_str())
    }
}

fn bootstrap_token_fingerprint(bootstrap_token: &str) -> String {
    let mut hasher = DefaultHasher::new();
    bootstrap_token.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn emit_device_code_prompt(handler: &DeviceCodeHandler, prompt: DeviceCodePrompt) {
    if let Some(callback) = &handler.0 {
        callback(prompt);
    } else {
        println!(
            "Sign in with GitHub Copilot:\n1) Visit {}\n2) Enter code: {}",
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

fn token_expired(expires_at: Option<i64>) -> bool {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or_default();

    match expires_at {
        Some(exp) => now >= exp,
        None => true,
    }
}

fn normalize_poll_interval_seconds(interval: Option<u64>) -> u64 {
    interval.unwrap_or(DEVICE_CODE_POLL_SLEEP_SECONDS).max(1)
}

fn next_poll_interval_seconds(
    current_interval: u64,
    error: Option<&str>,
    error_description: Option<&str>,
) -> Result<u64, AuthError> {
    match error {
        Some("authorization_pending") => Ok(current_interval),
        Some("slow_down") => Ok(current_interval.saturating_add(DEVICE_CODE_SLOW_DOWN_SECONDS)),
        Some("expired_token") => Err(AuthError::Message(
            "GitHub device authorization expired before it completed".into(),
        )),
        Some("access_denied") => Err(AuthError::Message(
            "GitHub device authorization was denied".into(),
        )),
        Some(other) => Err(AuthError::Message(format_oauth_error(
            "GitHub device authorization failed",
            other,
            error_description,
        ))),
        None => Err(AuthError::Message(
            "GitHub device authorization failed: unknown error".into(),
        )),
    }
}

fn format_oauth_error(prefix: &str, error: &str, description: Option<&str>) -> String {
    match description
        .map(str::trim)
        .filter(|description| !description.is_empty())
    {
        Some(description) => format!("{prefix}: {error} ({description})"),
        None => format!("{prefix}: {error}"),
    }
}

fn should_retry_with_fresh_access_token(err: &AuthError) -> bool {
    match err {
        AuthError::Http(err) => should_retry_with_fresh_access_token_status(err.status()),
        _ => false,
    }
}

fn should_retry_with_fresh_access_token_status(status: Option<reqwest::StatusCode>) -> bool {
    matches!(
        status,
        Some(reqwest::StatusCode::UNAUTHORIZED | reqwest::StatusCode::FORBIDDEN)
    )
}

#[cfg(test)]
mod tests {
    use super::{
        ApiKeyRecord, bootstrap_token_fingerprint, next_poll_interval_seconds,
        normalize_poll_interval_seconds, should_retry_with_fresh_access_token_status,
    };
    use reqwest::StatusCode;

    #[test]
    fn api_key_record_parses_dynamic_api_base() {
        let record: ApiKeyRecord = serde_json::from_str(
            r#"{
                "token": "copilot-token",
                "expires_at": 1775791135,
                "endpoints": {
                    "api": "https://api.individual.githubcopilot.com"
                }
            }"#,
        )
        .expect("parse api key record");

        assert_eq!(
            record.api_base().as_deref(),
            Some("https://api.individual.githubcopilot.com")
        );
    }

    #[test]
    fn api_key_record_reuse_requires_matching_bootstrap_token_for_explicit_auth() {
        let record = ApiKeyRecord {
            token: Some("copilot-token".into()),
            expires_at: Some(i64::MAX),
            endpoints: None,
            bootstrap_token_fingerprint: Some(bootstrap_token_fingerprint("github-token-a")),
        };

        assert!(record.can_reuse_for_bootstrap_token("github-token-a"));
        assert!(!record.can_reuse_for_bootstrap_token("github-token-b"));
    }

    #[test]
    fn api_key_record_oauth_reuse_requires_match_when_bootstrap_token_is_available() {
        let record = ApiKeyRecord {
            token: Some("copilot-token".into()),
            expires_at: Some(i64::MAX),
            endpoints: None,
            bootstrap_token_fingerprint: Some(bootstrap_token_fingerprint("github-token-a")),
        };

        assert!(record.can_reuse_for_oauth(Some("github-token-a")));
        assert!(!record.can_reuse_for_oauth(Some("github-token-b")));
        assert!(record.can_reuse_for_oauth(None));
    }

    #[test]
    fn legacy_api_key_record_without_fingerprint_forces_refresh_when_bootstrap_token_is_known() {
        let record = ApiKeyRecord {
            token: Some("copilot-token".into()),
            expires_at: Some(i64::MAX),
            endpoints: None,
            bootstrap_token_fingerprint: None,
        };

        assert!(!record.can_reuse_for_bootstrap_token("github-token-a"));
        assert!(!record.can_reuse_for_oauth(Some("github-token-a")));
        assert!(record.can_reuse_for_oauth(None));
    }

    #[test]
    fn poll_interval_defaults_and_clamps() {
        assert_eq!(normalize_poll_interval_seconds(None), 5);
        assert_eq!(normalize_poll_interval_seconds(Some(0)), 1);
        assert_eq!(normalize_poll_interval_seconds(Some(9)), 9);
    }

    #[test]
    fn poll_interval_handles_pending_and_slow_down() {
        assert_eq!(
            next_poll_interval_seconds(5, Some("authorization_pending"), None)
                .expect("authorization pending interval"),
            5
        );
        assert_eq!(
            next_poll_interval_seconds(5, Some("slow_down"), None).expect("slow_down interval"),
            10
        );
    }

    #[test]
    fn poll_interval_rejects_terminal_errors() {
        let denied = next_poll_interval_seconds(5, Some("access_denied"), None)
            .expect_err("access denied should fail");
        assert_eq!(denied.to_string(), "GitHub device authorization was denied");

        let unknown = next_poll_interval_seconds(
            5,
            Some("device_flow_disabled"),
            Some("OAuth app device flow is disabled"),
        )
        .expect_err("device flow disabled should fail");
        assert_eq!(
            unknown.to_string(),
            "GitHub device authorization failed: device_flow_disabled (OAuth app device flow is disabled)"
        );
    }

    #[test]
    fn stale_access_token_retries_only_on_auth_failures() {
        assert!(should_retry_with_fresh_access_token_status(Some(
            StatusCode::UNAUTHORIZED
        )));
        assert!(should_retry_with_fresh_access_token_status(Some(
            StatusCode::FORBIDDEN
        )));
        assert!(!should_retry_with_fresh_access_token_status(Some(
            StatusCode::BAD_GATEWAY
        )));
        assert!(!should_retry_with_fresh_access_token_status(None));
    }
}
