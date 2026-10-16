use super::{
    ApiKeyRecord, AuthContext, AuthError, Authenticator, DeviceCodePrompt, STORE_KEY,
    refresh_api_key,
};
use crate::http_client::HttpClientExt;
use crate::providers::internal::auth::{load_record, request, save_record, send_json};
use bytes::Bytes;
use http::Method;
use serde::Deserialize;

const GITHUB_CLIENT_ID: &str = "Iv1.b507a08c87ecfe98";
const GITHUB_DEVICE_CODE_URL: &str = "https://github.com/login/device/code";
const GITHUB_ACCESS_TOKEN_URL: &str = "https://github.com/login/oauth/access_token";
const DEVICE_CODE_POLL_SLEEP_SECONDS: u64 = 5;
const DEVICE_CODE_TIMEOUT_SECONDS: u64 = 15 * 60;
const DEVICE_CODE_SLOW_DOWN_SECONDS: u64 = 5;

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

impl Authenticator {
    pub(super) async fn auth_context_oauth<H>(&self, http: &H) -> Result<AuthContext, AuthError>
    where
        H: HttpClientExt,
    {
        if let Some(context) = self.0.remembered.live(0) {
            return Ok(context);
        }
        let _held = self.0.hold(STORE_KEY).await?;
        let mut record: ApiKeyRecord = load_record(&*self.0.store, STORE_KEY).await?;
        if record.can_reuse_for_oauth(record.github_token.as_deref()) {
            return Ok(self.remember(record));
        }

        let from_cache = record.github_token.is_some();
        let github_token = match record.github_token.clone() {
            Some(token) => token,
            None => self.sign_in_to_github(http, &mut record).await?,
        };
        let session = match refresh_api_key(http, &github_token).await {
            Ok(session) => session.bound_to(&github_token, Some(github_token.clone())),
            Err(err) if from_cache && should_retry_with_fresh_access_token(&err) => {
                record.github_token = None;
                save_record(&*self.0.store, STORE_KEY, &record).await?;
                let fresh = self.sign_in_to_github(http, &mut record).await?;
                refresh_api_key(http, &fresh)
                    .await?
                    .bound_to(&fresh, Some(fresh.clone()))
            }
            Err(err) => return Err(err),
        };
        save_record(&*self.0.store, STORE_KEY, &session).await?;
        Ok(self.remember(session))
    }

    /// Run the device flow when allowed, and keep the GitHub token it gets
    /// in `record` and the store before the exchange that may fail.
    async fn sign_in_to_github<H>(
        &self,
        http: &H,
        record: &mut ApiKeyRecord,
    ) -> Result<String, AuthError>
    where
        H: HttpClientExt,
    {
        if !self.0.allow_device_flow {
            return Err(AuthError::Message(
                "GitHub Copilot sign-in required. Reconnect Copilot in Settings before using this provider."
                    .into(),
            ));
        }
        let token = self.login_device_flow(http).await?;
        record.github_token = Some(token.clone());
        save_record(&*self.0.store, STORE_KEY, record).await?;
        Ok(token)
    }

    async fn login_device_flow<H>(&self, http: &H) -> Result<String, AuthError>
    where
        H: HttpClientExt,
    {
        let body = url::form_urlencoded::Serializer::new(String::new())
            .append_pair("client_id", GITHUB_CLIENT_ID)
            .append_pair("scope", "read:user")
            .finish();

        let device: DeviceCodeResponse = send_json(
            http,
            request(Method::POST, GITHUB_DEVICE_CODE_URL)
                .header(http::header::ACCEPT, "application/json")
                .header(
                    http::header::CONTENT_TYPE,
                    "application/x-www-form-urlencoded",
                )
                .body(Bytes::from(body)),
        )
        .await?;

        self.0.device_code_handler.show(
            DeviceCodePrompt {
                verification_uri: device.verification_uri.clone(),
                user_code: device.user_code.clone(),
            },
            &format!(
                "Sign in with GitHub Copilot:\n1) Visit {}\n2) Enter code: {}",
                device.verification_uri, device.user_code
            ),
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

            let response: AccessTokenResponse = send_json(
                http,
                request(Method::POST, GITHUB_ACCESS_TOKEN_URL)
                    .header(http::header::ACCEPT, "application/json")
                    .header(
                        http::header::CONTENT_TYPE,
                        "application/x-www-form-urlencoded",
                    )
                    .body(Bytes::from(body)),
            )
            .await?;

            if let Some(access_token) = response.access_token {
                return Ok(access_token);
            }

            interval = next_poll_interval_seconds(
                interval,
                response.error.as_deref(),
                response.error_description.as_deref(),
            )?;
            crate::wasm_compat::sleep(std::time::Duration::from_secs(interval)).await;
        }

        Err(AuthError::Message(
            "Timed out waiting for GitHub Copilot device authorization".into(),
        ))
    }
}

impl ApiKeyRecord {
    fn can_reuse_for_oauth(&self, bootstrap_token: Option<&str>) -> bool {
        if !self.has_live_api_key() {
            return false;
        }

        bootstrap_token.is_none_or(|bootstrap_token| self.matches_bootstrap_token(bootstrap_token))
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
        AuthError::Http(err) => {
            should_retry_with_fresh_access_token_status(err.non_success_status())
        }
        _ => false,
    }
}

fn should_retry_with_fresh_access_token_status(status: Option<http::StatusCode>) -> bool {
    matches!(
        status,
        Some(http::StatusCode::UNAUTHORIZED | http::StatusCode::FORBIDDEN)
    )
}

#[cfg(test)]
mod tests;
