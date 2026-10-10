//! Native ChatGPT OAuth and token cache implementation.

use super::{
    AuthContext, AuthError, Authenticator, BrowserSignInPrompt, DeviceCodePrompt, SignInMethod,
    SignInPrompt,
};
use crate::http_client::HttpClientExt;
use crate::providers::internal::auth::device::{
    emit_device_code_prompt, read_json_record, token_expired, write_json_record,
};
use crate::providers::internal::auth::{request, send_json};
use crate::wasm_compat::WasmCompatSend;
use base64::Engine;
use base64::prelude::BASE64_URL_SAFE_NO_PAD;
use bytes::Bytes;
use http::Method;
use serde::{Deserialize, Deserializer, Serialize};

mod browser;

const CHATGPT_AUTH_BASE: &str = "https://auth.openai.com";
const CHATGPT_DEVICE_CODE_URL: &str = "https://auth.openai.com/api/accounts/deviceauth/usercode";
const CHATGPT_DEVICE_TOKEN_URL: &str = "https://auth.openai.com/api/accounts/deviceauth/token";
const CHATGPT_OAUTH_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
const CHATGPT_DEVICE_VERIFY_URL: &str = "https://auth.openai.com/codex/device";
const CHATGPT_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const TOKEN_EXPIRY_SKEW_SECONDS: i64 = 60;
const DEVICE_CODE_TIMEOUT_SECONDS: i64 = 15 * 60;
const BROWSER_SIGN_IN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(15 * 60);
const DEVICE_CODE_POLL_SLEEP_SECONDS: u64 = 5;

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

impl SignInMethod {
    /// The browser when one opened here would reach the user and
    /// `device_asked` is false: not over SSH, and on Linux and the BSDs
    /// only inside an X11 or Wayland session. Otherwise the device code.
    /// Native only.
    pub fn detect(device_asked: bool) -> Self {
        let set = |name: &str| std::env::var_os(name).is_some_and(|value| !value.is_empty());
        let graphical = !(set("SSH_CONNECTION") || set("SSH_TTY"))
            && (cfg!(any(target_os = "macos", windows))
                || set("DISPLAY")
                || set("WAYLAND_DISPLAY"));
        if device_asked || !graphical {
            Self::DeviceCode
        } else {
            Self::Browser
        }
    }
}

impl Authenticator {
    pub(super) async fn auth_context_oauth<H>(&self, http: &H) -> Result<AuthContext, AuthError>
    where
        H: HttpClientExt,
    {
        let _refresh = self.refresh_lock.lock().await;
        let mut record: AuthRecord = read_json_record(self.auth_file.as_deref())?;

        if let Some(access_token) = record.access_token.clone()
            && !token_expired(record.expires_at, TOKEN_EXPIRY_SKEW_SECONDS)
        {
            let account_id = record
                .account_id
                .clone()
                .or_else(|| extract_account_id(record.id_token.as_deref()))
                .or_else(|| extract_account_id(Some(&access_token)));
            if account_id != record.account_id {
                record.account_id.clone_from(&account_id);
                write_json_record(self.auth_file.as_deref(), &record)?;
            }
            return Ok(AuthContext {
                access_token: access_token.into(),
                account_id,
            });
        }

        if let Some(refresh_token) = record.refresh_token.clone() {
            match self.refresh_tokens(http, &refresh_token).await {
                Ok(refreshed) => {
                    write_json_record(self.auth_file.as_deref(), &refreshed)?;
                    return Ok(AuthContext {
                        access_token: refreshed.access_token.unwrap_or_default().into(),
                        account_id: refreshed.account_id,
                    });
                }
                Err(RefreshTokensError::Reauthenticate) => {}
                Err(RefreshTokensError::Auth(err)) => return Err(err),
            }
        }

        if !self.allow_device_flow {
            return Err(AuthError::Message(
                "ChatGPT sign-in required. Reconnect ChatGPT in Settings before using this provider."
                    .into(),
            ));
        }

        let fresh = self.login_device_flow(http).await?;
        self.store(fresh)
    }

    /// Sign in again by `method`, whatever is cached, telling `prompt` what
    /// the user must do. A browser sign-in whose callback ports are taken
    /// says so and falls back to the device code. The credential is stored
    /// in the auth file. Native only.
    pub async fn sign_in<H, F>(
        &self,
        http: &H,
        method: SignInMethod,
        mut prompt: F,
    ) -> Result<AuthContext, AuthError>
    where
        H: HttpClientExt,
        F: FnMut(SignInPrompt) + WasmCompatSend,
    {
        if method == SignInMethod::Browser {
            let browser = self
                .sign_in_with_browser(http, |page| prompt(SignInPrompt::Browser(page)))
                .await;
            match browser {
                Err(AuthError::Io(error)) if error.kind() == std::io::ErrorKind::AddrInUse => {
                    prompt(SignInPrompt::BrowserUnavailable(error.to_string()));
                }
                other => return other,
            }
        }
        self.sign_in_with_device_code(http).await
    }

    /// Sign in again with the device-code flow, whatever is cached: the
    /// handler shows the code to enter, and the credential is stored in the
    /// auth file. Native only.
    pub async fn sign_in_with_device_code<H>(&self, http: &H) -> Result<AuthContext, AuthError>
    where
        H: HttpClientExt,
    {
        let fresh = self.login_device_flow(http).await?;
        let _refresh = self.refresh_lock.lock().await;
        self.store(fresh)
    }

    /// Sign in again in the browser, whatever is cached: an authorization
    /// code with PKCE, returned to a one-shot listener on `127.0.0.1` port
    /// 1455, or 1457 when that is taken. The default browser is opened on
    /// the sign-in page and `prompt` receives its URL, for the user to open
    /// when the browser did not. The credential is stored in the auth file.
    /// Native only.
    ///
    /// When both ports are taken, fails with [`AuthError::Io`] of kind
    /// [`std::io::ErrorKind::AddrInUse`], before prompting, so the caller can
    /// fall back to [`Self::sign_in_with_device_code`]. Gives up after 15
    /// minutes. Dropping the future stops the listener within a fraction of
    /// a second and frees its port.
    pub async fn sign_in_with_browser<H, F>(
        &self,
        http: &H,
        prompt: F,
    ) -> Result<AuthContext, AuthError>
    where
        H: HttpClientExt,
        F: FnOnce(BrowserSignInPrompt) + WasmCompatSend,
    {
        let (listener, port) = browser::bind()?;
        let redirect_uri = browser::redirect_uri(port);
        let pkce = browser::Pkce::generate();
        let state = browser::random_token(2);
        let authorize_url =
            browser::authorize_url(&redirect_uri, &pkce.challenge, &state, &originator());
        let callback = browser::spawn_listener(
            listener,
            state,
            std::time::Instant::now() + BROWSER_SIGN_IN_TIMEOUT,
        )?;
        let browser_launched = browser::open_browser(&authorize_url);
        prompt(BrowserSignInPrompt {
            authorize_url,
            browser_launched,
        });

        let received = callback.await.map_err(|_| {
            AuthError::Message("the sign-in listener stopped unexpectedly".into())
        })??;
        let exchanged =
            exchange_authorization_code(http, &received.code, &pkce.verifier, &redirect_uri).await;
        received.finish(
            exchanged
                .as_ref()
                .map(|_| ())
                .map_err(|error| error.to_string()),
        );
        let fresh = exchanged?;
        let _refresh = self.refresh_lock.lock().await;
        self.store(fresh)
    }

    /// Persist a fresh sign-in and return its context.
    fn store(&self, fresh: AuthRecord) -> Result<AuthContext, AuthError> {
        write_json_record(self.auth_file.as_deref(), &fresh)?;
        Ok(AuthContext {
            access_token: fresh.access_token.unwrap_or_default().into(),
            account_id: fresh.account_id,
        })
    }

    async fn login_device_flow<H>(&self, http: &H) -> Result<AuthRecord, AuthError>
    where
        H: HttpClientExt,
    {
        let device: DeviceCodeResponse = send_json(
            http,
            request(Method::POST, CHATGPT_DEVICE_CODE_URL)
                .header(http::header::CONTENT_TYPE, "application/json")
                .body(Bytes::from(serde_json::to_vec(
                    &serde_json::json!({ "client_id": CHATGPT_CLIENT_ID }),
                )?)),
        )
        .await?;

        emit_device_code_prompt(
            self.device_code_handler.0.as_ref(),
            DeviceCodePrompt {
                verification_uri: CHATGPT_DEVICE_VERIFY_URL.to_string(),
                user_code: device.user_code.clone(),
            },
            &format!(
                "Sign in with ChatGPT:\n1) Visit {CHATGPT_DEVICE_VERIFY_URL}\n2) Enter code: {}\nDo not share this device code.",
                device.user_code
            ),
        );

        let interval = device.interval.unwrap_or(DEVICE_CODE_POLL_SLEEP_SECONDS);
        let start = std::time::Instant::now();
        let code = loop {
            if start.elapsed().as_secs() as i64 >= DEVICE_CODE_TIMEOUT_SECONDS {
                return Err(AuthError::Message(
                    "Timed out waiting for ChatGPT device authorization".into(),
                ));
            }

            let poll = send_json::<_, DeviceTokenResponse>(
                http,
                request(Method::POST, CHATGPT_DEVICE_TOKEN_URL)
                    .header(http::header::CONTENT_TYPE, "application/json")
                    .body(Bytes::from(serde_json::to_vec(&serde_json::json!({
                        "device_auth_id": device.device_auth_id,
                        "user_code": device.user_code,
                    }))?)),
            )
            .await;

            match poll {
                Ok(token_response) => break token_response,
                // Still pending: the endpoint answers 403/404 until the user
                // completes authorization.
                Err(AuthError::Http(err))
                    if matches!(
                        err.non_success_status().map(|status| status.as_u16()),
                        Some(403 | 404)
                    ) =>
                {
                    crate::wasm_compat::sleep(std::time::Duration::from_secs(interval)).await;
                    continue;
                }
                Err(AuthError::Http(err)) if err.non_success_status().is_some() => {
                    let status = err.non_success_status().unwrap_or_default();
                    let text = err.non_success_body().unwrap_or_default();
                    return Err(AuthError::Message(format!(
                        "ChatGPT device authorization failed: {status} {text}"
                    )));
                }
                Err(err) => return Err(err),
            }
        };

        exchange_authorization_code(
            http,
            &code.authorization_code,
            &code.code_verifier,
            &format!("{CHATGPT_AUTH_BASE}/deviceauth/callback"),
        )
        .await
    }

    async fn refresh_tokens<H>(
        &self,
        http: &H,
        refresh_token: &str,
    ) -> Result<AuthRecord, RefreshTokensError>
    where
        H: HttpClientExt,
    {
        let form = [
            ("client_id", CHATGPT_CLIENT_ID),
            ("grant_type", "refresh_token"),
            ("refresh_token", refresh_token),
            ("scope", "openid profile email"),
        ];

        let body = url::form_urlencoded::Serializer::new(String::new())
            .extend_pairs(form)
            .finish();

        let response = send_json::<_, OAuthTokenResponse>(
            http,
            request(Method::POST, CHATGPT_OAUTH_TOKEN_URL)
                .header(
                    http::header::CONTENT_TYPE,
                    "application/x-www-form-urlencoded",
                )
                .body(Bytes::from(body)),
        )
        .await;

        let (status, body) = match response {
            Ok(tokens) => {
                return Ok(build_auth_record(tokens, Some(refresh_token.to_owned())));
            }
            Err(AuthError::Http(err)) if err.non_success_status().is_some() => (
                err.non_success_status().unwrap_or_default(),
                err.non_success_body().unwrap_or_default().to_owned(),
            ),
            Err(err) => return Err(RefreshTokensError::Auth(err)),
        };
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

/// Exchange an authorization code and its PKCE verifier for tokens.
async fn exchange_authorization_code<H>(
    http: &H,
    code: &str,
    code_verifier: &str,
    redirect_uri: &str,
) -> Result<AuthRecord, AuthError>
where
    H: HttpClientExt,
{
    let form = [
        ("grant_type", "authorization_code"),
        ("code", code),
        ("redirect_uri", redirect_uri),
        ("client_id", CHATGPT_CLIENT_ID),
        ("code_verifier", code_verifier),
    ];
    let body = url::form_urlencoded::Serializer::new(String::new())
        .extend_pairs(form)
        .finish();

    let tokens: OAuthTokenResponse = send_json(
        http,
        request(Method::POST, CHATGPT_OAUTH_TOKEN_URL)
            .header(
                http::header::CONTENT_TYPE,
                "application/x-www-form-urlencoded",
            )
            .body(Bytes::from(body)),
    )
    .await?;

    Ok(build_auth_record(tokens, None))
}

/// The originator the ChatGPT dialect sends, named to the sign-in page too.
fn originator() -> String {
    let identity = crate::providers::chatgpt::DIALECT.quirks.identity;
    identity
        .and_then(|identity| std::env::var(identity.originator_env).ok())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| {
            identity
                .map_or(crate::providers::chatgpt::DEFAULT_ORIGINATOR, |identity| {
                    identity.originator
                })
                .to_owned()
        })
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

fn should_reauthenticate_after_refresh(status: http::StatusCode, error_code: Option<&str>) -> bool {
    matches!(
        status,
        http::StatusCode::BAD_REQUEST | http::StatusCode::UNAUTHORIZED
    ) && matches!(error_code, Some("invalid_grant"))
}

fn format_refresh_error(
    status: http::StatusCode,
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
mod tests;
