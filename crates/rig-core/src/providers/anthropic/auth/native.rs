//! Native Anthropic OAuth and token cache implementation.
//!
//! We use the loopback authorization-code flow (browser redirect to
//! `http://localhost:53692/callback`) rather than the OAuth device-code flow.
//! Anthropic does expose a device-authorization endpoint
//! (`https://platform.claude.com/v1/oauth/device_authorization`), but the
//! Claude Code OAuth client id used here is not authorized for that grant
//! (the endpoint rejects it with `unauthorized_client`). The loopback flow
//! below is the grant this client supports.

use super::{
    AuthContext, AuthError, AuthSource, ManualCodeHandler, OAuthPrompt, OAuthPromptHandler,
};
use crate::http_client::{self, HttpClientExt};
use base64::Engine;
use base64::prelude::BASE64_URL_SAFE_NO_PAD;
use http::{Method, StatusCode};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Claude OAuth client id used by the Claude Code loopback flow.
pub const ANTHROPIC_OAUTH_CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
/// Claude OAuth authorization URL.
pub const ANTHROPIC_AUTHORIZE_URL: &str = "https://claude.ai/oauth/authorize";
/// Anthropic OAuth token URL.
pub const ANTHROPIC_TOKEN_URL: &str = "https://platform.claude.com/v1/oauth/token";
/// Claude Code loopback redirect URI registered for the OAuth client.
pub const ANTHROPIC_REDIRECT_URI: &str = "http://localhost:53692/callback";
/// Claude Code OAuth scopes required for subscription-backed inference.
pub const ANTHROPIC_SCOPES: &str = "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload";
const TOKEN_EXPIRY_SKEW_SECONDS: i64 = 5 * 60;
const CALLBACK_PORT: u16 = 53692;
const CALLBACK_TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Debug, Clone)]
pub(super) struct PlatformAuthenticator {
    auth_file: Option<PathBuf>,
    oauth_prompt_handler: OAuthPromptHandler,
    manual_code_handler: ManualCodeHandler,
    /// In-memory copy of the last known token record, shared across clones.
    ///
    /// Lets repeat requests reuse a still-valid access token without hitting
    /// the disk on every call. Reads are populated from the on-disk record and
    /// writes keep it consistent with the cache file.
    cached_record: Arc<Mutex<Option<AuthRecord>>>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub(super) struct AuthRecord {
    access: Option<String>,
    refresh: Option<String>,
    expires: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct TokenResponse {
    access_token: String,
    refresh_token: Option<String>,
    expires_in: Option<i64>,
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
    pub(super) fn new(
        auth_file: Option<PathBuf>,
        oauth_prompt_handler: OAuthPromptHandler,
        manual_code_handler: ManualCodeHandler,
    ) -> Self {
        Self {
            auth_file,
            oauth_prompt_handler,
            manual_code_handler,
            cached_record: Arc::new(Mutex::new(None)),
        }
    }

    pub(super) async fn auth_context_oauth<H>(
        &self,
        http_client: &H,
    ) -> Result<AuthContext, AuthError>
    where
        H: HttpClientExt,
    {
        let record = self.read_auth_record().await?;
        if let Some(access) = record.access.clone()
            && !token_expired(record.expires)
        {
            return Ok(AuthContext {
                access_token: access,
                source: AuthSource::OAuth,
            });
        }

        if let Some(refresh_token) = record.refresh.clone() {
            match self.refresh_tokens(http_client, &refresh_token).await {
                Ok(refreshed) => {
                    let access_token = auth_record_access_token(&refreshed)?;
                    self.write_auth_record(&refreshed).await?;
                    return Ok(AuthContext {
                        access_token,
                        source: AuthSource::OAuth,
                    });
                }
                Err(RefreshTokensError::Reauthenticate) => {}
                Err(RefreshTokensError::Auth(err)) => return Err(err),
            }
        }

        let fresh = self.login_browser_flow(http_client).await?;
        let access_token = auth_record_access_token(&fresh)?;
        self.write_auth_record(&fresh).await?;
        Ok(AuthContext {
            access_token,
            source: AuthSource::OAuth,
        })
    }

    /// Read the cached token record, preferring an in-memory copy with a
    /// still-valid access token to avoid per-request disk I/O.
    ///
    /// Returns the cached record when its access token has not expired,
    /// otherwise reads the on-disk record and refreshes the in-memory cache.
    async fn read_auth_record(&self) -> Result<AuthRecord, AuthError> {
        if let Some(record) = self.cached_valid_record() {
            return Ok(record);
        }
        let Some(path) = &self.auth_file else {
            return Ok(AuthRecord::default());
        };
        let record = match tokio::fs::read(path).await {
            Ok(bytes) => serde_json::from_slice(&bytes)?,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => AuthRecord::default(),
            Err(err) => return Err(err.into()),
        };
        self.store_cached_record(record.clone());
        Ok(record)
    }

    /// Return the in-memory record when it holds a non-expired access token.
    fn cached_valid_record(&self) -> Option<AuthRecord> {
        let guard = self
            .cached_record
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let record = guard.as_ref()?;
        if record
            .access
            .as_ref()
            .is_some_and(|access| !access.is_empty())
            && !token_expired(record.expires)
        {
            Some(record.clone())
        } else {
            None
        }
    }

    /// Replace the in-memory cache with the supplied record.
    fn store_cached_record(&self, record: AuthRecord) {
        let mut guard = self
            .cached_record
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        *guard = Some(record);
    }

    async fn write_auth_record(&self, record: &AuthRecord) -> Result<(), AuthError> {
        self.store_cached_record(record.clone());
        let Some(path) = &self.auth_file else {
            return Ok(());
        };
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let bytes = serde_json::to_vec_pretty(record)?;
        #[cfg(unix)]
        {
            let mut file = tokio::fs::OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .mode(0o600)
                .open(path)
                .await?;
            file.write_all(&bytes).await?;
        }
        #[cfg(not(unix))]
        tokio::fs::write(path, bytes).await?;
        Ok(())
    }

    async fn login_browser_flow<H>(&self, http_client: &H) -> Result<AuthRecord, AuthError>
    where
        H: HttpClientExt,
    {
        let verifier = pkce_verifier()?;
        let url = authorization_url(&verifier);
        if let Some(callback) = &self.oauth_prompt_handler.0 {
            callback(OAuthPrompt {
                authorization_url: url.clone(),
            });
        } else {
            tracing::info!(authorization_url = %url, "Sign in with Anthropic Claude OAuth");
        }

        let captured = capture_callback_code().await.or_else(|| {
            self.manual_code_handler
                .0
                .as_ref()
                .and_then(|handler| handler())
        });
        let Some(input) = captured else {
            return Err(AuthError::Message(format!(
                "Anthropic OAuth requires a callback code. Open {url} and provide the redirected URL or code#state."
            )));
        };
        let code = parse_callback_code(&input, &verifier)?;
        let tokens = exchange_code(http_client, &code, &verifier).await?;
        Ok(build_auth_record(tokens, None))
    }

    async fn refresh_tokens<H>(
        &self,
        http_client: &H,
        refresh_token: &str,
    ) -> Result<AuthRecord, RefreshTokensError>
    where
        H: HttpClientExt,
    {
        let body = serde_json::json!({
            "grant_type": "refresh_token",
            "client_id": ANTHROPIC_OAUTH_CLIENT_ID,
            "refresh_token": refresh_token,
        });

        match send_token_request(http_client, body).await {
            Ok(tokens) => Ok(build_auth_record(tokens, Some(refresh_token.to_owned()))),
            Err(AuthError::Transport(http_client::Error::InvalidStatusCodeWithMessage(
                status,
                body,
            ))) => {
                let oauth_error = serde_json::from_str::<OAuthErrorResponse>(&body).ok();
                if matches!(status, StatusCode::BAD_REQUEST | StatusCode::UNAUTHORIZED)
                    && oauth_error.as_ref().and_then(|e| e.error.as_deref())
                        == Some("invalid_grant")
                {
                    return Err(RefreshTokensError::Reauthenticate);
                }
                Err(RefreshTokensError::Auth(AuthError::Message(
                    format_refresh_error(status, oauth_error.as_ref(), &body),
                )))
            }
            Err(err) => Err(RefreshTokensError::Auth(err)),
        }
    }
}

async fn exchange_code<H>(
    http_client: &H,
    code: &str,
    verifier: &str,
) -> Result<TokenResponse, AuthError>
where
    H: HttpClientExt,
{
    send_token_request(
        http_client,
        serde_json::json!({
            "grant_type": "authorization_code",
            "client_id": ANTHROPIC_OAUTH_CLIENT_ID,
            "code": code,
            "state": verifier,
            "redirect_uri": ANTHROPIC_REDIRECT_URI,
            "code_verifier": verifier,
        }),
    )
    .await
}

async fn send_token_request<H>(
    http_client: &H,
    body: serde_json::Value,
) -> Result<TokenResponse, AuthError>
where
    H: HttpClientExt,
{
    let req = http::Request::builder()
        .method(Method::POST)
        .uri(ANTHROPIC_TOKEN_URL)
        .header(http::header::CONTENT_TYPE, "application/json")
        .body(serde_json::to_vec(&body)?)
        .map_err(http_client::Error::Protocol)?;
    let response = http_client.send::<_, Vec<u8>>(req).await?;
    let body = response.into_body().await?;
    serde_json::from_slice::<TokenResponse>(&body).map_err(|error| {
        AuthError::Message(format!(
            "Anthropic OAuth token response could not be parsed: {error}; body: {}",
            String::from_utf8_lossy(&body)
        ))
    })
}

async fn capture_callback_code() -> Option<String> {
    let listener = TcpListener::bind(("127.0.0.1", CALLBACK_PORT)).await.ok()?;
    let (mut stream, _) = tokio::time::timeout(CALLBACK_TIMEOUT, listener.accept())
        .await
        .ok()?
        .ok()?;

    let mut request_bytes = Vec::new();
    let mut buffer = [0_u8; 1024];
    loop {
        let read = tokio::time::timeout(CALLBACK_TIMEOUT, stream.read(&mut buffer))
            .await
            .ok()?
            .ok()?;
        if read == 0 {
            break;
        }
        request_bytes.extend_from_slice(buffer.get(..read)?);
        if request_bytes.windows(4).any(|window| window == b"\r\n\r\n")
            || request_bytes.len() >= 16 * 1024
        {
            break;
        }
    }

    let request = String::from_utf8_lossy(&request_bytes);
    let path = request
        .lines()
        .next()?
        .split_whitespace()
        .nth(1)?
        .to_string();
    let _ = stream
        .write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/plain\r\n\r\nClaude OAuth complete. You can close this tab.")
        .await;
    Some(format!("http://localhost:{CALLBACK_PORT}{path}"))
}

pub(super) fn authorization_url(verifier: &str) -> String {
    let challenge = pkce_challenge(verifier);
    url::form_urlencoded::Serializer::new(format!("{ANTHROPIC_AUTHORIZE_URL}?"))
        .append_pair("client_id", ANTHROPIC_OAUTH_CLIENT_ID)
        .append_pair("redirect_uri", ANTHROPIC_REDIRECT_URI)
        .append_pair("response_type", "code")
        .append_pair("scope", ANTHROPIC_SCOPES)
        .append_pair("code_challenge", &challenge)
        .append_pair("code_challenge_method", "S256")
        // Claude's OAuth flow echoes the verifier as state and returns `code#state`;
        // keep this wire shape for Claude Code compatibility.
        .append_pair("state", verifier)
        .append_pair("code", "true")
        .finish()
}

/// Extract the authorization code from an OAuth callback, validating state
/// whenever it is present.
///
/// # Arguments
/// * `input` - Either a `code#state` fragment, a full redirect URL, or a bare
///   authorization code pasted manually by the user.
/// * `expected_state` - The PKCE-derived state value sent in the authorization
///   request; used as a CSRF check against the value echoed by Anthropic.
///
/// # Returns
/// The authorization code, or [`AuthError::Message`] when the echoed state does
/// not match `expected_state` or no code is present.
///
/// # Security tradeoff
/// The `code#state` and URL forms carry the echoed state and are validated
/// against `expected_state`. A manually pasted bare code carries no state, so
/// it is accepted unchecked. This is a deliberate concession for the CLI
/// copy/paste flow: such a code is only obtained by the user completing the
/// browser authorization, and PKCE still binds the subsequent token exchange to
/// the locally generated verifier.
pub(super) fn parse_callback_code(input: &str, expected_state: &str) -> Result<String, AuthError> {
    let trimmed = input.trim();
    if let Some((code, state)) = trimmed.split_once('#') {
        if state != expected_state {
            return Err(AuthError::Message("Anthropic OAuth state mismatch".into()));
        }
        return Ok(code.to_string());
    }
    if let Ok(url) = url::Url::parse(trimmed) {
        let mut code = None;
        let mut state = None;
        let mut error = None;
        let mut error_description = None;
        for (key, value) in url.query_pairs() {
            match key.as_ref() {
                "code" => code = Some(value.to_string()),
                "state" => state = Some(value.to_string()),
                "error" => error = Some(value.to_string()),
                "error_description" => error_description = Some(value.to_string()),
                _ => {}
            }
        }
        if let Some(error) = error {
            return Err(AuthError::Message(format_oauth_error(
                "Anthropic OAuth callback failed",
                &error,
                error_description.as_deref(),
            )));
        }
        if state.as_deref() != Some(expected_state) {
            return Err(AuthError::Message("Anthropic OAuth state mismatch".into()));
        }
        return code
            .ok_or_else(|| AuthError::Message("Anthropic OAuth callback missing code".into()));
    }
    // Manual copy/paste may provide only the code; see the `Security tradeoff`
    // note above for why the missing state is accepted here.
    Ok(trimmed.to_string())
}

pub(super) fn pkce_challenge(verifier: &str) -> String {
    BASE64_URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()))
}

/// Generate a cryptographically random PKCE verifier.
///
/// Returns a base64url-encoded verifier suitable for use as both the PKCE
/// verifier and OAuth state value.
fn pkce_verifier() -> Result<String, AuthError> {
    let mut bytes = [0_u8; 32];
    getrandom::getrandom(&mut bytes).map_err(|err| {
        AuthError::Message(format!(
            "failed to generate cryptographically secure OAuth state: {err}"
        ))
    })?;
    Ok(BASE64_URL_SAFE_NO_PAD.encode(bytes))
}

fn auth_record_access_token(record: &AuthRecord) -> Result<String, AuthError> {
    record
        .access
        .clone()
        .filter(|access| !access.is_empty())
        .ok_or_else(|| {
            AuthError::Message("Anthropic OAuth token response missing access token".into())
        })
}

fn build_auth_record(tokens: TokenResponse, previous_refresh_token: Option<String>) -> AuthRecord {
    AuthRecord {
        access: Some(tokens.access_token),
        refresh: tokens.refresh_token.or(previous_refresh_token),
        expires: tokens
            .expires_in
            .map(|expires_in| now_seconds() + expires_in - TOKEN_EXPIRY_SKEW_SECONDS),
    }
}

pub(super) fn token_expired(expires: Option<i64>) -> bool {
    match expires {
        Some(expires) => now_seconds() >= expires,
        None => true,
    }
}

fn now_seconds() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or_default()
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

fn format_refresh_error(
    status: StatusCode,
    oauth_error: Option<&OAuthErrorResponse>,
    body: &str,
) -> String {
    if let Some(description) = oauth_error
        .and_then(|e| e.error_description.as_deref())
        .filter(|s| !s.trim().is_empty())
    {
        return format!(
            "Anthropic token refresh failed: {status} {} ({description})",
            oauth_error
                .and_then(|e| e.error.as_deref())
                .unwrap_or("unknown_error")
        );
    }
    if let Some(error) = oauth_error.and_then(|e| e.error.as_deref()) {
        return format!("Anthropic token refresh failed: {status} {error}");
    }
    if !body.trim().is_empty() {
        return format!("Anthropic token refresh failed: {status} {body}");
    }
    format!("Anthropic token refresh failed: {status}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_url_and_code_state_callbacks() {
        assert_eq!(
            parse_callback_code("abc#state", "state").expect("code"),
            "abc"
        );
        assert_eq!(
            parse_callback_code(
                "http://localhost:53692/callback?code=abc&state=state",
                "state"
            )
            .expect("url"),
            "abc"
        );
        assert!(parse_callback_code("abc#wrong", "state").is_err());
    }

    #[test]
    fn parses_callback_error_redirects() {
        let error = parse_callback_code(
            "http://localhost:53692/callback?error=access_denied&error_description=Nope&state=state",
            "state",
        )
        .expect_err("error redirects should be reported");

        assert_eq!(
            error.to_string(),
            "Anthropic OAuth callback failed: access_denied (Nope)"
        );
    }

    #[test]
    fn pkce_challenge_is_s256() {
        assert_eq!(
            pkce_challenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"),
            "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
        );
    }

    #[test]
    fn auth_record_uses_skewed_expiry_and_preserves_refresh() {
        let before = now_seconds();
        let record = build_auth_record(
            TokenResponse {
                access_token: "access".into(),
                refresh_token: None,
                expires_in: Some(600),
            },
            Some("refresh".into()),
        );
        assert_eq!(record.refresh.as_deref(), Some("refresh"));
        assert!(record.expires.expect("expires") >= before + 300);
    }
}
