//! Native SuperGrok subscription OAuth token-cache implementation.
//!
//! This intentionally consumes an existing xAI OAuth `auth.json` cache instead
//! of starting a browser/loopback sign-in flow inside Rig. The cache format is
//! interoperable with LiteLLM's `xai_oauth` cache and stores an access token,
//! refresh token, expiry, and token endpoint.

use super::AuthError;
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

const XAI_DISCOVERY_URL: &str = "https://auth.x.ai/.well-known/openid-configuration";
const XAI_FALLBACK_TOKEN_URL: &str = "https://auth.x.ai/oauth2/token";
const XAI_OAUTH_CLIENT_ID: &str = "b1a00492-073a-47ea-816f-4c329264a828";
const TOKEN_EXPIRY_SKEW_SECONDS: i64 = 120;
const AUTH_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const AUTH_REQUEST_TIMEOUT: Duration = Duration::from_secs(20);
const CACHE_LOCK_TIMEOUT: Duration = Duration::from_secs(25);

#[derive(Debug, Clone)]
pub(super) struct PlatformAuthenticator {
    auth_file: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct AuthRecord {
    access_token: Option<String>,
    refresh_token: Option<String>,
    id_token: Option<String>,
    token_type: Option<String>,
    token_endpoint: Option<String>,
    // LiteLLM writes `time.time() + expires_in`, which is a JSON float. Keep
    // integer timestamps compatible too through serde's f64 number handling.
    expires_at: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct DiscoveryResponse {
    token_endpoint: String,
}

#[derive(Debug, Deserialize)]
struct OAuthTokenResponse {
    access_token: String,
    refresh_token: Option<String>,
    id_token: Option<String>,
    token_type: Option<String>,
    expires_in: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct OAuthErrorResponse {
    error: Option<String>,
}

impl PlatformAuthenticator {
    pub(super) fn new(auth_file: Option<PathBuf>) -> Self {
        Self { auth_file }
    }

    pub(super) async fn access_token_oauth(&self) -> Result<String, AuthError> {
        let mut record = self.read_auth_record()?;

        if let Some(access_token) = record.access_token.clone()
            && !token_expired(record.expires_at)
        {
            return Ok(access_token);
        }

        // Refresh tokens may rotate. Serialize the read-refresh-write
        // transaction across all processes that share this cache, then re-read
        // under the lock in case another process already refreshed it.
        let _cache_lock = match &self.auth_file {
            Some(path) => Some(acquire_cache_lock(path).await?),
            None => None,
        };
        record = self.read_auth_record()?;
        if let Some(access_token) = record.access_token.clone()
            && !token_expired(record.expires_at)
        {
            return Ok(access_token);
        }

        let refresh_token = record.refresh_token.clone().ok_or_else(sign_in_required)?;
        let token_endpoint = match record
            .token_endpoint
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            Some(endpoint) => validate_xai_endpoint(endpoint)?,
            None => discover_token_endpoint().await?,
        };

        let refreshed = refresh_tokens(&token_endpoint, &refresh_token).await?;
        self.write_auth_record(&refreshed)?;
        refreshed.access_token.ok_or_else(sign_in_required)
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
        write_private_file_atomically(path, &serde_json::to_vec_pretty(record)?)?;
        Ok(())
    }
}

async fn discover_token_endpoint() -> Result<String, AuthError> {
    let discovery = auth_http_client()?
        .get(XAI_DISCOVERY_URL)
        .header(reqwest::header::ACCEPT, "application/json")
        .send()
        .await?
        .error_for_status()?
        .json::<DiscoveryResponse>()
        .await?;

    validate_xai_endpoint(&discovery.token_endpoint)
}

async fn refresh_tokens(
    token_endpoint: &str,
    refresh_token: &str,
) -> Result<AuthRecord, AuthError> {
    let form = [
        ("grant_type", "refresh_token"),
        ("refresh_token", refresh_token),
        ("client_id", XAI_OAUTH_CLIENT_ID),
    ];

    let body = url::form_urlencoded::Serializer::new(String::new())
        .extend_pairs(form)
        .finish();

    let response = auth_http_client()?
        .post(token_endpoint)
        .header(reqwest::header::ACCEPT, "application/json")
        .header(
            reqwest::header::CONTENT_TYPE,
            "application/x-www-form-urlencoded",
        )
        .body(body)
        .send()
        .await?;

    let status = response.status();
    if !status.is_success() {
        let code = response
            .json::<OAuthErrorResponse>()
            .await
            .ok()
            .and_then(|body| body.error)
            .and_then(|code| sanitized_error_code(&code));
        return Err(AuthError::Message(match code {
            Some(code) => format!("xAI OAuth token refresh failed: {status} {code}"),
            None => format!("xAI OAuth token refresh failed: {status}"),
        }));
    }

    let tokens: OAuthTokenResponse = response.json().await?;
    Ok(build_auth_record(
        tokens,
        token_endpoint,
        Some(refresh_token.to_owned()),
    ))
}

fn auth_http_client() -> Result<reqwest::Client, AuthError> {
    Ok(reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(AUTH_CONNECT_TIMEOUT)
        .timeout(AUTH_REQUEST_TIMEOUT)
        .build()?)
}

fn sanitized_error_code(code: &str) -> Option<String> {
    let code = code.trim();
    if code.is_empty()
        || code.len() > 64
        || !code
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
    {
        return None;
    }
    Some(code.to_string())
}

fn build_auth_record(
    tokens: OAuthTokenResponse,
    token_endpoint: &str,
    previous_refresh_token: Option<String>,
) -> AuthRecord {
    AuthRecord {
        access_token: Some(tokens.access_token),
        refresh_token: tokens.refresh_token.or(previous_refresh_token),
        id_token: tokens.id_token,
        token_type: tokens.token_type,
        token_endpoint: Some(token_endpoint.to_string()),
        expires_at: Some(now_epoch_seconds() + tokens.expires_in.unwrap_or(3600) as f64),
    }
}

fn token_expired(expires_at: Option<f64>) -> bool {
    expires_at
        .map(|expires| expires <= now_epoch_seconds() + TOKEN_EXPIRY_SKEW_SECONDS as f64)
        .unwrap_or(true)
}

fn now_epoch_seconds() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

fn validate_xai_endpoint(endpoint: &str) -> Result<String, AuthError> {
    let parsed = url::Url::parse(endpoint)
        .map_err(|err| AuthError::Message(format!("invalid xAI OAuth endpoint: {err}")))?;
    if parsed.scheme() != "https" {
        return Err(AuthError::Message(
            "xAI OAuth endpoint must use https".into(),
        ));
    }
    let host = parsed.host_str().unwrap_or_default();
    if host != "x.ai" && !host.ends_with(".x.ai") {
        return Err(AuthError::Message(format!(
            "refusing non-xAI OAuth endpoint: {endpoint}"
        )));
    }
    Ok(endpoint.to_string())
}

fn ensure_parent_dir(path: &Path) -> Result<(), std::io::Error> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}

async fn acquire_cache_lock(auth_file: &Path) -> Result<std::fs::File, AuthError> {
    ensure_parent_dir(auth_file)?;
    let lock_path = auth_file.with_file_name(format!(
        ".{}.lock",
        auth_file
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("auth.json")
    ));
    let mut options = std::fs::OpenOptions::new();
    options.read(true).write(true).create(true).truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let file = options.open(lock_path)?;
    let started = std::time::Instant::now();
    loop {
        match FileExt::try_lock_exclusive(&file) {
            Ok(()) => return Ok(file),
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                if started.elapsed() >= CACHE_LOCK_TIMEOUT {
                    return Err(AuthError::Message(
                        "timed out waiting for the shared xAI OAuth cache lock".into(),
                    ));
                }
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
            Err(error) => return Err(error.into()),
        }
    }
}

fn write_private_file_atomically(path: &Path, contents: &[u8]) -> Result<(), std::io::Error> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("auth.json");

    let mut temp = None;
    for _ in 0..16 {
        let candidate = parent.join(format!(
            ".{file_name}.{}.{}.tmp",
            std::process::id(),
            fastrand::u64(..)
        ));
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        match options.open(&candidate) {
            Ok(file) => {
                temp = Some((candidate, file));
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }

    let (temp_path, mut temp_file) = temp.ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "could not allocate a unique xAI OAuth cache temporary file",
        )
    })?;
    let result = (|| {
        temp_file.write_all(contents)?;
        temp_file.sync_all()?;
        drop(temp_file);
        replace_file(&temp_path, path)?;
        set_private_file_permissions(path)?;
        Ok(())
    })();

    if result.is_err() {
        let _ = std::fs::remove_file(&temp_path);
    }
    result
}

#[cfg(not(windows))]
fn replace_file(source: &Path, destination: &Path) -> Result<(), std::io::Error> {
    // POSIX rename replaces the destination atomically when both paths are in
    // the same directory (which write_private_file_atomically guarantees).
    std::fs::rename(source, destination)
}

#[cfg(windows)]
fn replace_file(source: &Path, destination: &Path) -> Result<(), std::io::Error> {
    // std::fs::rename cannot replace an existing file on Windows. Move the
    // prior cache aside first, but restore it if installing the fully-written
    // replacement fails.
    match std::fs::rename(source, destination) {
        Ok(()) => Ok(()),
        Err(error)
            if destination.exists()
                && matches!(
                    error.kind(),
                    std::io::ErrorKind::AlreadyExists | std::io::ErrorKind::PermissionDenied
                ) =>
        {
            let backup = destination.with_file_name(format!(
                ".{}.{}.bak",
                destination
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("auth.json"),
                fastrand::u64(..)
            ));
            std::fs::rename(destination, &backup)?;
            match std::fs::rename(source, destination) {
                Ok(()) => {
                    let _ = std::fs::remove_file(backup);
                    Ok(())
                }
                Err(install_error) => match std::fs::rename(&backup, destination) {
                    Ok(()) => Err(install_error),
                    Err(restore_error) => Err(std::io::Error::new(
                        restore_error.kind(),
                        format!(
                            "failed to install refreshed xAI cache ({install_error}); \
                             prior cache remains at {} because restoration failed: {restore_error}",
                            backup.display()
                        ),
                    )),
                },
            }
        }
        Err(error) => Err(error),
    }
}

fn set_private_file_permissions(path: &Path) -> Result<(), std::io::Error> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

fn sign_in_required() -> AuthError {
    AuthError::Message(
        "xAI OAuth sign-in required. Provide an xAI OAuth auth.json cache before using this provider."
            .into(),
    )
}

impl Default for AuthRecord {
    fn default() -> Self {
        Self {
            access_token: None,
            refresh_token: None,
            id_token: None,
            token_type: None,
            token_endpoint: Some(XAI_FALLBACK_TOKEN_URL.to_string()),
            expires_at: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_fs::TempDir;

    #[test]
    fn validates_only_xai_https_endpoints() {
        assert_eq!(
            validate_xai_endpoint("https://auth.x.ai/oauth2/token").unwrap(),
            "https://auth.x.ai/oauth2/token"
        );
        assert!(validate_xai_endpoint("http://auth.x.ai/oauth2/token").is_err());
        assert!(validate_xai_endpoint("https://example.com/oauth2/token").is_err());
        assert!(validate_xai_endpoint("https://auth.x.ai.evil.test/token").is_err());
    }

    #[test]
    fn oauth_error_codes_are_bounded_and_non_secret() {
        assert_eq!(
            sanitized_error_code("invalid_grant"),
            Some("invalid_grant".to_string())
        );
        assert_eq!(sanitized_error_code("token=secret&refresh=secret"), None);
        assert_eq!(sanitized_error_code(&"x".repeat(65)), None);
    }

    #[tokio::test]
    async fn auth_http_client_does_not_follow_redirects() {
        use std::io::{Read, Write as _};
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0u8; 1024];
            let _ = stream.read(&mut request);
            stream
                .write_all(
                    b"HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:9/leak\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                )
                .unwrap();
        });

        let response = auth_http_client()
            .unwrap()
            .get(format!("http://{addr}/token"))
            .send()
            .await
            .unwrap();
        server.join().unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::FOUND);
    }

    #[test]
    fn cache_lock_file_is_exclusive_across_handles() {
        let temp = TempDir::new().unwrap();
        let lock_path = temp.path().join(".auth.json.lock");
        let first = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)
            .unwrap();
        let second = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)
            .unwrap();
        FileExt::lock_exclusive(&first).unwrap();
        assert_eq!(
            FileExt::try_lock_exclusive(&second).unwrap_err().kind(),
            std::io::ErrorKind::WouldBlock
        );
        FileExt::unlock(&first).unwrap();
    }

    #[test]
    fn refreshed_records_rotate_or_preserve_refresh_tokens() {
        let rotated = build_auth_record(
            OAuthTokenResponse {
                access_token: "new-access".into(),
                refresh_token: Some("new-refresh".into()),
                id_token: None,
                token_type: Some("Bearer".into()),
                expires_in: Some(60),
            },
            XAI_FALLBACK_TOKEN_URL,
            Some("old-refresh".into()),
        );
        assert_eq!(rotated.refresh_token.as_deref(), Some("new-refresh"));

        let preserved = build_auth_record(
            OAuthTokenResponse {
                access_token: "new-access".into(),
                refresh_token: None,
                id_token: None,
                token_type: Some("Bearer".into()),
                expires_in: Some(60),
            },
            XAI_FALLBACK_TOKEN_URL,
            Some("old-refresh".into()),
        );
        assert_eq!(preserved.refresh_token.as_deref(), Some("old-refresh"));
    }

    #[test]
    fn reads_fractional_litellm_expiry() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("auth.json");
        std::fs::write(
            &path,
            format!(
                r#"{{"access_token":"fresh","refresh_token":"refresh","expires_at":{}}}"#,
                now_epoch_seconds() + 3600.5
            ),
        )
        .unwrap();

        let auth = PlatformAuthenticator::new(Some(path));
        let record = auth.read_auth_record().unwrap();
        assert_eq!(record.access_token.as_deref(), Some("fresh"));
        assert!(!token_expired(record.expires_at));
    }

    #[test]
    fn cache_write_replaces_content_and_keeps_tokens_private() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("auth.json");
        std::fs::write(&path, b"stale").unwrap();
        let auth = PlatformAuthenticator::new(Some(path.clone()));
        let record = AuthRecord {
            access_token: Some("access".into()),
            refresh_token: Some("refresh".into()),
            id_token: None,
            token_type: Some("Bearer".into()),
            token_endpoint: Some(XAI_FALLBACK_TOKEN_URL.into()),
            expires_at: Some(now_epoch_seconds() + 3600.0),
        };

        auth.write_auth_record(&record).unwrap();
        let stored: AuthRecord = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(stored.access_token.as_deref(), Some("access"));
        assert_eq!(stored.refresh_token.as_deref(), Some("refresh"));

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
    }

    #[test]
    fn relative_auth_file_does_not_require_a_parent_directory() {
        ensure_parent_dir(Path::new("auth.json")).unwrap();
    }
}
