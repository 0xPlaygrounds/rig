//! Browser OAuth and private, refreshable Anthropic credentials.

use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use fs2::FileExt;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt},
    path::{Path, PathBuf},
    process::Stdio,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};
use url::Url;

const CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
const TOKEN_URL: &str = "https://api.anthropic.com/v1/oauth/token";
const SCOPES: &str = "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload";

#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    #[error("Anthropic OAuth is not logged in; run /anthropic-oauth login explicitly")]
    NotLoggedIn,
    #[error("OAuth credential storage is unavailable")]
    Storage,
    #[error("OAuth credential storage has unsafe permissions or file type")]
    UnsafeStorage,
    #[error("Another OAuth operation is in progress; try again")]
    Busy,
    #[error("OAuth network request failed")]
    Network,
    #[error("OAuth async runtime is unavailable")]
    Runtime,
    #[error("OAuth server rejected the request (HTTP {0}); log in again if authorization expired")]
    Http(u16),
    #[error("OAuth server returned invalid credentials")]
    InvalidCredentials,
    #[error("Could not open the browser for OAuth login")]
    Browser,
    #[error("Could not bind the OAuth loopback callback")]
    Callback,
    #[error("OAuth login timed out")]
    Timeout,
    #[error("OAuth authorization was denied")]
    Denied,
}

#[derive(Serialize, Deserialize)]
struct Credentials {
    access_token: String,
    refresh_token: String,
    expires_at: u64,
    account_id: Option<String>,
    org_id: Option<String>,
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    refresh_token: Option<String>,
    expires_in: u64,
    account: Option<Account>,
    organization: Option<Account>,
}

#[derive(Deserialize)]
struct Account {
    uuid: Option<String>,
}

fn now() -> Result<u64, AuthError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|v| v.as_secs())
        .map_err(|_| AuthError::Storage)
}

fn private_dir(path: &Path) -> Result<(), AuthError> {
    match fs::DirBuilder::new().mode(0o700).create(path) {
        Ok(()) => (),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => (),
        Err(_) => return Err(AuthError::Storage),
    }
    let metadata = fs::symlink_metadata(path).map_err(|_| AuthError::Storage)?;
    if !metadata.is_dir() || metadata.permissions().mode() & 0o077 != 0 {
        return Err(AuthError::UnsafeStorage);
    }
    Ok(())
}

fn directory() -> Result<PathBuf, AuthError> {
    let home = if let Some(path) = std::env::var_os("RIG_HOME") {
        PathBuf::from(path)
    } else {
        PathBuf::from(std::env::var_os("HOME").ok_or(AuthError::Storage)?).join(".rig")
    };
    // The parent is the existing agent home; only this plugin's directory is managed here.
    let path = home.join("anthropic-oauth");
    private_dir(&path)?;
    Ok(path)
}

fn validate_file(path: &Path) -> Result<(), AuthError> {
    let metadata = fs::symlink_metadata(path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            AuthError::NotLoggedIn
        } else {
            AuthError::Storage
        }
    })?;
    if !metadata.is_file() || metadata.permissions().mode() & 0o077 != 0 {
        return Err(AuthError::UnsafeStorage);
    }
    Ok(())
}

fn lock(dir: &Path) -> Result<File, AuthError> {
    let path = dir.join("lock");
    match OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&path)
    {
        Ok(_) => (),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => (),
        Err(_) => return Err(AuthError::Storage),
    }
    validate_file(&path)?;
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_err(|_| AuthError::Storage)?;
    file.try_lock_exclusive().map_err(|e| {
        if e.kind() == std::io::ErrorKind::WouldBlock {
            AuthError::Busy
        } else {
            AuthError::Storage
        }
    })?;
    Ok(file)
}

async fn wait_lock(dir: &Path) -> Result<File, AuthError> {
    tokio::time::timeout(Duration::from_secs(60), async {
        loop {
            match lock(dir) {
                Err(AuthError::Busy) => tokio::time::sleep(Duration::from_millis(50)).await,
                result => return result,
            }
        }
    })
    .await
    .map_err(|_| AuthError::Busy)?
}

fn load(dir: &Path) -> Result<Credentials, AuthError> {
    let path = dir.join("credentials.json");
    validate_file(&path)?;
    let mut data = Vec::new();
    File::open(path)
        .map_err(|_| AuthError::Storage)?
        .take(65537)
        .read_to_end(&mut data)
        .map_err(|_| AuthError::Storage)?;
    if data.len() > 65536 {
        return Err(AuthError::InvalidCredentials);
    }
    let credentials: Credentials =
        serde_json::from_slice(&data).map_err(|_| AuthError::InvalidCredentials)?;
    if credentials.access_token.is_empty() || credentials.refresh_token.is_empty() {
        return Err(AuthError::InvalidCredentials);
    }
    Ok(credentials)
}

fn save(dir: &Path, credentials: &Credentials) -> Result<(), AuthError> {
    let mut file = tempfile::NamedTempFile::new_in(dir).map_err(|_| AuthError::Storage)?;
    file.as_file()
        .set_permissions(fs::Permissions::from_mode(0o600))
        .map_err(|_| AuthError::Storage)?;
    let data = serde_json::to_vec(credentials).map_err(|_| AuthError::Storage)?;
    file.write_all(&data).map_err(|_| AuthError::Storage)?;
    file.as_file().sync_all().map_err(|_| AuthError::Storage)?;
    file.persist(dir.join("credentials.json"))
        .map_err(|_| AuthError::Storage)?;
    File::open(dir)
        .and_then(|f| f.sync_all())
        .map_err(|_| AuthError::Storage)?;
    Ok(())
}

fn client() -> Result<reqwest::Client, AuthError> {
    reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|_| AuthError::Network)
}

async fn response_json<T: serde::de::DeserializeOwned>(
    mut response: reqwest::Response,
) -> Result<T, AuthError> {
    let mut data = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(|_| AuthError::Network)? {
        if data.len().saturating_add(chunk.len()) > 65536 {
            return Err(AuthError::InvalidCredentials);
        }
        data.extend_from_slice(&chunk);
    }
    serde_json::from_slice(&data).map_err(|_| AuthError::InvalidCredentials)
}

struct OAuthClient {
    http: reqwest::Client,
    #[cfg(test)]
    token_url: String,
    #[cfg(test)]
    bootstrap_url: String,
}

impl OAuthClient {
    fn production() -> Result<Self, AuthError> {
        Ok(Self {
            http: client()?,
            #[cfg(test)]
            token_url: TOKEN_URL.into(),
            #[cfg(test)]
            bootstrap_url: BOOTSTRAP_URL.into(),
        })
    }

    fn token_url(&self) -> &str {
        #[cfg(test)]
        {
            &self.token_url
        }
        #[cfg(not(test))]
        {
            TOKEN_URL
        }
    }

    fn bootstrap_url(&self) -> &str {
        #[cfg(test)]
        {
            &self.bootstrap_url
        }
        #[cfg(not(test))]
        {
            BOOTSTRAP_URL
        }
    }
}

const BOOTSTRAP_URL: &str =
    "https://api.anthropic.com/api/claude_cli/bootstrap?entrypoint=cli&model=claude-opus-4-8";

async fn exchange(
    client: &OAuthClient,
    body: serde_json::Value,
    refresh: bool,
) -> Result<TokenResponse, AuthError> {
    let mut request = client.http.post(client.token_url()).json(&body);
    if refresh {
        request = request.header("anthropic-beta", "oauth-2025-04-20").header(
            "User-Agent",
            "anthropic-sdk-typescript/0.112.1 userOAuthProvider",
        );
    }
    let response = request.send().await.map_err(|_| AuthError::Network)?;
    if !response.status().is_success() {
        return Err(AuthError::Http(response.status().as_u16()));
    }
    response_json(response).await
}

fn merge(
    response: TokenResponse,
    previous: Option<Credentials>,
    time: u64,
) -> Result<Credentials, AuthError> {
    let refresh_token = response
        .refresh_token
        .filter(|v| !v.is_empty())
        .or_else(|| previous.as_ref().map(|v| v.refresh_token.clone()))
        .ok_or(AuthError::InvalidCredentials)?;
    if response.access_token.is_empty() || response.expires_in == 0 {
        return Err(AuthError::InvalidCredentials);
    }
    Ok(Credentials {
        access_token: response.access_token,
        refresh_token,
        expires_at: time.saturating_add(response.expires_in.saturating_sub(300)),
        account_id: response
            .account
            .and_then(|v| v.uuid)
            .filter(|v| !v.is_empty())
            .or_else(|| previous.as_ref().and_then(|v| v.account_id.clone())),
        org_id: match previous {
            Some(v) => v.org_id,
            None => response.organization.and_then(|v| v.uuid),
        },
    })
}

async fn fill_account(client: &OAuthClient, credentials: &mut Credentials, login: bool) {
    if credentials.account_id.is_some() && (!login || credentials.org_id.is_some()) {
        return;
    }
    let Ok(response) = client
        .http
        .get(client.bootstrap_url())
        .bearer_auth(&credentials.access_token)
        .header("anthropic-beta", "oauth-2025-04-20")
        .header("User-Agent", "claude-code/2.1.280")
        .header("Accept", "application/json, text/plain, */*")
        .header("Content-Type", "application/json")
        .send()
        .await
    else {
        return;
    };
    if !response.status().is_success() {
        return;
    }
    let Ok(value) = response_json::<serde_json::Value>(response).await else {
        return;
    };
    merge_identity(credentials, &value, login);
}

fn merge_identity(credentials: &mut Credentials, value: &serde_json::Value, login: bool) {
    let field = |path: &str| {
        value
            .pointer(path)
            .and_then(|v| v.as_str())
            .filter(|v| !v.is_empty())
            .map(str::to_owned)
    };
    if credentials.account_id.is_none() {
        credentials.account_id = field("/oauth_account/account_uuid");
    }
    if login && credentials.org_id.is_none() {
        credentials.org_id = field("/oauth_account/organization_uuid");
    }
}

fn random_verifier() -> String {
    let mut bytes = [0u8; 96];
    rand::rng().fill_bytes(&mut bytes);
    URL_SAFE_NO_PAD.encode(bytes)
}

fn random_state() -> String {
    let mut bytes = [0u8; 16];
    rand::rng().fill_bytes(&mut bytes);
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn authorization_url(redirect: &str, state: &str, verifier: &str) -> Result<Url, AuthError> {
    let mut url =
        Url::parse("https://claude.ai/oauth/authorize").map_err(|_| AuthError::Callback)?;
    url.query_pairs_mut().extend_pairs([
        ("code", "true"),
        ("client_id", CLIENT_ID),
        ("response_type", "code"),
        ("redirect_uri", redirect),
        ("scope", SCOPES),
        ("state", state),
        (
            "code_challenge",
            URL_SAFE_NO_PAD
                .encode(Sha256::digest(verifier.as_bytes()))
                .as_str(),
        ),
        ("code_challenge_method", "S256"),
    ]);
    Ok(url)
}

fn callback_code(target: &str, state: &str) -> Option<Result<String, AuthError>> {
    if !target.starts_with("/callback?") {
        return None;
    }
    let url = Url::parse(&format!("http://localhost{target}")).ok()?;
    let pairs: Vec<_> = url.query_pairs().collect();
    let states: Vec<_> = pairs.iter().filter(|(k, _)| k == "state").collect();
    if states.len() != 1 || states.first()?.1 != state {
        return None;
    }
    if pairs.iter().any(|(k, _)| k == "error") {
        return Some(Err(AuthError::Denied));
    }
    let codes: Vec<_> = pairs
        .iter()
        .filter(|(k, v)| k == "code" && !v.is_empty())
        .collect();
    if codes.len() != 1 {
        return None;
    }
    Some(Ok(codes.first()?.1.to_string()))
}

async fn receive_code(
    listener: &TcpListener,
    ipv6: Option<&TcpListener>,
    state: &str,
) -> Result<String, AuthError> {
    tokio::time::timeout(Duration::from_secs(300), async {
        loop {
            let accepted = if let Some(ipv6) = ipv6 {
                tokio::select! {
                    accepted = listener.accept() => accepted,
                    accepted = ipv6.accept() => accepted,
                }
            } else {
                listener.accept().await
            };
            let (mut stream, _) = accepted.map_err(|_| AuthError::Callback)?;
            let request = tokio::time::timeout(Duration::from_secs(5), async {
                let mut data = Vec::new();
                loop {
                    let mut chunk = [0u8; 1024];
                    let count = stream.read(&mut chunk).await.ok()?;
                    if count == 0 { return None; }
                    data.extend_from_slice(chunk.get(..count)?);
                    if data.len() > 8192 { return None; }
                    if data.windows(4).any(|w| w == b"\r\n\r\n") { return String::from_utf8(data).ok(); }
                }
            }).await;
            let result = request.ok().flatten().and_then(|request| {
                let mut words = request.lines().next()?.split_whitespace();
                if words.next()? != "GET" { return None; }
                callback_code(words.next()?, state)
            });
            let reply = if result.is_some() { "HTTP/1.1 200 OK\r\nContent-Length: 35\r\nConnection: close\r\nCache-Control: no-store\r\nContent-Type: text/plain\r\n\r\nReturn to Rig to finish OAuth login." } else { "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n" };
            let _ = tokio::time::timeout(Duration::from_secs(2), stream.write_all(reply.as_bytes())).await;
            if let Some(result) = result { return result; }
        }
    }).await.map_err(|_| AuthError::Timeout)?
}

async fn callback_listeners(
    preferred_port: u16,
) -> Result<(TcpListener, Option<TcpListener>), AuthError> {
    let mut port = preferred_port;
    for _ in 0..6 {
        let listener = match TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, port)).await {
            Ok(listener) => listener,
            Err(_) if port != 0 => {
                port = 0;
                continue;
            }
            Err(_) => return Err(AuthError::Callback),
        };
        let bound_port = listener
            .local_addr()
            .map_err(|_| AuthError::Callback)?
            .port();
        match TcpListener::bind((std::net::Ipv6Addr::LOCALHOST, bound_port)).await {
            Ok(ipv6) => return Ok((listener, Some(ipv6))),
            Err(error) if error.kind() == std::io::ErrorKind::AddrInUse => {
                // localhost can resolve to IPv6 first; never advertise another server's port.
                drop(listener);
                port = 0;
            }
            Err(_) => return Ok((listener, None)),
        }
    }
    Err(AuthError::Callback)
}

/// Opens a browser for an explicit PKCE login and stores credentials privately.
pub async fn login() -> Result<(), AuthError> {
    let dir = directory()?;
    let _lock = lock(&dir)?;
    let (listener, ipv6) = callback_listeners(54545).await?;
    let redirect = format!(
        "http://localhost:{}/callback",
        listener
            .local_addr()
            .map_err(|_| AuthError::Callback)?
            .port()
    );
    let verifier = random_verifier();
    let state = random_state();
    let url = authorization_url(&redirect, &state, &verifier)?;
    let status = tokio::time::timeout(
        Duration::from_secs(15),
        tokio::process::Command::new("xdg-open")
            .arg(url.as_str())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .status(),
    )
    .await
    .map_err(|_| AuthError::Browser)?
    .map_err(|_| AuthError::Browser)?;
    if !status.success() {
        return Err(AuthError::Browser);
    }
    let code = receive_code(&listener, ipv6.as_ref(), &state).await?;
    let client = OAuthClient::production()?;
    let credentials = exchange_code(&client, &code, &state, &verifier, &redirect).await?;
    save(&dir, &credentials)
}

async fn exchange_code(
    client: &OAuthClient,
    code: &str,
    state: &str,
    verifier: &str,
    redirect: &str,
) -> Result<Credentials, AuthError> {
    let response = exchange(client, serde_json::json!({"grant_type":"authorization_code", "client_id":CLIENT_ID, "code":code, "state":state, "code_verifier":verifier, "redirect_uri":redirect}), false).await?;
    let mut credentials = merge(response, None, now()?)?;
    fill_account(client, &mut credentials, true).await;
    Ok(credentials)
}

/// Returns a matching token and account UUID under one refresh lock. Never starts login.
pub async fn credentials_for_request() -> Result<(String, Option<String>), AuthError> {
    let dir = directory()?;
    credentials_in(&dir, &OAuthClient::production()?).await
}

async fn credentials_in(
    dir: &Path,
    client: &OAuthClient,
) -> Result<(String, Option<String>), AuthError> {
    // Model calls can be polled by Bevy without a Tokio reactor. Bind both
    // refresh I/O and lock-wait timers on every poll, as Rig's HTTP transport does.
    rig_http::runtime::bind(async {
        let _lock = wait_lock(dir).await?;
        let credentials = load(dir)?;
        if credentials.expires_at > now()? {
            return Ok((credentials.access_token, credentials.account_id));
        }
        let response = exchange(client, serde_json::json!({"grant_type":"refresh_token", "client_id":CLIENT_ID, "refresh_token":credentials.refresh_token}), true).await?;
        let mut credentials = merge(response, Some(credentials), now()?)?;
        fill_account(client, &mut credentials, false).await;
        save(dir, &credentials)?;
        Ok((credentials.access_token, credentials.account_id))
    })
    .map_err(|_| AuthError::Runtime)?
    .await
}

/// Deletes local credentials. Fails if login or refresh is currently active.
pub fn logout() -> Result<(), AuthError> {
    logout_in(&directory()?)
}

fn logout_in(dir: &Path) -> Result<(), AuthError> {
    let _lock = lock(dir)?;
    match fs::remove_file(dir.join("credentials.json")) {
        Ok(()) => File::open(dir)
            .and_then(|f| f.sync_all())
            .map_err(|_| AuthError::Storage),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(_) => Err(AuthError::Storage),
    }
}

#[cfg(test)]
mod tests;
