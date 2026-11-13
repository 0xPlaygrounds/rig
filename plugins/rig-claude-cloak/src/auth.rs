//! Browser OAuth and private, refreshable Anthropic credentials.

use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use rand::RngCore;
use rig_harness::harness_protocol::without_secrets;
use rig_harness::rig_core::error::{BoxError, retryable_status};
use rig_harness::rig_core::providers::chatgpt::auth::{FileStoreError, FileTokenStore, TokenStore};
use rig_harness::rig_core::providers::internal::loopback::{Loopback, LoopbackError, LoopbackHost};
use rig_harness::rig_tools::process::{self, Process};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    os::unix::fs::{DirBuilderExt, PermissionsExt},
    path::{Path, PathBuf},
    process::Stdio,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use url::Url;

const CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
const TOKEN_URL: &str = "https://api.anthropic.com/v1/oauth/token";
/// The callback port registered with the provider.
const CALLBACK_PORT: u16 = 54545;
/// How long a login waits for the browser to come back.
const LOGIN_WAIT: Duration = Duration::from_secs(300);
const SCOPES: &str = "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload";

#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    #[error("Anthropic OAuth is not logged in; run /claude-cloak login explicitly")]
    NotLoggedIn,
    #[error("OAuth credential storage is unavailable")]
    Storage,
    #[error("OAuth credential storage has unsafe permissions or file type")]
    UnsafeStorage,
    #[error("Another OAuth operation is in progress; try again")]
    Busy,
    #[error("OAuth network request failed")]
    Network,
    #[error("Could not build the OAuth HTTP client")]
    Client,
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

impl AuthError {
    /// Whether trying again may pass: another refresh held the lock, the
    /// network failed, or the server answered a status the retry table
    /// retries. Not being logged in, or any local failure, never does.
    pub fn is_transient(&self) -> bool {
        match self {
            Self::Busy | Self::Network => true,
            Self::Http(status) => retryable_status(Some(*status)),
            _ => false,
        }
    }
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

/// The record the credentials are kept under, `credentials.json` in this
/// plugin's directory.
const KEY: &str = "credentials";

/// This plugin's directory in the agent's home.
const DIRECTORY: &str = "anthropic-oauth";

/// This plugin's directory under the agent's home, created private when
/// missing.
fn directory(home: &Path) -> Result<PathBuf, AuthError> {
    let path = home.join(DIRECTORY);
    private_dir(&path)?;
    Ok(path)
}

fn private_dir(path: &Path) -> Result<(), AuthError> {
    match fs::DirBuilder::new().mode(0o700).create(path) {
        Ok(()) => (),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => (),
        Err(_) => return Err(AuthError::Storage),
    }
    checked_dir(path)
}

/// Refuses a credential directory that is a symlink or that others may
/// access, since they could replace the credential or lock files in it. A
/// missing directory is `NotLoggedIn`.
fn checked_dir(path: &Path) -> Result<(), AuthError> {
    let metadata = fs::symlink_metadata(path).map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => AuthError::NotLoggedIn,
        _ => AuthError::Storage,
    })?;
    if !metadata.is_dir() || metadata.permissions().mode() & 0o077 != 0 {
        return Err(AuthError::UnsafeStorage);
    }
    Ok(())
}

/// The store's refusal as this plugin reports it: a lock held past the wait
/// is busy, and a file the store will not read is unsafe.
fn storage(error: BoxError) -> AuthError {
    match error.downcast_ref::<FileStoreError>() {
        Some(FileStoreError::Busy { .. }) => AuthError::Busy,
        Some(FileStoreError::TooLarge { .. }) => AuthError::InvalidCredentials,
        Some(_) => AuthError::UnsafeStorage,
        None => AuthError::Storage,
    }
}

async fn load(store: &FileTokenStore) -> Result<Credentials, AuthError> {
    let record = store
        .load(KEY)
        .await
        .map_err(storage)?
        .ok_or(AuthError::NotLoggedIn)?;
    let credentials: Credentials =
        serde_json::from_str(&record).map_err(|_| AuthError::InvalidCredentials)?;
    if credentials.access_token.is_empty() || credentials.refresh_token.is_empty() {
        return Err(AuthError::InvalidCredentials);
    }
    Ok(credentials)
}

async fn save(store: &FileTokenStore, credentials: &Credentials) -> Result<(), AuthError> {
    let record = serde_json::to_string(credentials).map_err(|_| AuthError::Storage)?;
    store.save(KEY, &record).await.map_err(storage)
}

fn client() -> Result<reqwest::Client, AuthError> {
    reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|_| AuthError::Client)
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
        request = request
            .header("anthropic-beta", crate::profile::OAUTH_BETA)
            .header(
                "User-Agent",
                format!(
                    "anthropic-sdk-typescript/{} userOAuthProvider",
                    crate::profile::SDK
                ),
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
        .header("anthropic-beta", crate::profile::OAUTH_BETA)
        .header(
            "User-Agent",
            format!("claude-code/{}", crate::profile::VERSION),
        )
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

async fn save_and_fill_account(
    store: &FileTokenStore,
    client: &OAuthClient,
    credentials: &mut Credentials,
    login: bool,
) -> Result<(), AuthError> {
    // Persist rotated tokens before optional network I/O can be cancelled.
    save(store, credentials).await?;
    let identity = (credentials.account_id.clone(), credentials.org_id.clone());
    fill_account(client, credentials, login).await;
    if identity != (credentials.account_id.clone(), credentials.org_id.clone()) {
        save(store, credentials).await?;
    }
    Ok(())
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

/// The callback registered with the provider: `localhost` on IPv4 and IPv6,
/// at `port`, or at a port the system picks when that is taken.
fn loopback(port: u16) -> Loopback {
    Loopback::new("Claude", LoopbackHost::Localhost, [port, 0], "/callback")
}

/// The listener's failure as this plugin reports it.
fn listen_error(error: LoopbackError) -> AuthError {
    match error {
        LoopbackError::TimedOut => AuthError::Timeout,
        LoopbackError::Refused(_) => AuthError::Denied,
        _ => AuthError::Callback,
    }
}

fn browser_command(url: &Url) -> std::process::Command {
    #[cfg(target_os = "linux")]
    let opener = "xdg-open";
    #[cfg(target_os = "macos")]
    let opener = "/usr/bin/open";
    let mut command = std::process::Command::new(opener);
    command
        .arg(url.as_str())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    // A browser it starts outlives rig with the user's own editor and pager.
    for (name, _) in process::NON_INTERACTIVE {
        match std::env::var_os(name) {
            Some(value) => command.env(name, value),
            None => command.env_remove(name),
        };
    }
    without_secrets(&mut command);
    command
}

/// Runs `opener`, which opens the browser: whether it said it did within
/// `wait`. It is handed over as it starts, so the browser it starts, and
/// an opener that waits for that browser, stay as rig goes on or exits.
async fn open_browser(opener: std::process::Command, wait: Duration) -> Result<(), AuthError> {
    let mut opener = Process::start_with_stdio(opener).map_err(|_| AuthError::Browser)?;
    opener.release();
    let waited = tokio::time::timeout(wait, async {
        loop {
            match opener.try_wait() {
                Ok(None) => tokio::time::sleep(Duration::from_millis(20)).await,
                ended => return ended,
            }
        }
    });
    match waited.await {
        Ok(Ok(Some(status))) if status.success() => Ok(()),
        Err(_) => {
            // Still waiting for its browser: reaped once it is done.
            let reaper = std::thread::Builder::new();
            reaper
                .spawn(move || opener.wait_for(Duration::MAX).ok())
                .ok();
            Err(AuthError::Browser)
        }
        _ => Err(AuthError::Browser),
    }
}

/// Opens a browser for an explicit PKCE login and stores credentials
/// privately. The credentials are held only while the new ones are saved,
/// not while the user is in the browser. The browser's page says how the
/// login ended.
pub async fn login(home: &Path) -> Result<(), AuthError> {
    let state = random_state();
    let listener = loopback(CALLBACK_PORT)
        .listen(state.clone(), Instant::now() + LOGIN_WAIT)
        .map_err(listen_error)?;
    let redirect = listener.redirect_uri().to_owned();
    let verifier = random_verifier();
    let url = authorization_url(&redirect, &state, &verifier)?;
    open_browser(browser_command(&url), Duration::from_secs(15)).await?;
    let received = listener.code().await.map_err(listen_error)?;
    let exchanged = async {
        let client = OAuthClient::production()?;
        let dir = directory(home)?;
        exchange_code(&dir, &client, received.code(), &state, &verifier, &redirect).await
    }
    .await;
    match &exchanged {
        Ok(_) => received.succeeded(),
        Err(error) => received.failed(&error.to_string()),
    }
    exchanged.map(drop)
}

async fn exchange_code(
    dir: &Path,
    client: &OAuthClient,
    code: &str,
    state: &str,
    verifier: &str,
    redirect: &str,
) -> Result<Credentials, AuthError> {
    let response = exchange(client, serde_json::json!({"grant_type":"authorization_code", "client_id":CLIENT_ID, "code":code, "state":state, "code_verifier":verifier, "redirect_uri":redirect}), false).await?;
    let mut credentials = merge(response, None, now()?)?;
    let store = FileTokenStore::new(dir);
    let _lock = store.lock(KEY).await.map_err(storage)?;
    save_and_fill_account(&store, client, &mut credentials, true).await?;
    Ok(credentials)
}

/// Returns a matching token and account UUID under one refresh lock. Never starts login.
pub async fn credentials_for_request(home: &Path) -> Result<(String, Option<String>), AuthError> {
    credentials_in(&directory(home)?, &OAuthClient::production()?).await
}

async fn credentials_in(
    dir: &Path,
    client: &OAuthClient,
) -> Result<(String, Option<String>), AuthError> {
    // Model calls can be polled by Bevy without a Tokio reactor. Bind both
    // refresh I/O and lock-wait timers on every poll, as Rig's HTTP transport does.
    rig_http::runtime::bind(async {
        let store = FileTokenStore::new(dir);
        let _lock = store.lock(KEY).await.map_err(storage)?;
        let credentials = load(&store).await?;
        if credentials.expires_at > now()? {
            return Ok((credentials.access_token, credentials.account_id));
        }
        let response = exchange(client, serde_json::json!({"grant_type":"refresh_token", "client_id":CLIENT_ID, "refresh_token":credentials.refresh_token}), true).await?;
        let mut credentials = merge(response, Some(credentials), now()?)?;
        save_and_fill_account(&store, client, &mut credentials, false).await?;
        Ok((credentials.access_token, credentials.account_id))
    })
    .map_err(|_| AuthError::Runtime)?
    .await
}

/// Whether `home` holds the credentials a login saved, read as a request
/// reads them, which may still need a refresh: what makes this plugin's
/// models usable; else why not.
pub fn signed_in(home: &Path) -> Result<(), AuthError> {
    let dir = home.join(DIRECTORY);
    checked_dir(&dir)?;
    let store = FileTokenStore::new(dir);
    // The store's file reads never wait, so this completes on first poll.
    match bevy_tasks::block_on(store.load(KEY)).map_err(storage)? {
        Some(_) => Ok(()),
        None => Err(AuthError::NotLoggedIn),
    }
}

/// Deletes local credentials. Fails at once if a login's save or a refresh
/// is currently active.
pub fn logout(home: &Path) -> Result<(), AuthError> {
    logout_in(&directory(home)?)
}

fn logout_in(dir: &Path) -> Result<(), AuthError> {
    let store = FileTokenStore::new(dir).with_lock_wait(Duration::ZERO);
    // A zero wait never sleeps, so this completes on first poll.
    bevy_tasks::block_on(async {
        let _lock = store.lock(KEY).await.map_err(storage)?;
        store.delete(KEY).await.map_err(storage)
    })
}

#[cfg(test)]
mod tests;
