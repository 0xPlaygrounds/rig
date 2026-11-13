use super::*;
use rig_harness::rig_core::providers::chatgpt::auth::StoreLock;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt};

mod http;
mod off_runtime;
mod tls;

#[test]
fn browser_opener_receives_the_authorization_url_as_one_argument() -> anyhow::Result<()> {
    let url = authorization_url(
        "http://localhost:54545/callback",
        "state-with-&-and-'",
        "synthetic-challenge",
    )?;
    let command = browser_command(&url);
    #[cfg(target_os = "linux")]
    anyhow::ensure!(command.get_program() == "xdg-open");
    #[cfg(target_os = "macos")]
    anyhow::ensure!(command.get_program() == "/usr/bin/open");
    anyhow::ensure!(command.get_args().collect::<Vec<_>>() == [std::ffi::OsStr::new(url.as_str())]);
    Ok(())
}

/// The secret this test runs a copy of itself with.
const PROBE: &str = "RIG_SECRET_PROBE";

/// The browser outlives rig, so it starts without any secret.
#[test]
#[allow(clippy::disallowed_methods, reason = "test fixtures")]
fn the_browser_gets_no_secret() -> anyhow::Result<()> {
    if std::env::var_os(PROBE).is_none() {
        let name = "auth::tests::the_browser_gets_no_secret";
        let output = std::process::Command::new(std::env::current_exe()?)
            .args(["--exact", name, "--test-threads", "1"])
            .env(PROBE, "hunter2")
            .output()?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        anyhow::ensure!(
            output.status.success() && stdout.contains("1 passed"),
            "the run with a secret failed: {stdout}"
        );
        return Ok(());
    }
    let url = authorization_url("http://localhost:54545/callback", "state", "challenge")?;
    let command = browser_command(&url);
    let removed = command
        .get_envs()
        .any(|(name, value)| name == PROBE && value.is_none());
    anyhow::ensure!(removed, "{command:?}");
    Ok(())
}

/// An opener that waits for the browser it started past its time fails
/// the login, and the browser stays.
#[cfg(target_os = "linux")]
#[tokio::test]
#[allow(clippy::disallowed_methods, reason = "test fixtures")]
async fn a_slow_opener_never_takes_its_browser_with_it() -> anyhow::Result<()> {
    let browser = std::env::temp_dir().join(format!("rig-browser-{}", std::process::id()));
    let mut opener = std::process::Command::new("sh");
    let script = format!("sleep 30 & echo $! > '{}'; wait", browser.display());
    opener.args(["-c", &script]);
    let opened = open_browser(opener, Duration::from_millis(500)).await;
    tokio::time::sleep(Duration::from_millis(100)).await;
    let pid = fs::read_to_string(&browser).unwrap_or_default();
    fs::remove_file(&browser).ok();
    let stat = fs::read_to_string(format!("/proc/{}/stat", pid.trim())).unwrap_or_default();
    let alive = !stat.is_empty()
        && !stat
            .rsplit(')')
            .next()
            .unwrap_or_default()
            .trim_start()
            .starts_with('Z');
    if let Ok(pid) = pid.trim().parse::<i32>() {
        std::process::Command::new("kill")
            .arg(pid.to_string())
            .status()
            .ok();
    }
    anyhow::ensure!(matches!(opened, Err(AuthError::Browser)));
    anyhow::ensure!(alive, "the browser died with its opener");
    Ok(())
}

/// Saves `credentials` in `dir` as the plugin does.
fn seed(dir: &Path, credentials: &Credentials) -> Result<(), AuthError> {
    bevy_tasks::block_on(save(&FileTokenStore::new(dir), credentials))
}

/// The credentials saved in `dir`, read as the plugin reads them.
fn saved(dir: &Path) -> Result<Credentials, AuthError> {
    bevy_tasks::block_on(load(&FileTokenStore::new(dir)))
}

/// The credential lock in `dir`, refused at once as busy when held.
fn hold(dir: &Path) -> Result<StoreLock, AuthError> {
    let store = FileTokenStore::new(dir).with_lock_wait(Duration::ZERO);
    bevy_tasks::block_on(store.lock(KEY)).map_err(storage)
}

fn credentials() -> Credentials {
    Credentials {
        access_token: "old-access".into(),
        refresh_token: "old-refresh".into(),
        expires_at: 0,
        account_id: Some("old-account".into()),
        org_id: Some("original-org".into()),
    }
}

#[test]
fn pkce_matches_rfc7636_vector_and_omp_authorization_contract() -> anyhow::Result<()> {
    let url = (authorization_url(
        "http://localhost:54545/callback",
        "state",
        "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk",
    ))?;
    let parameters: std::collections::HashMap<_, _> = url.query_pairs().collect();
    anyhow::ensure!((url.host_str()) == (Some("claude.ai")));
    anyhow::ensure!(
        (parameters.get("code_challenge").map(|v| v.as_ref()))
            == (Some("E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"))
    );
    anyhow::ensure!((parameters.get("scope").map(|v| v.as_ref())) == (Some(SCOPES)));
    anyhow::ensure!((parameters.get("client_id").map(|v| v.as_ref())) == (Some(CLIENT_ID)));
    anyhow::ensure!((parameters.get("code").map(|v| v.as_ref())) == (Some("true")));
    Ok(())
}

#[test]
fn callback_rejects_wrong_state_path_and_duplicate_parameters() -> anyhow::Result<()> {
    for target in [
        "/callback?state=wrong&code=secret",
        "/wrong?state=expected&code=secret",
        "/callback?state=expected&state=expected&code=secret",
        "/callback?state=expected&code=a&code=b",
        "/callback?state=expected&code=",
        "https://evil.test/callback?state=expected&code=a",
    ] {
        anyhow::ensure!(callback_code(target, "expected").is_none());
    }
    anyhow::ensure!(matches!(
        callback_code("/callback?state=expected&error=access_denied", "expected"),
        Some(Err(AuthError::Denied))
    ));
    anyhow::ensure!(
        matches!(callback_code("/callback?state=expected&code=abc%2Bdef", "expected"), Some(Ok(code)) if code == "abc+def")
    );
    Ok(())
}

#[test]
fn refresh_rotates_tokens_preserves_organization_and_applies_skew() -> anyhow::Result<()> {
    let response = TokenResponse {
        access_token: "new-access".into(),
        refresh_token: Some("new-refresh".into()),
        expires_in: 3600,
        account: Some(Account {
            uuid: Some("new-account".into()),
        }),
        organization: Some(Account {
            uuid: Some("drifted-org".into()),
        }),
    };
    let value = (merge(response, Some(credentials()), 1000))?;
    anyhow::ensure!((value.access_token) == ("new-access"));
    anyhow::ensure!((value.refresh_token) == ("new-refresh"));
    anyhow::ensure!((value.org_id.as_deref()) == (Some("original-org")));
    anyhow::ensure!((value.account_id.as_deref()) == (Some("new-account")));
    anyhow::ensure!((value.expires_at) == (4300));
    Ok(())
}

#[test]
fn refresh_can_retain_refresh_token_but_login_requires_one() -> anyhow::Result<()> {
    let response = || TokenResponse {
        access_token: "access".into(),
        refresh_token: None,
        expires_in: 30,
        account: None,
        organization: None,
    };
    anyhow::ensure!(matches!(
        merge(response(), None, 0),
        Err(AuthError::InvalidCredentials)
    ));
    let value = (merge(response(), Some(credentials()), 100))?;
    anyhow::ensure!((value.refresh_token) == ("old-refresh"));
    anyhow::ensure!((value.expires_at) == (100));
    Ok(())
}

#[test]
fn private_storage_roundtrip_atomic_replacement_and_exclusive_lock() -> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    let dir = root.path().join("oauth");
    let guard = hold(&dir)?;
    anyhow::ensure!(matches!(hold(&dir), Err(AuthError::Busy)));
    anyhow::ensure!(matches!(logout_in(&dir), Err(AuthError::Busy)));
    seed(&dir, &credentials())?;
    let mut value = saved(&dir)?;
    anyhow::ensure!(value.access_token == "old-access");
    value.access_token = "replacement".into();
    seed(&dir, &value)?;
    anyhow::ensure!(saved(&dir)?.access_token == "replacement");
    let mode = fs::metadata(dir.join("credentials.json"))?
        .permissions()
        .mode();
    anyhow::ensure!(mode & 0o777 == 0o600);
    anyhow::ensure!(fs::metadata(&dir)?.permissions().mode() & 0o777 == 0o700);
    drop(guard);
    let _guard = hold(&dir)?;
    Ok(())
}

/// A credential file others may read is narrowed to the owner and read; one
/// behind a symlink is refused.
#[test]
fn storage_narrows_a_shared_file_and_rejects_symlinks() -> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    let dir = root.path().join("oauth");
    seed(&dir, &credentials())?;
    let path = dir.join("credentials.json");
    fs::set_permissions(&path, fs::Permissions::from_mode(0o644))?;
    anyhow::ensure!(saved(&dir)?.access_token == "old-access");
    anyhow::ensure!(fs::metadata(&path)?.permissions().mode() & 0o777 == 0o600);
    fs::remove_file(&path)?;
    std::os::unix::fs::symlink(root.path().join("missing"), &path)?;
    anyhow::ensure!(matches!(saved(&dir), Err(AuthError::UnsafeStorage)));
    Ok(())
}

#[test]
fn loopback_callback_ignores_invalid_state_and_accepts_valid_code() -> anyhow::Result<()> {
    let runtime = (tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build())?;
    runtime.block_on(async {
        let listener = (TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).await)?;
        let address = (listener.local_addr())?;
        let receiver = tokio::spawn(async move { receive_code(&listener, None, "expected").await });
        for (state, status) in [("wrong", "400"), ("expected", "200")] {
            let mut stream = (tokio::net::TcpStream::connect(address).await)?;
            (stream
                .write_all(
                    format!(
                        "GET /callback?state={state}&code=code HTTP/1.1\r\nHost: localhost\r\n\r\n"
                    )
                    .as_bytes(),
                )
                .await)?;
            let mut response = String::new();
            (stream.read_to_string(&mut response).await)?;
            anyhow::ensure!(response.starts_with(&format!("HTTP/1.1 {status}")));
            anyhow::ensure!(!response.contains("expected"));
            anyhow::ensure!(!response.contains("code="));
        }
        anyhow::ensure!((((receiver.await)?)?) == ("code"));
        Ok::<_, anyhow::Error>(())
    })?;
    Ok(())
}

#[test]
fn callback_accepts_ipv6_when_available() -> anyhow::Result<()> {
    let runtime = (tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build())?;
    runtime.block_on(async {
        let listener = (TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).await)?;
        let port = (listener.local_addr())?.port();
        let Ok(ipv6) = TcpListener::bind((std::net::Ipv6Addr::LOCALHOST, port)).await else {
            return Ok(());
        };
        let address = (ipv6.local_addr())?;
        let receiver =
            tokio::spawn(async move { receive_code(&listener, Some(&ipv6), "expected").await });
        let mut stream = (tokio::net::TcpStream::connect(address).await)?;
        (stream
            .write_all(
                b"GET /callback?state=expected&code=ipv6 HTTP/1.1\r\nHost: localhost\r\n\r\n",
            )
            .await)?;
        anyhow::ensure!((((receiver.await)?)?) == ("ipv6"));
        Ok::<_, anyhow::Error>(())
    })?;
    Ok(())
}

#[test]
fn callback_falls_back_when_only_ipv6_preferred_port_is_occupied() -> anyhow::Result<()> {
    let runtime = (tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build())?;
    runtime.block_on(async {
        let Ok(occupied) = TcpListener::bind((std::net::Ipv6Addr::LOCALHOST, 0)).await else {
            return Ok(());
        };
        let preferred_port = (occupied.local_addr())?.port();
        let primary = (TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, preferred_port)).await)?;
        drop(primary);
        let (listener, ipv6) = (callback_listeners(preferred_port).await)?;
        let port = (listener.local_addr())?.port();
        anyhow::ensure!((port) != (preferred_port));
        let ipv6 = match ipv6 {
            Some(ipv6) => ipv6,
            None => anyhow::bail!("IPv6 listener missing despite available IPv6 loopback"),
        };
        anyhow::ensure!(((ipv6.local_addr())?.port()) == (port));
        let released = (TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, preferred_port)).await)?;
        let address = (ipv6.local_addr())?;
        let receiver =
            tokio::spawn(async move { receive_code(&listener, Some(&ipv6), "expected").await });
        let mut stream = (tokio::net::TcpStream::connect(address).await)?;
        (stream
            .write_all(
                b"GET /callback?state=expected&code=fallback HTTP/1.1\r\nHost: localhost\r\n\r\n",
            )
            .await)?;
        anyhow::ensure!((((receiver.await)?)?) == ("fallback"));
        drop(released);
        drop(occupied);
        Ok::<_, anyhow::Error>(())
    })?;
    Ok(())
}

#[test]
fn bootstrap_fills_only_missing_identity_and_never_refreshes_organization() -> anyhow::Result<()> {
    let identity = serde_json::json!({"oauth_account":{"account_uuid":"bootstrap-account", "organization_uuid":"bootstrap-org"}});
    let mut value = credentials();
    merge_identity(&mut value, &identity, true);
    anyhow::ensure!((value.account_id.as_deref()) == (Some("old-account")));
    anyhow::ensure!((value.org_id.as_deref()) == (Some("original-org")));
    value.account_id = None;
    value.org_id = None;
    merge_identity(&mut value, &identity, false);
    anyhow::ensure!((value.account_id.as_deref()) == (Some("bootstrap-account")));
    anyhow::ensure!(value.org_id.is_none());
    merge_identity(&mut value, &identity, true);
    anyhow::ensure!((value.org_id.as_deref()) == (Some("bootstrap-org")));
    Ok(())
}

#[test]
fn verifier_and_state_match_omp_encoding() -> anyhow::Result<()> {
    let verifier = random_verifier();
    let state = random_state();
    anyhow::ensure!((verifier.len()) == (128));
    anyhow::ensure!(((URL_SAFE_NO_PAD.decode(verifier))?.len()) == (96));
    anyhow::ensure!((state.len()) == (32));
    anyhow::ensure!(state.bytes().all(|byte| byte.is_ascii_hexdigit()));
    Ok(())
}

#[test]
fn errors_do_not_include_secrets_or_response_bodies() -> anyhow::Result<()> {
    anyhow::ensure!((AuthError::Network.to_string()) == ("OAuth network request failed"));
    anyhow::ensure!(!AuthError::Http(401).to_string().contains("https://"));
    Ok(())
}

/// Only a lock another refresh holds, the network, or a status the retry
/// table retries may pass on a second try.
#[test]
fn only_a_held_lock_the_network_and_retried_statuses_are_transient() -> anyhow::Result<()> {
    for (error, transient) in [
        (AuthError::Busy, true),
        (AuthError::Network, true),
        (AuthError::Http(429), true),
        (AuthError::Http(503), true),
        (AuthError::Http(401), false),
        (AuthError::NotLoggedIn, false),
        (AuthError::Client, false),
        (AuthError::Storage, false),
    ] {
        anyhow::ensure!(error.is_transient() == transient, "{error}");
    }
    Ok(())
}

/// A login saved before the plugin kept its credentials through rig-core's
/// file store is read as it is: no new login.
#[test]
fn a_login_saved_by_the_previous_storage_code_is_read_without_a_new_login() -> anyhow::Result<()> {
    let home = tempfile::tempdir()?;
    let dir = home.path().join("anthropic-oauth");
    fs::DirBuilder::new().mode(0o700).create(&dir)?;
    let expires_at = now()? + 3600;
    // The bytes and modes the previous `save` wrote, beside its lock file.
    let saved = format!(
        r#"{{"access_token":"fake-access","refresh_token":"fake-refresh","expires_at":{expires_at},"account_id":"fake-account","org_id":null}}"#
    );
    for (name, text) in [("credentials.json", saved.as_str()), ("lock", "")] {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(dir.join(name))?;
        file.write_all(text.as_bytes())?;
    }
    signed_in(home.path())?;
    let client = OAuthClient::production()?;
    let result = bevy_tasks::block_on(credentials_in(&dir, &client))?;
    anyhow::ensure!(result == ("fake-access".into(), Some("fake-account".into())));
    Ok(())
}

/// Where the login that `a_login_waiting_in_the_browser_leaves_the_credentials_free`
/// runs in a copy of itself keeps its credentials.
const LOGIN_HOME: &str = "RIG_CLOAK_TEST_LOGIN_HOME";

/// A login waiting for the user in the browser holds nothing another rig
/// needs: a request there reads the saved credentials at once. The login
/// runs in a copy of this test, its browser a script that only reports the
/// URL it was given.
#[cfg(target_os = "linux")]
#[test]
#[allow(clippy::disallowed_methods, reason = "test fixtures")]
fn a_login_waiting_in_the_browser_leaves_the_credentials_free() -> anyhow::Result<()> {
    if let Some(home) = std::env::var_os(LOGIN_HOME) {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        runtime.block_on(login(Path::new(&home)))?;
        return Ok(());
    }
    let root = tempfile::tempdir()?;
    let home = root.path().join("home");
    let dir = home.join("anthropic-oauth");
    fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(&dir)?;
    let mut value = credentials();
    value.expires_at = now()? + 3600;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(dir.join("credentials.json"))?;
    file.write_all(&serde_json::to_vec(&value)?)?;
    let bin = root.path().join("bin");
    fs::create_dir(&bin)?;
    let opened = root.path().join("opened");
    let opener = bin.join("xdg-open");
    fs::write(
        &opener,
        format!("#!/bin/sh\necho \"$1\" > '{}'\n", opened.display()),
    )?;
    fs::set_permissions(&opener, fs::Permissions::from_mode(0o700))?;
    let path = std::env::join_paths(std::iter::once(bin).chain(std::env::split_paths(
        &std::env::var_os("PATH").unwrap_or_default(),
    )))?;
    let name = "auth::tests::a_login_waiting_in_the_browser_leaves_the_credentials_free";
    let mut child = std::process::Command::new(std::env::current_exe()?)
        .args(["--exact", name, "--test-threads", "1"])
        .env(LOGIN_HOME, &home)
        .env("PATH", path)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    let waited = (|| {
        let started = std::time::Instant::now();
        while !opened.exists() {
            anyhow::ensure!(
                started.elapsed() < Duration::from_secs(20),
                "the login never opened the browser"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
        let client = OAuthClient::production()?;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        let result = runtime.block_on(async {
            tokio::time::timeout(Duration::from_secs(10), credentials_in(&dir, &client)).await
        });
        let Ok(result) = result else {
            anyhow::bail!("the request waited on the login's browser");
        };
        anyhow::ensure!(result? == ("old-access".into(), Some("old-account".into())));
        Ok(())
    })();
    child.kill().ok();
    child.wait().ok();
    waited
}
