use super::*;

mod http;
mod off_runtime;

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
    let root = (tempfile::tempdir())?;
    let dir = root.path().join("oauth");
    (private_dir(&dir))?;
    let guard = (lock(&dir))?;
    anyhow::ensure!(matches!(lock(&dir), Err(AuthError::Busy)));
    (save(&dir, &credentials()))?;
    let mut value = (load(&dir))?;
    anyhow::ensure!((value.access_token) == ("old-access"));
    value.access_token = "replacement".into();
    (save(&dir, &value))?;
    anyhow::ensure!(((load(&dir))?.access_token) == ("replacement"));
    let mode = (fs::metadata(dir.join("credentials.json")))?
        .permissions()
        .mode();
    anyhow::ensure!((mode & 0o777) == (0o600));
    anyhow::ensure!(((fs::metadata(&dir))?.permissions().mode() & 0o777) == (0o700));
    drop(guard);
    let _guard = (lock(&dir))?;
    Ok(())
}

#[test]
fn storage_rejects_permissive_files_and_symlinks() -> anyhow::Result<()> {
    let root = (tempfile::tempdir())?;
    let dir = root.path().join("oauth");
    (private_dir(&dir))?;
    (save(&dir, &credentials()))?;
    let path = dir.join("credentials.json");
    (fs::set_permissions(&path, fs::Permissions::from_mode(0o644)))?;
    anyhow::ensure!(matches!(load(&dir), Err(AuthError::UnsafeStorage)));
    (fs::remove_file(&path))?;
    (std::os::unix::fs::symlink(root.path().join("missing"), &path))?;
    anyhow::ensure!(matches!(load(&dir), Err(AuthError::UnsafeStorage)));
    let linked = root.path().join("linked");
    (std::os::unix::fs::symlink(&dir, &linked))?;
    anyhow::ensure!(matches!(
        private_dir(&linked),
        Err(AuthError::UnsafeStorage)
    ));
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
