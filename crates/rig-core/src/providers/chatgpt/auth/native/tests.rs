use super::{
    DeviceCodeResponse, OAuthErrorResponse, OAuthTokenResponse, build_auth_record,
    format_refresh_error, should_reauthenticate_after_refresh,
};
use crate::providers::chatgpt::auth::{AuthSource, Authenticator, DeviceCodeHandler};
use crate::test_utils::RecordingHttpClient;
use http::StatusCode;

#[test]
fn device_code_response_accepts_a_numeric_or_string_interval() {
    for (case, interval) in [("numeric", "5"), ("string", r#""5""#)] {
        let response: DeviceCodeResponse = serde_json::from_str(&format!(
            r#"{{
                "device_auth_id": "deviceauth_123",
                "user_code": "ABCD-EFGH",
                "interval": {interval}
            }}"#
        ))
        .unwrap_or_else(|error| panic!("{case}: device code response: {error}"));

        assert_eq!(response.interval, Some(5), "{case}");
    }
}

#[test]
fn refresh_reauth_only_on_invalid_grant() {
    assert!(should_reauthenticate_after_refresh(
        StatusCode::BAD_REQUEST,
        Some("invalid_grant")
    ));
    assert!(should_reauthenticate_after_refresh(
        StatusCode::UNAUTHORIZED,
        Some("invalid_grant")
    ));
    assert!(!should_reauthenticate_after_refresh(
        StatusCode::BAD_GATEWAY,
        Some("invalid_grant")
    ));
    assert!(!should_reauthenticate_after_refresh(
        StatusCode::BAD_REQUEST,
        Some("invalid_request")
    ));
    assert!(!should_reauthenticate_after_refresh(
        StatusCode::UNAUTHORIZED,
        None
    ));
}

#[tokio::test]
async fn noninteractive_oauth_requires_sign_in_instead_of_device_flow() {
    let auth = Authenticator::new(AuthSource::OAuth, None, DeviceCodeHandler::default(), false);
    let err = auth
        .auth_context(&RecordingHttpClient::new(""))
        .await
        .expect_err("missing cached auth should not start device flow")
        .to_string();

    assert!(err.contains("ChatGPT sign-in required"), "{err}");
}

#[test]
fn refresh_error_uses_oauth_description_when_present() {
    let oauth_error = OAuthErrorResponse {
        error: Some("temporarily_unavailable".into()),
        error_description: Some("please retry".into()),
    };

    assert_eq!(
        format_refresh_error(StatusCode::BAD_GATEWAY, Some(&oauth_error), ""),
        "ChatGPT token refresh failed: 502 Bad Gateway temporarily_unavailable (please retry)"
    );
}

#[test]
fn build_auth_record_preserves_existing_refresh_token_when_refresh_omits_one() {
    let record = build_auth_record(
        OAuthTokenResponse {
            access_token: "access-token".into(),
            refresh_token: None,
            id_token: None,
        },
        Some("cached-refresh-token".into()),
    );

    assert_eq!(
        record.refresh_token.as_deref(),
        Some("cached-refresh-token")
    );
}

/// Disk persistence and reuse are local contracts, not provider responses.
#[tokio::test]
async fn cached_credential_remains_persistent_and_reusable() -> anyhow::Result<()> {
    let dir = assert_fs::TempDir::new()?;
    let path = dir.path().join("auth.json");
    let fixture = serde_json::json!({
        "access_token": "synthetic-cached-chatgpt-token",
        "refresh_token": "synthetic-cached-refresh-token",
        "id_token": null,
        "expires_at": i64::MAX,
        "account_id": "synthetic-cached-account"
    });
    let record: super::AuthRecord = serde_json::from_value(fixture.clone())?;
    super::write_json_record(Some(&path), &record)?;
    let http = RecordingHttpClient::new("");
    let auth = Authenticator::new(
        AuthSource::OAuth,
        Some(path.clone()),
        DeviceCodeHandler::default(),
        false,
    );
    let context = auth.auth_context(&http).await?;
    anyhow::ensure!(http.requests().is_empty(), "a fresh cache must not refresh");
    anyhow::ensure!(
        context.access_token.expose() == "synthetic-cached-chatgpt-token",
        "cached access token changed"
    );
    anyhow::ensure!(
        context.account_id.as_deref() == Some("synthetic-cached-account"),
        "cached account changed"
    );
    let persisted: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
    anyhow::ensure!(persisted == fixture, "persisted cache changed");
    Ok(())
}

mod browser_sign_in {
    use super::super::browser::{
        CALLBACK_PATH, Callback, SCOPE, authorize_url, parse_callback, pkce_challenge,
        random_token, redirect_uri, spawn_listener,
    };
    use std::collections::HashMap;
    use std::io::{Read, Write};
    use std::net::{Ipv4Addr, TcpListener, TcpStream};
    use std::time::{Duration, Instant};

    #[test]
    fn authorize_url_carries_pkce_state_and_the_registered_redirect() {
        let url = authorize_url(&redirect_uri(1455), "challenge-123", "state-456", "rig");
        let parsed = url::Url::parse(&url).expect("authorize url parses");
        assert_eq!(
            format!("{}{}", parsed.origin().ascii_serialization(), parsed.path()),
            "https://auth.openai.com/oauth/authorize"
        );
        let query: HashMap<String, String> = parsed.query_pairs().into_owned().collect();
        let expected = [
            ("response_type", "code"),
            ("client_id", super::super::CHATGPT_CLIENT_ID),
            ("redirect_uri", "http://127.0.0.1:1455/auth/callback"),
            ("code_challenge", "challenge-123"),
            ("code_challenge_method", "S256"),
            ("state", "state-456"),
            ("scope", SCOPE),
            ("id_token_add_organizations", "true"),
            ("codex_cli_simplified_flow", "true"),
            ("originator", "rig"),
        ];
        for (key, value) in expected {
            assert_eq!(query.get(key).map(String::as_str), Some(value), "{key}");
        }
        assert_eq!(query.len(), expected.len());
        assert!(SCOPE.split(' ').any(|scope| scope == "offline_access"));
    }

    #[test]
    fn pkce_challenge_is_the_s256_of_the_verifier() {
        // `printf %s <verifier> | openssl dgst -sha256 -binary | base64url`
        assert_eq!(
            pkce_challenge("dBjftJeZ4CVP-mJ92K9XmOkXVMtKtV0eGWWjqtdbr8vE"),
            "2M9avNSz_FoDqpmzPAJTqw8c_f5UqtRV8Hb2lkcHnUA"
        );
        let verifier = random_token(4);
        assert_eq!(verifier.len(), 86, "64 bytes in unpadded base64");
        assert_ne!(verifier, random_token(4));
    }

    #[test]
    fn callback_parsing_checks_the_state_first() {
        let line = |target: &str| format!("GET {target} HTTP/1.1");
        assert_eq!(
            parse_callback(&line("/auth/callback?code=abc%2B1&state=s1"), "s1"),
            Callback::Code("abc+1".into())
        );
        assert_eq!(
            parse_callback(&line("/auth/callback?code=abc&state=other"), "s1"),
            Callback::StateMismatch
        );
        assert_eq!(
            parse_callback(&line("/auth/callback?code=abc"), "s1"),
            Callback::StateMismatch
        );
        assert_eq!(
            parse_callback(
                &line("/auth/callback?error=access_denied&error_description=No+thanks&state=s1"),
                "s1"
            ),
            Callback::Failed("the browser sign-in returned access_denied (No thanks)".into())
        );
        assert!(matches!(
            parse_callback(&line("/auth/callback?state=s1&code="), "s1"),
            Callback::Failed(_)
        ));
        assert_eq!(parse_callback(&line("/favicon.ico"), "s1"), Callback::Other);
        assert_eq!(
            parse_callback(&line("/auth/callback/x?code=a&state=s1"), "s1"),
            Callback::Other
        );
        assert_eq!(
            parse_callback("POST /auth/callback?code=a&state=s1 HTTP/1.1", "s1"),
            Callback::Other
        );
    }

    fn get(port: u16, target: &str) -> String {
        let mut stream = TcpStream::connect((Ipv4Addr::LOCALHOST, port)).expect("connect");
        write!(stream, "GET {target} HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n").expect("write");
        let mut response = String::new();
        stream.read_to_string(&mut response).expect("read");
        response
    }

    #[tokio::test]
    async fn listener_ignores_other_requests_and_reports_the_outcome_page() {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).expect("bind");
        let port = listener.local_addr().expect("addr").port();
        let callback = spawn_listener(
            listener,
            "s1".into(),
            Instant::now() + Duration::from_secs(30),
        )
        .expect("spawn");

        assert!(get(port, "/favicon.ico").starts_with("HTTP/1.1 404"));
        let stale = format!("{CALLBACK_PATH}?code=c&state=stale");
        assert!(get(port, &stale).starts_with("HTTP/1.1 400"));

        let browser = std::thread::spawn(move || get(port, "/auth/callback?code=c1&state=s1"));
        let received = callback.await.expect("listener alive").expect("code");
        assert_eq!(received.code, "c1");
        received.finish(Ok(()));
        let page = browser.join().expect("browser thread");
        assert!(page.starts_with("HTTP/1.1 200"), "{page}");
        assert!(page.contains("Signed in to ChatGPT"), "{page}");
    }

    #[test]
    fn dropping_the_waiter_closes_the_listener() {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).expect("bind");
        let port = listener.local_addr().expect("addr").port();
        let callback = spawn_listener(
            listener,
            "s1".into(),
            Instant::now() + Duration::from_secs(30),
        )
        .expect("spawn");
        drop(callback);

        let deadline = Instant::now() + Duration::from_secs(5);
        while TcpListener::bind((Ipv4Addr::LOCALHOST, port)).is_err() {
            assert!(Instant::now() < deadline, "listener still holds the port");
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}

/// The credential file and the directory created for it are the owner's alone.
#[cfg(unix)]
#[test]
fn credential_file_is_written_private() -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let dir = assert_fs::TempDir::new()?;
    let path = dir.path().join("chatgpt").join("auth.json");
    let record: super::AuthRecord = serde_json::from_value(serde_json::json!({
        "access_token": "synthetic-token",
        "refresh_token": null,
        "id_token": null,
        "expires_at": null,
        "account_id": null
    }))?;
    super::write_json_record(Some(&path), &record)?;
    let file_mode = std::fs::metadata(&path)?.permissions().mode() & 0o777;
    anyhow::ensure!(file_mode == 0o600, "file mode {file_mode:o}");
    let parent = path.parent().ok_or_else(|| anyhow::anyhow!("no parent"))?;
    let dir_mode = std::fs::metadata(parent)?.permissions().mode() & 0o777;
    anyhow::ensure!(dir_mode == 0o700, "directory mode {dir_mode:o}");
    Ok(())
}
