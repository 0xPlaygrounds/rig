use super::{
    DeviceCodeResponse, OAuthErrorResponse, OAuthTokenResponse, build_auth_record,
    format_refresh_error, should_reauthenticate_after_refresh,
};
use crate::error::BoxError;
use crate::providers::chatgpt::auth::{
    AuthError, AuthSource, Authenticator, DeviceCodeHandler, MemoryTokenStore, STORE_KEY,
    TokenStore,
};
use crate::test_utils::{MockHttpResponse, RecordingHttpClient, SequencedHttpClient};
use crate::wasm_compat::WasmBoxedFuture;
use http::StatusCode;
use std::sync::Arc;

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

        let interval = response
            .interval
            .as_ref()
            .and_then(crate::json_utils::Lenient::as_u64_lenient);
        assert_eq!(interval, Some(5), "{case}");
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
    let auth = Authenticator::new(
        AuthSource::OAuth,
        MemoryTokenStore::default(),
        DeviceCodeHandler::default(),
        false,
    );
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

/// The authenticator over `store`, never starting a device flow.
fn stored(store: &Arc<MemoryTokenStore>) -> Authenticator {
    Authenticator::new(
        AuthSource::OAuth,
        store.clone(),
        DeviceCodeHandler::default(),
        false,
    )
}

async fn kept(store: &MemoryTokenStore) -> anyhow::Result<serde_json::Value> {
    let text = store
        .load(STORE_KEY)
        .await
        .map_err(anyhow::Error::from_boxed)?
        .unwrap_or_default();
    Ok(serde_json::from_str(&text)?)
}

/// Reuse of a kept credential is a local contract, not a provider response.
#[tokio::test]
async fn kept_credential_is_reused_unchanged() -> anyhow::Result<()> {
    let fixture = serde_json::json!({
        "access_token": "synthetic-cached-chatgpt-token",
        "refresh_token": "synthetic-cached-refresh-token",
        "id_token": null,
        "expires_at": i64::MAX,
        "account_id": "synthetic-cached-account"
    });
    let store = Arc::new(MemoryTokenStore::default());
    store
        .save(STORE_KEY, &fixture.to_string())
        .await
        .map_err(anyhow::Error::from_boxed)?;
    let http = RecordingHttpClient::new("");
    let context = stored(&store).auth_context(&http).await?;
    anyhow::ensure!(
        http.requests().is_empty(),
        "a fresh credential must not refresh"
    );
    anyhow::ensure!(
        context.api_key.expose() == "synthetic-cached-chatgpt-token",
        "kept access token changed"
    );
    anyhow::ensure!(
        context.account_id.as_deref() == Some("synthetic-cached-account"),
        "kept account changed"
    );
    anyhow::ensure!(kept(&store).await? == fixture, "kept credential changed");
    Ok(())
}

/// A refresh keeps the rotated tokens in the store. The token endpoint's
/// reply is synthetic: what is under test is where the result goes.
#[tokio::test]
async fn refresh_keeps_rotated_tokens_in_the_store() -> anyhow::Result<()> {
    let store = Arc::new(MemoryTokenStore::default());
    let expired = serde_json::json!({
        "access_token": "synthetic-expired-token",
        "refresh_token": "synthetic-old-refresh",
        "expires_at": 0
    });
    store
        .save(STORE_KEY, &expired.to_string())
        .await
        .map_err(anyhow::Error::from_boxed)?;
    let http = RecordingHttpClient::new(
        r#"{"access_token":"synthetic-new-token","refresh_token":"synthetic-new-refresh"}"#,
    );
    let context = stored(&store).auth_context(&http).await?;
    anyhow::ensure!(context.api_key.expose() == "synthetic-new-token");
    let requests = http.requests();
    anyhow::ensure!(
        requests.first().map(|request| request.uri.as_str())
            == Some("https://auth.openai.com/oauth/token"),
        "refreshed at {:?}",
        requests.first().map(|request| request.uri.clone())
    );
    let record = kept(&store).await?;
    anyhow::ensure!(record.get("access_token") == Some(&"synthetic-new-token".into()));
    anyhow::ensure!(record.get("refresh_token") == Some(&"synthetic-new-refresh".into()));
    Ok(())
}

/// A store whose loads answer in turn from a list, and which keeps nothing.
struct Turns(std::sync::Mutex<std::collections::VecDeque<Option<String>>>);

impl TokenStore for Turns {
    fn load<'a>(&'a self, _: &'a str) -> WasmBoxedFuture<'a, Result<Option<String>, BoxError>> {
        let next = self
            .0
            .lock()
            .ok()
            .and_then(|mut turns| turns.pop_front())
            .flatten();
        Box::pin(async move { Ok(next) })
    }

    fn save<'a>(&'a self, _: &'a str, _: &'a str) -> WasmBoxedFuture<'a, Result<(), BoxError>> {
        Box::pin(async { Err("the vault is read-only".into()) })
    }

    fn delete<'a>(&'a self, _: &'a str) -> WasmBoxedFuture<'a, Result<(), BoxError>> {
        Box::pin(async { Ok(()) })
    }
}

/// A live credential is read from the store once and then served from
/// memory, by every clone, until it is forgotten. The store answers only
/// its first load, so a second read finds no sign-in.
#[tokio::test]
async fn a_live_credential_is_read_from_the_store_once() -> anyhow::Result<()> {
    let live = serde_json::json!({ "access_token": "live", "expires_at": i64::MAX });
    let store = Turns(std::sync::Mutex::new([Some(live.to_string())].into()));
    let auth = Authenticator::new(
        AuthSource::OAuth,
        store,
        DeviceCodeHandler::default(),
        false,
    );
    let http = RecordingHttpClient::new("");
    for auth in [&auth, &auth.clone(), &auth] {
        anyhow::ensure!(auth.auth_context(&http).await?.api_key.expose() == "live");
    }
    auth.forget();
    anyhow::ensure!(
        auth.auth_context(&http).await.is_err(),
        "a forgotten credential is read from the store again"
    );
    Ok(())
}

/// When another client spent the refresh token first, the credential it
/// kept is used instead of a new sign-in. The `invalid_grant` reply is
/// synthetic; what is under test is the second read of the store.
#[tokio::test]
async fn a_spent_refresh_token_takes_the_credential_another_client_kept() -> anyhow::Result<()> {
    let stale = serde_json::json!({ "refresh_token": "spent", "expires_at": 0 });
    let rotated = serde_json::json!({
        "access_token": "rotated-by-another-client",
        "refresh_token": "rotated",
        "expires_at": i64::MAX
    });
    let store = Turns(std::sync::Mutex::new(
        [Some(stale.to_string()), Some(rotated.to_string())].into(),
    ));
    let http =
        RecordingHttpClient::with_error(StatusCode::BAD_REQUEST, r#"{"error":"invalid_grant"}"#);
    let auth = Authenticator::new(
        AuthSource::OAuth,
        store,
        DeviceCodeHandler::default(),
        false,
    );
    let context = auth.auth_context(&http).await?;
    anyhow::ensure!(context.api_key.expose() == "rotated-by-another-client");
    anyhow::ensure!(http.requests().len() == 1, "one refresh");
    Ok(())
}

/// The replies of a device sign-in: the user code, the authorization, and
/// the tokens. Synthetic: what is under test is where the tokens go.
fn device_sign_in() -> SequencedHttpClient {
    SequencedHttpClient::new([
        MockHttpResponse::success(r#"{"device_auth_id":"d1","user_code":"UC-1","interval":"1"}"#),
        MockHttpResponse::success(r#"{"authorization_code":"code","code_verifier":"verifier"}"#),
        MockHttpResponse::success(r#"{"access_token":"fresh","refresh_token":"fresh-refresh"}"#),
    ])
}

/// A device sign-in keeps its tokens in the application's store.
#[tokio::test]
async fn device_sign_in_keeps_the_tokens_in_the_store() -> anyhow::Result<()> {
    let store = Arc::new(MemoryTokenStore::default());
    let auth = Authenticator::new(
        AuthSource::OAuth,
        store.clone(),
        DeviceCodeHandler::default(),
        true,
    )
    .with_device_code_handler(DeviceCodeHandler::new(|_| {}));
    anyhow::ensure!(
        format!("{auth:?}").starts_with("Authenticator { source: OAuth"),
        "{auth:?}"
    );
    let context = auth.auth_context(&device_sign_in()).await?;
    anyhow::ensure!(context.api_key.expose() == "fresh");
    let record = kept(&store).await?;
    anyhow::ensure!(record.get("refresh_token") == Some(&"fresh-refresh".into()));
    Ok(())
}

/// A sign-in the store cannot keep fails, rather than going unsaved. With
/// no handler the device code is printed.
#[tokio::test]
async fn a_sign_in_the_store_cannot_keep_fails() {
    let auth = Authenticator::new(
        AuthSource::OAuth,
        Turns(std::sync::Mutex::default()),
        DeviceCodeHandler::default(),
        true,
    );
    let err = auth
        .auth_context(&device_sign_in())
        .await
        .expect_err("a read-only store keeps nothing");
    assert!(matches!(err, AuthError::Store(_)), "{err}");
}

/// A store that cannot answer.
struct LockedVault;

impl TokenStore for LockedVault {
    fn load<'a>(&'a self, _: &'a str) -> WasmBoxedFuture<'a, Result<Option<String>, BoxError>> {
        Box::pin(async { Err("the vault is locked".into()) })
    }

    fn save<'a>(&'a self, _: &'a str, _: &'a str) -> WasmBoxedFuture<'a, Result<(), BoxError>> {
        Box::pin(async { Err("the vault is locked".into()) })
    }

    fn delete<'a>(&'a self, _: &'a str) -> WasmBoxedFuture<'a, Result<(), BoxError>> {
        Box::pin(async { Err("the vault is locked".into()) })
    }
}

/// A store failure fails the request, not as a missing sign-in, and sends
/// nothing.
#[tokio::test]
async fn a_failing_store_fails_the_request() {
    let http = RecordingHttpClient::new("");
    let auth = Authenticator::new(
        AuthSource::OAuth,
        LockedVault,
        DeviceCodeHandler::default(),
        true,
    );
    let err = auth
        .auth_context(&http)
        .await
        .expect_err("a locked vault has no credential");
    assert!(matches!(err, AuthError::Store(_)), "{err}");
    assert!(err.to_string().contains("the vault is locked"), "{err}");
    assert!(http.requests().is_empty());
}

mod browser_sign_in {
    use super::super::AuthError;
    use super::super::browser::{
        CALLBACK_PATH, Callback, Received, authorize_url, parse_callback, pkce_challenge,
        random_token, redirect_uri, spawn_listener,
    };
    use futures::channel::oneshot;
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
        for (key, value) in [
            ("redirect_uri", "http://127.0.0.1:1455/auth/callback"),
            ("code_challenge", "challenge-123"),
            ("state", "state-456"),
            ("originator", "rig"),
        ] {
            assert_eq!(query.get(key).map(String::as_str), Some(value), "{key}");
        }
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

    /// A listener for state `s1` on a free port, and its port.
    fn listen() -> (u16, oneshot::Receiver<Result<Received, AuthError>>) {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).expect("bind");
        let port = listener.local_addr().expect("addr").port();
        let deadline = Instant::now() + Duration::from_secs(30);
        let callback = spawn_listener(listener, "s1".into(), deadline).expect("spawn");
        (port, callback)
    }

    #[tokio::test]
    async fn listener_ignores_other_requests_and_reports_the_outcome_page() {
        let (port, callback) = listen();

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
        let (port, callback) = listen();
        drop(callback);

        let deadline = Instant::now() + Duration::from_secs(5);
        while TcpListener::bind((Ipv4Addr::LOCALHOST, port)).is_err() {
            assert!(Instant::now() < deadline, "listener still holds the port");
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}

/// A sign-out asks the provider to revoke the refresh and access tokens,
/// then deletes the record; a second one finds nothing to sign out of.
#[tokio::test]
async fn sign_out_revokes_the_tokens_and_deletes_the_record() -> anyhow::Result<()> {
    use crate::providers::chatgpt::auth::SignedOut;

    let store = Arc::new(MemoryTokenStore::default());
    let record = serde_json::json!({
        "access_token": "synthetic-access",
        "refresh_token": "synthetic-refresh",
        "expires_at": i64::MAX
    });
    store
        .save(STORE_KEY, &record.to_string())
        .await
        .map_err(anyhow::Error::from_boxed)?;
    let http = RecordingHttpClient::new("");
    let auth = stored(&store);
    let signed_out = auth.sign_out(&http).await?;
    anyhow::ensure!(
        signed_out == SignedOut::Forgotten { revoked: Ok(()) },
        "{signed_out:?}"
    );
    let sent: Vec<(String, String)> = http
        .requests()
        .into_iter()
        .map(|request| {
            let body = String::from_utf8_lossy(&request.body).into_owned();
            (request.uri, body)
        })
        .collect();
    let revoke = "https://auth.openai.com/api/accounts/oauth/revoke".to_owned();
    anyhow::ensure!(
        sent == [
            (
                revoke.clone(),
                "token=synthetic-refresh&token_type_hint=refresh_token&client_id=app_EMoamEEZ73f0CkXaXp7hrann".to_owned()
            ),
            (
                revoke,
                "token=synthetic-access&token_type_hint=access_token&client_id=app_EMoamEEZ73f0CkXaXp7hrann".to_owned()
            ),
        ],
        "{sent:?}"
    );
    anyhow::ensure!(
        store
            .load(STORE_KEY)
            .await
            .map_err(anyhow::Error::from_boxed)?
            .is_none()
    );
    anyhow::ensure!(auth.sign_out(&http).await? == SignedOut::NotSignedIn);
    Ok(())
}

/// A revocation the provider refuses still signs out here, and says so.
#[tokio::test]
async fn a_refused_revocation_still_deletes_the_record() -> anyhow::Result<()> {
    use crate::providers::chatgpt::auth::SignedOut;

    let store = Arc::new(MemoryTokenStore::default());
    store
        .save(STORE_KEY, r#"{"refresh_token":"synthetic-refresh"}"#)
        .await
        .map_err(anyhow::Error::from_boxed)?;
    let http = SequencedHttpClient::new([MockHttpResponse::error(
        StatusCode::SERVICE_UNAVAILABLE,
        "down",
    )]);
    let signed_out = stored(&store).sign_out(&http).await?;
    anyhow::ensure!(
        matches!(&signed_out, SignedOut::Forgotten { revoked: Err(why) } if why.contains("refresh_token")),
        "{signed_out:?}"
    );
    anyhow::ensure!(
        store
            .load(STORE_KEY)
            .await
            .map_err(anyhow::Error::from_boxed)?
            .is_none()
    );
    Ok(())
}
