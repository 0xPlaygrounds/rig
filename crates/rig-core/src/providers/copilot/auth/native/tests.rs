use super::super::bootstrap_token_fingerprint;
use super::{
    ApiKeyRecord, next_poll_interval_seconds, normalize_poll_interval_seconds, refresh_api_key,
    should_retry_with_fresh_access_token_status,
};
use crate::error::BoxError;
use crate::providers::copilot::auth::{
    AuthError, AuthSource, Authenticator, DeviceCodeHandler, MemoryTokenStore, STORE_KEY,
    TokenStore,
};
use crate::test_utils::{MockHttpResponse, RecordingHttpClient, SequencedHttpClient};
use crate::wasm_compat::WasmBoxedFuture;
use http::StatusCode;
use std::sync::Arc;

#[test]
fn api_key_record_parses_dynamic_api_base() {
    let record: ApiKeyRecord = serde_json::from_str(
        r#"{
                "token": "copilot-token",
                "expires_at": 1775791135,
                "endpoints": {
                    "api": "https://api.individual.githubcopilot.com"
                }
            }"#,
    )
    .expect("parse api key record");

    assert_eq!(
        record.api_base().as_deref(),
        Some("https://api.individual.githubcopilot.com")
    );
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
        .auth_context_oauth(&RecordingHttpClient::new(""))
        .await
        .expect_err("missing cached auth should not start device flow")
        .to_string();

    assert!(err.contains("GitHub Copilot sign-in required"), "{err}");
}

#[test]
fn api_key_record_reuse_requires_matching_bootstrap_token_for_explicit_auth() {
    let record = ApiKeyRecord {
        token: Some("copilot-token".into()),
        expires_at: Some(i64::MAX),
        endpoints: None,
        github_token: None,
        bootstrap_token_fingerprint: Some(bootstrap_token_fingerprint("github-token-a")),
    };

    assert!(record.can_reuse_for_bootstrap_token("github-token-a"));
    assert!(!record.can_reuse_for_bootstrap_token("github-token-b"));
}

#[test]
fn api_key_record_oauth_reuse_requires_match_when_bootstrap_token_is_available() {
    let record = ApiKeyRecord {
        token: Some("copilot-token".into()),
        expires_at: Some(i64::MAX),
        endpoints: None,
        github_token: None,
        bootstrap_token_fingerprint: Some(bootstrap_token_fingerprint("github-token-a")),
    };

    assert!(record.can_reuse_for_oauth(Some("github-token-a")));
    assert!(!record.can_reuse_for_oauth(Some("github-token-b")));
    assert!(record.can_reuse_for_oauth(None));
}

#[test]
fn api_key_record_without_fingerprint_forces_refresh_when_bootstrap_token_is_known() {
    let record = ApiKeyRecord {
        token: Some("copilot-token".into()),
        expires_at: Some(i64::MAX),
        endpoints: None,
        github_token: None,
        bootstrap_token_fingerprint: None,
    };

    assert!(!record.can_reuse_for_bootstrap_token("github-token-a"));
    assert!(!record.can_reuse_for_oauth(Some("github-token-a")));
    assert!(record.can_reuse_for_oauth(None));
}

#[test]
fn poll_interval_defaults_and_clamps() {
    assert_eq!(normalize_poll_interval_seconds(None), 5);
    assert_eq!(normalize_poll_interval_seconds(Some(0)), 1);
    assert_eq!(normalize_poll_interval_seconds(Some(9)), 9);
}

#[test]
fn poll_interval_rejects_terminal_errors() {
    let denied = next_poll_interval_seconds(5, Some("access_denied"), None)
        .expect_err("access denied should fail");
    assert_eq!(denied.to_string(), "GitHub device authorization was denied");

    let unknown = next_poll_interval_seconds(
        5,
        Some("device_flow_disabled"),
        Some("OAuth app device flow is disabled"),
    )
    .expect_err("device flow disabled should fail");
    assert_eq!(
        unknown.to_string(),
        "GitHub device authorization failed: device_flow_disabled (OAuth app device flow is disabled)"
    );
}

#[test]
fn stale_access_token_retries_only_on_auth_failures() {
    assert!(should_retry_with_fresh_access_token_status(Some(
        StatusCode::UNAUTHORIZED
    )));
    assert!(should_retry_with_fresh_access_token_status(Some(
        StatusCode::FORBIDDEN
    )));
    assert!(!should_retry_with_fresh_access_token_status(Some(
        StatusCode::BAD_GATEWAY
    )));
    assert!(!should_retry_with_fresh_access_token_status(None));
}

/// The authenticator for `source` over `store`, never starting a device
/// flow.
fn stored(source: AuthSource, store: &Arc<MemoryTokenStore>) -> Authenticator {
    Authenticator::new(source, store.clone(), DeviceCodeHandler::default(), false)
}

async fn seed(store: &MemoryTokenStore, record: &serde_json::Value) -> anyhow::Result<()> {
    store
        .save(STORE_KEY, &record.to_string())
        .await
        .map_err(anyhow::Error::from_boxed)
}

async fn kept(store: &MemoryTokenStore) -> anyhow::Result<serde_json::Value> {
    let text = store
        .load(STORE_KEY)
        .await
        .map_err(anyhow::Error::from_boxed)?
        .unwrap_or_default();
    Ok(serde_json::from_str(&text)?)
}

/// Reuse of a kept session is a local contract, not a provider response.
#[tokio::test]
async fn kept_session_is_reused_unchanged() -> anyhow::Result<()> {
    let fixture = serde_json::json!({
        "github_token": "synthetic-bootstrap",
        "token": "synthetic-cached-copilot-key",
        "expires_at": i64::MAX,
        "endpoints": { "api": "https://cached-copilot.example.invalid" },
        "bootstrap_token_fingerprint": bootstrap_token_fingerprint("synthetic-bootstrap")
    });
    let store = Arc::new(MemoryTokenStore::default());
    seed(&store, &fixture).await?;
    let http = RecordingHttpClient::new("");
    let auth = stored(AuthSource::OAuth, &store);
    let oauth = auth.auth_context_oauth(&http).await?;
    let explicit = auth
        .auth_context_with_github_access_token(&http, "synthetic-bootstrap")
        .await?;
    anyhow::ensure!(
        http.requests().is_empty(),
        "a fresh session must not exchange"
    );
    for context in [oauth, explicit] {
        anyhow::ensure!(
            context.api_key.expose() == "synthetic-cached-copilot-key",
            "kept API key changed"
        );
        anyhow::ensure!(
            context.api_base.as_deref() == Some("https://cached-copilot.example.invalid"),
            "kept API endpoint changed"
        );
    }
    anyhow::ensure!(kept(&store).await? == fixture, "kept record changed");
    Ok(())
}

/// An expired session is exchanged again with the kept GitHub token, and
/// the new session is kept with it. The exchange reply is synthetic: what
/// is under test is where the result goes.
#[tokio::test]
async fn exchange_keeps_the_github_token_and_the_new_session() -> anyhow::Result<()> {
    let store = Arc::new(MemoryTokenStore::default());
    seed(
        &store,
        &serde_json::json!({ "github_token": "gho_synthetic", "token": "old", "expires_at": 0 }),
    )
    .await?;
    let http = RecordingHttpClient::new(r#"{"token":"tid=new","expires_at":4102444800}"#);
    let context = stored(AuthSource::OAuth, &store)
        .auth_context_oauth(&http)
        .await?;
    anyhow::ensure!(context.api_key.expose() == "tid=new");
    anyhow::ensure!(http.requests().len() == 1, "one exchange");
    let record = kept(&store).await?;
    anyhow::ensure!(record.get("github_token") == Some(&"gho_synthetic".into()));
    anyhow::ensure!(record.get("token") == Some(&"tid=new".into()));
    anyhow::ensure!(
        record.get("bootstrap_token_fingerprint")
            == Some(&bootstrap_token_fingerprint("gho_synthetic").into())
    );
    Ok(())
}

/// A GitHub token the exchange refuses is forgotten in the store before
/// the sign-in it then needs.
#[tokio::test]
async fn a_refused_github_token_is_forgotten() -> anyhow::Result<()> {
    let store = Arc::new(MemoryTokenStore::default());
    seed(
        &store,
        &serde_json::json!({ "github_token": "gho_revoked" }),
    )
    .await?;
    let http = RecordingHttpClient::with_error(StatusCode::UNAUTHORIZED, "Bad credentials");
    let err = stored(AuthSource::OAuth, &store)
        .auth_context_oauth(&http)
        .await
        .err()
        .map(|err| err.to_string())
        .unwrap_or_default();
    anyhow::ensure!(err.contains("GitHub Copilot sign-in required"), "{err}");
    anyhow::ensure!(kept(&store).await?.get("github_token").is_none());
    Ok(())
}

/// A device sign-in keeps the GitHub token and the session it exchanges
/// for in the application's store. The GitHub and exchange replies are
/// synthetic: what is under test is where the results go.
#[tokio::test]
async fn device_sign_in_keeps_both_tokens_in_the_store() -> anyhow::Result<()> {
    let store = Arc::new(MemoryTokenStore::default());
    let http = SequencedHttpClient::new([
        MockHttpResponse::success(
            r#"{"device_code":"dc","user_code":"UC-1","verification_uri":"https://github.com/login/device","interval":1}"#,
        ),
        MockHttpResponse::success(r#"{"access_token":"gho_fresh"}"#),
        MockHttpResponse::success(r#"{"token":"tid=fresh","expires_at":4102444800}"#),
    ]);
    let shown = Arc::new(std::sync::Mutex::new(Vec::new()));
    let seen = shown.clone();
    let auth = Authenticator::new(
        AuthSource::OAuth,
        store.clone(),
        DeviceCodeHandler::new(move |prompt| {
            if let Ok(mut codes) = seen.lock() {
                codes.push(prompt.user_code);
            }
        }),
        true,
    );
    anyhow::ensure!(
        format!("{auth:?}").starts_with("Authenticator { source: OAuth"),
        "{auth:?}"
    );
    let context = auth.auth_context(&http).await?;
    anyhow::ensure!(context.api_key.expose() == "tid=fresh");
    anyhow::ensure!(
        shown.lock().map(|codes| codes.clone()).unwrap_or_default() == ["UC-1"],
        "the handler shows the code"
    );
    let record = kept(&store).await?;
    anyhow::ensure!(record.get("github_token") == Some(&"gho_fresh".into()));
    anyhow::ensure!(record.get("token") == Some(&"tid=fresh".into()));
    Ok(())
}

/// A store that loads nothing and saves nothing.
struct ReadOnlyVault;

impl TokenStore for ReadOnlyVault {
    fn load<'a>(&'a self, _: &'a str) -> WasmBoxedFuture<'a, Result<Option<String>, BoxError>> {
        Box::pin(async { Ok(None) })
    }

    fn save<'a>(&'a self, _: &'a str, _: &'a str) -> WasmBoxedFuture<'a, Result<(), BoxError>> {
        Box::pin(async { Err("the vault is read-only".into()) })
    }

    fn delete<'a>(&'a self, _: &'a str) -> WasmBoxedFuture<'a, Result<(), BoxError>> {
        Box::pin(async { Err("the vault is read-only".into()) })
    }
}

/// A session the store cannot keep fails the request.
#[tokio::test]
async fn a_failed_save_fails_the_request() {
    let auth = Authenticator::new(
        AuthSource::GitHubAccessToken("gho_synthetic".into()),
        ReadOnlyVault,
        DeviceCodeHandler::default(),
        false,
    );
    let http = RecordingHttpClient::new(r#"{"token":"tid=new","expires_at":4102444800}"#);
    let err = auth
        .auth_context(&http)
        .await
        .expect_err("the session cannot be kept");
    assert!(matches!(err, AuthError::Store(_)), "{err}");
}

#[tokio::test]
async fn api_key_exchange_rejects_a_blank_token() {
    let http = RecordingHttpClient::new(r#"{"token": "  "}"#);
    let err = refresh_api_key(&http, "synthetic-bootstrap")
        .await
        .expect_err("a blank API key cannot authenticate")
        .to_string();

    assert!(err.contains("did not include a token"), "{err}");
}
