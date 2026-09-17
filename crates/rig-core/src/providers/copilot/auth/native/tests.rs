use super::{
    ApiKeyRecord, DeviceCodeHandler, PlatformAuthenticator, bootstrap_token_fingerprint,
    next_poll_interval_seconds, normalize_poll_interval_seconds,
    should_retry_with_fresh_access_token_status,
};
use crate::test_utils::RecordingHttpClient;
use http::StatusCode;

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
    let auth = PlatformAuthenticator::new(None, None, DeviceCodeHandler::default(), false);
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
fn poll_interval_handles_pending_and_slow_down() {
    assert_eq!(
        next_poll_interval_seconds(5, Some("authorization_pending"), None)
            .expect("authorization pending interval"),
        5
    );
    assert_eq!(
        next_poll_interval_seconds(5, Some("slow_down"), None).expect("slow_down interval"),
        10
    );
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

/// Disk persistence and reuse are local contracts, not provider responses.
#[tokio::test]
async fn cached_credential_remains_persistent_and_reusable() -> anyhow::Result<()> {
    let dir = assert_fs::TempDir::new()?;
    let path = dir.path().join("api-key.json");
    let fixture = serde_json::json!({
        "token": "synthetic-cached-copilot-key",
        "expires_at": i64::MAX,
        "endpoints": { "api": "https://cached-copilot.example.invalid" },
        "bootstrap_token_fingerprint": bootstrap_token_fingerprint("synthetic-bootstrap")
    });
    let record: ApiKeyRecord = serde_json::from_value(fixture.clone())?;
    super::write_json_record(Some(&path), &record)?;
    let http = RecordingHttpClient::new("");
    let auth = PlatformAuthenticator::new(
        None,
        Some(path.clone()),
        DeviceCodeHandler::default(),
        false,
    );
    let oauth = auth.auth_context_oauth(&http).await?;
    let explicit = auth
        .auth_context_with_github_access_token(&http, "synthetic-bootstrap")
        .await?;
    anyhow::ensure!(
        http.requests().is_empty(),
        "a fresh cache must not exchange"
    );
    for context in [oauth, explicit] {
        anyhow::ensure!(
            context.api_key.expose() == "synthetic-cached-copilot-key",
            "cached API key changed"
        );
        anyhow::ensure!(
            context.api_base.as_deref() == Some("https://cached-copilot.example.invalid"),
            "cached API endpoint changed"
        );
    }
    let persisted: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
    anyhow::ensure!(persisted == fixture, "persisted cache changed");
    Ok(())
}
