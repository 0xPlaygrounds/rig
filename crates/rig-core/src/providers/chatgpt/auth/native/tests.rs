use super::{
    DeviceCodeHandler, DeviceCodeResponse, OAuthErrorResponse, OAuthTokenResponse,
    PlatformAuthenticator, build_auth_record, format_refresh_error,
    should_reauthenticate_after_refresh,
};
use crate::test_utils::RecordingHttpClient;
use http::StatusCode;

#[test]
fn device_code_response_accepts_numeric_interval() {
    let response: DeviceCodeResponse = serde_json::from_str(
        r#"{
                "device_auth_id": "deviceauth_123",
                "user_code": "ABCD-EFGH",
                "interval": 5
            }"#,
    )
    .expect("device code response");

    assert_eq!(response.interval, Some(5));
}

#[test]
fn device_code_response_accepts_string_interval() {
    let response: DeviceCodeResponse = serde_json::from_str(
        r#"{
                "device_auth_id": "deviceauth_123",
                "user_code": "ABCD-EFGH",
                "interval": "5"
            }"#,
    )
    .expect("device code response");

    assert_eq!(response.interval, Some(5));
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
    let auth = PlatformAuthenticator::new(None, DeviceCodeHandler::default(), false);
    let err = auth
        .auth_context_oauth(&RecordingHttpClient::new(""))
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
