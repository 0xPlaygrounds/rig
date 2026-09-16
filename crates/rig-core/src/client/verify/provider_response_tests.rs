use super::*;
use http::StatusCode;

#[test]
fn verify_error_provider_response_helpers_with_preserved_json_body() {
    let body = r#"{"error":{"message":"rate limited"}}"#;
    let error = VerifyError::ProviderResponse(
        provider_response::ProviderResponseError::without_status(body.to_string()),
    );

    assert_eq!(error.provider_response_body(), Some(body));
    assert_eq!(error.provider_response_status(), None);
    assert_eq!(
        error.provider_response_json().expect("valid JSON"),
        Some(serde_json::json!({ "error": { "message": "rate limited" } }))
    );
}

#[test]
fn verify_error_provider_response_helpers_with_http_non_success() {
    let body = r#"{"error":{"message":"bad request"}}"#;
    let error = VerifyError::from_transport_error(http_client::Error::non_success_with_details(
        StatusCode::BAD_REQUEST,
        http::HeaderMap::new(),
        body.to_string(),
    ));

    assert_eq!(error.provider_response_body(), Some(body));
    assert_eq!(
        error.provider_response_status(),
        Some(StatusCode::BAD_REQUEST)
    );
    assert_eq!(
        error.provider_response_json().expect("valid JSON"),
        Some(serde_json::json!({ "error": { "message": "bad request" } }))
    );
}

#[test]
fn verify_error_provider_response_helpers_with_preserved_plain_text_body() {
    let error = VerifyError::ProviderResponse(
        provider_response::ProviderResponseError::without_status("not json".to_string()),
    );

    assert_eq!(error.provider_response_body(), Some("not json"));
    assert!(error.provider_response_json().is_err());
}

#[test]
fn verify_error_provider_error_is_not_a_provider_response() {
    let error = VerifyError::ProviderError("internal diagnostic".to_string());

    assert_eq!(error.provider_response_body(), None);
    assert_eq!(error.provider_response_status(), None);
    assert_eq!(error.provider_response_json().expect("no body"), None);
}

#[test]
fn verify_error_provider_response_helpers_with_unrelated_variant() {
    let error = VerifyError::InvalidAuthentication;

    assert_eq!(error.provider_response_body(), None);
    assert_eq!(error.provider_response_status(), None);
    assert_eq!(error.provider_response_json().expect("no body"), None);
}

/// rig#2210, on the wire path: the 401/403 reading is the only thing
/// `VerifyError`'s `WireError` impl may collapse. Every other rejected
/// verification must still carry the reply the driver stamped onto it —
/// status, body, and headers — so a caller can retry on the server's
/// schedule instead of guessing.
#[test]
fn a_rejected_verification_keeps_its_reply_unless_it_is_an_auth_failure() {
    let body = r#"{"error":{"message":"slow down"}}"#;
    let mut headers = http::HeaderMap::new();
    headers.insert(http::header::RETRY_AFTER, "20".parse().expect("value"));

    for status in [
        StatusCode::INTERNAL_SERVER_ERROR,
        StatusCode::from_u16(529).expect("overloaded"),
        StatusCode::TOO_MANY_REQUESTS,
    ] {
        let error = <VerifyError as WireError>::http_response(status, body)
            .with_response_headers(Some(headers.clone()));

        assert_eq!(error.provider_response_status(), Some(status));
        assert_eq!(error.provider_response_body(), Some(body));
        assert_eq!(
            error
                .provider_response_headers()
                .and_then(|headers| headers.get(http::header::RETRY_AFTER))
                .and_then(|value| value.to_str().ok()),
            Some("20"),
            "{status}: Retry-After not recoverable from a failed verify",
        );
    }

    for status in [StatusCode::UNAUTHORIZED, StatusCode::FORBIDDEN] {
        assert!(
            matches!(
                <VerifyError as WireError>::http_response(status, body),
                VerifyError::InvalidAuthentication
            ),
            "{status} is a verdict on the credential, not a provider response",
        );
        assert!(
            matches!(
                <VerifyError as WireError>::transport(
                    http_client::Error::non_success_with_details(
                        status,
                        http::HeaderMap::new(),
                        body.to_string(),
                    )
                ),
                VerifyError::InvalidAuthentication
            ),
            "{status} seen by the transport is the same verdict",
        );
    }
}
