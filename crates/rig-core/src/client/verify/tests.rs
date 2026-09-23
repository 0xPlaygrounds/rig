use super::*;
use crate::http_client;
use http::StatusCode;

/// A rejected verification keeps the reply the driver stamped onto it, so a
/// caller can retry on the server's schedule. A 401 or 403 is read as a
/// verdict on the credential, and keeps its reply too.
#[test]
fn a_rejected_verification_keeps_its_reply() {
    let body = r#"{"error":{"message":"slow down"}}"#;
    let mut headers = http::HeaderMap::new();
    headers.insert(http::header::RETRY_AFTER, "20".parse().expect("value"));

    for status in [
        StatusCode::INTERNAL_SERVER_ERROR,
        StatusCode::from_u16(529).expect("overloaded"),
        StatusCode::TOO_MANY_REQUESTS,
    ] {
        let error = authentication(
            ProviderError::from_http_response(status, body)
                .with_response_headers(Some(headers.clone())),
        );

        assert!(matches!(error, ProviderError::ProviderResponse(_)));
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
        for error in [
            ProviderError::from_http_response(status, body),
            ProviderError::from_transport_error(http_client::Error::non_success_with_details(
                status,
                headers.clone(),
                body.to_string(),
            )),
        ] {
            let error = authentication(error.with_provider_request_id(Some("req_1".into())));
            let ProviderError::InvalidAuthentication(response) = &error else {
                panic!("{status} is a verdict on the credential: {error:?}");
            };
            assert_eq!(response.status, Some(status));
            assert_eq!(response.body, body);
            assert_eq!(error.provider_request_id(), Some("req_1"));
            assert!(!error.is_retryable());
        }
    }
}

/// Only a 401 or 403 reply is a verdict on the credential. A transport
/// failure, and a reply that did not decode, classify as they do for every
/// other operation.
#[test]
fn other_verification_failures_are_unchanged() {
    let json = serde_json::from_str::<serde_json::Value>("{").expect_err("malformed");
    for error in [
        ProviderError::Http(http_client::Error::StreamEnded),
        ProviderError::Json(json),
        ProviderError::Response("verify reply carried no payload".into()),
    ] {
        let kind = error.kind();
        let error = authentication(error);
        assert!(!matches!(error, ProviderError::InvalidAuthentication(_)));
        assert_eq!(error.kind(), kind);
    }
}
