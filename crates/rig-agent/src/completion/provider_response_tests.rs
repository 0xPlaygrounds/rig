use crate::run::CanonicalHistory;
use rig_core::error::ProviderError;
use rig_core::{ProviderResponseError, http_client};

use super::*;

#[test]
fn prompt_error_provider_response_helpers_forward_http_status_and_body() {
    let body = r#"{"error":{"message":"unauthorized"}}"#;
    let error = PromptError::Provider(ProviderError::from_transport_error(
        http_client::Error::non_success_with_details(
            http::StatusCode::UNAUTHORIZED,
            http::HeaderMap::new(),
            body.to_string(),
        ),
    ));

    assert_eq!(error.provider_response_body(), Some(body));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::UNAUTHORIZED)
    );
    assert_eq!(
        error.provider_response_json().expect("valid JSON body"),
        Some(serde_json::json!({
            "error": { "message": "unauthorized" }
        }))
    );
}

/// rig#2210: the response headers forward through both wrappers, so an
/// agent-level caller can back off on `Retry-After` without unwrapping to
/// the transport error by hand. Covered on both classifications, since
/// the two store the headers in different places.
#[test]
fn prompt_error_forwards_captured_response_headers() {
    let mut headers = http::HeaderMap::new();
    headers.insert(
        http::header::RETRY_AFTER,
        http::HeaderValue::from_static("20"),
    );
    let body = r#"{"error":{"message":"rate limited"}}"#;

    for completion_error in [
        // A preserved response, with and without a request id.
        ProviderError::from_http_response(http::StatusCode::TOO_MANY_REQUESTS, body)
            .with_provider_request_id(Some("req_abc".to_string()))
            .with_response_headers(Some(headers.clone())),
        ProviderError::from_http_response(http::StatusCode::TOO_MANY_REQUESTS, body)
            .with_response_headers(Some(headers.clone())),
        // A transport that reported the reply as an error routes through
        // the same funnel, headers included.
        ProviderError::from_transport_error(http_client::Error::InvalidStatusCodeWithDetails {
            status: http::StatusCode::TOO_MANY_REQUESTS,
            body: body.to_string(),
            headers: headers.clone(),
        }),
    ] {
        let prompt_error = PromptError::Provider(completion_error);
        assert_eq!(
            prompt_error
                .provider_response_headers()
                .and_then(|headers| headers.get(http::header::RETRY_AFTER))
                .and_then(|value| value.to_str().ok()),
            Some("20"),
            "PromptError dropped the captured headers",
        );

        let structured = StructuredOutputError::Prompt(prompt_error);
        assert_eq!(
            structured
                .provider_response_headers()
                .and_then(|headers| headers.get(http::header::RETRY_AFTER))
                .and_then(|value| value.to_str().ok()),
            Some("20"),
            "StructuredOutputError dropped the captured headers",
        );
    }
}

/// Variants that wrap no provider response report no headers.
#[test]
fn prompt_error_reports_no_headers_for_unrelated_variants() {
    let error = PromptError::Cancelled {
        chat_history: CanonicalHistory::validate(vec![Message::user("hi")])
            .expect("a lone user message is canonical"),
        reason: "cancelled".to_string(),
    };
    assert!(error.provider_response_headers().is_none());
    assert!(
        StructuredOutputError::EmptyResponse
            .provider_response_headers()
            .is_none()
    );
}

/// rig#2314: a wrapped completion error's transport request id forwards
/// through `PromptError` (and, transitively, `StructuredOutputError`).
#[test]
fn prompt_error_forwards_the_provider_request_id() {
    let error = PromptError::Provider(ProviderError::ProviderResponse(
        ProviderResponseError::new(http::StatusCode::NOT_FOUND, "{}")
            .with_provider_request_id(Some("req_failed_call".to_string())),
    ));
    assert_eq!(error.provider_request_id(), Some("req_failed_call"));
}

#[test]
fn prompt_error_provider_response_helpers_return_none_for_unrelated_variant() {
    let error = PromptError::Cancelled {
        chat_history: CanonicalHistory::validate(vec![Message::user("hi")])
            .expect("a lone user message is canonical"),
        reason: "cancelled".to_string(),
    };

    assert_eq!(error.provider_response_body(), None);
    assert_eq!(error.provider_response_status(), None);
    assert_eq!(
        error
            .provider_response_json()
            .expect("no body is not an error"),
        None
    );
}
