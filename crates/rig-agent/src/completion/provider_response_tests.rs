use rig_core::{ProviderResponseError, http_client};

use super::*;

#[test]
fn prompt_error_forwards_provider_response_to_completion_error() {
    let body = r#"{"error":{"message":"boom"}}"#;
    let inner = CompletionError::from_http_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
    let error = PromptError::CompletionError(inner);

    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::SERVICE_UNAVAILABLE),
    );
    assert_eq!(error.provider_response_body(), Some(body));
    assert_eq!(
        error
            .provider_response_json()
            .expect("valid json")
            .expect("present json")["error"]["message"],
        "boom",
    );
}

#[test]
fn prompt_error_provider_response_helpers_forward_http_status_and_body() {
    let body = r#"{"error":{"message":"unauthorized"}}"#;
    let error = PromptError::CompletionError(CompletionError::HttpError(
        http_client::Error::InvalidStatusCodeWithMessage(
            http::StatusCode::UNAUTHORIZED,
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

#[test]
fn prompt_error_provider_response_helpers_forward_wrapped_completion_error() {
    let body = r#"{"error":{"code":"invalid_request","message":"bad input"}}"#;
    let error = PromptError::CompletionError(CompletionError::ProviderResponse(
        ProviderResponseError::without_status(body),
    ));

    assert_eq!(error.provider_response_body(), Some(body));
    assert_eq!(error.provider_response_status(), None);
    // rig#2314: the transport request id forwards through the wrapper too.
    assert_eq!(error.provider_request_id(), None);
    assert_eq!(
        error.provider_response_json().expect("valid JSON body"),
        Some(serde_json::json!({
            "error": {
                "code": "invalid_request",
                "message": "bad input"
            }
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
        // Contract provider: headers live on the ProviderResponse.
        CompletionError::from_http_response_with_request_id(
            http::StatusCode::TOO_MANY_REQUESTS,
            body,
            Some("req_abc".to_string()),
        )
        .with_response_headers(Some(Box::new(headers.clone()))),
        // Contract-less provider: headers live on the transport error.
        CompletionError::from_http_response(http::StatusCode::TOO_MANY_REQUESTS, body)
            .with_response_headers(Some(Box::new(headers.clone()))),
    ] {
        let prompt_error = PromptError::CompletionError(completion_error);
        assert_eq!(
            prompt_error
                .provider_response_headers()
                .and_then(|headers| headers.get(http::header::RETRY_AFTER))
                .and_then(|value| value.to_str().ok()),
            Some("20"),
            "PromptError dropped the captured headers",
        );

        let structured = StructuredOutputError::PromptError(Box::new(prompt_error));
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
    let error = PromptError::PromptCancelled {
        chat_history: vec![Message::user("hi")],
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
    let error = PromptError::CompletionError(CompletionError::ProviderResponse(
        ProviderResponseError::new(http::StatusCode::NOT_FOUND, "{}")
            .with_provider_request_id(Some("req_failed_call".to_string())),
    ));
    assert_eq!(error.provider_request_id(), Some("req_failed_call"));
}

#[test]
fn prompt_error_provider_response_helpers_return_none_for_unrelated_variant() {
    let error = PromptError::PromptCancelled {
        chat_history: vec![Message::user("hi")],
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

#[test]
fn structured_output_error_provider_response_helpers_forward_prompt_error() {
    let body = r#"{"error":{"message":"bad input"}}"#;
    let error = StructuredOutputError::PromptError(Box::new(PromptError::CompletionError(
        CompletionError::ProviderResponse(ProviderResponseError::new(
            http::StatusCode::BAD_REQUEST,
            body,
        )),
    )));

    assert_eq!(error.provider_response_body(), Some(body));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::BAD_REQUEST)
    );
}
