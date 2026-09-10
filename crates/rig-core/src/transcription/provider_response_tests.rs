use super::*;
use crate::{http_client, provider_response};
use http::StatusCode;

#[test]
fn transcription_error_provider_response_helpers_with_preserved_json_body() {
    let body = r#"{"error":{"message":"rate limited"}}"#;
    let error = TranscriptionError::ProviderResponse(
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
fn transcription_error_provider_response_helpers_with_http_non_success() {
    let body = r#"{"error":{"message":"bad request"}}"#;
    let error = TranscriptionError::HttpError(http_client::Error::InvalidStatusCodeWithMessage(
        StatusCode::BAD_REQUEST,
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
fn transcription_error_provider_response_helpers_with_preserved_plain_text_body() {
    let error = TranscriptionError::ProviderResponse(
        provider_response::ProviderResponseError::without_status("not json".to_string()),
    );

    assert_eq!(error.provider_response_body(), Some("not json"));
    assert!(error.provider_response_json().is_err());
}

#[test]
fn transcription_error_provider_error_is_not_a_provider_response() {
    let error = TranscriptionError::ProviderError("internal diagnostic".to_string());

    assert_eq!(error.provider_response_body(), None);
    assert_eq!(error.provider_response_status(), None);
    assert_eq!(error.provider_response_json().expect("no body"), None);
}

#[test]
fn transcription_error_provider_response_helpers_with_unrelated_variant() {
    let error = TranscriptionError::ResponseError("parse failed".to_string());

    assert_eq!(error.provider_response_body(), None);
    assert_eq!(error.provider_response_status(), None);
    assert_eq!(error.provider_response_json().expect("no body"), None);
}
