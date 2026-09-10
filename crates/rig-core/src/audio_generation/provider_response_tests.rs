use super::*;
use crate::{http_client, provider_response};
use http::StatusCode;

#[test]
fn audio_generation_error_provider_response_helpers_with_preserved_json_body() {
    let body = r#"{"error":{"message":"invalid voice"}}"#;
    let error = AudioGenerationError::ProviderResponse(
        provider_response::ProviderResponseError::without_status(body.to_string()),
    );

    assert_eq!(error.provider_response_body(), Some(body));
    assert_eq!(error.provider_response_status(), None);
    assert_eq!(
        error.provider_response_json().expect("valid JSON"),
        Some(serde_json::json!({ "error": { "message": "invalid voice" } }))
    );
}

#[test]
fn audio_generation_error_provider_response_helpers_with_http_non_success() {
    let body = r#"{"error":{"message":"bad request"}}"#;
    let error = AudioGenerationError::HttpError(http_client::Error::InvalidStatusCodeWithMessage(
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
fn audio_generation_error_provider_error_is_not_a_provider_response() {
    let error = AudioGenerationError::ProviderError("internal diagnostic".to_string());

    assert_eq!(error.provider_response_body(), None);
    assert_eq!(error.provider_response_status(), None);
    assert_eq!(error.provider_response_json().expect("no body"), None);
}

#[test]
fn audio_generation_error_provider_response_helpers_with_unrelated_variant() {
    let error = AudioGenerationError::ResponseError("parse failed".to_string());

    assert_eq!(error.provider_response_body(), None);
    assert_eq!(error.provider_response_status(), None);
    assert_eq!(error.provider_response_json().expect("no body"), None);
}
