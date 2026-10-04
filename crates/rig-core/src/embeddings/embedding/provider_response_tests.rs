use super::*;
use crate::provider_response;

#[test]
fn embedding_error_provider_response_helpers_with_preserved_json_body() {
    let body = r#"{"error":{"message":"rate limited"}}"#;
    let error = ProviderError::ProviderResponse(
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
fn embedding_error_provider_response_helpers_with_preserved_plain_text_body() {
    let error = ProviderError::ProviderResponse(
        provider_response::ProviderResponseError::without_status("not json".to_string()),
    );

    assert_eq!(error.provider_response_body(), Some("not json"));
    assert!(error.provider_response_json().is_err());
}
