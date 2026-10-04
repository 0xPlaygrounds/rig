use super::*;
use crate::provider_response;

#[test]
fn image_generation_error_provider_response_helpers_with_preserved_json_body() {
    let body = r#"{"error":{"message":"content policy"}}"#;
    let error = ProviderError::ProviderResponse(
        provider_response::ProviderResponseError::without_status(body.to_string()),
    );

    assert_eq!(error.provider_response_body(), Some(body));
    assert_eq!(error.provider_response_status(), None);
    assert_eq!(
        error.provider_response_json().expect("valid JSON"),
        Some(serde_json::json!({ "error": { "message": "content policy" } }))
    );
}
