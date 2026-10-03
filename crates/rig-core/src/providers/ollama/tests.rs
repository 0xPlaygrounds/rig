use super::*;
use crate::error::ProviderError;

// Proves a non-success HTTP response from `/api/embed` preserves the
// provider's status + body through the `provider_response_*` helpers
// (issue #1931).
#[tokio::test]
async fn embeddings_non_success_preserves_status_and_body() {
    use crate::test_utils::RecordingHttpClient;

    let body = r#"{"error":"model not found"}"#;
    let http_client =
        RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
    let model =
        crate::driver::Model::new(OllamaConfig::new().embedding(ALL_MINILM, None), http_client);

    let error = model
        .call(vec!["hello".to_string()])
        .await
        .map(|response| response.embeddings)
        .expect_err("should fail with non-success status");

    assert!(matches!(error, ProviderError::ProviderResponse(_)));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::SERVICE_UNAVAILABLE)
    );
    assert_eq!(error.provider_response_body(), Some(body));
}
