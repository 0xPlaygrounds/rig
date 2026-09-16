/// The driver preserves Voyage's status and body on the rerank route: a
/// caller reading a 503 must see what Voyage actually said.
#[tokio::test]
async fn rerank_non_success_preserves_status_and_body() {
    use crate::rerank::{RerankError, RerankModel as _};
    use crate::test_utils::RecordingHttpClient;

    let body = r#"{"error":{"message":"boom"}}"#;
    let http_client =
        RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
    let model = crate::driver::Bound::new(super::VoyageAi::new("test-key"), http_client)
        .rerank(super::RERANK_2_5);

    let error = model
        .rerank("query", vec!["doc one".to_string(), "doc two".to_string()])
        .await
        .expect_err("rerank should fail with non-success status");

    assert!(matches!(error, RerankError::ProviderResponse(_)));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::SERVICE_UNAVAILABLE)
    );
    assert_eq!(error.provider_response_body(), Some(body));
}
