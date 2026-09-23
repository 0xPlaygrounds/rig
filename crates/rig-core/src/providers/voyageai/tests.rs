/// The driver preserves Voyage's status and body on the rerank route: a
/// caller reading a 503 must see what Voyage actually said.
#[tokio::test]
async fn rerank_non_success_preserves_status_and_body() {
    use crate::error::ProviderError;
    use crate::rerank::RerankModel as _;
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

    assert!(matches!(error, ProviderError::ProviderResponse(_)));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::SERVICE_UNAVAILABLE)
    );
    assert_eq!(error.provider_response_body(), Some(body));
}

/// Voyage can answer `/rerank` with **200** and an error envelope in the
/// body. That is still the provider's reply: the caller must read the
/// status it arrived under and the bytes it arrived as, not a decode
/// failure about a missing ordering.
#[tokio::test]
async fn rerank_2xx_error_envelope_preserves_status_and_body() {
    use crate::error::ProviderError;
    use crate::rerank::RerankModel as _;
    use crate::test_utils::RecordingHttpClient;

    let body = r#"{"message":"boom"}"#;
    let http_client = RecordingHttpClient::new(body); // 200 OK
    let model = crate::driver::Bound::new(super::VoyageAi::new("test-key"), http_client)
        .rerank(super::RERANK_2_5);

    let error = model
        .rerank("query", vec!["doc one".to_string(), "doc two".to_string()])
        .await
        .expect_err("rerank should fail with provider error envelope");

    let ProviderError::ProviderResponse(stored) = &error else {
        panic!("expected ProviderResponse, got {error:?}");
    };
    // Byte-equal, not "contains": a preserved reply is the provider's bytes
    // or it is a rendering of them.
    assert_eq!(stored.body, body);
    assert_eq!(stored.status, Some(http::StatusCode::OK));
}
