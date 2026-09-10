#[test]
fn test_client_initialization() {
    let _client = crate::providers::voyageai::Client::new_with(
        "dummy-key",
        crate::test_utils::RecordingHttpClient::new(""),
    )
    .expect("Client::new() failed");
    let _client_from_builder = crate::providers::voyageai::Client::builder()
        .api_key("dummy-key")
        .http_client(crate::test_utils::RecordingHttpClient::new(""))
        .build()
        .expect("Client::builder() failed");
}

#[tokio::test]
async fn rerank_non_success_preserves_status_and_body() {
    use crate::client::RerankingClient;
    use crate::rerank::{RerankError, RerankModel as _};
    use crate::test_utils::RecordingHttpClient;

    let body = r#"{"error":{"message":"boom"}}"#;
    let http_client =
        RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
    let client = super::Client::builder()
        .api_key("test-key")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.rerank_model(super::RERANK_2_5);

    let error = model
        .rerank("query", vec!["doc one".to_string(), "doc two".to_string()])
        .await
        .expect_err("rerank should fail with non-success status");

    assert!(matches!(error, RerankError::HttpError(_)));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::SERVICE_UNAVAILABLE)
    );
    assert_eq!(error.provider_response_body(), Some(body));
}

#[tokio::test]
async fn rerank_2xx_error_envelope_preserves_status_and_body() {
    use crate::client::RerankingClient;
    use crate::rerank::{RerankError, RerankModel as _};
    use crate::test_utils::RecordingHttpClient;

    let body = r#"{"message":"boom"}"#;
    let http_client = RecordingHttpClient::new(body); // 200 OK
    let client = super::Client::builder()
        .api_key("test-key")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.rerank_model(super::RERANK_2_5);

    let error = model
        .rerank("query", vec!["doc one".to_string(), "doc two".to_string()])
        .await
        .expect_err("rerank should fail with provider error envelope");

    match &error {
        RerankError::ProviderResponse(stored) => {
            assert_eq!(stored.body, body);
            assert_eq!(stored.status, Some(http::StatusCode::OK));
        }
        other => panic!("expected ProviderResponse, got {other:?}"),
    }
}

#[tokio::test]
async fn embedding_request_includes_options_when_set() {
    use crate::client::EmbeddingsClient;
    use crate::embeddings::EmbeddingModel as _;
    use crate::test_utils::RecordingHttpClient;

    let response_body = r#"{
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}],
            "model": "voyage-3-large",
            "usage": {"total_tokens": 7}
        }"#;
    let http_client = RecordingHttpClient::new(response_body);
    let client = super::Client::builder()
        .api_key("test-key")
        .http_client(http_client.clone())
        .build()
        .expect("build client");
    let model = client.embedding_model(super::VOYAGE_3_LARGE);

    model
        .with_options(super::EmbeddingOptions {
            input_type: Some("document".to_string()),
            truncation: Some(true),
            output_dimension: Some(256),
        })
        .embed_texts_response(vec!["doc".to_string()])
        .await
        .expect("embed should succeed");

    let captured = http_client.requests();
    assert_eq!(captured.len(), 1);
    let body: serde_json::Value =
        serde_json::from_slice(&captured[0].body).expect("request body is valid JSON");
    assert_eq!(body["model"], super::VOYAGE_3_LARGE);
    assert_eq!(body["input_type"], "document");
    assert_eq!(body["truncation"], true);
    assert_eq!(body["output_dimension"], serde_json::json!(256));
    assert_eq!(body.get("output_dtype"), None);
}

#[tokio::test]
async fn embedding_request_omits_options_when_unset() {
    use crate::client::EmbeddingsClient;
    use crate::embeddings::EmbeddingModel as _;
    use crate::test_utils::RecordingHttpClient;

    let response_body = r#"{
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}],
            "model": "voyage-3-large",
            "usage": {"total_tokens": 7}
        }"#;
    let http_client = RecordingHttpClient::new(response_body);
    let client = super::Client::builder()
        .api_key("test-key")
        .http_client(http_client.clone())
        .build()
        .expect("build client");
    let model = client.embedding_model(super::VOYAGE_3_LARGE);

    model
        .embed_texts_response(vec!["doc".to_string()])
        .await
        .expect("embed should succeed");

    let captured = http_client.requests();
    assert_eq!(captured.len(), 1);
    let body: serde_json::Value =
        serde_json::from_slice(&captured[0].body).expect("request body is valid JSON");
    assert_eq!(body["model"], super::VOYAGE_3_LARGE);
    assert_eq!(body.get("input_type"), None);
    assert_eq!(body.get("truncation"), None);
    assert_eq!(body.get("output_dimension"), None);
}
