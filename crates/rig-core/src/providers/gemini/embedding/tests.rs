use super::*;
use crate::client::EmbeddingsClient;

#[test]
fn test_embedding_values_deserializes_without_empty_values_field() {
    let values: gemini_api_types::EmbeddingValues =
        serde_json::from_str("{}").expect("empty embedding values should deserialize");
    assert!(values.values.is_empty());
}

#[test]
fn test_model_default_ndims_lookup() {
    assert_eq!(model_default_ndims(EMBEDDING_001), Some(3072));
    assert_eq!(model_default_ndims(EMBEDDING_004), Some(768));
    assert_eq!(model_default_ndims("unknown-model"), None);
}

#[test]
fn test_make_resolves_default_dims() {
    let client =
        Client::new_with("test_key", crate::test_utils::RecordingHttpClient::new("")).unwrap();

    // EMBEDDING_001 defaults to 3072
    let model = client.embedding_model(EMBEDDING_001);
    assert_eq!(embeddings::EmbeddingModel::ndims(&model), 3072);

    // EMBEDDING_004 defaults to 768
    let model = client.embedding_model(EMBEDDING_004);
    assert_eq!(embeddings::EmbeddingModel::ndims(&model), 768);

    // Unknown model falls back to 768
    let model = client.embedding_model("some-future-model");
    assert_eq!(embeddings::EmbeddingModel::ndims(&model), 768);
}

#[test]
fn test_make_respects_explicit_dims() {
    let client =
        Client::new_with("test_key", crate::test_utils::RecordingHttpClient::new("")).unwrap();

    let model = client.embedding_model_with_ndims(EMBEDDING_001, 256);
    assert_eq!(embeddings::EmbeddingModel::ndims(&model), 256);
}

#[test]
fn test_new_uses_provided_ndims() {
    let client =
        Client::new_with("test_key", crate::test_utils::RecordingHttpClient::new("")).unwrap();

    let model = EmbeddingModel::new(client, EMBEDDING_001, 512);
    assert_eq!(embeddings::EmbeddingModel::ndims(&model), 512);
}

#[tokio::test]
async fn embedding_non_success_preserves_status_and_body() {
    use crate::client::embeddings::EmbeddingsClient;
    use crate::embeddings::EmbeddingModel as _;
    use crate::test_utils::RecordingHttpClient;

    // The non-success status guard preserves the raw provider body without
    // depending on its envelope shape.
    let body = r#"{"error":{"code":503,"message":"service unavailable","status":"UNAVAILABLE"}}"#;
    let http_client =
        RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
    let client = Client::builder()
        .api_key("test-key")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.embedding_model(EMBEDDING_001);

    let error = model
        .embed_texts(vec!["hello".to_string()])
        .await
        .expect_err("should fail with non-success status");

    assert!(matches!(error, EmbeddingError::HttpError(_)));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::SERVICE_UNAVAILABLE)
    );
    assert_eq!(error.provider_response_body(), Some(body));
}

#[tokio::test]
async fn embedding_2xx_error_envelope_preserves_status_and_body() {
    use crate::client::embeddings::EmbeddingsClient;
    use crate::embeddings::EmbeddingModel as _;
    use crate::test_utils::RecordingHttpClient;

    // 200 OK carrying Gemini's standard nested error envelope.
    let body = r#"{"error":{"code":503,"message":"boom","status":"UNAVAILABLE"}}"#;
    let http_client = RecordingHttpClient::new(body); // 200 OK
    let client = Client::builder()
        .api_key("test-key")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.embedding_model(EMBEDDING_001);

    let error = model
        .embed_texts(vec!["hello".to_string()])
        .await
        .expect_err("should fail with provider error envelope");

    match &error {
        EmbeddingError::ProviderResponse(stored) => {
            assert_eq!(stored.body, body);
            assert_eq!(stored.status, Some(http::StatusCode::OK));
        }
        other => panic!("expected ProviderResponse, got {other:?}"),
    }
}
