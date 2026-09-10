use super::*;

#[tokio::test]
async fn embeddings_non_success_preserves_status_and_body() {
    use crate::embeddings::EmbeddingModel as _;
    use crate::test_utils::RecordingHttpClient;

    let body = r#"{"error":{"message":"boom"}}"#;
    let http_client =
        RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
    let client = crate::providers::cohere::Client::builder()
        .api_key("test-key")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.embedding_model(
        crate::providers::cohere::EMBED_ENGLISH_V3,
        "search_document",
    );

    let error = model
        .embed_texts(["hello".to_string()])
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
async fn embeddings_2xx_error_envelope_preserves_status_and_body() {
    use crate::embeddings::EmbeddingModel as _;
    use crate::test_utils::RecordingHttpClient;

    // Deserializes to `ApiResponse::Err(ApiErrorResponse { message })` on a 200 OK.
    let body = r#"{"message":"boom"}"#;
    let http_client = RecordingHttpClient::new(body);
    let client = crate::providers::cohere::Client::builder()
        .api_key("test-key")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.embedding_model(
        crate::providers::cohere::EMBED_ENGLISH_V3,
        "search_document",
    );

    let error = model
        .embed_texts(["hello".to_string()])
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

#[test]
fn image_data_urls_detect_every_cohere_image_format() {
    let cases: &[(&[u8], &str)] = &[
        (b"\x89PNG\r\n\x1a\n", "image/png"),
        (b"\xff\xd8\xff", "image/jpeg"),
        (b"GIF89a", "image/gif"),
        (b"RIFF\0\0\0\0WEBP", "image/webp"),
    ];

    for &(bytes, expected_media_type) in cases {
        let result = validate_image(bytes);
        assert!(
            matches!(result, Ok(media_type) if media_type == expected_media_type),
            "expected {expected_media_type}"
        );
        assert!(
            image_data_url(bytes, expected_media_type)
                .starts_with(&format!("data:{expected_media_type};base64,"))
        );
    }
}

#[test]
fn image_documents_are_stable_without_retaining_image_bytes() {
    let first = image_document(b"\x89PNG\r\n\x1a\nfirst", "image/png");
    let second = image_document(b"\x89PNG\r\n\x1a\nother", "image/png");

    assert_eq!(
        first,
        image_document(b"\x89PNG\r\n\x1a\nfirst", "image/png")
    );
    assert_ne!(first, second);
    assert!(first.starts_with("image/png;sha256="));
    assert!(!first.contains("first"));
}

#[test]
fn image_data_url_rejects_unsupported_and_oversized_inputs() {
    assert!(matches!(
        validate_image(b"not an image"),
        Err(EmbeddingError::DocumentError(_))
    ));
    assert!(matches!(
        validate_image(&vec![0; MAX_IMAGE_BYTES + 1]),
        Err(EmbeddingError::DocumentError(_))
    ));
}

#[tokio::test]
async fn image_batches_are_fully_validated_before_any_request() {
    use crate::embeddings::ImageEmbeddingModel as _;
    use crate::test_utils::RecordingHttpClient;

    let http_client = RecordingHttpClient::default();
    let client = crate::providers::cohere::Client::builder()
        .api_key("test-key")
        .http_client(http_client.clone())
        .build()
        .expect("build client");

    let error = client
        .image_embedding_model()
        .embed_images([b"\x89PNG\r\n\x1a\n".to_vec(), b"not an image".to_vec()])
        .await
        .expect_err("invalid batch should fail before transport");

    assert!(matches!(error, EmbeddingError::DocumentError(_)));
    assert!(http_client.requests().is_empty());
}

#[tokio::test]
async fn image_embeddings_non_success_preserves_status_and_body() {
    use crate::embeddings::ImageEmbeddingModel as _;
    use crate::test_utils::RecordingHttpClient;

    let body = r#"{"error":{"message":"boom"}}"#;
    let http_client =
        RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
    let client = crate::providers::cohere::Client::builder()
        .api_key("test-key")
        .http_client(http_client)
        .build()
        .expect("build client");

    let error = client
        .image_embedding_model()
        .embed_image(b"\x89PNG\r\n\x1a\n")
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
async fn image_embeddings_2xx_error_envelope_preserves_status_and_body() {
    use crate::embeddings::ImageEmbeddingModel as _;
    use crate::test_utils::RecordingHttpClient;

    let body = r#"{"message":"boom"}"#;
    let http_client = RecordingHttpClient::new(body);
    let client = crate::providers::cohere::Client::builder()
        .api_key("test-key")
        .http_client(http_client)
        .build()
        .expect("build client");

    let error = client
        .image_embedding_model()
        .embed_image(b"\x89PNG\r\n\x1a\n")
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
