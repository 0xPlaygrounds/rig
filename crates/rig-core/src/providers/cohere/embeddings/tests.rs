use super::*;
use crate::error::ProviderError;

#[tokio::test]
async fn embeddings_non_success_preserves_status_and_body() {
    use crate::embeddings::EmbeddingModel as _;
    use crate::test_utils::RecordingHttpClient;

    let body = r#"{"error":{"message":"boom"}}"#;
    let http_client =
        RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
    let model = crate::driver::Bound::new(
        crate::providers::cohere::Cohere::new("test-key"),
        http_client,
    )
    .embedding(crate::providers::cohere::EMBED_ENGLISH_V3, None);

    let error = model
        .embed_texts(["hello".to_string()])
        .await
        .expect_err("should fail with non-success status");

    assert!(matches!(error, ProviderError::ProviderResponse(_)));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::SERVICE_UNAVAILABLE)
    );
    assert_eq!(error.provider_response_body(), Some(body));
}

/// Cohere can answer `/v1/embed` with **200** and an error envelope in the
/// body. That is still the provider's reply: the caller must read the
/// status it arrived under and the bytes it arrived as, not a decode
/// failure about missing embeddings.
#[tokio::test]
async fn embeddings_2xx_error_envelope_preserves_status_and_body() {
    use crate::embeddings::EmbeddingModel as _;
    use crate::test_utils::RecordingHttpClient;

    let body = r#"{"message":"boom"}"#;
    let http_client = RecordingHttpClient::new(body); // 200 OK
    let model = crate::driver::Bound::new(
        crate::providers::cohere::Cohere::new("test-key"),
        http_client,
    )
    .embedding(crate::providers::cohere::EMBED_ENGLISH_V3, None);

    let error = model
        .embed_texts(["hello".to_string()])
        .await
        .expect_err("should fail with provider error envelope");

    let ProviderError::ProviderResponse(stored) = &error else {
        panic!("expected ProviderResponse, got {error:?}");
    };
    // Byte-equal, not "contains": a preserved reply is the provider's bytes
    // or it is a rendering of them.
    assert_eq!(stored.body, body);
    assert_eq!(stored.status, Some(http::StatusCode::OK));
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

/// The identifier an image vector carries is the shared one now
/// ([`crate::embeddings::image_document`], which the `ImageEmbedding`
/// operation seeds its fold with), and Cohere is its only caller in this
/// crate: a digest that tells two images apart and cannot be reversed into
/// the bytes the trait forbids returning.
#[test]
fn image_documents_are_stable_without_retaining_image_bytes() {
    let first = crate::embeddings::image_document(b"\x89PNG\r\n\x1a\nfirst");
    let second = crate::embeddings::image_document(b"\x89PNG\r\n\x1a\nother");

    assert_eq!(
        first,
        crate::embeddings::image_document(b"\x89PNG\r\n\x1a\nfirst")
    );
    assert_ne!(first, second);
    assert!(first.starts_with("image/png;sha256="));
    assert!(!first.contains("first"));
}

#[test]
fn image_data_url_rejects_unsupported_and_oversized_inputs() {
    assert!(matches!(
        validate_image(b"not an image").map_err(ProviderError::from),
        Err(ProviderError::Request(_))
    ));
    assert!(matches!(
        validate_image(&vec![0; MAX_IMAGE_BYTES + 1]).map_err(ProviderError::from),
        Err(ProviderError::Request(_))
    ));
}

/// Cohere's acceptance policy runs inside `ImageEmbeddings::encode`, so an
/// unsniffable image in a batch fails the whole operation before the driver
/// opens a socket.
#[tokio::test]
async fn image_batches_are_fully_validated_before_any_request() {
    use crate::embeddings::ImageEmbeddingModel as _;
    use crate::test_utils::RecordingHttpClient;

    let http_client = RecordingHttpClient::default();
    let model = crate::driver::Bound::new(
        crate::providers::cohere::Cohere::new("test-key"),
        http_client.clone(),
    )
    .image_embedding(crate::providers::cohere::EMBED_ENGLISH_V3, None);

    let error = model
        .embed_images([b"\x89PNG\r\n\x1a\n".to_vec(), b"not an image".to_vec()])
        .await
        .expect_err("invalid batch should fail before transport");

    assert!(matches!(error, ProviderError::Request(_)));
    assert!(http_client.requests().is_empty());
}

#[tokio::test]
async fn image_embeddings_non_success_preserves_status_and_body() {
    use crate::embeddings::ImageEmbeddingModel as _;
    use crate::test_utils::RecordingHttpClient;

    let body = r#"{"error":{"message":"boom"}}"#;
    let http_client =
        RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
    let model = crate::driver::Bound::new(
        crate::providers::cohere::Cohere::new("test-key"),
        http_client,
    )
    .image_embedding(crate::providers::cohere::EMBED_ENGLISH_V3, None);

    let error = model
        .embed_image(b"\x89PNG\r\n\x1a\n")
        .await
        .expect_err("should fail with non-success status");

    assert!(matches!(error, ProviderError::ProviderResponse(_)));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::SERVICE_UNAVAILABLE)
    );
    assert_eq!(error.provider_response_body(), Some(body));
}

/// The image route answers on the same `/v1/embed` endpoint, so it can
/// answer a 200 with the same envelope — and preserves it the same way.
#[tokio::test]
async fn image_embeddings_2xx_error_envelope_preserves_status_and_body() {
    use crate::embeddings::ImageEmbeddingModel as _;
    use crate::test_utils::RecordingHttpClient;

    let body = r#"{"message":"boom"}"#;
    let http_client = RecordingHttpClient::new(body); // 200 OK
    let model = crate::driver::Bound::new(
        crate::providers::cohere::Cohere::new("test-key"),
        http_client,
    )
    .image_embedding(crate::providers::cohere::EMBED_ENGLISH_V3, None);

    let error = model
        .embed_image(b"\x89PNG\r\n\x1a\n")
        .await
        .expect_err("should fail with provider error envelope");

    let ProviderError::ProviderResponse(stored) = &error else {
        panic!("expected ProviderResponse, got {error:?}");
    };
    assert_eq!(stored.body, body);
    assert_eq!(stored.status, Some(http::StatusCode::OK));
}
