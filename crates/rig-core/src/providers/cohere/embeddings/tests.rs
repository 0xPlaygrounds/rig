use super::*;
use crate::test_utils::json_body;
use crate::test_utils::{MockHttpResponse, RecordingHttpClient, SequencedHttpClient};

fn cohere() -> CohereConfig {
    CohereConfig::new("cohere-test-key")
}

#[test]
fn embedding_dimensions_cover_every_live_embed_model() {
    assert_eq!(model_dimensions_from_identifier(EMBED_V4), Some(1_536));
    assert_eq!(
        model_dimensions_from_identifier(EMBED_ENGLISH_V3),
        Some(1_024)
    );
    assert_eq!(
        model_dimensions_from_identifier(EMBED_MULTILINGUAL_V3),
        Some(1_024)
    );
    assert_eq!(
        model_dimensions_from_identifier(EMBED_ENGLISH_LIGHT_V3),
        Some(384)
    );
    assert_eq!(
        model_dimensions_from_identifier(EMBED_MULTILINGUAL_LIGHT_V3),
        Some(384)
    );
    assert_eq!(model_dimensions_from_identifier("embed-unknown"), None);
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

/// The image route answers on the same `/v1/embed` endpoint, so it can
/// answer a 200 with the same envelope — and preserves it the same way.
#[tokio::test]
async fn image_embeddings_2xx_error_envelope_preserves_status_and_body() {
    use crate::test_utils::RecordingHttpClient;

    let body = r#"{"message":"boom"}"#;
    let http_client = RecordingHttpClient::new(body); // 200 OK
    let model = crate::driver::Model::new(
        crate::providers::cohere::CohereConfig::new("test-key").image_embedding(),
        http_client,
    );

    let error = model
        .call(vec![b"\x89PNG\r\n\x1a\n".to_vec()])
        .await
        .expect_err("should fail with provider error envelope");

    let ProviderError::ProviderResponse(stored) = &error else {
        panic!("expected ProviderResponse, got {error:?}");
    };
    assert_eq!(stored.body, body);
    assert_eq!(stored.status, Some(http::StatusCode::OK));
}

/// A `/v1/embed` reply, shaped as `crates/rig-cassette/fixtures/cassettes/cohere/embeddings/
/// embed_texts_smoke.yaml` records it (`id`, `embeddings`, `texts`, and
/// `meta.billed_units`), with two-element vectors in place of the recorded
/// 1024-element ones.
const EMBED_BODY: &str = r#"{"id":"b2e4b0f7-0000-0000-0000-000000000000","texts":["first","second"],"embeddings":[[0.5,-0.25],[0.125,0.0]],"meta":{"api_version":{"version":"1"},"billed_units":{"input_tokens":7,"search_units":0,"classifications":0,"images":0}}}"#;

#[tokio::test]
async fn an_embedding_reply_pairs_its_vectors_with_the_texts_that_were_sent() {
    let response = crate::driver::Model::new(
        cohere().embedding("embed-v4.0", None),
        RecordingHttpClient::new(EMBED_BODY),
    )
    .call(vec!["first".to_owned(), "second".to_owned()])
    .await
    .expect("the reply decodes");

    assert_eq!(
        response
            .embeddings
            .iter()
            .map(|embedding| (embedding.document.as_str(), embedding.vec.as_slice()))
            .collect::<Vec<_>>(),
        vec![
            ("first", [0.5, -0.25].as_slice()),
            ("second", [0.125, 0.0].as_slice()),
        ]
    );
    assert_eq!(response.usage.input_tokens, Some(7));
    assert_eq!(response.usage.total_tokens, Some(7));
    assert_eq!(
        response.response_id.as_deref(),
        Some("b2e4b0f7-0000-0000-0000-000000000000")
    );
}

/// A one-pixel PNG-headed byte string: enough for the media-type sniff the
/// wire runs before it builds a request.
fn png(tail: &[u8]) -> Vec<u8> {
    let mut bytes = b"\x89PNG\r\n\x1a\n".to_vec();
    bytes.extend_from_slice(tail);
    bytes
}

/// Cohere embeds ONE image per call: a larger batch is the caller's to
/// split, and the one image travels as a data URL.
#[test]
fn an_image_request_carries_exactly_one_image() {
    let wire = cohere().image_embedding();
    let refused = wire.encode(vec![png(b"first"), png(b"second")], Mode::Unary);
    assert!(refused.is_err(), "two images are two calls");

    let encoded = wire
        .encode(vec![png(b"first")], Mode::Unary)
        .expect("the image is a PNG");
    let body = json_body(&encoded.request);
    assert!(
        body["images"][0]
            .as_str()
            .is_some_and(|url| url.starts_with("data:image/png;base64,"))
    );
}

/// One `/v1/embed` image reply, shaped as the recorded image cells are:
/// `embeddings.float` holds exactly one vector.
fn image_reply(first: f64) -> MockHttpResponse {
    MockHttpResponse::success(format!(
        r#"{{"id":"img-{first}","embeddings":{{"float":[[{first},1.0]]}},"meta":{{"api_version":{{"version":"1"}},"billed_units":{{"search_units":0,"classifications":0,"images":1}}}}}}"#
    ))
}

#[tokio::test]
async fn a_single_image_embed_captures_the_bare_document() {
    let http = SequencedHttpClient::new([image_reply(0.5)]);
    let response = crate::driver::Model::new(cohere().image_embedding(), http.clone())
        .call(vec![png(b"only")])
        .await
        .expect("the reply decodes");

    assert_eq!(http.requests().len(), 1);
    // One request, one page: `raw` is that document itself, not a one-element
    // array, so a single-image embed reads the same as any non-batched wire.
    assert!(
        response.raw.is_object(),
        "one page is captured bare: {}",
        response.raw
    );
    let page: super::super::embeddings::ImageEmbeddingResponse =
        serde_json::from_value(response.raw.clone()).expect("raw is Cohere's own answer");
    assert_eq!(page.id.as_deref(), Some("img-0.5"));
}
