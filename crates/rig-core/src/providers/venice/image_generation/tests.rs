use super::*;
use crate::client::image_generation::ImageGenerationClient;
use crate::image_generation::ImageGenerationModel as _;

fn request() -> ImageGenerationRequest {
    ImageGenerationRequest {
        prompt: "a red circle on white".to_string(),
        width: 256,
        height: 256,
        additional_params: None,
    }
}

/// Venice answers a bad request with a flat `{"error": "…"}` body, not
/// OpenAI's nested error object; the shared envelope must still classify
/// it as an error and preserve the body verbatim.
#[tokio::test]
async fn image_generation_non_success_preserves_status_and_body() {
    use crate::test_utils::RecordingHttpClient;

    let body = r#"{"error":"Specified model not found: nope"}"#;
    let http_client = RecordingHttpClient::with_error_response(http::StatusCode::NOT_FOUND, body);
    let client = crate::providers::venice::Client::builder()
        .api_key("test-key")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.image_generation_model(VENICE_SD35);

    let error = model
        .image_generation(request())
        .await
        .expect_err("should fail with non-success status");

    assert!(matches!(error, ImageGenerationError::HttpError(_)));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::NOT_FOUND)
    );
    assert_eq!(error.provider_response_body(), Some(body));
}

#[tokio::test]
async fn image_generation_posts_venice_native_body() {
    use crate::test_utils::RecordingHttpClient;

    let http_client = RecordingHttpClient::new(r#"{"id":"abc","images":["aGVsbG8="]}"#);
    let client = crate::providers::venice::Client::builder()
        .api_key("test-key")
        .http_client(http_client.clone())
        .build()
        .expect("build client");
    let model = client.image_generation_model(VENICE_SD35);

    let response = model
        .image_generation(request())
        .await
        .expect("image generation should succeed");

    assert_eq!(response.image, b"hello");
    assert_eq!(response.provider, "venice");
    assert_eq!(response.raw["id"], "abc");

    let requests = http_client.requests();
    let recorded = requests.first().expect("one request");
    assert!(recorded.uri.ends_with("/image/generate"));
    let body: serde_json::Value =
        serde_json::from_slice(&recorded.body).expect("body should be JSON");
    assert_eq!(
        body,
        serde_json::json!({
            "model": VENICE_SD35,
            "prompt": "a red circle on white",
            "width": 256,
            "height": 256,
        })
    );
}
