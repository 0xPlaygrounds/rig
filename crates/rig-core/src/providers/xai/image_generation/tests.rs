use super::*;
use crate::client::image_generation::ImageGenerationClient;
use crate::image_generation::ImageGenerationModel as _;

fn request() -> ImageGenerationRequest {
    ImageGenerationRequest {
        prompt: "draw a cat".to_string(),
        width: 256,
        height: 256,
        additional_params: None,
    }
}

#[tokio::test]
async fn image_generation_non_success_preserves_status_and_body() {
    use crate::test_utils::RecordingHttpClient;

    let body = r#"{"error":"boom","code":"503"}"#;
    let http_client =
        RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
    let client = crate::providers::xai::Client::builder()
        .api_key("test-key")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.image_generation_model(GROK_IMAGINE_IMAGE);

    let error = model
        .image_generation(request())
        .await
        .expect_err("should fail with non-success status");

    assert!(matches!(error, ImageGenerationError::HttpError(_)));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::SERVICE_UNAVAILABLE)
    );
    assert_eq!(error.provider_response_body(), Some(body));
}

#[tokio::test]
async fn image_generation_2xx_error_envelope_preserves_status_and_body() {
    use crate::test_utils::RecordingHttpClient;

    // Deserializes to `ApiResponse::Err(ApiErrorResponse)` on a 200 OK.
    let body = r#"{"error":"boom","code":"503"}"#;
    let http_client = RecordingHttpClient::new(body);
    let client = crate::providers::xai::Client::builder()
        .api_key("test-key")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.image_generation_model(GROK_IMAGINE_IMAGE);

    let error = model
        .image_generation(request())
        .await
        .expect_err("should fail with provider error envelope");

    match &error {
        ImageGenerationError::ProviderResponse(stored) => {
            assert_eq!(stored.body, body);
            assert_eq!(stored.status, Some(http::StatusCode::OK));
        }
        other => panic!("expected ProviderResponse, got {other:?}"),
    }
}
