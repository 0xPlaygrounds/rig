use super::*;
use crate::client::image_generation::ImageGenerationClient;
use crate::image_generation::ImageGenerationModel as _;
use crate::test_utils::RecordingHttpClient;

#[tokio::test]
async fn image_generation_non_success_preserves_status_and_body() {
    let body = r#"{"error":{"message":"boom"}}"#;
    let http_client =
        RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
    let client = Client::builder()
        .api_key("test-key")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.image_generation_model(Flux1);

    let request = model
        .image_generation_request()
        .prompt("draw a cat")
        .build();

    let error = model
        .image_generation(request)
        .await
        .expect_err("should fail with non-success status");

    assert!(matches!(error, ImageGenerationError::HttpError(_)));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::SERVICE_UNAVAILABLE)
    );
    assert_eq!(error.provider_response_body(), Some(body));
}
