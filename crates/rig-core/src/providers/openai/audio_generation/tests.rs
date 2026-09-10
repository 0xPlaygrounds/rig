use super::*;
use crate::audio_generation::{AudioGenerationError, AudioGenerationModel as _};
use crate::client::audio_generation::AudioGenerationClient;
use crate::providers::openai::Client;
use crate::test_utils::RecordingHttpClient;

#[tokio::test]
async fn audio_generation_non_success_preserves_status_and_body() {
    let body = r#"{"error":{"message":"boom"}}"#;
    let http_client =
        RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
    let client = Client::builder()
        .api_key("test-key")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.audio_generation_model(TTS_1);

    let request = model
        .audio_generation_request()
        .text("hello")
        .voice("alloy")
        .build();

    let error = model
        .audio_generation(request)
        .await
        .expect_err("should fail with non-success status");

    assert!(matches!(error, AudioGenerationError::HttpError(_)));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::SERVICE_UNAVAILABLE)
    );
    assert_eq!(error.provider_response_body(), Some(body));
}
