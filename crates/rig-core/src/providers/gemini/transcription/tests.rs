use super::*;
use crate::client::transcription::TranscriptionClient;
use crate::providers::gemini::Client;
use crate::providers::gemini::completion::GEMINI_2_0_FLASH;
use crate::test_utils::RecordingHttpClient;
use crate::transcription::TranscriptionModel as _;

fn transcription_request() -> transcription::TranscriptionRequest {
    transcription::TranscriptionRequest {
        data: b"audio bytes".to_vec(),
        filename: "audio.mp3".to_string(),
        language: None,
        prompt: None,
        temperature: None,
        additional_params: None,
    }
}

#[tokio::test]
async fn transcription_non_success_preserves_status_and_body() {
    let body = r#"{"error":{"code":503,"message":"boom","status":"UNAVAILABLE"}}"#;
    let http_client =
        RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
    let client = Client::builder()
        .api_key("test-key")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.transcription_model(GEMINI_2_0_FLASH);

    let error = model
        .transcription(transcription_request())
        .await
        .expect_err("should fail with non-success status");

    assert!(matches!(error, TranscriptionError::HttpError(_)));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::SERVICE_UNAVAILABLE)
    );
    assert_eq!(error.provider_response_body(), Some(body));
}
