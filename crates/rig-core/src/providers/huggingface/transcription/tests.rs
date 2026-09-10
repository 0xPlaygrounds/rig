use super::*;
use crate::client::transcription::TranscriptionClient;
use crate::providers::huggingface::Client;
use crate::test_utils::RecordingHttpClient;
use crate::transcription::TranscriptionModel as _;

#[tokio::test]
async fn transcription_non_success_preserves_status_and_body() {
    let body = r#"{"error":{"message":"boom"}}"#;
    let http_client =
        RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
    let client = Client::builder()
        .api_key("test-key")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.transcription_model(WHISPER_LARGE_V3);

    let request = model.transcription_request().data(vec![0u8; 16]).build();

    let error = model
        .transcription(request)
        .await
        .expect_err("should fail with non-success status");

    assert!(matches!(error, TranscriptionError::HttpError(_)));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::SERVICE_UNAVAILABLE)
    );
    assert_eq!(error.provider_response_body(), Some(body));
}

#[tokio::test]
async fn transcription_2xx_error_envelope_preserves_status_and_body() {
    // A 200 OK body that is not a valid `TranscriptionResponse` (no `text`
    // field) falls through the untagged `ApiResponse` to its `Err(Value)`
    // variant, which the provider routes through `from_http_response`.
    let body = r#"{"error":"Model openai/whisper-large-v3 is currently loading"}"#;
    let http_client = RecordingHttpClient::new(body);
    let client = Client::builder()
        .api_key("test-key")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.transcription_model(WHISPER_LARGE_V3);

    let request = model.transcription_request().data(vec![0u8; 16]).build();

    let error = model
        .transcription(request)
        .await
        .expect_err("should fail with provider error envelope");

    match &error {
        TranscriptionError::ProviderResponse(stored) => {
            assert_eq!(stored.body, body);
            assert_eq!(stored.status, Some(http::StatusCode::OK));
        }
        other => panic!("expected ProviderResponse, got {other:?}"),
    }
}
