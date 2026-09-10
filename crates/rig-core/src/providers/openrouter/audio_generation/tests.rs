use super::*;
use crate::audio_generation::AudioGenerationModel as _;
use crate::client::audio_generation::AudioGenerationClient;
use crate::providers::openrouter::Client;
use crate::test_utils::RecordingHttpClient;
use bytes::Bytes;

#[tokio::test]
async fn shared_driver_keeps_openrouter_request_and_binary_response() {
    let http_client = RecordingHttpClient::new(Bytes::from_static(b"audio"));
    let client = Client::builder()
        .api_key("test-key")
        .http_client(http_client.clone())
        .build()
        .expect("build client");
    let model = client.audio_generation_model(GPT_4O_MINI_TTS);

    let response = model
        .audio_generation(
            model
                .audio_generation_request()
                .text("hello")
                .voice("alloy")
                .build(),
        )
        .await
        .expect("audio generation should succeed");

    assert_eq!(response.audio, b"audio");
    let requests = http_client.requests();
    assert_eq!(requests[0].uri, "https://openrouter.ai/api/v1/audio/speech");
    assert_eq!(
        requests[0]
            .headers
            .get(http::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("application/json")
    );
    let body: serde_json::Value =
        serde_json::from_slice(&requests[0].body).expect("request body should be JSON");
    assert_eq!(body["model"], GPT_4O_MINI_TTS);
    assert_eq!(body["input"], "hello");
    assert_eq!(body["voice"], "alloy");
}

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
    let model = client.audio_generation_model(GPT_4O_MINI_TTS);

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
