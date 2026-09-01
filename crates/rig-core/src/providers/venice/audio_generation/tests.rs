use super::*;
use crate::audio_generation::AudioGenerationModel as _;
use crate::client::audio_generation::AudioGenerationClient;
use crate::test_utils::RecordingHttpClient;

#[tokio::test]
async fn audio_generation_non_success_preserves_status_and_body() {
    let body = r#"{"error":"Insufficient USD or DIEM balance"}"#;
    let http_client =
        RecordingHttpClient::with_error_response(http::StatusCode::PAYMENT_REQUIRED, body);
    let client = crate::providers::venice::Client::builder()
        .api_key("test-key")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.audio_generation_model(TTS_KOKORO);

    let request = model
        .audio_generation_request()
        .text("hello")
        .voice("af_sky")
        .build();

    let error = model
        .audio_generation(request)
        .await
        .expect_err("should fail with non-success status");

    assert!(matches!(error, AudioGenerationError::HttpError(_)));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::PAYMENT_REQUIRED)
    );
    assert_eq!(error.provider_response_body(), Some(body));
}

/// An unset voice must not reach Venice as `""`, which it rejects.
#[tokio::test]
async fn empty_voice_falls_back_to_the_model_default() {
    let http_client = RecordingHttpClient::new("audio-bytes");
    let client = crate::providers::venice::Client::builder()
        .api_key("test-key")
        .http_client(http_client.clone())
        .build()
        .expect("build client");
    let model = client.audio_generation_model(TTS_KOKORO);

    let request = model
        .audio_generation_request()
        .text("hello")
        .voice("")
        .build();
    model
        .audio_generation(request)
        .await
        .expect("audio generation should succeed");

    let requests = http_client.requests();
    let recorded = requests.first().expect("one request");
    assert!(recorded.uri.ends_with("/audio/speech"));
    let body: serde_json::Value =
        serde_json::from_slice(&recorded.body).expect("body should be JSON");
    assert_eq!(body["voice"], DEFAULT_VOICE);
    assert_eq!(body["model"], TTS_KOKORO);
    assert_eq!(body["input"], "hello");
}
