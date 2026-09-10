use super::*;
use crate::providers::openrouter::Client;

#[test]
fn test_infer_format_from_filename() {
    assert_eq!(infer_format_from_filename("audio.wav"), "wav");
    assert_eq!(infer_format_from_filename("audio.mp3"), "mp3");
    assert_eq!(infer_format_from_filename("audio.flac"), "flac");
    assert_eq!(infer_format_from_filename("audio.m4a"), "m4a");
    assert_eq!(infer_format_from_filename("audio.ogg"), "ogg");
    assert_eq!(infer_format_from_filename("audio.webm"), "webm");
    assert_eq!(infer_format_from_filename("audio.aac"), "aac");
    assert_eq!(infer_format_from_filename("audio.WAV"), "wav");
    assert_eq!(infer_format_from_filename("audio.MP3"), "mp3");
    assert_eq!(infer_format_from_filename("unknown"), "wav");
    assert_eq!(infer_format_from_filename("noextension"), "wav");
    assert_eq!(infer_format_from_filename("meeting.final.mp3"), "mp3");
    assert_eq!(infer_format_from_filename("audio.tar.gz"), "wav");
}

#[test]
fn test_transcription_response_deserialization() {
    let json = r#"{"text": "Hello world", "usage": {"seconds": 1.5, "cost": 0.001}}"#;
    let resp: TranscriptionResponse = serde_json::from_str(json).unwrap();
    assert_eq!(resp.text, "Hello world");
    let usage = resp.usage.unwrap();
    assert_eq!(usage.seconds, Some(1.5));
}

#[test]
fn test_transcription_response_without_usage() {
    let json = r#"{"text": "Hello world"}"#;
    let resp: TranscriptionResponse = serde_json::from_str(json).unwrap();
    assert_eq!(resp.text, "Hello world");
    assert!(resp.usage.is_none());
}

#[tokio::test]
async fn transcription_non_success_preserves_status_and_body() {
    use crate::client::transcription::TranscriptionClient;
    use crate::test_utils::RecordingHttpClient;
    use crate::transcription::TranscriptionModel as _;

    let body = r#"{"error":{"message":"boom"}}"#;
    let http_client =
        RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
    let client = Client::builder()
        .api_key("test-key")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.transcription_model(WHISPER_1);

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
