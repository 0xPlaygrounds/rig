use super::*;
use crate::test_utils::RecordingHttpClient;

fn rate_limited() -> RecordingHttpClient {
    let mut headers = http::HeaderMap::new();
    headers.insert(http::header::RETRY_AFTER, "20".parse().expect("value"));
    RecordingHttpClient::with_error_response_headers(
        http::StatusCode::TOO_MANY_REQUESTS,
        r#"{"error":"slow down"}"#,
        headers,
    )
}

fn assert_retry_after(error: &TranscriptionError, driver: &str) {
    assert_eq!(
        error
            .provider_response_headers()
            .and_then(|headers| headers.get(http::header::RETRY_AFTER))
            .and_then(|value| value.to_str().ok()),
        Some("20"),
        "{driver}: Retry-After not recoverable",
    );
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::TOO_MANY_REQUESTS),
        "{driver}: status lost",
    );
}

#[tokio::test]
async fn json_driver_non_success_response_preserves_headers() {
    let error = send_json_transcription::<_, serde_json::Value>(
        &rate_limited(),
        http_client::Request::builder()
            .method(http::Method::POST)
            .uri("https://example.test/v1/audio/transcriptions"),
        b"{}".to_vec(),
        None,
        |_, _| unreachable!("a 429 never reaches the decoder"),
    )
    .await
    .expect_err("a 429 should fail");

    assert_retry_after(&error, "send_json_transcription");
}

#[tokio::test]
async fn multipart_driver_non_success_response_preserves_headers() {
    let error = send_transcription::<
        _,
        crate::providers::openai::client::ApiResponse<
            crate::providers::openai::TranscriptionResponse,
        >,
    >(
        &rate_limited(),
        http_client::Request::builder()
            .method(http::Method::POST)
            .uri("https://example.test/v1/audio/transcriptions"),
        MultipartForm::default(),
        None,
    )
    .await
    .expect_err("a 429 should fail");

    assert_retry_after(&error, "send_transcription");
}
