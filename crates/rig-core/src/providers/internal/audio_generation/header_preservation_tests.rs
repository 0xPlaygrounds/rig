use super::*;
use crate::test_utils::RecordingHttpClient;

#[tokio::test]
async fn non_success_response_preserves_headers() {
    let mut headers = http::HeaderMap::new();
    headers.insert(http::header::RETRY_AFTER, "20".parse().expect("value"));
    let client = RecordingHttpClient::with_error_response_headers(
        http::StatusCode::TOO_MANY_REQUESTS,
        r#"{"error":"slow down"}"#,
        headers,
    );

    let error = send_audio_generation(
        &client,
        http_client::Request::builder()
            .method(http::Method::POST)
            .uri("https://example.test/v1/audio/speech"),
        serde_json::json!({}),
        None,
    )
    .await
    .expect_err("a 429 should fail");

    assert_eq!(
        error
            .provider_response_headers()
            .and_then(|headers| headers.get(http::header::RETRY_AFTER))
            .and_then(|value| value.to_str().ok()),
        Some("20"),
    );
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::TOO_MANY_REQUESTS)
    );
}
