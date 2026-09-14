use super::*;
use crate::providers::internal::envelope::DirectPayload;
use crate::test_utils::RecordingHttpClient;

/// Minimal payload satisfying the driver's `TryInto` bound; the 429 path
/// returns before any decoding, so its conversion is never reached.
#[derive(serde::Deserialize)]
struct Payload;

#[tokio::test]
async fn non_success_response_preserves_headers() {
    let mut headers = http::HeaderMap::new();
    headers.insert(http::header::RETRY_AFTER, "20".parse().expect("value"));
    let client = RecordingHttpClient::with_error_response_headers(
        http::StatusCode::TOO_MANY_REQUESTS,
        r#"{"error":"slow down"}"#,
        headers,
    );

    let error = send_image_generation::<_, DirectPayload<Payload>>(
        &client,
        http_client::Request::builder()
            .method(http::Method::POST)
            .uri("https://example.test/v1/images/generations"),
        serde_json::json!({}),
        None,
    )
    .await
    .err()
    .expect("a 429 should fail");

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
