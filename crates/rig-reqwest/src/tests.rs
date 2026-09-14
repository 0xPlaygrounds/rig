use super::*;
use http::StatusCode;

/// rig#2210: the bundled transport's own error constructor is where the
/// headers are captured, so drive it with a real `reqwest::Response`.
#[tokio::test]
async fn non_success_status_error_preserves_response_headers() {
    let response = http::Response::builder()
        .status(StatusCode::TOO_MANY_REQUESTS)
        .header("retry-after", "20")
        .header("x-ratelimit-remaining", "0")
        .body(r#"{"error":{"message":"rate limited"}}"#)
        .expect("valid response");

    let error = non_success_status_error(reqwest::Response::from(response)).await;

    assert!(matches!(
        &error,
        Error::InvalidStatusCodeWithDetails { status, .. } if *status == StatusCode::TOO_MANY_REQUESTS
    ));
    let headers = error
        .non_success_headers()
        .expect("headers captured at error construction");
    assert_eq!(
        headers.get("retry-after").and_then(|v| v.to_str().ok()),
        Some("20")
    );
    assert_eq!(
        headers
            .get("x-ratelimit-remaining")
            .and_then(|v| v.to_str().ok()),
        Some("0")
    );
}
