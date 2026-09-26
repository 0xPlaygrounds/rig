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

/// Every `default()` is a clone of the one process-wide client.
#[test]
fn default_hands_out_clones_of_one_client() {
    let first = ReqwestClient::default();
    let second = ReqwestClient::default();
    assert!(first.same(&second));
    assert!(first.inner().is_some());
    let custom = ReqwestClient::from(reqwest::Client::new());
    assert!(!first.same(&custom));
}

/// A client reqwest could not build reports the build error on every send,
/// in-band, rather than panicking where it was named.
#[tokio::test]
async fn an_unbuilt_client_reports_its_build_error_on_every_send() {
    let refused = reqwest::Client::new()
        .get("not a url")
        .build()
        .expect_err("an unparseable URL");
    let client = ReqwestClient(Built::Failed(Arc::new(refused)));
    assert!(client.inner().is_none());
    let request = || {
        Request::builder()
            .uri("https://example.test/")
            .body(Bytes::new())
            .expect("a request")
    };
    for _ in 0..2 {
        let Err(error) = client.send::<_, Bytes>(request()).await else {
            panic!("an unbuilt client sent a request");
        };
        assert!(
            error
                .to_string()
                .contains("could not build the bundled reqwest transport"),
            "{error}"
        );
        assert!(client.send_streaming(request()).await.is_err());
    }
}
