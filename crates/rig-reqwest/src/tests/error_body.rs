//! Synthetic bodies exercise read boundaries and failures absent from cassettes.

use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

fn response(body: reqwest::Body) -> reqwest::Response {
    let mut response = http::Response::new(body);
    *response.status_mut() = StatusCode::BAD_GATEWAY;
    response
        .headers_mut()
        .insert("request-id", http::HeaderValue::from_static("bounded"));
    reqwest::Response::from(response)
}

#[tokio::test]
async fn exact_limit_is_not_truncated_but_one_more_byte_is() {
    for (length, truncated) in [(64 * 1024, false), (64 * 1024 + 1, true)] {
        let error = non_success_status_error(response(vec![b'x'; length].into())).await;
        let mut expected = "x".repeat(64 * 1024);
        if truncated {
            expected.push_str("\n[Response body truncated after 64 KiB]");
        }
        assert_eq!(error.non_success_body(), Some(expected.as_str()));
        assert_eq!(error.non_success_status(), Some(StatusCode::BAD_GATEWAY));
        assert_eq!(
            error
                .non_success_headers()
                .and_then(|h| h.get("request-id"))
                .and_then(|v| v.to_str().ok()),
            Some("bounded")
        );
    }
}

#[tokio::test]
async fn reads_past_exact_limit_only_until_excess_is_observed() {
    let reads = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&reads);
    let stream = futures::stream::repeat_with(move || {
        observed.fetch_add(1, Ordering::SeqCst);
        Ok::<_, std::io::Error>(Bytes::from(vec![b'x'; 64 * 1024]))
    });
    let error = non_success_status_error(response(reqwest::Body::wrap_stream(stream))).await;
    assert_eq!(reads.load(Ordering::SeqCst), 2);
    assert!(
        error
            .non_success_body()
            .is_some_and(|body| body.ends_with("[Response body truncated after 64 KiB]"))
    );
}

#[tokio::test]
async fn partial_body_survives_transport_failure_and_empty_chunks() {
    let stream = futures::stream::iter([
        Ok(Bytes::new()),
        Ok(Bytes::from_static(b"partial detail")),
        Ok(Bytes::new()),
        Err(std::io::Error::other("socket closed")),
    ]);
    let error = non_success_status_error(response(reqwest::Body::wrap_stream(stream))).await;
    assert_eq!(error.non_success_body(), Some("partial detail"));
}

#[tokio::test]
async fn empty_failed_body_retains_read_failure_diagnostic() {
    let stream = futures::stream::iter([Err::<Bytes, _>(std::io::Error::other("socket closed"))]);
    let error = non_success_status_error(response(reqwest::Body::wrap_stream(stream))).await;
    assert!(
        error
            .non_success_body()
            .is_some_and(|body| body.starts_with("failed to read error response body:"))
    );
}

#[tokio::test]
async fn utf8_split_across_chunks_is_decoded_together() {
    let stream = futures::stream::iter([
        Ok::<_, std::io::Error>(Bytes::from_static(b"\xe2")),
        Ok(Bytes::new()),
        Ok(Bytes::from_static(b"\x82\xac")),
    ]);
    let error = non_success_status_error(response(reqwest::Body::wrap_stream(stream))).await;
    assert_eq!(error.non_success_body(), Some("€"));
}

#[tokio::test]
async fn incomplete_utf8_keeps_reqwest_replacement_semantics() {
    for failed in [false, true] {
        let mut chunks = vec![Ok(Bytes::from_static(b"detail \xe2\x82"))];
        if failed {
            chunks.push(Err(std::io::Error::other("socket closed")));
        }
        let error = non_success_status_error(response(reqwest::Body::wrap_stream(
            futures::stream::iter(chunks),
        )))
        .await;
        assert_eq!(error.non_success_body(), Some("detail �"));
    }
    let mut bytes = vec![b'x'; 64 * 1024 - 1];
    bytes.extend_from_slice("€".as_bytes());
    let error = non_success_status_error(response(bytes.into())).await;
    let expected = format!(
        "{}�\n[Response body truncated after 64 KiB]",
        "x".repeat(64 * 1024 - 1)
    );
    assert_eq!(error.non_success_body(), Some(expected.as_str()));
}

#[tokio::test]
async fn configured_timeout_preserves_partial_stalled_response() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let outcome = async {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await?;
            let mut request = Vec::new();
            while !request.ends_with(b"\r\n\r\n") {
                if request.len() == 4096 {
                    return Err(std::io::Error::other("request headers exceed test limit"));
                }
                let mut byte = [0];
                if socket.read(&mut byte).await? == 0 {
                    return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof));
                }
                request.extend_from_slice(&byte);
            }
            socket
                .write_all(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 100\r\n\r\npartial detail")
                .await?;
            let mut buffer = [0; 1];
            let _ = socket.read(&mut buffer).await;
            Ok::<_, std::io::Error>(())
        });
        let client = reqwest::Client::builder()
            .no_proxy()
            .timeout(std::time::Duration::from_millis(200))
            .build()?;
        let response = client.get(format!("http://{address}")).send().await?;
        let error = non_success_status_error(response).await;
        server.abort();
        Ok::<_, Box<dyn std::error::Error>>(error)
    }
    .await;
    let Ok(error) = outcome else {
        panic!("local timeout scenario failed: {outcome:?}");
    };
    assert_eq!(error.non_success_body(), Some("partial detail"));
    assert_eq!(error.non_success_status(), Some(StatusCode::BAD_GATEWAY));
}

#[cfg(any(
    feature = "rustls",
    feature = "native-tls",
    feature = "reqwest-middleware"
))]
#[tokio::test]
async fn declared_charset_is_preserved() {
    let mut response = response(vec![0xe9].into());
    response.headers_mut().insert(
        http::header::CONTENT_TYPE,
        http::HeaderValue::from_static("text/plain; charset=windows-1252"),
    );
    let error = non_success_status_error(response).await;
    assert_eq!(error.non_success_body(), Some("é"));
}
