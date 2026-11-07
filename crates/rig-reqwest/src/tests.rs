use super::*;
use http::StatusCode;

#[cfg(not(target_family = "wasm"))]
mod error_body;

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
/// in-band, rather than panicking where it was named, and as `Permanent`,
/// which callers do not retry.
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
        assert!(matches!(error, Error::Permanent(_)), "{error:?}");
        assert!(client.send_streaming(request()).await.is_err());
    }
}

/// A redirect to another origin (here, another port) arrives without any key
/// header, whatever its name, and keeps the method, body and content type; a
/// redirect on the same origin arrives with the keys.
#[cfg(not(target_family = "wasm"))]
#[tokio::test]
async fn only_a_redirect_on_the_same_origin_keeps_the_keys() {
    use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
    use tokio::net::TcpListener;
    use tokio::sync::mpsc;

    /// Answers `/to/<port>` with a 307 to that port, and records any other
    /// request, in lower case, before answering it with a 200.
    async fn serve(listener: TcpListener, landed: mpsc::UnboundedSender<String>) {
        while let Ok((socket, _)) = listener.accept().await {
            let mut socket = BufReader::new(socket);
            let mut request = String::new();
            while !request.ends_with("\r\n\r\n") {
                socket.read_line(&mut request).await.expect("a request");
            }
            request.make_ascii_lowercase();
            let length = request
                .split_once("content-length: ")
                .and_then(|(_, rest)| rest.split_once("\r\n"))
                .map_or(0, |(length, _)| length.parse().expect("a length"));
            let mut body = vec![0; length];
            socket.read_exact(&mut body).await.expect("a body");
            let path = request.split_whitespace().nth(1).expect("a path");
            let head = match path.strip_prefix("/to/") {
                Some(port) => {
                    format!("307 Temporary Redirect\r\nlocation: http://127.0.0.1:{port}/end")
                }
                None => {
                    request.push_str(&String::from_utf8_lossy(&body));
                    landed.send(request).expect("the test is waiting");
                    "200 OK".to_string()
                }
            };
            let reply =
                format!("HTTP/1.1 {head}\r\ncontent-length: 0\r\nconnection: close\r\n\r\n");
            socket.write_all(reply.as_bytes()).await.expect("a reply");
        }
    }

    async fn server() -> (u16, mpsc::UnboundedReceiver<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let port = listener.local_addr().expect("an address").port();
        let (sender, receiver) = mpsc::unbounded_channel();
        tokio::spawn(serve(listener, sender));
        (port, receiver)
    }

    const KEY_HEADERS: [&str; 4] = ["x-api-key", "api-key", "x-goog-api-key", "x-unheard-of-key"];

    /// What landed after a POST from the first server to `/to/<port>`.
    async fn post(first: u16, port: u16, landed: &mut mpsc::UnboundedReceiver<String>) -> String {
        let mut request = Request::builder()
            .method("POST")
            .uri(format!("http://127.0.0.1:{first}/to/{port}"))
            .header("content-type", "application/json");
        for name in KEY_HEADERS {
            request = request.header(name, "secret-key");
        }
        let request = request.body(Bytes::from("the-body")).expect("a request");
        ReqwestClient::default()
            .send::<_, Bytes>(request)
            .await
            .expect("the redirect is followed");
        landed.recv().await.expect("a request landed")
    }

    let (first, mut landed_on_first) = server().await;
    let (second, mut landed_on_second) = server().await;

    let other_origin = post(first, second, &mut landed_on_second).await;
    assert!(
        !other_origin.contains("secret-key"),
        "the other origin got a key: {other_origin}"
    );
    assert!(other_origin.starts_with("post /end "), "{other_origin}");
    assert!(
        other_origin.contains("content-type: application/json"),
        "{other_origin}"
    );
    assert!(other_origin.ends_with("the-body"), "{other_origin}");

    let same_origin = post(first, first, &mut landed_on_first).await;
    for name in KEY_HEADERS {
        assert!(
            same_origin.contains(&format!("{name}: secret-key")),
            "{same_origin}"
        );
    }
    assert!(same_origin.ends_with("the-body"), "{same_origin}");
}

/// A status error, as a middleware's `error_for_status` makes one, keeps
/// reqwest's error and is retried only for a status the table retries.
#[test]
fn a_status_error_is_classified_by_its_status() {
    for (status, transient) in [
        (StatusCode::SERVICE_UNAVAILABLE, true),
        (StatusCode::BAD_REQUEST, false),
    ] {
        let error = from_reqwest(status_error(status));
        assert_eq!(matches!(error, Error::Transient(_)), transient, "{error:?}");
        assert!(
            matches!(&error, Error::Transient(inner) | Error::Permanent(inner)
                if inner.downcast_ref::<reqwest::Error>().is_some()),
            "{error:?}"
        );
    }
}

/// The error `error_for_status` makes of a `status` reply.
fn status_error(status: StatusCode) -> reqwest::Error {
    let response = http::Response::builder()
        .status(status)
        .body("")
        .expect("a response");
    reqwest::Response::from(response)
        .error_for_status()
        .expect_err("a non-success status")
}

/// A middleware's error keeps the class of the reqwest or rig error it
/// passed on, also one that came back from the middleware chain, with the
/// context it added in the message; a middleware says a failure is final
/// with a permanent rig error, and an unclassified one may hide a dropped
/// connection, so it is retried.
#[cfg(any(
    feature = "reqwest-middleware-rustls",
    feature = "reqwest-middleware-native-tls"
))]
#[test]
fn a_middleware_error_keeps_the_class_of_what_it_passed_on() {
    /// A wrapper that hides what it wraps from the source chain, as
    /// `reqwest-retry`'s `RetryError::Error` does.
    #[derive(Debug, thiserror::Error)]
    #[error(transparent)]
    struct GaveUp(reqwest_middleware::Error);

    let sent = |status| reqwest_middleware::Error::Reqwest(status_error(status));
    let mut passed_on = vec![
        (
            anyhow::Error::new(status_error(StatusCode::SERVICE_UNAVAILABLE)),
            true,
        ),
        (
            anyhow::Error::new(sent(StatusCode::SERVICE_UNAVAILABLE)),
            true,
        ),
        (anyhow::Error::new(sent(StatusCode::BAD_REQUEST)), false),
        (
            anyhow::Error::new(GaveUp(sent(StatusCode::BAD_REQUEST))),
            true,
        ),
        (anyhow::Error::new(std::io::Error::other("no key")), true),
    ];
    #[cfg(not(target_family = "wasm"))]
    passed_on.extend([
        (anyhow::Error::new(Error::transient("reset")), true),
        (anyhow::Error::new(Error::permanent("no key")), false),
    ]);
    for (error, transient) in passed_on {
        let error = reqwest_middleware::Error::Middleware(error.context("refreshing the key"));
        let error = middleware::from_middleware(error);
        assert_eq!(error.is_transient(), transient, "{error:?}");
        assert!(error.to_string().contains("refreshing the key"), "{error}");
    }
}
