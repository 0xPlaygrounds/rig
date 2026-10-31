//! Each reqwest failure says whether sending the same request again may
//! work: a connection that drops may, a request reqwest cannot build never
//! does. A timeout is checked in `cancellation.rs`.
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use bytes::Bytes;
use futures::StreamExt;
use rig_http::http_client::{Error, HttpClientExt, NoBody, Request};
use rig_reqwest::ReqwestClient;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::time::Duration;

fn client() -> ReqwestClient {
    ReqwestClient::from(
        reqwest::Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(10))
            .build()
            .expect("client"),
    )
}

/// A server that answers one request with a 200 promising 1000 bytes, sends
/// five, and closes the connection.
fn cut_short() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let url = format!("http://{}/", listener.local_addr().expect("address"));
    std::thread::spawn(move || {
        let (mut socket, _) = listener.accept().expect("accept");
        socket
            .set_read_timeout(Some(Duration::from_secs(10)))
            .expect("read deadline");
        let mut request = Vec::new();
        let mut byte = [0];
        while !request.ends_with(b"\r\n\r\n") {
            socket.read_exact(&mut byte).expect("request");
            request.push(byte[0]);
        }
        socket
            .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 1000\r\n\r\nhello")
            .expect("reply");
    });
    url
}

#[tokio::test]
async fn a_body_cut_short_is_transient() {
    let request = Request::builder()
        .uri(cut_short())
        .body(NoBody)
        .expect("request");
    let response = client().send::<_, Bytes>(request).await.expect("headers");
    let error = response
        .into_body()
        .await
        .expect_err("the body is cut short");
    assert!(matches!(error, Error::Transient(_)), "{error:?}");

    let request = Request::builder()
        .uri(cut_short())
        .body(NoBody)
        .expect("request");
    let response = client().send_streaming(request).await.expect("headers");
    let mut body = response.into_body();
    let mut failure = None;
    while let Some(chunk) = body.next().await {
        if let Err(error) = chunk {
            failure = Some(error);
            break;
        }
    }
    let error = failure.expect("the stream is cut short");
    assert!(matches!(error, Error::Transient(_)), "{error:?}");
}

#[tokio::test]
async fn a_request_reqwest_cannot_build_is_permanent() {
    let request = Request::builder()
        .uri("/relative")
        .body(NoBody)
        .expect("request");
    let Err(error) = client().send::<_, Bytes>(request).await else {
        panic!("a relative URL has no host");
    };
    assert!(matches!(error, Error::Permanent(_)), "{error:?}");
}
