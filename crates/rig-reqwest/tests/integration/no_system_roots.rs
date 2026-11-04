//! A host with no CA certificates (a bare container image) still gets a
//! working shared client: plain http is sent, and https trusts the bundled
//! roots. Its own binary, because the root store is read from the
//! environment when the process-wide client is built.

#![cfg(feature = "rustls")]
#![allow(clippy::expect_used)]

use bytes::Bytes;
use rig_http::http_client::{HttpClientExt, NoBody, Request};
use rig_reqwest::ReqwestClient;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::time::Duration;

/// Answers one request on a std thread with `200 OK` and `body`.
fn serve_once(body: &'static str) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("addr");
    std::thread::spawn(move || {
        let Ok((mut stream, _)) = listener.accept() else {
            return;
        };
        stream
            .set_read_timeout(Some(Duration::from_secs(3)))
            .expect("read deadline");
        let mut buf = [0u8; 4096];
        let _ = stream.read(&mut buf);
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        );
        let _ = stream.write_all(response.as_bytes());
    });
    format!("http://{addr}/")
}

#[tokio::test]
async fn plain_http_works_on_a_host_without_ca_certificates() {
    let empty = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("no-system-roots");
    std::fs::create_dir_all(&empty).expect("an empty certificate directory");
    // SAFETY: this test binary has one test, and it sets the environment
    // before anything else runs or builds the shared client.
    unsafe {
        std::env::set_var("SSL_CERT_FILE", empty.join("none.pem"));
        std::env::set_var("SSL_CERT_DIR", &empty);
    }

    let client = ReqwestClient::default();
    let request = Request::get(serve_once("hello"))
        .body(NoBody)
        .expect("a request");
    let response = tokio::time::timeout(Duration::from_secs(10), client.send::<_, Bytes>(request))
        .await
        .expect("a reply in time")
        .expect("a plain-http reply without system roots");
    let body = response.into_body().await.expect("the body");
    assert_eq!(body, Bytes::from_static(b"hello"));
}
