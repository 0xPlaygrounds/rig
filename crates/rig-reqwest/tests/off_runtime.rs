//! The bundled transport must work when the caller has no tokio runtime —
//! Bevy task pools, smol, `futures::executor` — for both a unary request and a
//! streamed body. A local hyper-less server is not available without tokio,
//! so the server side runs on its own tokio runtime thread while the client
//! side is driven entirely by `futures::executor::block_on`.

#![allow(clippy::expect_used, clippy::indexing_slicing)]

use bytes::Bytes;
use futures::StreamExt;
use rig_core::http_client::{HttpClientExt, NoBody, Request};
use rig_reqwest::ReqwestClient;
use std::io::{Read, Write};
use std::net::TcpListener;

/// Minimal blocking HTTP/1.1 server on a std thread: answers every request
/// with a fixed body, chunked when `chunks > 1` so the client sees a real
/// multi-frame stream.
fn serve_once(body: &'static str, chunks: usize) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("addr");
    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept");
        let mut buf = [0u8; 4096];
        let _ = stream.read(&mut buf);
        if chunks <= 1 {
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-length: {}\r\ncontent-type: text/plain\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).expect("write");
        } else {
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\ntransfer-encoding: chunked\r\ncontent-type: text/plain\r\nconnection: close\r\n\r\n",
                )
                .expect("write head");
            let piece = body.len().div_ceil(chunks);
            for chunk in body.as_bytes().chunks(piece.max(1)) {
                let frame = format!("{:x}\r\n", chunk.len());
                stream.write_all(frame.as_bytes()).expect("write");
                stream.write_all(chunk).expect("write");
                stream.write_all(b"\r\n").expect("write");
                stream.flush().expect("flush");
            }
            stream.write_all(b"0\r\n\r\n").expect("write end");
        }
        stream.flush().expect("flush");
    });
    format!("http://{addr}/")
}

#[test]
fn unary_request_without_a_tokio_runtime() {
    let url = serve_once("hello from std", 1);
    let client = ReqwestClient::default();
    let request = Request::builder()
        .method(http::Method::GET)
        .uri(url)
        .body(NoBody)
        .expect("request");

    // No tokio runtime anywhere on this thread: `futures::executor` only.
    let body: Bytes = futures::executor::block_on(async {
        let response = client.send::<_, Bytes>(request).await.expect("send");
        assert_eq!(response.status(), http::StatusCode::OK);
        response.into_body().await.expect("body")
    });
    assert_eq!(&body[..], b"hello from std");
}

#[test]
fn streamed_body_without_a_tokio_runtime() {
    let url = serve_once("one two three four five six seven eight", 4);
    let client = ReqwestClient::default();
    let request = Request::builder()
        .method(http::Method::GET)
        .uri(url)
        .body(NoBody)
        .expect("request");

    let collected: Vec<u8> = futures::executor::block_on(async {
        let response = client
            .send_streaming(request)
            .await
            .expect("send_streaming");
        assert_eq!(response.status(), http::StatusCode::OK);
        let mut body = response.into_body();
        let mut out = Vec::new();
        while let Some(chunk) = body.next().await {
            out.extend_from_slice(&chunk.expect("chunk"));
        }
        out
    });
    assert_eq!(collected, b"one two three four five six seven eight");
}

#[tokio::test]
async fn unary_request_inside_a_tokio_runtime_takes_the_direct_path() {
    let url = serve_once("hello from tokio", 1);
    let client = ReqwestClient::default();
    let request = Request::builder()
        .method(http::Method::GET)
        .uri(url)
        .body(NoBody)
        .expect("request");
    let response = client.send::<_, Bytes>(request).await.expect("send");
    let body = response.into_body().await.expect("body");
    assert_eq!(&body[..], b"hello from tokio");
}
