//! The bundled transport must work when the caller has no tokio runtime —
//! Bevy task pools, smol, `futures::executor` — for both a unary request and a
//! streamed body. A blocking local server runs on its own std thread while
//! the client is driven by `futures::executor::block_on`.

#![allow(clippy::expect_used)]

use bytes::Bytes;
use futures::StreamExt;
use rig_core::http_client::{HttpClientExt, NoBody, Request};
use rig_reqwest::ReqwestClient;
use std::io::{Read, Write};
use std::net::TcpListener;

fn block_on<F: std::future::Future>(future: F) -> F::Output {
    futures::executor::block_on(rig_core::wasm_compat::timeout(
        std::time::Duration::from_secs(10),
        future,
    ))
    .expect("client operation deadline")
}

/// Minimal blocking HTTP/1.1 server on a std thread: answers every request
/// with a fixed body, chunked when `chunks > 1` so the client sees a real
/// multi-frame stream.
fn serve_once(body: &'static str, chunks: usize) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("addr");
    listener.set_nonblocking(true).expect("nonblocking");
    std::thread::spawn(move || {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let mut stream = loop {
            match listener.accept() {
                Ok((stream, _)) => break stream,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    if std::time::Instant::now() >= deadline {
                        return;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                Err(_) => return,
            }
        };
        stream.set_nonblocking(false).expect("blocking");
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(3)))
            .expect("read deadline");
        stream
            .set_write_timeout(Some(std::time::Duration::from_secs(3)))
            .expect("write deadline");
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
    let client = ReqwestClient::new(
        reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("client"),
    );
    let request = Request::builder()
        .method(http::Method::GET)
        .uri(url)
        .body(NoBody)
        .expect("request");

    // No tokio runtime anywhere on this thread: `futures::executor` only.
    let body: Bytes = block_on(async {
        let response = client.send::<_, Bytes>(request).await.expect("send");
        assert_eq!(response.status(), http::StatusCode::OK);
        response.into_body().await.expect("body")
    });
    assert_eq!(&body[..], b"hello from std");
}

#[test]
fn streamed_body_without_a_tokio_runtime() {
    let url = serve_once("one two three four five six seven eight", 4);
    let client = ReqwestClient::new(
        reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("client"),
    );
    let request = Request::builder()
        .method(http::Method::GET)
        .uri(url)
        .body(NoBody)
        .expect("request");

    let collected: Vec<u8> = block_on(async {
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
async fn unary_request_inside_a_tokio_runtime() {
    let url = serve_once("hello from tokio", 1);
    let client = ReqwestClient::new(
        reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("client"),
    );
    let request = Request::builder()
        .method(http::Method::GET)
        .uri(url)
        .body(NoBody)
        .expect("request");
    let body = tokio::time::timeout(std::time::Duration::from_secs(10), async {
        let response = client.send::<_, Bytes>(request).await.expect("send");
        response.into_body().await.expect("body")
    })
    .await
    .expect("request/body deadline");
    assert_eq!(&body[..], b"hello from tokio");
}

#[test]
fn bodies_can_move_off_the_request_executor() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .worker_threads(1)
        .build()
        .expect("runtime");
    let client = ReqwestClient::new(
        reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("client"),
    );
    let request = Request::builder()
        .uri(serve_once("moved unary", 1))
        .body(NoBody)
        .expect("request");
    let response = runtime
        .block_on(rig_core::wasm_compat::timeout(
            std::time::Duration::from_secs(10),
            client.send::<_, Bytes>(request),
        ))
        .expect("request deadline")
        .expect("response");
    let body = block_on(response.into_body()).expect("body");
    assert_eq!(&body[..], b"moved unary");

    let request = Request::builder()
        .uri(serve_once("moved stream", 4))
        .body(NoBody)
        .expect("request");
    let response = runtime
        .block_on(rig_core::wasm_compat::timeout(
            std::time::Duration::from_secs(10),
            client.send_streaming(request),
        ))
        .expect("request deadline")
        .expect("response");
    let body = block_on(async {
        let mut stream = response.into_body();
        let mut bytes = Vec::new();
        while let Some(chunk) = stream.next().await {
            bytes.extend_from_slice(&chunk.expect("chunk"));
        }
        bytes
    });
    assert_eq!(body, b"moved stream");
}
