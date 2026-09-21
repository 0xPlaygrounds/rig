//! Supplied middleware runs lazily on foreign executors without losing wire metadata.
#![cfg(all(
    not(target_family = "wasm"),
    any(
        feature = "reqwest-middleware-rustls",
        feature = "reqwest-middleware-native-tls"
    )
))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use bytes::Bytes;
use rig_core::{
    http_client::{HttpClientExt, NoBody, Request},
    wasm_compat::WasmBoxedFuture,
};
use rig_reqwest::ReqwestMiddlewareClient;
use std::{
    io::{Read, Write},
    net::TcpListener,
    sync::atomic::{AtomicUsize, Ordering},
};

static CALLS: AtomicUsize = AtomicUsize::new(0);

fn stamp<'a>(
    mut request: reqwest::Request,
    extensions: &'a mut http::Extensions,
    next: reqwest_middleware::Next<'a>,
) -> WasmBoxedFuture<'a, reqwest_middleware::Result<reqwest::Response>> {
    Box::pin(async move {
        CALLS.fetch_add(1, Ordering::SeqCst);
        request
            .headers_mut()
            .insert("x-host-policy", "enforced".parse().unwrap());
        next.run(request, extensions).await
    })
}

#[test]
fn supplied_middleware_is_lazy_and_preserves_status_headers_and_body() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let uri = format!("http://{}/", listener.local_addr().unwrap());
    listener.set_nonblocking(true).unwrap();
    let server = std::thread::spawn(move || {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let mut socket = loop {
            match listener.accept() {
                Ok((socket, _)) => break socket,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    assert!(std::time::Instant::now() < deadline, "accept deadline");
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                Err(error) => return Err(error),
            }
        };
        socket.set_nonblocking(false).unwrap();
        socket
            .set_write_timeout(Some(std::time::Duration::from_secs(3)))
            .unwrap();
        socket
            .set_read_timeout(Some(std::time::Duration::from_secs(3)))
            .unwrap();
        let mut request = Vec::new();
        while !request.ends_with(b"\r\n\r\n") {
            assert!(std::time::Instant::now() < deadline, "request deadline");
            let mut byte = [0];
            socket.read_exact(&mut byte).unwrap();
            request.push(byte[0]);
        }
        assert!(
            String::from_utf8(request)
                .unwrap()
                .contains("x-host-policy: enforced")
        );
        socket.write_all(b"HTTP/1.1 429 Too Many Requests\r\nretry-after: 17\r\ncontent-length: 6\r\nconnection: close\r\n\r\nquota!").unwrap();
        Ok(())
    });
    let client = reqwest_middleware::ClientBuilder::new(
        reqwest::Client::builder().no_proxy().build().unwrap(),
    )
    .with(stamp)
    .build();
    let client = ReqwestMiddlewareClient::new(client);
    let operation = client.send::<_, Bytes>(Request::builder().uri(uri).body(NoBody).unwrap());
    assert_eq!(CALLS.load(Ordering::SeqCst), 0);
    let error = futures::executor::block_on(rig_core::wasm_compat::timeout(
        std::time::Duration::from_secs(10),
        operation,
    ))
    .expect("middleware operation deadline")
    .err()
    .expect("non-success response");
    assert_eq!(CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(error.non_success_status().unwrap().as_u16(), 429);
    assert_eq!(
        error
            .non_success_headers()
            .unwrap()
            .get("retry-after")
            .unwrap(),
        "17"
    );
    assert_eq!(error.non_success_body(), Some("quota!"));
    server.join().unwrap().unwrap();
}
