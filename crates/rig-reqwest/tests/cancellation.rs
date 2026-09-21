//! Cancellation must release the HTTP operation, not merely stop its delivery.
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use bytes::Bytes;
use futures::FutureExt;
use rig_core::http_client::{HttpClientExt, NoBody, Request};
use rig_reqwest::ReqwestClient;
use std::{
    io::{Read, Write},
    net::TcpListener,
    sync::mpsc,
    time::Duration,
};

fn block_on<F: std::future::Future>(future: F) -> F::Output {
    futures::executor::block_on(rig_core::wasm_compat::timeout(
        Duration::from_secs(10),
        future,
    ))
    .expect("operation reached its deadline")
}

/// Hold the response open until the client releases it. The server deadline
/// bounds failures without making a successful test depend on a sleep.
fn held_response(
    headers: bool,
) -> (
    String,
    futures::channel::oneshot::Receiver<()>,
    mpsc::Receiver<bool>,
) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let url = format!("http://{}/", listener.local_addr().expect("address"));
    let (accepted, received) = futures::channel::oneshot::channel();
    let (closed, disconnected) = mpsc::channel();
    listener.set_nonblocking(true).expect("nonblocking");
    std::thread::spawn(move || {
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        let mut socket = loop {
            match listener.accept() {
                Ok((socket, _)) => break socket,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    if std::time::Instant::now() >= deadline {
                        return;
                    }
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(error) => panic!("accept: {error}"),
            }
        };
        socket.set_nonblocking(false).expect("blocking socket");
        socket
            .set_write_timeout(Some(Duration::from_secs(3)))
            .expect("write deadline");
        socket
            .set_read_timeout(Some(Duration::from_secs(3)))
            .expect("deadline");
        let mut request = Vec::new();
        let mut byte = [0];
        while !request.ends_with(b"\r\n\r\n") {
            assert!(std::time::Instant::now() < deadline, "request deadline");
            socket.read_exact(&mut byte).expect("request");
            request.push(byte[0]);
        }
        if headers {
            socket
                .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 1000\r\nconnection: close\r\n\r\n")
                .expect("headers");
            socket.flush().expect("flush");
        }
        accepted.send(()).expect("signal");
        let released = match socket.read(&mut byte) {
            Ok(0) => true,
            Err(error) => matches!(
                error.kind(),
                std::io::ErrorKind::ConnectionReset | std::io::ErrorKind::ConnectionAborted
            ),
            _ => false,
        };
        let _ = closed.send(released);
    });
    (url, received, disconnected)
}

#[test]
fn dropping_a_pending_request_releases_the_connection() {
    let (url, received, disconnected) = held_response(false);
    let client = ReqwestClient::new(
        reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("client"),
    );
    let request = Request::builder().uri(url).body(NoBody).expect("request");
    let mut send = Box::pin(client.send::<_, Bytes>(request));
    block_on(async {
        match futures::future::select(send.as_mut(), received).await {
            futures::future::Either::Right((received, _)) => {
                received.expect("request reached server")
            }
            futures::future::Either::Left(_) => {
                panic!("held response completed before cancellation")
            }
        }
    });
    drop(send);
    assert!(
        disconnected
            .recv_timeout(Duration::from_secs(4))
            .expect("server finished")
    );
}

#[test]
fn the_supplied_timeout_releases_a_pending_request() {
    let (url, received, disconnected) = held_response(false);
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(1))
        .build()
        .expect("client");
    let client = ReqwestClient::new(client);
    let started = std::time::Instant::now();
    let request = Request::builder().uri(url).body(NoBody).expect("request");
    let error = block_on(client.send::<_, Bytes>(request))
        .err()
        .expect("request times out");
    match error {
        rig_core::http_client::Error::Instance(error) => assert!(
            error
                .downcast_ref::<reqwest::Error>()
                .is_some_and(reqwest::Error::is_timeout),
            "the supplied client deadline, not an unrelated transport failure"
        ),
        other => panic!("expected request timeout, got {other}"),
    }
    assert!(started.elapsed() < Duration::from_secs(2));
    block_on(received).expect("request reached server");
    assert!(
        disconnected
            .recv_timeout(Duration::from_secs(4))
            .expect("server finished")
    );
}

#[test]
fn dropping_before_first_poll_never_connects() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    listener.set_nonblocking(true).expect("nonblocking");
    let client = ReqwestClient::new(
        reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("client"),
    );
    let request = Request::builder()
        .uri(format!(
            "http://{}/",
            listener.local_addr().expect("address")
        ))
        .body(NoBody)
        .expect("request");
    drop(client.send::<_, Bytes>(request));
    assert_eq!(
        listener.accept().expect_err("no connection").kind(),
        std::io::ErrorKind::WouldBlock
    );
}

#[test]
fn dropping_a_lazy_unary_body_releases_the_connection() {
    let (url, received, disconnected) = held_response(true);
    let client = ReqwestClient::new(
        reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("client"),
    );
    let request = Request::builder().uri(url).body(NoBody).expect("request");
    let response =
        block_on(client.send::<_, Bytes>(request)).expect("headers without waiting for body");
    block_on(received).expect("request reached server");
    drop(response.into_body());
    assert!(
        disconnected
            .recv_timeout(Duration::from_secs(4))
            .expect("server finished")
    );
}

#[test]
fn dropping_a_polled_unary_body_releases_the_connection() {
    let (url, received, disconnected) = held_response(true);
    let client = ReqwestClient::new(
        reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("client"),
    );
    let request = Request::builder().uri(url).body(NoBody).expect("request");
    let response = block_on(client.send::<_, Bytes>(request)).expect("response");
    block_on(received).expect("request reached server");
    assert!(
        response.into_body().now_or_never().is_none(),
        "body is pending and dropped"
    );
    assert!(
        disconnected
            .recv_timeout(Duration::from_secs(4))
            .expect("server finished")
    );
}

#[test]
fn a_host_runtime_allows_body_drop_from_a_foreign_executor() {
    let runtime = tokio::runtime::Runtime::new().expect("runtime");
    let (url, received, disconnected) = held_response(true);
    let client = ReqwestClient::new(
        reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("client"),
    );
    let request = Request::builder().uri(url).body(NoBody).expect("request");
    let response = runtime
        .block_on(rig_core::wasm_compat::timeout(
            Duration::from_secs(10),
            client.send::<_, Bytes>(request),
        ))
        .expect("request deadline")
        .expect("response");
    block_on(received).expect("request reached server");
    drop(response); // No entered Tokio context: drop must still release I/O.
    assert!(
        disconnected
            .recv_timeout(Duration::from_secs(4))
            .expect("server finished")
    );
    drop(client);
    drop(runtime);
}

#[test]
fn host_runtime_shutdown_releases_io_and_late_body_polling_returns_an_error() {
    let runtime = tokio::runtime::Runtime::new().expect("runtime");
    let (url, received, disconnected) = held_response(true);
    let client = ReqwestClient::new(
        reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("client"),
    );
    let request = Request::builder().uri(url).body(NoBody).expect("request");
    let response = runtime
        .block_on(rig_core::wasm_compat::timeout(
            Duration::from_secs(10),
            client.send::<_, Bytes>(request),
        ))
        .expect("request deadline")
        .expect("response");
    block_on(received).expect("request reached server");
    runtime.shutdown_background();
    assert!(
        disconnected
            .recv_timeout(Duration::from_secs(4))
            .expect("server finished")
    );
    assert!(block_on(response.into_body()).is_err());
}

#[test]
fn dropping_an_idle_stream_releases_the_connection_without_another_chunk() {
    let (url, received, disconnected) = held_response(true);
    let client = ReqwestClient::new(
        reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("client"),
    );
    let request = Request::builder().uri(url).body(NoBody).expect("request");
    let response = block_on(client.send_streaming(request)).expect("response");
    block_on(received).expect("request reached server");
    drop(response);
    assert!(
        disconnected
            .recv_timeout(Duration::from_secs(4))
            .expect("server finished")
    );
}
