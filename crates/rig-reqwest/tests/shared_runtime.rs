//! A process-wide pool must not borrow a short-lived caller's reactor.

#![cfg(not(target_family = "wasm"))]
#![allow(clippy::expect_used)]

use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    sync::mpsc,
    thread,
    time::Duration,
};

use bytes::Bytes;
use rig_http::http_client::{HttpClientExt, NoBody, Request};
use rig_reqwest::ReqwestClient;

const DEADLINE: Duration = Duration::from_secs(20);

fn read_head(socket: &mut TcpStream) -> std::io::Result<()> {
    let mut head = Vec::new();
    while !head.ends_with(b"\r\n\r\n") {
        let mut byte = [0];
        socket.read_exact(&mut byte)?;
        head.push(byte[0]);
        if head.len() >= 16_384 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "HTTP request head exceeded test server limit",
            ));
        }
    }
    Ok(())
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .expect("runtime")
}

async fn request(client: &ReqwestClient, url: &str) -> rig_http::http_client::Result<Bytes> {
    let request = Request::builder().uri(url).body(NoBody).expect("request");
    let response = client.send::<_, Bytes>(request).await?;
    response.into_body().await
}

#[test]
fn shared_pool_survives_the_first_callers_runtime_during_a_later_call() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("listen");
    let url = format!("http://{}/shared", listener.local_addr().expect("address"));
    let (second_seen, waiting) = mpsc::sync_channel(1);
    let (release, released) = mpsc::sync_channel(1);
    let server = thread::spawn(move || -> std::io::Result<()> {
        // One connection for both calls proves the shared pool was reused,
        // rather than merely proving a fresh connection can use a new runtime.
        let (mut socket, _) = listener.accept()?;
        socket.set_read_timeout(Some(DEADLINE))?;
        socket.set_write_timeout(Some(DEADLINE))?;
        read_head(&mut socket)?;
        socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")?;
        read_head(&mut socket)?;
        second_seen.send(()).expect("second request announced");
        released.recv_timeout(DEADLINE).expect("release response");
        socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
    });

    let first = runtime();
    let client = ReqwestClient::default();
    let body = first
        .block_on(async { tokio::time::timeout(DEADLINE, request(&client, &url)).await })
        .expect("first request deadline")
        .expect("first request");
    assert_eq!(body, Bytes::from_static(b"ok"));

    let second = thread::spawn(move || {
        let second = runtime();
        second.block_on(async { tokio::time::timeout(DEADLINE, request(&client, &url)).await })
    });
    waiting
        .recv_timeout(DEADLINE)
        .expect("pooled second request");
    drop(first);
    release.send(()).expect("release server");
    let result = second.join().expect("second runtime thread");
    let server_result = server.join().expect("server thread");
    let body = result
        .expect("second request deadline")
        .expect("a healthy second caller must not lose the process-wide pool's reactor");
    assert_eq!(body, Bytes::from_static(b"ok"));
    server_result.expect("server delivered the second response");
}
