//! What a multipart request actually puts on the wire.
//!
//! A part's content type is the piece most easily lost in translation: rig
//! carries it as a typed `mime::Mime`, reqwest wants a string, and until this
//! was fixed a content type reqwest would not take was dropped and the request
//! was sent anyway — so a provider saw a part with *no* content type and
//! answered with something unrelated to the real mistake.
//!
//! The provider-facing property is simply that the content type survives, so
//! that is what this asserts, against a socket rather than through reqwest's
//! `Form` (which exposes no way to read a part back).

#![cfg(not(target_family = "wasm"))]
#![allow(clippy::expect_used, clippy::indexing_slicing)]

use rig_core::http_client::multipart::{MultipartForm, Part};
use rig_core::http_client::{HttpClientExt, Request};
use rig_reqwest::ReqwestClient;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::mpsc;

/// Accept one request, hand its full text back to the test, and answer 200.
fn capture_one_request() -> (String, mpsc::Receiver<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let address = listener.local_addr().expect("addr");
    let (tx, rx) = mpsc::channel();

    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept");
        // One read is enough: these bodies are far below the socket buffer, and
        // the assertions only need the headers and part preamble.
        let mut buffer = vec![0u8; 8192];
        let read = stream.read(&mut buffer).unwrap_or(0);
        let _ = tx.send(String::from_utf8_lossy(&buffer[..read]).into_owned());
        let _ = stream
            .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\nconnection: close\r\n\r\n{}");
        let _ = stream.flush();
    });

    (format!("http://{address}/"), rx)
}

#[tokio::test]
async fn a_parts_content_type_reaches_the_wire() {
    let (url, requests) = capture_one_request();

    let form = MultipartForm::new().text("model", "whisper-1").part(
        Part::bytes("file", vec![0x89, 0x50, 0x4e, 0x47])
            .filename("audio.png")
            .content_type("image/png".parse().expect("a valid mime")),
    );

    let request = Request::builder()
        .method(http::Method::POST)
        .uri(&url)
        .body(form)
        .expect("request should build");

    let response = ReqwestClient::default()
        .send_multipart::<Vec<u8>>(request)
        .await
        .expect("the request should be sent");
    assert!(response.status().is_success());

    let sent = requests.recv().expect("the server should see the request");
    assert!(
        sent.contains("content-type: image/png") || sent.contains("Content-Type: image/png"),
        "the part's content type must reach the wire, got:\n{sent}"
    );
    assert!(
        sent.contains("audio.png"),
        "the part's filename must reach the wire, got:\n{sent}"
    );
    assert!(
        sent.contains("whisper-1"),
        "the text part must reach the wire, got:\n{sent}"
    );
}

/// A form with no content type on its binary part is still valid — the field
/// is optional, and rendering must not invent one.
#[tokio::test]
async fn a_part_without_a_content_type_is_still_rendered() {
    let (url, requests) = capture_one_request();

    let form = MultipartForm::new().part(Part::bytes("file", vec![1, 2, 3]));
    let request = Request::builder()
        .method(http::Method::POST)
        .uri(&url)
        .body(form)
        .expect("request should build");

    ReqwestClient::default()
        .send_multipart::<Vec<u8>>(request)
        .await
        .expect("the request should be sent");

    let sent = requests.recv().expect("the server should see the request");
    assert!(
        sent.contains("name=\"file\""),
        "the part must reach the wire, got:\n{sent}"
    );
}
