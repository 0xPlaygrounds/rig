use super::*;
use crate::http::ledger::{self, LedgerTarget};
use crate::http::{BodyEncoding, CassetteInteraction};

/// A recorder whose ledger lives in `dir`, and the interactions it keeps.
fn recorder(
    dir: &assert_fs::TempDir,
) -> (
    DirectRecorder,
    Arc<std::sync::Mutex<Vec<CassetteInteraction>>>,
) {
    let interactions = Arc::new(std::sync::Mutex::new(Vec::new()));
    let recorder = DirectRecorder {
        interactions: interactions.clone(),
        ledger: Arc::new(LedgerTarget {
            path: dir.path().join(ledger::LEDGER_FILE),
            provider: "openai".into(),
            scenario: "proxy/test".into(),
            origin: "https://api.openai.com".into(),
        }),
        refusals: Arc::default(),
        scrubber: Arc::default(),
        in_flight: Arc::default(),
    };
    (recorder, interactions)
}

fn client() -> rig_reqwest::reqwest::Client {
    rig_reqwest::reqwest::Client::builder()
        .no_proxy()
        .build()
        .expect("client")
}

#[tokio::test]
async fn the_proxy_forwards_a_reply_unchanged_and_writes_the_ledger_first() {
    let upstream = httpmock::MockServer::start_async().await;
    let mock = upstream
        .mock_async(|when, then| {
            when.method("POST")
                .path("/v1/responses")
                .header("accept", "text/plain")
                .body(r#"{"input":"hi"}"#);
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"id":"resp_proxy","object":"response"}"#);
        })
        .await;
    let dir = assert_fs::TempDir::new().expect("ledger directory");
    let (recorder, interactions) = recorder(&dir);
    let proxy = Proxy::start(upstream.base_url(), recorder)
        .await
        .expect("proxy starts");

    let body = client()
        .post(format!("{}/v1/responses", proxy.base_url))
        .header("accept", "text/plain")
        .body(r#"{"input":"hi"}"#)
        .send()
        .await
        .expect("forwarded")
        .text()
        .await
        .expect("body");
    assert_eq!(body, r#"{"id":"resp_proxy","object":"response"}"#);
    mock.assert_async().await;

    let outstanding = ledger::outstanding(&dir.path().join(ledger::LEDGER_FILE));
    assert_eq!(outstanding.len(), 1);
    assert_eq!(outstanding[0].id, "resp_proxy");
    assert_eq!(outstanding[0].scenario, "proxy/test");
    assert_eq!(
        outstanding[0].delete_url,
        "https://api.openai.com/v1/responses/resp_proxy"
    );
    let interactions = interactions.lock().expect("interactions");
    assert_eq!(interactions.len(), 1);
    assert_eq!(interactions[0].when.path, "/v1/responses");
    assert_eq!(
        interactions[0].then.body.as_deref(),
        Some(r#"{"id":"resp_proxy","object":"response"}"#)
    );
}

#[tokio::test]
async fn a_request_above_axums_default_body_limit_is_forwarded() {
    let upstream = httpmock::MockServer::start_async().await;
    let mock = upstream
        .mock_async(|when, then| {
            when.method("POST").path("/v1/files");
            then.status(200).body("{}");
        })
        .await;
    let dir = assert_fs::TempDir::new().expect("ledger directory");
    let (recorder, interactions) = recorder(&dir);
    let proxy = Proxy::start(upstream.base_url(), recorder)
        .await
        .expect("proxy starts");

    let status = client()
        .post(format!("{}/v1/files", proxy.base_url))
        .body(vec![b'a'; 3 * 1024 * 1024])
        .send()
        .await
        .expect("forwarded")
        .status();
    assert_eq!(status.as_u16(), 200);
    mock.assert_async().await;
    assert_eq!(interactions.lock().expect("interactions").len(), 1);
}

#[tokio::test]
async fn a_streamed_reply_and_an_error_reply_reach_the_client_as_sent() {
    let upstream = httpmock::MockServer::start_async().await;
    let events = "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_stream\"}}\n\ndata: [DONE]\n\n";
    upstream
        .mock_async(|when, then| {
            when.method("POST").path("/v1/responses");
            then.status(200)
                .header("content-type", "text/event-stream")
                .body(events);
        })
        .await;
    let refusal = r#"{"error":{"code":"invalid_api_key"}}"#;
    upstream
        .mock_async(|when, then| {
            when.method("GET").path("/v1/models");
            then.status(401)
                .header("content-type", "application/json")
                .header("x-request-id", "req_proxy")
                .body(refusal);
        })
        .await;
    let dir = assert_fs::TempDir::new().expect("ledger directory");
    let (recorder, interactions) = recorder(&dir);
    let proxy = Proxy::start(upstream.base_url(), recorder)
        .await
        .expect("proxy starts");

    let streamed = client()
        .post(format!("{}/v1/responses", proxy.base_url))
        .body(r#"{"stream":true}"#)
        .send()
        .await
        .expect("forwarded");
    assert_eq!(
        streamed
            .headers()
            .get("content-type")
            .map(|value| value.as_bytes()),
        Some(&b"text/event-stream"[..])
    );
    assert_eq!(streamed.text().await.expect("events"), events);
    let refused = client()
        .get(format!("{}/v1/models", proxy.base_url))
        .send()
        .await
        .expect("forwarded");
    assert_eq!(refused.status().as_u16(), 401);
    assert_eq!(
        refused
            .headers()
            .get("x-request-id")
            .map(|value| value.as_bytes()),
        Some(&b"req_proxy"[..])
    );
    assert_eq!(refused.text().await.expect("body"), refusal);

    let outstanding = ledger::outstanding(&dir.path().join(ledger::LEDGER_FILE));
    assert_eq!(outstanding.len(), 1);
    assert_eq!(outstanding[0].id, "resp_stream");
    let interactions = interactions.lock().expect("interactions");
    let statuses: Vec<_> = interactions
        .iter()
        .map(|interaction| interaction.then.status)
        .collect();
    assert_eq!(statuses, [200, 401]);
    // The cassette keeps each event's JSON canonically, as it always has.
    assert_eq!(
        interactions[0].then.body.as_deref(),
        Some(
            "data: {\"response\":{\"id\":\"resp_stream\"},\"type\":\"response.created\"}\n\ndata: [DONE]\n\n"
        )
    );
    assert_eq!(interactions[1].then.body.as_deref(), Some(refusal));
}

/// A websocket handshake the provider rejects is an ordinary reply: the
/// client gets the rejection and the recording keeps it, as the OpenAI
/// websocket error matrix records it.
#[tokio::test]
async fn a_rejected_websocket_handshake_is_recorded_like_any_reply() {
    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

    let upstream = httpmock::MockServer::start_async().await;
    let rejection = r#"{"error":{"code":"invalid_api_key"},"status":401}"#;
    upstream
        .mock_async(|when, then| {
            when.method("GET").path("/v1/responses");
            then.status(401)
                .header("content-type", "text/plain")
                .body(rejection);
        })
        .await;
    let dir = assert_fs::TempDir::new().expect("ledger directory");
    let (recorder, interactions) = recorder(&dir);
    let proxy = Proxy::start(upstream.base_url(), recorder)
        .await
        .expect("proxy starts");

    let address = proxy.base_url.trim_start_matches("http://").to_owned();
    let mut stream = tokio::net::TcpStream::connect(address)
        .await
        .expect("connect");
    stream
        .write_all(
            b"GET /v1/responses HTTP/1.1\r\nhost: proxy\r\nconnection: Upgrade\r\nupgrade: websocket\r\n\
              sec-websocket-version: 13\r\nsec-websocket-key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n",
        )
        .await
        .expect("handshake");
    let mut reply = Vec::new();
    let mut buffer = [0_u8; 4096];
    while !String::from_utf8_lossy(&reply).contains(rejection) {
        let read = stream.read(&mut buffer).await.expect("reply");
        assert!(read > 0, "{}", String::from_utf8_lossy(&reply));
        reply.extend_from_slice(&buffer[..read]);
    }
    assert!(
        reply.starts_with(b"HTTP/1.1 401"),
        "{}",
        String::from_utf8_lossy(&reply)
    );

    let interactions = interactions.lock().expect("interactions");
    assert_eq!(interactions.len(), 1);
    assert_eq!(interactions[0].when.method, "GET");
    assert_eq!(interactions[0].then.status, 401);
    assert_eq!(interactions[0].then.body.as_deref(), Some(rejection));
}

// The ledger is a FIFO whose reader opens late, so writing a line blocks the
// proxy until then: the client must not hold the whole reply before that
// moment. The upstream sends the reply as one chunk and ends the chunked body
// only later, so a proxy that passed each chunk on at once would hand the
// client the whole reply before the recorder saw it end. The client is a
// blocking socket on its own thread, so a proxy that blocks a runtime worker
// cannot delay the client's reads.
#[cfg(unix)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_client_cannot_read_the_reply_before_the_ledger_line_is_written() {
    use std::io::{Read as _, Write as _};
    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

    let reply = r#"{"id":"resp_fifo","object":"response"}"#;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("upstream listens");
    let origin = format!("http://{}", listener.local_addr().expect("address"));
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("connection");
        let mut request = [0_u8; 4096];
        let _ = socket.read(&mut request).await;
        let head = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ntransfer-encoding: chunked\r\n\r\n{:x}\r\n{reply}\r\n",
            reply.len()
        );
        socket.write_all(head.as_bytes()).await.expect("reply");
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        socket.write_all(b"0\r\n\r\n").await.expect("end of reply");
        let _ = socket.read(&mut request).await;
    });
    let dir = assert_fs::TempDir::new().expect("ledger directory");
    let ledger_path = dir.path().join(ledger::LEDGER_FILE);
    let made = std::process::Command::new("mkfifo")
        .arg(&ledger_path)
        .status()
        .expect("mkfifo runs");
    assert!(made.success());
    let (recorder, _) = recorder(&dir);
    let proxy = Proxy::start(origin, recorder).await.expect("proxy starts");

    let started = std::time::Instant::now();
    let (sender, received) = std::sync::mpsc::channel();
    let fifo = ledger_path.clone();
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(500));
        let opened = started.elapsed();
        let line = std::fs::read_to_string(&fifo).unwrap_or_default();
        let _ = sender.send((opened, line));
    });
    let address = proxy.base_url.trim_start_matches("http://").to_owned();
    let client = std::thread::spawn(move || {
        let mut stream = std::net::TcpStream::connect(address).expect("connect");
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(10)))
            .expect("timeout");
        stream
            .write_all(
                b"POST /v1/responses HTTP/1.1\r\nhost: proxy\r\ncontent-length: 2\r\nconnection: close\r\n\r\n{}",
            )
            .expect("request");
        let mut bytes = Vec::new();
        let mut buffer = [0_u8; 4096];
        let mut complete_at = None;
        while let Ok(read) = stream.read(&mut buffer) {
            if read == 0 {
                break;
            }
            bytes.extend_from_slice(&buffer[..read]);
            if complete_at.is_none()
                && bytes
                    .windows(reply.len())
                    .any(|window| window == reply.as_bytes())
            {
                complete_at = Some(started.elapsed());
            }
        }
        (complete_at, bytes)
    });
    let (complete_at, bytes) = tokio::task::spawn_blocking(move || client.join())
        .await
        .expect("join task")
        .expect("client thread");
    // A proxy that never writes would leave the reader blocked in `open`:
    // opening the FIFO for writing ourselves releases it with no line.
    let (opened, line) = received
        .recv_timeout(std::time::Duration::from_secs(5))
        .unwrap_or_else(|_| {
            drop(std::fs::OpenOptions::new().write(true).open(&ledger_path));
            received
                .recv_timeout(std::time::Duration::from_secs(5))
                .expect("the reader thread ends once the FIFO has a writer")
        });

    let complete_at = complete_at.unwrap_or_else(|| {
        panic!(
            "the client never read the whole reply: {}",
            String::from_utf8_lossy(&bytes)
        )
    });
    assert!(
        line.contains("resp_fifo"),
        "no ledger line was written: {line:?}"
    );
    assert!(
        complete_at >= opened,
        "the client held the reply at {complete_at:?}, before the ledger was read at {opened:?}"
    );
}

/// rig-reqwest keeps at most 64 KiB of a non-success body, decoded as text;
/// the proxy reaches the provider without that, so an error reply of any
/// size or encoding reaches the client and the cassette byte for byte.
#[tokio::test]
async fn oversized_and_binary_error_bodies_pass_through_byte_for_byte() {
    use base64::Engine as _;

    let oversized: String = "gateway timeout ".repeat(100 * 1024 / 16);
    let binary: Vec<u8> = (0..=255_u8).cycle().take(4096).collect();
    let upstream = httpmock::MockServer::start_async().await;
    upstream
        .mock_async(|when, then| {
            when.method("POST").path("/v1/responses");
            then.status(502)
                .header("content-type", "text/html")
                .body(oversized.clone());
        })
        .await;
    upstream
        .mock_async(|when, then| {
            when.method("POST").path("/v1/audio/speech");
            then.status(500)
                .header("content-type", "application/octet-stream")
                .body(binary.clone());
        })
        .await;
    let dir = assert_fs::TempDir::new().expect("ledger directory");
    let (recorder, interactions) = recorder(&dir);
    let proxy = Proxy::start(upstream.base_url(), recorder)
        .await
        .expect("proxy starts");

    let gateway = client()
        .post(format!("{}/v1/responses", proxy.base_url))
        .body("{}")
        .send()
        .await
        .expect("forwarded");
    assert_eq!(gateway.status().as_u16(), 502);
    assert_eq!(gateway.bytes().await.expect("body"), oversized.as_bytes());
    let speech = client()
        .post(format!("{}/v1/audio/speech", proxy.base_url))
        .body("{}")
        .send()
        .await
        .expect("forwarded");
    assert_eq!(speech.status().as_u16(), 500);
    assert_eq!(speech.bytes().await.expect("body"), binary.as_slice());

    let interactions = interactions.lock().expect("interactions");
    assert_eq!(interactions.len(), 2);
    assert_eq!(interactions[0].then.status, 502);
    assert_eq!(
        interactions[0].then.body.as_deref(),
        Some(oversized.as_str())
    );
    assert_eq!(interactions[1].then.status, 500);
    assert_eq!(interactions[1].then.body_encoding, BodyEncoding::Base64);
    let recorded = base64::prelude::BASE64_STANDARD
        .decode(interactions[1].then.body.as_deref().unwrap_or_default())
        .expect("a base64 body");
    assert_eq!(recorded, binary);
}

/// A stream names what it created on an early event: the ledger line is on
/// disk while the stream is still open, before the client can read that
/// event, and a later event naming the same response adds nothing.
#[tokio::test]
async fn a_stream_reports_each_event_as_it_lands_and_nothing_twice() {
    use futures::StreamExt as _;
    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

    let created = "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_early\"}}\n\n";
    let delta = "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n";
    let completed =
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_early\"}}\n\n";
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("upstream listens");
    let origin = format!("http://{}", listener.local_addr().expect("address"));
    let (finish, finished) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("connection");
        let mut request = [0_u8; 4096];
        let _ = socket.read(&mut request).await;
        let head =
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n";
        socket
            .write_all(format!("{head}{created}").as_bytes())
            .await
            .expect("created");
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        socket.write_all(delta.as_bytes()).await.expect("delta");
        let _ = finished.await;
        socket
            .write_all(completed.as_bytes())
            .await
            .expect("completed");
    });
    let dir = assert_fs::TempDir::new().expect("ledger directory");
    let ledger_path = dir.path().join(ledger::LEDGER_FILE);
    let (recorder, interactions) = recorder(&dir);
    let proxy = Proxy::start(origin, recorder).await.expect("proxy starts");

    let response = client()
        .post(format!("{}/v1/responses", proxy.base_url))
        .body(r#"{"stream":true}"#)
        .send()
        .await
        .expect("the stream starts");
    let mut stream = response.bytes_stream();
    let mut read = Vec::new();
    while !String::from_utf8_lossy(&read).contains("resp_early") {
        let chunk = stream.next().await.expect("an event").expect("a chunk");
        read.extend_from_slice(&chunk);
    }
    // The upstream holds the stream open: this line came from its first event.
    let outstanding = ledger::outstanding(&ledger_path);
    assert_eq!(outstanding.len(), 1);
    assert_eq!(outstanding[0].id, "resp_early");

    finish.send(()).expect("the upstream is waiting");
    while let Some(chunk) = stream.next().await {
        read.extend_from_slice(&chunk.expect("a chunk"));
    }
    assert!(String::from_utf8_lossy(&read).ends_with(completed));
    let lines = std::fs::read_to_string(&ledger_path).expect("ledger");
    assert_eq!(lines.lines().count(), 1, "{lines}");
    assert_eq!(interactions.lock().expect("interactions").len(), 1);
}

/// An event stream is passed on as each chunk arrives: a provider that sends
/// one event and then waits on the client still gets that event to it, and
/// the ledger line it names was written first.
#[tokio::test]
async fn an_event_reaches_the_client_while_the_provider_waits_on_it() {
    use futures::StreamExt as _;
    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

    let created = "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_waits\"}}\n\n";
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("upstream listens");
    let origin = format!("http://{}", listener.local_addr().expect("address"));
    let (answered, answer) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("connection");
        let mut request = [0_u8; 4096];
        let _ = socket.read(&mut request).await;
        let head =
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n";
        socket
            .write_all(format!("{head}{created}").as_bytes())
            .await
            .expect("the event");
        // Nothing more until the client has the event.
        let _ = answer.await;
        socket.write_all(b"data: [DONE]\n\n").await.expect("done");
    });
    let dir = assert_fs::TempDir::new().expect("ledger directory");
    let (recorder, _) = recorder(&dir);
    let proxy = Proxy::start(origin, recorder).await.expect("proxy starts");

    let response = client()
        .post(format!("{}/v1/responses", proxy.base_url))
        .body(r#"{"stream":true}"#)
        .send()
        .await
        .expect("the stream starts");
    let mut stream = response.bytes_stream();
    let first = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
        .await
        .expect("the event arrives while the provider waits")
        .expect("a chunk")
        .expect("bytes");
    assert_eq!(first.as_ref(), created.as_bytes());
    let outstanding = ledger::outstanding(&dir.path().join(ledger::LEDGER_FILE));
    assert_eq!(outstanding.len(), 1);
    assert_eq!(outstanding[0].id, "resp_waits");
    answered.send(()).expect("the provider waits");
    while stream.next().await.is_some() {}
}

/// The proxy follows no redirect: a 3xx reaches the client and the cassette
/// as the provider sent it.
#[tokio::test]
async fn a_redirect_is_forwarded_as_sent() {
    let upstream = httpmock::MockServer::start_async().await;
    let elsewhere = upstream
        .mock_async(|when, then| {
            when.path("/v1/elsewhere");
            then.status(200).body("followed");
        })
        .await;
    upstream
        .mock_async(|when, then| {
            when.method("GET").path("/v1/files/file_1/content");
            then.status(302)
                .header("location", "/v1/elsewhere")
                .body("moved");
        })
        .await;
    let dir = assert_fs::TempDir::new().expect("ledger directory");
    let (recorder, interactions) = recorder(&dir);
    let proxy = Proxy::start(upstream.base_url(), recorder)
        .await
        .expect("proxy starts");

    let client = rig_reqwest::reqwest::Client::builder()
        .no_proxy()
        .redirect(rig_reqwest::reqwest::redirect::Policy::none())
        .build()
        .expect("client");
    let reply = client
        .get(format!("{}/v1/files/file_1/content", proxy.base_url))
        .send()
        .await
        .expect("forwarded");
    assert_eq!(reply.status().as_u16(), 302);
    assert_eq!(
        reply
            .headers()
            .get("location")
            .map(|value| value.as_bytes()),
        Some(&b"/v1/elsewhere"[..])
    );
    assert_eq!(reply.text().await.expect("body"), "moved");
    elsewhere.assert_calls_async(0).await;

    let interactions = interactions.lock().expect("interactions");
    assert_eq!(interactions.len(), 1);
    assert_eq!(interactions[0].then.status, 302);
    assert_eq!(interactions[0].then.body.as_deref(), Some("moved"));
}
