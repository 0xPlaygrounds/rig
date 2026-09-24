use super::*;

fn request(path: &str, body: &str) -> Vec<u8> {
    format!(
        "POST {path} HTTP/1.1\r\nhost: proxy\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
        body.len()
    )
    .into_bytes()
}

fn json_response(body: &str) -> Vec<u8> {
    format!(
        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
        body.len()
    )
    .into_bytes()
}

fn chunked(body: &[u8], chunk: usize) -> Vec<u8> {
    let mut out = Vec::new();
    for piece in body.chunks(chunk) {
        out.extend(format!("{:x}\r\n", piece.len()).into_bytes());
        out.extend(piece);
        out.extend(b"\r\n");
    }
    out.extend(b"0\r\n\r\n");
    out
}

#[test]
fn a_whole_reply_pairs_with_its_request() {
    let mut tap = Tap::default();
    tap.request_bytes(&request("/v1/responses", r#"{"input":"hi"}"#));
    let created = tap.response_bytes(&json_response(r#"{"id":"resp_1"}"#));
    assert_eq!(
        created,
        [TappedCreation {
            status: 200,
            method: "POST".into(),
            path: "/v1/responses".into(),
            request_body: br#"{"input":"hi"}"#.to_vec(),
            response_body: br#"{"id":"resp_1"}"#.to_vec(),
        }]
    );
}

#[test]
fn an_interim_reply_does_not_take_the_request_and_an_upgrade_stops_the_tap() {
    let mut tap = Tap::default();
    tap.request_bytes(&request("/v1/responses", "{}"));
    let mut reply = b"HTTP/1.1 100 Continue\r\n\r\n".to_vec();
    reply.extend(json_response(r#"{"id":"resp_1"}"#));
    let created = tap.response_bytes(&reply);
    assert_eq!(created.len(), 1);
    assert_eq!(created[0].path, "/v1/responses");

    tap.request_bytes(b"GET /v1/responses HTTP/1.1\r\nupgrade: websocket\r\n\r\n");
    assert!(
        tap.response_bytes(b"HTTP/1.1 101 Switching Protocols\r\n\r\n")
            .is_empty()
    );
    // WebSocket frames are not HTTP: nothing is parsed or buffered.
    tap.request_bytes(&[0x81, 0x85, 1, 2, 3, 4, b'\r', b'\n', b'\r', b'\n']);
    assert!(
        tap.response_bytes(&json_response(r#"{"id":"resp_2"}"#))
            .is_empty()
    );
    assert!(tap.requests.buffer.is_empty() && tap.responses.buffer.is_empty());
}

#[test]
fn bytes_split_anywhere_still_parse() {
    let mut tap = Tap::default();
    for byte in request("/v1/files", r#"{"purpose":"x"}"#) {
        tap.request_bytes(&[byte]);
    }
    let reply = json_response(r#"{"id":"file-1"}"#);
    let mut created = Vec::new();
    for byte in reply {
        created.extend(tap.response_bytes(&[byte]));
    }
    assert_eq!(created.len(), 1);
    assert_eq!(created[0].response_body, br#"{"id":"file-1"}"#);
}

#[test]
fn a_chunked_reply_is_decoded() {
    let mut tap = Tap::default();
    tap.request_bytes(&request("/v1/responses", "{}"));
    let mut reply =
        b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ntransfer-encoding: chunked\r\n\r\n"
            .to_vec();
    reply.extend(chunked(br#"{"id":"resp_chunked"}"#, 5));
    let created = tap.response_bytes(&reply);
    assert_eq!(created[0].response_body, br#"{"id":"resp_chunked"}"#);
}

#[test]
fn a_stream_reports_each_event_as_it_lands_and_nothing_twice() {
    let mut tap = Tap::default();
    tap.request_bytes(&request("/v1/responses", r#"{"stream":true}"#));
    let head =
        b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n";
    let created_event =
        b"data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_s\"}}\n\n";
    let delta_event = b"data: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n";

    assert!(tap.response_bytes(head).is_empty());
    // The creation event arrives in two pieces; it is reported once whole,
    // before the stream ends.
    let (first, rest) = created_event.split_at(20);
    let mut wire = format!("{:x}\r\n", created_event.len()).into_bytes();
    wire.extend(first);
    assert!(tap.response_bytes(&wire).is_empty());
    let mut wire = rest.to_vec();
    wire.extend(b"\r\n");
    let early = tap.response_bytes(&wire);
    assert_eq!(early.len(), 1);
    assert_eq!(early[0].response_body, created_event);

    let mut wire = format!("{:x}\r\n", delta_event.len()).into_bytes();
    wire.extend(delta_event);
    wire.extend(b"\r\n0\r\n\r\n");
    let late = tap.response_bytes(&wire);
    let reported: Vec<u8> = late.iter().flat_map(|c| c.response_body.clone()).collect();
    assert!(
        !String::from_utf8_lossy(&reported).contains("resp_s"),
        "the creation event is not reported again: {}",
        String::from_utf8_lossy(&reported)
    );
}

#[test]
fn keep_alive_pairs_replies_with_requests_in_order() {
    let mut tap = Tap::default();
    tap.request_bytes(&[request("/v1/files", "{}"), request("/v1/responses", "{}")].concat());
    let created = tap.response_bytes(
        &[
            json_response(r#"{"id":"file-1"}"#),
            json_response(r#"{"id":"resp_2"}"#),
        ]
        .concat(),
    );
    let paths: Vec<_> = created.iter().map(|c| c.path.as_str()).collect();
    assert_eq!(paths, ["/v1/files", "/v1/responses"]);
}

#[test]
fn a_reply_delimited_by_close_completes_at_close() {
    let mut tap = Tap::default();
    tap.request_bytes(&request("/v1/responses", "{}"));
    assert!(
        tap.response_bytes(
            b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\r\n{\"id\":\"resp_c\"}"
        )
        .is_empty()
    );
    let created = tap.end_of_responses();
    assert_eq!(created[0].response_body, br#"{"id":"resp_c"}"#);
}

#[test]
fn error_replies_carry_their_status() {
    let mut tap = Tap::default();
    tap.request_bytes(&request("/v1/responses", "{}"));
    let body = r#"{"error":{"type":"invalid_request_error"}}"#;
    let reply = format!(
        "HTTP/1.1 400 Bad Request\r\ncontent-length: {}\r\n\r\n{body}",
        body.len()
    );
    assert_eq!(tap.response_bytes(reply.as_bytes())[0].status, 400);
}

#[tokio::test]
async fn the_relay_forwards_bytes_unchanged_and_writes_the_ledger_first() {
    let upstream = httpmock::MockServer::start_async().await;
    let mock = upstream
        .mock_async(|when, then| {
            when.method("POST")
                .path("/v1/responses")
                .header("accept", "text/plain")
                .body(r#"{"input":"hi"}"#);
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"id":"resp_relay","object":"response"}"#);
        })
        .await;
    let dir = assert_fs::TempDir::new().expect("ledger directory");
    let ledger_path = dir.path().join("ledger.jsonl");
    let relay = Relay::start(
        upstream.address().to_string(),
        LedgerTarget {
            path: ledger_path.clone(),
            provider: "openai".into(),
            scenario: "relay/test".into(),
            origin: "https://api.openai.com".into(),
        },
    )
    .await
    .expect("relay starts");

    let client = rig_reqwest::reqwest::Client::builder()
        .no_proxy()
        .build()
        .expect("client");
    let body = client
        .post(format!("{}/v1/responses", relay.base_url))
        .header("accept", "text/plain")
        .body(r#"{"input":"hi"}"#)
        .send()
        .await
        .expect("relayed")
        .text()
        .await
        .expect("body");
    assert_eq!(body, r#"{"id":"resp_relay","object":"response"}"#);
    mock.assert_async().await;

    // The line was written before the reply's bytes reached the client.
    let outstanding = crate::http::ledger::outstanding(&ledger_path);
    assert_eq!(outstanding.len(), 1);
    assert_eq!(outstanding[0].id, "resp_relay");
    assert_eq!(outstanding[0].scenario, "relay/test");
    assert_eq!(
        outstanding[0].delete_url,
        "https://api.openai.com/v1/responses/resp_relay"
    );
}

// The ledger is a FIFO whose reader opens late, so writing a line blocks the
// relay until then: the client must not hold the whole reply before that
// moment. The client is a blocking socket on its own thread, so a relay that
// blocks a runtime worker cannot delay the client's reads.
#[cfg(unix)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_client_cannot_read_the_reply_before_the_ledger_line_is_written() {
    use std::io::{Read as _, Write as _};

    let reply = r#"{"id":"resp_fifo","object":"response"}"#;
    let upstream = httpmock::MockServer::start_async().await;
    upstream
        .mock_async(|when, then| {
            when.method("POST").path("/v1/responses");
            then.status(200)
                .header("content-type", "application/json")
                .body(reply);
        })
        .await;
    let dir = assert_fs::TempDir::new().expect("ledger directory");
    let ledger_path = dir.path().join("ledger.jsonl");
    let made = std::process::Command::new("mkfifo")
        .arg(&ledger_path)
        .status()
        .expect("mkfifo runs");
    assert!(made.success());
    let relay = Relay::start(
        upstream.address().to_string(),
        LedgerTarget {
            path: ledger_path.clone(),
            provider: "openai".into(),
            scenario: "relay/fifo".into(),
            origin: "https://api.openai.com".into(),
        },
    )
    .await
    .expect("relay starts");

    let started = std::time::Instant::now();
    let (sender, received) = std::sync::mpsc::channel();
    let fifo = ledger_path.clone();
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(500));
        let opened = started.elapsed();
        let line = std::fs::read_to_string(&fifo).unwrap_or_default();
        let _ = sender.send((opened, line));
    });
    let address = relay.base_url.trim_start_matches("http://").to_owned();
    let client = std::thread::spawn(move || {
        let mut stream = std::net::TcpStream::connect(address).expect("connect");
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(10)))
            .expect("timeout");
        stream
            .write_all(
                b"POST /v1/responses HTTP/1.1\r\nhost: relay\r\ncontent-length: 2\r\nconnection: close\r\n\r\n{}",
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
            if complete_at.is_none() && bytes.ends_with(reply.as_bytes()) {
                complete_at = Some(started.elapsed());
            }
        }
        (complete_at, bytes)
    });
    let (complete_at, bytes) = tokio::task::spawn_blocking(move || client.join())
        .await
        .expect("join task")
        .expect("client thread");
    // A relay that never writes would leave the reader blocked in `open`:
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
