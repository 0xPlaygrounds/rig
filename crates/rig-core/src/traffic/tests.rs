use std::io::{self, Write};
use std::sync::{Arc, Mutex, PoisonError};

use bytes::Bytes;
use futures::StreamExt;
use http::{HeaderMap, HeaderValue, Request, StatusCode};
use serde_json::{Value, json};

use super::TrafficLog;
use crate::effect::{EffectId, HandlerKey};
use crate::http_client::HttpClientExt;
use crate::serve::Serving;
use crate::test_utils::{RecordingHttpClient, SequencedStreamingHttpClient};

/// A key of no known shape, so only the request it was sent with tells
/// the log it is one.
const KEY: &str = "plainkey-0123456789abcdef";

/// The lines a log wrote, shared with the test.
#[derive(Clone, Default)]
struct Lines(Arc<Mutex<Vec<u8>>>);

impl Write for Lines {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let mut lines = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        lines.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Lines {
    fn text(&self) -> String {
        let lines = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        String::from_utf8_lossy(&lines).into_owned()
    }

    /// The only entry written.
    fn entry(&self) -> Value {
        let text = self.text();
        let entries: Vec<Value> = text
            .lines()
            .map(|line| serde_json::from_str(line).expect("each line is JSON"))
            .collect();
        let [entry] = entries.as_slice() else {
            panic!("one entry: {text}");
        };
        entry.clone()
    }

    /// Fails when any of `secrets` was written.
    fn assert_hides(&self, secrets: &[&str]) {
        let text = self.text();
        for secret in secrets {
            assert!(!text.contains(secret), "{secret} in {text}");
        }
    }
}

fn log() -> (TrafficLog, Lines) {
    let lines = Lines::default();
    let log = TrafficLog::new(lines.clone()).expect("the writer thread starts");
    (log, lines)
}

#[tokio::test]
async fn an_error_reply_is_kept_and_no_credential_is() {
    let mut headers = HeaderMap::new();
    let header = HeaderValue::from_static;
    headers.insert("x-ratelimit-remaining-tokens", header("0"));
    headers.insert(
        "set-cookie",
        header("session=cookie-0123456789; Domain=example.com; Path=/v1"),
    );
    // The reply echoes each credential, one under a key, and sends one of
    // its own as an object, echoed too.
    let message = format!(
        "Bad key {KEY}, cookie cookie-0123456789 abcde, token refresh-0123456789, ek_opaque-0123"
    );
    let echoed = json!({
        "error": { "message": message },
        "client_secret": { "value": "ek_opaque-0123", "expires_at": 1 },
        KEY: "as a key",
    });
    let http = RecordingHttpClient::with_error_headers(
        StatusCode::TOO_MANY_REQUESTS,
        echoed.to_string(),
        headers,
    );
    let (log, lines) = log();
    // A token refresh: user info, the key under an encoded name, cookies
    // (one short) and a form body.
    let url = format!("https://user:pass@example.com/v1/token?%61pi_key={KEY}&alt=sse");
    let request = Request::post(url)
        .header("authorization", format!("Bearer {KEY}"))
        .header("cookie", "session=cookie-0123456789; lang=abcde")
        .header("content-type", "application/x-www-form-urlencoded")
        .body("grant_type=refresh_token&refresh_token=refresh-0123456789")
        .expect("a request");

    let sent = log.wrap_http(http).send::<_, Bytes>(request).await;

    assert!(sent.is_err());
    log.flush();
    lines.assert_hides(&[
        KEY,
        "cookie-0123456789",
        "refresh-0123456789",
        "abcde",
        "pass@",
        "ek_opaque-0123",
    ]);
    let entry = lines.entry();
    assert_eq!(entry["request"]["method"], "POST");
    assert_eq!(
        entry["request"]["url"],
        "https://[REDACTED]@example.com/v1/token?%61pi_key=[REDACTED]&alt=sse"
    );
    assert_eq!(
        entry["request"]["body"],
        "grant_type=refresh_token&refresh_token=[REDACTED]"
    );
    assert_eq!(entry["response"]["status"], 429);
    assert_eq!(
        entry["response"]["headers"]["x-ratelimit-remaining-tokens"],
        "0"
    );
    let message = "Bad key [REDACTED], cookie [REDACTED] [REDACTED], token [REDACTED], [REDACTED]";
    assert_eq!(
        entry["response"]["body"],
        json!({
            "error": { "message": message },
            "client_secret": { "value": "[REDACTED]", "expires_at": 1 },
            "[REDACTED]": "as a key",
        })
    );
    assert_eq!(entry["error"], "non-success status 429 Too Many Requests");
    assert_eq!(entry["end"], "error");
    assert!(entry["timings"]["end_ms"].as_f64().is_some());
}

#[tokio::test]
async fn a_stream_is_kept_as_its_events_with_when_each_came() {
    // Chunk boundaries split an event, the echoed key and a character; a
    // token sent in an event's JSON over two `data:` lines is echoed later;
    // the reader stops inside another.
    let chunks: [&[u8]; 5] = [
        b"data: {\"a\":\"caf\xc3",
        b"\xa9\"}\n\ndata: {\"echo\":\"plainkey-0123",
        b"456789abcdef\"}\n\nevent: auth\ndata: {\"access_token\":\n",
        b"data: \"opaque-1234\"}\n\ndata: {\"error\":\"bad token opaque-1234\"}\n\n",
        b"data: {\"clientSecret\":\ndata: {\"value\":\"opaque-5678",
    ];
    let http = SequencedStreamingHttpClient::new(
        chunks
            .iter()
            .map(|chunk| Ok(Bytes::copy_from_slice(chunk)))
            .collect(),
    );
    let (log, lines) = log();
    let mut request = Request::post("https://example.com/v1/responses")
        .header("x-api-key", KEY)
        .body(Bytes::from_static(b"{\"stream\":true}"))
        .expect("a request");
    request.extensions_mut().insert(Serving {
        effect: EffectId::from_raw(7),
        handler: HandlerKey::new("model:openai/gpt-5"),
        provider: None,
    });

    let response = log
        .wrap_http(http)
        .send_streaming(request)
        .await
        .expect("a streaming response");
    // Stop without reading the end of the body, as a reader does after the
    // provider's final event.
    let read: Vec<_> = response.into_body().take(chunks.len()).collect().await;

    assert_eq!(read.len(), chunks.len());
    log.flush();
    lines.assert_hides(&[KEY, "opaque-1234", "opaque-5678"]);
    let entry = lines.entry();
    assert_eq!(entry["effect"], 7);
    assert_eq!(entry["handler"], "model:openai/gpt-5");
    assert_eq!(entry["request"]["body"], json!({ "stream": true }));
    assert_eq!(entry["end"], "stopped");
    let events = entry["response"]["events"].as_array().expect("events");
    let data: Vec<_> = events.iter().map(|event| event["data"].clone()).collect();
    assert_eq!(
        data,
        [
            "data: {\"a\":\"café\"}",
            "data: {\"echo\":\"[REDACTED]\"}",
            "event: auth\ndata: {\"access_token\":\"[REDACTED]\"}",
            "data: {\"error\":\"bad token [REDACTED]\"}",
            "data: {\"clientSecret\":\ndata: \"[REDACTED]\"",
        ]
    );
    let times: Vec<f64> = events
        .iter()
        .map(|event| event["ms"].as_f64().expect("a time"))
        .collect();
    assert!(times.is_sorted());
    let timings = &entry["timings"];
    for timing in ["headers_ms", "first_byte_ms", "last_byte_ms", "end_ms"] {
        assert!(timings[timing].as_f64().is_some(), "{timing} in {timings}");
    }
}

#[tokio::test]
async fn a_compressed_body_is_read_back_and_redacted() {
    use std::io::Write as _;

    // A key of a known shape, built here so nothing that redacts this
    // file's text changes it.
    let secret = format!("sk-ant-api03-{}", "x".repeat(93));
    let mut gzip = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    gzip.write_all(json!({ "tool_output": secret }).to_string().as_bytes())
        .expect("the body compresses");
    let body = gzip.finish().expect("the body compresses");
    let (log, lines) = log();
    let request = Request::post("https://example.com/v1/messages")
        .header("content-encoding", "gzip")
        .body(body)
        .expect("a request");

    // The reply is dropped unread, which ends the exchange.
    let sent = log
        .wrap_http(RecordingHttpClient::new("{}"))
        .send::<_, Bytes>(request)
        .await
        .is_ok();

    assert!(sent);
    // The last handle gone, what is queued is written.
    drop(log);
    lines.assert_hides(&[&secret]);
    assert_eq!(
        lines.entry()["request"]["body"],
        json!({ "tool_output": "[REDACTED]" })
    );
}

#[tokio::test]
async fn json_in_a_string_is_redacted_and_a_schema_is_not() {
    // Tool parameters named like secrets; a call whose arguments, a JSON
    // string, hold a password after an escaped quote; a JSON tool result
    // with an object-valued secret; a command's JSON output with text after
    // it; each echoed in plain text.
    let body = json!({
        "tools": [{ "parameters": { "properties": {
            "token": { "type": "string", "description": "The account's token" },
            "api_key": { "type": ["string", "null"], "description": "An optional key" },
            "password": { "description": "The user's password" },
        } } }],
        "device_auth_id": "deviceauth-123456",
        "user_code": "ABCD-1234",
        "logprobs": { "content": [{ "token": "function", "logprob": -0.1 }] },
        "logprobsResult": { "chosenCandidates": [{ "token": "functional", "logProbability": -0.5 }] },
        "mcp_servers": [{ "authorization_token": "opaque-mcp-credential-123456789" }],
        "messages": [
            { "image_url": { "url": "https://storage.example/i.png?sig=opaque-signature-1234" } },
            { "content": "Error downloading https://storage.example/i.png?sig=opaque-signature-1234" },
            { "content": "Saved [it](https://u:pw-123456@cdn.example/f?X-Amz-Signature=sig-0123456789&v=2)." },
            { "tool_calls": [{ "arguments":
                r#"{"query":"say \"hi","password":"hunter22xx"}"#
            }] },
            { "content": r#"{"client_secret":{"value":"opaque-ephemeral-123"}}"# },
            { "content": "Login failed for hunter22xx with opaque-ephemeral-123" },
            { "content": "{\"password\":\"opaque-long-value\"}\n[exit code 0]" },
            { "content": "Login failed for opaque-long-value" },
            { "content": "MCP refused opaque-mcp-credential-123456789; functional words stay" },
        ],
    });
    let (log, lines) = log();
    let request = Request::post("https://example.com/v1/chat/completions")
        .body(body.to_string())
        .expect("a request");

    let sent = log
        .wrap_http(RecordingHttpClient::new("{}"))
        .send::<_, Bytes>(request)
        .await
        .is_ok();

    assert!(sent);
    log.flush();
    lines.assert_hides(&[
        "hunter22xx",
        "opaque-ephemeral-123",
        "opaque-long-value",
        "opaque-signature-1234",
        "sig-0123456789",
        "pw-123456",
        "deviceauth-123456",
        "ABCD-1234",
        "opaque-mcp-credential-123456789",
    ]);
    let entry = lines.entry();
    let body = &entry["request"]["body"];
    assert_eq!(
        body["tools"][0]["parameters"]["properties"],
        json!({
            "token": { "type": "string", "description": "The account's token" },
            "api_key": { "type": ["string", "null"], "description": "An optional key" },
            "password": { "description": "The user's password" },
        })
    );
    assert_eq!(body["logprobs"]["content"][0]["token"], "function");
    assert_eq!(
        body["logprobsResult"]["chosenCandidates"][0]["token"],
        "functional"
    );
    assert_eq!(
        body["messages"][2]["content"],
        "Saved [it](https://[REDACTED]@cdn.example/f?X-Amz-Signature=[REDACTED]&v=2)."
    );
    assert_eq!(
        body["messages"][5]["content"],
        "Login failed for [REDACTED] with [REDACTED]"
    );
    assert_eq!(
        body["messages"][3]["tool_calls"][0]["arguments"],
        r#"{"query":"say \"hi","password":"[REDACTED]"}"#
    );
}

#[tokio::test]
async fn a_websocket_session_keeps_each_frame_either_way() {
    use crate::wasm_compat::WasmBoxedFuture;
    use crate::ws_client::{
        BoxedWebSocketConnection, CloseFrame, ConnectOptions, Frame, WebSocketClientExt,
        WebSocketConnection,
    };

    /// A backend whose connection answers with `replies`, then ends.
    #[derive(Clone)]
    struct Scripted(Vec<Frame>);

    struct Connection(std::vec::IntoIter<Frame>);

    impl WebSocketClientExt for Scripted {
        async fn connect(
            &self,
            _request: Request<crate::http_client::NoBody>,
            _options: ConnectOptions,
        ) -> crate::http_client::Result<BoxedWebSocketConnection> {
            Ok(Box::new(Connection(self.0.clone().into_iter())))
        }
    }

    impl WebSocketConnection for Connection {
        fn send(&mut self, _frame: Frame) -> WasmBoxedFuture<'_, crate::http_client::Result<()>> {
            Box::pin(async { Ok(()) })
        }

        fn recv(&mut self) -> WasmBoxedFuture<'_, crate::http_client::Result<Option<Frame>>> {
            let next = self.0.next();
            Box::pin(async move { Ok(next) })
        }

        fn close(
            &mut self,
            _frame: Option<CloseFrame>,
        ) -> WasmBoxedFuture<'_, crate::http_client::Result<()>> {
            Box::pin(async { Ok(()) })
        }
    }

    let echo = format!(r#"{{"type":"error","message":"bad key {KEY}"}}"#);
    let backend = Scripted(vec![
        Frame::Text(echo),
        Frame::Binary(Bytes::from_static(b"\xff\x00")),
    ]);
    let (log, lines) = log();
    let request = Request::get("wss://example.com/v1/responses")
        .header("authorization", format!("Bearer {KEY}"))
        .body(crate::http_client::NoBody)
        .expect("a request");

    let mut socket = log
        .wrap_websocket(backend)
        .connect(request, ConnectOptions::new())
        .await
        .expect("a connection");
    socket
        .send(Frame::Text(r#"{"type":"response.create"}"#.to_owned()))
        .await
        .expect("a sent frame");
    while socket.recv().await.expect("a frame").is_some() {}

    log.flush();
    lines.assert_hides(&[KEY]);
    let entry = lines.entry();
    assert_eq!(entry["request"]["method"], "GET");
    assert_eq!(entry["end"], "complete");
    let frames: Vec<_> = entry["response"]["events"]
        .as_array()
        .expect("frames")
        .iter()
        .map(|frame| {
            (
                frame["dir"].clone(),
                frame["frame"].clone(),
                frame["data"].clone(),
            )
        })
        .collect();
    assert_eq!(
        frames,
        [
            (
                json!("sent"),
                json!("text"),
                json!({ "type": "response.create" })
            ),
            (
                json!("received"),
                json!("text"),
                json!({ "type": "error", "message": "bad key [REDACTED]" })
            ),
            (json!("received"), json!("binary"), json!({ "bytes": 2 })),
        ]
    );
}
