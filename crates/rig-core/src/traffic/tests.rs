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
    let echoed = format!(
        r#"{{"error":{{"message":"Bad key {KEY}, cookie cookie-0123456789 abcde, token refresh-0123456789"}}}}"#
    );
    let http =
        RecordingHttpClient::with_error_headers(StatusCode::TOO_MANY_REQUESTS, echoed, headers);
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
    let text = lines.text();
    for secret in [
        KEY,
        "cookie-0123456789",
        "refresh-0123456789",
        "abcde",
        "pass@",
    ] {
        assert!(!text.contains(secret), "{secret} in {text}");
    }
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
    assert_eq!(
        entry["response"]["body"]["error"]["message"],
        "Bad key [REDACTED], cookie [REDACTED] [REDACTED], token [REDACTED]"
    );
    assert_eq!(entry["end"], "error");
    assert!(entry["timings"]["end_ms"].as_f64().is_some());
}

#[tokio::test]
async fn a_stream_is_kept_as_its_events_with_when_each_came() {
    // Chunk boundaries split an event, the echoed key and a character; a
    // token sent in an event's JSON over two `data:` lines is echoed later.
    let chunks: [&[u8]; 5] = [
        b"data: {\"a\":\"caf\xc3",
        b"\xa9\"}\n\ndata: {\"echo\":\"plainkey-0123",
        b"456789abcdef\"}\n\nevent: auth\ndata: {\"access_token\":\n",
        b"data: \"opaque-1234\"}\n\ndata: {\"error\":\"bad token opaque-1234\"}\n\n",
        b"data: [DONE]\n\n",
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
    // As a provider's reader does, stop at the final event.
    let read: Vec<_> = response.into_body().take(chunks.len()).collect().await;

    assert_eq!(read.len(), chunks.len());
    log.flush();
    let text = lines.text();
    assert!(
        !text.contains(KEY) && !text.contains("opaque-1234"),
        "{text}"
    );
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
            "data: [DONE]",
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

    let secret = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789";
    let mut gzip = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    gzip.write_all(format!(r#"{{"tool_output":"{secret}"}}"#).as_bytes())
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
    log.flush();
    assert_eq!(
        lines.entry()["request"]["body"],
        json!({ "tool_output": "[REDACTED]" })
    );
}
