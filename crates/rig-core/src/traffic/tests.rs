use std::io::{self, Write};
use std::sync::{Arc, Mutex, PoisonError};

use bytes::Bytes;
use futures::StreamExt;
use http::{HeaderMap, HeaderValue, Request, StatusCode};
use serde_json::Value;

use super::TrafficLog;
use crate::effect::{EffectId, HandlerKey};
use crate::http_client::{self, HttpClientExt};
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

    fn entries(&self) -> Vec<Value> {
        self.text()
            .lines()
            .map(|line| serde_json::from_str(line).expect("each line is JSON"))
            .collect()
    }
}

fn log() -> (TrafficLog, Lines) {
    let lines = Lines::default();
    let log = TrafficLog::new(lines.clone()).expect("the writer thread starts");
    (log, lines)
}

#[tokio::test]
async fn an_error_reply_is_kept_and_the_key_is_not() {
    let mut headers = HeaderMap::new();
    headers.insert(
        "x-ratelimit-remaining-tokens",
        HeaderValue::from_static("0"),
    );
    headers.insert(
        "set-cookie",
        HeaderValue::from_static("session=abc123456789"),
    );
    let echoed = format!(r#"{{"error":{{"message":"Incorrect API key provided: {KEY}"}}}}"#);
    let http =
        RecordingHttpClient::with_error_headers(StatusCode::TOO_MANY_REQUESTS, echoed, headers);
    let (log, lines) = log();
    let request = Request::post(format!("https://example.com/v1/chat?key={KEY}&alt=sse"))
        .header("authorization", format!("Bearer {KEY}"))
        .header("x-api-key", KEY)
        .header("content-type", "application/json")
        .body(format!(r#"{{"model":"m","api_key":"{KEY}"}}"#))
        .expect("a request");

    let sent = log.wrap(http).send::<_, Bytes>(request).await;

    assert!(sent.is_err());
    log.flush();
    assert!(!lines.text().contains(KEY), "{}", lines.text());
    assert!(!lines.text().contains("abc123456789"));
    let entries = lines.entries();
    let [entry] = entries.as_slice() else {
        panic!("one entry: {entries:?}");
    };
    assert_eq!(entry["request"]["method"], "POST");
    assert_eq!(entry["request"]["body"]["model"], "m");
    assert_eq!(
        entry["request"]["url"],
        "https://example.com/v1/chat?key=[REDACTED]&alt=sse"
    );
    assert_eq!(
        entry["request"]["headers"]["content-type"],
        "application/json"
    );
    assert_eq!(entry["response"]["status"], 429);
    assert_eq!(
        entry["response"]["headers"]["x-ratelimit-remaining-tokens"],
        "0"
    );
    assert_eq!(
        entry["response"]["body"]["error"]["message"],
        "Incorrect API key provided: [REDACTED]"
    );
    assert!(entry["error"].as_str().is_some());
    assert!(entry["timings"]["end_ms"].as_f64().is_some());
}

#[tokio::test]
async fn a_streamed_reply_keeps_each_chunk_and_when_it_came() {
    let chunks = [
        "data: {\"a\":1}\n\n",
        "data: {\"b\":2}\n\n",
        "data: [DONE]\n\n",
    ];
    let http = SequencedStreamingHttpClient::new(
        chunks
            .iter()
            .map(|chunk| Ok(Bytes::from_static(chunk.as_bytes())))
            .collect(),
    );
    let (log, lines) = log();
    let mut request = Request::post("https://example.com/v1/responses")
        .body(Bytes::from_static(b"{\"stream\":true}"))
        .expect("a request");
    request.extensions_mut().insert(Serving {
        effect: EffectId::from_raw(7),
        handler: HandlerKey::new("model:openai/gpt-5"),
        provider: None,
    });

    let response = log
        .wrap(http)
        .send_streaming(request)
        .await
        .expect("a streaming response");
    let read: Vec<http_client::Result<Bytes>> = response.into_body().collect().await;

    assert_eq!(read.len(), chunks.len());
    log.flush();
    let entries = lines.entries();
    let [entry] = entries.as_slice() else {
        panic!("one entry: {entries:?}");
    };
    assert_eq!(entry["effect"], 7);
    assert_eq!(entry["handler"], "model:openai/gpt-5");
    assert_eq!(entry["request"]["body"]["stream"], true);
    assert_eq!(entry["response"]["status"], 200);
    let events = entry["response"]["events"].as_array().expect("events");
    let data: Vec<_> = events.iter().map(|event| event["data"].clone()).collect();
    assert_eq!(data, chunks);
    let times: Vec<f64> = events
        .iter()
        .map(|event| event["ms"].as_f64().expect("a time"))
        .collect();
    assert!(times.windows(2).all(|pair| pair[0] <= pair[1]));
    let timings = &entry["timings"];
    for timing in ["headers_ms", "first_byte_ms", "end_ms"] {
        assert!(timings[timing].as_f64().is_some(), "{timing} in {timings}");
    }
    assert!(entry.get("dropped").is_none());
}
