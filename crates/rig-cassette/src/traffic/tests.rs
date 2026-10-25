use std::io::{self, Write};
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use rig_core::effect::{EffectId, HandlerKey};
use rig_core::http_client::{
    self, BoxedStream, Bytes, HeaderMap, HeaderValue, HttpClientExt, LazyBody, MultipartForm,
    Request, Response, StatusCode, StreamingResponse,
};
use rig_core::serve::Serving;
use rig_core::wasm_compat::WasmCompatSend;
use serde_json::{Value, json};

use super::TrafficLog;

/// A key of no known shape, which only its place in the request reveals.
const KEY: &str = "plain-key-0f9e8d7c6b5a";
const QUERY_KEY: &str = "query-key-1a2b3c4d5e6f";

/// What the scripted provider answers: a status, headers, and a body or
/// the chunks of a stream.
#[derive(Clone)]
struct Scripted {
    status: StatusCode,
    headers: Vec<(&'static str, &'static str)>,
    chunks: Vec<String>,
}

impl Scripted {
    fn reply(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        for (name, value) in &self.headers {
            headers.append(*name, HeaderValue::from_static(value));
        }
        headers
    }
}

impl HttpClientExt for Scripted {
    fn send<T, U>(
        &self,
        _req: Request<T>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes> + WasmCompatSend,
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        let (status, headers, body) = (self.status, self.reply(), self.chunks.concat());
        async move {
            if !status.is_success() {
                return Err(http_client::Error::non_success_with_details(
                    status, headers, body,
                ));
            }
            let body: LazyBody<U> = Box::pin(async move { Ok(U::from(Bytes::from(body))) });
            let mut response = Response::new(body);
            *response.status_mut() = status;
            *response.headers_mut() = headers;
            Ok(response)
        }
    }

    fn send_multipart<U>(
        &self,
        req: Request<MultipartForm>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        self.send(req.map(|_| Bytes::new()))
    }

    fn send_streaming<T>(
        &self,
        _req: Request<T>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        let (status, headers) = (self.status, self.reply());
        let chunks: Vec<http_client::Result<Bytes>> = self
            .chunks
            .iter()
            .map(|chunk| Ok(Bytes::from(chunk.clone())))
            .collect();
        async move {
            let body: BoxedStream = Box::pin(futures::stream::iter(chunks));
            let mut response = Response::new(body);
            *response.status_mut() = status;
            *response.headers_mut() = headers;
            Ok(response)
        }
    }
}

/// The lines a log wrote, shared with the test.
#[derive(Clone, Default)]
struct Written(Arc<Mutex<Vec<u8>>>);

impl Write for Written {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0
            .lock()
            .map_err(|_| io::Error::other("poisoned"))?
            .extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Written {
    fn text(&self) -> String {
        self.0
            .lock()
            .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
            .unwrap_or_default()
    }

    fn lines(&self) -> Vec<Value> {
        self.text()
            .lines()
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect()
    }
}

fn request(body: &str) -> Request<Bytes> {
    let mut request = Request::post(format!(
        "https://user:{KEY}@api.example.com/v1/chat?key={QUERY_KEY}&stream=true"
    ))
    .header("authorization", format!("Bearer {KEY}"))
    .header("x-goog-api-key", QUERY_KEY)
    .header("content-type", "application/json")
    .body(Bytes::from(body.to_owned()))
    .unwrap_or_default();
    request.extensions_mut().insert(Serving::new(
        EffectId::from_raw(7),
        HandlerKey::new("model:example/m"),
    ));
    request
}

/// Sends `request` through a log over `provider`, reads the reply whole,
/// and returns what the log wrote.
async fn exchange(provider: Scripted, request: Request<Bytes>, streamed: bool) -> Written {
    let written = Written::default();
    let log = TrafficLog::new(written.clone()).unwrap_or_else(|_| unreachable!());
    let http = log.wrap_http(provider);
    if streamed {
        if let Ok(response) = http.send_streaming(request).await {
            let _: Vec<_> = response.into_body().collect().await;
        }
    } else if let Ok(response) = http.send::<_, Bytes>(request).await {
        response.into_body().await.ok();
    }
    log.flush();
    written
}

#[tokio::test]
async fn keys_set_in_a_request_reach_the_file_nowhere() {
    let authorization = "opaque-authorization-012345";
    let api_key = "opaque-header-123456";
    let query_key = "opaque-query-234567";
    let password = "opaque-userinfo-345678";
    let form_secret = "opaque-form-456789";
    let json_secret = "opaque-json-567890";
    for (body, body_secret) in [
        (
            format!("grant_type=refresh_token&refresh_token={form_secret}"),
            form_secret,
        ),
        (
            json!({"model": "m", "client_secret": {"value": json_secret}}).to_string(),
            json_secret,
        ),
    ] {
        let keys = [authorization, api_key, query_key, password, body_secret];
        let echo = json!({"error": {"message": keys.join(" ")}}).to_string();
        for streamed in [false, true] {
            let payload = if streamed {
                format!("data: {echo}\n\n")
            } else {
                echo.clone()
            };
            let middle = payload
                .find(body_secret)
                .map_or(payload.len() / 2, |at| at + body_secret.len() / 2);
            let provider = Scripted {
                status: if streamed {
                    StatusCode::OK
                } else {
                    StatusCode::UNAUTHORIZED
                },
                headers: vec![(
                    "content-type",
                    if streamed {
                        "text/event-stream"
                    } else {
                        "application/json"
                    },
                )],
                chunks: vec![
                    payload.get(..middle).unwrap_or_default().to_owned(),
                    payload.get(middle..).unwrap_or_default().to_owned(),
                ],
            };
            let request = Request::post(format!(
                "https://user:{password}@example.test/?%6Bey={query_key}"
            ))
            .header("authorization", format!("Bearer {authorization}"))
            .header("x-api-key", api_key)
            .body(Bytes::from(body.clone()))
            .unwrap_or_default();
            let written = exchange(provider, request, streamed).await;
            let text = written.text();
            assert!(!text.is_empty());
            for key in keys {
                assert!(!text.contains(key), "{key} in {text}");
            }
            let line = written.lines().into_iter().next().unwrap_or_default();
            assert_eq!(
                line.pointer("/request/headers/authorization"),
                Some(&json!("[REDACTED]"))
            );
            assert_eq!(
                line.pointer("/end"),
                Some(&json!(if streamed { "complete" } else { "error" }))
            );
        }
    }
}

#[tokio::test]
async fn a_secret_part_of_a_multipart_request_reaches_the_file_nowhere() {
    let secret = "opaque-multipart-012345";
    let provider = Scripted {
        status: StatusCode::BAD_REQUEST,
        headers: vec![("content-type", "application/json")],
        chunks: vec![json!({"error": {"message": format!("bad secret {secret}")}}).to_string()],
    };
    let form = MultipartForm::new()
        .text("client_secret", secret)
        .text("purpose", "fine-tune");
    let request = Request::post("https://api.example.com/v1/files")
        .body(form)
        .expect("a request");
    let written = Written::default();
    let log = TrafficLog::new(written.clone()).expect("the writer starts");
    log.wrap_http(provider)
        .send_multipart::<Bytes>(request)
        .await
        .ok();
    log.flush();
    let text = written.text();
    let line = written.lines().into_iter().next().unwrap_or_default();

    assert!(!text.contains(secret), "{text}");
    assert_eq!(
        line.pointer("/request/parts/1/value"),
        Some(&json!("fine-tune"))
    );
}

/// A body the recorder cannot read whole, here one sent with a
/// `content-encoding`, is kept by its size and content type only, so nothing
/// in it, nor an echo of it, can reach the line.
#[tokio::test]
async fn an_encoded_body_is_kept_by_its_size_only() {
    let secret = "opaque-body-credential-012345";
    let body = json!({"model": "m", "client_secret": secret}).to_string();
    let mut request = request("");
    *request.body_mut() = Bytes::from(body.clone());
    request
        .headers_mut()
        .insert("content-encoding", HeaderValue::from_static("gzip"));
    let provider = Scripted {
        status: StatusCode::BAD_REQUEST,
        headers: vec![("content-type", "text/plain")],
        chunks: vec!["invalid request".to_owned()],
    };

    let written = exchange(provider, request, false).await;
    let text = written.text();
    let line = written.lines().into_iter().next().unwrap_or_default();

    assert!(!text.contains(secret), "{text}");
    assert_eq!(
        line.pointer("/request/body"),
        Some(&json!({"bytes": body.len(), "content_type": "application/json"}))
    );
}

#[tokio::test]
async fn a_streamed_reply_keeps_each_event_and_its_time() {
    // The first event is cut across two chunks, and so is the second's data.
    let provider = Scripted {
        status: StatusCode::OK,
        headers: vec![
            ("content-type", "text/event-stream"),
            ("x-request-id", "req_1"),
        ],
        chunks: vec![
            "event: delta\ndata: {\"text\":".into(),
            "\"Hel\"}\n\nevent: delta\nda".into(),
            "ta: {\"text\":\"lo\"}\n\ndata: [DONE]\n\n".into(),
        ],
    };

    let written = exchange(provider, request("{}"), true).await;
    let line = written.lines().into_iter().next().unwrap_or_default();
    assert_eq!(line.pointer("/v"), Some(&json!(super::LINE_VERSION)));
    assert_eq!(line.pointer("/exchange"), Some(&json!(1)));
    assert_eq!(line.pointer("/effect"), Some(&json!(7)));
    assert_eq!(line.pointer("/handler"), Some(&json!("model:example/m")));
    let events = line
        .pointer("/response/events")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let data: Vec<_> = events
        .iter()
        .filter_map(|event| event.get("data"))
        .collect();

    assert_eq!(
        data,
        [
            &json!({"text": "Hel"}),
            &json!({"text": "lo"}),
            &json!("[DONE]")
        ]
    );
    assert!(events.iter().all(|event| event.get("ms").is_some()));
    assert_eq!(
        events.first().and_then(|event| event.get("event")),
        Some(&json!("delta"))
    );
    assert_eq!(
        line.pointer("/response/headers/x-request-id"),
        Some(&json!("req_1"))
    );
    for timing in ["headers_ms", "first_byte_ms", "last_byte_ms", "end_ms"] {
        assert!(
            line.pointer(&format!("/timings/{timing}")).is_some(),
            "{line}"
        );
    }
    assert_eq!(line.pointer("/end"), Some(&json!("complete")));
}

#[tokio::test]
async fn an_error_reply_keeps_its_status_headers_and_body() {
    let provider = Scripted {
        status: StatusCode::TOO_MANY_REQUESTS,
        headers: vec![
            ("content-type", "application/json"),
            ("retry-after", "7"),
            ("x-ratelimit-remaining-requests", "0"),
        ],
        chunks: vec![r#"{"error":{"type":"rate_limit"}}"#.into()],
    };

    let written = exchange(provider, request("{}"), false).await;
    let line = written.lines().into_iter().next().unwrap_or_default();

    assert_eq!(line.pointer("/response/status"), Some(&json!(429)));
    assert_eq!(
        line.pointer("/response/headers/x-ratelimit-remaining-requests"),
        Some(&json!("0"))
    );
    assert_eq!(
        line.pointer("/response/body/error/type"),
        Some(&json!("rate_limit"))
    );
    assert_eq!(
        line.pointer("/error"),
        Some(&json!("non-success status 429 Too Many Requests"))
    );
    assert_eq!(line.pointer("/end"), Some(&json!("error")));
}

#[tokio::test]
async fn a_call_is_written_against_its_own_conversations_last_one() {
    let provider = Scripted {
        status: StatusCode::OK,
        headers: vec![("content-type", "application/json")],
        chunks: vec!["{}".into()],
    };
    let first =
        r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[{"name":"t"}]}"#;
    // Another agent on the same model calls in between.
    let other =
        r#"{"model":"m","messages":[{"role":"user","content":"other"}],"tools":[{"name":"t"}]}"#;
    let second = r#"{"model":"m","messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"yo"}],"tools":[{"name":"t"}],"seed":1}"#;
    let written = Written::default();
    let log = TrafficLog::new(written.clone()).expect("the writer starts");
    let http = log.wrap_http(provider);
    for body in [first, other, second] {
        let response = http.send::<_, Bytes>(request(body)).await.expect("a reply");
        response.into_body().await.expect("its body");
    }
    log.flush();
    let lines = written.lines();
    let second_line = lines.get(2).cloned().unwrap_or_default();

    assert_eq!(
        lines
            .first()
            .and_then(|line| line.pointer("/request/body/model")),
        Some(&json!("m"))
    );
    assert_eq!(second_line.pointer("/request/after"), Some(&json!(1)));
    assert_eq!(
        second_line.pointer("/request/keep"),
        Some(&json!({"model": true, "messages": 1, "tools": true}))
    );
    assert_eq!(
        second_line.pointer("/request/body"),
        Some(&json!({"messages": [{"role": "assistant", "content": "yo"}], "seed": 1}))
    );
}

/// A writer that holds each write until `gate` lets it through, and says
/// when it finished one.
struct Gated {
    gate: std::sync::mpsc::Receiver<()>,
    done: std::sync::mpsc::Sender<Vec<u8>>,
}

impl Write for Gated {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let _ = self.gate.recv();
        let _ = self.done.send(bytes.to_vec());
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn ok_reply() -> Scripted {
    Scripted {
        status: StatusCode::OK,
        headers: vec![("content-type", "application/json")],
        chunks: vec!["{}".to_owned()],
    }
}

/// A request never waits on the file: with the writer stuck on a line, the
/// last exchange still completes after the log and its client are dropped,
/// and the writer writes what is queued once it can.
#[tokio::test]
async fn a_request_completes_while_the_writer_is_stuck_and_the_log_is_gone() {
    let (open, gate) = std::sync::mpsc::channel();
    let (done, written) = std::sync::mpsc::channel();
    let log = TrafficLog::new(Gated { gate, done }).expect("the writer starts");
    let http = log.wrap_http(ok_reply());
    // The first line takes the writer and holds it at the gate.
    let first = http.send::<_, Bytes>(request("{}")).await.expect("a reply");
    first.into_body().await.expect("a body");
    let last = http.send::<_, Bytes>(request("{}")).await.expect("a reply");
    drop(log);
    drop(http);

    // The body future now holds the last handle. A thread of its own lets a
    // regression that waits on the writer fail here instead of hanging.
    let body = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        tokio::task::spawn_blocking(move || futures::executor::block_on(last.into_body())),
    )
    .await
    .expect("the body completes without waiting on the writer")
    .expect("the body task");
    assert!(body.is_ok());

    for _ in 0..2 {
        open.send(()).expect("the writer waits");
        let line = written
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("the writer goes on writing what is queued");
        assert!(String::from_utf8_lossy(&line).contains("\"end\":\"complete\""));
    }
}

/// An exchange opens when its future first runs: a request handed to the
/// transport but never awaited, such as the losing branch of a `select!`,
/// writes nothing.
#[tokio::test]
async fn a_request_never_awaited_writes_nothing() {
    let written = Written::default();
    let log = TrafficLog::new(written.clone()).expect("the writer starts");
    let http = log.wrap_http(ok_reply());
    drop(http.send::<_, Bytes>(request("{}")));
    drop(http.send_streaming(request("{}")));
    log.flush();
    assert_eq!(written.text(), "");
}

/// The recorder asks for every reply uncompressed, whatever the client asked
/// for, and the line records the header as sent.
#[tokio::test]
async fn a_recorded_request_asks_for_its_reply_uncompressed() {
    let mut sent = request("{}");
    sent.headers_mut().insert(
        "accept-encoding",
        HeaderValue::from_static("gzip, br, zstd"),
    );
    let written = exchange(ok_reply(), sent, false).await;
    let lines = written.lines();
    assert_eq!(
        lines
            .first()
            .and_then(|line| line.pointer("/request/headers/accept-encoding")),
        Some(&json!("identity"))
    );
}

/// An exchange with `status`, a streamed reply of `chunks` or a unary `body`,
/// that ended with `end`; an event stream when `sse`.
fn ended(
    status: Option<u16>,
    chunks: Option<&[&str]>,
    body: Option<&str>,
    end: super::End,
    sse: bool,
) -> super::Exchange {
    let mut headers = HeaderMap::new();
    if sse {
        headers.insert(
            "content-type",
            HeaderValue::from_static("text/event-stream"),
        );
    }
    super::Exchange {
        number: 1,
        started_ms: 0,
        serving: None,
        method: "POST".to_owned(),
        uri: "https://api.example.com/v1/chat".to_owned(),
        request_headers: HeaderMap::new(),
        request_body: super::RequestBody::Bytes(Bytes::new()),
        status: status.and_then(|status| StatusCode::from_u16(status).ok()),
        response_headers: Some(headers),
        response_body: body.map(|body| Bytes::from(body.to_owned())),
        chunks: chunks.map(|chunks| {
            chunks
                .iter()
                .map(|chunk| (std::time::Duration::ZERO, Bytes::from((*chunk).to_owned())))
                .collect()
        }),
        headers_at: None,
        first_byte_at: None,
        last_byte_at: None,
        ended_at: None,
        error: None,
        end,
    }
}

/// One rule says whether a reply ended whole, for the cassette's refusals
/// and the line's withholding alike.
#[test]
fn whether_a_reply_ended_whole_is_one_rule() {
    use super::End::{Complete, Error, Stopped};

    let event = "data: {\"n\":1}\n\n";
    for (exchange, expected) in [
        (ended(None, None, None, Error, false), Some("got no reply")),
        (ended(Some(200), Some(&[event]), None, Complete, true), None),
        (ended(Some(200), None, Some("{}"), Complete, false), None),
        (
            ended(Some(200), Some(&[event]), None, Error, true),
            Some("failed midway"),
        ),
        (
            ended(Some(200), Some(&[]), None, Stopped, true),
            Some("was never read"),
        ),
        (ended(Some(200), Some(&[event]), None, Stopped, true), None),
        (
            ended(
                Some(200),
                Some(&[event, "data: {\"n\":"]),
                None,
                Stopped,
                true,
            ),
            Some("stopped before it was complete"),
        ),
        (
            ended(Some(200), Some(&["partial text"]), None, Stopped, false),
            Some("stopped before it was complete"),
        ),
        (
            ended(Some(429), None, Some("{\"error\":1}"), Error, false),
            None,
        ),
        (
            ended(Some(200), None, None, Error, false),
            Some("failed while its body was read"),
        ),
        (
            ended(Some(200), None, None, Stopped, false),
            Some("was never read"),
        ),
    ] {
        assert_eq!(
            exchange.incomplete(),
            expected,
            "{:?} {:?}",
            exchange.status,
            exchange.end
        );
    }
}
