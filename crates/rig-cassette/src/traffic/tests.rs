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
async fn a_line_names_the_effect_it_served() {
    let provider = Scripted {
        status: StatusCode::OK,
        headers: vec![("content-type", "application/json")],
        chunks: vec!["{}".into()],
    };

    let written = exchange(provider, request("{}"), false).await;
    let line = written.lines().into_iter().next().unwrap_or_default();

    assert_eq!(line.pointer("/v"), Some(&json!(super::LINE_VERSION)));
    assert_eq!(line.pointer("/exchange"), Some(&json!(1)));
    assert_eq!(line.pointer("/effect"), Some(&json!(7)));
    assert_eq!(line.pointer("/handler"), Some(&json!("model:example/m")));
    assert_eq!(line.pointer("/end"), Some(&json!("complete")));
}

#[tokio::test]
async fn a_call_repeating_the_last_ones_conversation_writes_only_what_is_new() {
    let provider = Scripted {
        status: StatusCode::OK,
        headers: vec![("content-type", "application/json")],
        chunks: vec!["{}".into()],
    };
    let first =
        r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[{"name":"t"}]}"#;
    let second = r#"{"model":"m","messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"yo"}],"tools":[{"name":"t"}],"seed":1}"#;
    let written = Written::default();
    let log = TrafficLog::new(written.clone()).expect("the writer starts");
    let http = log.wrap_http(provider);
    for body in [first, second] {
        let response = http.send::<_, Bytes>(request(body)).await.expect("a reply");
        response.into_body().await.expect("its body");
    }
    log.flush();
    let lines = written.lines();
    let second_line = lines.get(1).cloned().unwrap_or_default();

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
