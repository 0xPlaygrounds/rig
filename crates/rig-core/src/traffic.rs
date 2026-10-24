//! An opt-in log of the HTTP exchanges and websocket sessions a model
//! provider client makes: each request as sent, the reply's status, headers
//! and body or streamed chunks with their times, and the error when one
//! failed. Native only.
//!
//! A [`TrafficLog`] wraps a transport ([`TrafficLog::wrap`]) and hands each
//! finished exchange to a writer thread, which removes secrets and writes
//! one JSON line per exchange. The request path only copies cheap handles;
//! a failed write is warned about once and the requests go on.
//!
//! ```no_run
//! use rig_core::providers::anthropic::wire::AnthropicConfig;
//! use rig_core::traffic::TrafficLog;
//!
//! # fn run(http: rig_core::http_client::DynHttpClient) -> std::io::Result<()> {
//! let log = TrafficLog::create("traffic.jsonl")?;
//! let provider = AnthropicConfig::new("key").connect(log.wrap(http));
//! # Ok(())
//! # }
//! ```

use std::fmt;
use std::fs::{self, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use bytes::Bytes;
use futures::Stream;
use http::{HeaderMap, Request, Response, StatusCode};
use serde_json::{Map, Value, json};

use crate::http_client::{
    self, BoxedStream, DynHttpClient, HttpClientExt, LazyBody, MultipartForm, StreamingResponse,
};
use crate::redact;
use crate::serve::Serving;
use crate::wasm_compat::WasmCompatSend;
#[cfg(feature = "websocket")]
use crate::ws_client::{
    BoxedWebSocketConnection, CloseFrame, ConnectOptions, Frame, WebSocketClientExt,
    WebSocketConnection,
};

#[cfg(test)]
mod tests;

/// Where a log's exchanges go: a writer thread that redacts and writes
/// them. Cloning is cheap; the clones share the thread, which ends when the
/// last clone and every client it wrapped are gone.
#[derive(Clone)]
pub struct TrafficLog {
    shared: Arc<Shared>,
}

struct Shared {
    sender: Sender<Message>,
    exchanges: AtomicU64,
}

enum Message {
    Exchange(Box<Exchange>),
    Flush(Sender<()>),
}

impl fmt::Debug for TrafficLog {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("TrafficLog")
    }
}

impl TrafficLog {
    /// A log writing JSON lines to `writer` on a thread of its own. Fails
    /// only when the thread cannot start.
    pub fn new(writer: impl Write + Send + 'static) -> io::Result<Self> {
        let (sender, receiver) = mpsc::channel();
        std::thread::Builder::new()
            .name("rig-traffic-log".to_owned())
            .spawn(move || write_lines(&receiver, BufWriter::new(writer)))?;
        Ok(Self {
            shared: Arc::new(Shared {
                sender,
                exchanges: AtomicU64::new(0),
            }),
        })
    }

    /// A log appending to the file at `path`, creating it and its parent
    /// directories when missing.
    pub fn create(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }
        Self::new(OpenOptions::new().create(true).append(true).open(path)?)
    }

    /// `http`, recording every request sent through it in this log.
    pub fn wrap(&self, http: impl HttpClientExt + 'static) -> DynHttpClient {
        DynHttpClient::new(Logged {
            inner: DynHttpClient::new(http),
            log: self.clone(),
        })
    }

    /// `backend`, recording every websocket session opened through it in
    /// this log: the handshake, then each frame either way.
    #[cfg(feature = "websocket")]
    #[cfg_attr(docsrs, doc(cfg(feature = "websocket")))]
    pub fn websocket<W: WebSocketClientExt>(&self, backend: W) -> TrafficWebSocket<W> {
        TrafficWebSocket {
            inner: backend,
            log: self.clone(),
        }
    }

    /// Blocks until every exchange that finished before the call is
    /// written, such as before the process exits.
    pub fn flush(&self) {
        let (done, wait) = mpsc::channel();
        if self.shared.sender.send(Message::Flush(done)).is_ok() {
            wait.recv().ok();
        }
    }

    fn begin<B>(&self, request: &Request<B>, body: RequestBody) -> Recording {
        let number = self.shared.exchanges.fetch_add(1, Ordering::Relaxed) + 1;
        Recording {
            log: self.clone(),
            started: Instant::now(),
            exchange: Some(Box::new(Exchange {
                number,
                started_ms: crate::now_ms(),
                serving: request.extensions().get::<Serving>().cloned(),
                method: request.method().to_string(),
                uri: request.uri().to_string(),
                request_headers: request.headers().clone(),
                request_body: body,
                status: None,
                response_headers: None,
                response_body: None,
                events: Vec::new(),
                headers_at: None,
                first_byte_at: None,
                ended_at: None,
                error: None,
                dropped: false,
            })),
        }
    }
}

/// One exchange as it happened, secrets and all: the writer thread redacts
/// it.
struct Exchange {
    number: u64,
    started_ms: u64,
    serving: Option<Serving>,
    method: String,
    uri: String,
    request_headers: HeaderMap,
    request_body: RequestBody,
    status: Option<StatusCode>,
    response_headers: Option<HeaderMap>,
    response_body: Option<Bytes>,
    events: Vec<Event>,
    headers_at: Option<Duration>,
    first_byte_at: Option<Duration>,
    ended_at: Option<Duration>,
    error: Option<String>,
    dropped: bool,
}

enum RequestBody {
    Bytes(Bytes),
    Multipart,
    #[cfg(feature = "websocket")]
    None,
}

/// A streamed chunk, or a websocket frame, and when it came.
struct Event {
    at: Duration,
    data: EventData,
}

enum EventData {
    Chunk(Bytes),
    #[cfg(feature = "websocket")]
    Frame {
        sent: bool,
        frame: Frame,
    },
}

/// An exchange being recorded. It is written when it ends, or when it is
/// dropped before, as a reply whose body was never read.
struct Recording {
    log: TrafficLog,
    started: Instant,
    exchange: Option<Box<Exchange>>,
}

impl Recording {
    fn at(&self) -> Duration {
        self.started.elapsed()
    }

    fn update(&mut self, change: impl FnOnce(&mut Exchange, Duration)) {
        let at = self.at();
        if let Some(exchange) = self.exchange.as_deref_mut() {
            change(exchange, at);
        }
    }

    fn response<B>(&mut self, response: &Response<B>) {
        self.update(|exchange, at| {
            exchange.status = Some(response.status());
            exchange.response_headers = Some(response.headers().clone());
            exchange.headers_at = Some(at);
        });
    }

    /// Records `error`, with the status, headers and body of a non-success
    /// reply it kept, and ends the exchange.
    fn failed(&mut self, error: &http_client::Error) {
        self.update(|exchange, at| {
            if let (Some(status), Some(headers)) =
                (error.non_success_status(), error.non_success_headers())
            {
                exchange.status = Some(status);
                exchange.response_headers = Some(headers.clone());
                exchange.headers_at.get_or_insert(at);
            }
            if let Some(body) = error.non_success_body() {
                exchange.response_body = Some(Bytes::copy_from_slice(body.as_bytes()));
            }
            exchange.error = Some(error.to_string());
        });
        self.end();
    }

    /// Records a unary reply's body, or why it could not be read, and ends
    /// the exchange.
    fn body(&mut self, body: &http_client::Result<Bytes>) {
        match body {
            Ok(bytes) => self.update(|exchange, at| {
                exchange.first_byte_at = Some(at);
                exchange.response_body = Some(bytes.clone());
            }),
            Err(error) => return self.failed(error),
        }
        self.end();
    }

    fn event(&mut self, data: EventData) {
        self.update(|exchange, at| {
            exchange.first_byte_at.get_or_insert(at);
            exchange.events.push(Event { at, data });
        });
    }

    /// Writes the exchange, once.
    fn end(&mut self) {
        let at = self.at();
        if let Some(mut exchange) = self.exchange.take() {
            exchange.ended_at = Some(at);
            self.log
                .shared
                .sender
                .send(Message::Exchange(exchange))
                .ok();
        }
    }
}

impl Drop for Recording {
    fn drop(&mut self) {
        self.update(|exchange, _| exchange.dropped = exchange.error.is_none());
        self.end();
    }
}

/// A transport whose requests are recorded.
struct Logged {
    inner: DynHttpClient,
    log: TrafficLog,
}

impl HttpClientExt for Logged {
    fn send<T, U>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes> + WasmCompatSend,
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        let req = req.map(Into::into);
        let mut recording = self.log.begin(&req, RequestBody::Bytes(req.body().clone()));
        let sent = self.inner.send::<Bytes, Bytes>(req);
        async move {
            match sent.await {
                Ok(response) => {
                    recording.response(&response);
                    Ok(response.map(|body| -> LazyBody<U> {
                        Box::pin(async move {
                            let body = body.await;
                            recording.body(&body);
                            body.map(U::from)
                        })
                    }))
                }
                Err(error) => {
                    recording.failed(&error);
                    Err(error)
                }
            }
        }
    }

    fn send_multipart<U>(
        &self,
        req: Request<MultipartForm>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        let mut recording = self.log.begin(&req, RequestBody::Multipart);
        let sent = self.inner.send_multipart::<Bytes>(req);
        async move {
            match sent.await {
                Ok(response) => {
                    recording.response(&response);
                    Ok(response.map(|body| -> LazyBody<U> {
                        Box::pin(async move {
                            let body = body.await;
                            recording.body(&body);
                            body.map(U::from)
                        })
                    }))
                }
                Err(error) => {
                    recording.failed(&error);
                    Err(error)
                }
            }
        }
    }

    fn send_streaming<T>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        let req = req.map(Into::into);
        let mut recording = self.log.begin(&req, RequestBody::Bytes(req.body().clone()));
        let sent = self.inner.send_streaming(req);
        async move {
            match sent.await {
                Ok(response) => {
                    recording.response(&response);
                    Ok(
                        response
                            .map(|body| -> BoxedStream { Box::pin(Tapped { body, recording }) }),
                    )
                }
                Err(error) => {
                    recording.failed(&error);
                    Err(error)
                }
            }
        }
    }
}

/// A streaming body whose chunks are recorded as they pass.
struct Tapped {
    body: BoxedStream,
    recording: Recording,
}

impl Stream for Tapped {
    type Item = http_client::Result<Bytes>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let polled = this.body.as_mut().poll_next(cx);
        match &polled {
            Poll::Ready(Some(Ok(chunk))) => this.recording.event(EventData::Chunk(chunk.clone())),
            Poll::Ready(Some(Err(error))) => this.recording.failed(error),
            Poll::Ready(None) => this.recording.end(),
            Poll::Pending => {}
        }
        polled
    }
}

/// A websocket backend whose sessions are recorded, from
/// [`TrafficLog::websocket`].
#[cfg(feature = "websocket")]
#[cfg_attr(docsrs, doc(cfg(feature = "websocket")))]
#[derive(Clone)]
pub struct TrafficWebSocket<W> {
    inner: W,
    log: TrafficLog,
}

#[cfg(feature = "websocket")]
impl<W: WebSocketClientExt> WebSocketClientExt for TrafficWebSocket<W> {
    fn connect(
        &self,
        request: Request<http_client::NoBody>,
        options: ConnectOptions,
    ) -> impl Future<Output = http_client::Result<BoxedWebSocketConnection>> + WasmCompatSend {
        let mut recording = self.log.begin(&request, RequestBody::None);
        let connecting = self.inner.connect(request, options);
        async move {
            match connecting.await {
                Ok(socket) => {
                    recording.update(|exchange, at| exchange.headers_at = Some(at));
                    let logged: BoxedWebSocketConnection =
                        Box::new(LoggedSocket { socket, recording });
                    Ok(logged)
                }
                Err(error) => {
                    recording.failed(&error);
                    Err(error)
                }
            }
        }
    }
}

/// An open websocket whose frames are recorded.
#[cfg(feature = "websocket")]
struct LoggedSocket {
    socket: BoxedWebSocketConnection,
    recording: Recording,
}

#[cfg(feature = "websocket")]
impl WebSocketConnection for LoggedSocket {
    fn send(
        &mut self,
        frame: Frame,
    ) -> crate::wasm_compat::WasmBoxedFuture<'_, http_client::Result<()>> {
        self.recording.event(EventData::Frame {
            sent: true,
            frame: frame.clone(),
        });
        self.socket.send(frame)
    }

    fn recv(
        &mut self,
    ) -> crate::wasm_compat::WasmBoxedFuture<'_, http_client::Result<Option<Frame>>> {
        let Self { socket, recording } = self;
        Box::pin(async move {
            let received = socket.recv().await;
            match &received {
                Ok(Some(frame)) => recording.event(EventData::Frame {
                    sent: false,
                    frame: frame.clone(),
                }),
                Ok(None) => recording.end(),
                Err(error) => recording.failed(error),
            }
            received
        })
    }

    fn close(
        &mut self,
        frame: Option<CloseFrame>,
    ) -> crate::wasm_compat::WasmBoxedFuture<'_, http_client::Result<()>> {
        self.recording.event(EventData::Frame {
            sent: true,
            frame: Frame::Close(frame.clone()),
        });
        let Self { socket, recording } = self;
        Box::pin(async move {
            let closed = socket.close(frame).await;
            match &closed {
                Ok(()) => recording.end(),
                Err(error) => recording.failed(error),
            }
            closed
        })
    }
}

/// The writer thread: one redacted JSON line per exchange, flushed when no
/// more are waiting. A failed write is warned about once.
fn write_lines(receiver: &Receiver<Message>, mut out: impl Write) {
    let mut warned = false;
    let mut warn = |failure: io::Error| {
        if !std::mem::replace(&mut warned, true) {
            tracing::warn!("the traffic log could not write: {failure}");
        }
    };
    while let Ok(message) = receiver.recv() {
        let mut next = Some(message);
        let mut flushed = Vec::new();
        while let Some(message) = next {
            match message {
                Message::Exchange(exchange) => {
                    let mut line = exchange.to_json().to_string();
                    line.push('\n');
                    if let Err(failure) = out.write_all(line.as_bytes()) {
                        warn(failure);
                    }
                }
                Message::Flush(done) => flushed.push(done),
            }
            next = receiver.try_recv().ok();
        }
        if let Err(failure) = out.flush() {
            warn(failure);
        }
        for done in flushed {
            done.send(()).ok();
        }
    }
}

/// Headers whose values are credentials, or name the account they belong
/// to, matched without case.
const SECRET_HEADERS: &[&str] = &[
    "authorization",
    "proxy-authorization",
    "x-api-key",
    "api-key",
    "x-goog-api-key",
    "ocp-apim-subscription-key",
    "cookie",
    "set-cookie",
    "chatgpt-account-id",
    "openai-organization",
    "openai-project",
    "anthropic-organization-id",
    "x-amz-security-token",
    "x-amz-content-sha256",
    "x-amz-date",
    "key",
];

/// Whether a query parameter named `name` holds a credential.
fn is_secret_param(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    is_secret_field(&name)
        || matches!(
            name.as_str(),
            "key" | "api-key" | "token" | "secret" | "signature" | "sig"
        )
        || name.starts_with("x-amz-")
}

/// Whether a JSON field named `name` holds a credential, such as a token
/// exchange's reply.
fn is_secret_field(name: &str) -> bool {
    [
        "api_key",
        "apikey",
        "access_token",
        "refresh_token",
        "id_token",
        "client_secret",
        "password",
    ]
    .iter()
    .any(|secret| secret.eq_ignore_ascii_case(name))
}

/// Whether a header named `name` holds a credential: one of
/// [`SECRET_HEADERS`], or a name ending in `-key` or `-token`.
fn is_secret_header(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    SECRET_HEADERS.contains(&name.as_str()) || name.ends_with("-key") || name.ends_with("-token")
}

/// The secrets one exchange's request carried, removed wherever they show
/// up, such as an error body that echoes the key, and every secret
/// [`redact`] finds by its shape.
struct Scrubber {
    secrets: Vec<String>,
}

impl Scrubber {
    fn new(exchange: &Exchange) -> Self {
        let mut secrets = Vec::new();
        for (name, value) in &exchange.request_headers {
            if is_secret_header(name.as_str())
                && let Ok(value) = value.to_str()
            {
                secrets.push(value.to_owned());
                if let Some((_, credential)) = value.split_once(' ') {
                    secrets.push(credential.trim().to_owned());
                }
            }
        }
        for (name, value) in query_pairs(&exchange.uri) {
            if is_secret_param(name) {
                secrets.push(value.to_owned());
            }
        }
        // Short values would match ordinary text; a key is never that short.
        secrets.retain(|secret| secret.len() >= 8);
        secrets.sort_by_key(|secret| std::cmp::Reverse(secret.len()));
        Self { secrets }
    }

    fn text(&self, text: &str) -> String {
        let mut text = text.to_owned();
        for secret in &self.secrets {
            if text.contains(secret.as_str()) {
                text = text.replace(secret.as_str(), redact::REDACTED);
            }
        }
        redact::text(&text).into_owned()
    }

    fn json(&self, value: &mut Value) {
        match value {
            Value::String(text) => *text = self.text(text),
            Value::Array(items) => items.iter_mut().for_each(|item| self.json(item)),
            Value::Object(fields) => {
                for (name, field) in fields.iter_mut() {
                    if is_secret_field(name) && field.is_string() {
                        *field = Value::String(redact::REDACTED.to_owned());
                    } else {
                        self.json(field);
                    }
                }
            }
            Value::Null | Value::Bool(_) | Value::Number(_) => {}
        }
    }

    /// A body as JSON when it is, else as text.
    fn body(&self, bytes: &[u8]) -> Value {
        match serde_json::from_slice::<Value>(bytes) {
            Ok(mut value) => {
                self.json(&mut value);
                value
            }
            Err(_) => Value::String(self.text(&String::from_utf8_lossy(bytes))),
        }
    }

    fn url(&self, uri: &str) -> String {
        let url = match uri.split_once('?') {
            Some((path, query)) => {
                let query: Vec<String> = query
                    .split('&')
                    .map(|pair| match pair.split_once('=') {
                        Some((name, _)) if is_secret_param(name) => {
                            format!("{name}={}", redact::REDACTED)
                        }
                        _ => pair.to_owned(),
                    })
                    .collect();
                format!("{path}?{}", query.join("&"))
            }
            None => uri.to_owned(),
        };
        self.text(&url)
    }

    /// Headers as an object, duplicate names joined with `, `.
    fn headers(&self, headers: &HeaderMap) -> Value {
        let mut object = Map::new();
        for name in headers.keys() {
            let value = if is_secret_header(name.as_str()) {
                redact::REDACTED.to_owned()
            } else {
                let values: Vec<_> = headers
                    .get_all(name)
                    .iter()
                    .map(|value| String::from_utf8_lossy(value.as_bytes()).into_owned())
                    .collect();
                self.text(&values.join(", "))
            };
            object.insert(name.to_string(), Value::String(value));
        }
        Value::Object(object)
    }
}

fn query_pairs(uri: &str) -> impl Iterator<Item = (&str, &str)> {
    uri.split_once('?')
        .map(|(_, query)| query)
        .unwrap_or_default()
        .split('&')
        .filter_map(|pair| pair.split_once('='))
}

fn millis(duration: Duration) -> Value {
    json!((duration.as_secs_f64() * 1000.0 * 1000.0).round() / 1000.0)
}

impl Exchange {
    fn to_json(&self) -> Value {
        let scrub = Scrubber::new(self);
        let mut entry = Map::new();
        entry.insert("exchange".into(), json!(self.number));
        entry.insert("started_ms".into(), json!(self.started_ms));
        if let Some(serving) = &self.serving {
            entry.insert("effect".into(), json!(serving.effect));
            entry.insert("handler".into(), json!(serving.handler.as_str()));
            if let Some(provider) = &serving.provider {
                entry.insert("provider".into(), json!(provider.to_string()));
            }
        }

        let mut request = Map::new();
        request.insert("method".into(), json!(self.method));
        request.insert("url".into(), json!(scrub.url(&self.uri)));
        request.insert("headers".into(), scrub.headers(&self.request_headers));
        match &self.request_body {
            RequestBody::Bytes(bytes) if !bytes.is_empty() => {
                request.insert("body".into(), scrub.body(bytes));
            }
            RequestBody::Multipart => {
                request.insert("multipart".into(), json!(true));
            }
            _ => {}
        }
        entry.insert("request".into(), Value::Object(request));

        let mut response = Map::new();
        if let Some(status) = self.status {
            response.insert("status".into(), json!(status.as_u16()));
        }
        if let Some(headers) = &self.response_headers {
            response.insert("headers".into(), scrub.headers(headers));
        }
        if let Some(body) = &self.response_body {
            response.insert("body".into(), scrub.body(body));
        }
        if !self.events.is_empty() {
            let events = self.events.iter().map(|event| event.to_json(&scrub));
            response.insert("events".into(), Value::Array(events.collect()));
        }
        if !response.is_empty() {
            entry.insert("response".into(), Value::Object(response));
        }

        let mut timings = Map::new();
        for (name, at) in [
            ("headers_ms", self.headers_at),
            ("first_byte_ms", self.first_byte_at),
            ("end_ms", self.ended_at),
        ] {
            if let Some(at) = at {
                timings.insert(name.into(), millis(at));
            }
        }
        entry.insert("timings".into(), Value::Object(timings));
        if let Some(error) = &self.error {
            entry.insert("error".into(), json!(scrub.text(error)));
        }
        if self.dropped {
            entry.insert("dropped".into(), json!(true));
        }
        Value::Object(entry)
    }
}

impl Event {
    fn to_json(&self, scrub: &Scrubber) -> Value {
        let mut event = Map::new();
        event.insert("ms".into(), millis(self.at));
        match &self.data {
            EventData::Chunk(bytes) => {
                event.insert(
                    "data".into(),
                    json!(scrub.text(&String::from_utf8_lossy(bytes))),
                );
            }
            #[cfg(feature = "websocket")]
            EventData::Frame { sent, frame } => {
                event.insert("dir".into(), json!(if *sent { "sent" } else { "received" }));
                let (kind, data) = match frame {
                    Frame::Text(text) => ("text", scrub.text(text)),
                    Frame::Binary(bytes) => ("binary", scrub.text(&String::from_utf8_lossy(bytes))),
                    Frame::Ping(bytes) => ("ping", scrub.text(&String::from_utf8_lossy(bytes))),
                    Frame::Pong(bytes) => ("pong", scrub.text(&String::from_utf8_lossy(bytes))),
                    Frame::Close(close) => (
                        "close",
                        close.as_ref().map_or_else(String::new, |close| {
                            format!("{} {}", close.code, scrub.text(&close.reason))
                        }),
                    ),
                };
                event.insert("frame".into(), json!(kind));
                event.insert("data".into(), json!(data));
            }
        }
        Value::Object(event)
    }
}
