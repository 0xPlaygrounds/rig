//! An opt-in log of the HTTP exchanges and websocket sessions a model
//! provider client makes: each request as it reaches the log, the reply's
//! status, headers and body or streamed events with their times, and how
//! it ended. Native only.
//!
//! A [`TrafficLog`] wraps a transport ([`TrafficLog::wrap_http`]) and hands
//! each finished exchange to a writer thread, which removes secrets and
//! writes one JSON line per exchange, in the order they end. The request
//! path only copies cheap handles; a failed write is warned about once and
//! the requests go on.
//!
//! ```no_run
//! use rig_core::providers::anthropic::wire::AnthropicConfig;
//! use rig_core::traffic::TrafficLog;
//!
//! # fn run(http: rig_core::http_client::DynHttpClient) -> std::io::Result<()> {
//! let log = TrafficLog::create("traffic.jsonl")?;
//! let provider = AnthropicConfig::new("key").connect(log.wrap_http(http));
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

use base64::Engine as _;
use bytes::Bytes;
use futures::Stream;
use http::{HeaderMap, Request, Response, StatusCode, header};
use serde_json::{Map, Value, json};

use crate::http_client::multipart::PartContent;
use crate::http_client::{
    self, BoxedStream, DynHttpClient, HttpClientExt, LazyBody, MultipartForm, StreamingResponse,
};
use crate::redact;
use crate::serve::Serving;
#[cfg(feature = "websocket")]
use crate::wasm_compat::WasmBoxedFuture;
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
///
/// Each line is one exchange: `exchange` (its number, in the order they
/// started), `started_ms` (Unix milliseconds), the `effect`, `handler` and
/// `provider` of the model call it served when the request carried a
/// [`Serving`], the `request` (`method`, `url`, `headers`, `body`), the
/// `response` (`status`, `headers`, and a unary `body` or the streamed
/// `events`, each with its `ms` since the start), the `timings`
/// (`headers_ms`, `first_byte_ms` and `last_byte_ms` of a stream, `end_ms`),
/// the `error`, and `end`: `complete`, `error`, or `stopped` when the caller
/// stopped reading first, as a streaming reader does after the provider's
/// final event. A JSON body is written as JSON, re-serialized; a body that
/// is not UTF-8 as `{"base64": ...}`; a multipart body as its parts, binary
/// ones by size. Credentials (authentication headers, cookies, secret query
/// parameters and JSON fields, and wherever their values appear again) and
/// every secret [`redact`] finds are replaced by [`redact::REDACTED`].
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

    /// A log appending to the file at `path`, creating it, readable by the
    /// owner alone on Unix, and its parent directories when missing.
    pub fn create(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }
        let mut options = OpenOptions::new();
        options.create(true).append(true);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
        Self::new(options.open(path)?)
    }

    /// `http`, recording every request sent through it in this log, as the
    /// request reaches it: a wrapper inside it that rewrites requests or
    /// sends one again is not seen.
    pub fn wrap_http(&self, http: impl HttpClientExt + 'static) -> DynHttpClient {
        DynHttpClient::new(Logged {
            inner: DynHttpClient::new(http),
            log: self.clone(),
        })
    }

    /// `backend`, recording every websocket session opened through it in
    /// this log: the handshake request, then each frame either way, written
    /// when the session ends. A rejected handshake keeps its status,
    /// headers and body; an accepted one's are not part of the websocket
    /// contract.
    #[cfg(feature = "websocket")]
    #[cfg_attr(docsrs, doc(cfg(feature = "websocket")))]
    pub fn wrap_websocket<W: WebSocketClientExt>(&self, backend: W) -> TrafficWebSocket<W> {
        TrafficWebSocket {
            inner: backend,
            log: self.clone(),
        }
    }

    /// Blocks until every exchange that ended before the call is written,
    /// such as before the process exits.
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
                last_byte_at: None,
                ended_at: None,
                error: None,
                end: End::Stopped,
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
    last_byte_at: Option<Duration>,
    ended_at: Option<Duration>,
    error: Option<String>,
    end: End,
}

#[derive(Clone, Copy)]
enum End {
    Complete,
    Error,
    Stopped,
}

enum RequestBody {
    Bytes(Bytes),
    Multipart(MultipartForm),
    #[cfg(feature = "websocket")]
    None,
}

/// A streamed chunk, or a websocket frame, and when it came or went.
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

impl Event {
    fn chunk(&self) -> Option<&Bytes> {
        match &self.data {
            EventData::Chunk(bytes) => Some(bytes),
            #[cfg(feature = "websocket")]
            EventData::Frame { .. } => None,
        }
    }
}

/// An exchange being recorded. It is written when it ends, or when it is
/// dropped before, as a reply whose body was never read.
struct Recording {
    log: TrafficLog,
    started: Instant,
    exchange: Option<Box<Exchange>>,
}

impl Recording {
    fn update(&mut self, change: impl FnOnce(&mut Exchange, Duration)) {
        let at = self.started.elapsed();
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
        self.end(End::Error);
    }

    /// Records a unary reply's body, or why it could not be read, and ends
    /// the exchange.
    fn body(&mut self, body: &http_client::Result<Bytes>) {
        match body {
            Ok(bytes) => {
                self.update(|exchange, _| exchange.response_body = Some(bytes.clone()));
                self.end(End::Complete);
            }
            Err(error) => self.failed(error),
        }
    }

    /// Records an event; one that came in is a byte of the reply.
    fn event(&mut self, data: EventData, received: bool) {
        self.update(|exchange, at| {
            if received {
                exchange.first_byte_at.get_or_insert(at);
                exchange.last_byte_at = Some(at);
            }
            exchange.events.push(Event { at, data });
        });
    }

    /// Writes the exchange, once.
    fn end(&mut self, end: End) {
        let at = self.started.elapsed();
        if let Some(mut exchange) = self.exchange.take() {
            exchange.ended_at = Some(at);
            exchange.end = end;
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
        self.end(End::Stopped);
    }
}

/// A transport whose requests are recorded.
struct Logged {
    inner: DynHttpClient,
    log: TrafficLog,
}

/// The reply `sent` ended with, recorded: a unary body is recorded once
/// read.
async fn unary<U>(
    mut recording: Recording,
    sent: impl Future<Output = http_client::Result<Response<LazyBody<Bytes>>>>,
) -> http_client::Result<Response<LazyBody<U>>>
where
    U: From<Bytes> + WasmCompatSend + 'static,
{
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
        let recording = self.log.begin(&req, RequestBody::Bytes(req.body().clone()));
        unary(recording, self.inner.send::<Bytes, Bytes>(req))
    }

    fn send_multipart<U>(
        &self,
        req: Request<MultipartForm>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        let recording = self
            .log
            .begin(&req, RequestBody::Multipart(req.body().clone()));
        unary(recording, self.inner.send_multipart::<Bytes>(req))
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
            Poll::Ready(Some(Ok(chunk))) => {
                this.recording.event(EventData::Chunk(chunk.clone()), true);
            }
            Poll::Ready(Some(Err(error))) => this.recording.failed(error),
            Poll::Ready(None) => this.recording.end(End::Complete),
            Poll::Pending => {}
        }
        polled
    }
}

/// A websocket backend whose sessions are recorded, from
/// [`TrafficLog::wrap_websocket`].
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
    fn send(&mut self, frame: Frame) -> WasmBoxedFuture<'_, http_client::Result<()>> {
        let Self { socket, recording } = self;
        Box::pin(async move {
            let sent = socket.send(frame.clone()).await;
            match &sent {
                Ok(()) => recording.event(EventData::Frame { sent: true, frame }, false),
                Err(error) => recording.failed(error),
            }
            sent
        })
    }

    fn recv(&mut self) -> WasmBoxedFuture<'_, http_client::Result<Option<Frame>>> {
        let Self { socket, recording } = self;
        Box::pin(async move {
            let received = socket.recv().await;
            match &received {
                Ok(Some(frame)) => {
                    let frame = frame.clone();
                    recording.event(EventData::Frame { sent: false, frame }, true);
                }
                Ok(None) => recording.end(End::Complete),
                Err(error) => recording.failed(error),
            }
            received
        })
    }

    fn close(&mut self, frame: Option<CloseFrame>) -> WasmBoxedFuture<'_, http_client::Result<()>> {
        let Self { socket, recording } = self;
        Box::pin(async move {
            let closed = socket.close(frame.clone()).await;
            match &closed {
                Ok(()) => {
                    let frame = Frame::Close(frame);
                    recording.event(EventData::Frame { sent: true, frame }, false);
                    recording.end(End::Complete);
                }
                Err(error) => recording.failed(error),
            }
            closed
        })
    }
}

/// The writer thread: one redacted JSON line per exchange, flushed when no
/// more are waiting or a flush is asked. A failed write is warned about
/// once.
fn write_lines(receiver: &Receiver<Message>, mut out: impl Write) {
    let mut warned = false;
    let mut warn = |failure: io::Error| {
        if !std::mem::replace(&mut warned, true) {
            tracing::warn!("the traffic log could not write: {failure}");
        }
    };
    while let Ok(message) = receiver.recv() {
        let mut next = Some(message);
        while let Some(message) = next {
            match message {
                Message::Exchange(exchange) => {
                    let mut line = exchange.to_json().to_string();
                    line.push('\n');
                    if let Err(failure) = out.write_all(line.as_bytes()) {
                        warn(failure);
                    }
                }
                Message::Flush(done) => {
                    if let Err(failure) = out.flush() {
                        warn(failure);
                    }
                    done.send(()).ok();
                }
            }
            next = receiver.try_recv().ok();
        }
        if let Err(failure) = out.flush() {
            warn(failure);
        }
    }
}

/// Headers whose values are credentials, or name the account they belong
/// to, besides those [`is_secret_header`] finds by their names' shape.
const SECRET_HEADERS: &[&str] = &[
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

/// Whether a header named `name` holds a credential: one of
/// [`SECRET_HEADERS`], one naming authentication (`authorization`,
/// `x-auth-token`, ...), or one ending in `-key`, `-token`, `-secret` or
/// `-signature`.
fn is_secret_header(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    SECRET_HEADERS.contains(&name.as_str())
        || name.contains("auth")
        || ["-key", "-token", "-secret", "-signature"]
            .iter()
            .any(|end| name.ends_with(end))
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
        "token",
        "client_secret",
        "password",
        "device_code",
        "code_verifier",
    ]
    .iter()
    .any(|secret| secret.eq_ignore_ascii_case(name))
}

/// Whether a query or form parameter named `name` holds a credential.
fn is_secret_param(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    is_secret_field(&lower)
        || matches!(
            lower.as_str(),
            "key" | "api-key" | "secret" | "signature" | "sig"
        )
        || lower.starts_with("x-amz-")
}

/// A `name=value&...` text's pairs: each raw pair, with its name and value
/// decoded.
fn form_pairs(form: &str) -> impl Iterator<Item = (&str, String, String)> {
    form.split('&').map(|pair| {
        let decoded = url::form_urlencoded::parse(pair.as_bytes()).next();
        let (name, value) = decoded
            .map(|(name, value)| (name.into_owned(), value.into_owned()))
            .unwrap_or_default();
        (pair, name, value)
    })
}

/// The shortest value searched for elsewhere in an exchange: shorter ones
/// would match ordinary text, and no credential is that short.
const SHORTEST_SECRET: usize = 6;

/// The credentials one exchange carried, removed wherever they show up,
/// such as an error body that echoes the key, and every secret [`redact`]
/// finds by its shape.
struct Scrubber {
    secrets: Vec<String>,
}

impl Scrubber {
    fn new(exchange: &Exchange) -> Self {
        let mut secrets = Vec::new();
        for (name, value) in &exchange.request_headers {
            if is_secret_header(name.as_str()) {
                let value = String::from_utf8_lossy(value.as_bytes());
                credentials(name.as_str(), &value, &mut secrets);
            }
        }
        // A cookie set by the reply: its value, not its attributes.
        if let Some(headers) = &exchange.response_headers {
            for value in headers.get_all(header::SET_COOKIE) {
                let value = String::from_utf8_lossy(value.as_bytes());
                let cookie = value
                    .split(';')
                    .next()
                    .and_then(|cookie| cookie.split_once('='));
                secrets.extend(cookie.map(|(_, value)| value.trim().to_owned()));
            }
        }
        if let Ok(url) = url::Url::parse(&exchange.uri) {
            secrets.push(url.username().to_owned());
            secrets.extend(url.password().map(str::to_owned));
        }
        if let Some((_, query)) = exchange.uri.split_once('?') {
            form_secrets(query, &mut secrets);
        }
        match &exchange.request_body {
            RequestBody::Bytes(body) => body_secrets(body, &mut secrets),
            RequestBody::Multipart(form) => {
                for part in form.parts() {
                    if let PartContent::Text(text) = part.content()
                        && is_secret_param(part.name())
                    {
                        secrets.push(text.clone());
                    }
                }
            }
            #[cfg(feature = "websocket")]
            RequestBody::None => {}
        }
        if let Some(body) = &exchange.response_body {
            body_secrets(body, &mut secrets);
        }
        secrets.retain(|secret| secret.len() >= SHORTEST_SECRET);
        secrets.sort_by_key(|secret| std::cmp::Reverse(secret.len()));
        secrets.dedup();
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

    /// `form` with its secret parameters' values replaced.
    fn form(&self, form: &str) -> String {
        let pairs: Vec<String> = form_pairs(form)
            .map(|(pair, name, _)| match pair.split_once('=') {
                Some((raw, _)) if is_secret_param(&name) => {
                    format!("{raw}={}", redact::REDACTED)
                }
                _ => pair.to_owned(),
            })
            .collect();
        self.text(&pairs.join("&"))
    }

    /// A body as JSON when it is, a form's text with its secret fields
    /// replaced, other text, or `{"base64": ...}` when it is not UTF-8.
    fn body(&self, bytes: &[u8]) -> Value {
        if let Ok(mut value) = serde_json::from_slice::<Value>(bytes) {
            self.json(&mut value);
            return value;
        }
        match std::str::from_utf8(bytes) {
            Ok(text) if is_form(text) => Value::String(self.form(text)),
            Ok(text) => Value::String(self.text(text)),
            Err(_) => self.binary(bytes),
        }
    }

    /// Bytes that are not text, as base64, unless a credential of the
    /// exchange is in them.
    fn binary(&self, bytes: &[u8]) -> Value {
        let lossy = String::from_utf8_lossy(bytes);
        if self.text(&lossy) == lossy {
            json!({ "base64": base64::engine::general_purpose::STANDARD.encode(bytes) })
        } else {
            json!({ "bytes": bytes.len(), "redacted": true })
        }
    }

    /// One line of a stream: an SSE `data:` line's JSON, or a JSON line,
    /// redacted field by field.
    fn line(&self, line: &str) -> String {
        let (prefix, payload) = match line.strip_prefix("data:") {
            Some(payload) => ("data: ", payload.trim_start()),
            None => ("", line),
        };
        match serde_json::from_str::<Value>(payload) {
            Ok(mut value) if value.is_object() || value.is_array() => {
                self.json(&mut value);
                format!("{prefix}{value}")
            }
            _ => self.text(line),
        }
    }

    fn url(&self, uri: &str) -> String {
        match uri.split_once('?') {
            Some((path, query)) => format!("{}?{}", self.text(path), self.form(query)),
            None => self.text(uri),
        }
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

    /// A streamed body as its events: the body joined, so nothing is split
    /// between chunks, then cut into SSE events (`text/event-stream`) or
    /// lines, each with the time of the chunk that ended it. A body that is
    /// not UTF-8 stays in its chunks.
    fn stream(&self, events: &[Event], sse: bool) -> Vec<Value> {
        let chunks: Vec<(Duration, &Bytes)> = events
            .iter()
            .filter_map(|event| Some((event.at, event.chunk()?)))
            .collect();
        let joined: Vec<u8> = chunks
            .iter()
            .flat_map(|(_, bytes)| bytes.iter().copied())
            .collect();
        let Ok(text) = std::str::from_utf8(&joined) else {
            return chunks
                .iter()
                .map(|(at, bytes)| json!({ "ms": millis(*at), "data": self.binary(bytes) }))
                .collect();
        };
        // When each byte came: the time of the first chunk ending at or
        // after it.
        let mut ends = Vec::with_capacity(chunks.len());
        let mut end = 0;
        for (at, bytes) in &chunks {
            end += bytes.len();
            ends.push((end, *at));
        }
        let came = |offset: usize| {
            ends.iter()
                .find(|(end, _)| *end >= offset)
                .or(ends.last())
                .map_or(Duration::ZERO, |(_, at)| *at)
        };
        let mut out = Vec::new();
        let mut event: Vec<String> = Vec::new();
        let mut offset = 0;
        for line in text.split_inclusive('\n') {
            offset += line.len();
            let bare = line.trim_end_matches(['\r', '\n']);
            if !bare.is_empty() {
                event.push(self.line(bare));
            }
            let ended = if sse { bare.is_empty() } else { true };
            if ended && !event.is_empty() {
                out.push(json!({ "ms": millis(came(offset)), "data": event.join("\n") }));
                event.clear();
            }
        }
        if !event.is_empty() {
            out.push(json!({ "ms": millis(came(offset)), "data": event.join("\n") }));
        }
        out
    }
}

/// Whether `text` is a `name=value&...` form, as a token exchange sends.
fn is_form(text: &str) -> bool {
    !text.is_empty()
        && !text.contains(char::is_whitespace)
        && text.split('&').all(|pair| pair.contains('='))
}

/// The credential in a header's value: the value, the part after its
/// scheme (`Bearer ...`), and, for `cookie`, each cookie's value.
fn credentials(name: &str, value: &str, secrets: &mut Vec<String>) {
    secrets.push(value.to_owned());
    if let Some((_, credential)) = value.split_once(' ') {
        secrets.push(credential.trim().to_owned());
    }
    if name.eq_ignore_ascii_case("cookie") {
        for cookie in value.split(';') {
            secrets.extend(
                cookie
                    .split_once('=')
                    .map(|(_, value)| value.trim().to_owned()),
            );
        }
    }
}

/// The values of a form's or query's secret parameters, decoded and as
/// sent.
fn form_secrets(form: &str, secrets: &mut Vec<String>) {
    for (pair, name, value) in form_pairs(form) {
        if is_secret_param(&name) {
            secrets.extend(pair.split_once('=').map(|(_, raw)| raw.to_owned()));
            secrets.push(value);
        }
    }
}

/// The values of a body's secret JSON fields or form parameters.
fn body_secrets(body: &[u8], secrets: &mut Vec<String>) {
    fn walk(value: &Value, secrets: &mut Vec<String>) {
        match value {
            Value::Array(items) => items.iter().for_each(|item| walk(item, secrets)),
            Value::Object(fields) => {
                for (name, field) in fields {
                    match field {
                        Value::String(text) if is_secret_field(name) => secrets.push(text.clone()),
                        _ => walk(field, secrets),
                    }
                }
            }
            _ => {}
        }
    }
    if let Ok(value) = serde_json::from_slice::<Value>(body) {
        walk(&value, secrets);
    } else if let Ok(text) = std::str::from_utf8(body)
        && is_form(text)
    {
        form_secrets(text, secrets);
    }
}

fn millis(duration: Duration) -> Value {
    json!((duration.as_secs_f64() * 1_000_000.0).round() / 1000.0)
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
            RequestBody::Multipart(form) => {
                let parts = form.parts().iter().map(|part| {
                    let mut out = Map::new();
                    out.insert("name".into(), json!(part.name()));
                    if let Some(filename) = part.get_filename() {
                        out.insert("filename".into(), json!(scrub.text(filename)));
                    }
                    if let Some(content_type) = part.get_content_type() {
                        out.insert("content_type".into(), json!(content_type.to_string()));
                    }
                    match part.content() {
                        PartContent::Text(_) if is_secret_param(part.name()) => {
                            out.insert("text".into(), json!(redact::REDACTED));
                        }
                        PartContent::Text(text) => {
                            out.insert("text".into(), json!(scrub.text(text)));
                        }
                        PartContent::Binary(bytes) => {
                            out.insert("bytes".into(), json!(bytes.len()));
                        }
                    }
                    Value::Object(out)
                });
                request.insert("parts".into(), Value::Array(parts.collect()));
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
        let events = self.events_json(&scrub);
        if !events.is_empty() {
            response.insert("events".into(), Value::Array(events));
        }
        if !response.is_empty() {
            entry.insert("response".into(), Value::Object(response));
        }

        let mut timings = Map::new();
        for (name, at) in [
            ("headers_ms", self.headers_at),
            ("first_byte_ms", self.first_byte_at),
            ("last_byte_ms", self.last_byte_at),
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
        let end = match self.end {
            End::Complete => "complete",
            End::Error => "error",
            End::Stopped => "stopped",
        };
        entry.insert("end".into(), json!(end));
        Value::Object(entry)
    }

    fn events_json(&self, scrub: &Scrubber) -> Vec<Value> {
        #[cfg(feature = "websocket")]
        if matches!(self.request_body, RequestBody::None) {
            return self
                .events
                .iter()
                .map(|event| event.frame_json(scrub))
                .collect();
        }
        let sse = self
            .response_headers
            .as_ref()
            .and_then(|headers| headers.get(header::CONTENT_TYPE))
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.starts_with("text/event-stream"));
        if self.events.is_empty() {
            Vec::new()
        } else {
            scrub.stream(&self.events, sse)
        }
    }
}

#[cfg(feature = "websocket")]
impl Event {
    fn frame_json(&self, scrub: &Scrubber) -> Value {
        let EventData::Frame { sent, frame } = &self.data else {
            return json!({ "ms": millis(self.at) });
        };
        let (kind, data) = match frame {
            Frame::Text(text) => ("text", scrub.body(text.as_bytes())),
            Frame::Binary(bytes) => ("binary", scrub.body(bytes)),
            Frame::Ping(bytes) => ("ping", scrub.body(bytes)),
            Frame::Pong(bytes) => ("pong", scrub.body(bytes)),
            Frame::Close(close) => (
                "close",
                close.as_ref().map_or(
                    Value::Null,
                    |close| json!({ "code": close.code, "reason": scrub.text(&close.reason) }),
                ),
            ),
        };
        json!({
            "ms": millis(self.at),
            "dir": if *sent { "sent" } else { "received" },
            "frame": kind,
            "data": data,
        })
    }
}
