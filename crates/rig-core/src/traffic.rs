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

use bytes::Bytes;
use futures::Stream;
use http::{HeaderMap, Request, Response, StatusCode};

use crate::http_client::{
    self, BoxedStream, DynHttpClient, HttpClientExt, LazyBody, MultipartForm, StreamingResponse,
};
use crate::serve::Serving;
#[cfg(feature = "websocket")]
use crate::wasm_compat::WasmBoxedFuture;
use crate::wasm_compat::WasmCompatSend;
#[cfg(feature = "websocket")]
use crate::ws_client::{
    BoxedWebSocketConnection, CloseFrame, ConnectOptions, Frame, WebSocketClientExt,
    WebSocketConnection,
};

mod entry;
#[cfg(test)]
mod tests;

/// Where a log's exchanges go: a writer thread that redacts and writes
/// them. Cloning is cheap; the clones share the thread, which ends when the
/// last clone and every client it wrapped are gone.
///
/// Each line is one exchange: `exchange` (its number in this log, in the
/// order they started) and the `pid` of the process, `started_ms` (Unix milliseconds), the `effect`, `handler` and
/// `provider` of the model call it served when the request carried a
/// [`Serving`], the `request` (`method`, `url`, `headers`, `body`), the
/// `response` (`status`, `headers`, and a unary `body` or the streamed
/// `events`, each with its `ms` since the start), the `timings`
/// (`headers_ms`, `first_byte_ms` and `last_byte_ms` of a stream, `end_ms`),
/// the `error`, and `end`: `complete`, `error`, or `stopped` when the caller
/// stopped reading first, as a streaming reader does after the provider's
/// final event. A body is decoded from gzip or deflate; a JSON body, and
/// the JSON of each event, is written as JSON, re-serialized; a body that
/// is not text only by its size; a multipart body as its parts, binary ones
/// by size. A stream's chunks are joined and cut into SSE events, or lines,
/// each with the time of the chunk that ended it. Credentials
/// (authentication headers, cookies, URL user info, secret query
/// parameters and JSON fields, and wherever their values appear again in
/// the exchange) and every secret [`crate::redact`] finds are replaced by
/// [`crate::redact::REDACTED`]. A websocket session is kept in memory
/// until it ends.
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

    /// A log appending to the file at `path`, made readable by the owner
    /// alone on Unix, creating it and its parent directories when missing.
    pub fn create(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        #[cfg(unix)]
        fs::set_permissions(
            path,
            <fs::Permissions as std::os::unix::fs::PermissionsExt>::from_mode(0o600),
        )?;
        Self::new(file)
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
    let mut warn = |failure: &str| {
        if !std::mem::replace(&mut warned, true) {
            tracing::warn!("the traffic log could not write: {failure}");
        }
    };
    let mut torn = false;
    while let Ok(message) = receiver.recv() {
        let mut next = Some(message);
        while let Some(message) = next {
            match message {
                Message::Exchange(exchange) => {
                    // After a failed write a line may be cut short: start
                    // the next on a line of its own.
                    let mut line = if torn { "\n".to_owned() } else { String::new() };
                    line.push_str(&entry::line(&exchange).to_string());
                    line.push('\n');
                    torn = match out.write_all(line.as_bytes()) {
                        Ok(()) => false,
                        Err(failure) => {
                            warn(&failure.to_string());
                            true
                        }
                    };
                }
                Message::Flush(done) => {
                    if let Err(failure) = out.flush() {
                        warn(&failure.to_string());
                    }
                    done.send(()).ok();
                }
            }
            next = receiver.try_recv().ok();
        }
        if let Err(failure) = out.flush() {
            warn(&failure.to_string());
        }
    }
}
