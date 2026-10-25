//! Recording live provider traffic: every HTTP exchange a client sends, as one
//! scrubbed JSON line, for analysing what went over the wire. A [`TrafficLog`]
//! wraps a transport ([`TrafficLog::wrap_http`]) and hands each exchange, when
//! it ends, to a writer thread, which scrubs it with the cassette engine's
//! scrubber and writes it; a cassette recording, proxied or direct, captures
//! through the same wrapper. The request path only copies cheap handles and
//! never waits on the writer; a failed write is warned about once and requests
//! carry on. Native only: the writer is a thread.
//!
//! A line is version [`LINE_VERSION`] of this shape (absent fields are
//! omitted): `v`, `exchange` (its number in this log), `pid` and `started_ms`
//! (unique with `exchange` in a file several runs append to); `effect`,
//! `handler` and `provider` of the model call the request serves (the `id`,
//! `key` and provider of its record in an effect log); `request` with `method`,
//! `url`, `headers` and `body` (or a multipart request's `parts`); `response`
//! with `status`, `headers`, and a unary `body` or the streamed `events`, each
//! `{ms, event?, id?, data}`; `timings` with `headers_ms`, `first_byte_ms` (a
//! stream's), `last_byte_ms` and `end_ms` from the request's start (a
//! non-success reply reaches the recorder with its body, so it has no
//! `headers_ms`, and that body is the one its transport kept: rig-reqwest
//! keeps the first 64 KiB as text, with a marker where it cut); `error`; and
//! `end`: `complete`, `error` or `stopped`, when
//! the caller dropped the reply first, as a stream reader does after the
//! provider's final event. Every request asks for its reply uncompressed
//! (`accept-encoding: identity`, written as sent), and nothing is
//! decompressed. A JSON body or event is JSON and other text a string, both
//! scrubbed. A binary body, and one the scrubber cannot read whole (an
//! encoding other than identity, text that does not parse, a reply that did
//! not end whole), is `{"bytes": n, "content_type": …}`; so is the reply to a
//! request whose body was kept that way, which could echo it: its body or its
//! one event, with its error text as `"[n bytes]"`. A model call repeats its
//! conversation, so a JSON request
//! body is written against the one of the last few of its handler and URL in
//! this run that it shares the most with, when that is exact: `request.after`
//! names that exchange, `request.keep` maps each field equal to its own to
//! `true` and each array that starts with its first `n` items to `n`, and
//! `body` holds the other fields and those arrays' tails. Read in file order, a
//! body rebuilds from its base's fields in their order (those in `keep` or
//! `body`), then `body`'s new ones; a base can serve several later bodies.
//!
//! ```no_run
//! use rig_cassette::traffic::TrafficLog;
//! use rig_core::http_client::DynHttpClient;
//!
//! # fn run(http: DynHttpClient) -> std::io::Result<()> {
//! let log = TrafficLog::new(std::fs::File::create("traffic.jsonl")?)?;
//! let http = log.wrap_http(http);
//! # Ok(())
//! # }
//! ```

use std::fmt;
use std::io::{self, Write};
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use futures::Stream;
use rig_core::http_client::{
    self, BoxedStream, Bytes, DynHttpClient, HeaderMap, HeaderValue, HttpClientExt, LazyBody,
    MultipartForm, Request, Response, StatusCode, StreamingResponse,
};
use rig_core::serve::Serving;
use rig_core::wasm_compat::WasmCompatSend;

mod line;
#[cfg(test)]
mod tests;

pub use line::LINE_VERSION;

/// Where exchanges go: a writer thread that scrubs each and writes it as
/// one JSON line (see the [module docs](self)). Cloning is cheap; the clones
/// share the thread. Nothing waits on it but [`flush`](Self::flush): when the
/// last clone and every client it wrapped are gone, the thread writes what is
/// queued and ends on its own, so a program that exits right after its last
/// request calls `flush` first.
#[derive(Clone)]
pub struct TrafficLog {
    shared: Arc<Shared>,
}

pub(crate) struct Shared {
    sender: Sender<Message>,
    exchanges: AtomicU64,
    /// The writer thread is gone, such as after a panic in a writer, and
    /// that was warned about.
    gone: AtomicBool,
}

impl Shared {
    fn send(&self, message: Message) -> bool {
        let sent = self.sender.send(message).is_ok();
        if !sent && !self.gone.swap(true, Ordering::Relaxed) {
            tracing::warn!("the traffic log's writer stopped; exchanges are no longer written");
        }
        sent
    }

    /// Waits until the writer wrote everything sent before.
    fn flush(&self) {
        let (done, wait) = mpsc::channel();
        if self.send(Message::Flush(done)) {
            wait.recv().ok();
        }
    }
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
            .name("rig-traffic".to_owned())
            .spawn(move || write_lines(&receiver, writer))?;
        Ok(Self {
            shared: Arc::new(Shared {
                sender,
                exchanges: AtomicU64::new(0),
                gone: AtomicBool::new(false),
            }),
        })
    }

    /// `http`, recording every exchange sent through it, as the request
    /// reaches it: a wrapper inside it that rewrites a request or sends it
    /// again is not seen. Each request asks for its reply uncompressed
    /// (`accept-encoding: identity`), whatever its client asked for.
    pub fn wrap_http(&self, http: impl HttpClientExt + 'static) -> DynHttpClient {
        recorded(DynHttpClient::new(http), Sink::Log(self.shared.clone()))
    }

    /// Blocks until every exchange that ended before the call is written,
    /// such as before the process exits while a handle is still alive.
    pub fn flush(&self) {
        self.shared.flush();
    }
}

/// Who a finished exchange is handed to.
#[derive(Clone)]
pub(crate) enum Sink {
    /// The writer thread of a [`TrafficLog`].
    Log(Arc<Shared>),
    /// A cassette recording, which scrubs and keeps the exchange at once.
    #[cfg(feature = "http")]
    Cassette(crate::http::DirectRecorder),
}

impl Sink {
    fn number(&self) -> u64 {
        match self {
            Self::Log(shared) => shared.exchanges.fetch_add(1, Ordering::Relaxed) + 1,
            #[cfg(feature = "http")]
            Self::Cassette(_) => 0,
        }
    }

    fn finished(&self, exchange: Box<Exchange>) {
        match self {
            Self::Log(shared) => {
                shared.send(Message::Exchange(exchange));
            }
            #[cfg(feature = "http")]
            Self::Cassette(recorder) => recorder.record_exchange(&exchange),
        }
    }
}

/// `inner`, every exchange through it handed to `sink`.
pub(crate) fn recorded(inner: DynHttpClient, sink: Sink) -> DynHttpClient {
    DynHttpClient::new(Recorded { inner, sink })
}

/// One exchange as it happened, secrets and all: it is scrubbed where it
/// is written.
pub(crate) struct Exchange {
    pub(crate) number: u64,
    pub(crate) started_ms: u64,
    pub(crate) serving: Option<Serving>,
    pub(crate) method: String,
    pub(crate) uri: String,
    pub(crate) request_headers: HeaderMap,
    pub(crate) request_body: RequestBody,
    pub(crate) status: Option<StatusCode>,
    pub(crate) response_headers: Option<HeaderMap>,
    /// A unary reply's body, or a non-success reply's that its error kept.
    pub(crate) response_body: Option<Bytes>,
    /// A streamed reply's chunks, each with when it came.
    pub(crate) chunks: Option<Vec<(Duration, Bytes)>>,
    pub(crate) headers_at: Option<Duration>,
    pub(crate) first_byte_at: Option<Duration>,
    pub(crate) last_byte_at: Option<Duration>,
    pub(crate) ended_at: Option<Duration>,
    pub(crate) error: Option<String>,
    pub(crate) end: End,
}

/// Whether `headers` name an event stream (`text/event-stream`) as the reply's
/// content type.
#[cfg(feature = "http")]
pub(crate) fn is_event_stream(headers: &HeaderMap) -> bool {
    headers
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .is_some_and(crate::scrub::is_sse)
}

impl Exchange {
    /// Why the reply did not end whole, as a reader would replay or analyse
    /// it; `None` when it did. A stream that failed midway or was never read,
    /// one that stopped anywhere but at a frame boundary of its framing
    /// ([`crate::scrub::stopped_at_frame_boundary`]: a provider's reader stops
    /// at its final event or record), a unary body that
    /// failed or was never read, and no reply at all are not whole; a
    /// non-success reply its transport captured with its body is.
    pub(crate) fn incomplete(&self) -> Option<&'static str> {
        if self.status.is_none() {
            return Some("got no reply");
        }
        match (&self.chunks, self.end) {
            (_, End::Complete) => None,
            (Some(_), End::Error) => Some("failed midway"),
            (Some(chunks), End::Stopped) if chunks.is_empty() => Some("was never read"),
            (Some(_), End::Stopped) => {
                let body = self.response_bytes().unwrap_or_default();
                let headers = self
                    .response_headers
                    .as_ref()
                    .into_iter()
                    .flat_map(crate::scrub::header_pairs);
                (!crate::scrub::stopped_at_frame_boundary(
                    crate::scrub::recordable_pairs(headers),
                    &body,
                ))
                .then_some("stopped before it was complete")
            }
            (None, End::Error)
                if self.response_body.is_some()
                    && self.status.is_some_and(|status| !status.is_success()) =>
            {
                None
            }
            (None, End::Error) => Some("failed while its body was read"),
            (None, End::Stopped) => Some("was never read"),
        }
    }

    /// Whether the reply is an event stream ([`is_event_stream`]).
    #[cfg(feature = "http")]
    pub(crate) fn is_event_stream(&self) -> bool {
        self.response_headers.as_ref().is_some_and(is_event_stream)
    }

    /// The reply's body as one buffer: the unary body or the chunks joined.
    pub(crate) fn response_bytes(&self) -> Option<Bytes> {
        match (&self.response_body, &self.chunks) {
            (Some(body), _) => Some(body.clone()),
            (None, Some(chunks)) => Some(
                chunks
                    .iter()
                    .flat_map(|(_, chunk)| chunk.iter().copied())
                    .collect::<Vec<u8>>()
                    .into(),
            ),
            (None, None) => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum End {
    Complete,
    Error,
    Stopped,
}

pub(crate) enum RequestBody {
    Bytes(Bytes),
    Multipart(MultipartForm),
}

/// An exchange being recorded. It is handed on when it ends, or when it is
/// dropped before, as a reply whose body was never read.
struct Recording {
    sink: Sink,
    started: Instant,
    exchange: Option<Box<Exchange>>,
    /// A cassette's created-resource ledger of the reply, fed each chunk
    /// before the caller reads it.
    #[cfg(feature = "http")]
    ledger: Option<crate::http::ledger::ReplyLedger>,
}

/// What a request is when it is handed to the transport, kept until its
/// future first runs: an exchange opens then, so a future that never runs,
/// such as the losing branch of a `select!`, records nothing.
struct Opening {
    sink: Sink,
    serving: Option<Serving>,
    method: String,
    uri: String,
    headers: HeaderMap,
    body: RequestBody,
}

impl Opening {
    fn of<B>(sink: &Sink, request: &Request<B>, body: RequestBody) -> Self {
        Self {
            sink: sink.clone(),
            serving: request.extensions().get::<Serving>().cloned(),
            method: request.method().to_string(),
            uri: request.uri().to_string(),
            headers: request.headers().clone(),
            body,
        }
    }

    /// The exchange, opened now.
    fn begin(self) -> Recording {
        #[cfg(feature = "http")]
        if let Sink::Cassette(recorder) = &self.sink {
            recorder.exchange_opened();
        }
        Recording {
            started: Instant::now(),
            exchange: Some(Box::new(Exchange {
                number: self.sink.number(),
                started_ms: rig_core::now_ms(),
                serving: self.serving,
                method: self.method,
                uri: self.uri,
                request_headers: self.headers,
                request_body: self.body,
                status: None,
                response_headers: None,
                response_body: None,
                chunks: None,
                headers_at: None,
                first_byte_at: None,
                last_byte_at: None,
                ended_at: None,
                error: None,
                end: End::Stopped,
            })),
            sink: self.sink,
            #[cfg(feature = "http")]
            ledger: None,
        }
    }
}

impl Recording {
    fn update(&mut self, change: impl FnOnce(&mut Exchange, Duration)) {
        let at = self.started.elapsed();
        if let Some(exchange) = self.exchange.as_deref_mut() {
            change(exchange, at);
        }
    }

    fn response<B>(&mut self, response: &Response<B>, streamed: bool) {
        self.update(|exchange, at| {
            exchange.status = Some(response.status());
            exchange.response_headers = Some(response.headers().clone());
            exchange.headers_at = Some(at);
            if streamed {
                exchange.chunks = Some(Vec::new());
            }
        });
        #[cfg(feature = "http")]
        if let (Sink::Cassette(recorder), Some(exchange)) = (&self.sink, &self.exchange) {
            self.ledger = recorder.reply_ledger(exchange);
        }
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
            }
            // A non-success reply's body is the response's, not the error's.
            exchange.error = Some(match error.non_success_body() {
                Some(body) => {
                    exchange.response_body = Some(Bytes::copy_from_slice(body.as_bytes()));
                    exchange.last_byte_at = Some(at);
                    match error.non_success_status() {
                        Some(status) => format!("non-success status {status}"),
                        None => "non-success reply".to_owned(),
                    }
                }
                None => error.to_string(),
            });
        });
        self.end(End::Error);
    }

    /// Records a unary reply's body, or why it could not be read, and ends
    /// the exchange.
    fn body(&mut self, body: &http_client::Result<Bytes>) {
        match body {
            Ok(bytes) => {
                self.update(|exchange, at| {
                    exchange.response_body = Some(bytes.clone());
                    exchange.last_byte_at = Some(at);
                });
                self.end(End::Complete);
            }
            Err(error) => self.failed(error),
        }
    }

    fn chunk(&mut self, chunk: &Bytes) {
        self.update(|exchange, at| {
            exchange.first_byte_at.get_or_insert(at);
            exchange.last_byte_at = Some(at);
            if let Some(chunks) = &mut exchange.chunks {
                chunks.push((at, chunk.clone()));
            }
        });
        #[cfg(feature = "http")]
        if let Some(ledger) = &mut self.ledger {
            ledger.push(chunk);
        }
    }

    /// Hands the exchange on, once.
    fn end(&mut self, end: End) {
        let at = self.started.elapsed();
        if let Some(mut exchange) = self.exchange.take() {
            exchange.ended_at = Some(at);
            exchange.end = end;
            #[cfg(feature = "http")]
            if let Some(ledger) = self.ledger.take() {
                ledger.finish(&exchange.response_bytes().unwrap_or_default());
            }
            self.sink.finished(exchange);
            #[cfg(feature = "http")]
            if let Sink::Cassette(recorder) = &self.sink {
                recorder.exchange_closed();
            }
        }
    }
}

impl Drop for Recording {
    fn drop(&mut self) {
        self.end(End::Stopped);
    }
}

/// A transport whose exchanges are recorded.
struct Recorded {
    inner: DynHttpClient,
    sink: Sink,
}

/// The reply `sent` ended with, recorded: a unary body once it is read.
async fn unary<U>(
    opening: Opening,
    sent: impl Future<Output = http_client::Result<Response<LazyBody<Bytes>>>>,
) -> http_client::Result<Response<LazyBody<U>>>
where
    U: From<Bytes> + WasmCompatSend + 'static,
{
    let mut recording = opening.begin();
    match sent.await {
        Ok(response) => {
            recording.response(&response, false);
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

/// `request` asking for its reply uncompressed, whatever its client asked
/// for: a recorded reply is read and scrubbed as the provider sent it, and
/// one that comes compressed all the same is kept by its size only. The
/// request is recorded with the header as sent.
fn uncompressed<B>(mut request: Request<B>) -> Request<B> {
    request
        .headers_mut()
        .insert("accept-encoding", HeaderValue::from_static("identity"));
    request
}

impl HttpClientExt for Recorded {
    fn send<T, U>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes> + WasmCompatSend,
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        let req = uncompressed(req.map(Into::into));
        let opening = Opening::of(&self.sink, &req, RequestBody::Bytes(req.body().clone()));
        unary(opening, self.inner.send::<Bytes, Bytes>(req))
    }

    fn send_multipart<U>(
        &self,
        req: Request<MultipartForm>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        let req = uncompressed(req);
        let opening = Opening::of(&self.sink, &req, RequestBody::Multipart(req.body().clone()));
        unary(opening, self.inner.send_multipart::<Bytes>(req))
    }

    fn send_streaming<T>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        let req = uncompressed(req.map(Into::into));
        let opening = Opening::of(&self.sink, &req, RequestBody::Bytes(req.body().clone()));
        let sent = self.inner.send_streaming(req);
        async move {
            let mut recording = opening.begin();
            match sent.await {
                Ok(response) => {
                    recording.response(&response, true);
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

/// A streamed body whose chunks are recorded as they pass.
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
            Poll::Ready(Some(Ok(chunk))) => this.recording.chunk(chunk),
            Poll::Ready(Some(Err(error))) => this.recording.failed(error),
            Poll::Ready(None) => this.recording.end(End::Complete),
            Poll::Pending => {}
        }
        polled
    }
}

/// The writer thread: one scrubbed JSON line per exchange, each in one
/// write, so a failed write is the only way a line goes missing. A failed
/// write is warned about once; after one, the next line starts on a line
/// of its own and is written against no earlier body.
fn write_lines(receiver: &Receiver<Message>, mut out: impl Write) {
    let pid = std::process::id();
    let mut bases = line::Bases::default();
    let mut warned = false;
    let mut warn = |failure: io::Error| {
        if !std::mem::replace(&mut warned, true) {
            tracing::warn!("the traffic log could not write: {failure}");
        }
    };
    let mut torn = false;
    while let Ok(message) = receiver.recv() {
        match message {
            Message::Exchange(exchange) => {
                let mut text = if torn { "\n".to_owned() } else { String::new() };
                text.push_str(&line::line(&exchange, pid, &mut bases).to_string());
                text.push('\n');
                torn = out
                    .write_all(text.as_bytes())
                    .and_then(|()| out.flush())
                    .map_err(&mut warn)
                    .is_err();
                if torn {
                    bases = line::Bases::default();
                }
            }
            Message::Flush(done) => {
                done.send(()).ok();
            }
        }
    }
}
