//! Calls a model. A [`Model`] pairs a [`Wire`] (what to send and how to read
//! the reply) with a [`Transport`] (how the payload travels). Its `stream`
//! yields any operation's events as a [`Streamed`], and its `call` is that
//! stream in unary mode, drained; both run the one private driver. The
//! `_observed` twins take the observation context a bus records under.
//!
//! ```no_run
//! use rig_core::completion::CompletionRequest;
//! use rig_core::driver::Model;
//! use rig_core::providers::openai::{self, OpenAI};
//!
//! # async fn example(http: rig_core::http_client::DynHttpClient) -> Result<(), Box<dyn std::error::Error>> {
//! let model = OpenAI::from_env()?.with_http(http).responses(openai::GPT_5_2);
//! let response = model.call(CompletionRequest::new("Hello")).await?;
//! # let _ = response;
//! # Ok(())
//! # }
//! ```

use std::future::Future;
use std::task::{Context, Poll};

use futures::StreamExt;

use crate::error::{ProviderError, RigError};
use crate::observe::{AdapterContext, AdapterEnding, AdapterSlot};
use crate::streaming::Streamed;
use crate::wasm_compat::{WasmBoxedStream, WasmCompatSend, WasmCompatSync};
use crate::wire::{
    Call, Capabilities, Decoder, End, Fold, Mode, Operation, Out, Ready, Reply, Request, Response,
    Wire, WireEvent, WireFrame,
};

mod dyn_model;
mod http_transport;
mod local;

pub use dyn_model::DynModel;
pub use local::{Local, Passthrough};

/// An endpoint of one provider: a wire bound to a transport.
///
/// The pair holds no invariant, so both halves are public. To share one
/// transport across models, clone it; to send a model's requests another
/// way, replace its transport. The transport defaults to the erased HTTP
/// client, so a model on the default transport is `Model<W>`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Model<W, T = crate::http_client::DynHttpClient> {
    /// What to send and how to read the reply.
    pub wire: W,
    /// How the payload travels.
    pub transport: T,
}

impl<W, T> Model<W, T> {
    /// Pair `wire` with the `transport` that sends it.
    pub fn new(wire: W, transport: T) -> Self {
        Self { wire, transport }
    }
}

/// Sends a wire's payloads and delivers its replies' frames.
///
/// HTTP clients ([`HttpClientExt`](crate::http_client::HttpClientExt)) are
/// transports for every wire that encodes [`Encoded`](crate::wire::Encoded)
/// requests and reads [`WireFrame`]s. A local runtime, a proxy or a mock
/// is a transport for the wires it serves.
pub trait Transport<W: Wire>: Clone + WasmCompatSend + WasmCompatSync + 'static {
    /// Prepare one payload. A payload the transport cannot send in the
    /// exchange's mode is refused here, before anything is sent. Every
    /// failure after that, including one to open a reply, is the last frame
    /// of that reply. Nothing is sent until the result is first polled.
    fn send(
        &self,
        payload: W::Payload,
        exchange: Exchange,
    ) -> Result<Sending<W::Frame>, ProviderError>;
}

/// What the driver tells a transport about one send.
pub struct Exchange {
    /// How the reply is read.
    pub mode: Mode,
    /// The observation an observing transport records the send under.
    pub(crate) observation: Option<AdapterContext>,
}

/// The replies one payload opens, in order: one for most payloads, several
/// for a payload that carries a batch of requests. A transport that wraps
/// another composes it as the stream it is.
pub struct Sending<F>(WasmBoxedStream<'static, Opened<F>>);

impl<F> futures::Stream for Sending<F> {
    type Item = Opened<F>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Opened<F>>> {
        self.0.as_mut().poll_next(cx)
    }
}

impl<F: WasmCompatSend + 'static> Sending<F> {
    /// A reply that is already open.
    pub fn opened(opened: Opened<F>) -> Self {
        Self::later(std::future::ready(opened))
    }

    /// The reply `opening` opens when first polled.
    pub fn later(opening: impl Future<Output = Opened<F>> + WasmCompatSend + 'static) -> Self {
        Self(Box::pin(futures::stream::once(opening)))
    }

    /// Replies opened one after another, each once the one before it was
    /// read.
    pub fn each(
        replies: impl futures::Stream<Item = Opened<F>> + WasmCompatSend + 'static,
    ) -> Self {
        Self(Box::pin(replies))
    }
}

/// One reply a transport opened: its frames, and the facts the transport
/// owns.
pub struct Opened<F> {
    pub(crate) frames: WasmBoxedStream<'static, Result<F, ProviderError>>,
    pub(crate) request_id: Option<String>,
    pub(crate) status: Option<http::StatusCode>,
    pub(crate) headers: Option<http::HeaderMap>,
    pub(crate) route: Option<String>,
    pub(crate) document: Option<serde_json::Value>,
    pub(crate) slot: Option<AdapterSlot>,
    pub(crate) analysis_only: Option<fn(&F) -> bool>,
}

impl<F: WasmCompatSend + 'static> Opened<F> {
    /// A reply of `frames`, in order. A transport failure is the last item.
    pub fn new(
        frames: impl futures::Stream<Item = Result<F, ProviderError>> + WasmCompatSend + 'static,
    ) -> Self {
        Self {
            frames: Box::pin(frames),
            request_id: None,
            status: None,
            headers: None,
            route: None,
            document: None,
            slot: None,
            analysis_only: None,
        }
    }

    /// A reply that failed with `error` before any frame.
    pub fn failed(error: ProviderError) -> Self {
        Self::new(futures::stream::once(async move { Err(error) }))
    }

    /// The provider's transport request id, when the reply carried one.
    pub fn with_request_id(mut self, request_id: Option<String>) -> Self {
        self.request_id = request_id;
        self
    }

    /// The reply's HTTP status and headers, which enrich its unary failure.
    pub fn with_http(mut self, status: http::StatusCode, headers: http::HeaderMap) -> Self {
        self.status = Some(status);
        self.headers = Some(headers);
        self
    }

    /// The concrete request path, which a listing's failure names.
    pub fn with_route(mut self, route: impl Into<String>) -> Self {
        self.route = Some(route.into());
        self
    }

    /// The whole reply as one document: a response's `raw`.
    pub fn with_document(mut self, document: serde_json::Value) -> Self {
        self.document = Some(document);
        self
    }

    /// The same reply, its frames passed through `frames`: what a transport
    /// that wraps another uses to watch or rewrite a reply in flight.
    pub fn map_frames<S>(
        mut self,
        frames: impl FnOnce(WasmBoxedStream<'static, Result<F, ProviderError>>) -> S,
    ) -> Self
    where
        S: futures::Stream<Item = Result<F, ProviderError>> + WasmCompatSend + 'static,
    {
        self.frames = Box::pin(frames(self.frames));
        self
    }
}

impl<W, T> Model<W, T>
where
    W: Wire,
    T: Transport<W>,
{
    /// The wire's provider descriptor name (`"anthropic"`).
    pub fn name(&self) -> &str {
        self.wire.describe().name
    }

    /// The model id the wire addresses, when the operation addresses one.
    pub fn id(&self) -> Option<&str> {
        self.wire.describe().model
    }

    /// What a runtime accounts for about this model, such as an embedding
    /// model's width.
    pub fn capabilities(&self) -> Capabilities {
        self.wire.describe().capabilities
    }

    /// Send `request` and fold the whole reply into the operation's
    /// response: [`Self::stream`] in unary mode, drained. A completion takes
    /// a prompt, a conversation or a
    /// [`CompletionRequest`](crate::completion::CompletionRequest).
    pub fn call(
        &self,
        request: impl Into<Request<W>>,
    ) -> impl Future<Output = Result<Response<W>, RigError>> + WasmCompatSend + 'static {
        self.drained(request.into(), None)
    }

    /// [`Self::call`], with the attempt observed under `observation`.
    pub fn call_observed(
        &self,
        request: impl Into<Request<W>>,
        observation: AdapterContext,
    ) -> impl Future<Output = Result<Response<W>, RigError>> + WasmCompatSend + 'static {
        self.drained(request.into(), Some(observation))
    }

    /// The call opens when first polled, so its span is created under the
    /// caller's instrumented context. The error converts here, in the call's
    /// own future: a caller receives a [`RigError`], and a verb that
    /// reclassifies the provider's failure first asks for the
    /// [`ProviderError`].
    pub(crate) fn drained<E: From<ProviderError>>(
        &self,
        request: Request<W>,
        observation: Option<AdapterContext>,
    ) -> impl Future<Output = Result<Response<W>, E>> + WasmCompatSend + 'static {
        let model = self.clone();
        async move {
            Ok(model
                .open(request, Mode::Unary, observation)?
                .drain()
                .await?)
        }
    }

    /// Open a streamed reply. Encoding errors, and requests the transport
    /// cannot stream, return here; every later failure arrives in-band.
    /// Nothing is sent until the stream is first polled.
    pub fn stream(&self, request: impl Into<Request<W>>) -> Result<Streamed<W::Op>, RigError> {
        Ok(self.open(request.into(), Mode::Streaming, None)?)
    }

    /// [`Self::stream`], with the attempt observed under `observation`.
    pub fn stream_observed(
        &self,
        request: impl Into<Request<W>>,
        observation: AdapterContext,
    ) -> Result<Streamed<W::Op>, RigError> {
        Ok(self.open(request.into(), Mode::Streaming, Some(observation))?)
    }

    /// [`Self::call`], with a failure paired with the request path of the
    /// reply that failed, for an operation whose errors name their route.
    pub(crate) async fn call_routed(
        &self,
        request: Request<W>,
    ) -> Result<Response<W>, (ProviderError, String)> {
        let mut stream = self
            .open(request, Mode::Unary, None)
            .map_err(|error| (error, String::new()))?;
        while let Some(item) = futures::future::poll_fn(|cx| stream.poll_step(cx)).await {
            if let Err(error) = item {
                return Err((error, stream.route().to_owned()));
            }
        }
        let route = stream.route().to_owned();
        stream.finished().map_err(|error| (error, route))
    }

    /// The one entry to the driver: the operation's fold for the reply,
    /// the encoded payload, and the transport's replies, in `mode`.
    pub(crate) fn open(
        &self,
        request: Request<W>,
        mode: Mode,
        observation: Option<AdapterContext>,
    ) -> Result<Streamed<W::Op>, ProviderError> {
        let describe = self.wire.describe();
        let provider = describe.name.to_owned();
        let mut call = Call::new(&describe, mode);
        let fold = <W::Op as Operation>::fold(&request, &mut call);
        let span = call.span;
        let payload = self.wire.encode(request, mode)?;
        let sending = self
            .transport
            .send(payload, Exchange { mode, observation })?;
        let source = Driven::<W> {
            wire: self.wire.clone(),
            mode,
            replies: Some(sending.0),
            reading: None,
            documents: Vec::new(),
            route: String::new(),
            span: span.clone(),
            provider: provider.clone(),
        };
        Ok(Streamed::new(Box::new(source), fold, span, provider))
    }
}

/// What a reply's source did on one poll.
pub(crate) enum Progress {
    /// Items may be waiting in the ready queue.
    Pushed,
    /// The reply is complete: what the driver learned about it.
    Closed(Reply),
}

/// Where one reply's items come from: the driver over a transport, or a
/// relayed stream. It pushes items through the fold into the ready queue.
pub(crate) trait Source<Op: Operation>: WasmCompatSend {
    fn poll_into(
        &mut self,
        cx: &mut Context<'_>,
        fold: &mut Op::Fold,
        ready: &mut Ready<Op>,
    ) -> Poll<Progress>;

    /// The request path of the reply being read, when the transport named
    /// one.
    fn route(&self) -> &str {
        ""
    }
}

/// The driver: send the payload, decode each reply's frames through the
/// operation's fold, and close with what the transport reported.
struct Driven<W: Wire> {
    wire: W,
    mode: Mode,
    /// The transport's replies; `None` once the call ended early.
    replies: Option<WasmBoxedStream<'static, Opened<W::Frame>>>,
    reading: Option<Reading<W>>,
    /// Every reply's document, so a batch keeps the earlier ones.
    documents: Vec<serde_json::Value>,
    /// The request path of the latest reply.
    route: String,
    span: tracing::Span,
    provider: String,
}

/// One opened reply being decoded.
struct Reading<W: Wire> {
    frames: WasmBoxedStream<'static, Result<W::Frame, ProviderError>>,
    driver: FrameDriver<W::Decoder, W::Frame>,
    /// Items staged until they may leave: after each frame for a stream,
    /// at the end of the reply for a unary call.
    staged: Ready<W::Op>,
    status: Option<http::StatusCode>,
    headers: Option<http::HeaderMap>,
    document: Option<serde_json::Value>,
    /// Whether an error left this reply.
    failed: bool,
}

impl<W: Wire> Source<W::Op> for Driven<W> {
    fn poll_into(
        &mut self,
        cx: &mut Context<'_>,
        fold: &mut <W::Op as Operation>::Fold,
        ready: &mut Ready<W::Op>,
    ) -> Poll<Progress> {
        // A stream decodes under the call's span; a unary call sends under
        // it.
        let span = self.span.clone();
        let _decoding = (self.mode == Mode::Streaming).then(|| span.enter());
        loop {
            let Some(reading) = &mut self.reading else {
                let Some(replies) = &mut self.replies else {
                    return Poll::Ready(Progress::Closed(self.close(ready)));
                };
                let opened = match self.mode {
                    Mode::Unary => span.in_scope(|| replies.poll_next_unpin(cx)),
                    Mode::Streaming => replies.poll_next_unpin(cx),
                };
                match opened {
                    Poll::Pending => return Poll::Pending,
                    Poll::Ready(None) => {
                        self.replies = None;
                        return Poll::Ready(Progress::Closed(self.close(ready)));
                    }
                    Poll::Ready(Some(opened)) => {
                        record_request_id(&self.span, opened.request_id.as_deref());
                        ready.set_request_id(opened.request_id.clone());
                        let mut staged = Ready::default();
                        staged.set_request_id(opened.request_id.clone());
                        self.route = opened.route.unwrap_or_default();
                        self.reading = Some(Reading {
                            frames: opened.frames,
                            driver: FrameDriver::new(
                                self.wire.decoder(self.mode),
                                opened.slot,
                                opened.analysis_only,
                            ),
                            staged,
                            status: opened.status,
                            headers: opened.headers,
                            document: opened.document,
                            failed: false,
                        });
                        continue;
                    }
                }
            };
            // A unary reply is read to its end even past the terminal, so
            // every payload is observed.
            let frame = if reading.driver.done && self.mode == Mode::Streaming {
                None
            } else {
                match reading.frames.poll_next_unpin(cx) {
                    Poll::Pending => return Poll::Pending,
                    Poll::Ready(frame) => frame,
                }
            };
            // Whether the call failed with this reply; otherwise the reply
            // ended and a batch's next reply may follow.
            let failed_call = match (self.mode, frame) {
                // Each frame's items leave as it decodes; a failure is
                // in-band, and the reply ends at its terminal.
                (Mode::Streaming, Some(Ok(frame))) => {
                    reading.driver.push(frame, fold, &mut reading.staged);
                    reading.release(ready);
                    return Poll::Ready(Progress::Pushed);
                }
                (Mode::Streaming, Some(Err(error))) => {
                    reading.driver.fail(error, fold, &mut reading.staged);
                    reading.release(ready);
                    false
                }
                (Mode::Streaming, None) => {
                    reading.driver.eof(fold, &mut reading.staged);
                    reading.release(ready);
                    false
                }
                // A unary reply is read whole: EOF is a complete answer,
                // every frame is read even after the terminal, and the
                // first failure, enriched with what the transport reported,
                // fails the call.
                (Mode::Unary, Some(Ok(frame))) => {
                    reading.driver.push(frame, fold, &mut reading.staged);
                    continue;
                }
                (Mode::Unary, Some(Err(error))) => {
                    if let Some(slot) = &reading.driver.slot {
                        slot.fail(&error);
                    }
                    ready.push(Err(error));
                    true
                }
                (Mode::Unary, None) => {
                    reading.driver.eof(fold, &mut reading.staged);
                    let failed = reading.fail_first(ready);
                    if !failed {
                        reading.release(ready);
                    }
                    failed
                }
            };
            if failed_call {
                self.replies = None;
                self.reading = None;
            } else {
                self.end_reply();
            }
            return Poll::Ready(Progress::Pushed);
        }
    }

    fn route(&self) -> &str {
        &self.route
    }
}

impl<W: Wire> Driven<W> {
    /// The reply being read ended: keep its document, and stop after a
    /// streamed reply that failed.
    fn end_reply(&mut self) {
        let Some(mut reading) = self.reading.take() else {
            return;
        };
        if !reading.failed
            && let Some(slot) = &reading.driver.slot
        {
            slot.finish(AdapterEnding::Decoded);
        }
        // The reply's own bytes when they are one document; otherwise the
        // envelope the decoder reassembled from its frames.
        let document = reading
            .document
            .take()
            .or_else(|| reading.driver.decoded.take())
            .unwrap_or(serde_json::Value::Null);
        crate::providers::internal::trace_json(
            crate::providers::internal::LogTarget::Completions,
            &format!("{} reply", self.provider),
            &document,
        );
        self.documents.push(document);
        if reading.failed {
            self.replies = None;
        }
    }

    fn close(&mut self, ready: &Ready<W::Op>) -> Reply {
        // One reply answered with one document; several answered with the
        // sequence.
        let raw = if self.documents.len() > 1 {
            serde_json::Value::Array(std::mem::take(&mut self.documents))
        } else {
            self.documents.pop().unwrap_or(serde_json::Value::Null)
        };
        Reply {
            provider: self.provider.clone(),
            raw,
            provider_request_id: ready.request_id().map(str::to_owned),
        }
    }
}

impl<W: Wire> Reading<W> {
    /// Hand the staged items over to the consumer.
    fn release(&mut self, ready: &mut Ready<W::Op>) {
        while let Some(item) = self.staged.pop() {
            self.failed |= item.is_err();
            ready.push(item);
        }
    }

    /// A unary reply's first failure fails the call, enriched with what the
    /// transport reported. Returns whether one did.
    fn fail_first(&mut self, ready: &mut Ready<W::Op>) -> bool {
        while let Some(item) = self.staged.pop() {
            match item {
                Ok(event) => ready.push(Ok(event)),
                Err(error) => {
                    let error = error
                        .with_provider_status(self.status)
                        .with_provider_request_id(self.staged.request_id().map(str::to_owned))
                        .with_response_headers(self.headers.take());
                    if let Some(slot) = &self.driver.slot {
                        slot.fail(&error);
                    }
                    ready.push(Err(error));
                    return true;
                }
            }
        }
        false
    }
}

/// Drives one reply's frames through a decoder and the operation's fold.
/// Known frames are interpreted; unknown frames produce metadata-only
/// warnings and reach the fold's passthrough; corrupt frames yield errors
/// without stopping consumption. A transport failure flushes delivered
/// content before one final error.
struct FrameDriver<D, F> {
    decoder: D,
    slot: Option<AdapterSlot>,
    analysis_only: Option<fn(&F) -> bool>,
    /// Frames counted for observation's EOF and corruption positions.
    frames: usize,
    done: bool,
    /// The document the decoder reassembled, when it did.
    decoded: Option<serde_json::Value>,
}

impl<D, F> FrameDriver<D, F> {
    fn new(decoder: D, slot: Option<AdapterSlot>, analysis_only: Option<fn(&F) -> bool>) -> Self {
        Self {
            decoder,
            slot,
            analysis_only,
            frames: 0,
            done: false,
            decoded: None,
        }
    }

    fn push<Op>(&mut self, frame: F, fold: &mut Op::Fold, ready: &mut Ready<Op>)
    where
        Op: Operation,
        D: Decoder<Op, F>,
    {
        if self.done {
            return;
        }
        let analysis_only = self.slot.is_some()
            && self
                .analysis_only
                .is_some_and(|analysis_only| analysis_only(&frame));
        let classified = self.decoder.classify(frame);
        // Never exempt a corrupt frame, even when the provider's metadata
        // predicate accepts its shape.
        let corrupt = matches!(classified, WireEvent::Corrupt(_));
        if (corrupt || !analysis_only) && self.slot.is_some() {
            self.frames += 1;
        }
        let corruption = match classified {
            WireEvent::Known(event) => {
                if self.step(fold, ready, None, |decoder, out| {
                    decoder.interpret(event, out)
                }) {
                    self.done = true;
                }
                None
            }
            // Skipped semantically, but surfaced verbatim where the
            // operation has a raw passthrough channel; aggregation never
            // folds it into the answer.
            WireEvent::Unknown { event_type, value } => {
                warn_unmodeled(&event_type, &value);
                self.step(fold, ready, None, |_, out| {
                    out.fold.unknown(value, out.ready)
                });
                None
            }
            WireEvent::Corrupt(error) => {
                if let Some(slot) = &self.slot {
                    slot.corrupt(self.frames);
                }
                self.step(fold, ready, None, |_, _| {});
                Some(error)
            }
        };
        if let Some(error) = corruption {
            ready.push(Err(ProviderError::Json(error)));
        }
    }

    /// Flush delivered content before a final transport error. Does
    /// nothing after the reply ended.
    fn fail<Op>(&mut self, error: ProviderError, fold: &mut Op::Fold, ready: &mut Ready<Op>)
    where
        Op: Operation,
        D: Decoder<Op, F>,
    {
        if self.done {
            return;
        }
        if let Some(slot) = &self.slot {
            slot.fail(&error);
        }
        self.step(fold, ready, Some(End::Failed), |decoder, out| {
            decoder.end(out, End::Failed);
        });
        ready.push(Err(error));
        self.done = true;
    }

    /// End of reply: flush what the decoder still holds. Never runs after a
    /// transport error or after a terminal.
    fn eof<Op>(&mut self, fold: &mut Op::Fold, ready: &mut Ready<Op>)
    where
        Op: Operation,
        D: Decoder<Op, F>,
    {
        if self.done {
            return;
        }
        if let Some(slot) = &self.slot {
            slot.transport_eof(self.frames);
        }
        self.step(fold, ready, Some(End::Eof), |decoder, out| {
            decoder.end(out, End::Eof);
        });
        if let Some(slot) = &self.slot {
            slot.eof(self.frames);
        }
        self.done = true;
    }

    /// Run one decoder step through the fold and report what it produced.
    /// Returns whether the decoder consumed its wire's own terminal
    /// failure, which ends the reply.
    fn step<Op>(
        &mut self,
        fold: &mut Op::Fold,
        ready: &mut Ready<Op>,
        end: Option<End>,
        run: impl FnOnce(&mut D, &mut Out<'_, Op>),
    ) -> bool
    where
        Op: Operation,
        D: Decoder<Op, F>,
    {
        let before = ready.items.len();
        let mut finished = false;
        run(
            &mut self.decoder,
            &mut Out {
                fold: &mut *fold,
                ready: &mut *ready,
                finished: &mut finished,
                document: &mut self.decoded,
            },
        );
        fold.settle(ready, end.or(finished.then_some(End::Eof)));
        self.collect(ready, before);
        finished
    }

    /// Report the items one step produced to observation, and stop at the
    /// provider's terminal.
    fn collect<Op: Operation>(&mut self, ready: &Ready<Op>, before: usize) {
        let mut terminal = false;
        for item in ready.items.iter().skip(before) {
            let is_terminal = matches!(item, Ok(event) if Op::is_terminal(event));
            if let Some(slot) = &self.slot {
                match item {
                    Ok(_) if is_terminal => slot.finish(AdapterEnding::Terminal),
                    Err(error) => slot.fail(error),
                    Ok(_) => {}
                }
            }
            terminal |= is_terminal;
        }
        if terminal {
            self.done = true;
        }
    }
}

/// Drives one reply's frames through a decoder, outside any transport: what
/// a caller that reads a provider's frames itself (a websocket session, a
/// test) uses. Items pass the operation's fold as they do in a call.
pub struct WireDriver<Op: Operation, D, F = WireFrame> {
    fold: Op::Fold,
    ready: Ready<Op>,
    driver: FrameDriver<D, F>,
}

impl<Op, D, F> WireDriver<Op, D, F>
where
    Op: Operation,
    Op::Fold: Default,
    D: Decoder<Op, F>,
{
    /// A driver over one reply, without observation.
    pub fn new(decoder: D) -> Self {
        Self {
            fold: Op::Fold::default(),
            ready: Ready::default(),
            driver: FrameDriver::new(decoder, None, None),
        }
    }

    /// Whether the provider's genuine terminal already arrived: the driver
    /// stops consuming, and never runs the EOF flush.
    pub fn done(&self) -> bool {
        self.driver.done
    }

    /// Feed one frame.
    pub fn push(&mut self, frame: F) {
        self.driver.push(frame, &mut self.fold, &mut self.ready);
    }

    /// Flush delivered content before a final transport error. Does nothing
    /// after termination.
    pub fn fail(&mut self, error: ProviderError) {
        self.driver.fail(error, &mut self.fold, &mut self.ready);
    }

    /// End of reply: flush what the decoder still holds.
    pub fn finish(&mut self) {
        self.driver.eof(&mut self.fold, &mut self.ready);
    }

    /// Take the items the pushed frames produced.
    pub fn drain(&mut self) -> impl Iterator<Item = Result<Op::Event, ProviderError>> + '_ {
        std::iter::from_fn(|| self.ready.pop())
    }

    /// The reply as one document, when the decoder reassembled it.
    pub fn document(&self) -> Option<serde_json::Value> {
        self.driver.decoded.clone()
    }
}

/// Read pages until one names no cursor, names the cursor it was asked
/// with, or [`MAX_CONTINUATION_PAGES`] were read. `page` reads the page at a
/// cursor (`None` for the first) and returns it with the next cursor.
pub(crate) async fn follow_cursors<P, E, Fut>(
    provider: &str,
    operation: &str,
    mut page: impl FnMut(Option<String>) -> Fut,
) -> Result<Vec<P>, E>
where
    Fut: Future<Output = Result<(P, Option<String>), E>>,
{
    let mut pages = Vec::new();
    let mut cursor = None;
    loop {
        let (read, next) = page(cursor.clone()).await?;
        pages.push(read);
        // Warn only when an offered page is refused: normal exhaustion is
        // not truncation, while repeated or cycling cursors need a bound.
        let Some(next) = next else { break };
        if cursor.as_deref() == Some(next.as_str()) {
            // The next request would be identical to the one just answered,
            // so the page would repeat forever.
            tracing::warn!(
                provider,
                operation,
                pages = pages.len(),
                "listing repeated its pagination cursor; returning the pages fetched so far"
            );
            break;
        }
        if pages.len() >= MAX_CONTINUATION_PAGES {
            tracing::warn!(
                provider,
                operation,
                pages = pages.len(),
                "listing hit its page ceiling with a cursor still advancing; returning the pages \
                 fetched so far"
            );
            break;
        }
        cursor = Some(next);
    }
    Ok(pages)
}

/// One frame after [`triage_frame`]: a modeled event for `interpret`, or an
/// unknown frame's raw payload for the passthrough channel.
#[derive(Debug)]
pub enum TriagedFrame<T> {
    /// A modeled event, ready for [`Decoder::interpret`].
    Event(T),
    /// Unknown payload, already logged without content. Forward through a raw
    /// channel when available; do not interpret it as modeled content.
    Unknown(crate::streaming::UnknownPayload),
}

/// Returns known events or unknown payloads, warning without content for the
/// latter. Corrupt frames return a JSON error.
pub fn triage_frame<T>(event: WireEvent<T>) -> Result<TriagedFrame<T>, ProviderError> {
    match event {
        WireEvent::Known(event) => Ok(TriagedFrame::Event(event)),
        WireEvent::Unknown { event_type, value } => {
            warn_unmodeled(&event_type, &value);
            Ok(TriagedFrame::Unknown(value))
        }
        WireEvent::Corrupt(error) => Err(ProviderError::Json(error)),
    }
}

/// Logs an unmodeled payload's kind and serialized size, never its content.
/// Callers must supply a structural kind label without sensitive data.
pub fn warn_unmodeled(kind: &str, payload: &impl serde::Serialize) {
    tracing::warn!(
        kind,
        payload_bytes = unknown_payload_bytes(payload),
        "skipping unmodeled wire payload"
    );
}

/// Serialized byte size of an unknown frame's payload, for the structural
/// warn log (the log never carries the payload itself).
fn unknown_payload_bytes(value: &impl serde::Serialize) -> u64 {
    /// Counter sink: measures how many bytes serialization would write
    /// without buffering them.
    struct CountingWriter(u64);

    impl std::io::Write for CountingWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0 += buf.len() as u64;
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    let mut counter = CountingWriter(0);
    // A `Value` cannot fail to serialize; degrade to 0 rather than panic.
    let _ = serde_json::to_writer(&mut counter, value);
    counter.0
}

/// Page count after which a listing's cursors are ignored, preventing
/// infinite cursor cycles.
const MAX_CONTINUATION_PAGES: usize = 1000;

/// Record the transport request id on the call's span, success or failure.
pub(crate) fn record_request_id(span: &tracing::Span, request_id: Option<&str>) {
    if let Some(request_id) = request_id
        && !span.is_disabled()
    {
        span.record(crate::telemetry::PROVIDER_REQUEST_ID_FIELD, request_id);
    }
}

#[cfg(test)]
pub(crate) mod tests;
