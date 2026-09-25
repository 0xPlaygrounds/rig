//! Calls a model. A [`Model`] pairs a [`Wire`] (what to send and how to read
//! the reply) with a [`Transport`] (how the payload travels). Its `call`
//! folds a whole reply and its `stream` yields a completion's events; both
//! run the one private driver, so the two modes share every step. The
//! `_observed` twins take the observation context a bus records under.
//!
//! ```no_run
//! use rig_core::completion::CompletionRequestBuilder;
//! use rig_core::driver::Model;
//! use rig_core::providers::openai::{self, OpenAI};
//!
//! # async fn example(http: rig_core::http_client::BoxedHttpClient) -> Result<(), Box<dyn std::error::Error>> {
//! let model = Model::new(OpenAI::from_env()?.responses(openai::GPT_5_2), http);
//! let response = model.call(CompletionRequestBuilder::new("Hello").build()).await?;
//! # let _ = response;
//! # Ok(())
//! # }
//! ```

use std::collections::VecDeque;
use std::future::Future;

use futures::StreamExt;

use crate::error::ProviderError;
use crate::observe::{AdapterContext, AdapterEnding, AdapterSlot};
use crate::operation::{Completion, CompletionFold};
use crate::providers::internal::wire::WireEvent;
use crate::streaming::{CompletionStream, StreamEvent};
use crate::telemetry::SpanCombinator;
use crate::wasm_compat::{WasmBoxedStream, WasmCompatSend, WasmCompatSync};
use crate::wire::{
    Decoder, Event, Fold, Mode, ObservationSink, Operation, Reply, Request, Response, Sink, Wire,
    WireFrame,
};

mod http_transport;

/// An endpoint of one provider: a wire bound to a transport.
///
/// The pair holds no invariant, so both halves are public. To share one
/// transport across models, clone it. The transport defaults to the erased
/// HTTP client, so a model on the default transport is `Model<W>`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Model<W, T = crate::http_client::BoxedHttpClient> {
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
/// requests and reads [`WireFrame`]s. A transport for another kind of wire
/// is a type local to that wire's crate.
pub trait Transport<W: Wire>: Clone + WasmCompatSend + WasmCompatSync + 'static {
    /// Prepare one payload and return the future that sends it.
    ///
    /// A payload the transport cannot send in `mode` is refused here,
    /// before anything is sent. Every failure after that, including one to
    /// open the reply, is the last item of [`Opened::frames`]. The future
    /// sends nothing until it is polled. A transport that records no
    /// observation ignores `observation`.
    fn send(
        &self,
        payload: W::Payload,
        mode: Mode,
        observation: Option<Observation>,
    ) -> Result<
        impl Future<Output = Opened<W::Payload, W::Frame>> + WasmCompatSend + 'static + use<Self, W>,
        ProviderError,
    >;
}

/// A reply the transport opened: its frames, and the facts the transport owns.
pub struct Opened<P, F> {
    /// The reply's frames in order. A transport failure is the last item.
    pub frames: WasmBoxedStream<'static, Result<F, ProviderError>>,
    /// The provider's transport request id, when the reply carried one.
    pub request_id: Option<String>,
    /// The reply's status, when the transport has one.
    pub status: Option<http::StatusCode>,
    /// The reply's headers, when the transport has them.
    pub headers: Option<http::HeaderMap>,
    /// The concrete request path, for [`Operation::with_route`].
    pub route: Option<String>,
    /// The whole reply as one JSON document, when the transport buffered it.
    pub document: Option<serde_json::Value>,
    /// What remains of a payload that carried several requests. The driver
    /// sends it next.
    pub rest: Option<P>,
}

impl<P, F> Opened<P, F> {
    /// A reply of `frames` with no transport facts.
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
            rest: None,
        }
    }

    /// A reply that failed with `error` before any frame.
    pub fn failed(error: ProviderError) -> Self
    where
        F: WasmCompatSend + 'static,
    {
        Self::new(futures::stream::once(async move { Err(error) }))
    }
}

/// The observation of one attempt: what an observing transport records
/// about the exchange it performs. Built by the driver; a transport that
/// records nothing ignores it.
pub struct Observation {
    pub(crate) context: AdapterContext,
    pub(crate) slot: AdapterSlot,
    pub(crate) project: Projector,
}

/// Reads a payload's observation facts through the decoder that
/// understands it.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub(crate) type Projector = Box<dyn Fn(&[u8], &mut dyn ObservationSink) + Send>;

/// Reads a payload's observation facts (browser wasm: `!Send` allowed).
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub(crate) type Projector = Box<dyn Fn(&[u8], &mut dyn ObservationSink)>;

impl Observation {
    /// Project a payload's facts onto the attempt.
    pub(crate) fn project(&self, payload: &[u8]) {
        self.slot.project(|sink| (self.project)(payload, sink));
    }
}

/// One step of a reply: the opened reply's transport request id, an event,
/// or the folded response of a unary call.
enum Step<W: Wire> {
    /// Exactly once, first, and only for a streamed call.
    Opened(Option<String>),
    Event(Event<W>),
    /// Exactly once, last, and only for a unary call.
    Done(Response<W>),
}

impl<W, T> Model<W, T>
where
    W: Wire,
    T: Transport<W>,
{
    /// Send `request` and fold the whole reply into the operation's
    /// response. A paged operation follows every page the reply names.
    pub fn call(
        &self,
        request: Request<W>,
    ) -> impl Future<Output = Result<Response<W>, ProviderError>> + WasmCompatSend + '_ {
        self.unary(request, None)
    }

    /// [`Self::call`], with the attempt observed under `observation`.
    pub fn call_observed(
        &self,
        request: Request<W>,
        observation: AdapterContext,
    ) -> impl Future<Output = Result<Response<W>, ProviderError>> + WasmCompatSend + '_ {
        self.unary(request, Some(observation))
    }

    async fn unary(
        &self,
        request: Request<W>,
        observation: Option<AdapterContext>,
    ) -> Result<Response<W>, ProviderError> {
        let span = self.span(&request, false);
        let result = self.fold(request, observation, &span).await;
        if let Err(error) = &result {
            record_request_id(&span, error.provider_request_id());
        }
        let response = result?;
        <W::Op as Operation>::accept(&self.wire.capabilities(), self.wire.name(), &response)?;
        Ok(response)
    }

    async fn fold(
        &self,
        request: Request<W>,
        observation: Option<AdapterContext>,
        span: &tracing::Span,
    ) -> Result<Response<W>, ProviderError> {
        let steps = self.run(request, Mode::Unary, observation, span.clone())?;
        futures::pin_mut!(steps);
        while let Some(step) = steps.next().await {
            if let Step::Done(response) = step? {
                return Ok(response);
            }
        }
        Err(ProviderError::Response(format!(
            "{} reply ended before the transport delivered it whole",
            <W::Op as Operation>::NAME
        )))
    }

    fn span(&self, request: &Request<W>, streaming: bool) -> tracing::Span {
        <W::Op as Operation>::span(
            self.wire.name(),
            self.wire.model(),
            self.wire.telemetry(streaming),
            request,
        )
    }

    /// Open the attempt for one payload: its observation, and the future
    /// that sends it.
    fn open(
        wire: &W,
        transport: &T,
        payload: W::Payload,
        mode: Mode,
        context: Option<&AdapterContext>,
    ) -> Result<
        (
            impl Future<Output = Opened<W::Payload, W::Frame>> + WasmCompatSend + 'static + use<W, T>,
            Option<AdapterSlot>,
        ),
        ProviderError,
    > {
        let slot = context.map(|_| AdapterSlot::default());
        let observation = context.zip(slot.clone()).map(|(context, slot)| {
            let decoder = wire.decoder(mode);
            Observation {
                context: context.clone(),
                slot,
                project: Box::new(move |payload: &[u8], sink: &mut dyn ObservationSink| {
                    decoder.project(payload, sink)
                }),
            }
        });
        Ok((transport.send(payload, mode, observation)?, slot))
    }

    /// The driver: encode, send each page through the transport, decode its
    /// frames, and yield the events (streaming) or the folded response
    /// (unary). Encoding and send refusals return before any stream exists.
    fn run(
        &self,
        request: Request<W>,
        mode: Mode,
        observation: Option<AdapterContext>,
        span: tracing::Span,
    ) -> Result<
        impl futures::Stream<Item = Result<Step<W>, ProviderError>>
        + WasmCompatSend
        + 'static
        + use<W, T>,
        ProviderError,
    > {
        let wire = self.wire.clone();
        let transport = self.transport.clone();
        let mut fold = <W::Op as Operation>::fold(&request);
        let mut request = request;
        <W::Op as Operation>::scope_to_wire(&mut request, &wire);
        let payload = wire.encode(request, mode)?;
        let first = Self::open(&wire, &transport, payload, mode, observation.as_ref())?;

        Ok(async_stream::stream! {
            let mut next = Some((first, None::<String>));
            let mut queue: VecDeque<(W::Payload, Option<String>)> = VecDeque::new();
            // Every page's document, so batched replies keep earlier data.
            let mut documents: Vec<serde_json::Value> = Vec::new();
            let mut request_id = None;
            // Pages read in this call, which is what MAX_CONTINUATION_PAGES bounds.
            let mut pages: usize = 0;
            loop {
                let ((sending, slot), cursor) = match next.take() {
                    Some(opening) => opening,
                    None => match queue.pop_front() {
                        Some((payload, cursor)) => {
                            match Self::open(&wire, &transport, payload, mode, observation.as_ref()) {
                                Ok(opening) => (opening, cursor),
                                Err(error) => {
                                    yield Err(error);
                                    return;
                                }
                            }
                        }
                        None => break,
                    },
                };
                let opened = match mode {
                    Mode::Unary => tracing::Instrument::instrument(sending, span.clone()).await,
                    Mode::Streaming => sending.await,
                };
                let Opened { frames, request_id: page_request_id, status, headers, route, document, rest } =
                    opened;
                if let Some(rest) = rest {
                    queue.push_front((rest, None));
                }
                let route = route.unwrap_or_default();
                let mut driver = WireDriver::<W::Op, _, W::Frame>::observed(wire.decoder(mode), slot.clone());
                let mut frames = frames;

                if mode == Mode::Streaming {
                    yield Ok(Step::Opened(page_request_id));
                    while let Some(frame) = frames.next().await {
                        match frame {
                            Ok(frame) => driver.push(frame),
                            Err(error) => driver.fail(error),
                        }
                        for item in driver.drain() {
                            yield item.map(Step::Event);
                        }
                        if driver.done() {
                            return;
                        }
                    }
                    driver.finish();
                    for item in driver.drain() {
                        yield item.map(Step::Event);
                    }
                    return;
                }

                // A unary page is read whole: EOF is a complete answer, and
                // every frame is projected even after the terminal.
                let mut failed = None;
                while let Some(frame) = frames.next().await {
                    match frame {
                        Ok(frame) => driver.push(frame),
                        Err(error) => {
                            failed = Some(error);
                            break;
                        }
                    }
                }
                if let Some(error) = failed {
                    let error = <W::Op as Operation>::with_route(error, wire.name(), &route);
                    if let Some(slot) = &slot {
                        slot.fail(&error);
                    }
                    yield Err(error);
                    return;
                }
                driver.finish();
                let mut failure = None;
                for item in driver.drain() {
                    let absorbed = item.and_then(|event| fold.absorb(event));
                    if let Err(error) = absorbed {
                        failure = Some(error);
                        break;
                    }
                }
                if let Some(error) = failure {
                    let error = <W::Op as Operation>::with_route(error, wire.name(), &route)
                        .with_provider_status(status)
                        .with_provider_request_id(page_request_id.clone())
                        .with_response_headers(headers);
                    if let Some(slot) = &slot {
                        slot.fail(&error);
                    }
                    yield Err(error);
                    return;
                }
                if let Some(slot) = &slot {
                    slot.finish(AdapterEnding::Decoded);
                }

                // The page's own bytes when they are one document; otherwise
                // the envelope the decoder reassembled from its frames.
                let document = document
                    .or_else(|| driver.document())
                    .unwrap_or(serde_json::Value::Null);
                request_id = page_request_id;
                crate::providers::internal::trace_json(
                    crate::providers::internal::LogTarget::Completions,
                    &format!("{} {} reply", wire.name(), <W::Op as Operation>::NAME),
                    &document,
                );
                documents.push(document);

                pages += 1;
                // Warn only when an offered page is refused: normal exhaustion
                // is not truncation, while repeated or cycling cursors need a bound.
                if let Some(next_cursor) = driver.cursor() {
                    if cursor.as_deref() == Some(next_cursor.as_str()) {
                        // The next request would be identical to the one just
                        // answered, so the page would repeat forever.
                        tracing::warn!(
                            provider = wire.name(),
                            operation = <W::Op as Operation>::NAME,
                            pages,
                            "listing repeated its pagination cursor; returning the pages fetched \
                             so far"
                        );
                    } else if pages >= MAX_CONTINUATION_PAGES {
                        tracing::warn!(
                            provider = wire.name(),
                            operation = <W::Op as Operation>::NAME,
                            pages,
                            "listing hit its page ceiling with a cursor still advancing; returning \
                             the pages fetched so far"
                        );
                    } else if let Ok(payload) = wire.page(&next_cursor) {
                        queue.push_front((payload, Some(next_cursor)));
                    }
                }
            }

            // One page answered with one document; several answered with
            // the sequence.
            let raw = if documents.len() > 1 {
                serde_json::Value::Array(documents)
            } else {
                documents.pop().unwrap_or(serde_json::Value::Null)
            };
            let reply = Reply {
                provider: wire.name().to_owned(),
                raw,
                provider_request_id: request_id.clone(),
            };
            match fold.finish(reply) {
                Ok(response) => {
                    <W::Op as Operation>::record(&span, &response);
                    record_request_id(&span, request_id.as_deref());
                    yield Ok(Step::Done(response));
                }
                Err(error) => yield Err(error),
            }
        })
    }
}

impl<W, T> Model<W, T>
where
    W: Wire<Op = Completion>,
    T: Transport<W>,
{
    /// Open a streamed completion. Encoding errors, and requests the
    /// transport cannot stream, return here; every later failure arrives
    /// in-band. Nothing is sent until the stream is first polled.
    pub fn stream(
        &self,
        request: crate::completion::CompletionRequest,
    ) -> Result<CompletionStream, ProviderError> {
        self.streamed(request, None)
    }

    /// [`Self::stream`], with the attempt observed under `observation`.
    pub fn stream_observed(
        &self,
        request: crate::completion::CompletionRequest,
        observation: AdapterContext,
    ) -> Result<CompletionStream, ProviderError> {
        self.streamed(request, Some(observation))
    }

    fn streamed(
        &self,
        request: crate::completion::CompletionRequest,
        observation: Option<AdapterContext>,
    ) -> Result<CompletionStream, ProviderError> {
        let span = self.span(&request, true);
        let model = request.model.clone();
        let issuer = self
            .wire
            .reasoning_issuer(model.as_deref().or(self.wire.model()))
            .map(str::to_owned);
        let steps = self.run(request, Mode::Streaming, observation, span.clone())?;
        // The transport request id read off the reply's headers is stamped
        // onto the terminal record and onto errors; an id an upstream
        // constructor already attached wins, since it saw the reply.
        let mut request_id: Option<String> = None;
        let recorder = span.clone();
        let events = tracing_futures::Instrument::instrument(steps, span).filter_map(move |step| {
            futures::future::ready(match step {
                Ok(Step::Opened(id)) => {
                    record_request_id(&recorder, id.as_deref());
                    request_id = id;
                    None
                }
                Ok(Step::Event(mut event)) => {
                    if let StreamEvent::Final(terminal) = &mut event {
                        if terminal.provider_request_id.is_none() {
                            terminal.provider_request_id = request_id.clone();
                        }
                        recorder.record_response(
                            terminal
                                .response_id
                                .as_deref()
                                .or(terminal.message_id.as_deref()),
                            terminal.model.as_deref(),
                            &terminal.usage,
                        );
                    }
                    Some(Ok::<StreamEvent, _>(event))
                }
                Ok(Step::Done(_)) => None,
                Err(error) => {
                    let error = error.with_provider_request_id(request_id.clone());
                    record_request_id(&recorder, error.provider_request_id());
                    Some(Err(crate::error::ErrorReport::from(&error)))
                }
            })
        });
        let fold = CompletionFold::opened(self.wire.name(), issuer);
        Ok(CompletionStream::opened(fold, Box::pin(events)))
    }
}

/// Drives classified frames through an operation decoder. Known frames are
/// interpreted; unknown frames produce metadata-only warnings and optional raw
/// passthrough events. Corrupt frames yield errors without stopping consumption.
/// Transport failure flushes delivered content before one final error.
/// [`Model::call`] fails on the first error; streams expose errors in-band.
pub struct WireDriver<Op: Operation, D, F = WireFrame> {
    decoder: D,
    out: Op::Output,
    ready: Vec<Result<Op::Event, ProviderError>>,
    /// Frames counted for observation's EOF/corruption positions.
    frames: usize,
    observation: Option<AdapterSlot>,
    done: bool,
    laws: <Op::Output as Sink<Op>>::Laws,
    frame: std::marker::PhantomData<fn(F)>,
}

impl<Op, D, F> WireDriver<Op, D, F>
where
    Op: Operation,
    D: Decoder<Op, F>,
{
    /// A driver over one reply, without observation.
    pub fn new(decoder: D) -> Self {
        Self::observed(decoder, None)
    }

    pub(crate) fn observed(decoder: D, observation: Option<AdapterSlot>) -> Self {
        Self {
            decoder,
            out: Op::Output::default(),
            ready: Vec::new(),
            frames: 0,
            observation,
            done: false,
            laws: Default::default(),
            frame: std::marker::PhantomData,
        }
    }

    /// Whether the provider's genuine terminal already arrived: the driver
    /// stops consuming, and never runs the EOF flush.
    pub fn done(&self) -> bool {
        self.done
    }

    /// Feed one frame.
    pub fn push(&mut self, frame: F) {
        if self.done {
            return;
        }
        let analysis_only = self.observation.is_some() && self.decoder.is_analysis_only(&frame);
        let classified = self.decoder.classify(frame);
        // Never exempt a corrupt frame, even when the provider's metadata
        // predicate accepts its shape.
        let corrupt = matches!(classified, WireEvent::Corrupt(_));
        if (corrupt || !analysis_only) && self.observation.is_some() {
            self.frames += 1;
        }
        match classified {
            WireEvent::Known(event) => self.decoder.interpret(event, &mut self.out),
            // Skipped semantically, but surfaced verbatim where the
            // operation has a raw passthrough channel; aggregation never
            // folds it into the answer.
            WireEvent::Unknown { event_type, value } => {
                warn_unmodeled(&event_type, &value);
                self.out.unknown(value);
            }
            WireEvent::Corrupt(error) => {
                if let Some(observation) = &self.observation {
                    observation.corrupt(self.frames);
                }
                self.ready.push(Err(ProviderError::Json(error)));
            }
        }
        self.out.check_laws(&mut self.laws);
        self.collect();
        if self.decoder.is_finished() {
            self.done = true;
        }
    }

    /// Flushes delivered content before a final transport error.
    /// Does nothing after termination.
    pub fn fail(&mut self, error: ProviderError) {
        if self.done {
            return;
        }
        if let Some(observation) = &self.observation {
            observation.fail(&error);
        }
        self.decoder.flush_before_terminal_error(&mut self.out);
        self.collect();
        self.ready.push(Err(error));
        self.done = true;
    }

    /// End of reply: flush what the decoder still holds. Never runs after a
    /// transport error or after a terminal.
    pub fn finish(&mut self) {
        if self.done {
            return;
        }
        if let Some(observation) = &self.observation {
            observation.transport_eof(self.frames);
        }
        self.decoder.finish(&mut self.out);
        self.out.check_laws(&mut self.laws);
        self.collect();
        if let Some(observation) = &self.observation {
            observation.eof(self.frames);
        }
        self.done = true;
    }

    /// Take the items the pushed frames produced.
    pub fn drain(&mut self) -> std::vec::Drain<'_, Result<Op::Event, ProviderError>> {
        self.ready.drain(..)
    }

    /// The next page's cursor, if the reply named one.
    pub fn cursor(&self) -> Option<String> {
        self.decoder.cursor()
    }

    /// The reply as one document, when the decoder reassembled it.
    pub fn document(&self) -> Option<serde_json::Value> {
        self.decoder.document()
    }

    /// Move one step's output into the ready queue, reporting a terminal to
    /// observation as it passes.
    fn collect(&mut self) {
        let mut terminal = false;
        for item in self.out.drain() {
            if let Some(observation) = &self.observation {
                match &item {
                    Ok(event) if Op::is_terminal(event) => {
                        observation.finish(AdapterEnding::Terminal);
                    }
                    Err(error) => observation.fail(error),
                    _ => {}
                }
            }
            terminal |= matches!(&item, Ok(event) if Op::is_terminal(event));
            self.ready.push(item);
        }
        if terminal {
            self.done = true;
        }
    }
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

/// Page count after which a paged reply's cursors are ignored, preventing
/// infinite cursor cycles.
const MAX_CONTINUATION_PAGES: usize = 1000;

/// Record the transport request id on the call's span, success or failure.
fn record_request_id(span: &tracing::Span, request_id: Option<&str>) {
    if let Some(request_id) = request_id
        && !span.is_disabled()
    {
        span.record(crate::telemetry::PROVIDER_REQUEST_ID_FIELD, request_id);
    }
}

#[cfg(test)]
pub(crate) mod tests;
