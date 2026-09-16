//! The one driver: the only `async` in the provider layer.
//!
//! [`call`] and [`stream`] are the whole HTTP half of every provider in this
//! crate. Everything that used to be duplicated per provider or per
//! `_observed` sibling lives here once: the status split and the non-success
//! funnel (rig#2210/#2314), transport request-id capture, the `Accept`
//! header, observation facts, the telemetry span and usage recording, and
//! the frame-triage policy table.
//!
//! **Unary is a stream of one frame.** [`call`] frames the whole reply body
//! with the same framer and drives it through the same [`Decoder`] a stream
//! would, then folds the events into the operation's response. There is no
//! `decode`, so a provider's unary and streaming replies cannot drift.
//!
//! [`Bound`] is a wire plus its socket, and the one implementation of each
//! consumer-facing trait that rig-core ships.

use bytes::Bytes;
use futures::{Stream, StreamExt};

use crate::http_client::framing::{Framing, NdjsonFramer, SseFramer};
use crate::http_client::{self, HttpClientExt};
use crate::observe::{AdapterContext, AdapterEnding, AdapterErrorBoundary, AdapterSlot};
use crate::providers::internal::wire::WireEvent;
use crate::wasm_compat::WasmCompatSend;
use crate::wire::{
    Body, Decoder, Encoded, Error, Event, Fold, Mode, Operation, Reply, Request, Response, Sink,
    Wire, WireError, WireFrame,
};

mod bound;
mod consumers;

pub use bound::{Bind, Bound};
#[cfg(feature = "audio")]
pub use consumers::HasAudioGeneration;
#[cfg(feature = "image")]
pub use consumers::HasImageGeneration;
pub use consumers::{
    CompletionProvider, HasCompletion, HasEmbedding, HasImageEmbedding, HasModelListing, HasRerank,
    HasTranscription, HasVerify, Socket,
};

/// The frame-triage policy table, in one place for every operation:
///
/// | classify             | driver action                                     |
/// |----------------------|---------------------------------------------------|
/// | [`WireEvent::Known`] | `interpret`, then yield what it pushed            |
/// | `WireEvent::Unknown` | warn (metadata only), skip semantically, and hand |
/// |                      | the raw value to the operation's passthrough      |
/// | `WireEvent::Corrupt` | an in-band `Err` item; keep consuming             |
/// | transport `Err`      | flush delivered content, one `Err`, then end      |
///
/// The same table serves a streamed reply and a buffered one: `call` stops
/// at the first `Err` item because its fold is fallible, so a corrupt frame
/// fails a unary call and surfaces in-band on a stream without the driver
/// naming the difference.
pub struct WireDriver<Op: Operation, D, F = WireFrame> {
    decoder: D,
    out: Op::Output,
    ready: Vec<Result<Op::Event, Op::Error>>,
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
                if let Some(event) = Op::unknown(value) {
                    self.out.push(Ok(event));
                }
            }
            WireEvent::Corrupt(error) => {
                if let Some(observation) = &self.observation {
                    observation.corrupt(self.frames);
                }
                self.ready.push(Err(Op::Error::json(error)));
            }
        }
        self.out.check_laws(&mut self.laws);
        self.collect();
        if self.decoder.is_finished() {
            self.done = true;
        }
    }

    /// Report a transport failure: truncation semantics — content the
    /// provider fully delivered flushes first, then the error is the last
    /// item and nothing follows it.
    pub fn fail(&mut self, error: Op::Error) {
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
    pub fn drain(&mut self) -> std::vec::Drain<'_, Result<Op::Event, Op::Error>> {
        self.ready.drain(..)
    }

    /// The next page's request, if the reply named one.
    pub fn continuation(&self) -> Option<http::Request<Body>> {
        self.decoder.continuation()
    }

    /// The reply as one document, when the decoder reassembled it.
    pub fn document(&self) -> Option<serde_json::Value> {
        self.decoder.document()
    }

    /// Project a payload's observation facts through the decoder.
    fn project(&self, payload: &[u8]) {
        if let Some(observation) = &self.observation {
            observation.project(|sink| self.decoder.project(payload, sink));
        }
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

impl<Op, D> WireDriver<Op, D>
where
    Op: Operation,
    D: Decoder<Op>,
{
    /// Project and decode one framed payload: the step every byte reply
    /// takes, whatever produced its bytes. The payload is projected whether
    /// or not it is a frame (a heartbeat still carries facts), and
    /// [`Self::push`] already no-ops once the reply is done.
    ///
    /// One payload at a time, because a streamed reply yields between them:
    /// a consumer that stops reading must not have facts recorded for the
    /// frames it never saw.
    fn absorb(&mut self, payload: Framed) {
        self.project(payload.payload());
        if let Some(frame) = payload.into_frame() {
            self.push(frame);
        }
    }
}

/// Drive an already-framed transport stream through a decoder.
///
/// The async wrapper over [`WireDriver`], and nothing else: it exists so a
/// typed transport — an AWS event stream, a gRPC stream, an in-process
/// generator — can reuse the fold without an HTTP request behind it. Every
/// byte wire in this crate goes through [`stream`] instead, which frames the
/// bytes, captures the request id and records the span on top of the same
/// fold.
pub fn run_wire_stream<D, F, S>(transport: S, decoder: D) -> crate::streaming::StreamingResult
where
    D: Decoder<crate::operation::Completion, F> + WasmCompatSend + 'static,
    F: WasmCompatSend + 'static,
    S: Stream<Item = Result<F, crate::completion::CompletionError>> + WasmCompatSend + 'static,
{
    let mut driver = WireDriver::<crate::operation::Completion, _, F>::new(decoder);
    Box::pin(async_stream::stream! {
        let mut transport = Box::pin(transport);
        while let Some(frame) = transport.next().await {
            match frame {
                Ok(frame) => driver.push(frame),
                Err(error) => driver.fail(error),
            }
            for item in driver.drain() {
                yield item;
            }
            if driver.done() {
                return;
            }
        }
        driver.finish();
        for item in driver.drain() {
            yield item;
        }
    })
}

/// One frame after [`triage_frame`]: a modeled event for `interpret`, or an
/// unknown frame's raw payload for the passthrough channel.
#[derive(Debug)]
pub enum TriagedFrame<T> {
    /// A modeled event, ready for [`Decoder::interpret`].
    Event(T),
    /// An unknown frame's raw payload. Already warned; the caller forwards it
    /// as `StreamEvent::Unknown` where the surface has a raw channel
    /// (openai-agents' raw-event precedent), and never interprets it — the
    /// semantic path skips it.
    Unknown(crate::streaming::UnknownPayload),
}

/// Triage one classified frame under the policy table on [`WireDriver`]:
/// `Known` passes through, `Unknown` is warned (structural metadata only)
/// and handed back raw for the passthrough channel, `Corrupt` is the
/// operation's JSON error.
///
/// [`WireDriver::push`]'s per-frame policy factored out for the non-stream
/// surfaces that classify frames one at a time (the websocket pre-dispatch),
/// so they share the table instead of restating it.
pub fn triage_frame<T>(
    event: WireEvent<T>,
) -> Result<TriagedFrame<T>, crate::completion::CompletionError> {
    match event {
        WireEvent::Known(event) => Ok(TriagedFrame::Event(event)),
        WireEvent::Unknown { event_type, value } => {
            // Structural metadata only — see `warn_unmodeled`. The full
            // payload survives on the `Unknown` raw passthrough channel;
            // that channel IS the opt-in for consumers who want the content.
            warn_unmodeled(&event_type, &value);
            Ok(TriagedFrame::Unknown(value))
        }
        WireEvent::Corrupt(error) => Err(crate::completion::CompletionError::JsonError(error)),
    }
}

/// Warn about an unmodeled wire payload with **structural metadata only** —
/// its kind and serialized byte size, never the payload itself. Unmodeled
/// frames and parts can carry model output or other sensitive provider
/// data, which must not leak into production WARN logs; the one redaction
/// policy lives here, used by the driver's Unknown arm and by adapters that
/// skip an unmodeled part kind. `driver_adoption.rs` scans streaming
/// modules for direct `warn!(?...)` payload captures, so bypassing this
/// helper fails CI.
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

/// Ceiling on the pages one [`call`] will follow.
///
/// Generous by orders of magnitude: a model catalog's page holds hundreds of
/// entries, so a real listing finishes in one or two requests. This exists
/// only so a cursor that keeps *changing* without advancing — a gateway
/// alternating `c1, c2, c1, …`, or minting a fresh cursor per request —
/// terminates. It bounds continuation-following only; the requests a batched
/// wire encodes up front are all sent however many there are.
const MAX_CONTINUATION_PAGES: usize = 1000;

/// Send one request and fold its whole reply into the operation's response.
///
/// `context` observes the attempt; `None` records nothing. A paged operation
/// loops on [`Decoder::continuation`].
pub async fn call<W, H>(
    wire: &W,
    http: &H,
    request: Request<W>,
    context: Option<AdapterContext>,
) -> Result<Response<W>, Error<W>>
where
    W: Wire,
    H: HttpClientExt,
{
    let span =
        <W::Op as Operation>::span(wire.name(), wire.model(), wire.telemetry(false), &request);
    let mut fold = <W::Op as Operation>::fold(&request);
    let Encoded {
        requests,
        framing,
        request_id_header,
        relaxed_content_type,
    } = wire.encode(request, Mode::Unary)?;

    let mut reply = Reply {
        provider: wire.name().to_owned(),
        raw: serde_json::Value::Null,
        provider_request_id: None,
    };

    // One document per page. A batched wire — Cohere embeds one image per
    // call — answers a single operation with several, and the last one is
    // not the reply.
    let mut documents: Vec<serde_json::Value> = Vec::new();
    let mut pending: std::collections::VecDeque<http::Request<Body>> = requests.into();
    // Replies read in this call, which is what MAX_CONTINUATION_PAGES bounds.
    let mut pages: usize = 0;
    while let Some(mut http_request) = pending.pop_front() {
        accept_header(&mut http_request, framing);
        // Two different facts, deliberately not one variable. `route` is the
        // URL that was actually sent, which is what a human debugging a 404
        // needs from `with_route`'s decoration. The observation records the
        // *template* the wire declares, so a consumer grouping attempts by
        // route gets one bucket per endpoint rather than one per base URL.
        let route = http_request.uri().path().to_owned();
        let declared = wire.route().unwrap_or(&route);
        // What was sent, kept to recognize a continuation that would re-send
        // it. `Uri` and `Method` clones are refcount-cheap.
        let sent_target = (http_request.method().clone(), http_request.uri().clone());
        let observation = context.as_ref().map(|_| AdapterSlot::default());
        let attempt = context
            .as_ref()
            .and_then(|context| context.attempt_for(&http_request, declared));
        if let Some(observation) = &observation {
            observation.install(attempt);
        }
        let mut page = WireDriver::<W::Op, _>::observed(
            // Every page this loop reads is read to its end, so the
            // decoder's EOF is the end of an answer rather than a stream
            // stopping early — which is what `Mode::Unary` tells it.
            wire.decoder(Mode::Unary),
            observation.clone(),
        );
        let sent = tracing::Instrument::instrument(
            send::<W::Op, H>(http, http_request, request_id_header, observation.as_ref()),
            span.clone(),
        )
        .await;
        let page_reply = match sent {
            Ok(page_reply) => page_reply,
            Err(error) => {
                let error = error.with_route(wire.name(), &route);
                // The reply the failure carries is still the provider's:
                // project its facts before reporting the ending.
                if let Some(body) = error.provider_response_body() {
                    page.project(body.as_bytes());
                }
                if let Some(observation) = &observation {
                    observation.fail(&error);
                }
                return Err(error);
            }
        };
        // The status said success, but an SSE framer over a body that is not
        // an event stream yields no frames at all, which would fold to a
        // contentless success. The reply the provider actually sent is the
        // error.
        if let Some(rejected) =
            wrong_content_type(&page_reply.headers, framing, relaxed_content_type)
        {
            let error = <Error<W> as WireError>::transport(rejected)
                .with_route(wire.name(), &route)
                .with_provider_status(Some(page_reply.status))
                .with_provider_request_id(page_reply.provider_request_id.clone())
                .with_response_headers(Some(page_reply.headers.clone()));
            page.project(&page_reply.body);
            if let Some(observation) = &observation {
                observation.fail(&error);
            }
            return Err(error);
        }

        // Frame the reply the way a stream is framed, and project each
        // payload rather than the body: they are the same bytes only when
        // the framing is `Whole`, and a wire whose unary reply is an event
        // stream (the Responses endpoint on an always-streaming dialect)
        // would otherwise hand every projector a document it cannot parse.
        let mut framer = Framer::new(framing);
        for payload in framer
            .push(&page_reply.body)
            .into_iter()
            .chain(framer.finish())
        {
            page.absorb(payload);
        }
        page.finish();
        let mut failure = None;
        for item in page.drain() {
            match item {
                Ok(event) => {
                    if let Err(error) = fold.absorb(event) {
                        failure = Some(error);
                        break;
                    }
                }
                Err(error) => {
                    failure = Some(error);
                    break;
                }
            }
        }
        if let Some(error) = failure {
            let error = error
                .with_route(wire.name(), &route)
                .with_provider_status(Some(page_reply.status))
                .with_provider_request_id(page_reply.provider_request_id.clone())
                .with_response_headers(Some(page_reply.headers.clone()));
            if let Some(observation) = &observation {
                observation.fail(&error);
            }
            return Err(error);
        }
        if let Some(observation) = &observation {
            observation.finish(AdapterEnding::Decoded);
        }

        // The page's own bytes when they are one document; otherwise the
        // envelope the decoder reassembled from the event stream.
        let document = serde_json::from_slice(&page_reply.body)
            .ok()
            .or_else(|| page.document())
            .unwrap_or(serde_json::Value::Null);
        reply.provider_request_id = page_reply.provider_request_id;
        crate::providers::internal::trace_json(
            crate::providers::internal::LogTarget::Completions,
            &format!("{} {} reply", wire.name(), <W::Op as Operation>::NAME),
            &document,
        );
        documents.push(document);

        pages += 1;
        // When the loop stops following a cursor the provider is still
        // naming, the catalog handed back is truncated, so say which rule
        // stopped it. Both of these are unbounded loops if left unchecked
        // (rig#2334): the listing never returns and the fold grows without
        // limit.
        //
        // Only running out of budget is a ceiling. A reply naming no
        // continuation is the provider ending the listing — the normal path,
        // and silent. Inferring the ceiling from "a cursor is still held"
        // would report one on every listing past its first page.
        if let Some(next) = page.continuation() {
            if (next.method(), next.uri()) == (&sent_target.0, &sent_target.1) {
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
            } else {
                pending.push_front(next);
            }
        }
    }

    // One page answered with one document; several answered with the
    // sequence, which is what the per-request escape hatch returned.
    reply.raw = if documents.len() > 1 {
        serde_json::Value::Array(documents)
    } else {
        documents.pop().unwrap_or(serde_json::Value::Null)
    };

    let response = fold.finish(reply)?;
    <W::Op as Operation>::record(&span, &response);
    Ok(response)
}

/// Open a streamed reply.
///
/// An encoding error surfaces here; a connect error is the stream's only
/// item — the semantics the per-provider `async fn stream` had.
pub fn stream<W, H>(
    wire: &W,
    http: &H,
    request: Request<W>,
    context: Option<AdapterContext>,
) -> Result<impl Stream<Item = Result<Event<W>, Error<W>>> + WasmCompatSend + 'static, Error<W>>
where
    W: Wire,
    H: HttpClientExt + Clone + 'static,
{
    let span =
        <W::Op as Operation>::span(wire.name(), wire.model(), wire.telemetry(true), &request);
    let Encoded {
        requests,
        framing,
        request_id_header,
        relaxed_content_type,
    } = wire.encode(request, Mode::Streaming)?;
    // No streamed operation sends a batch: a batch exists for providers
    // that take one item per request, and those are all unary.
    let [http_request] = <[_; 1]>::try_from(requests).map_err(|requests| {
        <Error<W> as WireError>::decode(format!(
            "a streamed reply takes exactly one request, not {}",
            requests.len()
        ))
    })?;
    let mut http_request = http_request;
    accept_header(&mut http_request, framing);
    let http_request = byte_request(http_request)?;

    let http = http.clone();
    let observation = context.as_ref().map(|_| AdapterSlot::default());
    let mut driver =
        WireDriver::<W::Op, _>::observed(wire.decoder(Mode::Streaming), observation.clone());
    let recording = span.clone();
    // Read here rather than inside the stream: `wire` is borrowed, and the
    // generated stream outlives this call.
    let declared_route = wire.route().map(str::to_owned);

    let frames = async_stream::stream! {
        // The attempt is announced on the first poll, not here: a stream
        // that is built and dropped without being polled never reaches the
        // transport, and an observation log that already carried its
        // `Started` would claim a send that never happened. Only the
        // encoding above is eager, because `stream` returns a `Result` and
        // an encode failure is not the stream's problem to deliver.
        if let Some(observation) = &observation {
            // The declared template, not the concrete path — see `call`.
            let path = http_request.uri().path().to_owned();
            let declared = declared_route.as_deref().unwrap_or(&path);
            observation.install(
                context
                    .as_ref()
                    .and_then(|context| context.attempt_for(&http_request, declared)),
            );
        }
        let response = match http.send_streaming(http_request).await {
            // The bundled transports reject a non-success reply before it
            // gets here; a custom `HttpClientExt` may hand it back as a
            // response. Either way the server answered, and its answer —
            // status, headers, body — is the error, never a bare status.
            Ok(response) if response.status() != http::StatusCode::OK => {
                Err(reject_response(response).await)
            }
            Ok(response) => {
                match wrong_content_type(response.headers(), framing, relaxed_content_type) {
                    Some(error) => Err(error),
                    None => Ok(response),
                }
            }
            other => other,
        };
        let response = match response {
            Ok(response) => response,
            Err(error) => {
                if let Some(observation) = &observation {
                    observation.error_boundary(AdapterErrorBoundary::from_http(&error));
                    if let Some(status) = error.non_success_status() {
                        observation.response_with_headers(status, error.non_success_headers());
                    }
                    if let Some(body) = error.non_success_body() {
                        driver.project(body.as_bytes());
                    }
                }
                let request_id = error
                    .non_success_headers()
                    .and_then(|headers| request_id_from(headers, request_id_header));
                driver.fail(
                    <Error<W> as WireError>::transport(error)
                        .with_provider_request_id(request_id),
                );
                for item in driver.drain() {
                    yield item;
                }
                return;
            }
        };
        if let Some(observation) = &observation {
            observation.response_with_headers(response.status(), Some(response.headers()));
        }
        let request_id = request_id_from(response.headers(), request_id_header);
        let mut body = response.into_body();
        let mut framer = Framer::new(framing);
        while let Some(chunk) = body.next().await {
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(error) => {
                    // The body stream's items are already
                    // `http_client::Result<Bytes>`, so the error IS the
                    // transport's own: boxing it into
                    // `http_client::Error::Instance` discarded the
                    // non-success status, headers and body a mid-stream
                    // failure carries, and doubled its `Display` prefix.
                    if let Some(observation) = &observation {
                        observation.error_boundary(AdapterErrorBoundary::Transport);
                    }
                    driver.fail(<Error<W> as WireError>::transport(error));
                    for item in driver.drain() {
                        yield stamped::<W>(item, &request_id, &recording);
                    }
                    return;
                }
            };
            if let Some(observation) = &observation {
                observation.bytes(&chunk);
            }
            for payload in framer.push(&chunk) {
                driver.absorb(payload);
                for item in driver.drain() {
                    yield stamped::<W>(item, &request_id, &recording);
                }
                if driver.done() {
                    return;
                }
            }
        }
        for payload in framer.finish() {
            driver.absorb(payload);
            for item in driver.drain() {
                yield stamped::<W>(item, &request_id, &recording);
            }
            if driver.done() {
                return;
            }
        }
        driver.finish();
        for item in driver.drain() {
            yield stamped::<W>(item, &request_id, &recording);
        }
    };
    Ok(tracing_futures::Instrument::instrument(frames, span))
}

/// Stamp the transport request id captured off the reply onto a terminal
/// event or a preserved provider error. An id an upstream constructor
/// already attached is never replaced: it saw the reply that carried it.
fn stamped<W: Wire>(
    item: Result<Event<W>, Error<W>>,
    request_id: &Option<String>,
    span: &tracing::Span,
) -> Result<Event<W>, Error<W>> {
    match item {
        Ok(mut event) => {
            <W::Op as Operation>::stamp_request_id(&mut event, request_id);
            <W::Op as Operation>::record_event(span, &event);
            Ok(event)
        }
        Err(error) => Err(error.with_provider_request_id(request_id.clone())),
    }
}

/// What one send learned about its reply.
struct Sent {
    status: http::StatusCode,
    headers: http::HeaderMap,
    body: Bytes,
    provider_request_id: Option<String>,
}

/// Send one unary request and split its reply, funnelling every non-success
/// shape the four transport/provider cells produce (rig#2210, rig#2314).
async fn send<Op, H>(
    http: &H,
    request: http::Request<Body>,
    request_id_header: Option<&'static str>,
    observation: Option<&AdapterSlot>,
) -> Result<Sent, Op::Error>
where
    Op: Operation,
    H: HttpClientExt,
{
    let (parts, body) = request.into_parts();
    let response = match body {
        Body::Bytes(bytes) => {
            http.send::<_, Bytes>(http::Request::from_parts(parts, bytes))
                .await
        }
        Body::Multipart(form) => {
            http.send_multipart::<Bytes>(http::Request::from_parts(parts, form))
                .await
        }
    };
    let response = match response {
        Ok(response) => response,
        // A transport that reports the non-success reply as an error: the
        // reply is the provider's, so it funnels to a preserved provider
        // response with the id read off its headers and the headers
        // themselves; a response-less failure stays a transport error.
        Err(error) => {
            if let Some(observation) = observation
                && let Some(status) = error.non_success_status()
            {
                observation.response_with_headers(status, error.non_success_headers());
            }
            let request_id = error
                .non_success_headers()
                .and_then(|headers| request_id_from(headers, request_id_header));
            return Err(Op::Error::transport(error).with_provider_request_id(request_id));
        }
    };

    // Take the reply apart before awaiting the body: the headers are then
    // owned, so preserving them onto an error costs no clone and every
    // error path below can afford them.
    let (parts, body) = response.into_parts();
    let status = parts.status;
    if let Some(observation) = observation {
        observation.response_with_headers(status, Some(&parts.headers));
    }
    let provider_request_id = request_id_from(&parts.headers, request_id_header);
    let body = body.await.map_err(Op::Error::transport)?;

    if !status.is_success() {
        return Err(
            Op::Error::http_response(status, &String::from_utf8_lossy(&body))
                .with_provider_request_id(provider_request_id)
                .with_response_headers(Some(parts.headers)),
        );
    }
    Ok(Sent {
        status,
        headers: parts.headers,
        body,
        provider_request_id,
    })
}

/// The provider's transport request id, when it names such a header and the
/// reply carries a non-empty value.
fn request_id_from(headers: &http::HeaderMap, header: Option<&str>) -> Option<String> {
    crate::providers::internal::request_id_from_headers(headers, header)
}

/// Every JSON request carries `Content-Type: application/json`, including a
/// bodyless `GET`.
///
/// The deleted client layer put it in the default headers of every request
/// it built, so recorded traffic pins it on the catalog and verification
/// GETs as well as the POSTs. It is one line here rather than a line in
/// every wire's `encode`, and a wire that needs another content type sets
/// it itself.
fn content_type(request: &mut http::Request<Body>) {
    if matches!(request.body(), Body::Bytes(_)) {
        request
            .headers_mut()
            .entry(http::header::CONTENT_TYPE)
            .or_insert(http::HeaderValue::from_static("application/json"));
    }
}

/// Add `Accept: text/event-stream` to an SSE request, without overriding an
/// `Accept` the wire set itself.
fn accept_header(request: &mut http::Request<Body>, framing: Framing) {
    content_type(request);
    if framing == Framing::Sse {
        request
            .headers_mut()
            .entry("Accept")
            .or_insert(http::HeaderValue::from_static("text/event-stream"));
    }
}

/// Whether a reply's content type is not the event stream an SSE wire asked
/// for: the framer would silently produce no frames, which reads as
/// truncation rather than as the wrong endpoint. A reply that names no
/// content type at all is accepted only by a wire that opted in.
///
/// The one predicate both paths ask, so a unary reply and a streamed one
/// cannot disagree about what the provider sent.
fn wrong_content_type(
    headers: &http::HeaderMap,
    framing: Framing,
    relaxed: bool,
) -> Option<http_client::Error> {
    if framing != Framing::Sse {
        return None;
    }
    let Some(content_type) = headers.get(&http::header::CONTENT_TYPE) else {
        return (!relaxed)
            .then(|| http_client::Error::InvalidContentType(http::HeaderValue::from_static("")));
    };
    let event_stream = content_type
        .to_str()
        .ok()
        .and_then(|value| value.parse::<mime::Mime>().ok())
        .is_some_and(|mime_type| {
            matches!(
                (mime_type.type_(), mime_type.subtype()),
                (mime::TEXT, mime::EVENT_STREAM)
            )
        });
    (!event_stream).then(|| http_client::Error::InvalidContentType(content_type.clone()))
}

/// A request whose body is bytes. Multipart replies are never streamed.
fn byte_request<E: WireError>(request: http::Request<Body>) -> Result<http::Request<Vec<u8>>, E> {
    let (parts, body) = request.into_parts();
    match body {
        Body::Bytes(bytes) => Ok(http::Request::from_parts(parts, bytes)),
        Body::Multipart(_) => Err(E::decode(
            "a multipart request cannot open a streamed reply".to_owned(),
        )),
    }
}

/// One framed payload: the bytes the observer sees, and the frame the
/// decoder sees — a whitespace-only SSE payload is a heartbeat, which is not
/// a frame on any wire.
struct Framed {
    payload: Vec<u8>,
    frame: bool,
}

impl Framed {
    fn payload(&self) -> &[u8] {
        &self.payload
    }

    fn into_frame(self) -> Option<WireFrame> {
        self.frame.then(|| match String::from_utf8(self.payload) {
            Ok(text) => WireFrame::Text(text),
            Err(error) => WireFrame::Bytes(error.into_bytes()),
        })
    }
}

/// The framer for one reply's bytes.
enum Framer {
    Sse(SseFramer),
    Ndjson(NdjsonFramer),
    Whole(Vec<u8>),
}

impl Framer {
    fn new(framing: Framing) -> Self {
        match framing {
            Framing::Sse => Self::Sse(SseFramer::new()),
            Framing::Ndjson => Self::Ndjson(NdjsonFramer::new()),
            Framing::Whole => Self::Whole(Vec::new()),
        }
    }

    fn push(&mut self, chunk: &[u8]) -> Vec<Framed> {
        match self {
            Self::Sse(framer) => framer
                .push(chunk)
                .map(|event| Framed {
                    frame: !event.data.trim().is_empty(),
                    payload: event.data.into_bytes(),
                })
                .collect(),
            Self::Ndjson(framer) => framer
                .push(chunk)
                .map(|line| Framed {
                    frame: true,
                    payload: line,
                })
                .collect(),
            Self::Whole(buffer) => {
                buffer.extend_from_slice(chunk);
                Vec::new()
            }
        }
    }

    fn finish(&mut self) -> Vec<Framed> {
        match self {
            // The grammar dispatches only on a blank line: an unterminated
            // trailing event is not a frame.
            Self::Sse(_) => Vec::new(),
            Self::Ndjson(framer) => framer
                .finish()
                .map(|line| Framed {
                    frame: true,
                    payload: line,
                })
                .into_iter()
                .collect(),
            Self::Whole(buffer) => {
                let payload = std::mem::take(buffer);
                if payload.is_empty() {
                    Vec::new()
                } else {
                    vec![Framed {
                        frame: true,
                        payload,
                    }]
                }
            }
        }
    }
}

/// Bytes of a rejected reply's body kept on the error; a reply longer than
/// this is cut there.
const REJECTED_BODY_LIMIT: usize = 1 << 20;

/// Chunks read off a rejected reply before giving up on it, so a transport
/// that keeps yielding empty chunks cannot hold the opener.
const REJECTED_CHUNK_LIMIT: usize = 4096;

/// Turn a reply the driver will not stream (any status but 200, a 204
/// included: a status is a status) into the non-success error, reading the
/// body to its end (bounded in bytes and chunks) so the provider's payload
/// and the transport's headers ride on the error.
async fn reject_response(
    response: http::Response<crate::http_client::BoxedStream>,
) -> http_client::Error {
    let status = response.status();
    let headers = response.headers().clone();
    let mut body = response.into_body();
    let mut bytes: Vec<u8> = Vec::new();
    let mut chunks = 0usize;
    while let Some(chunk) = body.next().await {
        chunks += 1;
        if let Ok(chunk) = chunk {
            let room = REJECTED_BODY_LIMIT.saturating_sub(bytes.len());
            bytes.extend_from_slice(chunk.get(..chunk.len().min(room)).unwrap_or_default());
        }
        if bytes.len() >= REJECTED_BODY_LIMIT || chunks >= REJECTED_CHUNK_LIMIT {
            break;
        }
    }
    http_client::Error::InvalidStatusCodeWithDetails {
        status,
        body: String::from_utf8_lossy(&bytes).into_owned(),
        headers,
    }
}

#[cfg(test)]
mod tests;
