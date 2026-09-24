//! Calls a model. A [`Model`] pairs a [`Wire`] (what to send and how to read
//! the reply) with a [`Transport`] (how the payload travels), and implements
//! the model trait of the wire's operation. [`Model::call`] folds a whole
//! reply; a completion model also streams. Both modes run the one private
//! driver, so they share every step.
//!
//! ```no_run
//! use rig_core::completion::CompletionModel;
//! use rig_core::driver::Model;
//! use rig_core::providers::openai::{self, OpenAI};
//!
//! # async fn example(http: rig_core::http_client::BoxedHttpClient) -> Result<(), Box<dyn std::error::Error>> {
//! let model = Model::new(OpenAI::from_env()?.responses(openai::GPT_5_2), http);
//! let response = model.completion_request("Hello").send().await?;
//! # let _ = response;
//! # Ok(())
//! # }
//! ```

use std::collections::VecDeque;
use std::future::Future;

use futures::StreamExt;

use crate::error::ProviderError;
use crate::http_client::{BoxedHttpClient, HttpClientExt};
use crate::observe::{AdapterContext, AdapterEnding, AdapterSlot};
use crate::providers::internal::wire::WireEvent;
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use crate::wire::{
    Capabilities, Decoder, Event, Fold, Mode, Operation, Reply, Request, Response, Sink, Wire,
    WireFrame,
};

mod consumers;
mod transport;

pub use transport::{Opened, Transport};

/// An endpoint of one provider: a wire bound to a transport.
///
/// The pair holds no invariant, so both halves are public. The transport
/// defaults to the erased [`BoxedHttpClient`] in type annotations;
/// [`Self::new`] infers it. To share one transport across models, clone it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Model<W, T = BoxedHttpClient> {
    /// What to send and how to read the reply.
    pub wire: W,
    /// How the payload travels.
    pub transport: T,
}

impl<W, T> Model<W, T> {
    /// Bind `wire` to `transport`.
    pub fn new(wire: W, transport: T) -> Self {
        Self { wire, transport }
    }
}

impl<W: Wire, T> Model<W, T> {
    /// The provider descriptor name.
    pub fn provider(&self) -> &str {
        self.wire.name()
    }

    /// What a runtime should account for about this model.
    pub fn capabilities(&self) -> Capabilities<W> {
        self.wire.capabilities()
    }
}

impl<W, T> Model<W, T>
where
    T: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Erase the transport, so models of many providers share one type.
    pub fn boxed(self) -> Model<W, BoxedHttpClient> {
        Model {
            wire: self.wire,
            transport: BoxedHttpClient::new(self.transport),
        }
    }
}

/// One step of a reply: an event, or the folded response of a unary call.
enum Step<W: Wire> {
    Event(Event<W>),
    /// Exactly once, last, and only for a unary call.
    Done(Response<W>),
}

/// Adds what a consumer knows about a failed page to its error: the
/// provider's name and the concrete request path.
type RouteError = fn(ProviderError, &str, &str) -> ProviderError;

impl<W, T> Model<W, T>
where
    W: Wire + Clone,
    T: Transport<Payload = W::Payload, Frame = W::Frame>,
{
    /// Send `request` and fold the whole reply into the operation's
    /// response. The call observes nothing; a completion call observed by a
    /// runtime goes through [`CompletionModel`](crate::completion::CompletionModel),
    /// whose request carries its context.
    pub async fn call(&self, request: Request<W>) -> Result<Response<W>, ProviderError> {
        self.unary(request, http::Extensions::new(), |error, _, _| error)
            .await
    }

    /// [`Self::call`] with the call's extensions, decorating a failed
    /// page's error with its route.
    async fn unary(
        &self,
        request: Request<W>,
        extensions: http::Extensions,
        route_error: RouteError,
    ) -> Result<Response<W>, ProviderError> {
        let span = self.span(&request, false);
        let result = async {
            let steps = self.run(request, Mode::Unary, extensions, span.clone(), route_error)?;
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
        .await;
        if let Err(error) = &result {
            record_request_id(&span, error.provider_request_id());
        }
        result
    }

    fn span(&self, request: &Request<W>, streaming: bool) -> tracing::Span {
        <W::Op as Operation>::span(
            self.wire.name(),
            self.wire.model(),
            self.wire.telemetry(streaming),
            request,
        )
    }

    /// The driver: encode, send each request of the payload through the
    /// transport, decode its frames, and yield the events (streaming) or
    /// the folded response (unary). Encoding and send refusals return
    /// before any stream exists; nothing is sent until the stream is polled.
    fn run(
        &self,
        request: Request<W>,
        mode: Mode,
        extensions: http::Extensions,
        span: tracing::Span,
        route_error: RouteError,
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
        scope_to_wire(&wire, &mut request);
        let payload = wire.encode(request, mode)?;
        let first = open(&transport, payload, mode, &extensions)?;

        Ok(async_stream::stream! {
            let mut next = Some(first);
            let mut queue: VecDeque<W::Payload> = VecDeque::new();
            // Every request's document, so batched replies keep earlier data.
            let mut documents: Vec<serde_json::Value> = Vec::new();
            let mut request_id = None;
            loop {
                let (sending, slot) = match next.take() {
                    Some(opening) => opening,
                    None => match queue.pop_front() {
                        Some(payload) => match open(&transport, payload, mode, &extensions) {
                            Ok(opening) => opening,
                            Err(error) => {
                                yield Err(error);
                                return;
                            }
                        },
                        None => break,
                    },
                };
                let opened = match mode {
                    Mode::Unary => tracing::Instrument::instrument(sending, span.clone()).await,
                    Mode::Streaming => sending.await,
                };
                let Opened { frames, request_id: page_request_id, status, headers, route, body, rest } =
                    opened;
                if let Some(rest) = rest {
                    queue.push_front(rest);
                }
                let route = route.unwrap_or_default();
                let mut driver = WireDriver::<W::Op, _, W::Frame>::observed(wire.decoder(mode), slot.clone());
                let mut frames = frames;

                if mode == Mode::Streaming {
                    record_request_id(&span, page_request_id.as_deref());
                    while let Some(frame) = frames.next().await {
                        match frame {
                            Ok(frame) => {
                                driver.project(frame.as_ref());
                                driver.push(frame);
                            }
                            Err(error) => {
                                // The reply the failure carries is still the
                                // provider's: project its facts first.
                                if let Some(body) = error.provider_response_body() {
                                    driver.project(body.as_bytes());
                                }
                                driver.fail(error);
                            }
                        }
                        for item in driver.drain() {
                            yield stamped::<W>(item, &page_request_id, &span).map(Step::Event);
                        }
                        if driver.done() {
                            return;
                        }
                    }
                    driver.finish();
                    for item in driver.drain() {
                        yield stamped::<W>(item, &page_request_id, &span).map(Step::Event);
                    }
                    return;
                }

                // A unary reply is read whole: EOF is a complete answer, and
                // every frame is projected, even after the terminal.
                let mut failed = None;
                while let Some(frame) = frames.next().await {
                    match frame {
                        Ok(frame) => {
                            driver.project(frame.as_ref());
                            driver.push(frame);
                        }
                        Err(error) => {
                            failed = Some(error);
                            break;
                        }
                    }
                }
                if let Some(error) = failed {
                    // The reply the failure carries is still the provider's:
                    // project its facts before reporting the ending.
                    match (error.provider_response_body(), &body) {
                        (Some(reply), _) => driver.project(reply.as_bytes()),
                        (None, Some(body)) => driver.project(body),
                        (None, None) => {}
                    }
                    let error = route_error(error, wire.name(), &route);
                    if let Some(slot) = &slot {
                        slot.fail(&error);
                    }
                    yield Err(error);
                    return;
                }
                driver.finish();
                let mut failure = None;
                for item in driver.drain() {
                    if let Err(error) = item.and_then(|event| fold.absorb(event)) {
                        failure = Some(error);
                        break;
                    }
                }
                if let Some(error) = failure {
                    let error = route_error(error, wire.name(), &route)
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

                let document = body
                    .and_then(|body| serde_json::from_slice(&body).ok())
                    .unwrap_or(serde_json::Value::Null);
                request_id = page_request_id;
                crate::providers::internal::trace_json(
                    crate::providers::internal::LogTarget::Completions,
                    &format!("{} {} reply", wire.name(), <W::Op as Operation>::NAME),
                    &document,
                );
                documents.push(document);
            }

            // One request answered with one document; several answered with
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

/// Open the attempt for one payload: its observation slot, when the call's
/// extensions carry a context, and the future that sends it.
#[allow(clippy::type_complexity)]
fn open<T: Transport>(
    transport: &T,
    payload: T::Payload,
    mode: Mode,
    extensions: &http::Extensions,
) -> Result<
    (
        impl Future<Output = Opened<T::Payload, T::Frame>> + WasmCompatSend + 'static + use<T>,
        Option<AdapterSlot>,
    ),
    ProviderError,
> {
    let mut extensions = extensions.clone();
    let slot = extensions.get::<AdapterContext>().is_some().then(|| {
        let slot = AdapterSlot::default();
        extensions.insert(slot.clone());
        slot
    });
    Ok((transport.send(payload, mode, extensions)?, slot))
}

/// Drives classified frames through an operation decoder. Known frames are
/// interpreted; unknown frames produce metadata-only warnings and optional raw
/// passthrough events. Corrupt frames yield errors without stopping consumption.
/// Transport failure flushes delivered content before one final error.
/// A unary call fails on the first error; streams expose errors in-band.
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
                if let Some(event) = Op::unknown(value) {
                    self.out.push(Ok(event));
                }
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

    /// Project a reply payload's observation facts through the decoder.
    pub(crate) fn project(&self, payload: &[u8]) {
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

/// Stamp the transport request id captured off the reply onto a terminal
/// event or a preserved provider error. An id an upstream constructor
/// already attached is never replaced: it saw the reply that carried it.
fn stamped<W: Wire>(
    item: Result<Event<W>, ProviderError>,
    request_id: &Option<String>,
    span: &tracing::Span,
) -> Result<Event<W>, ProviderError> {
    match item {
        Ok(mut event) => {
            <W::Op as Operation>::stamp_request_id(&mut event, request_id);
            <W::Op as Operation>::record_event(span, &event);
            Ok(event)
        }
        Err(error) => {
            let error = error.with_provider_request_id(request_id.clone());
            record_request_id(span, error.provider_request_id());
            Err(error)
        }
    }
}

/// Drop request content no issuer this wire accepts for the request's model
/// can interpret.
fn scope_to_wire<W: Wire>(wire: &W, request: &mut Request<W>) {
    let model = <W::Op as Operation>::request_model(request).or(wire.model());
    let issuers = wire.replay_issuers(model);
    let issuers: Vec<&str> = issuers.iter().map(String::as_str).collect();
    <W::Op as Operation>::scope_to_wire(request, &issuers);
}

/// Record the transport request id on the call's span, success or failure.
fn record_request_id(span: &tracing::Span, request_id: Option<&str>) {
    if let Some(request_id) = request_id
        && !span.is_disabled()
    {
        span.record(crate::telemetry::PROVIDER_REQUEST_ID_FIELD, request_id);
    }
}

#[cfg(test)]
/// The driver's unary path over `wire` and `http`, observed by `context`.
pub(crate) async fn call<W, H>(
    wire: &W,
    http: &H,
    request: Request<W>,
    context: Option<AdapterContext>,
) -> Result<Response<W>, ProviderError>
where
    W: Wire<Payload = crate::wire::Encoded, Frame = WireFrame> + Clone,
    H: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    let mut extensions = http::Extensions::new();
    if let Some(context) = context {
        extensions.insert(context);
    }
    Model::new(wire.clone(), http.clone())
        .unary(request, extensions, |error, _, _| error)
        .await
}

#[cfg(test)]
/// The driver's streamed path over `wire` and `http`, observed by
/// `context`: the decoder's events, before any completion fold.
pub(crate) fn stream<W, H>(
    wire: &W,
    http: &H,
    request: Request<W>,
    context: Option<AdapterContext>,
) -> Result<
    impl futures::Stream<Item = Result<Event<W>, ProviderError>> + WasmCompatSend + use<W, H>,
    ProviderError,
>
where
    W: Wire<Payload = crate::wire::Encoded, Frame = WireFrame> + Clone,
    H: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    let mut extensions = http::Extensions::new();
    if let Some(context) = context {
        extensions.insert(context);
    }
    let model = Model::new(wire.clone(), http.clone());
    let span = model.span(&request, true);
    let steps = model.run(request, Mode::Streaming, extensions, span, |error, _, _| {
        error
    })?;
    Ok(steps.filter_map(|step| {
        futures::future::ready(match step {
            Ok(Step::Event(event)) => Some(Ok(event)),
            Ok(Step::Done(_)) => None,
            Err(error) => Some(Err(error)),
        })
    }))
}

#[cfg(test)]
mod tests;
