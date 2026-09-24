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

use futures::StreamExt;

use crate::error::ProviderError;
use crate::http_client::{BoxedHttpClient, HttpClientExt};
use crate::observe::{AdapterEnding, AdapterSlot};
use crate::providers::internal::wire::WireEvent;
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use crate::wire::{
    Capabilities, Decoder, End, Event, Fold, Mode, Operation, Reply, Request, Response, Sink, Wire,
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

    /// The model of another wire on this transport, built from this one's.
    /// With a provider configuration in place of a wire, this is how one
    /// transport serves each of the provider's endpoints:
    /// `openai.endpoint(|openai| openai.completion(model))`.
    pub fn endpoint<V>(&self, wire: impl FnOnce(&W) -> V) -> Model<V, T>
    where
        T: Clone,
    {
        Model::new(wire(&self.wire), self.transport.clone())
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

/// One step of a reply: a streamed reply's opening, an event, or the folded
/// response of a unary call.
enum Step<W: Wire> {
    /// A streamed reply opened, with its transport request id when the
    /// transport read one. Exactly once, before the stream's events.
    Opened(Option<String>),
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
    /// response, without the telemetry span the model traits record.
    pub async fn call(&self, request: Request<W>) -> Result<Response<W>, ProviderError> {
        self.unary(request, tracing::Span::none(), |error, _, _| error)
            .await
    }

    /// [`Self::call`] with the call's span, decorating a failed page's error
    /// with its route. The span records the transport request id; the
    /// caller records the response.
    async fn unary(
        &self,
        request: Request<W>,
        span: tracing::Span,
        route_error: RouteError,
    ) -> Result<Response<W>, ProviderError> {
        let result = async {
            let steps = self.run(request, Mode::Unary, span.clone(), route_error)?;
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

    /// The driver: encode, send each request of the payload through the
    /// transport, decode its frames, and yield the events (streaming) or
    /// the folded response (unary). Encoding refusals return before any
    /// stream exists; nothing is sent until the stream is polled.
    fn run(
        &self,
        request: Request<W>,
        mode: Mode,
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
        let payload = wire.encode(request, mode)?;

        Ok(async_stream::stream! {
            let mut queue = VecDeque::from([payload]);
            // Every request's document, so batched replies keep earlier data.
            let mut documents: Vec<serde_json::Value> = Vec::new();
            let mut request_id = None;
            while let Some(payload) = queue.pop_front() {
                let sending = transport.send(payload, mode);
                let opened = match mode {
                    Mode::Unary => tracing::Instrument::instrument(sending, span.clone()).await,
                    Mode::Streaming => sending.await,
                };
                let opened = match opened {
                    Ok(opened) => opened,
                    Err(error) => {
                        yield Err(error);
                        return;
                    }
                };
                let Opened {
                    frames,
                    request_id: page_request_id,
                    status,
                    headers,
                    route,
                    body,
                    rest,
                    observation: slot,
                } = opened;
                if let Some(rest) = rest {
                    queue.push_front(rest);
                }
                let route = route.unwrap_or_default();
                let mut driver = WireDriver::<W::Op, _, W::Frame>::observed(wire.decoder(mode), slot.clone());
                let mut frames = frames;

                if mode == Mode::Streaming {
                    record_request_id(&span, page_request_id.as_deref());
                    yield Ok(Step::Opened(page_request_id.clone()));
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
                            yield stamped(item, &page_request_id, &span).map(Step::Event);
                        }
                        if driver.done() {
                            return;
                        }
                    }
                    driver.finish();
                    for item in driver.drain() {
                        yield stamped(item, &page_request_id, &span).map(Step::Event);
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
                    record_request_id(&span, request_id.as_deref());
                    yield Ok(Step::Done(response));
                }
                Err(error) => yield Err(error),
            }
        })
    }
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
        let classified = self.decoder.classify(frame);
        // A frame of analysis metadata alone does not advance observation's
        // EOF and corruption positions.
        if self.observation.is_some() && !matches!(classified, WireEvent::Metadata(_)) {
            self.frames += 1;
        }
        match classified {
            WireEvent::Known(event) | WireEvent::Metadata(event) => {
                self.decoder.interpret(event, &mut self.out)
            }
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
        self.decoder.finish(&mut self.out, End::Failed);
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
        self.decoder.finish(&mut self.out, End::Eof);
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
        WireEvent::Known(event) | WireEvent::Metadata(event) => Ok(TriagedFrame::Event(event)),
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

/// Stamp the transport request id captured off the reply onto a preserved
/// provider error. An id an upstream constructor already attached is never
/// replaced: it saw the reply that carried it.
fn stamped<E>(
    item: Result<E, ProviderError>,
    request_id: &Option<String>,
    span: &tracing::Span,
) -> Result<E, ProviderError> {
    item.map_err(|error| {
        let error = error.with_provider_request_id(request_id.clone());
        record_request_id(span, error.provider_request_id());
        error
    })
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
/// The completion stream over `wire` and `http`, observed by `context`: the
/// events as the model stamps them, before the completion fold.
pub(crate) fn stream<W, H>(
    wire: &W,
    http: &H,
    request: crate::completion::CompletionRequest,
    context: Option<crate::observe::AdapterContext>,
) -> Result<
    impl futures::Stream<Item = Result<crate::streaming::StreamEvent, ProviderError>>
    + WasmCompatSend
    + use<W, H>,
    ProviderError,
>
where
    W: Wire<Op = crate::operation::Completion, Payload = crate::wire::Encoded, Frame = WireFrame>
        + Clone,
    H: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    let mut request = request;
    if let Some(context) = context {
        request.extensions.insert(context);
    }
    Model::new(wire.clone(), http.clone()).completion_events(request)
}

#[cfg(test)]
mod tests;
