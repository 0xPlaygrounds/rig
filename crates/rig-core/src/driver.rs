//! Calls a model. A [`Model`] pairs a [`Wire`] (what to send and how to read
//! the reply) with a [`Transport`] (how the payload travels). Its `stream`
//! yields any operation's events as a [`Streamed`], and its `call` is that
//! stream, finished: one decoder, one fold and one response constructor for
//! both. The `_observed` twins take the observation context a recorder
//! records under.
//!
//! ```no_run
//! use rig_core::completion::CompletionRequest;
//! use rig_core::driver::Model;
//! use rig_core::providers::openai::{self, OpenAIConfig};
//!
//! # async fn example(http: rig_core::http_client::DynHttpClient) -> Result<(), Box<dyn std::error::Error>> {
//! let model = OpenAIConfig::from_env()?.connect(http).responses(openai::GPT_5_2);
//! let response = model.call(CompletionRequest::new("Hello")).await?;
//! # let _ = response;
//! # Ok(())
//! # }
//! ```

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;

use futures::StreamExt;
use tracing::Instrument;

use crate::error::ProviderError;
use crate::observe::{AdapterContext, AdapterSlot};
use crate::streaming::{Item, Streamed};
use crate::wasm_compat::{WasmBoxedFuture, WasmBoxedStream, WasmCompatSend, WasmCompatSync};
use crate::wire::document::Reassemble;
use crate::wire::{
    Call, Capabilities, Decoder, Descriptor, Flow, Mode, Operation, Out, Request, Response, Shared,
    Wire, WireEvent,
};

mod dyn_model;
pub(crate) mod http_transport;
mod local;

pub use dyn_model::DynModel;
pub use local::{Local, Step};

/// An endpoint of one provider: a wire bound to a transport.
///
/// The pair holds no invariant, so both halves are public. To share one
/// transport across models, clone it; to send a model's requests another
/// way, replace its transport. The transport defaults to the erased HTTP
/// client, so a model on the default transport is `Model<W>`.
#[derive(Clone, Debug, PartialEq)]
pub struct Model<W, T = crate::http_client::DynHttpClient> {
    /// What to send and how to read the reply.
    pub wire: W,
    /// How the payload travels.
    pub transport: T,
    /// What preparing the last request leaves for the next.
    prepared: crate::wire::Prepared,
    /// How long a streamed reply may send no frame before it fails.
    idle_timeout: Option<Duration>,
}

/// How long a streamed reply may send no frame, from when it was sent to its
/// first frame or between two, before it fails as
/// [`ProviderError::Stalled`]: five minutes. Only a frame restarts it: any
/// event the wire reads, a provider's ping event included. Bytes that are no
/// frame, such as an event stream's comments, never do. A model that thinks
/// at length is cut only if its provider sends no event for that long.
pub const STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(300);

impl<W: Default, T: Default> Default for Model<W, T> {
    fn default() -> Self {
        Self::new(W::default(), T::default())
    }
}

impl<W, T> Model<W, T> {
    /// Pair `wire` with the `transport` that sends it. A streamed reply
    /// that sends no frame for [`STREAM_IDLE_TIMEOUT`] fails.
    pub fn new(wire: W, transport: T) -> Self {
        Self {
            wire,
            transport,
            prepared: crate::wire::Prepared::default(),
            idle_timeout: Some(STREAM_IDLE_TIMEOUT),
        }
    }

    /// This model with a streamed reply failing as
    /// [`ProviderError::Stalled`] once it sends no frame for `timeout`, from
    /// when it was sent to its first frame or between two, whatever other
    /// bytes arrive; `None`, or a timeout longer than a browser's timer
    /// takes (about 24.8 days), waits indefinitely. A unary reply is never
    /// timed: it sends nothing until it is whole.
    pub fn with_idle_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.idle_timeout = timeout;
        self
    }
}

impl<W, T> Model<W, T>
where
    W: Wire<Op = crate::operation::Completion>,
    T: Transport<W>,
{
    /// Every option of `request` this model would refuse, found by preparing
    /// and encoding it as [`Self::call`] does, with nothing sent. It lists
    /// the refusals of the model's catalog entry
    /// ([`ModelSpec::refusals`](crate::catalog::ModelSpec::refusals)) and
    /// those of the wire and its route, all of them rather than the first.
    /// The request's [`OnUnsupported`](crate::completion::OnUnsupported)
    /// policy does not change what is listed, but a request whose
    /// [`GenerationOptions`](crate::completion::GenerationOptions) are
    /// default is not checked against the catalog, as when it is sent.
    ///
    /// The request's messages are not read: its system prompt and a
    /// stand-in prompt take their place, so a check costs the same for any
    /// history. A refusal that depends on what the messages hold, such as
    /// Bedrock's of a cache checkpoint after a reasoning turn, is reported
    /// when the request is sent.
    ///
    /// # Errors
    ///
    /// [`CheckError::Unsupported`](crate::completion::CheckError::Unsupported)
    /// with every refusal, or
    /// [`CheckError::Invalid`](crate::completion::CheckError::Invalid) when the request cannot be built for another
    /// reason.
    pub fn check(
        &self,
        request: &crate::completion::CompletionRequest,
    ) -> Result<(), crate::completion::CheckError> {
        check_with(request, |request| self.dry_run(request))
    }
}

/// The answer of `run`, a dry run of `request` without its messages, for
/// [`Model::check`]. The run goes on past each refusal as under
/// [`OnUnsupported::Ignore`](crate::completion::OnUnsupported::Ignore), so
/// it meets every one; a request whose options are default stays default,
/// so it is not checked against the catalog, as when it is sent.
pub(crate) fn check_with(
    request: &crate::completion::CompletionRequest,
    run: impl FnOnce(crate::completion::CompletionRequest) -> Result<(), ProviderError>,
) -> Result<(), crate::completion::CheckError> {
    use crate::completion::{CheckError, OnUnsupported};
    let mut request = without_messages(request);
    if !request.options.is_default() {
        request.options.on_unsupported = Some(OnUnsupported::Ignore);
    }
    match crate::completion::options::collect_refusals(|| run(request)) {
        (Err(error), _) => Err(CheckError::Invalid(error)),
        (Ok(()), refusals) if refusals.is_empty() => Ok(()),
        (Ok(()), refusals) => Err(CheckError::Unsupported(refusals)),
    }
}

/// `request` with its leading system prompt and a stand-in prompt as its
/// messages, everything else as it is: what [`check_with`] dry-runs.
pub(crate) fn without_messages(
    request: &crate::completion::CompletionRequest,
) -> crate::completion::CompletionRequest {
    use crate::completion::{CompletionRequest, Message};
    let CompletionRequest {
        model,
        chat_history,
        documents,
        tools,
        temperature,
        max_tokens,
        tool_choice,
        additional_params,
        output_schema,
        record_telemetry_content,
        accept_unknown_finish_reasons,
        foreign_reasoning,
        options,
        provider_options,
        cache_points,
    } = request;
    let system = chat_history
        .first()
        .filter(|message| matches!(&***message, Message::System { .. }));
    CompletionRequest {
        model: model.clone(),
        chat_history: system
            .cloned()
            .into_iter()
            .chain([std::sync::Arc::new(Message::user("check"))])
            .collect(),
        documents: documents.clone(),
        tools: tools.clone(),
        temperature: *temperature,
        max_tokens: *max_tokens,
        tool_choice: tool_choice.clone(),
        additional_params: additional_params.clone(),
        output_schema: output_schema.clone(),
        record_telemetry_content: *record_telemetry_content,
        accept_unknown_finish_reasons: *accept_unknown_finish_reasons,
        foreign_reasoning: *foreign_reasoning,
        options: options.clone(),
        provider_options: provider_options.clone(),
        cache_points: cache_points.clone(),
    }
}

/// Sends a wire's payloads and delivers its replies' frames.
///
/// HTTP clients ([`HttpClientExt`](crate::http_client::HttpClientExt)) are
/// transports for every wire that encodes [`Encoded`](crate::wire::Encoded)
/// requests and reads [`WireFrame`](crate::wire::WireFrame)s. A local
/// runtime, a proxy or a mock is a transport for the wires it serves.
pub trait Transport<W: Wire>: Clone + WasmCompatSend + WasmCompatSync + 'static {
    /// Open the reply to one payload. Nothing is sent until the opening is
    /// first polled. A payload the transport cannot send in the exchange's
    /// mode fails the opening; a failure after the reply opened is its last
    /// frame.
    fn send(&self, payload: W::Payload, exchange: Exchange) -> Opening<W::Frame>;
}

/// What the driver tells a transport about one send.
#[derive(Clone)]
pub struct Exchange {
    /// How the reply is read.
    pub mode: Mode,
    /// The observation an observing transport records the send under.
    pub(crate) observation: Option<AdapterContext>,
    /// What the HTTP transport puts in the request's extensions, such as
    /// the [`Serving`](crate::serve::Serving) of a dispatched effect.
    pub(crate) extensions: http::Extensions,
}

/// What a caller adds to one send besides the request: the observation to
/// record it under, and the HTTP request's extensions.
#[derive(Default)]
pub(crate) struct Sending {
    pub(crate) observation: Option<AdapterContext>,
    pub(crate) extensions: http::Extensions,
}

impl Sending {
    /// A send observed under `observation`.
    pub(crate) fn observed(observation: AdapterContext) -> Self {
        Self {
            observation: Some(observation),
            extensions: http::Extensions::new(),
        }
    }
}

/// One reply, opened when first polled.
pub struct Opening<F> {
    open: WasmBoxedFuture<'static, Result<Opened<F>, ProviderError>>,
    /// The attempt the transport records while opening, which a reply that
    /// stalls before it opened ends.
    slot: Option<AdapterSlot>,
}

impl<F: WasmCompatSend + 'static> Opening<F> {
    /// The reply `open` opens.
    pub fn new(
        open: impl Future<Output = Result<Opened<F>, ProviderError>> + WasmCompatSend + 'static,
    ) -> Self {
        Self {
            open: Box::pin(open),
            slot: None,
        }
    }

    /// This opening recording its attempt in `slot`.
    pub(crate) fn with_slot(mut self, slot: Option<AdapterSlot>) -> Self {
        self.slot = slot;
        self
    }

    /// A reply that is already open.
    pub fn ready(opened: Opened<F>) -> Self {
        Self::new(std::future::ready(Ok(opened)))
    }

    /// A reply that failed to open with `error`.
    pub fn failed(error: ProviderError) -> Self {
        Self::new(std::future::ready(Err(error)))
    }
}

impl<F> Future for Opening<F> {
    type Output = Result<Opened<F>, ProviderError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.open.as_mut().poll(cx)
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

    /// The provider's transport request id, when the reply carried one. An
    /// empty id is no id.
    pub fn with_request_id(mut self, request_id: Option<String>) -> Self {
        self.request_id = crate::provider_response::reported(request_id);
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

    /// The whole reply as one document: the response's `raw`. A reply
    /// with one is not fed to the wire's reassembler.
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
    /// response: [`Self::stream`] of a unary reply, finished. A completion
    /// takes a prompt, a conversation or a
    /// [`CompletionRequest`](crate::completion::CompletionRequest).
    pub fn call(
        &self,
        request: impl Into<Request<W>>,
    ) -> impl Future<Output = Result<Response<W>, ProviderError>> + WasmCompatSend + 'static {
        self.finished(request.into(), Sending::default())
    }

    /// [`Self::call`], with the attempt observed under `observation`.
    pub fn call_observed(
        &self,
        request: impl Into<Request<W>>,
        observation: AdapterContext,
    ) -> impl Future<Output = Result<Response<W>, ProviderError>> + WasmCompatSend + 'static {
        self.finished(request.into(), Sending::observed(observation))
    }

    /// The call opens when first polled, so its span is created under the
    /// caller's instrumented context.
    fn finished(
        &self,
        request: Request<W>,
        sending: Sending,
    ) -> impl Future<Output = Result<Response<W>, ProviderError>> + WasmCompatSend + 'static {
        let model = self.clone();
        async move { model.open(request, Mode::Unary, sending)?.finish().await }
    }

    /// Open a streamed reply. Encoding errors return here; every later
    /// failure arrives in-band, as the stream's last item. Nothing is sent
    /// until the stream is first polled.
    pub fn stream(&self, request: impl Into<Request<W>>) -> Result<Streamed<W::Op>, ProviderError> {
        self.open(request.into(), Mode::Streaming, Sending::default())
    }

    /// [`Self::stream`], with the attempt observed under `observation`.
    pub fn stream_observed(
        &self,
        request: impl Into<Request<W>>,
        observation: AdapterContext,
    ) -> Result<Streamed<W::Op>, ProviderError> {
        self.open(
            request.into(),
            Mode::Streaming,
            Sending::observed(observation),
        )
    }

    /// [`Self::call`], with a failure paired with the request path of the
    /// reply that failed, for an operation whose errors name their route.
    pub(crate) async fn call_routed(
        &self,
        request: Request<W>,
    ) -> Result<Response<W>, (ProviderError, String)> {
        self.open(request, Mode::Unary, Sending::default())
            .map_err(|error| (error, String::new()))?
            .finish_routed()
            .await
    }

    /// `request` prepared and encoded as [`Self::call`] would, with nothing
    /// sent.
    pub(crate) fn dry_run(&self, request: Request<W>) -> Result<(), ProviderError> {
        let describe = Descriptor {
            prepared: Some(&self.prepared),
            ..self.wire.describe()
        };
        let request = <W::Op as Operation>::prepare(request, &describe)?;
        self.wire.encode(request, Mode::Unary)?;
        Ok(())
    }

    /// The one entry to the driver: the operation's fold for the reply,
    /// the encoded payload, and the transport's reply, in `mode`. A request
    /// the operation rejects fails here, before it is encoded.
    pub(crate) fn open(
        &self,
        request: Request<W>,
        mode: Mode,
        sending: Sending,
    ) -> Result<Streamed<W::Op>, ProviderError> {
        let describe = Descriptor {
            prepared: Some(&self.prepared),
            ..self.wire.describe()
        };
        let request = <W::Op as Operation>::prepare(request, &describe)?;
        let provider = describe.name.to_owned();
        let mut call = Call::new(&describe, mode);
        let fold = <W::Op as Operation>::fold(&request, &mut call);
        let span = call.span;
        let payload = self.wire.encode(request, mode)?;
        let Sending {
            observation,
            extensions,
        } = sending;
        let exchange = Exchange {
            mode,
            observation,
            extensions,
        };
        let opening = self.transport.send(payload, exchange);
        let shared = Arc::new(Mutex::new(Shared::new(fold)));
        // Only a streamed reply sends as it goes, so only it can stall.
        let idle = match mode {
            Mode::Streaming => self.idle_timeout,
            Mode::Unary => None,
        };
        let reading = read(
            self.wire.clone(),
            opening,
            Arc::clone(&shared),
            span.clone(),
            (mode, idle),
        );
        Ok(Streamed::new(reading, shared, span, provider))
    }
}

/// Read one reply: open it, classify each frame, decode it into the reply's
/// writer, and stop at the provider's end. The stream yields once per frame
/// read, so its items reach the consumer frame by frame. When the transport
/// reported no whole document, the wire's reassembler sees every known
/// event before the decoder decodes it, and what it rebuilds is the reply's
/// `raw` however the reply ends.
///
/// The reply's `'id` brand is the borrow of its shared state inside this
/// stream: the decoder and every part handle it holds are tied to it, and
/// none can leave.
fn read<W: Wire>(
    wire: W,
    opening: Opening<W::Frame>,
    shared: Arc<Mutex<Shared<W::Op>>>,
    span: tracing::Span,
    (mode, idle): (Mode, Option<Duration>),
) -> WasmBoxedStream<'static, ()> {
    let decoding = span.clone();
    let reading = async_stream::stream! {
        let reply: &Mutex<Shared<W::Op>> = &shared;
        let opening_slot = opening.slot.clone();
        // Timed from the send and restarted only by a frame, so opening the
        // reply and its first frame share one wait.
        let mut last_frame = web_time::Instant::now();
        // A unary call sends under the call's span; a stream reads under it.
        let opened = match mode {
            Mode::Unary => opening.instrument(span.clone()).await,
            Mode::Streaming => idle_within(idle, last_frame, opening)
                .await
                .and_then(|opened| opened),
        };
        let Opened {
            mut frames,
            request_id,
            status,
            headers,
            route,
            document,
            slot,
            analysis_only,
        } = match opened {
            Ok(opened) => opened,
            Err(error) => {
                fail(reply, opening_slot.as_ref(), error);
                return;
            }
        };
        record_request_id(&span, request_id.as_deref());
        let mut reassembler = document.is_none().then(|| wire.reassembler());
        {
            let mut state = lock(reply);
            state.request_id.clone_from(&request_id);
            state.document = document;
            state.route = route.unwrap_or_default();
        }
        let enrich = |error: ProviderError| match mode {
            // A unary reply's failure carries what the transport reported.
            Mode::Unary => error
                .with_provider_status(status)
                .with_provider_request_id(request_id.clone())
                .with_response_headers(headers.clone()),
            Mode::Streaming => error,
        };
        let mut decoder = wire.decoder();
        let mut tally = slot.as_ref().map(|slot| Tally {
            slot,
            analysis_only,
            counted: 0,
        });
        loop {
            let next = match idle_within(idle, last_frame, frames.next()).await {
                Ok(next) => {
                    last_frame = web_time::Instant::now();
                    next
                }
                Err(stalled) => {
                    record(reply, reassembler.map(|document| document.finish()));
                    fail(reply, slot.as_ref(), stalled);
                    return;
                }
            };
            let flow = match next {
                Some(Ok(frame)) => step(
                    &mut decoder,
                    reassembler.as_mut(),
                    reply,
                    frame,
                    tally.as_mut(),
                ),
                Some(Err(error)) => {
                    record(reply, reassembler.map(|document| document.finish()));
                    fail(reply, slot.as_ref(), error);
                    return;
                }
                None => eof(&mut decoder, reply, tally.as_ref()),
            };
            match flow {
                Ok(Flow::More) => yield (),
                Ok(Flow::Ended(_)) => break,
                Err(error) => {
                    record(reply, reassembler.map(|document| document.finish()));
                    fail(reply, slot.as_ref(), enrich(error));
                    return;
                }
            }
        }
        record(reply, reassembler.map(|document| document.finish()));
        if let Some(slot) = &slot {
            let usage = lock(reply)
                .end
                .as_ref()
                .and_then(W::Op::observed_usage)
                .cloned();
            slot.terminal(usage.as_ref());
        }
        {
            let state = lock(reply);
            if let Some(document) = state.raw.as_ref().or(state.document.as_ref()) {
                crate::providers::internal::trace_json(
                    crate::providers::internal::LogTarget::Completions,
                    "reply",
                    document,
                );
            }
        }
        yield ();
    };
    let mut reading: WasmBoxedStream<'static, ()> = Box::pin(reading);
    match mode {
        // A stream decodes under the call's span.
        Mode::Streaming => Box::pin(futures::stream::poll_fn(move |cx| {
            let _decoding = decoding.enter();
            reading.as_mut().poll_next(cx)
        })),
        Mode::Unary => reading,
    }
}

/// The longest wait a browser's timer takes, about 24.8 days; past it, the
/// timer fires at once. A longer idle timeout waits indefinitely.
const LONGEST_TIMER: Duration = Duration::from_millis(2_147_483_647);

/// `next` awaited until `idle` has passed since `since`, or for as long as
/// it takes without `idle`: past it, the reply stalled. Timed from `since`,
/// not from this wait, so a reader that paused between frames does not move
/// the stall; a frame already in hand still comes.
pub(crate) async fn idle_within<F: Future>(
    idle: Option<Duration>,
    since: web_time::Instant,
    next: F,
) -> Result<F::Output, ProviderError> {
    let Some(idle) = idle.filter(|idle| *idle <= LONGEST_TIMER) else {
        return Ok(next.await);
    };
    crate::wasm_compat::timeout(idle.saturating_sub(since.elapsed()), next)
        .await
        .map_err(|_| ProviderError::Stalled(idle))
}

/// A reply decoded from frames already in hand: its items, then its
/// response or the error that ended it.
pub(crate) struct Decoded<Op: Operation> {
    /// Read only by relays and tests; a websocket session needs the outcome.
    #[cfg(any(test, feature = "test-utils"))]
    pub(crate) items: Vec<Result<crate::streaming::Item<Op::Event>, ProviderError>>,
    pub(crate) outcome: Result<Op::Response, ProviderError>,
}

/// Observation's count of one observed reply's frames, for its EOF and
/// corruption positions.
pub(crate) struct Tally<'a, F> {
    slot: &'a AdapterSlot,
    /// Frames observation does not count, unless they are corrupt.
    analysis_only: Option<fn(&F) -> bool>,
    counted: usize,
}

/// What the driver does with a classified frame: a known event is decoded,
/// an unmodeled payload is warned about and delivered (aggregation never
/// folds it into the answer), and a corrupt frame is the error that ends
/// the reply.
pub(crate) fn triage<E>(event: WireEvent<E>) -> Result<Item<E>, ProviderError> {
    match event {
        WireEvent::Known(event) => Ok(Item::Event(event)),
        WireEvent::Unknown { event_type, value } => {
            warn_unmodeled(&event_type, &value);
            Ok(Item::Unknown(value))
        }
        WireEvent::Corrupt(error) => Err(ProviderError::from(error)),
    }
}

/// Classify one frame and decode it into `reply`, handing a known event to
/// `reassembler` first, and count the frame in `tally` when the reply is
/// observed.
pub(crate) fn step<'id, Op, F, D, R>(
    decoder: &mut D,
    reassembler: Option<&mut R>,
    reply: &'id Mutex<Shared<Op>>,
    frame: F,
    tally: Option<&mut Tally<'_, F>>,
) -> Result<Flow, ProviderError>
where
    Op: Operation,
    D: Decoder<'id, Op, F>,
    R: Reassemble<D::Event>,
{
    let exempt = tally
        .as_ref()
        .and_then(|tally| tally.analysis_only)
        .is_some_and(|analysis_only| analysis_only(&frame));
    let classified = decoder.classify(frame);
    if let Some(tally) = tally {
        // A corrupt frame is never exempt, whatever its shape.
        let corrupt = matches!(classified, WireEvent::Corrupt(_));
        if corrupt || !exempt {
            tally.counted += 1;
        }
        if corrupt {
            tally.slot.corrupt(tally.counted);
        }
    }
    match triage(classified)? {
        Item::Event(event) => {
            if let Some(reassembler) = reassembler {
                reassembler.absorb(&event);
            }
            decoder.decode(event, Out::new(reply))
        }
        Item::Unknown(value) => {
            Out::new(reply).unknown(value);
            Ok(Flow::More)
        }
    }
}

/// The frames ran out: a decoder that saw the provider's end ends the reply
/// here; otherwise it is [`ProviderError::Truncated`].
fn eof<'id, Op, F, D>(
    decoder: &mut D,
    reply: &'id Mutex<Shared<Op>>,
    tally: Option<&Tally<'_, F>>,
) -> Result<Flow, ProviderError>
where
    Op: Operation,
    D: Decoder<'id, Op, F>,
{
    if let Some(tally) = tally {
        tally.slot.transport_eof(tally.counted);
    }
    let step = decoder.eof(Out::new(reply));
    if !matches!(step, Ok(Flow::Ended(_)))
        && let Some(tally) = tally
    {
        tally.slot.eof(tally.counted);
    }
    match step {
        Ok(Flow::More) => Err(ProviderError::Truncated),
        step => step,
    }
}

/// Feed frames already in hand through `reassembler` and `decoder` into
/// `reply`, as [`read`] does: the reply ends at the provider's end, or the
/// decoder decides at EOF, and what the reassembler rebuilt is recorded.
pub(crate) fn feed<'id, Op, F, D, R>(
    decoder: &mut D,
    mut reassembler: Option<R>,
    reply: &'id Mutex<Shared<Op>>,
    frames: impl IntoIterator<Item = F>,
) -> Result<(), ProviderError>
where
    Op: Operation,
    D: Decoder<'id, Op, F>,
    R: Reassemble<D::Event>,
{
    let fed = feed_until_end(decoder, reassembler.as_mut(), reply, frames);
    record(reply, reassembler.map(|document| document.finish()));
    fed
}

fn feed_until_end<'id, Op, F, D, R>(
    decoder: &mut D,
    mut reassembler: Option<&mut R>,
    reply: &'id Mutex<Shared<Op>>,
    frames: impl IntoIterator<Item = F>,
) -> Result<(), ProviderError>
where
    Op: Operation,
    D: Decoder<'id, Op, F>,
    R: Reassemble<D::Event>,
{
    for frame in frames {
        if let Flow::Ended(_) = step(decoder, reassembler.as_deref_mut(), reply, frame, None)? {
            return Ok(());
        }
    }
    eof(decoder, reply, None).map(drop)
}

/// Record `document`, what a reassembler rebuilt, as the reply's `raw`. A
/// `Null` document records nothing.
pub(crate) fn record<Op: Operation>(
    reply: &Mutex<Shared<Op>>,
    document: Option<serde_json::Value>,
) {
    if let Some(document) = document.filter(|document| !document.is_null()) {
        lock(reply).raw = Some(document);
    }
}

/// Fold a fed reply as a stream does: its items, then its response, or the
/// error that ended it. `reply.raw` stands unless it is null, when the
/// decoder's record does.
pub(crate) fn settle<Op: Operation>(
    shared: Mutex<Shared<Op>>,
    fed: Result<(), ProviderError>,
    reply: crate::wire::Reply,
) -> Decoded<Op> {
    let mut shared = shared
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Err(error) = fed {
        shared.items.push_back(Err(error));
    }
    shared.document = Some(reply.raw).filter(|raw| !raw.is_null());
    shared.request_id = reply.provider_request_id;
    let mut items = Vec::new();
    while let Some(item) = shared.take() {
        let failed = item.is_err();
        items.push(item);
        if failed {
            break;
        }
    }
    let outcome = match items.last() {
        Some(Err(error)) => Err(error.clone()),
        _ => shared.conclude(&reply.provider),
    };
    #[cfg(any(test, feature = "test-utils"))]
    if let Err(error) = &outcome
        && !matches!(items.last(), Some(Err(_)))
    {
        items.push(Err(error.clone()));
    }
    Decoded {
        #[cfg(any(test, feature = "test-utils"))]
        items,
        outcome,
    }
}

/// [`decode_reply`](crate::wire::decode_reply), with `raw` the reply's
/// whole document as a transport reports it (`Null` for none, so the
/// reassembler's stands), for tests that state what the transport read.
pub(crate) fn decode_reply<W: Wire>(
    wire: &W,
    request: &Request<W>,
    mode: Mode,
    frames: impl IntoIterator<Item = W::Frame>,
    raw: serde_json::Value,
) -> Result<Response<W>, ProviderError> {
    let describe = wire.describe();
    let fold = <W::Op as Operation>::fold(request, &mut Call::new(&describe, mode));
    let shared = Mutex::new(Shared::new(fold));
    let fed = feed(
        &mut wire.decoder(),
        Some(wire.reassembler()),
        &shared,
        frames,
    );
    let reply = crate::wire::Reply {
        provider: describe.name.to_owned(),
        raw,
        provider_request_id: None,
    };
    settle(shared, fed, reply).outcome
}

/// A completion reply decoded from frames already in hand, as a handler
/// relays it: its items, then the response, or the error that ended it.
#[cfg(any(test, feature = "test-utils"))]
pub(crate) fn relay_frames<W>(
    wire: &W,
    frames: impl IntoIterator<Item = W::Frame>,
) -> crate::streaming::StreamEvents
where
    W: Wire<Op = crate::operation::Completion>,
{
    use crate::error::ErrorReport;
    use crate::streaming::Relayed;

    let provider = wire.describe().name.to_owned();
    let shared = Mutex::new(Shared::new(crate::operation::ReplyFold::relayed(
        provider.clone(),
    )));
    let fed = feed(
        &mut wire.decoder(),
        Some(wire.reassembler()),
        &shared,
        frames,
    );
    let decoded = settle(
        shared,
        fed,
        crate::wire::Reply {
            provider,
            raw: serde_json::Value::Null,
            provider_request_id: None,
        },
    );
    let origin = decoded
        .outcome
        .as_ref()
        .map(|response| response.origin.clone())
        .ok();
    let relayed: Vec<Result<Relayed, ErrorReport>> = origin
        .map(|origin| Ok(Relayed::Origin(origin)))
        .into_iter()
        .chain(decoded.items.into_iter().map(|item| match item {
            Ok(item) => Ok(Relayed::Item(item)),
            Err(error) => Err(ErrorReport::from(&error)),
        }))
        // An error that ended the reply is already its last item.
        .chain(
            decoded
                .outcome
                .ok()
                .map(|response| Ok(Relayed::Done(Arc::new(response)))),
        )
        .collect();
    Box::pin(futures::stream::iter(relayed))
}

#[cfg(test)]
impl Decoded<crate::operation::Completion> {
    /// The events the reply yielded, without unmodeled payloads.
    pub(crate) fn events(&self) -> Vec<&crate::streaming::StreamEvent> {
        self.items
            .iter()
            .filter_map(|item| match item {
                Ok(crate::streaming::Item::Event(event)) => Some(event),
                _ => None,
            })
            .collect()
    }

    /// What each part ended with, in the order the parts ended.
    pub(crate) fn ended(&self) -> Vec<crate::message::AssistantContent> {
        self.events()
            .into_iter()
            .filter_map(|event| match event {
                crate::streaming::StreamEvent::End { content, .. } => Some(content.clone()),
                _ => None,
            })
            .collect()
    }
}

/// Decode typed `events` through `decoder` as one completion reply from
/// `provider`, then EOF: a decoder's test harness, without frames.
#[cfg(test)]
macro_rules! decode_events {
    ($decoder:expr, $provider:expr, $events:expr) => {
        $crate::driver::decode_with(
            $crate::operation::ReplyFold::relayed($provider),
            $provider,
            |reply| {
                let mut decoder = $decoder;
                for event in $events {
                    if let $crate::wire::Flow::Ended(_) =
                        $crate::wire::Decoder::decode(&mut decoder, event, reply.out())?
                    {
                        return Ok(());
                    }
                }
                match $crate::wire::Decoder::eof(&mut decoder, reply.out())? {
                    $crate::wire::Flow::Ended(_) => Ok(()),
                    $crate::wire::Flow::More => Err($crate::error::ProviderError::Truncated),
                }
            },
        )
    };
}
#[cfg(test)]
pub(crate) use decode_events;

/// Decode `frames` through `decoder`, classifier included, as one
/// completion reply from `provider`: what the driver does with frames it
/// read. A reassembler, when given, sees each known event first and its
/// document is the reply's `raw`.
#[cfg(test)]
macro_rules! feed_frames {
    ($decoder:expr, $provider:expr, $frames:expr) => {
        $crate::driver::decode_with(
            $crate::operation::ReplyFold::relayed($provider),
            $provider,
            |reply| {
                let mut decoder = $decoder;
                reply.feed(
                    &mut decoder,
                    None::<$crate::wire::document::Unreassembled>,
                    $frames,
                )
            },
        )
    };
    ($decoder:expr, $reassembler:expr, $provider:expr, $frames:expr) => {
        $crate::driver::decode_with(
            $crate::operation::ReplyFold::relayed($provider),
            $provider,
            |reply| {
                let mut decoder = $decoder;
                reply.feed(&mut decoder, Some($reassembler), $frames)
            },
        )
    };
}
#[cfg(test)]
pub(crate) use feed_frames;

/// One reply's writer, for a test that drives a decoder by hand.
#[cfg(test)]
pub(crate) struct Replying<'id, Op: Operation>(&'id Mutex<Shared<Op>>);

#[cfg(test)]
impl<'id, Op: Operation> Replying<'id, Op> {
    /// A writer for the next event.
    pub(crate) fn out(&self) -> Out<'id, Op> {
        Out::new(self.0)
    }

    /// Feed frames through `reassembler` and `decoder`, as the driver does.
    pub(crate) fn feed<F, D: Decoder<'id, Op, F>, R: Reassemble<D::Event>>(
        &self,
        decoder: &mut D,
        reassembler: Option<R>,
        frames: impl IntoIterator<Item = F>,
    ) -> Result<(), ProviderError> {
        feed(decoder, reassembler, self.0, frames)
    }
}

/// Drive one reply by hand: `run` decodes into it until the reply ends, or
/// fails with what ended it, and the reply folds into `fold` as a stream
/// would.
#[cfg(test)]
pub(crate) fn decode_with<Op: Operation>(
    fold: Op::Fold,
    provider: &str,
    run: impl for<'id> FnOnce(Replying<'id, Op>) -> Result<(), ProviderError>,
) -> Decoded<Op> {
    let shared = Mutex::new(Shared::new(fold));
    let fed = run(Replying(&shared));
    settle(
        shared,
        fed,
        crate::wire::Reply {
            provider: provider.to_owned(),
            raw: serde_json::Value::Null,
            provider_request_id: None,
        },
    )
}

/// The reply failed: the error is its last item.
fn fail<Op: Operation>(
    reply: &Mutex<Shared<Op>>,
    slot: Option<&AdapterSlot>,
    error: ProviderError,
) {
    if let Some(slot) = slot {
        slot.fail(&error);
    }
    lock(reply).items.push_back(Err(error));
}

pub(crate) fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Read pages until one names no cursor, names the cursor it was asked
/// with, or [`MAX_CONTINUATION_PAGES`] were read. `page` reads the page at a
/// cursor (`None` for the first) and returns it with the next cursor.
pub(crate) async fn follow_cursors<P, Fut>(
    provider: &str,
    operation: &str,
    mut page: impl FnMut(Option<String>) -> Fut,
) -> Result<Vec<P>, ProviderError>
where
    Fut: Future<Output = Result<(P, Option<String>), ProviderError>>,
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
