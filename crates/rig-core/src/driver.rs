//! Calls a model. A [`Model`] pairs a [`Wire`] (what to send and how to read
//! the reply) with a [`Transport`] (how the payload travels). Its `stream`
//! yields any operation's events as a [`Streamed`], and its `call` is that
//! stream, finished: one decoder, one fold and one response constructor for
//! both. The `_observed` twins take the observation context a bus records
//! under.
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
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use futures::StreamExt;
use tracing::Instrument;

use crate::error::ProviderError;
use crate::observe::{AdapterContext, AdapterEnding, AdapterSlot};
use crate::streaming::Streamed;
use crate::wasm_compat::{WasmBoxedFuture, WasmBoxedStream, WasmCompatSend, WasmCompatSync};
use crate::wire::{
    Call, Capabilities, Decoder, Flow, Mode, Operation, Out, Request, Response, Shared, Wire,
    WireEvent,
};

mod dyn_model;
mod http_transport;
mod local;

pub use dyn_model::DynModel;
pub use local::{Local, Step};

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
pub struct Exchange {
    /// How the reply is read.
    pub mode: Mode,
    /// The observation an observing transport records the send under.
    pub(crate) observation: Option<AdapterContext>,
}

/// One reply, opened when first polled.
pub struct Opening<F>(WasmBoxedFuture<'static, Result<Opened<F>, ProviderError>>);

impl<F: WasmCompatSend + 'static> Opening<F> {
    /// The reply `open` opens.
    pub fn new(
        open: impl Future<Output = Result<Opened<F>, ProviderError>> + WasmCompatSend + 'static,
    ) -> Self {
        Self(Box::pin(open))
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
        self.0.as_mut().poll(cx)
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

    /// The whole reply as one document: a response's `raw` when the decoder
    /// records none.
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
        self.finished(request.into(), None)
    }

    /// [`Self::call`], with the attempt observed under `observation`.
    pub fn call_observed(
        &self,
        request: impl Into<Request<W>>,
        observation: AdapterContext,
    ) -> impl Future<Output = Result<Response<W>, ProviderError>> + WasmCompatSend + 'static {
        self.finished(request.into(), Some(observation))
    }

    /// The call opens when first polled, so its span is created under the
    /// caller's instrumented context.
    fn finished(
        &self,
        request: Request<W>,
        observation: Option<AdapterContext>,
    ) -> impl Future<Output = Result<Response<W>, ProviderError>> + WasmCompatSend + 'static {
        let model = self.clone();
        async move {
            model
                .open(request, Mode::Unary, observation)?
                .finish()
                .await
        }
    }

    /// Open a streamed reply. Encoding errors return here; every later
    /// failure arrives in-band, as the stream's last item. Nothing is sent
    /// until the stream is first polled.
    pub fn stream(&self, request: impl Into<Request<W>>) -> Result<Streamed<W::Op>, ProviderError> {
        self.open(request.into(), Mode::Streaming, None)
    }

    /// [`Self::stream`], with the attempt observed under `observation`.
    pub fn stream_observed(
        &self,
        request: impl Into<Request<W>>,
        observation: AdapterContext,
    ) -> Result<Streamed<W::Op>, ProviderError> {
        self.open(request.into(), Mode::Streaming, Some(observation))
    }

    /// [`Self::call`], with a failure paired with the request path of the
    /// reply that failed, for an operation whose errors name their route.
    pub(crate) async fn call_routed(
        &self,
        request: Request<W>,
    ) -> Result<Response<W>, (ProviderError, String)> {
        self.open(request, Mode::Unary, None)
            .map_err(|error| (error, String::new()))?
            .finish_routed()
            .await
    }

    /// The one entry to the driver: the operation's fold for the reply,
    /// the encoded payload, and the transport's reply, in `mode`.
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
        let opening = self.transport.send(payload, Exchange { mode, observation });
        let shared = Arc::new(Mutex::new(Shared::new(fold)));
        let reading = read(
            self.wire.clone(),
            opening,
            Arc::clone(&shared),
            span.clone(),
            mode,
        );
        Ok(Streamed::new(reading, shared, span, provider))
    }
}

/// Read one reply: open it, classify each frame, decode it into the reply's
/// writer, and stop at the provider's end. The stream yields once per frame
/// read, so its items reach the consumer frame by frame.
///
/// The reply's `'id` brand is the borrow of its shared state inside this
/// stream: the decoder and every part handle it holds are tied to it, and
/// none can leave.
fn read<W: Wire>(
    wire: W,
    opening: Opening<W::Frame>,
    shared: Arc<Mutex<Shared<W::Op>>>,
    span: tracing::Span,
    mode: Mode,
) -> WasmBoxedStream<'static, ()> {
    let decoding = span.clone();
    let reading = async_stream::stream! {
        let reply: &Mutex<Shared<W::Op>> = &shared;
        // A unary call sends under the call's span; a stream reads under it.
        let opened = match mode {
            Mode::Unary => opening.instrument(span.clone()).await,
            Mode::Streaming => opening.await,
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
                fail(reply, slot_none(), error);
                return;
            }
        };
        record_request_id(&span, request_id.as_deref());
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
        // Frames counted for observation's EOF and corruption positions.
        let mut counted = 0usize;
        loop {
            let step = match frames.next().await {
                Some(Ok(frame)) => {
                    let analysis = slot.is_some()
                        && analysis_only.is_some_and(|analysis_only| analysis_only(&frame));
                    let classified = decoder.classify(frame);
                    // A corrupt frame is never exempt, whatever its shape.
                    let corrupt = matches!(classified, WireEvent::Corrupt(_));
                    if slot.is_some() && (corrupt || !analysis) {
                        counted += 1;
                    }
                    match classified {
                        WireEvent::Known(event) => decoder.decode(event, Out::new(reply)),
                        // Unmodeled, but always delivered: the consumer sees
                        // it, and aggregation never folds it into the answer.
                        WireEvent::Unknown { event_type, value } => {
                            warn_unmodeled(&event_type, &value);
                            Out::new(reply).unknown(value);
                            Ok(Flow::More)
                        }
                        WireEvent::Corrupt(error) => {
                            if let Some(slot) = &slot {
                                slot.corrupt(counted);
                            }
                            Err(ProviderError::from(error))
                        }
                    }
                }
                Some(Err(error)) => {
                    fail(reply, slot.as_ref(), error);
                    return;
                }
                None => {
                    if let Some(slot) = &slot {
                        slot.transport_eof(counted);
                    }
                    // A decoder that saw the provider's end ends the reply
                    // here; otherwise the frames ran out on it.
                    let step = decoder.eof(Out::new(reply));
                    if !matches!(step, Ok(Flow::Ended(_)))
                        && let Some(slot) = &slot
                    {
                        slot.eof(counted);
                    }
                    match step {
                        Ok(Flow::More) => Err(ProviderError::Truncated),
                        step => step,
                    }
                }
            };
            match step {
                Ok(Flow::More) => yield (),
                Ok(Flow::Ended(_)) => break,
                Err(error) => {
                    fail(reply, slot.as_ref(), enrich(error));
                    return;
                }
            }
        }
        if let Some(slot) = &slot {
            slot.finish(AdapterEnding::Terminal);
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

/// A reply decoded from frames already in hand: its items, then its
/// response or the error that ended it.
#[cfg(any(test, feature = "websocket", feature = "test-utils"))]
pub(crate) struct Decoded<Op: Operation> {
    pub(crate) items: Vec<Result<crate::streaming::Item<Op::Event>, ProviderError>>,
    pub(crate) outcome: Result<Op::Response, ProviderError>,
}

/// What the driver does with a classified frame: a known event is decoded,
/// an unmodeled payload is warned about and delivered, and a corrupt frame
/// is the error that ends the reply.
#[cfg(any(test, feature = "websocket", feature = "test-utils"))]
pub(crate) fn triage<E>(event: WireEvent<E>) -> Result<crate::streaming::Item<E>, ProviderError> {
    match event {
        WireEvent::Known(event) => Ok(crate::streaming::Item::Event(event)),
        WireEvent::Unknown { event_type, value } => {
            warn_unmodeled(&event_type, &value);
            Ok(crate::streaming::Item::Unknown(value))
        }
        WireEvent::Corrupt(error) => Err(ProviderError::from(error)),
    }
}

/// Decode one frame already in hand into `reply`, as [`read`] does.
#[cfg(any(test, feature = "websocket", feature = "test-utils"))]
pub(crate) fn step<'id, Op, F, D>(
    decoder: &mut D,
    reply: &'id Mutex<Shared<Op>>,
    frame: F,
) -> Result<Flow, ProviderError>
where
    Op: Operation,
    D: crate::wire::Decoder<'id, Op, F>,
{
    match triage(decoder.classify(frame))? {
        crate::streaming::Item::Event(event) => decoder.decode(event, Out::new(reply)),
        crate::streaming::Item::Unknown(value) => {
            Out::new(reply).unknown(value);
            Ok(Flow::More)
        }
    }
}

/// Feed frames already in hand through `decoder` into `reply` as [`read`]
/// does: the reply ends at the provider's end, or the decoder decides at EOF.
#[cfg(any(test, feature = "websocket", feature = "test-utils"))]
fn feed<'id, Op, F, D>(
    decoder: &mut D,
    reply: &'id Mutex<Shared<Op>>,
    frames: impl IntoIterator<Item = F>,
) -> Result<(), ProviderError>
where
    Op: Operation,
    D: crate::wire::Decoder<'id, Op, F>,
{
    for frame in frames {
        if let Flow::Ended(_) = step(decoder, reply, frame)? {
            return Ok(());
        }
    }
    match decoder.eof(Out::new(reply))? {
        Flow::Ended(_) => Ok(()),
        Flow::More => Err(ProviderError::Truncated),
    }
}

/// Fold a fed reply: its items, then its response, or the error that ended
/// it. `reply.raw` stands unless it is null, when the decoder's record does.
#[cfg(any(test, feature = "websocket", feature = "test-utils"))]
pub(crate) fn settle<Op: Operation>(
    shared: Mutex<Shared<Op>>,
    fed: Result<(), ProviderError>,
    reply: crate::wire::Reply,
) -> Decoded<Op> {
    let Shared {
        mut fold,
        items,
        end,
        raw,
        ..
    } = shared
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut absorbed = Ok(());
    for item in &items {
        if let (Ok(crate::streaming::Item::Event(event)), Ok(())) = (item, &absorbed) {
            absorbed = crate::wire::Fold::absorb(&mut fold, event);
        }
    }
    let outcome = fed.and(absorbed).and_then(|()| {
        let reply = crate::wire::Reply {
            raw: if reply.raw.is_null() {
                raw.unwrap_or(serde_json::Value::Null)
            } else {
                reply.raw
            },
            ..reply
        };
        crate::wire::Fold::finish(fold, end.ok_or(ProviderError::Truncated)?, reply)
    });
    let mut items: Vec<_> = items.into_iter().collect();
    if let Err(error) = &outcome
        && !items.iter().any(Result::is_err)
    {
        items.push(Err(error.clone()));
    }
    Decoded { items, outcome }
}

/// Decode a reply whose frames are already in hand through `wire`'s
/// decoder and `fold`, without a transport: what a caller that reads a
/// provider's frames itself (a websocket session, a whole body) finishes a
/// reply with.
#[cfg(any(test, feature = "websocket"))]
pub(crate) fn decode_frames<W: Wire>(
    wire: &W,
    fold: <W::Op as Operation>::Fold,
    frames: impl IntoIterator<Item = W::Frame>,
    reply: crate::wire::Reply,
) -> Result<Response<W>, ProviderError> {
    let shared = Mutex::new(Shared::new(fold));
    let fed = feed(&mut wire.decoder(), &shared, frames);
    settle(shared, fed, reply).outcome
}

/// A whole reply body of an HTTP wire, decoded as its one frame: what a
/// caller holding the body finishes a reply with.
#[cfg(any(test, feature = "websocket"))]
pub(crate) fn decode_body<W: Wire<Frame = crate::wire::WireFrame>>(
    wire: &W,
    fold: <W::Op as Operation>::Fold,
    body: String,
    reply: crate::wire::Reply,
) -> Result<Response<W>, ProviderError> {
    decode_frames(wire, fold, [crate::wire::WireFrame::Text(body)], reply)
}

/// A completion reply decoded from frames already in hand, as the bus
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
    let shared = Mutex::new(Shared::new(crate::operation::Turn::new(provider.clone())));
    let fed = feed(&mut wire.decoder(), &shared, frames);
    let decoded = settle(
        shared,
        fed,
        crate::wire::Reply {
            provider,
            raw: serde_json::Value::Null,
            provider_request_id: None,
        },
    );
    let relayed: Vec<Result<Relayed, ErrorReport>> = decoded
        .items
        .into_iter()
        .map(|item| match item {
            Ok(item) => Ok(Relayed::Item(item)),
            Err(error) => Err(ErrorReport::from(&error)),
        })
        // An error that ended the reply is already its last item.
        .chain(
            decoded
                .outcome
                .ok()
                .map(|response| Ok(Relayed::Done(Box::new(response)))),
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
            $crate::operation::Turn::new($provider),
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
/// read.
#[cfg(test)]
macro_rules! feed_frames {
    ($decoder:expr, $provider:expr, $frames:expr) => {
        $crate::driver::decode_with(
            $crate::operation::Turn::new($provider),
            $provider,
            |reply| {
                let mut decoder = $decoder;
                reply.feed(&mut decoder, $frames)
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

    /// Feed frames through `decoder`, as the driver does.
    pub(crate) fn feed<F, D: crate::wire::Decoder<'id, Op, F>>(
        &self,
        decoder: &mut D,
        frames: impl IntoIterator<Item = F>,
    ) -> Result<(), ProviderError> {
        feed(decoder, self.0, frames)
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

/// No observation slot, for a reply that failed before it opened.
fn slot_none() -> Option<&'static AdapterSlot> {
    None
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
