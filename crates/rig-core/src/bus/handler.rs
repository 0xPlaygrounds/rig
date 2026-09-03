//! The impl side of the bus: what a handler is and how it answers.

use std::task::{Context, Poll};

use futures::{SinkExt, channel::mpsc, channel::oneshot};

use crate::{
    completion::CompletionResponse,
    effect::{EffectId, EffectKind, HandlerDescriptor, Outcome},
    error::{ErrorKind, ErrorReport},
    streaming::{BlockAccumulator, StreamEvent, StreamFinal},
    wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync},
};

/// The future a handler returns: boxed, because the driver holds handlers
/// as `Arc<dyn Handler>` (an in-flight task holds its handler while the
/// table is replaced) and the trait must be object-safe. `Send` on native
/// (the `WasmBoxedFuture` fork), which is what makes `BusDriver: Send`.
pub type HandlerFuture<'a> = WasmBoxedFuture<'a, ()>;

/// Something registered on the bus that serves effects.
///
/// Provider and tool authors do not implement this directly: the adapters in
/// [`crate::bus::adapters`] wrap the impl-side traits (`CompletionModel`,
/// `Tool`, `EmbeddingModel`, `ConversationMemory`, `VectorStoreIndex`). A
/// host implements it for out-of-tree kinds ([`EffectKind::Custom`]) or for
/// a replayer.
///
/// A handler answers through the [`OutcomeSink`] it is given: a unary effect
/// resolves it once, a streaming effect feeds it [`StreamEvent`]s ending in
/// [`StreamEvent::Final`]. There is one sink type so a handler body cannot
/// answer on the wrong channel — the sink adapts the shape it receives to
/// the shape the dispatch asked for.
pub trait Handler: WasmCompatSend + WasmCompatSync {
    /// What this handler is: the family-keyed description a typed view
    /// checks at bind time and a scene serializes.
    fn descriptor(&self) -> HandlerDescriptor;

    /// Serve one effect. The future completes when the answer has been
    /// delivered (or the consumer went away — see [`OutcomeSink::send`]).
    fn handle(&self, kind: EffectKind, sink: OutcomeSink) -> HandlerFuture<'_>;
}

/// A handler behind the bus's one erasure, shareable between the driver
/// and every dispatcher: `Clone + Send + Sync + 'static` on every target.
///
/// On native this is `Arc<dyn Handler + Send + Sync>` (every handler is,
/// through the `WasmCompat*` supertraits). On browser wasm the supertraits
/// are no-op markers — a provider client there is `!Send` — and the target
/// has no threads, so the cell asserts `Send + Sync` for the single-threaded
/// runtime the same way the markers do; nothing on that target can move a
/// handler across a thread that does not exist.
#[derive(Clone)]
pub struct ErasedHandler(ErasedInner);

#[cfg(not(target_family = "wasm"))]
type ErasedInner = std::sync::Arc<dyn Handler + Send + Sync>;
#[cfg(target_family = "wasm")]
type ErasedInner = std::sync::Arc<dyn Handler>;

// SAFETY: `wasm32-unknown-unknown` is single-threaded; there is no thread to
// send to or share with, which is the premise of `WasmCompatSend` and
// `WasmCompatSync` being no-op markers on this target. `wasm_compat.rs`
// refuses threaded wasm (`+atomics`) at compile time, so the premise holds
// wherever this compiles. This is the bus's one `unsafe`, and it cannot be
// removed by moving the handler table to the driver: `Dispatcher` must be
// `Send + Sync` on wasm (a Bevy `Resource`), so whatever carries a `!Send`
// handler from `Dispatcher::register` to the driver — a table, a command
// queue, a channel — needs this same assertion, or a thread-local side
// channel, which is global state the bus does not have.
#[cfg(target_family = "wasm")]
unsafe impl Send for ErasedHandler {}
#[cfg(target_family = "wasm")]
unsafe impl Sync for ErasedHandler {}

impl ErasedHandler {
    /// Erase `handler`.
    pub fn new(handler: impl Handler + 'static) -> Self {
        Self(std::sync::Arc::new(handler))
    }

    /// Whether two erased handlers are the same allocation.
    pub fn ptr_eq(&self, other: &Self) -> bool {
        std::sync::Arc::ptr_eq(&self.0, &other.0)
    }
}

impl std::fmt::Debug for ErasedHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ErasedHandler")
            .field("key", &self.0.descriptor().key)
            .finish_non_exhaustive()
    }
}

impl Handler for ErasedHandler {
    fn descriptor(&self) -> HandlerDescriptor {
        self.0.descriptor()
    }

    fn handle(&self, kind: EffectKind, sink: OutcomeSink) -> HandlerFuture<'_> {
        self.0.handle(kind, sink)
    }
}

/// Serve one effect on `handler` right here, without a bus: the inline
/// path a standalone tool set or catalog uses. The bus is still the only
/// erasure — this is a direct call on a [`Handler`].
pub async fn serve_inline(handler: &dyn Handler, kind: EffectKind) -> Result<Outcome, ErrorReport> {
    let id = EffectId::from_raw(0);
    let (reply, receiver) = oneshot::channel();
    handler.handle(kind, OutcomeSink::unary(id, reply)).await;
    match receiver.await {
        Ok(outcome) => outcome,
        Err(oneshot::Canceled) => Err(ErrorReport::new(
            ErrorKind::Internal,
            "the handler dropped its outcome sink without answering",
        )),
    }
}

/// The consumer dropped its [`Pending`](super::Pending) or
/// [`EffectStream`](super::EffectStream): nobody is listening. A streaming
/// handler stops on it — that is how cancellation reaches a provider stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SinkClosed;

impl std::fmt::Display for SinkClosed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("the dispatch's consumer is gone")
    }
}

impl std::error::Error for SinkClosed {}

/// The reply half of one dispatch, handed to the handler by the driver.
pub struct OutcomeSink {
    id: EffectId,
    inner: SinkInner,
    tap: Option<Tap>,
}

/// What a bus tap observes: the outcome, as it resolves.
pub(super) type OnOutcome = Box<dyn Fn(&Result<Outcome, ErrorReport>) + Send + Sync>;

/// A bus tap installed by the driver: observes the outcome as it resolves.
struct Tap {
    on_outcome: OnOutcome,
    stream: super::driver::StreamTap,
    fired: bool,
}

impl Tap {
    fn fire(&mut self, outcome: &Result<Outcome, ErrorReport>) {
        if !self.fired {
            self.fired = true;
            (self.on_outcome)(outcome);
        }
    }
}

/// A streaming dispatch answered with a non-completion outcome: what the
/// consumer receives, and what the tap records.
fn wrong_stream_answer(other: &Outcome) -> ErrorReport {
    ErrorReport::new(
        ErrorKind::Internal,
        format!(
            "a streaming dispatch was answered with a {} outcome",
            other.family()
        ),
    )
}

impl Drop for OutcomeSink {
    fn drop(&mut self) {
        // A sink dropped before it answered is a dispatch the consumer sees
        // fail — a stream cut short before its `Final`, a unary handler
        // that never resolved — and the log records the same failure the
        // consumer receives rather than losing the dispatch.
        let unanswered = match &self.inner {
            SinkInner::Unary { reply, .. } => reply.is_some(),
            SinkInner::Stream { finished, .. } => !*finished,
        };
        if unanswered && self.tap.as_ref().is_some_and(|tap| !tap.fired) {
            let report = match &self.inner {
                SinkInner::Unary { .. } => ErrorReport::new(
                    ErrorKind::Internal,
                    "the handler dropped its outcome sink without answering",
                ),
                SinkInner::Stream { .. } => super::dispatcher::stream_truncated(),
            };
            self.tap_outcome(&Err(report));
        }
    }
}

#[allow(
    clippy::large_enum_variant,
    reason = "one sink per dispatch, moved into the handler once; the unary arm carries the fold state"
)]
enum SinkInner {
    /// A unary dispatch. A streaming handler answering here is folded by the
    /// accumulator and resolved at `Final`.
    Unary {
        reply: Option<oneshot::Sender<Result<Outcome, ErrorReport>>>,
        accumulator: BlockAccumulator,
        message_id: Option<String>,
    },
    /// A streaming dispatch. A unary handler answering here has its
    /// completion re-emitted as events. `finished` is set once a unary
    /// answer was re-emitted, so a later `send` is refused.
    Stream {
        events: mpsc::Sender<Result<StreamEvent, ErrorReport>>,
        finished: bool,
    },
}

impl OutcomeSink {
    pub(super) fn unary(
        id: EffectId,
        reply: oneshot::Sender<Result<Outcome, ErrorReport>>,
    ) -> Self {
        Self {
            id,
            inner: SinkInner::Unary {
                reply: Some(reply),
                accumulator: BlockAccumulator::new(),
                message_id: None,
            },
            tap: None,
        }
    }

    pub(super) fn stream(
        id: EffectId,
        events: mpsc::Sender<Result<StreamEvent, ErrorReport>>,
    ) -> Self {
        Self {
            id,
            inner: SinkInner::Stream {
                events,
                finished: false,
            },
            tap: None,
        }
    }

    pub(super) fn with_tap(mut self, on_outcome: OnOutcome) -> Self {
        self.tap = Some(Tap {
            on_outcome,
            stream: super::driver::StreamTap::new(),
            fired: false,
        });
        self
    }

    fn tap_outcome(&mut self, outcome: &Result<Outcome, ErrorReport>) {
        if let Some(tap) = &mut self.tap {
            tap.fire(outcome);
        }
    }

    fn tap_item(&mut self, item: &Result<StreamEvent, ErrorReport>) {
        if let Some(tap) = &mut self.tap
            && let Some(outcome) = tap.stream.observe(item)
        {
            tap.fire(&outcome);
        }
    }

    /// The dispatch this sink answers.
    pub const fn id(&self) -> EffectId {
        self.id
    }

    /// Whether the dispatch asked for a stream (`dispatch_stream`).
    pub const fn is_stream(&self) -> bool {
        matches!(self.inner, SinkInner::Stream { .. })
    }

    /// Whether the consumer is still listening.
    pub fn is_closed(&self) -> bool {
        match &self.inner {
            SinkInner::Unary { reply, .. } => reply.as_ref().is_none_or(|r| r.is_canceled()),
            SinkInner::Stream { events, finished } => *finished || events.is_closed(),
        }
    }

    /// Resolve a unary dispatch. On a streaming dispatch a completion is
    /// re-emitted as its events followed by `Final`; any other outcome, or an
    /// error, is delivered as the stream's one item.
    pub fn resolve(mut self, outcome: Result<Outcome, ErrorReport>) -> HandlerFuture<'static> {
        Box::pin(async move {
            // What the tap records is what the consumer receives: a stream
            // dispatch answered with a non-completion outcome is delivered
            // as an error, and recorded as that error.
            let delivered: Result<Outcome, ErrorReport> = match (&self.inner, &outcome) {
                (SinkInner::Stream { .. }, Ok(other))
                    if !matches!(other, Outcome::Completion(_)) =>
                {
                    Err(ErrorReport::new(
                        ErrorKind::Internal,
                        format!(
                            "a streaming dispatch was answered with a {} outcome",
                            other.family()
                        ),
                    ))
                }
                _ => outcome.clone(),
            };
            self.tap_outcome(&delivered);
            match &mut self.inner {
                SinkInner::Unary { reply, .. } => {
                    if let Some(reply) = reply.take() {
                        // A dropped receiver is the consumer cancelling; nothing to do.
                        let _ = reply.send(outcome);
                    }
                }
                SinkInner::Stream { events, finished } => {
                    if *finished {
                        return;
                    }
                    match delivered {
                        Ok(Outcome::Completion(response)) => {
                            for item in events_from_response(&response) {
                                if events.send(item).await.is_err() {
                                    break;
                                }
                            }
                        }
                        Ok(other) => {
                            let _ = events.send(Err(wrong_stream_answer(&other))).await;
                        }
                        Err(report) => {
                            let _ = events.send(Err(report)).await;
                        }
                    }
                    *finished = true;
                }
            }
        })
    }

    /// Feed one stream item. On a unary dispatch the item is folded into the
    /// accumulator and `Final` resolves the dispatch with the aggregated
    /// [`CompletionResponse`]; an error resolves it with the report.
    ///
    /// `Err(SinkClosed)` means the consumer is gone: stop producing.
    pub async fn send(&mut self, item: Result<StreamEvent, ErrorReport>) -> Result<(), SinkClosed> {
        self.tap_item(&item);
        match &mut self.inner {
            SinkInner::Stream { events, finished } => {
                if *finished {
                    return Err(SinkClosed);
                }
                // `Final` is not the end of the channel: a wire may still
                // deliver frames after its terminal record (a late message
                // id, a provider error), and the consumer's post-final rules
                // are its own. The stream ends when the handler drops the
                // sink.
                events.send(item).await.map_err(|_| SinkClosed)
            }
            SinkInner::Unary {
                reply,
                accumulator,
                message_id,
            } => {
                let Some(sender) = reply.as_ref() else {
                    return Err(SinkClosed);
                };
                if sender.is_canceled() {
                    *reply = None;
                    return Err(SinkClosed);
                }
                match item {
                    Ok(StreamEvent::Final(terminal)) => {
                        let outcome = finish_unary(accumulator, message_id.take(), terminal);
                        if let Some(reply) = reply.take() {
                            let _ = reply.send(outcome);
                        }
                        Ok(())
                    }
                    Ok(event) => {
                        if let StreamEvent::BlockStart {
                            id,
                            kind: crate::streaming::BlockKind::Message,
                        } = &event
                            && let Some(wire) = id.wire_str()
                        {
                            *message_id = Some(wire.to_owned());
                        }
                        if let Err(report) = accumulator.apply(&event)
                            && let Some(reply) = reply.take()
                        {
                            let _ = reply.send(Err(report));
                        }
                        Ok(())
                    }
                    Err(report) => {
                        if let Some(reply) = reply.take() {
                            let _ = reply.send(Err(report));
                        }
                        Ok(())
                    }
                }
            }
        }
    }

    /// Wait until the consumer can take another item without the driver
    /// stalling — the back-pressure point a streaming handler may poll
    /// explicitly. Resolves immediately on a unary dispatch.
    pub fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), SinkClosed>> {
        match &mut self.inner {
            SinkInner::Stream { events, .. } => events.poll_ready(cx).map_err(|_| SinkClosed),
            SinkInner::Unary { .. } => Poll::Ready(Ok(())),
        }
    }
}

pub(super) fn finish_unary(
    accumulator: &mut BlockAccumulator,
    message_id: Option<String>,
    terminal: StreamFinal,
) -> Result<Outcome, ErrorReport> {
    let choice = std::mem::replace(accumulator, BlockAccumulator::new()).finish();
    let mut response = CompletionResponse::new(choice, terminal.usage, terminal.provider.clone())
        .with_optional_finish_reason(terminal.finish_reason.clone());
    response.message_id = message_id.or(terminal.message_id.clone());
    response.response_id = terminal.response_id.clone();
    response.provider_request_id = terminal.provider_request_id.clone();
    response.model = terminal.model.clone();
    response.raw = terminal.raw;
    Ok(Outcome::Completion(response))
}

/// Re-emit a completed response as the events a stream consumer expects:
/// one block per content item, then `Final`. Used when a unary answer meets
/// a streaming dispatch (a replayed log, a unary-only custom handler).
pub fn events_from_response(
    response: &CompletionResponse,
) -> Vec<Result<StreamEvent, ErrorReport>> {
    use crate::{
        message::AssistantContent,
        providers::internal::adapter::AdapterOutput,
        streaming::{BlockId, MintKind, ToolCallEnd},
    };

    let mut out = AdapterOutput::new();
    if let Some(message_id) = &response.message_id {
        out.message_id(message_id.clone());
    }
    for (index, content) in response.choice.iter().enumerate() {
        let index = index as u64;
        match content {
            AssistantContent::Text(text) => {
                let id = BlockId::minted(MintKind::Text, index);
                out.text_start(id.clone(), text.additional_params.clone());
                out.text(text.text.clone());
                out.text_end(id);
            }
            AssistantContent::Reasoning(reasoning) => {
                let id = reasoning
                    .id
                    .as_deref()
                    .map(BlockId::wire)
                    .unwrap_or_else(|| BlockId::minted(MintKind::Reasoning, index));
                out.reasoning_end(id, Some(reasoning.clone()), None, true);
            }
            // Images never stream (no adapter emits one, the accumulator has
            // no block for one); a unary answer carrying an image reaches a
            // stream consumer as an unmodeled item, verbatim.
            AssistantContent::Image(image) => match serde_json::to_value(image) {
                Ok(value) => out.unknown(crate::streaming::UnknownPayload::new(value)),
                Err(error) => out.error(crate::completion::CompletionError::JsonError(error)),
            },
            AssistantContent::ToolCall(call) => {
                let id = BlockId::wire(call.id.as_str());
                let mut end =
                    ToolCallEnd::whole(call.function.name.clone(), call.function.arguments.clone())
                        .with_tool_id(call.id.as_str());
                if let Some(provider) = &call.provider {
                    end = end.with_call_id(provider.call_id.clone());
                }
                out.tool_call(id, end);
            }
        }
    }
    let mut terminal = StreamFinal::new(response.provider.clone(), response.usage)
        .with_optional_finish_reason(response.finish_reason());
    terminal.message_id = response.message_id.clone();
    terminal.response_id = response.response_id.clone();
    terminal.provider_request_id = response.provider_request_id.clone();
    terminal.model = response.model.clone();
    terminal.raw = response.raw.clone();
    out.final_record(terminal);
    // An item that failed to re-emit (an image that did not serialize) is
    // delivered as the error it is, not dropped.
    out.drain()
        .map(|item| item.map_err(|error| ErrorReport::from(&error)))
        .collect()
}
