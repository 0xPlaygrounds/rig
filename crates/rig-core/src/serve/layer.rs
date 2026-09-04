//! Interception as handler composition: a [`Layer`] is a [`Serve`] that
//! wraps another handler and an [`Intercept`] — the policy that sees every
//! dispatch before the handler does ([`Intercept::before`]) and every
//! answer after ([`Intercept::after`]). Layers nest by wrapping; a
//! [`Decision`] and a [`Verdict`] are data. Decisions are program, never
//! record: the driver's tap moves to the innermost hop, so a denial leaves
//! no record and a replacement leaves the handler's real answer in it — a
//! replay re-makes the decision.

use std::{
    pin::Pin,
    task::{Context, Poll},
};

use futures::{
    StreamExt,
    channel::{mpsc, oneshot},
};
use serde::{Deserialize, Serialize};

use crate::{
    effect::{EffectId, EffectKind, HandlerDescriptor, Outcome, family},
    error::{ErrorKind, ErrorReport},
    streaming::StreamEvent,
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};

use super::{ErasedHandler, HandlerFuture, OutcomeSink, Serve, StreamTap, stream_truncated};

/// What a layer decides about a dispatch before the handler sees it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case")]
pub enum Decision {
    /// Serve it as it is.
    Proceed,
    /// Serve this instead. A patch never changes the family: one that does
    /// is an `Internal` report on the consumer's outcome, and no dispatch.
    Patch(EffectKind),
    /// Do not serve it: the consumer's outcome is this report, and the
    /// record holds nothing. [`Decision::deny`] builds the usual one
    /// (`ErrorKind::Denied`); a report of another kind — `Cancelled`, the
    /// way a program stops — travels as given.
    Deny(ErrorReport),
}

impl Decision {
    /// A denial by policy: `ErrorKind::Denied`, never retryable.
    pub fn deny(reason: impl Into<String>) -> Self {
        Self::Deny(ErrorReport::new(ErrorKind::Denied, reason).with_retryable(false))
    }
}

/// What a layer decides about an answer on its way out.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
#[allow(
    clippy::large_enum_variant,
    reason = "a verdict is made once per dispatch and moved once into the sink; boxing the replacement would cost every layer author an allocation for nothing"
)]
pub enum Verdict {
    /// The consumer receives what the handler answered.
    Keep,
    /// The consumer receives this instead; the record keeps the handler's
    /// answer. Over a streaming dispatch the events were already delivered
    /// as they came, so only an error can replace the answer there: a
    /// `Replace(Ok(_))` reaches the consumer as an `Internal` error naming
    /// the layer.
    Replace(Result<Outcome, ErrorReport>),
}

/// The policy a [`Layer`] runs: both methods take `&self`, both are
/// `async`, and a layer that suspends inside `before` (an approval gate
/// answered by a system next tick) keeps the dispatch in flight and its
/// serial slot busy until it decides — like a detached sink. Its await
/// returns on cancellation because the future is dropped when the
/// consumer goes; the *world side* holding the answer channel sees
/// `is_canceled()` on its sender and must not panic on a closed one. The
/// name is the layer's identity in a log's handler table and hook list:
/// a program recorded under one layer stack refuses a replay under
/// another, so a layer is host policy and must say what it is.
pub trait Intercept: WasmCompatSend + WasmCompatSync + 'static {
    /// The layer's name, as the log records it.
    fn name(&self) -> String;

    /// Before the handler: the dispatch as it will be served, or not.
    fn before(
        &self,
        id: EffectId,
        kind: &EffectKind,
    ) -> impl Future<Output = Decision> + WasmCompatSend;

    /// After the handler: the answer as the consumer will receive it. For a
    /// streaming dispatch `outcome` is the fold of the events (the
    /// completion the record stores).
    fn after(
        &self,
        id: EffectId,
        kind: &EffectKind,
        outcome: &Result<Outcome, ErrorReport>,
    ) -> impl Future<Output = Verdict> + WasmCompatSend;
}

/// A handler wrapped in a policy: a [`Serve`] like any other, registered
/// under the inner handler's descriptor (with the layer's name added,
/// outermost first). Built with [`ErasedHandler::layered`].
pub struct Layer<I: Intercept> {
    inner: ErasedHandler,
    intercept: I,
}

impl<I: Intercept> Layer<I> {
    /// `intercept` around `inner`.
    pub fn new(inner: ErasedHandler, intercept: I) -> Self {
        Self { inner, intercept }
    }

    /// The policy.
    pub fn intercept(&self) -> &I {
        &self.intercept
    }

    /// The handler beneath.
    pub fn inner(&self) -> &ErasedHandler {
        &self.inner
    }

    fn internal(&self, message: String) -> ErrorReport {
        ErrorReport::new(
            ErrorKind::Internal,
            format!("layer `{}`: {message}", self.intercept.name()),
        )
        .with_retryable(false)
    }

    /// Serve a unary dispatch beneath: the inner handler answers into a
    /// sink of its own, which carries the driver's tap, so the record is
    /// its answer; the fold comes back here for the verdict.
    async fn serve_unary(
        &self,
        kind: &EffectKind,
        outer: &mut OutcomeSink,
    ) -> Result<Outcome, ErrorReport> {
        let (reply, receiver) = oneshot::channel();
        let inner = OutcomeSink::unary(outer.id(), reply)
            .with_tap_slot(outer.take_tap())
            .inheriting(outer);
        let mut serving = Serving {
            answer: receiver,
            handler: self.inner.handle(kind.clone(), inner),
            handler_done: false,
        };
        let answer = std::future::poll_fn(|cx| serving.poll_answer(cx)).await;
        match answer {
            Ok(outcome) => outcome,
            Err(oneshot::Canceled) => Err(ErrorReport::new(
                ErrorKind::Internal,
                "the handler dropped its outcome sink without answering",
            )),
        }
    }

    /// Serve a streaming dispatch beneath: every event is forwarded to the
    /// consumer as it comes (the inner sink taps it for the record); the
    /// terminal is folded for the verdict, which decides what ends the
    /// consumer's stream.
    async fn serve_stream(&self, kind: &EffectKind, outer: &mut OutcomeSink) {
        let (events, receiver) = mpsc::channel(0);
        let inner = OutcomeSink::stream(outer.id(), events)
            .with_tap_slot(outer.take_tap())
            .inheriting(outer);
        let mut serving = Serving {
            answer: receiver,
            handler: self.inner.handle(kind.clone(), inner),
            handler_done: false,
        };
        let mut fold = StreamTap::new();
        let mut decided = false;
        while let Some(item) = std::future::poll_fn(|cx| serving.poll_item(cx)).await {
            if decided {
                // A wire may deliver frames after its terminal record; they
                // pass through as they are.
                let _ = outer.send(item).await;
                continue;
            }
            let Some(outcome) = fold.observe(&item) else {
                let _ = outer.send(item).await;
                continue;
            };
            decided = true;
            let ending = match self.intercept.after(outer.id(), kind, &outcome).await {
                Verdict::Keep => item,
                Verdict::Replace(Err(report)) => Err(report),
                Verdict::Replace(Ok(_)) => Err(self.internal(
                    "cannot replace a streamed answer already delivered; replace with an error, or decide before"
                        .to_owned(),
                )),
            };
            let _ = outer.send(ending).await;
        }
        if !decided {
            // The handler dropped its sink before the terminal: the record
            // says so (the inner sink's tap); the consumer hears it through
            // the verdict.
            let outcome: Result<Outcome, ErrorReport> = Err(stream_truncated());
            let report = match self.intercept.after(outer.id(), kind, &outcome).await {
                Verdict::Keep => stream_truncated(),
                Verdict::Replace(Err(report)) => report,
                Verdict::Replace(Ok(_)) => self.internal(
                    "cannot replace a streamed answer already delivered; replace with an error, or decide before"
                        .to_owned(),
                ),
            };
            let _ = outer.send(Err(report)).await;
        }
    }
}

/// The inner handler's future and the channel its sink answers on, polled
/// together. The channel comes first so that, when the layer's future is
/// dropped (the consumer cancelled), the receiver goes before the handler:
/// the inner sink then sees its consumer gone and records the cancel, as
/// a bare handler's would.
struct Serving<'a, R> {
    answer: R,
    handler: HandlerFuture<'a>,
    handler_done: bool,
}

impl<R> Serving<'_, R> {
    fn poll_handler(&mut self, cx: &mut Context<'_>) {
        if !self.handler_done && self.handler.as_mut().poll(cx).is_ready() {
            self.handler_done = true;
        }
    }
}

impl Serving<'_, oneshot::Receiver<Result<Outcome, ErrorReport>>> {
    fn poll_answer(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Result<Outcome, ErrorReport>, oneshot::Canceled>> {
        self.poll_handler(cx);
        Pin::new(&mut self.answer).poll(cx)
    }
}

impl Serving<'_, mpsc::Receiver<Result<StreamEvent, ErrorReport>>> {
    fn poll_item(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<StreamEvent, ErrorReport>>> {
        self.poll_handler(cx);
        self.answer.poll_next_unpin(cx)
    }
}

impl<I: Intercept> Serve for Layer<I> {
    type Family = family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        let mut descriptor = self.inner.descriptor();
        descriptor.layers.insert(0, self.intercept.name());
        descriptor
    }

    async fn serve(&self, kind: EffectKind, mut sink: OutcomeSink) {
        let id = sink.id();
        let kind = match self.intercept.before(id, &kind).await {
            Decision::Proceed => kind,
            Decision::Patch(patched) => {
                if patched.family() != kind.family() {
                    sink.discard();
                    sink.resolve(Err(self.internal(format!(
                        "patched a {} effect into a {} effect; a layer never changes the family",
                        kind.family(),
                        patched.family()
                    ))))
                    .await;
                    return;
                }
                sink.patched(&patched);
                patched
            }
            Decision::Deny(report) => {
                sink.discard();
                sink.resolve(Err(report)).await;
                return;
            }
        };
        if sink.is_stream() {
            self.serve_stream(&kind, &mut sink).await;
            return;
        }
        let outcome = self.serve_unary(&kind, &mut sink).await;
        let delivered = match self.intercept.after(id, &kind, &outcome).await {
            Verdict::Keep => outcome,
            Verdict::Replace(replacement) => replacement,
        };
        sink.resolve(delivered).await;
    }
}

#[cfg(test)]
mod tests;
