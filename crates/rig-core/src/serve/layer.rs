//! Interception as handler composition: a [`Layer`] is a [`Serve`] that
//! wraps another handler and an [`Intercept`] — the policy that sees every
//! dispatch before the handler does ([`Intercept::before`]) and every
//! answer after ([`Intercept::after`]). Layers nest by wrapping; a
//! [`Decision`] and a [`Verdict`] are data. Decisions are program, never
//! record: the driver's observer moves to the innermost hop, so a denial leaves
//! no record and a replacement leaves the handler's real answer in it — a
//! replay re-makes the decision.

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use serde::{Deserialize, Serialize};

use crate::{
    effect::{EffectId, EffectKind, HandlerDescriptor, Outcome, family},
    error::{ErrorKind, ErrorReport},
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};

use super::{Dispatch, ErasedHandler, Reply, Serve, stream_truncated};

/// What a layer decides about a dispatch before the handler sees it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case")]
pub enum Decision {
    /// Serve it as it is.
    Proceed,
    /// Serve this instead. A patch never changes the family or a tool call's
    /// target name: either change returns an `Internal` report without reaching
    /// the next layer or dispatching. Tool argument patches remain supported.
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
    reason = "a verdict is made once per dispatch and returned once; boxing the replacement would cost every layer author an allocation for nothing"
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

/// Host policy before a handler and after its first answer. A suspended
/// verdict keeps execution in flight; driver cancellation drops its future.
/// Streamed verdict futures are polled on the stream consumer's thread.
/// The layer name identifies policy when recording and validating replay.
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
    intercept: Arc<I>,
}

impl<I: Intercept> Layer<I> {
    /// `intercept` around `inner`.
    pub(crate) fn new(inner: ErasedHandler, intercept: I) -> Self {
        Self {
            inner,
            intercept: Arc::new(intercept),
        }
    }

    fn internal(&self, message: String) -> ErrorReport {
        ErrorReport::new(
            ErrorKind::Internal,
            format!("layer `{}`: {message}", self.intercept.name()),
        )
        .with_retryable(false)
    }
}

impl<I: Intercept> Serve for Layer<I> {
    type Family = family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        let mut descriptor = self.inner.descriptor();
        descriptor.layers.insert(0, self.intercept.name());
        descriptor
    }

    async fn serve(&self, kind: EffectKind, mut dispatch: Dispatch) -> Reply {
        let id = dispatch.id();
        let name = self.intercept.name();
        let kind = match self.intercept.before(id, &kind).await {
            Decision::Proceed => kind,
            Decision::Patch(patched) => {
                if patched.family() != kind.family() {
                    dispatch.discard(&name);
                    return Reply::Outcome(Err(self.internal(format!(
                        "patched a {} effect into a {} effect; a layer never changes the family",
                        kind.family(),
                        patched.family()
                    ))));
                }
                if let (
                    EffectKind::ToolCall { name: original, .. },
                    EffectKind::ToolCall {
                        name: replacement, ..
                    },
                ) = (&kind, &patched)
                    && original != replacement
                {
                    dispatch.discard(&name);
                    return Reply::Outcome(Err(self.internal(format!(
                        "patched tool target `{original}` into `{replacement}`; a layer never changes the bound tool"
                    ))));
                }
                dispatch.patched(&name, &patched);
                patched
            }
            Decision::Deny(report) => {
                dispatch.discard(&name);
                return Reply::Outcome(Err(report));
            }
        };
        if !dispatch.is_stream() {
            let folded = Arc::new(Mutex::new(None));
            let inner = dispatch.inner(Some(folded.clone()));
            let attribution = dispatch.attribution();
            let outcome = self
                .inner
                .handle(kind.clone(), inner)
                .await
                .folded_outcome(Some(folded))
                .await;
            return Reply::Outcome(match self.intercept.after(id, &kind, &outcome).await {
                Verdict::Keep => outcome,
                Verdict::Replace(replacement) => {
                    attribution.replaced(&name);
                    replacement
                }
            });
        }
        let folded = Arc::new(Mutex::new(None));
        let inner = dispatch.inner(Some(folded.clone()));
        let attribution = dispatch.attribution();
        let stream = self
            .inner
            .handle(kind.clone(), inner)
            .await
            .into_stream()
            .fuse();
        let intercept = self.intercept.clone();
        Reply::Stream(Box::pin(futures::stream::unfold(
            (stream, intercept, kind, folded, false, attribution),
            move |(mut stream, intercept, kind, folded, mut decided, attribution)| async move {
                let item = stream.next().await;
                if decided && item.is_none() {
                    return None;
                }
                let outcome = if decided {
                    None
                } else {
                    folded
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .take()
                };
                let item = if let Some(outcome) = outcome {
                    decided = true;
                    match intercept.after(id, &kind, &outcome).await {
                        Verdict::Keep => item.unwrap_or_else(|| Err(stream_truncated())),
                        Verdict::Replace(Err(report)) => {
                            attribution.replaced(&intercept.name());
                            Err(report)
                        }
                        Verdict::Replace(Ok(_)) => {
                            attribution.replaced(&intercept.name());
                            Err(ErrorReport::new(
                                ErrorKind::Internal,
                                format!("layer `{}`: cannot replace a streamed answer already delivered; replace with an error, or decide before", intercept.name()),
                            ).with_retryable(false))
                        }
                    }
                } else {
                    item?
                };
                Some((
                    item,
                    (stream, intercept, kind, folded, decided, attribution),
                ))
            },
        )))
    }
}

#[cfg(test)]
mod tests;
