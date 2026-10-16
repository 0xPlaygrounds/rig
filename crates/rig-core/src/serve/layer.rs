//! Handler composition with pre-dispatch decisions and post-answer verdicts.
//! Recording observes the handler a layer wraps: denied dispatches leave no record,
//! and replaced answers retain the handler's original outcome for replay.
//!
//! ```
//! use rig_core::serve::BeforeServe;
//!
//! let decision = BeforeServe::deny("Not authorized");
//! assert!(matches!(decision, BeforeServe::Deny(_)));
//! ```

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use serde::{Deserialize, Serialize};

use crate::{
    effect::{Answered, EffectId, EffectKind, HandlerDescriptor, family},
    error::{ErrorKind, ErrorReport},
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};

use super::{Dispatch, ErasedHandler, Reply, Serve, stream_truncated};
use crate::streaming::Relayed;

/// What a layer decides about a dispatch before the handler sees it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case")]
pub enum BeforeServe {
    /// Serve it as it is.
    Proceed,
    /// Serve this instead. A patch never changes the family or a tool call's
    /// target name: either change returns an `Internal` report without reaching
    /// the next layer or dispatching. Tool argument patches remain supported.
    Patch(EffectKind),
    /// Returns this report without invoking the handler or recording a dispatch.
    /// The report's kind is preserved; [`BeforeServe::deny`] uses `ErrorKind::Denied`.
    Deny(ErrorReport),
}

impl BeforeServe {
    /// A denial by policy: `ErrorKind::Denied`, never retryable.
    pub fn deny(reason: impl Into<String>) -> Self {
        Self::Deny(ErrorReport::new(ErrorKind::Denied, reason).with_retryable(false))
    }
}

/// What a layer decides about an answer on its way out.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum AfterServe {
    /// The consumer receives what the handler answered.
    Keep,
    /// The consumer receives this instead; the record keeps the handler's
    /// answer. Over a streaming dispatch the events were already delivered
    /// as they came, so a replacement is the reply's end: a completion
    /// replaces the response the stream ends with (its
    /// [`Relayed::Done`](crate::streaming::Relayed::Done)), which is what
    /// the consumer's fold returns, and an error ends the stream with that
    /// error. A replacement of another family reaches the consumer as an
    /// `Internal` error naming the layer.
    Replace(Result<Answered, ErrorReport>),
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
    ) -> impl Future<Output = BeforeServe> + WasmCompatSend;

    /// After the handler: the answer as the consumer will receive it. For a
    /// streaming dispatch `outcome` is the fold of the events (the
    /// completion the record stores).
    fn after(
        &self,
        id: EffectId,
        kind: &EffectKind,
        outcome: &Result<Answered, ErrorReport>,
    ) -> impl Future<Output = AfterServe> + WasmCompatSend;
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

    fn key(&self) -> crate::effect::HandlerKey {
        self.inner.key().clone()
    }

    fn footprint(&self, kind: &EffectKind) -> Option<crate::tool::Touches> {
        self.inner.footprint(kind)
    }

    fn check(&self, kind: &EffectKind) -> Vec<crate::completion::UnsupportedOption> {
        self.inner.check(kind)
    }

    async fn serve(&self, kind: EffectKind, mut dispatch: Dispatch) -> Reply {
        let id = dispatch.id();
        let name = self.intercept.name();
        let kind = match self.intercept.before(id, &kind).await {
            BeforeServe::Proceed => kind,
            BeforeServe::Patch(patched) => {
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
                dispatch.patched(&patched);
                patched
            }
            BeforeServe::Deny(report) => {
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
                AfterServe::Keep => outcome,
                AfterServe::Replace(replacement) => {
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
                        AfterServe::Keep => item.unwrap_or_else(|| Err(stream_truncated())),
                        AfterServe::Replace(Err(report)) => {
                            attribution.replaced(&intercept.name());
                            Err(report)
                        }
                        AfterServe::Replace(Ok(Answered::Completion(response))) => {
                            attribution.replaced(&intercept.name());
                            Ok(Relayed::Done(response))
                        }
                        AfterServe::Replace(Ok(other)) => {
                            attribution.replaced(&intercept.name());
                            Err(ErrorReport::new(
                                ErrorKind::Internal,
                                format!(
                                    "layer `{}`: replaced a streamed completion with a {} outcome",
                                    intercept.name(),
                                    other.family()
                                ),
                            )
                            .with_retryable(false))
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
