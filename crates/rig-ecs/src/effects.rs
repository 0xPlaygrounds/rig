//! The one dispatch path. Every effect entity ([`stage`](super::stage)),
//! a model call, a tool call or a plugin's effect, runs through
//! [`Effects`] under its effect id, recorded in the kernel's own
//! [`RecordBuffer`], and its record is written as a
//! [`EffectRecorded`](super::stage::EffectRecorded) when it settles. A plugin that
//! keeps an effect log reads those, and continues the ids of the log it
//! appends to with [`Effects::continue_after`].
//!
//! A streamed model call is recorded as the response it ended with, not
//! its stream items: views follow the stream on the call ([`Partial`]),
//! and a replay answers the call with its recorded outcome before it
//! starts, in Gate, so nothing re-emits a stream.
//!
//! [`Partial`]: super::agent::Partial

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use bevy_ecs::prelude::*;
use bevy_tasks::ConditionalSendFuture;
use rig_core::effect::Answered;
use rig_core::effect::{EffectId, EffectKind};
use rig_core::error::ErrorReport;
use rig_core::serve::{
    Dispatch, DispatchContext, ErasedHandler, RecordBuffer, Recorder, Reply, catch_panics,
};
use rig_core::tool::{PublishedContext, ToolContext, ToolResultContext};

/// What a handler's run left: its answer, and what a tool published beside
/// it, if anything.
pub(crate) type Ran = (Result<Answered, ErrorReport>, Option<ToolResultContext>);

/// The effect ids, and the records of the effects running.
#[derive(Resource)]
pub struct Effects {
    /// The last id taken.
    last: AtomicU64,
    /// The records of effect entities' dispatches, timed, until they settle;
    /// without stream items.
    buffer: RecordBuffer,
}

impl Default for Effects {
    fn default() -> Self {
        Self {
            last: AtomicU64::new(0),
            buffer: RecordBuffer::new().timed(),
        }
    }
}

impl Effects {
    /// Effects with ids after `last`, such as the highest id of the effect
    /// log a plugin continues, so ids keep increasing across restarts.
    pub fn continue_after(last: Option<EffectId>) -> Self {
        Self {
            last: AtomicU64::new(last.map_or(0, EffectId::as_u64)),
            ..Self::default()
        }
    }

    /// Runs `kind` through `handler` as the effect `id` of an entity, from
    /// `origin`, recorded in the kernel's buffer; `consume` turns the reply
    /// into the outcome, such as `Reply::into_outcome`. A panic in the
    /// handler or its stream becomes an internal error, recorded as its
    /// outcome. What a tool publishes into its context comes back beside it.
    pub(crate) fn run<F>(
        &self,
        id: EffectId,
        origin: DispatchContext,
        handler: ErasedHandler,
        kind: EffectKind,
        consume: impl FnOnce(Reply) -> F + Send + 'static,
    ) -> impl ConditionalSendFuture<Output = Ran> + 'static
    where
        F: ConditionalSendFuture<Output = Result<Answered, ErrorReport>> + 'static,
    {
        let recorder: Arc<dyn Recorder + Send + Sync> = Arc::new(self.buffer.clone());
        recorder.begin(id, handler.key().clone(), kind.clone(), origin);
        let published = PublishedContext::new();
        let dispatch = Dispatch::new(id, kind.streams())
            .with_scope(published.clone())
            .recorded_by(recorder.clone());
        let work = async move { consume(handler.handle(kind, dispatch).await).await };
        let work = catch_panics(Some(recorder), id, work);
        async move {
            let outcome = work.await;
            let facts = published.take().map(ToolContext::into_result_context);
            (
                outcome,
                facts.filter(|facts| *facts != ToolResultContext::default()),
            )
        }
    }

    /// The records of effect entities' dispatches.
    pub(crate) fn buffer(&self) -> &RecordBuffer {
        &self.buffer
    }

    /// Takes the next effect id.
    pub(crate) fn next_id(&self) -> EffectId {
        EffectId::from_raw(self.last.fetch_add(1, Ordering::Relaxed) + 1)
    }
}

/// An effect handler as a component or resource keeps it. rig-core's
/// [`ErasedHandler`] is `Send` and `Sync` except on browser wasm, where its
/// provider clients hold JavaScript values, and Bevy asks for both on
/// every target.
#[derive(Clone)]
pub struct Handler(pub ErasedHandler);

// SAFETY: browser wasm without the `atomics` target feature runs one
// thread, so a `Handler` is never sent to or shared with another one.
#[cfg(all(
    target_arch = "wasm32",
    target_os = "unknown",
    not(target_feature = "atomics")
))]
unsafe impl Send for Handler {}
// SAFETY: as for `Send`: there is no other thread.
#[cfg(all(
    target_arch = "wasm32",
    target_os = "unknown",
    not(target_feature = "atomics")
))]
unsafe impl Sync for Handler {}
