//! The one dispatch path. Every model call and tool call goes through
//! [`Effects::dispatch`] under the agent's stable id and an effect id; an
//! open tool call, which no handler answers, gets its id when it is opened.
//! A plugin that keeps an effect log, such as rig-harness's, inserts
//! [`Effects::recorded_by`] its recorder, which then sees every effect with
//! rig-core's effect types; without one nothing is recorded.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use bevy_ecs::prelude::*;
use rig_core::effect::{EffectId, EffectKind, tool_key};
use rig_core::error::ErrorReport;
use rig_core::serve::{Dispatch, ErasedHandler, OpenRecord, Origin, Recorder, Reply, catch_panics};

use bevy_tasks::ConditionalSendFuture;

/// The effect ids, and the recorder of every effect, if any.
#[derive(Resource, Default)]
pub struct Effects {
    recorder: Option<Arc<dyn Recorder + Send + Sync>>,
    /// The last id taken.
    last: AtomicU64,
}

impl Effects {
    /// Effects recorded by `recorder`, with ids after `last`, such as the
    /// highest id of the log it continues, so ids keep increasing across
    /// restarts.
    pub fn recorded_by(recorder: Arc<dyn Recorder + Send + Sync>, last: Option<EffectId>) -> Self {
        Self {
            recorder: Some(recorder),
            last: AtomicU64::new(last.map_or(0, EffectId::as_u64)),
        }
    }

    /// Dispatch `kind` to `handler`, recorded under `scope` (the agent's
    /// id) with `parent` as the dispatch that asked for it. Returns the
    /// effect id and the reply. Dropping the reply before it resolves
    /// records the effect as cancelled.
    pub fn dispatch(
        &self,
        scope: &str,
        parent: Option<EffectId>,
        handler: ErasedHandler,
        kind: EffectKind,
    ) -> (
        EffectId,
        impl ConditionalSendFuture<Output = Reply> + 'static,
    ) {
        let id = self.next_id();
        let mut dispatch = Dispatch::new(id, kind.streams());
        if let Some(recorder) = &self.recorder {
            let key = handler.descriptor().key;
            recorder.begin(id, key, kind.clone(), origin(scope, parent));
            dispatch = dispatch.recorded_by(recorder.clone());
        }
        (id, async move { handler.handle(kind, dispatch).await })
    }

    /// Records the call of the tool `name` with `args`, under `scope` and
    /// `parent` as [`dispatch`](Self::dispatch) does, for a call no handler
    /// answers: its outcome is recorded when the returned [`OpenRecord`] is
    /// settled, or as cancelled when it is dropped first.
    pub(crate) fn open(
        &self,
        scope: &str,
        parent: Option<EffectId>,
        name: &str,
        args: String,
    ) -> OpenRecord {
        let kind = EffectKind::ToolCall {
            name: name.to_owned(),
            args,
        };
        OpenRecord::begin(
            self.recorder.clone(),
            self.next_id(),
            tool_key(name),
            kind,
            origin(scope, parent),
        )
    }

    /// Takes the next effect id.
    fn next_id(&self) -> EffectId {
        EffectId::from_raw(self.last.fetch_add(1, Ordering::Relaxed) + 1)
    }

    /// Run `work`, the task that drives the effect `id`. A panic in it, in
    /// the handler or in the reply's stream, becomes an internal error that
    /// is also recorded as the effect's outcome.
    pub(crate) fn caught<T: 'static>(
        &self,
        id: EffectId,
        work: impl ConditionalSendFuture<Output = Result<T, ErrorReport>> + 'static,
    ) -> impl ConditionalSendFuture<Output = Result<T, ErrorReport>> + 'static {
        catch_panics(self.recorder.clone(), id, work)
    }
}

/// An effect handler as a component or resource keeps it. rig-core's
/// [`ErasedHandler`] is `Send` and `Sync` except on browser wasm, where its
/// provider clients hold JavaScript values, and Bevy asks for both on
/// every target.
#[derive(Clone)]
pub struct Handler(pub ErasedHandler);

impl Handler {
    /// The handler, to dispatch to.
    pub fn erased(&self) -> ErasedHandler {
        self.0.clone()
    }
}

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

/// Where an effect of the agent `scope` comes from.
fn origin(scope: &str, parent: Option<EffectId>) -> Origin {
    Origin {
        parent,
        scope: Some(Arc::from(scope)),
    }
}
