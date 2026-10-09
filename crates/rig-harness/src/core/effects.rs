//! The one dispatch path. Every model call and tool call goes through
//! [`Effects::dispatch`], which records it with rig-core's effect types
//! under the agent's stable id; an open tool call, which no handler
//! answers, is recorded the same way when it is opened. The session's `effects.jsonl` holds one
//! resolved record per line, written by rig-cassette's
//! [`jsonl::Writer`], with a `{"header": …}` line before them whenever the
//! header (the tools and models described, the keys used) changed; it
//! reads back with [`jsonl::read`] for rig-cassette's replayer.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use bevy_ecs::prelude::*;
use rig_cassette::effect_log::{EffectLog, EffectLogRecorder, jsonl};
use rig_core::catalog::ModelSpec;
use rig_core::effect::{EffectId, EffectKind, HandlerDescriptor, HandlerKey, tool_key};
use rig_core::error::ErrorReport;
use rig_core::providers::registry::ConnectError;
use rig_core::serve::{Dispatch, ErasedHandler, OpenRecord, Origin, Recorder, Reply, catch_panics};

use super::models::ModelConnector;

/// The session's effect recorder, effect id counter and model handlers.
#[derive(Resource)]
pub struct Effects {
    recorder: EffectLogRecorder,
    /// The same recorder, as dispatches and open records hold it.
    shared: Arc<dyn Recorder + Send + Sync>,
    next: AtomicU64,
    /// One handler per catalog model, built on first use and shared by
    /// every agent that picks the model.
    models: HashMap<String, ErasedHandler>,
}

impl Effects {
    /// A recorder whose ids continue after the highest id already in the
    /// effect log at `log`, if any, so ids keep increasing across restarts.
    pub(crate) fn continuing(log: Option<&Path>) -> Self {
        let last = log
            .and_then(|log| jsonl::last_id(log).ok().flatten())
            .map_or(0, EffectId::as_u64);
        let recorder = EffectLogRecorder::new();
        Self {
            shared: Arc::new(recorder.clone()),
            recorder,
            next: AtomicU64::new(last + 1),
            models: HashMap::new(),
        }
    }

    /// The handler serving `spec`, built by `connector` the first time any
    /// agent picks it, and
    /// described in the log header. A signed-in handler reads the
    /// credential on each request, so a refreshed token needs no rebuild.
    pub(crate) fn model_handler(
        &mut self,
        spec: &'static ModelSpec,
        connector: &ModelConnector,
    ) -> Result<ErasedHandler, ConnectError> {
        let reference = spec.reference();
        if let Some(handler) = self.models.get(&reference) {
            return Ok(handler.clone());
        }
        let handler = connector.handler(spec)?;
        self.describe(vec![handler.descriptor()]);
        self.models.insert(reference, handler.clone());
        Ok(handler)
    }

    /// Forgets the handlers of `vendor`'s models, so the next agent that
    /// picks one connects it again, such as after its sign-in is deleted.
    pub fn forget_vendor(&mut self, vendor: &str) {
        self.models
            .retain(|reference, _| reference.split_once('/').map(|(of, _)| of) != Some(vendor));
    }

    /// Adds `handlers` to the ones the log header describes.
    pub fn describe(&self, handlers: Vec<HandlerDescriptor>) {
        self.recorder.handlers(handlers);
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
    ) -> (EffectId, impl Future<Output = Reply> + Send + 'static) {
        let streaming = kind.streams();
        let id = self.begin(scope, parent, handler.descriptor().key, kind.clone());
        let dispatch = Dispatch::new(id, streaming).recorded_by(self.shared.clone());
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
            self.shared.clone(),
            self.next_id(),
            tool_key(name),
            kind,
            origin(scope, parent),
        )
    }

    /// Takes the next effect id and records the effect's start.
    fn begin(
        &self,
        scope: &str,
        parent: Option<EffectId>,
        key: HandlerKey,
        kind: EffectKind,
    ) -> EffectId {
        let id = self.next_id();
        self.recorder.begin(id, key, kind, origin(scope, parent));
        id
    }

    /// Takes the next effect id.
    fn next_id(&self) -> EffectId {
        EffectId::from_raw(self.next.fetch_add(1, Ordering::Relaxed))
    }

    /// Run `work`, the task that drives the effect `id`. A panic in it, in
    /// the handler or in the reply's stream, becomes an internal error that
    /// is also recorded as the effect's outcome.
    pub(crate) fn caught<T: 'static>(
        &self,
        id: EffectId,
        work: impl Future<Output = Result<T, ErrorReport>> + Send + 'static,
    ) -> impl Future<Output = Result<T, ErrorReport>> + Send + 'static {
        catch_panics(self.shared.clone(), id, work)
    }

    /// Every resolved effect, taken out of the recorder; effects still in
    /// flight stay for a later take.
    pub(crate) fn take(&self) -> EffectLog {
        self.recorder.take()
    }
}

/// Where an effect of the agent `scope` comes from.
fn origin(scope: &str, parent: Option<EffectId>) -> Origin {
    Origin {
        parent,
        scope: Some(Arc::from(scope)),
    }
}
