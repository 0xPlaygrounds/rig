//! The one path from the agent to a handler. Every model call and tool call
//! is dispatched here: it gets an effect id, is recorded under the agent's
//! stable id, and its handler's panics are caught.

use std::any::Any;
use std::io::Write;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use bevy_ecs::prelude::*;
use futures::FutureExt;
use rig_cassette::effect_log::EffectLogRecorder;
use rig_core::effect::{EffectId, EffectKind, HandlerKey, Outcome};
use rig_core::message::Origin;
use rig_core::serve::{Dispatch, ErasedHandler, Observe, Recorder, Reply};
use rig_core::streaming::{Item, StreamEvent};
use rig_core::{ErrorKind, ErrorReport};

use super::agent::AgentId;
use super::session::{Session, append};

/// The session's effect recorder and the next effect id.
#[derive(Resource, Default)]
pub struct Effects {
    /// Records every dispatch until [`flush_effects`] writes it out.
    pub recorder: EffectLogRecorder,
    /// The id the next dispatch gets.
    pub next_id: AtomicU64,
}

/// Dispatches `kind` to `handler` on behalf of `agent`: the effect is begun
/// in the recorder, scoped by the agent's id, and the handler's answer,
/// stream items or cancellation are recorded as they happen. Returns the
/// handler's reply, or an internal error if the handler panicked.
pub fn dispatch(
    effects: &Effects,
    agent: &AgentId,
    key: HandlerKey,
    handler: ErasedHandler,
    kind: EffectKind,
) -> impl Future<Output = Result<Reply, ErrorReport>> + Send + 'static {
    let id = EffectId::from_raw(effects.next_id.fetch_add(1, Ordering::Relaxed));
    effects.recorder.begin(
        id,
        key,
        kind.clone(),
        rig_core::serve::Origin {
            parent: None,
            scope: Some(Arc::clone(&agent.0)),
        },
    );
    let dispatch = Dispatch::new(id, kind.streams()).with_observer(Box::new(Recorded {
        recorder: effects.recorder.clone(),
        id,
    }));
    caught(async move { handler.handle(kind, dispatch).await })
}

/// Runs `future`, turning a panic into an internal error that carries the
/// panic's message.
pub fn caught<T>(
    future: impl Future<Output = T> + Send + 'static,
) -> impl Future<Output = Result<T, ErrorReport>> + Send + 'static {
    AssertUnwindSafe(future).catch_unwind().map(|result| {
        result.map_err(|panic| {
            ErrorReport::new(
                ErrorKind::Internal,
                format!("panicked: {}", panic_message(panic.as_ref())),
            )
        })
    })
}

fn panic_message(panic: &(dyn Any + Send)) -> &str {
    panic
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("no message")
}

/// Tells the recorder what happens to one dispatch.
struct Recorded {
    recorder: EffectLogRecorder,
    id: EffectId,
}

impl Observe for Recorded {
    fn adapter_context(&self) -> Option<rig_core::observe::AdapterContext> {
        self.recorder.adapter_context(self.id)
    }

    fn outcome(&mut self, outcome: &Result<Outcome, ErrorReport>) {
        self.recorder.resolve(self.id, outcome.clone());
    }

    fn keep_events(&self) -> bool {
        self.recorder.keep_events()
    }

    fn event(&mut self, item: &Item<StreamEvent>) {
        self.recorder.event(self.id, item);
    }

    fn origin(&mut self, origin: &Origin) {
        self.recorder.origin(self.id, origin);
    }

    fn stream_error(&mut self, error: &ErrorReport) {
        self.recorder.stream_error(self.id, error);
    }

    fn discard(&mut self, _layer: &str) {
        self.recorder.discard(self.id);
    }

    fn patch(&mut self, kind: &EffectKind) {
        self.recorder.patch(self.id, kind.clone());
    }
}

/// Appends the resolved effect records to the session's `effects.jsonl`,
/// one JSON object per line, in dispatch order.
pub(crate) fn flush_effects(effects: Res<Effects>, session: Res<Session>) -> Result {
    let log = effects.recorder.take();
    if log.records.is_empty() {
        return Ok(());
    }
    let mut lines = String::new();
    for record in &log.records {
        lines.push_str(&serde_json::to_string(record)?);
        lines.push('\n');
    }
    append(&session.effects_path())?.write_all(lines.as_bytes())?;
    Ok(())
}
