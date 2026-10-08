//! The one dispatch path. Every model call and tool call goes through
//! [`Effects::dispatch`], which records it with rig-core's effect types
//! under the agent's stable id.

use std::fs::OpenOptions;
use std::io::{self, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use bevy_ecs::prelude::*;
use rig_cassette::effect_log::EffectLogRecorder;
use rig_core::effect::{EffectId, EffectKind, Outcome};
use rig_core::error::ErrorReport;
use rig_core::serve::{Dispatch, ErasedHandler, Observe, Origin, Recorder, Reply};
use rig_core::streaming::{Item, StreamEvent};

/// The session's effect recorder and effect id counter.
#[derive(Resource)]
pub struct Effects {
    recorder: EffectLogRecorder,
    next: AtomicU64,
}

impl Effects {
    /// A recorder whose ids continue after the highest id already in the
    /// effect log at `log`, if any, so ids keep increasing across restarts.
    pub fn continuing(log: Option<&Path>) -> Self {
        let last = log
            .and_then(|log| std::fs::read_to_string(log).ok())
            .unwrap_or_default()
            .lines()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .filter_map(|record| record.get("id").and_then(serde_json::Value::as_u64))
            .max()
            .unwrap_or(0);
        Self {
            recorder: EffectLogRecorder::new(),
            next: AtomicU64::new(last + 1),
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
    ) -> (EffectId, impl Future<Output = Reply> + Send + 'static) {
        let id = EffectId::from_raw(self.next.fetch_add(1, Ordering::Relaxed));
        let streaming = kind.streams();
        self.recorder.begin(
            id,
            handler.descriptor().key,
            kind.clone(),
            Origin {
                parent,
                scope: Some(Arc::from(scope)),
            },
        );
        let dispatch = Dispatch::new(id, streaming).with_observer(Box::new(Recorded {
            recorder: self.recorder.clone(),
            id,
        }));
        (id, async move { handler.handle(kind, dispatch).await })
    }

    /// Append every resolved effect to the JSON-lines log at `path`. Effects
    /// still in flight stay for a later flush.
    pub fn flush(&self, path: &Path) -> io::Result<()> {
        let records = self.recorder.take().records;
        if records.is_empty() {
            return Ok(());
        }
        let mut lines = Vec::new();
        for record in &records {
            serde_json::to_writer(&mut lines, record)?;
            lines.push(b'\n');
        }
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?
            .write_all(&lines)
    }
}

/// The recorder's view of one dispatch.
struct Recorded {
    recorder: EffectLogRecorder,
    id: EffectId,
}

impl Observe for Recorded {
    fn outcome(&mut self, outcome: &Result<Outcome, ErrorReport>) {
        self.recorder.resolve(self.id, outcome.clone());
    }

    fn keep_events(&self) -> bool {
        self.recorder.keep_events()
    }

    fn event(&mut self, item: &Item<StreamEvent>) {
        self.recorder.event(self.id, item);
    }

    fn stream_error(&mut self, error: &ErrorReport) {
        self.recorder.stream_error(self.id, error);
    }

    fn origin(&mut self, origin: &rig_core::message::Origin) {
        self.recorder.origin(self.id, origin);
    }

    fn discard(&mut self, _layer: &str) {
        self.recorder.discard(self.id);
    }

    fn patch(&mut self, kind: &EffectKind) {
        self.recorder.patch(self.id, kind.clone());
    }
}
