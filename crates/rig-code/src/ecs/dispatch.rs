//! The one dispatch path. Every model call and tool call goes through
//! [`Effects::dispatch`], which records it with rig-core's effect types,
//! scoped by the agent's stable id, and turns a panicking handler into an
//! error.

use std::{
    any::Any, collections::HashSet, fs::OpenOptions, io::Write, panic::AssertUnwindSafe,
    path::PathBuf, sync::Arc,
};

use bevy::{prelude::*, tasks::futures_lite::FutureExt};
use rig_cassette::effect_log::{EffectLogRecorder, LogHeader};
use rig_core::{
    effect::{EffectId, EffectKind, EffectRecord, HandlerKey, Outcome},
    error::{ErrorKind, ErrorReport},
    serve::{Dispatch, ErasedHandler, Observe, Origin, Recorder, Reply},
    streaming::{Item, StreamEvent},
};
use serde::Serialize;

use super::agent::AgentId;

/// Records every dispatch and appends resolved ones to the session's
/// `effects.jsonl`.
#[derive(Resource)]
pub struct Effects {
    recorder: EffectLogRecorder,
    next_id: u64,
    described: HashSet<HandlerKey>,
    file: PathBuf,
    header_written: bool,
}

/// One line of `effects.jsonl`: the log header once per process, then one
/// record per resolved dispatch.
#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum Line<'a> {
    Header(&'a LogHeader),
    Record(&'a EffectRecord),
}

impl Effects {
    /// Record into `file`, keeping streamed events verbatim.
    pub fn new(file: PathBuf) -> Self {
        Self {
            recorder: EffectLogRecorder::keeping_stream_events(),
            next_id: 0,
            described: HashSet::new(),
            file,
            header_written: false,
        }
    }

    /// Serve `kind` with `handler` for the agent `agent`, recorded. `scopes`
    /// reach a tool through its context. The returned future owns everything
    /// it needs; dropping it before it resolves records a cancellation.
    pub fn dispatch(
        &mut self,
        agent: &AgentId,
        handler: ErasedHandler,
        kind: EffectKind,
        scopes: Vec<Arc<dyn Any + Send + Sync>>,
    ) -> impl Future<Output = Reply> + Send + 'static {
        self.next_id += 1;
        let id = EffectId::from_raw(self.next_id);
        let descriptor = handler.descriptor();
        let key = descriptor.key.clone();
        if self.described.insert(key.clone()) {
            self.recorder.handlers(vec![descriptor]);
        }
        self.recorder.begin(
            id,
            key,
            kind.clone(),
            Origin {
                parent: None,
                scope: Some(agent.0.clone()),
            },
        );
        let dispatch = scopes
            .into_iter()
            .fold(Dispatch::new(id, kind.streams()), Dispatch::with_scope)
            .with_observer(Box::new(Recorded {
                recorder: self.recorder.clone(),
                id,
            }));
        async move {
            AssertUnwindSafe(handler.handle(kind, dispatch))
                .catch_unwind()
                .await
                .unwrap_or_else(|_| {
                    Reply::Outcome(Err(ErrorReport::new(
                        ErrorKind::Internal,
                        "the handler panicked",
                    )))
                })
        }
    }

    /// Append the dispatches resolved since the last flush to the effect
    /// log, after the header on the first flush of this process.
    pub fn flush(&mut self) -> std::io::Result<()> {
        let log = self.recorder.take();
        if log.records.is_empty() && self.header_written {
            return Ok(());
        }
        let mut text = String::new();
        if !self.header_written {
            text.push_str(&serde_json::to_string(&Line::Header(&log.header))?);
            text.push('\n');
        }
        for record in &log.records {
            text.push_str(&serde_json::to_string(&Line::Record(record))?);
            text.push('\n');
        }
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.file)?
            .write_all(text.as_bytes())?;
        self.header_written = true;
        Ok(())
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

    fn origin(&mut self, origin: &rig_core::message::Origin) {
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
