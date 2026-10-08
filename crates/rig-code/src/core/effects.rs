//! The one dispatch path. Every model call and tool call goes through
//! [`Effects::dispatch`], which records it with rig-core's effect types
//! under the agent's stable id. The session's `effects.jsonl` holds one
//! resolved record per line, with a `{"header": …}` line before them
//! whenever the set of described handlers (the tools and the models used)
//! grew.

use std::any::Any;
use std::collections::HashMap;
use std::error::Error;
use std::fs::OpenOptions;
use std::io::{self, Write};
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use bevy_ecs::prelude::*;
use futures::FutureExt;
use rig_cassette::effect_log::EffectLogRecorder;
use rig_core::catalog::ModelSpec;
use rig_core::effect::{EffectId, EffectKind, HandlerDescriptor, Outcome};
use rig_core::error::{ErrorKind, ErrorReport};
use rig_core::serve::{Dispatch, ErasedHandler, Observe, Origin, Recorder, Reply};
use rig_core::streaming::{Item, StreamEvent};
use serde::Serialize;

use super::models;

/// The session's effect recorder, effect id counter and model handlers.
#[derive(Resource)]
pub struct Effects {
    recorder: EffectLogRecorder,
    next: AtomicU64,
    /// One handler per catalog model, built on first use and shared by
    /// every agent that picks the model.
    models: HashMap<String, ErasedHandler>,
    /// How many handlers the last header written to the log described.
    described: AtomicUsize,
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
            models: HashMap::new(),
            // No header written yet by this process.
            described: AtomicUsize::new(usize::MAX),
        }
    }

    /// The handler serving `spec`, built from the environment's credentials
    /// the first time any agent picks it, and described in the log header.
    pub fn model_handler(
        &mut self,
        spec: &'static ModelSpec,
    ) -> Result<ErasedHandler, Box<dyn Error + Send + Sync>> {
        let reference = models::reference(spec);
        if let Some(handler) = self.models.get(&reference) {
            return Ok(handler.clone());
        }
        let handler = models::handler(spec)?;
        self.describe(vec![handler.descriptor()]);
        self.models.insert(reference, handler.clone());
        Ok(handler)
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

    /// Run `work`, the task that drives the effect `id`. A panic in it, in
    /// the handler or in the reply's stream, becomes an internal error that
    /// is also recorded as the effect's outcome.
    pub fn caught<T>(
        &self,
        id: EffectId,
        work: impl Future<Output = Result<T, ErrorReport>> + Send + 'static,
    ) -> impl Future<Output = Result<T, ErrorReport>> + Send + 'static {
        let recorder = self.recorder.clone();
        async move {
            // Unwinding drops the dispatch's observer, which records a
            // cancellation; the panic recorded here replaces it.
            AssertUnwindSafe(work)
                .catch_unwind()
                .await
                .unwrap_or_else(|panic| {
                    let report = ErrorReport::new(
                        ErrorKind::Internal,
                        format!("panicked: {}", panic_message(panic.as_ref())),
                    );
                    recorder.resolve(id, Err(report.clone()));
                    Err(report)
                })
        }
    }

    /// Append every resolved effect to the JSON-lines log at `path`, after
    /// a header line when the described handlers grew since the last one.
    /// Effects still in flight stay for a later flush.
    pub fn flush(&self, path: &Path) -> io::Result<()> {
        let log = self.recorder.take();
        let handlers = log.header.handlers.len();
        let header_due = self.described.load(Ordering::Relaxed) != handlers;
        if log.records.is_empty() && !header_due {
            return Ok(());
        }
        let mut lines = Vec::new();
        if header_due {
            serde_json::to_writer(
                &mut lines,
                &HeaderLine {
                    header: &log.header,
                },
            )?;
            lines.push(b'\n');
        }
        for record in &log.records {
            serde_json::to_writer(&mut lines, record)?;
            lines.push(b'\n');
        }
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?
            .write_all(&lines)?;
        self.described.store(handlers, Ordering::Relaxed);
        Ok(())
    }
}

/// The log's header line, `{"header": …}`; record lines are bare records.
#[derive(Serialize)]
struct HeaderLine<'a> {
    header: &'a rig_cassette::effect_log::LogHeader,
}

/// The message a panic was raised with.
fn panic_message(panic: &(dyn Any + Send)) -> &str {
    panic
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("no message")
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
