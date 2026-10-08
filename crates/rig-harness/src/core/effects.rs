//! The one dispatch path. Every model call and tool call goes through
//! [`Effects::dispatch`], which records it with rig-core's effect types
//! under the agent's stable id; an open tool call, which no handler
//! answers, is recorded the same way when it is opened. The session's `effects.jsonl` holds one
//! resolved record per line, with a `{"header": …}` line before them
//! whenever the set of described handlers (the tools and the models used)
//! grew.

use std::any::Any;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use bevy_ecs::prelude::*;
use futures::FutureExt;
use rig_cassette::effect_log::EffectLogRecorder;
use rig_core::catalog::ModelSpec;
use rig_core::effect::{EffectId, EffectKind, HandlerDescriptor, HandlerKey, Outcome, tool_key};
use rig_core::error::{ErrorKind, ErrorReport};
use rig_core::providers::registry::ConnectError;
use rig_core::serve::{Dispatch, ErasedHandler, Observe, Origin, Recorder, Reply, cancelled};
use rig_core::streaming::{Item, StreamEvent};
use serde::{Deserialize, Serialize};

use super::models::{self, ModelConnector};

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
    pub(crate) fn continuing(log: Option<&Path>) -> Self {
        let last = log.and_then(|log| last_id(log).ok()).unwrap_or(0);
        Self {
            recorder: EffectLogRecorder::new(),
            next: AtomicU64::new(last + 1),
            models: HashMap::new(),
            // No header written yet by this process.
            described: AtomicUsize::new(usize::MAX),
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
        let reference = models::reference(spec);
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
        let dispatch = Dispatch::new(id, streaming).with_observer(Box::new(Recorded {
            recorder: self.recorder.clone(),
            id,
        }));
        (id, async move { handler.handle(kind, dispatch).await })
    }

    /// Records the call of the tool `name` with `args`, under `scope` and
    /// `parent` as [`dispatch`](Self::dispatch) does, for a call no handler
    /// answers: its outcome is recorded when the returned [`OpenEffect`] is
    /// settled, or as cancelled when it is dropped first.
    pub(crate) fn open(
        &self,
        scope: &str,
        parent: Option<EffectId>,
        name: &str,
        args: String,
    ) -> OpenEffect {
        let kind = EffectKind::ToolCall {
            name: name.to_owned(),
            args,
        };
        let id = self.begin(scope, parent, tool_key(name), kind);
        OpenEffect {
            recorder: Some(self.recorder.clone()),
            id,
        }
    }

    /// Takes the next effect id and records the effect's start.
    fn begin(
        &self,
        scope: &str,
        parent: Option<EffectId>,
        key: HandlerKey,
        kind: EffectKind,
    ) -> EffectId {
        let id = EffectId::from_raw(self.next.fetch_add(1, Ordering::Relaxed));
        self.recorder.begin(
            id,
            key,
            kind,
            Origin {
                parent,
                scope: Some(Arc::from(scope)),
            },
        );
        id
    }

    /// Run `work`, the task that drives the effect `id`. A panic in it, in
    /// the handler or in the reply's stream, becomes an internal error that
    /// is also recorded as the effect's outcome.
    pub(crate) fn caught<T>(
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
    pub(crate) fn flush(&self, path: &Path) -> io::Result<()> {
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

/// How many of the log's last lines [`last_id`] reads.
const TAIL_LINES: usize = 64;

/// The highest effect id among the last [`TAIL_LINES`] lines of the log at
/// `log`. Ids are taken in order and records are appended as they resolve,
/// so the highest id is among the last few records; reading backwards from
/// the end keeps startup independent of the log's length.
fn last_id(log: &Path) -> io::Result<u64> {
    /// A record line's id; header lines have none.
    #[derive(Deserialize)]
    struct IdOnly {
        id: u64,
    }
    let mut file = File::open(log)?;
    let mut start = file.metadata()?.len();
    let mut tail = Vec::new();
    let mut chunk: u64 = 64 * 1024;
    while start > 0 && tail.iter().filter(|byte| **byte == b'\n').count() <= TAIL_LINES {
        let from = start.saturating_sub(chunk);
        let mut read = vec![0; usize::try_from(start - from).map_err(io::Error::other)?];
        file.seek(SeekFrom::Start(from))?;
        file.read_exact(&mut read)?;
        read.append(&mut tail);
        tail = read;
        start = from;
        chunk = chunk.saturating_mul(2);
    }
    // Unless the whole file was read, the first piece may be part of a line.
    Ok(tail
        .split(|byte| *byte == b'\n')
        .skip(usize::from(start > 0))
        .filter_map(|line| serde_json::from_slice::<IdOnly>(line).ok())
        .map(|record| record.id)
        .max()
        .unwrap_or(0))
}

/// The record of an open tool call, begun when the call opens. Settling
/// it records the call's outcome; dropping it unsettled, as despawning the
/// call does, records the call as cancelled.
pub struct OpenEffect {
    recorder: Option<EffectLogRecorder>,
    id: EffectId,
}

impl OpenEffect {
    /// The call's effect id, as `effects.jsonl` records it.
    pub fn id(&self) -> EffectId {
        self.id
    }

    /// Records `outcome` as the call's outcome; only the first counts.
    pub(crate) fn settle(&mut self, outcome: Result<Outcome, ErrorReport>) {
        if let Some(recorder) = self.recorder.take() {
            recorder.resolve(self.id, outcome);
        }
    }
}

impl Drop for OpenEffect {
    fn drop(&mut self) {
        self.settle(Err(cancelled()));
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

    /// A layer decided the dispatch before any handler saw it: the record
    /// is dropped, as rig-agent's bus does. Replay reruns the layers, so a
    /// recorded refusal would be replayed as if a handler had answered it.
    fn discard(&mut self, _: &str) {
        self.recorder.discard(self.id);
    }

    fn patch(&mut self, kind: &EffectKind) {
        self.recorder.patch(self.id, kind.clone());
    }
}
