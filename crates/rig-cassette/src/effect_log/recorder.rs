//! The effect-log recorder: a [`Recorder`] that folds every served dispatch
//! into an [`EffectLog`].
//!
//! ```
//! let recorder = rig_cassette::effect_log::EffectLogRecorder::keeping_stream_events();
//! assert_eq!(recorder.in_flight(), 0);
//! ```

use std::{
    collections::BTreeMap,
    fmt,
    sync::{Arc, Mutex, MutexGuard, PoisonError},
};

use rig_core::serve::{DispatchContext, RecordBuffer, Recorder};
use rig_core::{
    effect::{Answered, EffectId, EffectKind, HandlerDescriptor, HandlerKey},
    error::ErrorReport,
    streaming::{Item, StreamEvent},
};

use super::{EffectLog, LogHeader};

/// Records served dispatches in dispatch order, regardless of resolution order.
/// Clones share the log. Streams retain their folded completion and optionally
/// their original events. The records are rig-core's [`RecordBuffer`]'s; this
/// adds the log's header: the handlers, the signature and stream errors.
#[derive(Clone, Default)]
pub struct EffectLogRecorder {
    records: RecordBuffer,
    /// Locked before the records whenever both are.
    header: Arc<Mutex<LogHeader>>,
    /// Records per key, taken or not: the signature names a key while one
    /// exists, and forgets it when a layer's decision discards the last.
    touched: Arc<Mutex<BTreeMap<HandlerKey, usize>>>,
}

impl EffectLogRecorder {
    /// An empty recorder: streams are recorded as their folded completion.
    pub fn new() -> Self {
        Self::default()
    }

    /// A recorder that keeps a streamed dispatch's events verbatim in its
    /// record (`EffectRecord::events`), so a replay re-emits the original
    /// delta boundaries. Costs the events' size per streamed dispatch; the
    /// fold is the default.
    pub fn keeping_stream_events() -> Self {
        Self {
            records: RecordBuffer::new().keeping_stream_events(),
            ..Self::default()
        }
    }

    /// This recorder stamping each record with the wall-clock times its
    /// dispatch began and resolved (`EffectRecord::started_ms`,
    /// `EffectRecord::finished_ms`). Without it a log is the same on every
    /// run, as a golden needs.
    #[must_use]
    pub fn timed(mut self) -> Self {
        self.records = self.records.timed();
        self
    }

    /// Return a snapshot of registered handlers and the effect signature
    /// accumulated from served dispatches.
    pub fn header(&self) -> LogHeader {
        self.lock_header().clone()
    }

    /// A copy of every resolved dispatch so far, in dispatch order, under
    /// the header. A dispatch still in flight is not in the log yet; it
    /// takes its place (ahead of everything served after it) when it
    /// resolves.
    #[must_use = "the log is a copy of the recorder's state"]
    pub fn log(&self) -> EffectLog {
        let header = self.lock_header();
        let records = self.records.records();
        let mut log = EffectLog {
            header: header.clone(),
            records,
        };
        drop(header);
        log.retain_recorded_errors();
        log
    }

    /// Take the resolved dispatches, leaving the recorder holding only the
    /// ones still in flight; the header stays (the signature keeps growing).
    #[must_use = "the taken log is the only copy"]
    pub fn take(&self) -> EffectLog {
        let mut header = self.lock_header();
        let taken = self.records.take_resolved();
        let taken_ids: std::collections::BTreeSet<_> =
            taken.iter().map(|record| record.id).collect();
        let taken_header = header.clone();
        header.stream_errors.retain(|id, _| !taken_ids.contains(id));
        drop(header);
        let mut log = EffectLog {
            header: taken_header,
            records: taken,
        };
        log.retain_recorded_errors();
        log
    }

    /// Stamp one program's identity under its scope
    /// ([`LogHeader::programs`]): a host writing several programs' effects
    /// into one log names each by its scope.
    pub fn set_program_identity(&self, scope: impl Into<String>, identity: super::ProgramIdentity) {
        self.lock_header().programs.insert(scope.into(), identity);
    }

    /// Dispatches resolved and not yet taken: what [`take`](Self::take)
    /// would take, without copying the header.
    pub fn resolved(&self) -> usize {
        self.records.resolved()
    }

    /// Dispatches recorded and not yet resolved.
    pub fn in_flight(&self) -> usize {
        self.records.in_flight()
    }

    fn lock_header(&self) -> MutexGuard<'_, LogHeader> {
        self.header.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl fmt::Debug for EffectLogRecorder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EffectLogRecorder")
            .field("records", &self.records.resolved())
            .field("in_flight", &self.records.in_flight())
            .finish()
    }
}

impl Recorder for EffectLogRecorder {
    fn stream_error(&self, id: EffectId, error: &ErrorReport) {
        let mut header = self.lock_header();
        let Some(kept) = self.records.kept_events(id) else {
            return;
        };
        let errors = header.stream_errors.entry(id).or_default();
        errors.push(super::RecordedStreamError {
            item: kept + errors.len(),
            error: error.clone(),
        });
    }

    fn tool_output(&self, id: EffectId, output: rig_core::tool::ToolResultContext) {
        self.records.tool_output(id, output);
    }

    /// Handlers: a key already described is re-described in place, a new
    /// one appended, so the header lists every handler the driver served
    /// during the recording, in installation order.
    fn handlers(&self, handlers: Vec<HandlerDescriptor>) {
        self.lock_header().describe(handlers);
    }

    fn begin(&self, id: EffectId, key: HandlerKey, kind: EffectKind, origin: DispatchContext) {
        let mut header = self.lock_header();
        header
            .signature
            .insert_if_absent(key.clone(), kind.family());
        *self
            .touched
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .entry(key.clone())
            .or_insert(0) += 1;
        drop(header);
        self.records.begin(id, key, kind, origin);
    }

    /// A layer decided the dispatch before any handler served it: no
    /// record.
    fn discard(&self, id: EffectId) {
        let mut header = self.lock_header();
        let Some(key) = self.records.remove(id) else {
            return;
        };
        header.stream_errors.remove(&id);
        // A key belongs in the signature only while it has retained or taken records.
        let mut touched = self.touched.lock().unwrap_or_else(PoisonError::into_inner);
        let remaining = touched.get(&key).copied().unwrap_or(0).saturating_sub(1);
        if remaining == 0 {
            touched.remove(&key);
            header.signature.remove(&key);
        } else {
            touched.insert(key, remaining);
        }
    }

    fn patch(&self, id: EffectId, kind: EffectKind) {
        self.records.patch(id, kind);
    }

    fn keep_events(&self) -> bool {
        self.records.keep_events()
    }

    fn event(&self, id: EffectId, item: &Item<StreamEvent>) {
        self.records.event(id, item);
    }

    fn origin(&self, id: EffectId, origin: &rig_core::message::Origin) {
        self.records.origin(id, origin);
    }

    fn resolve(&self, id: EffectId, outcome: Result<Answered, ErrorReport>) {
        self.records.resolve(id, outcome);
    }
}
