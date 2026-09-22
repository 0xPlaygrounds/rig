//! The effect-log recorder: a [`Recorder`] that folds every served dispatch
//! into an [`EffectLog`].
//!
//! ```
//! let recorder = rig_cassette::effect_log::EffectLogRecorder::keeping_stream_events();
//! assert_eq!(recorder.in_flight(), 0);
//! ```

use std::{
    fmt,
    sync::{Arc, Mutex, PoisonError},
};

use rig_core::serve::{Origin, Recorder};
use rig_core::{
    effect::{EffectId, EffectKind, EffectRecord, HandlerDescriptor, HandlerKey, Outcome},
    error::ErrorReport,
    streaming::StreamEvent,
};

use super::{EffectLog, LogHeader};

/// Records served dispatches in dispatch order, regardless of resolution order.
/// Clones share the log. Streams retain their folded completion and optionally
/// their original events.
#[derive(Clone, Default)]
pub struct EffectLogRecorder {
    slots: Arc<Mutex<Vec<RecordSlot>>>,
    header: Arc<Mutex<LogHeader>>,
    /// Records per key, taken or not: the signature names a key while one
    /// exists, and forgets it when a layer's decision discards the last.
    touched: Arc<Mutex<std::collections::BTreeMap<HandlerKey, usize>>>,
    /// Keep a streamed dispatch's events verbatim (see
    /// [`Self::keeping_stream_events`]).
    keep_events: bool,
}

/// One dispatch the recorder has seen: opened at serve time, filled at
/// resolution.
struct RecordSlot {
    tool_output: Option<rig_core::tool::ToolResultContext>,
    id: EffectId,
    origin: Origin,
    key: HandlerKey,
    kind: EffectKind,
    outcome: Option<Result<Outcome, ErrorReport>>,
    events: Option<Vec<StreamEvent>>,
}

impl RecordSlot {
    fn record(&self) -> Option<EffectRecord> {
        self.outcome.as_ref().map(|outcome| EffectRecord {
            tool_output: self.tool_output.clone(),
            parent: self.origin.parent,
            scope: self.origin.scope.clone(),
            id: self.id,
            key: self.key.clone(),
            kind: self.kind.clone(),
            outcome: outcome.clone(),
            events: self.events.clone(),
        })
    }
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
            keep_events: true,
            ..Self::default()
        }
    }

    /// Return a snapshot of registered handlers, program identity, and the
    /// effect signature accumulated from served dispatches.
    pub fn header(&self) -> LogHeader {
        self.header
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Stamp one program's identity under its scope
    /// ([`LogHeader::programs`]): a world writing several programs' effects
    /// into one log names each by its scope.
    pub fn set_program_identity(&self, scope: impl Into<String>, identity: super::ProgramIdentity) {
        self.header
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .programs
            .insert(scope.into(), identity);
    }

    /// Describe handlers: a key already described is re-described in place,
    /// a new one appended, so the header lists every handler the driver
    /// served during the recording, in installation order.
    fn set_handlers(&self, handlers: Vec<HandlerDescriptor>) {
        let mut header = self.header.lock().unwrap_or_else(PoisonError::into_inner);
        for handler in handlers {
            match header
                .handlers
                .iter_mut()
                .find(|known| known.key == handler.key)
            {
                Some(known) => *known = handler,
                None => header.handlers.push(handler),
            }
        }
    }

    /// A copy of every resolved dispatch so far, in dispatch order, under
    /// the header. A dispatch still in flight is not in the log yet; it
    /// takes its place (ahead of everything served after it) when it
    /// resolves.
    #[must_use = "the log is a copy of the recorder's state"]
    pub fn log(&self) -> EffectLog {
        let slots = self.slots.lock().unwrap_or_else(PoisonError::into_inner);
        let records = slots.iter().filter_map(RecordSlot::record).collect();
        let mut log = EffectLog {
            header: self.header(),
            records,
        };
        drop(slots);
        log.retain_recorded_deliveries();
        log
    }

    /// Take the resolved dispatches, leaving the recorder holding only the
    /// ones still in flight; the header stays (the signature keeps growing).
    #[must_use = "the taken log is the only copy"]
    pub fn take(&self) -> EffectLog {
        let mut slots = self.slots.lock().unwrap_or_else(PoisonError::into_inner);
        let mut taken = Vec::new();
        slots.retain(|slot| match slot.record() {
            Some(record) => {
                taken.push(record);
                false
            }
            None => true,
        });
        let taken_ids: std::collections::BTreeSet<_> =
            taken.iter().map(|record| record.id).collect();
        let mut header = self.header.lock().unwrap_or_else(PoisonError::into_inner);
        let taken_header = header.clone();
        header.stream_errors.retain(|id, _| !taken_ids.contains(id));
        if let Some(deliveries) = &mut header.deliveries {
            deliveries.retain(|delivery| !taken_ids.contains(&delivery.id));
        }
        drop(header);
        drop(slots);
        let mut log = EffectLog {
            header: taken_header,
            records: taken,
        };
        log.retain_recorded_deliveries();
        log
    }

    /// Dispatches recorded and not yet resolved.
    pub fn in_flight(&self) -> usize {
        self.slots
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .filter(|slot| slot.outcome.is_none())
            .count()
    }

    fn begin_slot(&self, id: EffectId, key: HandlerKey, kind: EffectKind, origin: Origin) {
        self.header
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .signature
            .insert_if_absent(key.clone(), kind.family());
        *self
            .touched
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .entry(key.clone())
            .or_insert(0) += 1;
        let events = (self.keep_events && kind.streams()).then(Vec::new);
        self.slots
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(RecordSlot {
                tool_output: None,
                id,
                origin,
                key,
                kind,
                outcome: None,
                events,
            });
    }

    // Active slots are usually newest; reverse scanning avoids traversing
    // completed records for each streamed event.
    fn event_slot(&self, id: EffectId, event: &StreamEvent) {
        let mut slots = self.slots.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(slot) = slots.iter_mut().rev().find(|slot| slot.id == id)
            && let Some(events) = slot.events.as_mut()
        {
            events.push(event.clone());
        }
    }

    /// A layer decided the dispatch before any handler served it: no
    /// record. The slot is the newest for the id, as `resolve_slot` finds it.
    fn discard_slot(&self, id: EffectId) {
        let mut slots = self.slots.lock().unwrap_or_else(PoisonError::into_inner);
        let Some(position) = slots.iter().rposition(|slot| slot.id == id) else {
            return;
        };
        let key = slots.remove(position).key;
        self.header
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .stream_errors
            .remove(&id);
        if let Some(deliveries) = &mut self
            .header
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .deliveries
        {
            deliveries.retain(|delivery| delivery.id != id);
        }
        drop(slots);
        // A key belongs in the signature only while it has retained or taken records.
        let mut touched = self.touched.lock().unwrap_or_else(PoisonError::into_inner);
        let remaining = touched.get(&key).copied().unwrap_or(0).saturating_sub(1);
        if remaining == 0 {
            touched.remove(&key);
            self.header
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .signature
                .remove(&key);
        } else {
            touched.insert(key, remaining);
        }
    }

    /// A layer served `kind` in place of what began: the record's request
    /// is what the innermost handler served.
    fn patch_slot(&self, id: EffectId, kind: EffectKind) {
        let mut slots = self.slots.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(slot) = slots.iter_mut().rev().find(|slot| slot.id == id) {
            slot.kind = kind;
        }
    }

    fn resolve_slot(&self, id: EffectId, outcome: Result<Outcome, ErrorReport>) {
        let mut slots = self.slots.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(slot) = slots.iter_mut().rev().find(|slot| slot.id == id) {
            slot.outcome = Some(outcome);
        }
    }
}

impl fmt::Debug for EffectLogRecorder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let slots = self.slots.lock().unwrap_or_else(PoisonError::into_inner);
        let resolved = slots.iter().filter(|slot| slot.outcome.is_some()).count();
        f.debug_struct("EffectLogRecorder")
            .field("records", &resolved)
            .field("in_flight", &(slots.len() - resolved))
            .finish()
    }
}

impl Recorder for EffectLogRecorder {
    fn unsupported_delivery(&self, reason: &str) {
        let mut header = self.header.lock().unwrap_or_else(PoisonError::into_inner);
        if !header
            .delivery_limitations
            .iter()
            .any(|known| known == reason)
        {
            header.delivery_limitations.push(reason.to_owned());
        }
    }

    fn stream_error(&self, id: EffectId, error: &ErrorReport) {
        let slots = self.slots.lock().unwrap_or_else(PoisonError::into_inner);
        let Some(events) = slots
            .iter()
            .rev()
            .find(|slot| slot.id == id)
            .and_then(|slot| slot.events.as_ref())
        else {
            return;
        };
        let mut header = self.header.lock().unwrap_or_else(PoisonError::into_inner);
        let errors = header.stream_errors.entry(id).or_default();
        errors.push(super::RecordedStreamError {
            item: events.len() + errors.len(),
            error: error.clone(),
        });
    }

    fn begin_delivery_tracking(&self) {
        self.header
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .deliveries
            .get_or_insert_with(Vec::new);
    }

    fn delivery(&self, delivery: rig_core::effect::Delivery) {
        // Hold the slot lock through insertion so a layer cannot discard the
        // record between the membership check and delivery recording.
        let slots = self.slots.lock().unwrap_or_else(PoisonError::into_inner);
        if !slots.iter().any(|slot| slot.id == delivery.id) {
            return;
        }
        self.header
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .deliveries
            .get_or_insert_with(Vec::new)
            .push(delivery);
        drop(slots);
    }

    fn tool_output(&self, id: EffectId, output: rig_core::tool::ToolResultContext) {
        let mut slots = self.slots.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(slot) = slots
            .iter_mut()
            .rev()
            .find(|slot| slot.id == id && slot.outcome.is_none())
        {
            slot.tool_output = Some(output);
        }
    }
    fn handlers(&self, handlers: Vec<HandlerDescriptor>) {
        self.set_handlers(handlers);
    }

    fn begin(&self, id: EffectId, key: HandlerKey, kind: EffectKind, origin: Origin) {
        self.begin_slot(id, key, kind, origin);
    }

    fn discard(&self, id: EffectId) {
        self.discard_slot(id);
    }

    fn patch(&self, id: EffectId, kind: EffectKind) {
        self.patch_slot(id, kind);
    }

    fn keep_events(&self) -> bool {
        self.keep_events
    }

    fn event(&self, id: EffectId, event: &StreamEvent) {
        self.event_slot(id, event);
    }

    fn resolve(&self, id: EffectId, outcome: Result<Outcome, ErrorReport>) {
        self.resolve_slot(id, outcome);
    }
}
