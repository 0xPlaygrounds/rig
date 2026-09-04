//! The effect-log recorder: a [`Recorder`] that folds every served dispatch
//! into an [`EffectLog`].

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

/// A bus tap: every dispatch the driver serves is recorded, as an
/// [`EffectRecord`], **in dispatch order** — the slot is opened when the
/// driver takes the command and filled when the dispatch resolves, so two
/// concurrent dispatches to one key are logged in the order they were
/// served, not the order they happened to finish. Cloning shares the log; a
/// streaming dispatch is recorded as the aggregated completion its events
/// fold to.
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

    /// The header the log will carry: set by the driver at
    /// [`BusDriver::record_to`](rig_bus::BusDriver::record_to) (the registered handlers) and by an agent
    /// (the run spec hash); the signature accumulates as dispatches are
    /// served.
    pub fn header(&self) -> LogHeader {
        self.header
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Stamp the run spec hash into the header.
    pub fn set_run_spec(&self, hash: u64) {
        self.header
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .run_spec = Some(hash);
    }

    /// Stamp what names the program into the header: its hook stack, its
    /// required effect row, and the bus policy it runs under.
    pub fn set_program(
        &self,
        hooks: Vec<String>,
        required: rig_core::effect::EffectRow,
        bus: Option<rig_core::serve::ServingPolicy>,
    ) {
        let mut header = self.header.lock().unwrap_or_else(PoisonError::into_inner);
        header.hooks = hooks;
        header.required = required;
        header.bus = bus;
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
    pub fn log(&self) -> EffectLog {
        let records = self
            .slots
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .filter_map(RecordSlot::record)
            .collect();
        EffectLog {
            header: self.header(),
            records,
        }
    }

    /// Take the resolved dispatches, leaving the recorder holding only the
    /// ones still in flight; the header stays (the signature keeps growing).
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
        drop(slots);
        EffectLog {
            header: self.header(),
            records: taken,
        }
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
                id,
                origin,
                key,
                kind,
                outcome: None,
                events,
            });
    }

    // The open slot is almost always the last one begun: a stream's events
    // and a dispatch's outcome land on the newest slots, and every resolved
    // slot before them is dead weight to a scan from the front. Searching
    // from the back makes a long streamed run linear in its events rather
    // than in its records times its events.
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
        drop(slots);
        // The signature is the trace's row: a key with no record left —
        // none taken, none in flight — is not in it.
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
