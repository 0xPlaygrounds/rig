//! A [`Recorder`] that collects whole [`EffectRecord`]s by effect id, for
//! its owner to take.
//!
//! ```
//! use rig_core::effect::{EffectId, EffectKind, Answered, tool_key};
//! use rig_core::serve::{DispatchContext, RecordBuffer, Recorder};
//! use rig_core::tool::ToolResult;
//!
//! let buffer = RecordBuffer::new();
//! let id = EffectId::from_raw(1);
//! let kind = EffectKind::ToolCall { name: "add".into(), args: "{}".into() };
//! buffer.begin(id, tool_key("add"), kind, DispatchContext::default());
//! assert_eq!(buffer.in_flight(), 1);
//! buffer.resolve(id, Ok(Answered::ToolResult { result: ToolResult::skipped("no") }));
//! let record = buffer.take(id).ok_or("resolved")?;
//! assert!(record.outcome.is_ok());
//! # Ok::<(), &str>(())
//! ```

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use super::{DispatchContext, Recorder};
use crate::{
    effect::{
        Answered, Disposition, EffectId, EffectKind, EffectRecord, HandlerDescriptor, HandlerKey,
    },
    error::ErrorReport,
    streaming::{Item, StreamEvent, StreamRecord},
    tool::ToolResultContext,
};

/// Collects the record of every dispatch it is told about, in the order the
/// dispatches began, whatever order they resolve in. A dispatch recorded by
/// it ([`Dispatch::recorded_by`](super::Dispatch::recorded_by)), and every
/// dispatch nested in it, becomes one [`EffectRecord`] once resolved: a
/// layer's [`discard`](Recorder::discard) drops it and a
/// [`patch`](Recorder::patch) replaces its request, as an effect log keeps
/// them. Clones share the records.
///
/// It keeps no handler descriptions and no delivery boundaries: an owner
/// that writes a log keeps those itself. Its records are
/// [`Disposition::Ran`] with no reviewed outcome; a host that answers,
/// reviews or cancels effects itself sets those on the records it takes.
#[derive(Clone, Default)]
pub struct RecordBuffer {
    slots: Arc<Mutex<Vec<Slot>>>,
    keep_events: bool,
    timed: bool,
}

/// One dispatch: opened when it begins, filled when it resolves.
#[derive(Clone)]
struct Slot {
    id: EffectId,
    key: HandlerKey,
    kind: EffectKind,
    origin: DispatchContext,
    outcome: Option<Result<Answered, ErrorReport>>,
    events: Option<Vec<Item<StreamEvent>>>,
    stream_origin: Option<crate::message::Origin>,
    tool_output: Option<ToolResultContext>,
    started_ms: Option<u64>,
    first_item_ms: Option<u64>,
    finished_ms: Option<u64>,
}

impl Slot {
    /// The record, once the dispatch resolved.
    fn into_record(self) -> Option<EffectRecord> {
        Some(EffectRecord {
            published: self.tool_output,
            id: self.id,
            key: self.key,
            kind: self.kind,
            outcome: self.outcome?,
            events: self
                .events
                .and_then(|items| StreamRecord::from_items(items).ok()),
            stream_origin: self.stream_origin,
            parent: self.origin.parent,
            scope: self.origin.scope,
            purpose: self.origin.purpose,
            turn: None,
            call: None,
            started_ms: self.started_ms,
            first_item_ms: self.first_item_ms,
            finished_ms: self.finished_ms,
            reviewed: None,
            disposition: Disposition::Ran,
            decided_by: None,
        })
    }
}

impl RecordBuffer {
    /// An empty buffer that records a stream as its folded outcome and keeps
    /// no times.
    pub fn new() -> Self {
        Self::default()
    }

    /// This buffer keeping a streamed dispatch's items verbatim in its
    /// record ([`EffectRecord::events`]), so a replay re-emits the original
    /// delta boundaries.
    #[must_use]
    pub fn keeping_stream_events(mut self) -> Self {
        self.keep_events = true;
        self
    }

    /// This buffer stamping each record with the wall-clock time its
    /// dispatch began ([`EffectRecord::started_ms`]), streamed its first
    /// item ([`EffectRecord::first_item_ms`]) and resolved
    /// ([`EffectRecord::finished_ms`]). Without it records carry no times,
    /// so a recording is the same on every run.
    #[must_use]
    pub fn timed(mut self) -> Self {
        self.timed = true;
        self
    }

    /// Takes the record of `id` once it resolved; `None` while it is in
    /// flight, or when the buffer has none.
    pub fn take(&self, id: EffectId) -> Option<EffectRecord> {
        let mut slots = self.slots();
        let position = slots
            .iter()
            .rposition(|slot| slot.id == id && slot.outcome.is_some())?;
        slots.remove(position).into_record()
    }

    /// Takes every resolved record, in the order the dispatches began,
    /// leaving the ones in flight.
    pub fn take_resolved(&self) -> Vec<EffectRecord> {
        self.take_resolved_where(|_| true)
    }

    /// Takes every resolved record whose id `take` accepts, in the order
    /// the dispatches began, leaving the rest.
    pub fn take_resolved_where(&self, mut take: impl FnMut(EffectId) -> bool) -> Vec<EffectRecord> {
        let mut slots = self.slots();
        let (taken, left) = std::mem::take(&mut *slots)
            .into_iter()
            .partition(|slot| slot.outcome.is_some() && take(slot.id));
        *slots = left;
        drop(slots);
        taken.into_iter().filter_map(Slot::into_record).collect()
    }

    /// A copy of every resolved record, in the order the dispatches began.
    pub fn records(&self) -> Vec<EffectRecord> {
        let slots = self.slots();
        let resolved: Vec<Slot> = slots
            .iter()
            .filter(|slot| slot.outcome.is_some())
            .cloned()
            .collect();
        drop(slots);
        resolved.into_iter().filter_map(Slot::into_record).collect()
    }

    /// Whether the buffer holds a record of `id`, resolved or not.
    pub fn contains(&self, id: EffectId) -> bool {
        self.slots().iter().any(|slot| slot.id == id)
    }

    /// Drops the record of `id`, resolved or not; the key its dispatch was
    /// routed to, or `None` when the buffer held none.
    pub fn remove(&self, id: EffectId) -> Option<HandlerKey> {
        let mut slots = self.slots();
        let position = slots.iter().rposition(|slot| slot.id == id)?;
        Some(slots.remove(position).key)
    }

    /// How many streamed items the record of `id` keeps so far, or `None`
    /// when it keeps none.
    pub fn kept_events(&self, id: EffectId) -> Option<usize> {
        let slots = self.slots();
        let slot = slots.iter().rev().find(|slot| slot.id == id)?;
        slot.events.as_ref().map(Vec::len)
    }

    /// Records resolved and not yet taken.
    pub fn resolved(&self) -> usize {
        let slots = self.slots();
        slots.iter().filter(|slot| slot.outcome.is_some()).count()
    }

    /// Records begun and not yet resolved.
    pub fn in_flight(&self) -> usize {
        let slots = self.slots();
        slots.iter().filter(|slot| slot.outcome.is_none()).count()
    }

    fn slots(&self) -> MutexGuard<'_, Vec<Slot>> {
        self.slots.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Edits the newest record of `id` that `keep` accepts. Records in
    /// flight are usually the newest, so the search starts there.
    fn with_slot(&self, id: EffectId, keep: impl Fn(&Slot) -> bool, edit: impl FnOnce(&mut Slot)) {
        let mut slots = self.slots();
        if let Some(slot) = slots
            .iter_mut()
            .rev()
            .find(|slot| slot.id == id && keep(slot))
        {
            edit(slot);
        }
    }

    /// Milliseconds since the Unix epoch, when the buffer keeps times.
    fn now(&self) -> Option<u64> {
        self.timed.then(crate::now_ms)
    }
}

impl std::fmt::Debug for RecordBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecordBuffer")
            .field("resolved", &self.resolved())
            .field("in_flight", &self.in_flight())
            .finish()
    }
}

impl Recorder for RecordBuffer {
    fn handlers(&self, _handlers: Vec<HandlerDescriptor>) {}

    fn begin(&self, id: EffectId, key: HandlerKey, kind: EffectKind, origin: DispatchContext) {
        let events = (self.keep_events && kind.streams()).then(Vec::new);
        let started_ms = self.now();
        self.slots().push(Slot {
            id,
            key,
            kind,
            origin,
            outcome: None,
            events,
            stream_origin: None,
            tool_output: None,
            started_ms,
            first_item_ms: None,
            finished_ms: None,
        });
    }

    fn discard(&self, id: EffectId) {
        self.remove(id);
    }

    fn patch(&self, id: EffectId, kind: EffectKind) {
        self.with_slot(id, |_| true, |slot| slot.kind = kind);
    }

    fn keep_events(&self) -> bool {
        self.keep_events
    }

    fn event(&self, id: EffectId, item: &Item<StreamEvent>) {
        if !self.keep_events && !self.timed {
            return;
        }
        self.with_slot(
            id,
            |_| true,
            |slot| {
                slot.first_item_ms = slot.first_item_ms.or_else(|| self.now());
                if let Some(events) = &mut slot.events {
                    events.push(item.clone());
                }
            },
        );
    }

    fn origin(&self, id: EffectId, origin: &crate::message::Origin) {
        let keeps = |slot: &Slot| slot.events.is_some();
        self.with_slot(id, keeps, |slot| {
            slot.stream_origin = Some(origin.clone());
        });
    }

    fn tool_output(&self, id: EffectId, output: ToolResultContext) {
        let open = |slot: &Slot| slot.outcome.is_none();
        self.with_slot(id, open, |slot| slot.tool_output = Some(output));
    }

    fn resolve(&self, id: EffectId, outcome: Result<Answered, ErrorReport>) {
        let finished_ms = self.now();
        self.with_slot(
            id,
            |_| true,
            |slot| {
                slot.outcome = Some(outcome);
                slot.finished_ms = finished_ms;
            },
        );
    }
}

#[cfg(test)]
mod tests;
