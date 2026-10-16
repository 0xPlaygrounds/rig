//! Typed runtime and policy observations, separate from effect replay records.
//! [`Witness`] receives facts and [`ObservationLog`] retains a bounded trace.
//! Optional timestamps come from a host-supplied [`Clock`], not replay identity.
//!
//! ```
//! use rig_core::observe::ObservationLog;
//!
//! let log = ObservationLog::with_capacity(128);
//! log.finalize();
//! assert!(log.trace().finalized);
//! ```

use std::{
    sync::{Arc, Mutex, PoisonError},
    time::Duration,
};

use serde::{Deserialize, Serialize};

use crate::{
    effect::{EffectFamily, EffectId, HandlerKey},
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};

mod adapter;
pub use adapter::{
    AdapterAnalysis, AdapterContext, AdapterEnding, AdapterErrorBoundary, AdapterErrorEnvelope,
    AdapterEvent, AdapterObservation, AdapterUsage, AdapterVerdict, ObservationSink,
    diagnostic_url_secrets, scrub_diagnostic,
};
pub(crate) use adapter::{AdapterSlot, ObservedError};

#[cfg(test)]
mod tests;

/// One observed fact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Observation {
    /// Sink-assigned sequence position in observation order.
    pub seq: u64,
    /// What the fact is about.
    pub subject: Subject,
    /// Where in the pipeline it was seen.
    pub stage: Stage,
    /// Who owns the decision or the observation.
    pub emitter: Emitter,
    /// The fact.
    pub action: Action,
    /// When, as a host-owned monotonic elapsed duration; a measurement,
    /// never a semantic field. `None` when the sink has no [`Clock`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at: Option<Duration>,
}

impl Observation {
    /// A fact with no sequence yet; the sink assigns one.
    pub fn new(subject: Subject, stage: Stage, emitter: Emitter, action: Action) -> Self {
        Self {
            seq: 0,
            subject,
            stage,
            emitter,
            action,
            at: None,
        }
    }
}

/// What an observation is about. Every field is optional because facts
/// exist before an effect has an id (a gate decides a pending intent), and
/// some have no effect at all (a run ending). Correlation never depends on
/// a runtime handle: `scope` is the program's serde id, `order` the
/// driver's dispatch order, `effect`/`parent` the record's ids.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Subject {
    /// The program scope (the run or agent), as the record names it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    /// The driver's dispatch order for the effect, when it has one; stable
    /// before an id is issued, so a pre-dispatch decision correlates.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub order: Option<u64>,
    /// The effect's id once issued.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effect: Option<EffectId>,
    /// The dispatch this one was made from, when a handler made it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<EffectId>,
    /// The key the effect is routed to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<HandlerKey>,
    /// The family of the effect, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub family: Option<EffectFamily>,
}

impl Subject {
    /// A subject with nothing but a scope: a run-level fact.
    pub fn scoped(scope: impl Into<String>) -> Self {
        Self {
            scope: Some(scope.into()),
            ..Self::default()
        }
    }
}

/// Where a fact was seen: the handler side (the provider request
/// boundary), or the host.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    /// Handler-side: the provider request boundary.
    Handler,
    /// The application's own policies and state.
    Host,
}

/// Stable emitter name and optional version.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Emitter {
    /// The emitter's stable name (a layer's name, a host
    /// system's name).
    pub name: String,
    /// Its declared version, when it has one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}

impl Emitter {
    /// A named, unversioned emitter.
    pub fn named(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: None,
        }
    }

    /// A named, versioned emitter.
    pub fn versioned(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: Some(version.into()),
        }
    }
}

/// A structured reason: a stable code and an optional free-text detail.
/// The code is what a comparison keys on; the detail is for a reader.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Reason {
    /// A stable, machine-readable code, such as `incomplete_expected`.
    pub code: String,
    /// What a reader wants to know.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl Reason {
    /// A reason with a code and a detail.
    pub fn with_detail(code: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            detail: Some(detail.into()),
        }
    }
}

/// The fact: one the provider request boundary saw, or a host's own
/// ([`Action::Host`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum Action {
    /// A fact emitted by the provider request boundary.
    Adapter {
        /// Correlation and typed boundary metadata.
        observation: AdapterObservation,
    },
    /// A host policy's own fact: named by its kind, carried as its serde
    /// payload. Build one through [`HostAction`].
    Host {
        /// The host action's declared kind.
        kind: String,
        /// Its payload.
        payload: serde_json::Value,
    },
}

/// A host-defined action: a named, serde-typed fact a host policy emits
/// through [`Action::Host`]. The kind is declared once per type, so a
/// trace names every host fact and a consumer deserializes it back.
pub trait HostAction: Serialize + serde::de::DeserializeOwned {
    /// The stable kind name (`rigcoder/approval`).
    const KIND: &'static str;

    /// This fact as an [`Action::Host`]; an unserializable fact is an error,
    /// never a silent omission.
    fn action(&self) -> Result<Action, serde_json::Error> {
        Ok(Action::Host {
            kind: Self::KIND.to_owned(),
            payload: serde_json::to_value(self)?,
        })
    }

    /// The fact back out of an [`Action::Host`] of this kind.
    fn from_action(action: &Action) -> Option<Result<Self, serde_json::Error>> {
        match action {
            Action::Host { kind, payload } if kind == Self::KIND => {
                Some(serde_json::from_value(payload.clone()))
            }
            _ => None,
        }
    }
}

/// A host-owned monotonic clock: what a sink stamps [`Observation::at`]
/// from. rig-core reads no clock itself; a test supplies a counter.
pub trait Clock: WasmCompatSend + WasmCompatSync {
    /// Elapsed time since the clock's origin.
    fn elapsed(&self) -> Duration;
}

/// Where observations go: the seam a driver and a host emit through. A
/// witness is shared, so it takes `&self`; it must never block the caller.
pub trait Witness: WasmCompatSend + WasmCompatSync + 'static {
    /// One fact. The sink assigns its sequence.
    fn observe(&self, observation: Observation);
}

/// The serializable trace a sink produces: the analysis artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservationTrace {
    /// The sink's declared session, when it has one (a run's scope, a
    /// trial id). Lineage, not semantics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session: Option<String>,
    /// The facts, in sequence.
    pub observations: Vec<Observation>,
    /// Facts discarded because sink capacity was reached. Nonzero means the
    /// trace is incomplete.
    #[serde(default)]
    pub dropped: u64,
    /// Whether the sink was told the session finished normally.
    #[serde(default)]
    pub finalized: bool,
}

impl ObservationTrace {
    /// Whether every fact the session produced is here.
    pub fn is_complete(&self) -> bool {
        self.dropped == 0
    }
}

/// The bounded in-memory sink, in one of two shapes. A capture
/// ([`Self::with_capacity`], the default) keeps the first `capacity` facts
/// and counts the rest as dropped: the trace is a complete prefix or says
/// how much of the session it missed, which is what a comparison wants. A
/// ring ([`Self::ring`]) keeps the *last* `capacity` facts and counts what
/// it let go: a long-running host always sees its recent past, which is
/// what a dashboard wants. Either way [`Self::drain`] takes what is kept
/// and starts over, so a host that exports periodically never fills up.
pub struct ObservationLog {
    inner: Mutex<LogState>,
    capacity: usize,
    ring: bool,
    clock: Option<Arc<dyn Clock + Send + Sync>>,
}

struct LogState {
    session: Option<String>,
    observations: std::collections::VecDeque<Observation>,
    next: u64,
    dropped: u64,
    finalized: bool,
}

impl std::fmt::Debug for ObservationLog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.lock();
        f.debug_struct("ObservationLog")
            .field("observations", &state.observations.len())
            .field("dropped", &state.dropped)
            .field("capacity", &self.capacity)
            .finish_non_exhaustive()
    }
}

/// The default capacity of an [`ObservationLog`].
pub const DEFAULT_CAPACITY: usize = 65_536;

impl Default for ObservationLog {
    fn default() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }
}

impl ObservationLog {
    /// A capture keeping the first `capacity` facts; later ones are counted
    /// as dropped.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(LogState {
                session: None,
                observations: std::collections::VecDeque::new(),
                next: 0,
                dropped: 0,
                finalized: false,
            }),
            capacity,
            ring: false,
            clock: None,
        }
    }

    /// Keeps the last `capacity` facts and counts evictions as dropped.
    /// Capacity zero is raised to one.
    pub fn ring(capacity: usize) -> Self {
        Self {
            ring: true,
            // A ring of nothing would keep every fact and count it dropped.
            ..Self::with_capacity(capacity.max(1))
        }
    }

    /// Take the kept facts and start over: the session name and the
    /// sequence continue, the kept facts and the dropped count reset. A
    /// host that exports its trace periodically drains rather than
    /// letting a capture fill and stop.
    pub fn drain(&self) -> ObservationTrace {
        let mut state = self.lock();
        let trace = ObservationTrace {
            session: state.session.clone(),
            observations: state.observations.drain(..).collect(),
            dropped: state.dropped,
            finalized: state.finalized,
        };
        state.dropped = 0;
        trace
    }

    /// Stamp every fact with `clock`'s elapsed time.
    pub fn with_clock(mut self, clock: Arc<dyn Clock + Send + Sync>) -> Self {
        self.clock = Some(clock);
        self
    }

    /// Name the session the trace belongs to.
    pub fn with_session(self, session: impl Into<String>) -> Self {
        self.lock().session = Some(session.into());
        self
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, LogState> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The session finished normally: later readers know the trace is not
    /// a partial artifact of a killed process. A later fact reopens the capture.
    /// Hosts must drain their producers before exporting a finalized snapshot.
    pub fn finalize(&self) {
        self.lock().finalized = true;
    }

    /// The facts so far.
    pub fn trace(&self) -> ObservationTrace {
        let state = self.lock();
        ObservationTrace {
            session: state.session.clone(),
            observations: state.observations.iter().cloned().collect(),
            dropped: state.dropped,
            finalized: state.finalized,
        }
    }

    /// How many facts are kept.
    pub fn len(&self) -> usize {
        self.lock().observations.len()
    }

    /// Whether no observations are currently retained.
    pub fn is_empty(&self) -> bool {
        self.lock().observations.is_empty()
    }
}

impl Witness for ObservationLog {
    fn observe(&self, mut observation: Observation) {
        let at = self.clock.as_ref().map(|clock| clock.elapsed());
        let mut state = self.lock();
        state.finalized = false;
        observation.seq = state.next;
        state.next += 1;
        if state.observations.len() >= self.capacity {
            state.dropped += 1;
            if !self.ring {
                return;
            }
            state.observations.pop_front();
        }
        observation.at = at;
        state.observations.push_back(observation);
    }
}

impl<W: Witness + ?Sized> Witness for Arc<W> {
    fn observe(&self, observation: Observation) {
        (**self).observe(observation);
    }
}

// A trace serializes and crosses threads on every target.
const _: fn() = || {
    fn assert_wire<T: Clone + Send + Sync + 'static + Serialize + serde::de::DeserializeOwned>() {}
    assert_wire::<Observation>();
    assert_wire::<ObservationTrace>();
};
