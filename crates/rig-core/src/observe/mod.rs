//! Observations: what a runtime, a policy or a host *decided* or *saw* at a
//! boundary, as typed data beside — never inside — the exchange record.
//!
//! An [`EffectRecord`](crate::effect::EffectRecord) is the replay oracle: the
//! request a handler served and what it answered. Everything else worth
//! knowing after a failure is program, not record — a gate held a call
//! and later denied it, a judge replaced an
//! answer, a stream ended before its terminal record, a run was cancelled
//! with a tool in flight — and today it is either gone by the time a
//! component is inspected, or scattered over log lines. An [`Observation`]
//! is one such fact at the moment it happened: its [`Subject`] (which
//! effect, in which program scope, at which dispatch order), the
//! [`Stage`] of the pipeline that saw it, the [`Emitter`] that owns the
//! decision, and the typed [`Action`] with its outcome data or a
//! structured [`Reason`].
//!
//! The contract is runtime-neutral. rig-core names the vocabulary and the
//! [`Witness`] seam a sink implements; a driver (the ECS bus, an agent
//! loop) feeds it at its own decision sites; a host feeds it from its own
//! policies through [`Action::Host`] and the [`HostAction`] trait, so a
//! domain-specific fact travels typed and named, not as an anonymous blob.
//! [`ObservationLog`] is the bounded in-memory sink; its
//! [`ObservationTrace`] is the serializable analysis artifact, distinct
//! from an effect log and never part of replay identity.
//!
//! Measurements are separate from semantics: an observation may carry
//! [`Observation::at`], an elapsed duration from a host-owned [`Clock`],
//! and consumers decide how to compare it. Nothing here reads a clock or a random
//! source on its own.

use std::{
    sync::{Arc, Mutex, PoisonError},
    time::Duration,
};

use serde::{Deserialize, Serialize};

use crate::{
    effect::{EffectFamily, EffectId, HandlerKey, Outcome},
    error::ErrorReport,
    streaming::StreamEvent,
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};

mod adapter;
pub(crate) use adapter::AdapterSlot;
pub use adapter::{
    AdapterAnalysis, AdapterContext, AdapterEnding, AdapterErrorBoundary, AdapterErrorEnvelope,
    AdapterEvent, AdapterObservation, AdapterUsage, AdapterVerdict, diagnostic_url_secrets,
    scrub_diagnostic,
};
pub(crate) use adapter::{AdapterAttempt, PayloadObserver};

#[cfg(test)]
mod tests;

/// One observed fact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Observation {
    /// The position in its trace: assigned by the sink, in observation
    /// order. Meaningful order — two traces that disagree on it diverged.
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

/// Where in the pipeline a fact was seen. The bus's four sets, the agent
/// runtime's sets folded into one, the handler side, and the host.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    /// Before dispatch: a policy held, released or denied an intent.
    Gate,
    /// The driver took or refused an intent.
    Dispatch,
    /// Inside the handler side: a layer's decision on the way in or out.
    /// (`Handler`, not `Serve`: a variant named `Serve` would make the
    /// trait's name ambiguous in compiler diagnostics.)
    Handler,
    /// The driver landed what a handler produced.
    Collect,
    /// After the record: a policy replaced an answer.
    Judge,
    /// The agent runtime: run endings.
    Runtime,
    /// The application's own policies and state.
    Host,
}

/// Who owns a decision or an observation: a stable name and, when the
/// emitter versions itself, a version. [`Emitter::unknown`] is the explicit
/// value for a fact the runtime saw land without knowing which policy made
/// it — it is never inferred from the outcome.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Emitter {
    /// The emitter's stable name (`rig-ecs/bus`, a layer's name, a host
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

    /// The fact landed; the runtime does not know which policy made it.
    /// The name `unknown` is reserved for this: a host emitter must name
    /// itself otherwise.
    pub fn unknown() -> Self {
        Self::named("unknown")
    }

    /// Whether this is the explicit unknown emitter (by its reserved name).
    pub fn is_unknown(&self) -> bool {
        self.name == "unknown"
    }
}

/// A structured reason: a stable code and an optional free-text detail.
/// The code is what a comparison keys on; the detail is for a reader.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Reason {
    /// A stable, machine-readable code: an [`crate::error::ErrorKind::code`]
    /// for a fact carrying a report, or an emitter's own (`intake_bound`,
    /// `serial_key_busy`, `reentrant`, `ids_exhausted`, `layer_discarded`,
    /// `despawned_before_dispatch`, `never_served`, `settled`, `max_turns`,
    /// …).
    pub code: String,
    /// What a reader wants to know.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl Reason {
    /// A reason with a code and no detail.
    pub fn code(code: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            detail: None,
        }
    }

    /// A reason with a code and a detail.
    pub fn with_detail(code: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            detail: Some(detail.into()),
        }
    }

    /// The reason an error report carries: its kind's stable code
    /// ([`crate::error::ErrorKind::code`]), its message as the detail.
    pub fn from_report(report: &ErrorReport) -> Self {
        Self::with_detail(report.kind.code(), report.message.clone())
    }

    /// The explicit unknown reason.
    pub fn unknown() -> Self {
        Self::code("unknown")
    }
}

/// The compact form of an outcome an observation carries: enough to
/// classify without duplicating the exchange record that holds the value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum OutcomeSummary {
    /// The handler answered with an outcome of this family.
    Ok {
        /// The answer's family.
        family: EffectFamily,
    },
    /// The handler (or a decision) answered with this report.
    Err {
        /// The report's kind and message.
        reason: Reason,
        /// Whether the report says a retry may succeed.
        retryable: bool,
    },
}

impl OutcomeSummary {
    /// The summary of an outcome.
    pub fn of(outcome: &Result<Outcome, ErrorReport>) -> Self {
        match outcome {
            Ok(outcome) => Self::Ok {
                family: outcome.family(),
            },
            Err(report) => Self::Err {
                reason: Reason::from_report(report),
                retryable: report.retryable,
            },
        }
    }
}

/// The fact. Every variant carries its own before/after data or reason;
/// a host adds its own through [`Action::Host`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum Action {
    /// A fact emitted by the provider request boundary.
    Adapter {
        /// Correlation and typed boundary metadata.
        observation: AdapterObservation,
    },
    /// A pending intent was held before dispatch.
    Held {
        /// Why, when the holder said.
        reason: Reason,
    },
    /// A held intent was released to dispatch.
    Released,
    /// An intent was denied before any handler served it.
    Denied {
        /// The report the consumer receives.
        reason: Reason,
    },
    /// The driver took an intent: the id it issued.
    Issued,
    /// The driver refused an intent before any handler: no record.
    Refused {
        /// Why: `handler_unavailable`, `reentrant` or `ids_exhausted`, with
        /// the report's message as the detail.
        reason: Reason,
    },
    /// An outcome landed and the record closed.
    Landed {
        /// The outcome, in brief.
        outcome: OutcomeSummary,
    },
    /// A stream ended before its terminal record.
    StreamTruncated {
        /// Items the consumer had received.
        delivered: usize,
        /// The last events seen, bounded, for after-the-fact classification.
        tail: Vec<StreamEvent>,
        /// Error items seen in the stream, if any.
        errors: Vec<Reason>,
    },
    /// An answer was replaced after the record closed.
    Replaced {
        /// What the record holds.
        recorded: OutcomeSummary,
        /// What the consumer received.
        consumed: OutcomeSummary,
    },
    /// An in-flight dispatch was cancelled.
    Cancelled {
        /// Why.
        reason: Reason,
    },
    /// A program ended.
    Ended {
        /// How (`settled`, `max_turns`, `provider`, `cancelled`, …).
        ending: Reason,
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

/// The most bytes a truncation tail may hold in its serialized form. A larger value is elided to a
/// marker naming its size: the sink is bounded by count, this bounds each
/// entry, and the exchange record still holds the full value.
pub const LARGEST_PAYLOAD_BYTES: usize = 64 * 1024;

impl Action {
    /// A truncation observation whose `tail` is cut from the front until it
    /// fits [`LARGEST_PAYLOAD_BYTES`]; `delivered` and `errors` are kept.
    pub fn stream_truncated(
        delivered: usize,
        mut tail: Vec<StreamEvent>,
        errors: Vec<Reason>,
    ) -> Self {
        while !tail.is_empty()
            && serde_json::to_vec(&tail).map_or(usize::MAX, |bytes| bytes.len())
                > LARGEST_PAYLOAD_BYTES
        {
            tail.remove(0);
        }
        Self::StreamTruncated {
            delivered,
            tail,
            errors,
        }
    }
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
    /// Facts the sink could not keep: its capacity was reached. A trace
    /// with dropped facts is incomplete and never compares equal.
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

/// The bounded in-memory sink. Facts beyond `capacity` are counted, not
/// kept, so an analysis sink never grows without bound; the trace says
/// how many were lost.
pub struct ObservationLog {
    inner: Mutex<LogState>,
    capacity: usize,
    clock: Option<Arc<dyn Clock + Send + Sync>>,
}

struct LogState {
    session: Option<String>,
    observations: Vec<Observation>,
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
    /// A log keeping at most `capacity` facts.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(LogState {
                session: None,
                observations: Vec::new(),
                next: 0,
                dropped: 0,
                finalized: false,
            }),
            capacity,
            clock: None,
        }
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
            observations: state.observations.clone(),
            dropped: state.dropped,
            finalized: state.finalized,
        }
    }

    /// How many facts are kept.
    pub fn len(&self) -> usize {
        self.lock().observations.len()
    }

    /// Whether nothing was observed.
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
            return;
        }
        observation.at = at;
        state.observations.push(observation);
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
