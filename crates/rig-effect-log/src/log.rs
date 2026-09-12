//! The log: its header and its records.

use std::collections::BTreeMap;

use rig_core::effect::{EffectId, EffectRecord, EffectRow, HandlerDescriptor};
use rig_core::error::{ErrorKind, ErrorReport};
use rig_core::serve::ServingPolicy;
use serde::{Deserialize, Serialize};

/// The checkpoint envelope format this crate writes and reads. This versions
/// [`Checkpoint`], not [`LogHeader`]; logs have no global format number.
pub const CHECKPOINT_FORMAT: u32 = 6;

/// What a log says about the run it records, so a replay can refuse a log
/// the program has outgrown before the first dispatch diverges.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogHeader {
    /// Consumer-visible deliveries, in observation order, when the runtime
    /// records schedule boundaries. `None` supplies no delivery guarantee.
    /// A batch groups transitions observed in one pass; it is not a clock.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deliveries: Option<Vec<rig_core::effect::Delivery>>,
    /// Reasons this recording cannot establish policy-visible delivery.
    /// Exchange replay remains possible; exact policy mode must refuse it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub delivery_limitations: Vec<String>,
    /// Error items omitted from `EffectRecord::events`, at their original
    /// positions among all stream items. Empty for streams without errors.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub stream_errors: BTreeMap<EffectId, Vec<RecordedStreamError>>,
    /// A hash of the run spec the run was recorded under, when an agent
    /// recorded it (`None` for a bare-bus record). An agent that replays
    /// compares it with its own.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_spec: Option<u64>,
    /// The handlers registered on the bus when recording began, stamped
    /// with their keys.
    #[serde(default)]
    pub handlers: Vec<HandlerDescriptor>,
    /// The effect signature: which keys the run performed effects on, and
    /// of which family — the effect row read off the trace.
    #[serde(default)]
    pub signature: EffectRow,
    /// The program's hook stack at record time: the ordered type names of
    /// every hook (nested stacks flattened). Hooks are program, not record —
    /// a hook's decision is re-made on replay — so a log replayed under
    /// another stack is another program, and the agent refuses it.
    #[serde(default)]
    pub hooks: Vec<String>,
    /// The program's required effect row at record time: every key it could
    /// dispatch to (its model, its tools, its memory, its retrieval
    /// indexes) with the family it needs. A replay checks this row against
    /// what the log's handlers serve, not only against what happened to be
    /// dispatched.
    #[serde(default)]
    pub required: EffectRow,
    /// The serving policy the run was recorded under. Per-key order is
    /// dispatch order under either policy; the header says which so a
    /// replay under a different one is a stated choice, not a surprise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bus: Option<ServingPolicy>,
    /// Program identity as data, per scope (`EffectRecord::scope`): what
    /// each program that wrote to this log could dispatch to and the hash
    /// of its policy. Written by a world that runs several programs in one
    /// log; absent from a log one agent wrote (`run_spec` and `required`
    /// are that agent's), so nothing re-stamps.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub programs: BTreeMap<String, ProgramIdentity>,
}

/// One program's identity in a shared log: its required effect row and
/// the stable hash of its policy (what `run_spec` is for a single agent).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgramIdentity {
    /// Every key the program could dispatch to, with the family it needs.
    pub required: EffectRow,
    /// [`stable_hash`] of the program's policy.
    pub policy: u64,
}

/// An error's place in a kept stream, including errors after a terminal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordedStreamError {
    /// Zero-based position among successful events and error items together.
    pub item: usize,
    /// The original error item; it need not be the stream's folded outcome.
    pub error: ErrorReport,
}

impl Default for LogHeader {
    fn default() -> Self {
        Self {
            deliveries: None,
            delivery_limitations: Vec::new(),
            stream_errors: BTreeMap::new(),
            run_spec: None,
            handlers: Vec::new(),
            signature: EffectRow::new(),
            hooks: Vec::new(),
            required: EffectRow::new(),
            bus: None,
            programs: BTreeMap::new(),
        }
    }
}

/// A recorded run: its header, then every exchange in dispatch order.
/// Derefs to the records, so `log[i]`, `log.len()` and iteration read as
/// they did when the log was a plain vector.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EffectLog {
    /// What the log says about the run.
    pub header: LogHeader,
    /// The exchanges, in dispatch order.
    pub records: Vec<EffectRecord>,
}

impl EffectLog {
    /// A log over `records` with a default header (no spec, no handlers,
    /// the signature read off the records).
    pub fn from_records(records: Vec<EffectRecord>) -> Self {
        let mut header = LogHeader::default();
        for record in &records {
            header
                .signature
                .insert_if_absent(record.key.clone(), record.kind.family());
        }
        Self { header, records }
    }

    /// The records from `at` on, under a copy of this header — the
    /// continuation a resumed run replays.
    pub fn tail(&self, at: usize) -> Self {
        let mut tail = Self {
            header: self.header.clone(),
            records: self.records.get(at..).unwrap_or_default().to_vec(),
        };
        tail.retain_recorded_deliveries();
        tail
    }

    /// Drop delivery entries outside this log's records, preserving order and
    /// batch identities. Used for snapshots and tails of a shared recorder.
    pub(crate) fn retain_recorded_deliveries(&mut self) {
        let ids: std::collections::BTreeSet<_> =
            self.records.iter().map(|record| record.id).collect();
        self.header.stream_errors.retain(|id, _| ids.contains(id));
        if let Some(deliveries) = &mut self.header.deliveries {
            deliveries.retain(|delivery| ids.contains(&delivery.id));
        }
    }

    /// Cut the log at `at`: a [`Checkpoint`] naming the position, the id of
    /// the record that follows it, and `state` — what the driver persists,
    /// in its own type (a world's scene; the frozen engine's serialized
    /// run as JSON) — beside the tail the continuation replays. `at` past
    /// the end is a checkpoint with an empty tail.
    pub fn checkpoint<S>(&self, at: usize, state: S) -> (Checkpoint<S>, Self) {
        let checkpoint = Checkpoint {
            format: CHECKPOINT_FORMAT,
            at,
            next: self.records.get(at).map(|record| record.id),
            state,
        };
        (checkpoint, self.tail(at))
    }

    /// The continuation `checkpoint` names, over `tail`: refused by name
    /// when the checkpoint is of another format, when the tail's first
    /// record is not the one the checkpoint expects next (ids are total, so
    /// a tail begins exactly at the checkpoint's next id).
    pub fn from_checkpoint<S>(checkpoint: &Checkpoint<S>, tail: Self) -> Result<Self, ErrorReport> {
        if checkpoint.format != CHECKPOINT_FORMAT {
            return Err(ErrorReport::new(
                ErrorKind::Internal,
                format!(
                    "resume refused: the checkpoint is format {}, this rig reads format {}",
                    checkpoint.format, CHECKPOINT_FORMAT
                ),
            ));
        }
        let first = tail.records.first().map(|record| record.id);
        if first != checkpoint.next {
            return Err(ErrorReport::new(
                ErrorKind::Internal,
                match (checkpoint.next, first) {
                    (Some(next), Some(first)) => format!(
                        "resume refused: the checkpoint at {} expects record {next} next, the tail begins at {first}",
                        checkpoint.at
                    ),
                    (Some(next), None) => format!(
                        "resume refused: the checkpoint at {} expects record {next} next, the tail is empty",
                        checkpoint.at
                    ),
                    (None, Some(first)) => format!(
                        "resume refused: the checkpoint at {} ends the log, the tail begins at {first}",
                        checkpoint.at
                    ),
                    (None, None) => unreachable_refusal(),
                },
            ));
        }
        Ok(tail)
    }
}

/// `(None, None)` is the equal case, taken by the comparison above; this
/// arm types the match.
fn unreachable_refusal() -> String {
    "resume refused: the tail does not follow the checkpoint".to_owned()
}

/// A cut in a log: where a run was suspended, what follows, and what the
/// driver persisted — beside the tail, so a resumed run replays only what
/// it has not performed, and a full log offered in a tail's place is
/// refused by its first id. `S` is the driver's state in its own type: a
/// world's scene is a scene here, not a blob; only the frozen engine's
/// serialized run is a `serde_json::Value`, until it retires.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Checkpoint<S> {
    /// The checkpoint envelope format ([`CHECKPOINT_FORMAT`]).
    pub format: u32,
    /// The position in the log: `at` records were performed before it.
    pub at: usize,
    /// The id of the record the tail begins with; `None` when the
    /// checkpoint ends the log.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next: Option<EffectId>,
    /// What the driver persists, in its own type.
    pub state: S,
}

impl std::ops::Deref for EffectLog {
    type Target = [EffectRecord];

    fn deref(&self) -> &[EffectRecord] {
        &self.records
    }
}

impl From<Vec<EffectRecord>> for EffectLog {
    fn from(records: Vec<EffectRecord>) -> Self {
        Self::from_records(records)
    }
}

impl FromIterator<EffectRecord> for EffectLog {
    fn from_iter<I: IntoIterator<Item = EffectRecord>>(records: I) -> Self {
        Self::from_records(records.into_iter().collect())
    }
}

impl IntoIterator for EffectLog {
    type Item = EffectRecord;
    type IntoIter = std::vec::IntoIter<EffectRecord>;

    fn into_iter(self) -> Self::IntoIter {
        self.records.into_iter()
    }
}

impl<'a> IntoIterator for &'a EffectLog {
    type Item = &'a EffectRecord;
    type IntoIter = std::slice::Iter<'a, EffectRecord>;

    fn into_iter(self) -> Self::IntoIter {
        self.records.iter()
    }
}

/// A stable 64-bit hash of `value`'s JSON form (FNV-1a over the bytes of
/// its canonical rendering): the same on every platform, toolchain and
/// build, unlike `std`'s hasher. What [`LogHeader::run_spec`] holds.
///
/// Canonical means every object's keys are sorted before the bytes are
/// hashed. `serde_json` keeps insertion order in a build that enables its
/// `preserve_order` feature (the root `rig` package with every feature on
/// does, through a dependency) and sorts otherwise, so a hash over the
/// raw serialization of a spec holding a `serde_json::Value` — an
/// `additional_params`, an `output_schema` — differed between the crate
/// that recorded a golden and the crate that replays it. The program's
/// identity cannot depend on which crate computes it.
pub fn stable_hash<T: Serialize>(value: &T) -> Result<u64, serde_json::Error> {
    let json = serde_json::to_vec(&Canonical::from(serde_json::to_value(value)?))?;
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in json {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    Ok(hash)
}

/// A JSON value whose objects serialize with sorted keys whatever
/// `serde_json`'s map type is.
#[derive(Serialize)]
#[serde(untagged)]
enum Canonical {
    Null,
    Bool(bool),
    Number(serde_json::Number),
    String(String),
    Array(Vec<Canonical>),
    Object(BTreeMap<String, Canonical>),
}

impl From<serde_json::Value> for Canonical {
    fn from(value: serde_json::Value) -> Self {
        match value {
            serde_json::Value::Null => Self::Null,
            serde_json::Value::Bool(bool) => Self::Bool(bool),
            serde_json::Value::Number(number) => Self::Number(number),
            serde_json::Value::String(string) => Self::String(string),
            serde_json::Value::Array(items) => {
                Self::Array(items.into_iter().map(Self::from).collect())
            }
            serde_json::Value::Object(fields) => Self::Object(
                fields
                    .into_iter()
                    .map(|(key, value)| (key, Self::from(value)))
                    .collect(),
            ),
        }
    }
}

// A log serializes and crosses threads on every target.
const _: fn() = || {
    fn assert_wire<T: Clone + Send + Sync + 'static + Serialize + serde::de::DeserializeOwned>() {}
    assert_wire::<LogHeader>();
    assert_wire::<EffectLog>();
};

#[cfg(test)]
mod stable_hash_tests;
