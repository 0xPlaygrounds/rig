//! Effect log headers, checkpoint envelopes, and stable JSON hashing.
//!
//! ```
//! let hash = rig_cassette::effect_log::stable_hash(&"run")?;
//! # Ok::<(), serde_json::Error>(())
//! ```

use std::collections::BTreeMap;

use rig_core::effect::{EffectId, EffectRecord, EffectRow, HandlerDescriptor};
use rig_core::error::{ErrorKind, ErrorReport};
use rig_core::json_utils::to_canonical_string;
use serde::{Deserialize, Serialize};

/// The checkpoint envelope format this crate writes and reads. This versions
/// [`Checkpoint`], not [`LogHeader`]; logs have no global format number.
pub const CHECKPOINT_FORMAT: u32 = 7;

/// Recorded run identity and handler declarations used to validate replay
/// compatibility. Deserialization rejects unknown fields; headers have no
/// global format number.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LogHeader {
    /// Error items omitted from `EffectRecord::events`, at their original
    /// positions among all stream items. Empty for streams without errors.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub stream_errors: BTreeMap<EffectId, Vec<RecordedStreamError>>,
    /// A hash of the run spec the run was recorded under, when an agent
    /// recorded it (`None` for a record made without an agent). An agent that replays
    /// compares it with its own.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_spec: Option<u64>,
    /// The handlers registered when recording began, stamped
    /// with their keys.
    pub handlers: Vec<HandlerDescriptor>,
    /// The effect signature: which keys the run performed effects on, and
    /// of which family, derived from the trace.
    pub signature: EffectRow,
    /// Ordered hook type names, with nested stacks flattened.
    /// Replay requires the same stack because hooks execute again.
    pub hooks: Vec<String>,
    /// The program's required effect row at record time: every key it could
    /// dispatch to (its model, its tools, its retrieval indexes) with the
    /// family it needs. A replay checks this row against
    /// what the log's handlers serve, not only against what happened to be
    /// dispatched.
    pub required: EffectRow,
    /// Required effects and policy hashes per `EffectRecord::scope`, for a
    /// log several programs write into. A single agent's log uses
    /// `run_spec` and `required` instead, and has no `programs` field.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub programs: BTreeMap<String, ProgramIdentity>,
}

/// One program's identity in a shared log: its required effect row and
/// the stable hash of its policy (what `run_spec` is for a single agent).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramIdentity {
    /// Every key the program could dispatch to, with the family it needs.
    pub required: EffectRow,
    /// [`stable_hash`] of the program's policy. Replay never compares it:
    /// the host compares it with its program's own before it replays.
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

impl LogHeader {
    /// Describes each of `handlers`: one already described in place, a new
    /// one after the others.
    pub fn describe(&mut self, handlers: impl IntoIterator<Item = HandlerDescriptor>) {
        for handler in handlers {
            match self
                .handlers
                .iter_mut()
                .find(|known| known.key == handler.key)
            {
                Some(known) => *known = handler,
                None => self.handlers.push(handler),
            }
        }
    }

    /// Names `record`'s key, with its family, in the signature, unless it
    /// is named.
    pub fn sign(&mut self, record: &EffectRecord) {
        let family = record.kind.family();
        self.signature.insert_if_absent(record.key.clone(), family);
    }
}

/// A recorded run with metadata and exchanges in dispatch order.
/// Dereferences to the record slice and supports owned or borrowed iteration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
        records.iter().for_each(|record| header.sign(record));
        Self { header, records }
    }

    /// Return records from `at` onward with cloned metadata restricted to those
    /// records. An out-of-range position yields an empty tail.
    pub fn tail(&self, at: usize) -> Self {
        let mut tail = Self {
            header: self.header.clone(),
            records: self.records.get(at..).unwrap_or_default().to_vec(),
        };
        tail.retain_recorded_errors();
        tail
    }

    /// A checkpoint holding `state`, position `at`, the highest id before
    /// it and the id of the record after it, together with the rest of the
    /// log. A position at or past the end gives an empty tail.
    pub fn checkpoint<S>(&self, at: usize, state: S) -> (Checkpoint<S>, Self) {
        let checkpoint = Checkpoint {
            format: CHECKPOINT_FORMAT,
            at,
            last: self.records.iter().take(at).map(|record| record.id).max(),
            next: self.records.get(at).map(|record| record.id),
            state,
        };
        (checkpoint, self.tail(at))
    }

    /// `tail`, once it is checked to follow `checkpoint`.
    ///
    /// # Errors
    ///
    /// `Divergence` when the checkpoint's format is not
    /// [`CHECKPOINT_FORMAT`], or the tail does not begin with the record the
    /// checkpoint names next (a checkpoint that ends the log needs an empty
    /// tail).
    pub fn from_checkpoint<S>(checkpoint: &Checkpoint<S>, tail: Self) -> Result<Self, ErrorReport> {
        if checkpoint.format != CHECKPOINT_FORMAT {
            return Err(ErrorReport::new(
                ErrorKind::Divergence,
                format!(
                    "resume refused: the checkpoint is format {}, this rig reads format {CHECKPOINT_FORMAT}",
                    checkpoint.format
                ),
            ));
        }
        let at = checkpoint.at;
        let refusal = match (
            checkpoint.next,
            tail.records.first().map(|record| record.id),
        ) {
            (Some(next), Some(first)) if next == first => return Ok(tail),
            (None, None) => return Ok(tail),
            (Some(next), Some(first)) => format!(
                "resume refused: the checkpoint at {at} expects record {next} next, the tail begins at {first}"
            ),
            (Some(next), None) => format!(
                "resume refused: the checkpoint at {at} expects record {next} next, the tail is empty"
            ),
            (None, Some(first)) => format!(
                "resume refused: the checkpoint at {at} ends the log, the tail begins at {first}"
            ),
        };
        Err(ErrorReport::new(ErrorKind::Divergence, refusal))
    }

    /// Drop stream errors of effects outside this log's records. Used for
    /// snapshots and tails of a shared recorder.
    pub(crate) fn retain_recorded_errors(&mut self) {
        let ids: std::collections::BTreeSet<_> =
            self.records.iter().map(|record| record.id).collect();
        self.header.stream_errors.retain(|id, _| ids.contains(id));
    }
}

/// Where a host's own runtime stopped in a log: the position, the id of
/// the record it expects next, and the state it persists, in its own type.
///
/// The log alone does not restore an application: a host saves its state
/// here beside the log, and on resume restores that state, checks the tail
/// with [`EffectLog::from_checkpoint`], numbers its dispatches after
/// [`last`](Self::last), and replays the tail.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Checkpoint<S> {
    /// The checkpoint envelope format ([`CHECKPOINT_FORMAT`]).
    pub format: u32,
    /// The position in the log: `at` records were performed before it.
    pub at: usize,
    /// The highest id among the records before `at`; `None` when there
    /// are none. A resumed host numbers its dispatches after it (rig-agent's
    /// `Handlers::resume_ids_after`), so a by-id replay of the tail
    /// ([`EffectLogReplayer::for_log_by_id`](super::EffectLogReplayer::for_log_by_id))
    /// meets the ids it recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last: Option<EffectId>,
    /// The id of the record the tail begins with; `None` when the
    /// checkpoint ends the log.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next: Option<EffectId>,
    /// What the host persists, in its own type.
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

/// Return the 64-bit FNV-1a hash of `value` serialized as JSON with recursively
/// sorted object keys, or a serialization error.
/// Sorting makes identity independent of JSON map insertion order.
pub fn stable_hash<T: Serialize>(value: &T) -> Result<u64, serde_json::Error> {
    let json = to_canonical_string(&serde_json::to_value(value)?);
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in json.into_bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    Ok(hash)
}

/// `log` with each rig-issued call id (a random uuid) renamed, in order of
/// first appearance, to a fixed one. Rig issues its ids afresh on every run,
/// so two runs that differ only in them record and replay the same log.
///
/// ```
/// let issued = serde_json::json!({"local": "7ecaf57b-5bbb-4b9e-a440-63738a5939ec"});
/// assert_eq!(
///     rig_cassette::effect_log::canonical_local_ids(issued),
///     serde_json::json!({"local": "00000000-0000-4000-8000-000000000000"}),
/// );
/// ```
pub fn canonical_local_ids(mut log: serde_json::Value) -> serde_json::Value {
    use serde_json::Value;
    fn issued(value: &Value, ids: &mut Vec<String>) {
        match value {
            Value::Object(object) => {
                if let (1, Some(Value::String(id))) = (object.len(), object.get("local"))
                    && !ids.contains(id)
                {
                    ids.push(id.clone());
                }
                object.values().for_each(|value| issued(value, ids));
            }
            Value::Array(values) => values.iter().for_each(|value| issued(value, ids)),
            _ => {}
        }
    }
    fn rename(value: &mut Value, ids: &[String], order: &mut Vec<String>) {
        match value {
            Value::String(text) => {
                // Mark every id first, so a fixed id that is also an issued
                // one is never renamed twice. A string without one is left.
                let (mut marked, mut hits) = (None, Vec::new());
                for id in ids.iter().filter(|id| text.contains(id.as_str())) {
                    let index = order.iter().position(|seen| seen == id).unwrap_or_else(|| {
                        order.push(id.clone());
                        order.len() - 1
                    });
                    let into: &mut String = marked.get_or_insert_with(|| text.clone());
                    *into = into.replace(id.as_str(), &format!("\u{0}{index}\u{0}"));
                    hits.push(index);
                }
                if let Some(mut marked) = marked {
                    for index in hits {
                        marked = marked.replace(
                            &format!("\u{0}{index}\u{0}"),
                            &format!("00000000-0000-4000-8000-{index:012}"),
                        );
                    }
                    *text = marked;
                }
            }
            Value::Object(object) => object
                .values_mut()
                .for_each(|value| rename(value, ids, order)),
            Value::Array(values) => values
                .iter_mut()
                .for_each(|value| rename(value, ids, order)),
            _ => {}
        }
    }
    let mut ids = Vec::new();
    issued(&log, &mut ids);
    rename(&mut log, &ids, &mut Vec::new());
    log
}

/// A tool call's JSON arguments with object keys sorted, or the text as is
/// when it is not JSON. Two argument strings that differ only in key order
/// describe the same call; which order a build emits depends on whether
/// `serde_json`'s `preserve_order` feature is unified into it.
pub(crate) fn canonical_tool_args(args: &str) -> String {
    serde_json::from_str::<serde_json::Value>(args)
        .map(|value| to_canonical_string(&value))
        .unwrap_or_else(|_| args.to_owned())
}

// A log serializes and crosses threads on every target.
const _: fn() = || {
    fn assert_wire<T: Clone + Send + Sync + 'static + Serialize + serde::de::DeserializeOwned>() {}
    assert_wire::<LogHeader>();
    assert_wire::<EffectLog>();
    assert_wire::<Checkpoint<serde_json::Value>>();
};

#[cfg(test)]
mod stable_hash_tests;

#[cfg(test)]
mod canonical_args_tests;
