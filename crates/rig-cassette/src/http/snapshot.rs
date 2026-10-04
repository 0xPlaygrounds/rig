//! Request snapshots: the request bodies Rig sends during replay, pinned
//! beside each fixture, so a change to an encoder shows as a reviewable diff.
//!
//! `RIG_CASSETTE_SNAPSHOTS` selects what a replay session does with them.
//! Unset or `off` changes nothing. `check` fails the session when a request
//! differs from its snapshot, and `write` rewrites the snapshot from the
//! requests replay received. Recording never reads or writes a snapshot.
//!
//! A snapshot `<fixture>.requests.json` lists, per interaction, only how the
//! body Rig sends differs from the recorded body, so a fixture whose requests
//! all equal their recordings has none. Bodies are compared after the
//! cassette scrubber, as canonical JSON, and a multipart body by its parts,
//! a binary or large part reduced to its length and FNV-1a hash.
//!
//! ```
//! use rig_cassette::http::request_snapshot;
//! use std::path::Path;
//! let snapshot = request_snapshot(Path::new("openai/chat.yaml"));
//! assert_eq!(snapshot, Path::new("openai/chat.requests.json"));
//! ```
#![deny(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unreachable
)]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value, json};

use super::{
    CassetteError, CassettePolicy, CassetteScrubber, MultipartPart, canonical_json,
    multipart_boundary, parse_multipart_parts, temporary_cassette_path,
};

const SNAPSHOT_ENV: &str = "RIG_CASSETTE_SNAPSHOTS";
/// A multipart part up to this many UTF-8 bytes is kept as text.
const VERBATIM_PART_LIMIT: usize = 4096;
const PREVIEW_LIMIT: usize = 160;
/// Differences listed per interaction before the rest are counted.
const LISTED_DIFFERENCES: usize = 20;

/// The snapshot file that belongs to the fixture at `cassette_path`.
pub fn request_snapshot(cassette_path: &Path) -> PathBuf {
    cassette_path.with_extension("requests.json")
}

/// What a replay session does with its fixture's request snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SnapshotMode {
    Off,
    Check,
    Write,
}

impl SnapshotMode {
    /// Read `RIG_CASSETTE_SNAPSHOTS`, ignoring case. Unset, empty and
    /// non-Unicode values are `Off`; a value other than `off`, `check` or
    /// `write` is an error.
    pub(super) fn current(cassette_path: &Path) -> Result<Self, CassetteError> {
        Self::parse(std::env::var(SNAPSHOT_ENV).ok(), cassette_path)
    }

    fn parse(value: Option<String>, cassette_path: &Path) -> Result<Self, CassetteError> {
        let Some(value) = value else {
            return Ok(Self::Off);
        };
        match value.to_ascii_lowercase().as_str() {
            "" | "off" => Ok(Self::Off),
            "check" => Ok(Self::Check),
            "write" => Ok(Self::Write),
            _ => Err(CassetteError::InvalidSnapshotMode {
                path: cassette_path.to_path_buf(),
                value,
            }),
        }
    }
}

#[derive(Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SnapshotFile {
    interactions: Vec<InteractionChanges>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct InteractionChanges {
    index: usize,
    changes: Vec<Change>,
}

/// One difference between a recorded body and the body sent, at a JSON
/// pointer into the recorded body. `recorded` alone removes an object key,
/// `sent` alone adds one, and both replace the value. With `splice`, `path`
/// names an array and `recorded` and `sent` are the items removed and
/// inserted at that index.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Change {
    path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    splice: Option<usize>,
    #[serde(
        default,
        deserialize_with = "present",
        skip_serializing_if = "Option::is_none"
    )]
    recorded: Option<Value>,
    #[serde(
        default,
        deserialize_with = "present",
        skip_serializing_if = "Option::is_none"
    )]
    sent: Option<Value>,
}

/// A present field is `Some`, even when it holds `null`.
fn present<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<Value>, D::Error> {
    Value::deserialize(deserializer).map(Some)
}

/// The snapshot state of one replay session in `Check` or `Write` mode.
pub(super) struct RequestSnapshots {
    mode: SnapshotMode,
    cassette_path: PathBuf,
    snapshot_path: PathBuf,
    /// Each interaction's recorded body, as [`body_view`] reads it.
    recorded: Vec<Value>,
    /// Each interaction's expected body: the recording with the snapshot's
    /// changes applied.
    expected: Vec<Value>,
    /// What replay received for each interaction, kept in `Write` mode.
    sent: Vec<Option<Value>>,
    /// A readable difference for each request that differed, in `Check` mode.
    differences: Vec<String>,
}

impl RequestSnapshots {
    /// The session state for a fixture whose interactions recorded `recorded`
    /// bodies, or `None` when `mode` is `Off`. `Check` reads the snapshot and
    /// applies it to the recording; a missing snapshot means every request
    /// equals its recording. An unreadable or malformed snapshot, or one that
    /// does not apply to the recording, is an error.
    pub(super) fn load(
        mode: SnapshotMode,
        cassette_path: &Path,
        recorded: Vec<Value>,
    ) -> Result<Option<Self>, CassetteError> {
        if mode == SnapshotMode::Off {
            return Ok(None);
        }
        let snapshot_path = request_snapshot(cassette_path);
        let invalid = |reason: String| CassetteError::InvalidSnapshot {
            path: cassette_path.to_path_buf(),
            snapshot: snapshot_path.clone(),
            reason,
        };
        let mut expected = recorded.clone();
        if mode == SnapshotMode::Check {
            let file = match std::fs::read_to_string(&snapshot_path) {
                Ok(text) => serde_json::from_str::<SnapshotFile>(&text)
                    .map_err(|error| invalid(error.to_string()))?,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    SnapshotFile::default()
                }
                Err(error) => return Err(invalid(error.to_string())),
            };
            let mut seen = BTreeSet::new();
            for entry in file.interactions {
                if !seen.insert(entry.index) {
                    return Err(invalid(format!(
                        "interaction {} is listed twice",
                        entry.index
                    )));
                }
                let slot = expected.get_mut(entry.index).ok_or_else(|| {
                    invalid(format!(
                        "interaction {} is not in the fixture's {} interaction(s)",
                        entry.index,
                        recorded.len()
                    ))
                })?;
                *slot = apply(slot, &entry.changes).map_err(|reason| {
                    invalid(format!(
                        "interaction {} does not apply to the recording: {reason}",
                        entry.index
                    ))
                })?;
            }
        }
        Ok(Some(Self {
            mode,
            cassette_path: cassette_path.to_path_buf(),
            snapshot_path,
            sent: vec![None; recorded.len()],
            recorded,
            expected,
            differences: Vec::new(),
        }))
    }

    /// Note that replay served interaction `index` for a request whose body
    /// reads as `sent`.
    pub(super) fn observe(&mut self, index: usize, sent: Value) {
        match self.mode {
            SnapshotMode::Check => {
                if let Some(difference) = self.difference(index, &sent) {
                    self.differences.push(difference);
                }
            }
            SnapshotMode::Write => {
                if let Some(slot) = self.sent.get_mut(index) {
                    *slot = Some(sent);
                }
            }
            SnapshotMode::Off => {}
        }
    }

    /// How a request whose body reads as `sent` differs from interaction
    /// `index`'s expected body, as readable lines, or `None` when it does not.
    pub(super) fn difference(&self, index: usize, sent: &Value) -> Option<String> {
        let expected = self.expected.get(index)?;
        let lines = describe(expected, sent);
        (!lines.is_empty()).then(|| format!("interaction {index}:\n{}", lines.join("\n")))
    }

    /// How many changes turn interaction `index`'s expected body into
    /// `sent`: 0 when they are equal, `usize::MAX` for no such interaction.
    pub(super) fn distance(&self, index: usize, sent: &Value) -> usize {
        self.expected
            .get(index)
            .map_or(usize::MAX, |expected| diff(expected, sent).len())
    }

    /// End the session: in `Check` mode fail when a request differed; in
    /// `Write` mode write the changes replay saw, or remove the snapshot when
    /// every request equalled its recording.
    pub(super) fn finish(&self) -> Result<(), CassetteError> {
        match self.mode {
            SnapshotMode::Check if !self.differences.is_empty() => {
                Err(CassetteError::SnapshotMismatch {
                    path: self.cassette_path.clone(),
                    snapshot: self.snapshot_path.clone(),
                    differences: self.differences.clone(),
                })
            }
            SnapshotMode::Write => self.write(),
            SnapshotMode::Check | SnapshotMode::Off => Ok(()),
        }
    }

    fn write(&self) -> Result<(), CassetteError> {
        let interactions: Vec<InteractionChanges> = self
            .recorded
            .iter()
            .zip(&self.sent)
            .enumerate()
            .filter_map(|(index, (recorded, sent))| {
                let changes = diff(recorded, sent.as_ref()?);
                (!changes.is_empty()).then_some(InteractionChanges { index, changes })
            })
            .collect();
        let failed = |source| CassetteError::WriteSnapshot {
            path: self.cassette_path.clone(),
            snapshot: self.snapshot_path.clone(),
            source,
        };
        if interactions.is_empty() {
            return match std::fs::remove_file(&self.snapshot_path) {
                Err(error) if error.kind() != std::io::ErrorKind::NotFound => Err(failed(error)),
                _ => Ok(()),
            };
        }
        let mut text = serde_json::to_string_pretty(&SnapshotFile { interactions })
            .map_err(|error| failed(std::io::Error::other(error)))?;
        text.push('\n');
        write_atomically(&self.snapshot_path, text.as_bytes()).map_err(failed)
    }
}

/// Write `contents` to a temporary file beside `path`, then rename it over
/// `path`, so a concurrent reader never sees a partial snapshot.
fn write_atomically(path: &Path, contents: &[u8]) -> std::io::Result<()> {
    let temp_path = temporary_cassette_path(path);
    let result =
        std::fs::write(&temp_path, contents).and_then(|()| std::fs::rename(&temp_path, path));
    if result.is_err() {
        let _ = std::fs::remove_file(&temp_path);
    }
    result
}

/// A request body as snapshots compare it: `null` when empty; for a
/// multipart `content_type`, `{"multipart": [part, ...]}`; else the scrubbed
/// body as canonical JSON, or as a string when it is not JSON, or as
/// `{"bytes", "fnv1a64"}` when it is not UTF-8.
pub(super) fn body_view(policy: CassettePolicy, content_type: Option<&str>, body: &[u8]) -> Value {
    if body.is_empty() {
        return Value::Null;
    }
    if let Some(boundary) = content_type
        .filter(|value| value.starts_with("multipart/form-data;"))
        .and_then(multipart_boundary)
        && let Some(parts) = parse_multipart_parts(body, &boundary)
    {
        let parts = parts
            .into_iter()
            .map(|part| part_view(policy, part))
            .collect();
        return json!({ "multipart": Value::Array(parts) });
    }
    match std::str::from_utf8(body) {
        Ok(text) => {
            let scrubbed = CassetteScrubber::new(policy).scrub_body(text);
            canonical_json(&scrubbed).unwrap_or(Value::String(scrubbed))
        }
        Err(_) => opaque(body),
    }
}

/// A multipart part as `{"headers": {...}, "body": ...}`: the body as
/// scrubbed text up to [`VERBATIM_PART_LIMIT`] UTF-8 bytes, else opaque.
fn part_view(policy: CassettePolicy, part: MultipartPart) -> Value {
    let headers: Map<String, Value> = part
        .headers
        .into_iter()
        .map(|(name, value)| (name, Value::String(value)))
        .collect();
    let body = match std::str::from_utf8(&part.body) {
        Ok(text) if text.len() <= VERBATIM_PART_LIMIT => {
            Value::String(CassetteScrubber::new(policy).scrub_body(text))
        }
        _ => opaque(&part.body),
    };
    json!({ "headers": headers, "body": body })
}

fn opaque(bytes: &[u8]) -> Value {
    json!({ "bytes": bytes.len(), "fnv1a64": format!("{:016x}", fnv1a64(bytes)) })
}

/// 64-bit FNV-1a: a fixed, dependency-free hash for telling binary parts apart.
fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// The changes that turn `recorded` into `sent`, in a fixed order: object
/// keys sorted, array items compared after their common prefix and suffix,
/// and arrays whose middles differ in length spliced whole.
pub(super) fn diff(recorded: &Value, sent: &Value) -> Vec<Change> {
    let mut changes = Vec::new();
    diff_into(String::new(), recorded, sent, &mut changes);
    changes
}

fn diff_into(path: String, recorded: &Value, sent: &Value, changes: &mut Vec<Change>) {
    if recorded == sent {
        return;
    }
    match (recorded, sent) {
        (Value::Object(recorded), Value::Object(sent)) => {
            let keys: BTreeSet<&String> = recorded.keys().chain(sent.keys()).collect();
            for key in keys {
                let at = format!("{path}/{}", escape_token(key));
                match (recorded.get(key), sent.get(key)) {
                    (Some(recorded), Some(sent)) => diff_into(at, recorded, sent, changes),
                    (Some(recorded), None) => changes.push(Change {
                        path: at,
                        splice: None,
                        recorded: Some(recorded.clone()),
                        sent: None,
                    }),
                    (None, Some(sent)) => changes.push(Change {
                        path: at,
                        splice: None,
                        recorded: None,
                        sent: Some(sent.clone()),
                    }),
                    (None, None) => {}
                }
            }
        }
        (Value::Array(recorded), Value::Array(sent)) => {
            let prefix = recorded
                .iter()
                .zip(sent)
                .take_while(|(recorded, sent)| recorded == sent)
                .count();
            let recorded_rest = recorded.get(prefix..).unwrap_or_default();
            let sent_rest = sent.get(prefix..).unwrap_or_default();
            let suffix = recorded_rest
                .iter()
                .rev()
                .zip(sent_rest.iter().rev())
                .take_while(|(recorded, sent)| recorded == sent)
                .count();
            let recorded_middle = recorded_rest
                .get(..recorded_rest.len() - suffix)
                .unwrap_or_default();
            let sent_middle = sent_rest
                .get(..sent_rest.len() - suffix)
                .unwrap_or_default();
            if recorded_middle.len() == sent_middle.len() {
                for (offset, (recorded, sent)) in
                    recorded_middle.iter().zip(sent_middle).enumerate()
                {
                    diff_into(
                        format!("{path}/{}", prefix + offset),
                        recorded,
                        sent,
                        changes,
                    );
                }
            } else {
                changes.push(Change {
                    path,
                    splice: Some(prefix),
                    recorded: Some(Value::Array(recorded_middle.to_vec())),
                    sent: Some(Value::Array(sent_middle.to_vec())),
                });
            }
        }
        _ => changes.push(Change {
            path,
            splice: None,
            recorded: Some(recorded.clone()),
            sent: Some(sent.clone()),
        }),
    }
}

/// `recorded` with `changes` applied in order. Each change must find what it
/// says was recorded, so a snapshot taken against an older recording fails
/// with the pointer that no longer fits.
pub(super) fn apply(recorded: &Value, changes: &[Change]) -> Result<Value, String> {
    let mut value = recorded.clone();
    for change in changes {
        apply_change(&mut value, change)
            .map_err(|reason| format!("{}: {reason}", display_path(&change.path)))?;
    }
    Ok(value)
}

fn apply_change(value: &mut Value, change: &Change) -> Result<(), &'static str> {
    match (change.splice, &change.recorded, &change.sent) {
        (Some(at), Some(Value::Array(removed)), Some(Value::Array(inserted))) => {
            let Some(Value::Array(items)) = value.pointer_mut(&change.path) else {
                return Err("the recording holds no array here");
            };
            let end = at + removed.len();
            if items.get(at..end) != Some(removed.as_slice()) {
                return Err("the recording holds other items here");
            }
            items.splice(at..end, inserted.iter().cloned());
            Ok(())
        }
        (Some(_), _, _) => Err("a splice needs recorded and sent arrays"),
        (None, Some(recorded), Some(sent)) => {
            let slot = value
                .pointer_mut(&change.path)
                .ok_or("the recording holds no value here")?;
            if slot != recorded {
                return Err("the recording holds another value here");
            }
            *slot = sent.clone();
            Ok(())
        }
        (None, recorded, sent) => {
            let (parent, token) = change
                .path
                .rsplit_once('/')
                .ok_or("only an object key can be added or removed")?;
            let Some(Value::Object(map)) = value.pointer_mut(parent) else {
                return Err("the recording holds no object here");
            };
            let key = unescape_token(token);
            match (recorded, sent) {
                (None, Some(sent)) => {
                    if map.contains_key(&key) {
                        return Err("the recording already holds this key");
                    }
                    map.insert(key, sent.clone());
                    Ok(())
                }
                (Some(recorded), None) => {
                    if map.get(&key) != Some(recorded) {
                        return Err("the recording holds another value here");
                    }
                    map.shift_remove(&key);
                    Ok(())
                }
                _ => Err("a change needs a recorded or a sent value"),
            }
        }
    }
}

/// How `sent` differs from `expected`, one readable line per change, the
/// list cut after [`LISTED_DIFFERENCES`] lines.
pub(super) fn describe(expected: &Value, sent: &Value) -> Vec<String> {
    let changes = diff(expected, sent);
    let mut lines: Vec<String> = changes
        .iter()
        .take(LISTED_DIFFERENCES)
        .map(|change| {
            let path = display_path(&change.path);
            match (change.splice, &change.recorded, &change.sent) {
                (Some(at), Some(expected), Some(sent)) => format!(
                    "  {path} from item {at}: snapshot {}, sent {}",
                    preview(expected),
                    preview(sent)
                ),
                (None, Some(expected), Some(sent)) => format!(
                    "  {path}: snapshot {}, sent {}",
                    preview(expected),
                    preview(sent)
                ),
                (None, None, Some(sent)) => {
                    format!("  {path}: not in the snapshot, sent {}", preview(sent))
                }
                (_, Some(expected), _) => {
                    format!("  {path}: snapshot {}, not sent", preview(expected))
                }
                _ => format!("  {path}: differs"),
            }
        })
        .collect();
    if changes.len() > LISTED_DIFFERENCES {
        lines.push(format!(
            "  ... and {} more difference(s)",
            changes.len() - LISTED_DIFFERENCES
        ));
    }
    lines
}

fn preview(value: &Value) -> String {
    let text = value.to_string();
    if text.chars().count() <= PREVIEW_LIMIT {
        return text;
    }
    let mut cut: String = text.chars().take(PREVIEW_LIMIT).collect();
    cut.push_str("...");
    cut
}

fn display_path(path: &str) -> &str {
    if path.is_empty() { "body" } else { path }
}

fn escape_token(key: &str) -> String {
    key.replace('~', "~0").replace('/', "~1")
}

fn unescape_token(token: &str) -> String {
    token.replace("~1", "/").replace("~0", "~")
}

#[cfg(test)]
mod tests;
