//! Effect logs as JSON lines on disk: a `{"header": …}` line whenever the
//! header changed since the last one written, then one resolved record per
//! line. Appending keeps a long-running host's log durable as it goes;
//! [`read`] folds the lines back into one [`EffectLog`] that
//! [`EffectLogReplayer`](super::EffectLogReplayer) replays.
//!
//! The file grows with what happened, not with the length of the
//! conversation. A completion request mostly repeats the one before it on
//! the same handler and scope (an agent's previous call) plus a few new
//! messages, so [`Writer`] writes it as a continuation of that request:
//! `{"id": …, "after": <the earlier record's id>, "keep": <messages kept>,
//! "same_tools": true, "record": …}`, whose record holds only the appended
//! messages, and no tools when they are the earlier request's. [`read`]
//! restores the whole request, so a replayer sees what was dispatched. The
//! writer also leaves out the parts of a completion's raw provider document
//! that echo the request (`instructions`, `tools`) and the per-item usage
//! attribution (`usage.attribution`); nothing reads them back.
//!
//! ```no_run
//! use rig_cassette::effect_log::{EffectLogRecorder, jsonl};
//!
//! let recorder = EffectLogRecorder::new();
//! let mut writer = jsonl::Writer::new("effects.jsonl");
//! writer.append(&recorder.take())?;
//! let log = jsonl::read("effects.jsonl")?;
//! let next = jsonl::last_id("effects.jsonl")?.map_or(0, |id| id.as_u64() + 1);
//! # Ok::<(), std::io::Error>(())
//! ```

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use rig_core::effect::{EffectId, EffectKind, EffectRecord};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{EffectLog, LogHeader};

/// Appends effect logs, such as successive [`EffectLogRecorder::take`]s,
/// to a JSON-lines file.
///
/// [`EffectLogRecorder::take`]: super::EffectLogRecorder::take
#[derive(Debug)]
pub struct Writer {
    path: PathBuf,
    /// The header last written by this writer.
    written: Option<LogHeader>,
    /// The completion request last written per chain, which the next one
    /// on that chain is written against.
    heads: HashMap<String, Head>,
}

/// A chain's latest completion request as written: each message and the
/// tools as JSON text.
#[derive(Debug)]
struct Head {
    id: EffectId,
    history: Vec<String>,
    tools: String,
}

impl Writer {
    /// A writer appending to the file at `path`, created on first write.
    /// Its first completion request per handler and scope is written
    /// whole, later ones as continuations.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            written: None,
            heads: HashMap::new(),
        }
    }

    /// The file written to.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Appends `log`'s records, after its header when that differs from
    /// the last one this writer wrote. Writes nothing when there is
    /// neither.
    pub fn append(&mut self, log: &EffectLog) -> io::Result<()> {
        let header_due = self.written.as_ref() != Some(&log.header);
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
            if let Err(failure) = self.encode(record, &mut lines) {
                self.heads.clear();
                return Err(failure);
            }
            lines.push(b'\n');
        }
        let written = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .and_then(|mut file| file.write_all(&lines));
        if let Err(failure) = written {
            // The heads may name records that never reached the file.
            self.heads.clear();
            return Err(failure);
        }
        if header_due {
            self.written = Some(log.header.clone());
        }
        Ok(())
    }

    /// Writes `record` to `out`: a completion as a continuation of its
    /// chain's previous request when the two share a first message, any
    /// other record as it is.
    fn encode(&mut self, record: &EffectRecord, out: &mut Vec<u8>) -> io::Result<()> {
        if !matches!(record.kind, EffectKind::Completion { .. }) {
            return Ok(serde_json::to_writer(out, record)?);
        }
        let mut value = serde_json::to_value(record)?;
        for (parent, key) in RAW_ECHOES {
            if let Some(parent) = value.pointer_mut(parent).and_then(Value::as_object_mut) {
                parent.remove(key);
            }
        }
        let chain = chain_of(&value);
        let request = value
            .pointer_mut("/kind/request")
            .and_then(Value::as_object_mut)
            .ok_or_else(|| io::Error::other("a completion record without a request object"))?;
        let head = Head {
            id: record.id,
            history: request
                .get("chat_history")
                .and_then(Value::as_array)
                .map(|messages| messages.iter().map(Value::to_string).collect())
                .unwrap_or_default(),
            tools: request
                .get("tools")
                .map(Value::to_string)
                .unwrap_or_default(),
        };
        let continued = self.heads.get(&chain).and_then(|base| {
            let keep = base
                .history
                .iter()
                .zip(&head.history)
                .take_while(|(kept, sent)| kept == sent)
                .count();
            (keep > 0).then_some((base.id, keep, base.tools == head.tools))
        });
        match continued {
            None => serde_json::to_writer(&mut *out, &value)?,
            Some((after, keep, same_tools)) => {
                if let Some(history) = request
                    .get_mut("chat_history")
                    .and_then(Value::as_array_mut)
                {
                    history.drain(..keep.min(history.len()));
                }
                if same_tools {
                    request.remove("tools");
                }
                if let Some(fields) = value.as_object_mut() {
                    fields.remove("id");
                }
                serde_json::to_writer(
                    &mut *out,
                    &DeltaLine {
                        id: record.id,
                        after,
                        keep,
                        same_tools,
                        record: value,
                    },
                )?;
            }
        }
        self.heads.insert(chain, head);
        Ok(())
    }
}

/// The parts of a completion record's raw provider document a log leaves
/// out, as (parent pointer, key): the request's echoes, which the record's
/// request already holds, and the per-item usage attribution.
const RAW_ECHOES: [(&str, &str); 3] = [
    ("/outcome/Ok/raw", "instructions"),
    ("/outcome/Ok/raw", "tools"),
    ("/outcome/Ok/raw/usage", "attribution"),
];

/// The chain a serialized record belongs to: its handler key and scope.
/// Completion requests on one chain continue one another.
fn chain_of(record: &Value) -> String {
    let field = |name: &str| record.get(name).map(Value::to_string).unwrap_or_default();
    format!("{}\u{0}{}", field("key"), field("scope"))
}

/// Reads the JSON-lines log at `path` into one [`EffectLog`]: every record
/// in file order, under the headers merged. A handler described again
/// takes its latest description; signatures, required rows, program
/// identities, stream errors and deliveries accumulate; the hook stack,
/// run spec and serving policy are the latest stated. A continued
/// completion request is restored whole. A line that is neither a header
/// nor a record, or a continuation of a request that is not its chain's
/// latest, is an error.
pub fn read(path: impl AsRef<Path>) -> io::Result<EffectLog> {
    let mut log = EffectLog::default();
    let mut heads: HashMap<String, ReadHead> = HashMap::new();
    for line in BufReader::new(File::open(path)?).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(&line)?;
        if value.get("header").is_some() {
            let HeaderLine { header } = serde_json::from_value::<HeaderLine<LogHeader>>(value)?;
            merge(&mut log.header, header);
            continue;
        }
        let value = if value.get("after").is_some() {
            restore(serde_json::from_value(value)?, &heads)?
        } else {
            value
        };
        if value.pointer("/kind/effect").and_then(Value::as_str) == Some("completion") {
            let request = value.pointer("/kind/request");
            let field = |name: &str| request.and_then(|request| request.get(name)).cloned();
            heads.insert(
                chain_of(&value),
                ReadHead {
                    id: serde_json::from_value(value.get("id").cloned().unwrap_or_default())?,
                    history: match field("chat_history") {
                        Some(Value::Array(messages)) => messages,
                        _ => Vec::new(),
                    },
                    tools: field("tools"),
                },
            );
        }
        log.records.push(serde_json::from_value(value)?);
    }
    Ok(log)
}

/// A chain's latest completion request as read back.
struct ReadHead {
    id: EffectId,
    history: Vec<Value>,
    tools: Option<Value>,
}

/// The record a continuation line stands for, its request made whole
/// from its chain's latest one.
fn restore(line: DeltaLine<Value>, heads: &HashMap<String, ReadHead>) -> io::Result<Value> {
    let invalid = |what: String| io::Error::new(io::ErrorKind::InvalidData, what);
    let DeltaLine {
        id,
        after,
        keep,
        same_tools,
        mut record,
    } = line;
    let base = heads
        .get(&chain_of(&record))
        .filter(|base| base.id == after)
        .ok_or_else(|| {
            invalid(format!(
                "record {id} continues completion {after}, which is not the latest on its handler and scope"
            ))
        })?;
    let kept = base.history.get(..keep).ok_or_else(|| {
        invalid(format!(
            "record {id} keeps {keep} messages of completion {after}, which has {}",
            base.history.len()
        ))
    })?;
    let request = record
        .pointer_mut("/kind/request")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| invalid(format!("record {id} continues a request but has none")))?;
    let mut history = kept.to_vec();
    if let Some(Value::Array(appended)) = request.remove("chat_history") {
        history.extend(appended);
    }
    request.insert("chat_history".to_owned(), Value::Array(history));
    if same_tools {
        let tools = base.tools.clone().ok_or_else(|| {
            invalid(format!(
                "record {id} reuses the tools of completion {after}, which has none"
            ))
        })?;
        request.insert("tools".to_owned(), tools);
    }
    if let Some(fields) = record.as_object_mut() {
        fields.insert("id".to_owned(), serde_json::to_value(id)?);
    }
    Ok(record)
}

/// How many of the log's last lines [`last_id`] reads.
const TAIL_LINES: usize = 64;

/// How many bytes [`last_id`] reads at a time, from the end backwards.
const TAIL_CHUNK: u64 = 64 * 1024;

/// The highest effect id among the last lines of the log at `path`, `None`
/// when it has no record there. Ids are taken in order and records are
/// appended as they resolve, so the highest id is among the last few
/// records; reading backwards from the end keeps a restarting host's
/// startup independent of the log's length. A line's id is read off its
/// start, so a long record is not parsed.
pub fn last_id(path: impl AsRef<Path>) -> io::Result<Option<EffectId>> {
    let mut file = File::open(path)?;
    let mut end = file.metadata()?.len();
    // The bytes after the last newline seen, as read: last chunk first.
    let mut partial: Vec<Vec<u8>> = Vec::new();
    let mut lines = 0;
    let mut highest: Option<EffectId> = None;
    while end > 0 && lines < TAIL_LINES {
        let from = end.saturating_sub(TAIL_CHUNK);
        let mut chunk = vec![0; usize::try_from(end - from).map_err(io::Error::other)?];
        file.seek(SeekFrom::Start(from))?;
        file.read_exact(&mut chunk)?;
        end = from;
        let mut pieces = chunk.rsplit(|byte| *byte == b'\n').peekable();
        while let Some(piece) = pieces.next() {
            if pieces.peek().is_none() {
                // Before the chunk's first newline: the line goes on in an
                // earlier chunk, unless this is the file's start.
                partial.push(piece.to_vec());
                break;
            }
            let mut line = piece.to_vec();
            for later in partial.drain(..).rev() {
                line.extend(later);
            }
            take(&line, &mut lines, &mut highest);
            if lines >= TAIL_LINES {
                break;
            }
        }
    }
    if end == 0 && lines < TAIL_LINES {
        let line: Vec<u8> = partial.drain(..).rev().flatten().collect();
        take(&line, &mut lines, &mut highest);
    }
    Ok(highest)
}

/// Counts a non-blank `line` among those read and keeps its id when it is
/// the highest yet.
fn take(line: &[u8], lines: &mut usize, highest: &mut Option<EffectId>) {
    if line.iter().all(u8::is_ascii_whitespace) {
        return;
    }
    *lines += 1;
    if let Some(id) = line_id(line) {
        *highest = (*highest).max(Some(id));
    }
}

/// A record line's id, read off its start when it begins with it (a
/// continuation always does, and a whole record does after a `null`
/// tool output), else by parsing the line. Header lines have none.
fn line_id(line: &[u8]) -> Option<EffectId> {
    /// A record line's id.
    #[derive(Deserialize)]
    struct IdOnly {
        id: EffectId,
    }
    if line.starts_with(b"{\"header\":") {
        return None;
    }
    let leading = line.strip_prefix(b"{").and_then(|rest| {
        let rest = rest.strip_prefix(b"\"tool_output\":null,").unwrap_or(rest);
        let rest = rest.strip_prefix(b"\"id\":")?;
        let digits = rest.iter().take_while(|byte| byte.is_ascii_digit()).count();
        let (digits, after) = rest.split_at_checked(digits)?;
        matches!(after.first(), Some(b',' | b'}')).then_some(())?;
        std::str::from_utf8(digits).ok()?.parse::<u64>().ok()
    });
    match leading {
        Some(id) => Some(EffectId::from_raw(id)),
        None => serde_json::from_slice::<IdOnly>(line)
            .ok()
            .map(|record| record.id),
    }
}

/// Folds a later header into the merged one.
fn merge(into: &mut LogHeader, header: LogHeader) {
    for handler in header.handlers {
        match into
            .handlers
            .iter_mut()
            .find(|known| known.key == handler.key)
        {
            Some(known) => *known = handler,
            None => into.handlers.push(handler),
        }
    }
    for (key, family) in header.signature.iter() {
        into.signature.insert_if_absent(key.clone(), *family);
    }
    for (key, family) in header.required.iter() {
        into.required.insert_if_absent(key.clone(), *family);
    }
    into.programs.extend(header.programs);
    into.stream_errors.extend(header.stream_errors);
    if let Some(deliveries) = header.deliveries {
        into.deliveries.get_or_insert_default().extend(deliveries);
    }
    for limitation in header.delivery_limitations {
        if !into.delivery_limitations.contains(&limitation) {
            into.delivery_limitations.push(limitation);
        }
    }
    if !header.hooks.is_empty() {
        into.hooks = header.hooks;
    }
    into.run_spec = header.run_spec.or(into.run_spec);
    into.bus = header.bus.or(into.bus.take());
}

/// A header line, `{"header": …}`; record lines are bare records.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct HeaderLine<H> {
    header: H,
}

/// A completion record written as a continuation of its chain's latest
/// request: `record` holds the messages after the first `keep` of request
/// `after`, no tools when `same_tools`, and no id.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DeltaLine<R> {
    id: EffectId,
    after: EffectId,
    keep: usize,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    same_tools: bool,
    record: R,
}

#[cfg(test)]
mod tests;
