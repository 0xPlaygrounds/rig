//! Effect logs as JSON lines on disk: a `{"header": …}` line whenever the
//! header changed since the last one written, then one resolved record per
//! line. Appending keeps a long-running host's log durable as it goes;
//! [`read`] folds the lines back into one [`EffectLog`] that
//! [`EffectLogReplayer`](super::EffectLogReplayer) replays.
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

use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use rig_core::effect::{EffectId, EffectRecord};
use serde::{Deserialize, Serialize};

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
}

impl Writer {
    /// A writer appending to the file at `path`, created on first write.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            written: None,
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
            serde_json::to_writer(&mut lines, record)?;
            lines.push(b'\n');
        }
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?
            .write_all(&lines)?;
        if header_due {
            self.written = Some(log.header.clone());
        }
        Ok(())
    }
}

/// Reads the JSON-lines log at `path` into one [`EffectLog`]: every record
/// in file order, under the headers merged. A handler described again
/// takes its latest description; signatures, required rows, program
/// identities, stream errors and deliveries accumulate; the hook stack,
/// run spec and serving policy are the latest stated. A line that is
/// neither a header nor a record is an error.
pub fn read(path: impl AsRef<Path>) -> io::Result<EffectLog> {
    let mut log = EffectLog::default();
    for line in BufReader::new(File::open(path)?).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<Line>(&line)? {
            Line::Header(HeaderLine { header }) => merge(&mut log.header, header),
            Line::Record(record) => log.records.push(*record),
        }
    }
    Ok(log)
}

/// How many of the log's last lines [`last_id`] reads.
const TAIL_LINES: usize = 64;

/// The highest effect id among the last lines of the log at `path`, `None`
/// when it has no record there. Ids are taken in order and records are
/// appended as they resolve, so the highest id is among the last few
/// records; reading backwards from the end keeps a restarting host's
/// startup independent of the log's length.
pub fn last_id(path: impl AsRef<Path>) -> io::Result<Option<EffectId>> {
    /// A record line's id; header lines have none.
    #[derive(Deserialize)]
    struct IdOnly {
        id: EffectId,
    }
    let mut file = File::open(path)?;
    let mut start = file.metadata()?.len();
    let mut tail = Vec::new();
    let mut chunk: u64 = 64 * 1024;
    while start > 0 && tail.iter().filter(|byte| **byte == b'\n').count() <= TAIL_LINES {
        let from = start.saturating_sub(chunk);
        let mut read = vec![0; usize::try_from(start - from).map_err(io::Error::other)?];
        file.seek(SeekFrom::Start(from))?;
        file.read_exact(&mut read)?;
        read.append(&mut tail);
        tail = read;
        start = from;
        chunk = chunk.saturating_mul(2);
    }
    // Unless the whole file was read, the first piece may be part of a line.
    Ok(tail
        .split(|byte| *byte == b'\n')
        .skip(usize::from(start > 0))
        .filter_map(|line| serde_json::from_slice::<IdOnly>(line).ok())
        .map(|record| record.id)
        .max())
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

/// One line of the file.
#[derive(Deserialize)]
#[serde(untagged)]
enum Line {
    Header(HeaderLine<LogHeader>),
    Record(Box<EffectRecord>),
}

#[cfg(test)]
mod tests;
