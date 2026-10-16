//! The audit log on disk: one JSON line per [`Entry`] in
//! `<dir>/<date>.jsonl`, each naming the hash of the line before it (the
//! first line of a day's file names the last line of the day before), and
//! its own hash over its content, so an edited line is found where it was
//! edited. Appends hold the file's lock and are synced before they return.

use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use rig_core::effect::Decider;
use rig_core::tool::content_hash;
use serde::{Deserialize, Serialize};

/// The `prev` of the very first entry.
pub const GENESIS: &str = "";

/// One audited event.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Entry {
    /// When it was written, UTC, RFC 3339.
    pub time: String,
    /// What happened.
    pub event: Event,
    /// The session's id.
    pub session: Option<String>,
    /// The agent's id.
    pub agent: Option<String>,
    /// The agents that spawned it, nearest first.
    #[serde(default)]
    pub parents: Vec<String>,
    /// The tool called.
    pub tool: Option<String>,
    /// Its arguments, secrets redacted.
    pub arguments: Option<serde_json::Value>,
    /// How it went.
    pub outcome: Option<Ran>,
    /// The exit code of the process it ran.
    pub exit_code: Option<i32>,
    /// Why it was refused or failed.
    pub error: Option<String>,
    /// The files it could change, with their content before and after.
    #[serde(default)]
    pub files: Vec<FileHashes>,
    /// Who decided it: a person, or a plugin by name (a rule, a judge, a
    /// headless default). `None` for a call no one decided, which runs.
    pub decided_by: Option<Decider>,
    /// Anything else: a question and its answer, why the session ended.
    pub detail: Option<String>,
    /// The hash of the line before, [`GENESIS`] for the first.
    pub prev: String,
    /// The hash of this entry with `hash` empty.
    pub hash: String,
}

/// What an [`Entry`] records.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Event {
    /// A tool call that changed something or reached outside.
    #[default]
    ToolCall,
    /// A question that decided whether something may run.
    Approval,
    /// A new session started.
    SessionStart,
    /// A session started again: resumed or reloaded.
    SessionResume,
    /// The app exited.
    SessionEnd,
}

/// How a call went.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Ran {
    /// It ran.
    Ran,
    /// It was refused before it ran.
    Refused,
    /// It failed.
    Failed,
    /// It was stopped.
    Cancelled,
}

/// A file a call could change: the SHA-256 of its bytes before the call
/// and after, `None` where there was no file.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileHashes {
    /// The file.
    pub path: String,
    /// Before the call.
    pub before: Option<String>,
    /// After it.
    pub after: Option<String>,
}

/// Where the chain breaks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Broken {
    /// The day's file.
    pub file: PathBuf,
    /// The entry's line in it, from 1.
    pub line: usize,
    /// What is wrong with it.
    pub why: String,
}

impl fmt::Display for Broken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}: {}", self.file.display(), self.line, self.why)
    }
}

/// The hash a line is chained by.
pub fn line_hash(line: &str) -> String {
    content_hash(line.as_bytes()).0
}

/// The SHA-256 of the file at `path`, `None` when it is no file.
pub fn file_hash(path: &Path) -> Option<String> {
    fs::read(path).ok().map(|bytes| content_hash(&bytes).0)
}

/// The file of `date` (`YYYY-MM-DD`) in `dir`.
pub fn file_for(dir: &Path, date: &str) -> PathBuf {
    dir.join(format!("{date}.jsonl"))
}

/// Today's date, UTC.
pub fn today() -> String {
    let (date, _) = civil(SystemTime::now());
    date
}

/// `time` as a UTC date and an RFC 3339 time.
fn civil(time: SystemTime) -> (String, String) {
    let since = time.duration_since(UNIX_EPOCH).unwrap_or_default();
    let secs = i64::try_from(since.as_secs()).unwrap_or_default();
    let (days, rest) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // Howard Hinnant's days-to-civil.
    let z = days + 719_468;
    let (era, doe) = (z.div_euclid(146_097), z.rem_euclid(146_097));
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    let date = format!("{year:04}-{month:02}-{day:02}");
    let (hour, minute, second) = (rest / 3600, rest % 3600 / 60, rest % 60);
    let millis = since.subsec_millis();
    let stamp = format!("{date}T{hour:02}:{minute:02}:{second:02}.{millis:03}Z");
    (date, stamp)
}

/// Whether an entry could be appended in `dir` now: it can be made and
/// today's file opened for appending.
pub fn probe(dir: &Path) -> io::Result<()> {
    fs::create_dir_all(dir)?;
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(file_for(dir, &today()))
        .map(drop)
}

/// Appends `entry` to today's file in `dir`, timed, chained and hashed,
/// and syncs it, under the file's lock; the line written.
pub fn append(dir: &Path, mut entry: Entry) -> io::Result<String> {
    fs::create_dir_all(dir)?;
    let (date, stamp) = civil(SystemTime::now());
    let path = file_for(dir, &date);
    let mut file = OpenOptions::new()
        .create(true)
        .read(true)
        .append(true)
        .open(&path)?;
    file.lock()?;
    let last = match last_line(&mut file)? {
        Some(line) => Some(line),
        None => previous_day(dir, &date)?,
    };
    entry.time = stamp;
    entry.prev = last
        .as_deref()
        .map_or_else(|| GENESIS.to_owned(), line_hash);
    entry.hash = own_hash(&entry)?;
    let line = serde_json::to_string(&entry)?;
    file.write_all(format!("{line}\n").as_bytes())?;
    file.sync_data()?;
    Ok(line)
}

/// The hash of `entry` with its `hash` empty.
fn own_hash(entry: &Entry) -> io::Result<String> {
    let unsigned = Entry {
        hash: String::new(),
        ..entry.clone()
    };
    Ok(line_hash(&serde_json::to_string(&unsigned)?))
}

/// The days' files in `dir`, oldest first.
fn days(dir: &Path) -> io::Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = fs::read_dir(dir)?
        .filter_map(|entry| Some(entry.ok()?.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "jsonl"))
        .collect();
    files.sort();
    Ok(files)
}

/// The last line of the newest file before `date`'s.
fn previous_day(dir: &Path, date: &str) -> io::Result<Option<String>> {
    let today = file_for(dir, date);
    let before = days(dir)?.into_iter().rfind(|path| *path < today);
    match before {
        Some(path) => last_line(&mut File::open(path)?),
        None => Ok(None),
    }
}

/// The last line of `file`, read from its end.
fn last_line(file: &mut File) -> io::Result<Option<String>> {
    let mut at = file.seek(SeekFrom::End(0))?;
    let mut tail: Vec<u8> = Vec::new();
    loop {
        let body = tail.strip_suffix(b"\n").unwrap_or(&tail);
        let start = body
            .iter()
            .rposition(|&byte| byte == b'\n')
            .map(|at| at + 1);
        if let Some(line) = start.and_then(|start| body.get(start..)) {
            return Ok(Some(String::from_utf8_lossy(line).into_owned()));
        }
        if at == 0 {
            return Ok((!body.is_empty()).then(|| String::from_utf8_lossy(body).into_owned()));
        }
        let step = at.min(4096);
        at -= step;
        let mut chunk = vec![0; usize::try_from(step).unwrap_or_default()];
        file.seek(SeekFrom::Start(at))?;
        file.read_exact(&mut chunk)?;
        chunk.extend_from_slice(&tail);
        tail = chunk;
    }
}

/// Checks the chain in `dir` from the file of `from` (a date) on, or from
/// the first: the number of entries checked, or the first broken one.
pub fn verify(dir: &Path, from: Option<&str>) -> Result<usize, Broken> {
    let broken = |file: &Path, line, why: String| Broken {
        file: file.to_owned(),
        line,
        why,
    };
    let files = days(dir).map_err(|error| broken(dir, 0, error.to_string()))?;
    let start = from.map_or(0, |date| {
        let first = file_for(dir, date);
        files
            .iter()
            .position(|path| *path >= first)
            .unwrap_or(files.len())
    });
    let mut prev = match start.checked_sub(1).and_then(|before| files.get(before)) {
        Some(path) => File::open(path)
            .and_then(|mut file| last_line(&mut file))
            .map_err(|error| broken(path, 0, error.to_string()))?,
        None => None,
    };
    let mut count = 0;
    for path in files.iter().skip(start) {
        let text = fs::read_to_string(path).map_err(|error| broken(path, 0, error.to_string()))?;
        for (number, line) in text.lines().enumerate() {
            let at = number + 1;
            let entry: Entry = serde_json::from_str(line)
                .map_err(|error| broken(path, at, format!("not an entry: {error}")))?;
            if own_hash(&entry).ok().as_ref() != Some(&entry.hash) {
                return Err(broken(path, at, "the entry was changed".to_owned()));
            }
            let expected = prev
                .as_deref()
                .map_or_else(|| GENESIS.to_owned(), line_hash);
            if entry.prev != expected {
                let why = "the entry before it was changed, removed or added".to_owned();
                return Err(broken(path, at, why));
            }
            prev = Some(line.to_owned());
            count += 1;
        }
    }
    Ok(count)
}

/// Every entry in `dir`, oldest first; lines that are no entry are skipped.
pub fn entries(dir: &Path) -> io::Result<Vec<Entry>> {
    let mut all = Vec::new();
    for path in days(dir)? {
        let text = fs::read_to_string(path)?;
        all.extend(
            text.lines()
                .filter_map(|line| serde_json::from_str(line).ok()),
        );
    }
    Ok(all)
}

/// The entries of `session` in `dir`, one readable line each.
pub fn show(dir: &Path, session: &str) -> io::Result<Vec<String>> {
    let entries = entries(dir)?;
    let lines = entries
        .iter()
        .filter(|entry| entry.session.as_deref() == Some(session))
        .map(readable)
        .collect();
    Ok(lines)
}

/// One entry as a line: when, which agent, what, how it went and who
/// decided.
fn readable(entry: &Entry) -> String {
    let short = |id: &str| id.chars().take(8).collect::<String>();
    let agent = entry.agent.as_deref().map(short).unwrap_or_default();
    let chain: Vec<String> = entry.parents.iter().map(|id| short(id)).collect();
    let agent = match chain.is_empty() {
        true => agent,
        false => format!("{agent} (by {})", chain.join(" < ")),
    };
    let mut line = format!("{} {agent} {:?}", entry.time, entry.event);
    if let Some(tool) = &entry.tool {
        let arguments = entry.arguments.as_ref().map(ToString::to_string);
        line.push_str(&format!(" {tool} {}", arguments.unwrap_or_default()));
    }
    if let Some(outcome) = entry.outcome {
        line.push_str(&format!(" -> {outcome:?}"));
    }
    if let Some(code) = entry.exit_code {
        line.push_str(&format!(" (exit {code})"));
    }
    if let Some(error) = &entry.error {
        line.push_str(&format!(": {error}"));
    }
    for file in &entry.files {
        let hash = |hash: &Option<String>| hash.as_deref().map_or("-".to_owned(), short);
        let (before, after) = (hash(&file.before), hash(&file.after));
        line.push_str(&format!(" [{} {before}->{after}]", file.path));
    }
    let decided = match &entry.decided_by {
        Some(Decider::Person) => "a person".to_owned(),
        Some(Decider::Plugin(name)) => name.to_string(),
        Some(other) => format!("{other:?}"),
        None => "no one: allowed".to_owned(),
    };
    if matches!(entry.event, Event::ToolCall | Event::Approval) {
        line.push_str(&format!(" (decided by {decided})"));
    }
    if let Some(detail) = &entry.detail {
        line.push_str(&format!(" {detail}"));
    }
    line
}
