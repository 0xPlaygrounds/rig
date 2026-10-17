//! SP-19: `session-doctor <sessions-dir> [--fix <out-dir>] [--json]`, a
//! plain program on rig's library crates (rig-core and rig-memory; no Bevy,
//! no harness) that checks whether every saved session still loads.
//!
//! Each directory under `<sessions-dir>` is one saved session, a
//! [`DirStore`]. It is read without writing ([`Session::read_lenient`]) and
//! classed ([`Class`]): a log of a format other than this build's
//! ([`FORMAT`](rig_memory::FORMAT)) or one that cannot be read makes it `other-format` or
//! `unreadable`; otherwise a torn last record ([`Repair::TornTail`]) makes it
//! `truncated`, and asides of kinds neither rig-core nor rig-memory defines
//! `unknown-entries`.
//!
//! With `--fix`, each fixable session is copied store to store into
//! `<out-dir>/<name>` and opened there with a [`Writer`], the library's own
//! repair path: it sets each torn tail aside. The copy is read back and must hold the same messages,
//! saved values and unknown entries as the original read; one that does
//! not is deleted. The originals are only ever read.

use std::collections::BTreeMap;
use std::fmt::{self, Write as _};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rig_core::conversation::{Aside, AsideId, Conversation, TURN};
use rig_core::message::Message;
use rig_memory::{COMPACTED, DirStore, Repair, Session, SessionError, Store, Writer};
use serde::Serialize;

/// The aside kinds rig's library crates define; any other is unknown to
/// them.
const KNOWN: [&str; 2] = [TURN, COMPACTED];

/// What a saved session is to this build of the library.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Class {
    /// It loads as it is.
    Ok,
    /// It loads, and holds entries of kinds the library does not define.
    UnknownEntries,
    /// A log's last record is incomplete; the rest loads.
    Truncated,
    /// A log declares a format other than the one this build reads.
    OtherFormat,
    /// It cannot be read.
    Unreadable,
}

impl Class {
    /// Its name in the report.
    pub fn name(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::UnknownEntries => "unknown-entries",
            Self::Truncated => "truncated",
            Self::OtherFormat => "other-format",
            Self::Unreadable => "unreadable",
        }
    }

    /// Whether `--fix` writes a repaired copy of a session of this class.
    pub fn fixable(self) -> bool {
        matches!(self, Self::UnknownEntries | Self::Truncated)
    }
}

impl fmt::Display for Class {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The incomplete tail of a log, which a fixed copy drops.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Tail {
    /// The log.
    pub log: String,
    /// The bytes kept.
    pub kept: u64,
    /// The bytes dropped.
    pub cut: u64,
    /// The file of the copy the dropped bytes were set aside in.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub set_aside: Option<String>,
}

/// Why a log cannot be read.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Problem {
    /// The log, when one log is at fault.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub log: Option<String>,
    /// The line, from 1, when the library names one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<usize>,
    /// Why.
    pub reason: String,
}

/// What `--fix` did with a session.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "result", rename_all = "kebab-case")]
pub enum Fix {
    /// The copy was written and read back the same.
    Written {
        /// Where.
        path: PathBuf,
    },
    /// No copy was kept.
    Failed {
        /// Why.
        reason: String,
    },
}

/// One saved session, checked.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Checked {
    /// Its directory's name.
    pub session: String,
    /// What it is.
    pub class: Class,
    /// The number of its entries of each kind the library does not define.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub unknown: BTreeMap<String, usize>,
    /// Its logs' incomplete tails.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tails: Vec<Tail>,
    /// Why it cannot be read, log by log.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub problems: Vec<Problem>,
    /// What `--fix` did with it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fix: Option<Fix>,
}

/// Every session checked, the totals by class, and the exit code.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Report {
    /// The sessions, by name.
    pub sessions: Vec<Checked>,
    /// How many sessions are of each class.
    pub totals: BTreeMap<Class, usize>,
    /// 0 when every session is ok, 1 when any needed or got fixing, 2 when
    /// any is newer than this build or unreadable.
    pub exit: u8,
}

/// The command line.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Args {
    /// The directory holding the sessions.
    pub sessions: PathBuf,
    /// Where `--fix` writes repaired copies.
    pub fix: Option<PathBuf>,
    /// Whether the report is one JSON document.
    pub json: bool,
}

/// The usage line.
pub const USAGE: &str = "usage: session-doctor <sessions-dir> [--fix <out-dir>] [--json]";

impl Args {
    /// Reads the arguments after the program's name.
    ///
    /// # Errors
    ///
    /// The usage line, when they are not `<sessions-dir> [--fix <out-dir>]
    /// [--json]`.
    pub fn parse(args: &[String]) -> Result<Self, String> {
        let mut parsed = Self::default();
        let mut sessions = None;
        let mut args = args.iter();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--json" => parsed.json = true,
                "--fix" => parsed.fix = Some(args.next().ok_or(USAGE)?.into()),
                _ if sessions.is_none() && !arg.starts_with("--") => sessions = Some(arg.into()),
                _ => return Err(USAGE.to_owned()),
            }
        }
        parsed.sessions = sessions.ok_or(USAGE)?;
        Ok(parsed)
    }
}

/// Runs the program on `args`: the report, as text or JSON, and the exit
/// code.
///
/// # Errors
///
/// When the arguments are wrong or the sessions directory cannot be listed.
pub fn run(args: &[String]) -> Result<(String, u8), String> {
    let args = Args::parse(args)?;
    let report = doctor(&args.sessions, args.fix.as_deref()).map_err(|error| error.to_string())?;
    let text = if args.json {
        serde_json::to_string_pretty(&report).map_err(|error| error.to_string())? + "\n"
    } else {
        render(&report)
    };
    Ok((text, report.exit))
}

/// Checks every session under `sessions`, writing fixed copies to `fix`.
///
/// # Errors
///
/// When `sessions` cannot be listed, or `fix` is `sessions` itself.
pub fn doctor(sessions: &Path, fix: Option<&Path>) -> io::Result<Report> {
    if let Some(out) = fix
        && fs::canonicalize(out).ok() == fs::canonicalize(sessions).ok()
    {
        let why = "the copies go to another directory than the sessions";
        return Err(io::Error::new(io::ErrorKind::InvalidInput, why));
    }
    let mut dirs: Vec<(String, PathBuf)> = fs::read_dir(sessions)?
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
        .map(|entry| {
            (
                entry.file_name().to_string_lossy().into_owned(),
                entry.path(),
            )
        })
        .collect();
    dirs.sort();
    let mut checked = Vec::new();
    for (name, dir) in dirs {
        let (mut session, read) = check(&name, &dir);
        if let (Some(out), Some(read)) = (fix, read.filter(|_| session.class.fixable())) {
            session.fix = Some(match repair(&dir, &out.join(&name), &read) {
                Ok(tails) => {
                    session.tails = tails;
                    Fix::Written {
                        path: out.join(&name),
                    }
                }
                Err(reason) => Fix::Failed { reason },
            });
        }
        checked.push(session);
    }
    let mut totals = BTreeMap::new();
    for session in &checked {
        *totals.entry(session.class).or_insert(0) += 1;
    }
    let worst = checked.iter().map(|session| session.class).max();
    let exit = match worst {
        Some(Class::OtherFormat | Class::Unreadable) => 2,
        Some(Class::Ok) | None => 0,
        Some(_) => 1,
    };
    Ok(Report {
        sessions: checked,
        totals,
        exit,
    })
}

/// The session `name` in `dir`, checked without writing to it, and the
/// session as read when it reads.
fn check(name: &str, dir: &Path) -> (Checked, Option<Session>) {
    let mut checked = Checked {
        session: name.to_owned(),
        class: Class::Ok,
        unknown: BTreeMap::new(),
        tails: Vec::new(),
        problems: Vec::new(),
        fix: None,
    };
    let read = DirStore::open(dir).and_then(|store| Session::read_lenient(&store));
    let (session, unreadable) = match read {
        Ok(read) => read,
        Err(error) => {
            checked.class = class_of(&error);
            checked.problems.push(problem(None, &error));
            return (checked, None);
        }
    };
    if !unreadable.is_empty() {
        checked.class = unreadable
            .iter()
            .map(|log| class_of(&log.error))
            .max()
            .unwrap_or(Class::Unreadable);
        let problems = unreadable.iter();
        checked.problems = problems
            .map(|log| problem(Some(&log.log), &log.error))
            .collect();
        return (checked, None);
    }
    checked.tails = tails(&session.repairs);
    checked.unknown =
        unknown(&session)
            .into_iter()
            .fold(BTreeMap::new(), |mut counts, (_, _, aside)| {
                *counts.entry(aside.kind).or_insert(0) += 1;
                counts
            });
    checked.class = if !checked.tails.is_empty() {
        Class::Truncated
    } else if !checked.unknown.is_empty() {
        Class::UnknownEntries
    } else {
        Class::Ok
    };
    (checked, Some(session))
}

/// The class of a session the library refuses with `error`.
fn class_of(error: &SessionError) -> Class {
    match error {
        SessionError::Format { found: Some(_), .. } | SessionError::OldLayout => Class::OtherFormat,
        _ => Class::Unreadable,
    }
}

/// Why `log` cannot be read, as `error` says.
fn problem(log: Option<&str>, error: &SessionError) -> Problem {
    let (log, line) = match error {
        SessionError::Record { log, line, .. } => (Some(log.as_str()), Some(*line)),
        SessionError::Format { log, .. } => (Some(log.as_str()), Some(1)),
        _ => (log, None),
    };
    Problem {
        log: log.map(str::to_owned),
        line,
        reason: error.to_string(),
    }
}

/// The torn tails of `repairs`.
fn tails(repairs: &[Repair]) -> Vec<Tail> {
    let tails = repairs.iter().filter_map(|repair| match repair {
        Repair::TornTail {
            log,
            kept,
            cut,
            set_aside,
        } => Some(Tail {
            log: log.clone(),
            kept: *kept,
            cut: *cut,
            set_aside: set_aside.clone(),
        }),
        Repair::LostImage { .. } => None,
    });
    tails.collect()
}

/// Every aside of `session` of a kind the library does not define, by
/// agent, in order.
fn unknown(session: &Session) -> Vec<(String, AsideId, Aside)> {
    let agents = session.agents.iter().flat_map(|agent| {
        let asides = agent.conversation.asides();
        asides
            .filter(|(_, aside)| !KNOWN.contains(&aside.kind.as_str()))
            .map(|(id, aside)| (agent.id.clone(), id, aside.clone()))
    });
    agents.collect()
}

/// Copies every log and file (blobs among them) of `from` to `to`.
fn copy_store(from: &dyn Store, to: &dyn Store) -> io::Result<()> {
    for log in from.logs()? {
        to.replace_log(&log, &from.read_log(&log)?)?;
    }
    for file in from.files()? {
        to.write_file(&file, &from.read_file(&file)?)?;
    }
    Ok(())
}

/// Writes the session in `dir`, read as `read`, repaired to `out`, and
/// checks it reads back the same: the tails it dropped. A copy that fails
/// is removed.
fn repair(dir: &Path, out: &Path, read: &Session) -> Result<Vec<Tail>, String> {
    if out.exists() {
        return Err(format!("{} exists; it is left as it is", out.display()));
    }
    let copied = write_copy(dir, out, read);
    if copied.is_err() {
        // Best effort: the copy is not kept either way.
        let _ = fs::remove_dir_all(out);
    }
    copied
}

/// Copies the session in `dir` to `out`, opens it there with a [`Writer`]
/// (which sets torn tails aside), and compares it with
/// `read`.
fn write_copy(dir: &Path, out: &Path, read: &Session) -> Result<Vec<Tail>, String> {
    let fail = |error: &dyn fmt::Display| format!("{error}");
    let from = DirStore::open(dir).map_err(|error| fail(&error))?;
    let to = DirStore::open(out).map_err(|error| fail(&error))?;
    copy_store(&from, &to).map_err(|error| fail(&error))?;
    let to: Arc<dyn Store> = Arc::new(to);
    let (_writer, opened) = Writer::open(Arc::clone(&to)).map_err(|error| fail(&error))?;
    let back = Session::read(&*to).map_err(|error| format!("the copy does not read: {error}"))?;
    match differs(read, &back) {
        Some(why) => Err(format!("the copy does not read back the same: {why}")),
        None => Ok(tails(&opened.repairs)),
    }
}

/// What an agent is compared by: its messages, its saved values, and its
/// whole conversation.
type Shape<'a> = (
    &'a [Arc<Message>],
    &'a BTreeMap<String, serde_json::Value>,
    &'a Conversation,
);

/// Each agent of `session` by id, with what it is compared by.
fn shapes(session: &Session) -> BTreeMap<&str, Shape<'_>> {
    let agents = session.agents.iter().map(|agent| {
        let conversation = &agent.conversation;
        let shape = (conversation.messages(), &agent.saved, conversation);
        (agent.id.as_str(), shape)
    });
    agents.collect()
}

/// How `back`, a copy read back, differs from `original`; `None` when it
/// holds the same agents, messages, saved values and unknown entries.
fn differs(original: &Session, back: &Session) -> Option<String> {
    let (was, now) = (shapes(original), shapes(back));
    if was.keys().ne(now.keys()) {
        return Some("other agents".to_owned());
    }
    for ((agent, was), now) in was.iter().zip(now.values()) {
        let what = if was.0 != now.0 {
            "messages"
        } else if was.1 != now.1 {
            "saved values"
        } else if was.2 != now.2 {
            "conversation"
        } else {
            continue;
        };
        return Some(format!("agent {agent}: other {what}"));
    }
    if original.saved != back.saved {
        return Some("other session values".to_owned());
    }
    (unknown(original) != unknown(back)).then(|| "other unknown entries".to_owned())
}

/// The report as text: a line per session, then the totals.
pub fn render(report: &Report) -> String {
    let mut out = String::new();
    for session in &report.sessions {
        let mut details: Vec<String> = Vec::new();
        details.extend(
            session
                .unknown
                .iter()
                .map(|(kind, count)| format!("{kind}: {count}")),
        );
        details.extend(session.tails.iter().map(|tail| {
            let kept = format!(
                "log {}: dropped {} bytes after {}",
                tail.log, tail.cut, tail.kept
            );
            match &tail.set_aside {
                Some(file) => format!("{kept} (set aside in files/{file})"),
                None => kept,
            }
        }));
        details.extend(session.problems.iter().map(|problem| {
            let at = match (&problem.log, problem.line) {
                (Some(log), Some(line)) => format!("log {log} line {line}: "),
                (Some(log), None) => format!("log {log}: "),
                _ => String::new(),
            };
            format!("{at}{}", problem.reason)
        }));
        match &session.fix {
            Some(Fix::Written { path }) => details.push(format!("fixed copy {}", path.display())),
            Some(Fix::Failed { reason }) => details.push(format!("no copy: {reason}")),
            None => {}
        }
        let _ = writeln!(
            out,
            "{:<24} {:<16} {}",
            session.session,
            session.class.name(),
            details.join("; ")
        );
    }
    let totals: Vec<String> = report
        .totals
        .iter()
        .map(|(class, count)| format!("{count} {class}"))
        .collect();
    let _ = writeln!(
        out,
        "{} sessions: {}",
        report.sessions.len(),
        totals.join(", ")
    );
    out
}

#[cfg(test)]
mod tests;
