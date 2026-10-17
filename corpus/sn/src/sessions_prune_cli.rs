//! SN-25: `rig sessions prune`, a maintenance command safe for cron.
//!
//! `rig sessions prune` deletes saved sessions of `RIG_HOME` by age
//! (`--older-than 30d` or a date), by count (`--keep-last N` per project,
//! the newest kept) or by total size (`--max-size 2GB`, oldest first), in
//! one project (`--project PATH`) or all of them. It never deletes a
//! session another rig runs (its launcher holds the session's lock), nor
//! one the user named (unless `--include-named`).
//!
//! The sessions are the home's saved ones, read as the session picker reads
//! them ([`SessionEntry`]), and each is deleted with the harness's
//! [`DeleteSession`]: its store wherever it is kept (logs,
//! plugins' data, listing) and its directory with the plain files beside
//! the store (spilled output, pasted images). That delete is crash-safe: a
//! session is marked deleting first, which hides it from listings and
//! `--resume`, and the next start finishes it. [`SessionDeleted`] says which
//! sessions went.
//!
//! Without `--yes`, a run at a terminal shows the plan and asks; one whose
//! stdin is not a terminal refuses with exit code 2. `--dry-run` only
//! shows the plan, and `--json` prints it, or the result, as JSON
//! ([`PruneReport`]).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io::{self, BufRead, Write as _};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use rig_basics::session_picker::SessionEntry;
use rig_harness::harness_protocol::SessionDir;
use rig_harness::prelude::*;
use serde::Serialize;

/// The command, and the front that runs it.
const COMMAND: &str = "sessions";
/// Its usage.
const USAGE: &str = "prune [--older-than 30d|DATE] [--keep-last N] [--max-size 2GB] \
                     [--project PATH] [--include-named] [--dry-run] [--yes] [--json]";

/// `rig sessions prune`.
#[derive(Default)]
pub struct SessionsPrunePlugin;

impl Plugin for SessionsPrunePlugin {
    fn build(&self, app: &mut App) {
        let front = Front::new(COMMAND).sessionless();
        app.add_front(front.on_command(
            COMMAND,
            USAGE,
            "Delete old saved sessions by age, count or size",
        ));
    }

    fn finish(&self, app: &mut App) {
        if app.runs_front(COMMAND) {
            app.add_systems(
                Startup,
                (
                    prune_command,
                    report_deleted.run_if(resource_exists::<Deleting>),
                )
                    .chain(),
            );
        }
    }
}

/// What `rig sessions prune` was asked to do.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Prune {
    /// Sessions last changed before this, in milliseconds since the Unix
    /// epoch, or this long before now.
    pub older_than: Option<Cutoff>,
    /// Keep this many of the newest sessions of each project.
    pub keep_last: Option<usize>,
    /// Delete the oldest sessions until all of them take at most this many
    /// bytes.
    pub max_size: Option<u64>,
    /// Only the sessions that ran in this directory.
    pub project: Option<PathBuf>,
    /// Named sessions may be deleted too.
    pub include_named: bool,
    /// Show the plan, delete nothing.
    pub dry_run: bool,
    /// Delete without asking.
    pub yes: bool,
    /// Print JSON.
    pub json: bool,
}

/// When a session counts as old.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cutoff {
    /// Not changed for this long.
    Age(Duration),
    /// Not changed since this time, in milliseconds since the Unix epoch.
    Before(u64),
}

impl Cutoff {
    /// The time, in milliseconds since the Unix epoch, before which a
    /// session is old, given `now`.
    pub fn before(self, now: u64) -> u64 {
        match self {
            Self::Age(age) => {
                now.saturating_sub(u64::try_from(age.as_millis()).unwrap_or(u64::MAX))
            }
            Self::Before(at) => at,
        }
    }
}

impl Prune {
    /// Reads the words after `rig sessions`.
    ///
    /// # Errors
    ///
    /// When they are not `prune` and its options, or give none of
    /// `--older-than`, `--keep-last` and `--max-size`.
    pub fn parse(words: &[String]) -> Result<Self, String> {
        let mut words = words.iter().map(String::as_str);
        if words.next() != Some("prune") {
            return Err("the only subcommand is `prune`".to_owned());
        }
        let mut prune = Self::default();
        while let Some(word) = words.next() {
            let (name, inline) = match word.split_once('=') {
                Some((name, value)) => (name, Some(value)),
                None => (word, None),
            };
            let mut value = || {
                inline
                    .or_else(|| words.next())
                    .ok_or(format!("{name} needs a value"))
            };
            match name {
                "--older-than" => prune.older_than = Some(cutoff(value()?)?),
                "--keep-last" => {
                    let count = value()?;
                    prune.keep_last = Some(
                        count
                            .parse()
                            .map_err(|_| format!("--keep-last takes a count, not `{count}`"))?,
                    );
                }
                "--max-size" => prune.max_size = Some(size(value()?)?),
                "--project" => prune.project = Some(PathBuf::from(value()?)),
                "--include-named" => prune.include_named = true,
                "--dry-run" => prune.dry_run = true,
                "--yes" => prune.yes = true,
                "--json" => prune.json = true,
                other => return Err(format!("unknown option `{other}`")),
            }
        }
        if prune.older_than.is_none() && prune.keep_last.is_none() && prune.max_size.is_none() {
            return Err("give at least one of --older-than, --keep-last and --max-size".to_owned());
        }
        Ok(prune)
    }
}

/// `30d`, `12h`, `2w`, or a date, `2026-01-31`.
fn cutoff(text: &str) -> Result<Cutoff, String> {
    let unit = |unit: char| match unit {
        's' => Some(1),
        'm' => Some(60),
        'h' => Some(3600),
        'd' => Some(86_400),
        'w' => Some(7 * 86_400),
        _ => None,
    };
    if let Some(last) = text.chars().last()
        && let Some(seconds) = unit(last)
        && let Ok(count) = text.trim_end_matches(last).parse::<u64>()
    {
        return Ok(Cutoff::Age(Duration::from_secs(
            count.saturating_mul(seconds),
        )));
    }
    let date = chrono::NaiveDate::parse_from_str(text, "%Y-%m-%d").map_err(|_| {
        format!("--older-than takes an age such as 30d or a date such as 2026-01-31, not `{text}`")
    })?;
    let at = date
        .and_time(chrono::NaiveTime::MIN)
        .and_utc()
        .timestamp_millis();
    Ok(Cutoff::Before(u64::try_from(at).unwrap_or_default()))
}

/// `2GB`, `500M`, `1GiB`, `4096`.
fn size(text: &str) -> Result<u64, String> {
    let split = text
        .find(|c: char| !c.is_ascii_digit() && c != '.')
        .unwrap_or(text.len());
    let (number, unit) = text.split_at(split);
    let factor: f64 = match unit.trim().to_ascii_lowercase().as_str() {
        "" | "b" => 1.0,
        "k" | "kb" => 1e3,
        "m" | "mb" => 1e6,
        "g" | "gb" => 1e9,
        "t" | "tb" => 1e12,
        "kib" => 1024.0,
        "mib" => 1024.0 * 1024.0,
        "gib" => 1024.0 * 1024.0 * 1024.0,
        _ => return Err(format!("--max-size takes a size such as 2GB, not `{text}`")),
    };
    let number: f64 = number
        .parse()
        .map_err(|_| format!("--max-size takes a size such as 2GB, not `{text}`"))?;
    // A size in bytes, rounded.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a size in bytes"
    )]
    Ok((number * factor).round().max(0.0) as u64)
}

/// Why a session that would be deleted is kept.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case", tag = "reason", content = "by")]
pub enum KeptBecause {
    /// A running rig has it open.
    InUse,
    /// The user named it, and `--include-named` was not given.
    Named,
}

impl std::fmt::Display for KeptBecause {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InUse => f.write_str("in use"),
            Self::Named => f.write_str("named"),
        }
    }
}

/// A saved session as the plan sees it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PlannedSession {
    /// Its id.
    pub id: String,
    /// The directory it ran in, if known.
    pub project: Option<String>,
    /// What it takes on disk.
    pub bytes: u64,
    /// When it last changed, in milliseconds since the Unix epoch.
    pub updated: u64,
    /// Its name, if the user gave it one.
    pub name: Option<String>,
}

/// A session the rules chose that is kept, and why.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Kept {
    /// The session.
    #[serde(flatten)]
    pub session: PlannedSession,
    /// Why it is kept.
    #[serde(flatten)]
    pub because: KeptBecause,
}

/// What one project loses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct ProjectTotal {
    /// Sessions deleted.
    pub sessions: usize,
    /// Their bytes.
    pub bytes: u64,
}

/// The plan of a prune, then what it did: [`Resource`] once the command
/// ran, and what `--json` prints.
#[derive(Resource, Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct PruneReport {
    /// Whether it deleted, not only planned.
    pub done: bool,
    /// The sessions to delete, or deleted, newest first.
    pub deleted: Vec<PlannedSession>,
    /// The sessions the rules chose that are kept, and why.
    pub kept: Vec<Kept>,
    /// What each project loses, by its directory (`(unknown)` without one).
    pub projects: BTreeMap<String, ProjectTotal>,
    /// The bytes freed, or that would be.
    pub bytes_freed: u64,
    /// The sessions that could not be deleted; a notice says why.
    pub failed: Vec<String>,
}

/// A saved session, read once.
struct Found {
    planned: PlannedSession,
    in_use: bool,
}

/// The saved sessions of `home` other than `current`, newest first, each
/// read as the session picker reads it.
fn found(home: &Home, stores: &StoreOpener, current: &Path) -> Vec<Found> {
    let mut sessions: Vec<(SessionEntry, SessionDir)> = home
        .saved_sessions()
        .into_iter()
        .filter(|(_, dir)| dir.path() != current)
        .filter_map(|(_, dir)| Some((SessionEntry::read(&dir, stores)?, dir)))
        .collect();
    sessions.sort_by_key(|(entry, _)| std::cmp::Reverse(entry.updated));
    sessions
        .into_iter()
        .map(|(entry, dir)| {
            // The directory it names, even one removed since: its sessions
            // are still one project's.
            let project = fs::read_to_string(dir.directory())
                .ok()
                .map(|text| text.trim_end_matches('\n').to_owned());
            Found {
                planned: PlannedSession {
                    id: entry.id.to_string(),
                    project,
                    bytes: bytes_under(dir.path()),
                    updated: millis(entry.updated),
                    name: entry.name,
                },
                in_use: entry.running,
            }
        })
        .collect()
}

impl PruneReport {
    /// Counts `session` as deleted.
    fn delete(&mut self, session: &PlannedSession) {
        let project = session
            .project
            .clone()
            .unwrap_or_else(|| "(unknown)".to_owned());
        let total = self.projects.entry(project).or_default();
        total.sessions += 1;
        total.bytes += session.bytes;
        self.bytes_freed += session.bytes;
        self.deleted.push(session.clone());
    }
}

/// Plans a prune of the saved sessions of `home`, but `current`, at `now`
/// (milliseconds since the Unix epoch).
pub fn plan(
    home: &Home,
    stores: &StoreOpener,
    current: &Path,
    prune: &Prune,
    now: u64,
) -> PruneReport {
    let sessions = found(home, stores, current);
    let project = prune.project.as_deref().map(normal);
    let in_scope = |session: &Found| {
        project.as_ref().is_none_or(|project| {
            session
                .planned
                .project
                .as_deref()
                .map(|dir| normal(Path::new(dir)))
                .as_ref()
                == Some(project)
        })
    };
    let protected = |session: &Found| {
        if session.in_use {
            Some(KeptBecause::InUse)
        } else if session.planned.name.is_some() && !prune.include_named {
            Some(KeptBecause::Named)
        } else {
            None
        }
    };
    let mut report = PruneReport::default();
    let mut deleted: HashSet<&str> = HashSet::new();
    let mut kept: HashMap<&str, KeptBecause> = HashMap::new();
    // By age and count: sessions every rule given chooses.
    let mut rank: HashMap<Option<&str>, usize> = HashMap::new();
    for session in sessions.iter().filter(|session| in_scope(session)) {
        let newer = rank.entry(session.planned.project.as_deref()).or_default();
        let old = prune
            .older_than
            .is_none_or(|cutoff| session.planned.updated < cutoff.before(now));
        let beyond = prune.keep_last.is_none_or(|keep| *newer >= keep);
        *newer += 1;
        if (prune.older_than.is_some() || prune.keep_last.is_some()) && old && beyond {
            choose(
                protected(session),
                &session.planned.id,
                (&mut deleted, &mut kept),
            );
        }
    }
    // By size: the oldest left, until the rest fits.
    if let Some(max) = prune.max_size {
        let scope = sessions.iter().filter(|session| in_scope(session));
        let mut left: u64 = scope
            .clone()
            .filter(|session| !deleted.contains(session.planned.id.as_str()))
            .map(|session| session.planned.bytes)
            .sum();
        for session in scope.rev() {
            if left <= max {
                break;
            }
            if !deleted.contains(session.planned.id.as_str()) {
                choose(
                    protected(session),
                    &session.planned.id,
                    (&mut deleted, &mut kept),
                );
                if deleted.contains(session.planned.id.as_str()) {
                    left = left.saturating_sub(session.planned.bytes);
                }
            }
        }
    }
    for session in &sessions {
        let planned = &session.planned;
        if deleted.contains(planned.id.as_str()) {
            report.delete(planned);
        } else if let Some(because) = kept.remove(planned.id.as_str()) {
            report.kept.push(Kept {
                session: planned.clone(),
                because,
            });
        }
    }
    report
}

/// Marks the session `id` deleted, or kept `because`.
fn choose<'a>(
    because: Option<KeptBecause>,
    id: &'a str,
    (deleted, kept): (&mut HashSet<&'a str>, &mut HashMap<&'a str, KeptBecause>),
) {
    match because {
        Some(because) => {
            kept.entry(id).or_insert(because);
        }
        None => {
            deleted.insert(id);
        }
    }
}

/// A plan whose sessions were each sent a [`DeleteSession`].
#[derive(Resource)]
struct Deleting(PruneReport, Prune);

/// `rig sessions prune …`: plans, then asks or refuses and exits, or sends
/// a [`DeleteSession`] for each session the plan deletes.
fn prune_command(
    invoked: Res<Invoked>,
    (home, stores, paths): (Res<RigHome>, Res<StoreOpener>, Res<SessionPaths>),
    mut commands: Commands,
) {
    let words = invoked.command().map_or(&[][..], |(_, words)| words);
    let prune = match Prune::parse(words) {
        Ok(prune) => prune,
        Err(why) => {
            eprintln!("rig sessions: {why}\nusage: rig sessions {USAGE}");
            commands.trigger(Exit(ExitReason::Failed(2)));
            return;
        }
    };
    let now = millis(SystemTime::now());
    let report = plan(&home.0, &stores, paths.path(), &prune, now);
    let reason = if prune.dry_run || report.deleted.is_empty() {
        show(&report, &prune);
        ExitReason::Quit
    } else if !prune.yes && !invoked.terminal {
        eprintln!(
            "rig sessions prune: would delete {} sessions ({}); pass --yes to delete without asking.",
            report.deleted.len(),
            bytes_label(report.bytes_freed)
        );
        ExitReason::Failed(2)
    } else if !prune.yes && !confirmed(&report, &prune) {
        eprintln!("Nothing deleted.");
        ExitReason::Failed(1)
    } else {
        for session in &report.deleted {
            if let Ok(id) = session.id.parse() {
                commands.trigger(DeleteSession(id));
            }
        }
        commands.insert_resource(Deleting(report, prune));
        return;
    };
    commands.insert_resource(report);
    commands.trigger(Exit(reason));
}

/// Reports which sessions of the plan were deleted ([`SessionDeleted`]),
/// shows why the others were not, and exits.
fn report_deleted(
    deleting: Res<Deleting>,
    mut deleted: MessageReader<SessionDeleted>,
    mut notices: MessageReader<Notice>,
    mut commands: Commands,
) {
    let Deleting(planned, prune) = &*deleting;
    let gone: HashSet<String> = deleted.read().map(|gone| gone.0.to_string()).collect();
    let mut report = PruneReport {
        done: true,
        kept: planned.kept.clone(),
        ..PruneReport::default()
    };
    for session in &planned.deleted {
        if gone.contains(&session.id) {
            report.delete(session);
        } else {
            report.failed.push(session.id.clone());
        }
    }
    for notice in notices.read() {
        if notice.level == NoticeLevel::Error {
            eprintln!("rig: {}", notice.text);
        }
    }
    show(&report, prune);
    let reason = if report.failed.is_empty() {
        ExitReason::Quit
    } else {
        ExitReason::Failed(1)
    };
    commands.remove_resource::<Deleting>();
    commands.insert_resource(report);
    commands.trigger(Exit(reason));
}

/// Shows the plan, asks, and whether the answer was yes.
fn confirmed(report: &PruneReport, prune: &Prune) -> bool {
    if !prune.json {
        print!("{}", plan_text(report));
    }
    print!(
        "Delete {} sessions ({})? [y/N] ",
        report.deleted.len(),
        bytes_label(report.bytes_freed)
    );
    io::stdout().flush().ok();
    let mut answer = String::new();
    io::stdin().lock().read_line(&mut answer).ok();
    matches!(answer.trim(), "y" | "Y" | "yes")
}

/// Prints the plan or the result.
fn show(report: &PruneReport, prune: &Prune) {
    if prune.json {
        println!(
            "{}",
            serde_json::to_string_pretty(report).unwrap_or_default()
        );
        return;
    }
    if report.deleted.is_empty() && report.failed.is_empty() {
        println!("Nothing to prune.");
    } else if !report.done {
        print!("{}", plan_text(report));
    } else {
        println!(
            "Deleted {} sessions, freed {}.",
            report.deleted.len(),
            bytes_label(report.bytes_freed)
        );
        for id in &report.failed {
            println!("  could not delete {id}");
        }
    }
    for kept in &report.kept {
        println!("  kept {}: {}", kept.session.id, kept.because);
    }
}

/// The plan, a line per project.
fn plan_text(report: &PruneReport) -> String {
    let mut text = format!(
        "{} sessions, {} to delete:\n",
        report.deleted.len(),
        bytes_label(report.bytes_freed)
    );
    for (project, total) in &report.projects {
        text.push_str(&format!(
            "  {project}: {} sessions, {}\n",
            total.sessions,
            bytes_label(total.bytes)
        ));
    }
    text
}

/// `3.1 GB`.
fn bytes_label(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    #[expect(clippy::cast_precision_loss, reason = "shown to one decimal")]
    let mut value = bytes as f64;
    let mut unit = "B";
    for next in UNITS {
        unit = next;
        if value < 1000.0 {
            break;
        }
        value /= 1000.0;
    }
    if unit == "B" {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {unit}")
    }
}

/// The bytes of every file under `path`.
fn bytes_under(path: &Path) -> u64 {
    let Ok(entries) = fs::read_dir(path) else {
        return 0;
    };
    entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let kind = entry.file_type().ok()?;
            Some(if kind.is_dir() {
                bytes_under(&entry.path())
            } else {
                entry.metadata().ok()?.len()
            })
        })
        .sum()
}

/// `path` made absolute and without `..`, where it can be.
fn normal(path: &Path) -> PathBuf {
    fs::canonicalize(path)
        .or_else(|_| std::path::absolute(path))
        .unwrap_or_else(|_| path.to_owned())
}

/// `at` in milliseconds since the Unix epoch.
fn millis(at: SystemTime) -> u64 {
    let since = at.duration_since(UNIX_EPOCH).unwrap_or_default();
    u64::try_from(since.as_millis()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests;
