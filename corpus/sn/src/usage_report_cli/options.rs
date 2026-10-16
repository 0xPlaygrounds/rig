//! The options of `rig usage`, and its help.

use std::path::PathBuf;

use chrono::NaiveDate;

/// What `rig usage --help` prints.
pub const HELP: &str = "\
Usage: rig usage [options]

What the saved sessions spent on models: requests, input tokens, cached
input tokens, output tokens, cost, tool calls and active time, with a
totals row. It reads the saved sessions only, makes no model call and
changes nothing, so it is safe while sessions are open: an open session's
requests so far are counted. Subagents' requests and the model calls
plugins make are counted with the session that made them.

Options:
  --since DATE|Nd        From DATE (YYYY-MM-DD, local time) or the last N
                         days, such as 30d (default: 7d).
  --until DATE           Up to and including DATE (default: now).
  --by day|project|model|session
                         One row per local day, project directory, model
                         (vendor/model) or session (default: day).
  --project PATH         Sessions started in PATH or under it (default:
                         the current directory).
  --all-projects         Every session, wherever it was started.
  --format table|csv|json
                         A table (default), CSV with a header row, or one
                         JSON object with `rows` and `totals`.
  --fail-over DOLLARS    Exit with status 3 when the total cost is over
                         DOLLARS, after printing the report.
  -h, --help             This help.

Cost is the price recorded with each request. A request saved without one
is priced at the current price table, and its cost is approximate: `~` in
the table, `cost_approximate` in CSV and JSON. Active time is the time
model requests and tool calls ran, overlaps counted once per session.

A damaged or unreadable session is skipped with a warning on stderr;
stdout carries only the report.

Exit status: 0, 2 for a bad option, 3 over --fail-over.
";

/// When the report starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Since {
    /// At the start of this local day.
    Date(NaiveDate),
    /// This many days before now.
    Days(u32),
}

/// What one row of the report sums.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum By {
    /// A local day, `YYYY-MM-DD`.
    #[default]
    Day,
    /// The directory a session was started in.
    Project,
    /// The model that answered, `vendor/model`.
    Model,
    /// A session, by id.
    Session,
}

impl By {
    /// The row key's column name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Day => "day",
            Self::Project => "project",
            Self::Model => "model",
            Self::Session => "session",
        }
    }
}

/// How the report is written.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Format {
    /// Aligned columns for a person.
    #[default]
    Table,
    /// Comma-separated, with a header row.
    Csv,
    /// One JSON object.
    Json,
}

/// Which sessions the report reads.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Projects {
    /// Those started in the current directory or under it.
    #[default]
    Current,
    /// Those started in this directory or under it.
    In(PathBuf),
    /// Every one.
    All,
}

/// `rig usage`'s options.
#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    /// When the report starts.
    pub since: Since,
    /// The last local day it covers, if not today.
    pub until: Option<NaiveDate>,
    /// What one row sums.
    pub by: By,
    /// Which sessions it reads.
    pub projects: Projects,
    /// How it is written.
    pub format: Format,
    /// The total cost in USD over which the command exits with status 3.
    pub fail_over: Option<f64>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            since: Since::Days(7),
            until: None,
            by: By::default(),
            projects: Projects::default(),
            format: Format::default(),
            fail_over: None,
        }
    }
}

impl Options {
    /// Reads the words after `rig usage`; `None` asks for the help.
    ///
    /// # Errors
    ///
    /// Why an option is refused.
    pub fn parse(words: &[String]) -> Result<Option<Self>, String> {
        let mut options = Self::default();
        let mut project = None;
        let mut all = false;
        let mut words = words.iter().map(String::as_str);
        while let Some(word) = words.next() {
            let (name, inline) = match word.split_once('=') {
                Some((name, value)) => (name, Some(value)),
                None => (word, None),
            };
            let mut value = || {
                inline
                    .or_else(|| words.next())
                    .ok_or_else(|| format!("{name} needs a value"))
            };
            match name {
                "-h" | "--help" => return Ok(None),
                "--since" => options.since = since(value()?)?,
                "--until" => options.until = Some(date(value()?)?),
                "--by" => options.by = by(value()?)?,
                "--project" => project = Some(PathBuf::from(value()?)),
                "--all-projects" if inline.is_none() => all = true,
                "--format" => options.format = format(value()?)?,
                "--fail-over" => {
                    let text = value()?;
                    let dollars = text
                        .parse::<f64>()
                        .ok()
                        .filter(|d| d.is_finite() && *d >= 0.0);
                    let dollars = dollars
                        .ok_or_else(|| format!("--fail-over takes dollars, not `{text}`"))?;
                    options.fail_over = Some(dollars);
                }
                _ => return Err(format!("unknown option `{word}`")),
            }
        }
        options.projects = match (project, all) {
            (Some(_), true) => return Err("--project and --all-projects exclude each other".into()),
            (Some(path), false) => Projects::In(path),
            (None, true) => Projects::All,
            (None, false) => Projects::Current,
        };
        Ok(Some(options))
    }
}

fn date(text: &str) -> Result<NaiveDate, String> {
    NaiveDate::parse_from_str(text, "%Y-%m-%d")
        .map_err(|_| format!("`{text}` is not a date like 2026-10-01"))
}

fn since(text: &str) -> Result<Since, String> {
    match text.strip_suffix('d').map(str::parse::<u32>) {
        Some(Ok(days)) => Ok(Since::Days(days)),
        _ => date(text).map(Since::Date).map_err(|_| {
            format!("--since takes a date like 2026-10-01 or days like 7d, not `{text}`")
        }),
    }
}

fn by(text: &str) -> Result<By, String> {
    match text {
        "day" => Ok(By::Day),
        "project" => Ok(By::Project),
        "model" => Ok(By::Model),
        "session" => Ok(By::Session),
        _ => Err(format!(
            "--by takes day, project, model or session, not `{text}`"
        )),
    }
}

fn format(text: &str) -> Result<Format, String> {
    match text {
        "table" => Ok(Format::Table),
        "csv" => Ok(Format::Csv),
        "json" => Ok(Format::Json),
        _ => Err(format!("--format takes table, csv or json, not `{text}`")),
    }
}
