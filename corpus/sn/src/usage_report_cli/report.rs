//! Reading the saved sessions' effect logs into rows, and writing them.

use std::collections::BTreeMap;
use std::fmt::Write;
use std::io;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Local, NaiveDate, TimeZone};
use rig_cassette::effect_log::EffectLog;
use rig_cassette::effect_log::jsonl::{self, SESSION_EFFECT_LOG};
use rig_core::catalog::Catalog;
use rig_core::completion::Usage;
use rig_core::effect::{Answered, Disposition, EffectKind, EffectRecord, FamilyDescriptor};
use rig_harness::harness_protocol::{Home, SessionDir, SessionId};
use rig_harness::prelude::StoreOpener;
use serde::Serialize;

use crate::usage_report_cli::options::{By, Format, Options, Projects, Since};

/// What one row sums.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Row {
    /// Model requests a handler ran.
    pub requests: u64,
    /// Their input tokens, cached ones included.
    pub input_tokens: u64,
    /// The part of the input read from a cache.
    pub cached_input_tokens: u64,
    /// Their output tokens.
    pub output_tokens: u64,
    /// Their cost in USD.
    pub cost_usd: f64,
    /// Whether some request had no recorded price, so its cost is the
    /// current price table's, or none.
    pub cost_approximate: bool,
    /// Tool calls the models made.
    pub tool_calls: u64,
    /// How long requests and tool calls ran, overlaps counted once per
    /// session, in seconds.
    pub active_seconds: f64,
}

impl Row {
    fn add(&mut self, other: &Row) {
        self.requests += other.requests;
        self.input_tokens += other.input_tokens;
        self.cached_input_tokens += other.cached_input_tokens;
        self.output_tokens += other.output_tokens;
        self.cost_usd += other.cost_usd;
        self.cost_approximate |= other.cost_approximate;
        self.tool_calls += other.tool_calls;
        self.active_seconds += other.active_seconds;
    }
}

/// The rows of a report, by key, and their totals.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Report {
    /// What the rows are by.
    pub by: By,
    /// Each row, in the order shown.
    pub rows: Vec<(String, Row)>,
    /// Every row summed.
    pub totals: Row,
}

/// One request or tool call of a session, placed in its row.
struct Spent {
    key: String,
    row: Row,
    ran: Option<(u64, u64)>,
}

impl Report {
    /// The report `options` ask for over the sessions of `home`, with
    /// `root` the current directory, each session opened by `stores`,
    /// requests without a recorded price priced at `catalog`'s, at `now`;
    /// and a warning for each session skipped.
    pub fn collect(
        options: &Options,
        home: &Home,
        root: &Path,
        stores: &StoreOpener,
        catalog: &Catalog,
        now: DateTime<Local>,
    ) -> (Self, Vec<String>) {
        let from = match options.since {
            Since::Date(day) => start_of(day),
            Since::Days(days) => Some(now - chrono::Duration::days(days.into())),
        }
        .map_or(0, |from| from.timestamp_millis());
        let until = options
            .until
            .and_then(|day| start_of(day.succ_opt()?))
            .map_or(i64::MAX, |until| until.timestamp_millis());
        let wanted = match &options.projects {
            Projects::All => None,
            Projects::Current => Some(real(root)),
            Projects::In(path) => Some(real(&root.join(path))),
        };
        let mut warnings = Vec::new();
        // Per row, per session: the rows summed and when each part ran.
        let mut rows: BTreeMap<String, (Row, BTreeMap<String, Vec<(u64, u64)>>)> = BTreeMap::new();
        for (id, dir) in saved_sessions(home) {
            let project = std::fs::read_to_string(dir.directory())
                .ok()
                .map(|text| real(Path::new(text.trim_end_matches('\n'))));
            if let Some(wanted) = &wanted
                && !project.as_ref().is_some_and(|dir| dir.starts_with(wanted))
            {
                continue;
            }
            let log = match read_log(stores, &dir) {
                Ok(Some(log)) => log,
                Ok(None) => continue,
                Err(why) => {
                    warnings.push(format!(
                        "skipped session {id}: its {SESSION_EFFECT_LOG} cannot be read: {why}"
                    ));
                    continue;
                }
            };
            let project =
                project.map_or_else(|| "(unknown)".to_owned(), |p| p.display().to_string());
            for spent in spending(&log, options.by, id.as_str(), &project, catalog) {
                let Some((start, end)) = spent.ran else {
                    continue;
                };
                let at = i64::try_from(end).unwrap_or(i64::MAX);
                if at < from || at >= until {
                    continue;
                }
                let (row, ran) = rows.entry(spent.key).or_default();
                row.add(&spent.row);
                ran.entry(id.to_string()).or_default().push((start, end));
            }
        }
        let mut rows: Vec<(String, Row)> = rows
            .into_iter()
            .map(|(key, (mut row, ran))| {
                row.active_seconds = ran.into_values().map(covered).sum::<u64>() as f64 / 1000.0;
                (key, row)
            })
            .collect();
        if options.by != By::Day {
            rows.sort_by(|(_, a), (_, b)| b.cost_usd.total_cmp(&a.cost_usd));
        }
        let mut totals = Row::default();
        rows.iter().for_each(|(_, row)| totals.add(row));
        let report = Self {
            by: options.by,
            rows,
            totals,
        };
        (report, warnings)
    }

    /// The report as `format` writes it.
    pub fn render(&self, format: Format) -> String {
        match format {
            Format::Table => self.table(),
            Format::Csv => self.csv(),
            Format::Json => self.json(),
        }
    }

    fn table(&self) -> String {
        let header = [
            self.by.name(),
            "requests",
            "input",
            "cached input",
            "output",
            "cost",
            "tool calls",
            "active",
        ];
        let line = |key: &str, row: &Row| {
            let cost = format!(
                "{}${:.4}",
                if row.cost_approximate { "~" } else { "" },
                row.cost_usd
            );
            [
                key.to_owned(),
                row.requests.to_string(),
                row.input_tokens.to_string(),
                row.cached_input_tokens.to_string(),
                row.output_tokens.to_string(),
                cost,
                row.tool_calls.to_string(),
                duration(row.active_seconds),
            ]
        };
        let mut lines = vec![header.map(str::to_owned)];
        lines.extend(self.rows.iter().map(|(key, row)| line(key, row)));
        lines.push(line("total", &self.totals));
        let mut widths = [0; 8];
        for cells in &lines {
            for (width, cell) in widths.iter_mut().zip(cells) {
                *width = (*width).max(cell.chars().count());
            }
        }
        let mut text = String::new();
        for cells in &lines {
            let mut cells = cells.iter().zip(widths);
            if let Some((key, width)) = cells.next() {
                write!(text, "{key:<width$}").ok();
            }
            for (cell, width) in cells {
                write!(text, "  {cell:>width$}").ok();
            }
            text.push('\n');
        }
        text
    }

    fn csv(&self) -> String {
        let mut text = format!(
            "{},requests,input_tokens,cached_input_tokens,output_tokens,cost_usd,\
             cost_approximate,tool_calls,active_seconds\n",
            self.by.name()
        );
        let rows = self.rows.iter().map(|(key, row)| (key.as_str(), row));
        for (key, row) in rows.chain([("total", &self.totals)]) {
            let key = if key.contains([',', '"', '\n']) {
                format!("\"{}\"", key.replace('"', "\"\""))
            } else {
                key.to_owned()
            };
            writeln!(
                text,
                "{key},{},{},{},{},{:.6},{},{},{:.3}",
                row.requests,
                row.input_tokens,
                row.cached_input_tokens,
                row.output_tokens,
                row.cost_usd,
                row.cost_approximate,
                row.tool_calls,
                row.active_seconds
            )
            .ok();
        }
        text
    }

    fn json(&self) -> String {
        let rows: Vec<serde_json::Value> = self
            .rows
            .iter()
            .map(|(key, row)| {
                let mut value = serde_json::to_value(row).unwrap_or_default();
                if let Some(fields) = value.as_object_mut() {
                    fields.insert("key".to_owned(), key.clone().into());
                }
                value
            })
            .collect();
        let report = serde_json::json!({
            "by": self.by.name(),
            "rows": rows,
            "totals": self.totals,
        });
        format!("{report:#}\n")
    }
}

/// The start of local day `day`.
fn start_of(day: NaiveDate) -> Option<DateTime<Local>> {
    Local
        .from_local_datetime(&day.and_hms_opt(0, 0, 0)?)
        .earliest()
}

/// `path` with links resolved, when it exists.
fn real(path: &Path) -> PathBuf {
    path.canonicalize()
        .unwrap_or_else(|_| std::path::absolute(path).unwrap_or_else(|_| path.to_owned()))
}

/// Every saved session of `home`, by id.
fn saved_sessions(home: &Home) -> Vec<(SessionId, SessionDir)> {
    let entries = std::fs::read_dir(home.sessions())
        .into_iter()
        .flatten()
        .flatten();
    let mut sessions: Vec<_> = entries
        .filter_map(|entry| entry.file_name().to_str()?.parse::<SessionId>().ok())
        .map(|id| (id.clone(), home.session(&id)))
        .filter(|(_, dir)| dir.is_saved())
        .collect();
    sessions.sort_by(|(a, _), (b, _)| a.as_str().cmp(b.as_str()));
    sessions
}

/// The session's effect log, or `None` when it has none. A line being
/// written as it is read, by a session that is open, is left for later.
fn read_log(stores: &StoreOpener, dir: &SessionDir) -> io::Result<Option<EffectLog>> {
    let store = stores.open(dir)?;
    let mut bytes = match store.read_file(SESSION_EFFECT_LOG) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    let whole = bytes
        .iter()
        .rposition(|byte| *byte == b'\n')
        .map_or(0, |at| at + 1);
    bytes.truncate(whole);
    jsonl::read_from(bytes.as_slice()).map(Some)
}

/// Each request and tool call of `log`, the session `session` of
/// `project`, in its row by `by`.
fn spending(
    log: &EffectLog,
    by: By,
    session: &str,
    project: &str,
    catalog: &Catalog,
) -> Vec<Spent> {
    let mut records: Vec<&EffectRecord> = log.records.iter().collect();
    records.sort_by_key(|record| record.id);
    // The model each agent last asked: its tool calls are that model's.
    let mut asked: BTreeMap<Option<&str>, String> = BTreeMap::new();
    let mut spent = Vec::new();
    for record in records {
        let ran = record
            .finished_ms
            .map(|end| (record.started_ms.unwrap_or(end).min(end), end));
        let agent = record.scope.as_deref();
        let (model, row) = match (&record.kind, &record.outcome) {
            (EffectKind::Completion { .. }, Ok(Answered::Completion(response)))
                if record.disposition == Disposition::Ran =>
            {
                let model = model_of(log, record).unwrap_or_else(|| {
                    format!("{}/{}", response.origin.provider, response.origin.model)
                });
                let row = requested(&response.usage, &model, catalog);
                asked.insert(agent, model.clone());
                (model, row)
            }
            (EffectKind::ToolCall { .. }, _) => {
                let model = asked
                    .get(&agent)
                    .cloned()
                    .unwrap_or_else(|| "(none)".to_owned());
                let row = Row {
                    tool_calls: 1,
                    ..Row::default()
                };
                (model, row)
            }
            _ => continue,
        };
        let day = ran
            .and_then(|(_, end)| {
                Local
                    .timestamp_millis_opt(i64::try_from(end).ok()?)
                    .single()
            })
            .map(|at| at.date_naive().to_string());
        let key = match by {
            By::Day => day.unwrap_or_default(),
            By::Project => project.to_owned(),
            By::Model => model,
            By::Session => session.to_owned(),
        };
        spent.push(Spent { key, row, ran });
    }
    spent
}

/// The model the log's header names for `record`'s handler.
fn model_of(log: &EffectLog, record: &EffectRecord) -> Option<String> {
    let handler = log
        .header
        .handlers
        .iter()
        .find(|handler| handler.key == record.key)?;
    match &handler.family {
        FamilyDescriptor::Completion { model, .. } => Some(model.as_str().to_owned()),
        _ => None,
    }
}

/// One request's row: its tokens, and its recorded cost, else `catalog`'s
/// price of `model`, approximate.
fn requested(usage: &Usage, model: &str, catalog: &Catalog) -> Row {
    let (cost, approximate) = match usage.cost {
        Some(cost) => (cost.total, !cost.is_complete()),
        None => {
            let priced = catalog
                .resolve(model)
                .ok()
                .and_then(|found| found.spec.pricing?.cost(usage));
            (priced.map_or(0.0, |cost| cost.total), true)
        }
    };
    Row {
        requests: 1,
        input_tokens: usage.input_tokens.unwrap_or(0),
        cached_input_tokens: usage.cached_input_tokens.unwrap_or(0),
        output_tokens: usage.output_tokens.unwrap_or(0),
        cost_usd: cost,
        cost_approximate: approximate,
        ..Row::default()
    }
}

/// The milliseconds `spans` cover, each overlap once.
fn covered(mut spans: Vec<(u64, u64)>) -> u64 {
    spans.sort_unstable();
    let mut total = 0;
    let mut open: Option<(u64, u64)> = None;
    for (start, end) in spans {
        open = match open {
            Some((from, to)) if start <= to => Some((from, to.max(end))),
            Some((from, to)) => {
                total += to - from;
                Some((start, end))
            }
            None => Some((start, end)),
        };
    }
    total + open.map_or(0, |(from, to)| to - from)
}

/// `seconds` as `1h02m03s`, `2m03s` or `3s`.
fn duration(seconds: f64) -> String {
    let seconds = seconds.round() as u64;
    let (hours, minutes, seconds) = (seconds / 3600, seconds / 60 % 60, seconds % 60);
    match (hours, minutes) {
        (0, 0) => format!("{seconds}s"),
        (0, _) => format!("{minutes}m{seconds:02}s"),
        _ => format!("{hours}h{minutes:02}m{seconds:02}s"),
    }
}
