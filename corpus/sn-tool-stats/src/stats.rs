//! The statistics themselves: what one call adds ([`Totals::of`]), the
//! table of totals by day, agent and tool ([`Table`]), and the masking that
//! groups similar error messages ([`mask`]).

use std::collections::BTreeMap;

use rig_harness::rig_core::effect::{Answered, Disposition, EffectKind, EffectRecord};
use rig_harness::rig_core::error::ErrorKind;
use serde::{Deserialize, Serialize};

/// How many distinct error messages a row keeps; the rarest go first.
const KEPT_MESSAGES: usize = 20;

/// The ratio between two duration buckets: a percentile is right to
/// within it.
const BUCKET: f64 = 1.25;

/// Milliseconds in a day.
pub(crate) const DAY_MS: u64 = 86_400_000;

/// What a row of totals is about: the day its calls finished (days since
/// the Unix epoch, UTC), the agent that made them, and the tool.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Key {
    /// Days since the Unix epoch.
    pub day: u64,
    /// The agent: its name in a session, `main` or `subagent` in a project.
    pub agent: String,
    /// The tool's name.
    pub tool: String,
}

/// The totals of some tool calls.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Totals {
    /// Every call.
    pub calls: u64,
    /// Calls answered with an error result.
    pub errors: u64,
    /// Calls interrupted or stopped while they ran or waited.
    pub cancelled: u64,
    /// Calls refused before they ran, by the user or by a rule.
    pub refused: u64,
    /// Calls with a result whose size is counted.
    pub sized: u64,
    /// The bytes of those results, together.
    pub result_bytes: u64,
    /// How many calls ran how long: by bucket, `BUCKET` apart.
    pub durations: BTreeMap<u32, u64>,
    /// How often each error message came, masked.
    pub messages: BTreeMap<String, u64>,
}

impl Totals {
    /// What the tool call of `record` adds, and its tool's name; `None`
    /// for another kind of effect, or a call another one replaced (that one
    /// counts).
    pub fn of(record: &EffectRecord) -> Option<(String, Self)> {
        let EffectKind::ToolCall { name, .. } = &record.kind else {
            return None;
        };
        if record.disposition == Disposition::Superseded {
            return None;
        }
        let mut totals = Self {
            calls: 1,
            ..Self::default()
        };
        let outcome = record.reviewed.as_ref().unwrap_or(&record.outcome);
        let cancelled = matches!(
            record.disposition,
            Disposition::Cancelled | Disposition::AnsweredWhileRunning
        ) || matches!(outcome, Err(report) if report.kind == ErrorKind::Cancelled);
        match outcome {
            _ if cancelled => totals.cancelled = 1,
            Err(report) if report.refusal => totals.refused = 1,
            Err(report) => totals.error(&report.message),
            Ok(Answered::ToolResult { result }) => {
                totals.sized = 1;
                totals.result_bytes = result.output().render().len() as u64;
                if result.is_refused() {
                    totals.refused = 1;
                } else if let Some(error) = result.error() {
                    totals.error(error.message());
                }
            }
            Ok(_) => {}
        }
        if let (Some(started), Some(finished)) = (record.started_ms, record.finished_ms) {
            totals
                .durations
                .insert(bucket(finished.saturating_sub(started)), 1);
        }
        Some((name.clone(), totals))
    }

    fn error(&mut self, message: &str) {
        self.errors = 1;
        self.messages.insert(mask(message), 1);
    }

    /// Adds `other` in.
    pub fn add(&mut self, other: &Self) {
        self.calls += other.calls;
        self.errors += other.errors;
        self.cancelled += other.cancelled;
        self.refused += other.refused;
        self.sized += other.sized;
        self.result_bytes += other.result_bytes;
        for (bucket, count) in &other.durations {
            *self.durations.entry(*bucket).or_default() += count;
        }
        for (message, count) in &other.messages {
            *self.messages.entry(message.clone()).or_default() += count;
        }
        while self.messages.len() > KEPT_MESSAGES {
            let rarest = self
                .messages
                .iter()
                .min_by_key(|(_, count)| **count)
                .map(|(message, _)| message.clone());
            if let Some(rarest) = rarest {
                self.messages.remove(&rarest);
            }
        }
    }

    /// The duration, in milliseconds, that a share `q` (0 to 1) of the
    /// timed calls took at most; `None` when none was timed.
    pub fn percentile_ms(&self, q: f64) -> Option<u64> {
        let timed: u64 = self.durations.values().sum();
        let rank = ((q * timed as f64).ceil() as u64).clamp(1, timed.max(1));
        let mut seen = 0;
        for (bucket, count) in &self.durations {
            seen += count;
            if seen >= rank {
                return Some(bucket_ms(*bucket));
            }
        }
        None
    }

    /// The average result size in bytes, if any result was sized.
    pub fn average_bytes(&self) -> Option<u64> {
        self.result_bytes.checked_div(self.sized)
    }

    /// The `n` most frequent error messages, most frequent first.
    pub fn top_messages(&self, n: usize) -> Vec<(String, u64)> {
        let mut messages: Vec<(String, u64)> = self
            .messages
            .iter()
            .map(|(message, count)| (message.clone(), *count))
            .collect();
        messages.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        messages.truncate(n);
        messages
    }
}

/// The bucket of a duration of `ms` milliseconds.
fn bucket(ms: u64) -> u32 {
    ((ms as f64 + 1.0).ln() / BUCKET.ln()).ceil() as u32
}

/// The longest duration of a bucket, in milliseconds.
fn bucket_ms(bucket: u32) -> u64 {
    (BUCKET.powi(bucket as i32) - 1.0).round() as u64
}

/// Totals by day, agent and tool.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "Vec<Row>", into = "Vec<Row>")]
pub struct Table(pub BTreeMap<Key, Totals>);

/// A [`Table`]'s row as it is written.
#[derive(Serialize, Deserialize)]
struct Row {
    key: Key,
    totals: Totals,
}

impl From<Vec<Row>> for Table {
    fn from(rows: Vec<Row>) -> Self {
        let mut table = Self::default();
        for row in rows {
            table.add(row.key, &row.totals);
        }
        table
    }
}

impl From<Table> for Vec<Row> {
    fn from(table: Table) -> Self {
        table
            .0
            .into_iter()
            .map(|(key, totals)| Row { key, totals })
            .collect()
    }
}

impl Table {
    /// Adds `totals` to the row of `key`.
    pub fn add(&mut self, key: Key, totals: &Totals) {
        self.0.entry(key).or_default().add(totals);
    }

    /// The totals of the rows from day `since` on, by tool, or by tool and
    /// agent with `by_agent`, ordered by tool.
    pub fn summed(&self, since: u64, by_agent: bool) -> BTreeMap<(String, Option<String>), Totals> {
        let mut summed: BTreeMap<(String, Option<String>), Totals> = BTreeMap::new();
        for (key, totals) in self.0.range(
            Key {
                day: since,
                agent: String::new(),
                tool: String::new(),
            }..,
        ) {
            let agent = by_agent.then(|| key.agent.clone());
            summed
                .entry((key.tool.clone(), agent))
                .or_default()
                .add(totals);
        }
        summed
    }
}

/// `message`'s first line with what differs between similar errors
/// masked: paths and file names as `<path>`, ids and hashes as `<id>`,
/// numbers as `#`.
pub fn mask(message: &str) -> String {
    let line = message.lines().next().unwrap_or_default();
    let masked: Vec<String> = line.split_whitespace().map(mask_word).collect();
    let masked = masked.join(" ");
    match masked.char_indices().nth(160) {
        Some((cut, _)) => format!("{}…", masked.get(..cut).unwrap_or_default()),
        None => masked,
    }
}

fn mask_word(word: &str) -> String {
    let quote = |c: char| "`'\"()[]{},;:<>".contains(c);
    let core = word.trim_matches(quote);
    let Some(start) = (!core.is_empty()).then(|| word.find(core)).flatten() else {
        return word.to_owned();
    };
    let (before, after) = (
        word.get(..start).unwrap_or_default(),
        word.get(start + core.len()..).unwrap_or_default(),
    );
    let masked = if is_path(core) {
        "<path>".to_owned()
    } else if is_id(core) {
        "<id>".to_owned()
    } else {
        numbers_masked(core)
    };
    format!("{before}{masked}{after}")
}

/// A path, or a file name with an extension.
fn is_path(word: &str) -> bool {
    if word.contains('/') || word.contains('\\') {
        return true;
    }
    let Some((stem, extension)) = word.rsplit_once('.') else {
        return false;
    };
    !stem.is_empty()
        && (1..=5).contains(&extension.len())
        && extension.chars().all(|c| c.is_ascii_alphanumeric())
        && !extension.chars().all(|c| c.is_ascii_digit())
}

/// A uuid, a hash or another long id with digits in it.
fn is_id(word: &str) -> bool {
    word.len() >= 8
        && word.chars().any(|c| c.is_ascii_digit())
        && word.chars().all(|c| c.is_ascii_hexdigit() || c == '-')
}

fn numbers_masked(word: &str) -> String {
    let mut masked = String::with_capacity(word.len());
    let mut in_number = false;
    for c in word.chars() {
        if c.is_ascii_digit() {
            if !in_number {
                masked.push('#');
            }
            in_number = true;
        } else {
            masked.push(c);
            in_number = false;
        }
    }
    masked
}
