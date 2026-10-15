//! What the plugin reads out of the session's effect log: a row per
//! effect, filtered, as a table, and one effect in full.

use std::collections::HashMap;
use std::fmt::Write;

use rig_core::effect::{Answered, Disposition, EffectKind, EffectRecord};
use rig_core::error::{ErrorKind, ErrorReport};
use rig_core::tool::ToolErrorKind;
use schemars::JsonSchema;
use serde::Deserialize;

/// The most characters `effects_detail` answers with.
pub const DETAIL_CHARS: usize = 20_000;

/// The rows listed when no limit is given.
const DEFAULT_LIMIT: usize = 30;

/// The rows `slowest` lists when no limit is given.
const SLOWEST: usize = 10;

/// How an effect ended.
#[derive(Deserialize, JsonSchema, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Ok,
    Error,
    Denied,
    Interrupted,
}

impl Status {
    fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Error => "error",
            Self::Denied => "denied",
            Self::Interrupted => "interrupted",
        }
    }

    /// How `record`'s effect ended: interrupted when it was cancelled,
    /// else as the answer that applied, as a reviewer left it if one edited it.
    fn of_record(record: &EffectRecord) -> Self {
        match record.disposition {
            Disposition::Cancelled => Self::Interrupted,
            _ => Self::of(record.reviewed.as_ref().unwrap_or(&record.outcome)),
        }
    }

    /// How `outcome` ended.
    fn of(outcome: &Result<Answered, ErrorReport>) -> Self {
        match outcome {
            Ok(Answered::ToolResult { result }) if result.is_refused() => Self::Denied,
            Ok(Answered::ToolResult { result }) => match result.error().map(|e| e.kind()) {
                None => Self::Ok,
                Some(ToolErrorKind::Cancelled) => Self::Interrupted,
                Some(ToolErrorKind::PermissionDenied) => Self::Denied,
                Some(_) => Self::Error,
            },
            Ok(Answered::Completion(response)) if response.aborted.is_some() => Self::Interrupted,
            Ok(Answered::Completion(response)) if response.error.is_some() => Self::Error,
            Ok(_) => Self::Ok,
            Err(report) => match report.kind {
                ErrorKind::Cancelled | ErrorKind::Tool(ToolErrorKind::Cancelled) => {
                    Self::Interrupted
                }
                ErrorKind::Denied | ErrorKind::Tool(ToolErrorKind::PermissionDenied) => {
                    Self::Denied
                }
                _ if report.refusal => Self::Denied,
                _ => Self::Error,
            },
        }
    }
}

/// Which effects to list.
#[derive(Deserialize, JsonSchema, Default, Debug)]
#[serde(deny_unknown_fields)]
pub struct Filter {
    /// Only this agent's effects: its name, or the start of its id.
    pub agent: Option<String>,
    /// `model`, `tool`, another effect kind (such as a plugin's own), or
    /// `any` (the default).
    pub kind: Option<String>,
    /// Only effects of this tool or model name.
    pub name: Option<String>,
    /// Only effects that ended so.
    pub status: Option<Status>,
    /// Only effects of this turn of their agent and later (the first is 0).
    pub since_turn: Option<u64>,
    /// The most rows, the latest kept (30 by default).
    pub limit: Option<usize>,
    /// The slowest first, by how long each took, instead of the latest;
    /// 10 rows unless `limit` says.
    #[serde(default)]
    pub slowest: bool,
}

/// The arguments of `effects_detail`.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DetailArgs {
    /// The effect's id, from `effects`.
    pub id: u64,
}

/// One effect, as listed.
struct Row<'a> {
    record: &'a EffectRecord,
    agent: String,
    kind: String,
    name: String,
    status: Status,
    /// How long it took, in milliseconds, when its start and end were
    /// recorded.
    took_ms: Option<u64>,
}

/// The effect kind as the filter names it.
fn kind_label(kind: &EffectKind) -> String {
    match kind {
        EffectKind::Completion { .. } => "model".to_owned(),
        EffectKind::ToolCall { .. } => "tool".to_owned(),
        other => other.name().to_string(),
    }
}

fn name_of(record: &EffectRecord) -> String {
    match &record.kind {
        EffectKind::ToolCall { name, .. } => name.clone(),
        EffectKind::Completion { request, .. } => request
            .model
            .clone()
            .unwrap_or_else(|| record.key.parts().label.to_string()),
        _ => record.key.parts().label.to_string(),
    }
}

/// Every effect of `records` as a row, in id order, with its agent's
/// name from `names` (by agent id).
fn rows<'a>(records: &'a [EffectRecord], names: &HashMap<String, String>) -> Vec<Row<'a>> {
    let mut rows: Vec<Row> = records
        .iter()
        .map(|record| {
            let scope = record.scope.as_deref().unwrap_or("");
            Row {
                record,
                agent: names
                    .get(scope)
                    .cloned()
                    .unwrap_or_else(|| scope.get(..8).unwrap_or(scope).to_owned()),
                kind: kind_label(&record.kind),
                name: name_of(record),
                status: Status::of_record(record),
                took_ms: record
                    .started_ms
                    .zip(record.finished_ms)
                    .map(|(started, finished)| finished.saturating_sub(started)),
            }
        })
        .collect();
    rows.sort_by_key(|row| row.record.id);
    rows
}

impl Filter {
    fn keeps(&self, row: &Row, scope: &str) -> bool {
        let agent = self.agent.as_deref().is_none_or(|agent| {
            row.agent.eq_ignore_ascii_case(agent) || (!agent.is_empty() && scope.starts_with(agent))
        });
        let kind = self
            .kind
            .as_deref()
            .is_none_or(|kind| kind == "any" || kind.eq_ignore_ascii_case(&row.kind));
        let name = self.name.as_deref().is_none_or(|name| name == row.name);
        let status = self.status.is_none_or(|status| status == row.status);
        let turn = self
            .since_turn
            .is_none_or(|since| row.record.turn.is_some_and(|turn| turn >= since));
        agent && kind && name && status && turn
    }

    /// Reads `/effects` arguments: `key=value` pairs, or bare words for a
    /// status, a kind or a name.
    pub fn parse(words: &str) -> Result<Self, String> {
        let mut filter = Self::default();
        for word in words.split_whitespace() {
            let (key, value) = word.split_once('=').unwrap_or(("", word));
            let status = || {
                serde_json::from_value(serde_json::Value::from(value))
                    .map_err(|_| format!("unknown status `{value}`"))
            };
            match key {
                "agent" => filter.agent = Some(value.to_owned()),
                "kind" => filter.kind = Some(value.to_owned()),
                "name" => filter.name = Some(value.to_owned()),
                "status" => filter.status = Some(status()?),
                "since_turn" | "turn" => {
                    filter.since_turn = Some(value.parse().map_err(|_| "a turn is a number")?);
                }
                "limit" => filter.limit = Some(value.parse().map_err(|_| "a limit is a number")?),
                "" if value == "slow" => filter.slowest = true,
                "" if matches!(value, "ok" | "error" | "denied" | "interrupted") => {
                    filter.status = Some(status()?);
                }
                "" if matches!(value, "model" | "tool" | "any") => {
                    filter.kind = Some(value.to_owned());
                }
                "" => filter.name = Some(value.to_owned()),
                other => return Err(format!("unknown filter `{other}`")),
            }
        }
        Ok(filter)
    }
}

fn size(value: &impl serde::Serialize) -> usize {
    serde_json::to_string(value).map_or(0, |text| text.len())
}

/// The UTC time of day `ms` after the Unix epoch, as `HH:MM:SS`.
fn clock(ms: u64) -> String {
    let seconds = (ms / 1_000) % 86_400;
    format!(
        "{:02}:{:02}:{:02}",
        seconds / 3_600,
        seconds / 60 % 60,
        seconds % 60
    )
}

/// `ms` as seconds, such as `1.25s`.
fn seconds(ms: u64) -> String {
    format!("{:.2}s", ms as f64 / 1e3)
}

fn human(bytes: usize) -> String {
    match bytes {
        0..1_000 => format!("{bytes}B"),
        1_000..1_000_000 => format!("{:.1}kB", bytes as f64 / 1e3),
        _ => format!("{:.1}MB", bytes as f64 / 1e6),
    }
}

/// The effects of `records` that `filter` keeps, the latest `limit` or
/// the slowest, as a table; agents by their names in `names`.
pub fn table(records: &[EffectRecord], names: &HashMap<String, String>, filter: &Filter) -> String {
    let all = rows(records, names);
    let mut kept: Vec<&Row> = all
        .iter()
        .filter(|row| filter.keeps(row, row.record.scope.as_deref().unwrap_or("")))
        .collect();
    let shown = if filter.slowest {
        kept.sort_by_key(|row| std::cmp::Reverse(row.took_ms));
        let limit = filter.limit.unwrap_or(SLOWEST).max(1);
        kept.get(..limit.min(kept.len())).unwrap_or_default()
    } else {
        let limit = filter.limit.unwrap_or(DEFAULT_LIMIT).max(1);
        kept.get(kept.len().saturating_sub(limit)..)
            .unwrap_or_default()
    };
    let mut out = format!(
        "{} of {} effects match ({} shown). Start times are UTC.\n",
        kept.len(),
        all.len(),
        shown.len()
    );
    out.push_str(
        "id\tagent\tturn\tkind\tname\tstatus\tstart\tduration\trequest\tresult\tmodel\tcost\n",
    );
    for row in shown {
        let (model, cost) = match &row.record.outcome {
            Ok(Answered::Completion(response)) => (
                row.name.as_str(),
                response
                    .usage
                    .cost
                    .map_or("-".to_owned(), |cost| format!("${:.4}", cost.total)),
            ),
            _ => ("-", "-".to_owned()),
        };
        let (request, result) = (size(&row.record.kind), size(&row.record.outcome));
        let start = row.record.started_ms.map_or("-".to_owned(), clock);
        let took = row.took_ms.map_or("-".to_owned(), seconds);
        let _ = writeln!(
            out,
            "{}\t{}\t{}\t{}\t{}\t{}\t{start}\t{took}\t{}\t{}\t{model}\t{cost}",
            row.record.id.as_u64(),
            row.agent,
            row.record
                .turn
                .map_or("-".to_owned(), |turn| turn.to_string()),
            row.kind,
            row.name,
            row.status.as_str(),
            human(request),
            human(result),
        );
    }
    out
}

/// The effect `id` of `records` in full: its request and its result, cut
/// to [`DETAIL_CHARS`].
pub fn detail(records: &[EffectRecord], id: u64) -> Option<String> {
    let record = records.iter().find(|record| record.id.as_u64() == id)?;
    let request = match &record.kind {
        EffectKind::ToolCall { name, args } => format!("{name} {args}"),
        kind => pretty(kind),
    };
    let result = match &record.outcome {
        Ok(Answered::ToolResult { result }) => match result.error().or(result.refusal()) {
            Some(error) => format!(
                "{}: {}",
                Status::of(&record.outcome).as_str(),
                error.message()
            ),
            None => pretty(result.output()),
        },
        Ok(outcome) => pretty(outcome),
        Err(report) => format!("{}: {}", report.kind.code(), report.message),
    };
    let text = format!(
        "effect {} ({}, {})\n\nrequest:\n{request}\n\nresult:\n{result}",
        record.id.as_u64(),
        name_of(record),
        Status::of_record(record).as_str()
    );
    let mut cut: String = text.chars().take(DETAIL_CHARS).collect();
    if cut.len() < text.len() {
        cut.push_str("\n[cut at 20000 characters]");
    }
    Some(cut)
}

/// `value` as pretty JSON.
fn pretty(value: &impl serde::Serialize) -> String {
    serde_json::to_string_pretty(value).unwrap_or_default()
}
