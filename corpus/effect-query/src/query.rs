//! What the plugin reads out of the session's effect log: a row per
//! effect, filtered, as a table, and one effect in full.

use std::collections::HashMap;
use std::fmt::Write;

use rig_cassette::effect_log::EffectLog;
use rig_core::effect::{EffectKind, EffectRecord, Outcome};
use rig_core::error::{ErrorKind, ErrorReport};
use rig_core::message::{Message, UserContent};
use rig_core::tool::ToolErrorKind;
use schemars::JsonSchema;
use serde::Deserialize;

/// The most characters `effects_detail` answers with.
pub const DETAIL_CHARS: usize = 20_000;

/// The rows listed when no limit is given.
const DEFAULT_LIMIT: usize = 30;

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

    /// How `outcome` ended.
    fn of(outcome: &Result<Outcome, ErrorReport>) -> Self {
        match outcome {
            Ok(Outcome::ToolResult { result }) if result.is_refused() => Self::Denied,
            Ok(Outcome::ToolResult { result }) => match result.error().map(|e| e.kind()) {
                None => Self::Ok,
                Some(ToolErrorKind::Cancelled) => Self::Interrupted,
                Some(ToolErrorKind::PermissionDenied) => Self::Denied,
                Some(_) => Self::Error,
            },
            Ok(Outcome::Completion(response)) if response.aborted.is_some() => Self::Interrupted,
            Ok(Outcome::Completion(response)) if response.error.is_some() => Self::Error,
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
    /// Only effects of this turn of their agent and later (from 1).
    pub since_turn: Option<u32>,
    /// The most rows, the latest kept (30 by default).
    pub limit: Option<usize>,
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
    turn: u32,
    kind: String,
    name: String,
    status: Status,
}

/// The effect kind as the filter names it.
fn kind_label(kind: &EffectKind) -> String {
    match kind {
        EffectKind::Completion { .. } => "model".to_owned(),
        EffectKind::ToolCall { .. } => "tool".to_owned(),
        EffectKind::Custom { kind, .. } => kind.to_string(),
        other => other.family().to_string(),
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

/// Whether a model request starts a turn: its last message is the user's
/// (or a delivered) message rather than tool results.
fn starts_turn(kind: &EffectKind) -> bool {
    let EffectKind::Completion { request, .. } = kind else {
        return false;
    };
    matches!(request.chat_history.last(), Some(Message::User { content })
        if !content.iter().any(|item| matches!(item, UserContent::ToolResult(_))))
}

/// Every effect of `log` as a row, in id order, with its agent's name from
/// `names` (agent id to name) and the turn of its agent it belongs to.
fn rows<'a>(log: &'a EffectLog, names: &HashMap<String, String>) -> Vec<Row<'a>> {
    let mut records: Vec<&EffectRecord> = log.records.iter().collect();
    records.sort_by_key(|record| record.id);
    let mut turns: HashMap<&str, u32> = HashMap::new();
    let mut turn_of = HashMap::new();
    let mut rows = Vec::with_capacity(records.len());
    for record in records {
        let scope = record.scope.as_deref().unwrap_or("");
        let current = turns.entry(scope).or_default();
        if starts_turn(&record.kind) {
            *current += 1;
        }
        let turn = record
            .parent
            .filter(|_| !matches!(record.kind, EffectKind::Completion { .. }))
            .and_then(|parent| turn_of.get(&(scope, parent)).copied())
            .unwrap_or((*current).max(1));
        turn_of.insert((scope, record.id), turn);
        let agent = names
            .get(scope)
            .cloned()
            .unwrap_or_else(|| scope.get(..8).unwrap_or(scope).to_owned());
        rows.push(Row {
            record,
            agent,
            turn,
            kind: kind_label(&record.kind),
            name: name_of(record),
            status: Status::of(&record.outcome),
        });
    }
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
        let turn = self.since_turn.is_none_or(|since| row.turn >= since);
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

fn human(bytes: usize) -> String {
    match bytes {
        0..1_000 => format!("{bytes}B"),
        1_000..1_000_000 => format!("{:.1}kB", bytes as f64 / 1e3),
        _ => format!("{:.1}MB", bytes as f64 / 1e6),
    }
}

/// The effects of `log` that `filter` keeps, the latest `limit`, as a
/// table.
pub fn table(log: &EffectLog, names: &HashMap<String, String>, filter: &Filter) -> String {
    let all = rows(log, names);
    let kept: Vec<&Row> = all
        .iter()
        .filter(|row| filter.keeps(row, row.record.scope.as_deref().unwrap_or("")))
        .collect();
    let limit = filter.limit.unwrap_or(DEFAULT_LIMIT).max(1);
    let shown = kept
        .get(kept.len().saturating_sub(limit)..)
        .unwrap_or_default();
    let mut out = format!(
        "{} of {} effects match ({} shown). Start times and durations are not recorded.\n",
        kept.len(),
        all.len(),
        shown.len()
    );
    out.push_str("id\tagent\tturn\tkind\tname\tstatus\trequest\tresult\tmodel\tcost\n");
    for row in shown {
        let (model, cost) = match &row.record.outcome {
            Ok(Outcome::Completion(response)) => (
                row.name.as_str(),
                response
                    .usage
                    .cost
                    .map_or("-".to_owned(), |cost| format!("${:.4}", cost.total)),
            ),
            _ => ("-", "-".to_owned()),
        };
        let (request, result) = (size(&row.record.kind), size(&row.record.outcome));
        let _ = writeln!(
            out,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{model}\t{cost}",
            row.record.id.as_u64(),
            row.agent,
            row.turn,
            row.kind,
            row.name,
            row.status.as_str(),
            human(request),
            human(result),
        );
    }
    out
}

/// The effect `id` of `log` in full: its request and its result, cut to
/// [`DETAIL_CHARS`].
pub fn detail(log: &EffectLog, id: u64) -> Option<String> {
    let record = log.records.iter().find(|record| record.id.as_u64() == id)?;
    let request = match &record.kind {
        EffectKind::ToolCall { name, args } => format!("{name} {args}"),
        kind => pretty(kind),
    };
    let result = match &record.outcome {
        Ok(Outcome::ToolResult { result }) => match result.error().or(result.refusal()) {
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
        Status::of(&record.outcome).as_str()
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
