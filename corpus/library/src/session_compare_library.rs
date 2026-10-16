//! CT-24: `session-compare a b`, a plain program on rig's library crates
//! (rig-core, rig-memory, rig-tools; no Bevy, no harness) that compares two
//! saved sessions: where their main agents'
//! conversations part, what each side did (turns, replies, tool calls by
//! name, failed calls, tokens, cost, models), and which files each side's
//! file tools changed.
//!
//! A side is a session directory ([`DirStore`]).

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use rig_core::completion::UsageTotals;
use rig_core::conversation::Conversation;
use rig_core::message::{AssistantContent, Message, ToolCall, UserContent};
use rig_core::tool::Tool;
use rig_memory::{DirStore, Session, Upgrades, title_of};
use rig_tools::{Delete, DeleteArgs, Edit, EditArgs, Move, MoveArgs, Write, WriteArgs};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The key rig-telemetry's usage plugin saves an agent's spending under
/// (its type path), so a program without the harness can read it.
pub const SPENDING: &str = "rig_telemetry::usage::Spending";

/// The key rig-telemetry's usage plugin saves the session's spending
/// under, in the session's own log.
pub const SESSION_SPENDING: &str = "rig_telemetry::usage::SessionSpending";

/// What the usage plugin saves, as far as this program reads it.
#[derive(Deserialize, Default)]
#[serde(default)]
struct Spending {
    total: Option<UsageTotals>,
    models: BTreeMap<String, Value>,
}

/// One side of a comparison: a session's main agent.
#[derive(Clone, Debug)]
pub struct Side {
    /// What the side was named on the command line.
    pub label: String,
    /// The main agent's conversation.
    pub conversation: Conversation,
    /// The models the session's calls went to, or its main agent's model.
    pub models: Vec<String>,
    /// What the session's model calls used, when the session recorded it.
    pub spending: Option<UsageTotals>,
    /// Logs of the session that could not be read and were left out, and
    /// what reading it repaired.
    pub warnings: usize,
}

/// Loads the side `path`: a session directory.
///
/// # Errors
///
/// A message saying why, when `path` is no rig session or cannot be read.
pub fn load(path: &str) -> Result<Side, String> {
    let dir = Path::new(path);
    if !dir.is_dir() {
        let why = if dir.exists() {
            "is not a rig session directory"
        } else {
            "does not exist"
        };
        return Err(format!("{path} {why}"));
    }
    let store = DirStore::open(dir).map_err(|error| format!("{path}: {error}"))?;
    let (session, unreadable) =
        Session::read_lenient(&store).map_err(|error| format!("{path}: {error}"))?;
    let Some(agent) = session.main_agent() else {
        let why = unreadable.first().map_or_else(
            || "it holds no agent's log".to_owned(),
            |log| log.error.to_string(),
        );
        return Err(format!("{path} is not a readable rig session: {why}"));
    };
    let upgrades = Upgrades::default();
    let spending: Option<Spending> = session
        .get(SESSION_SPENDING, &upgrades)
        .or_else(|| agent.get(SPENDING, &upgrades))
        .and_then(Result::ok);
    Ok(Side {
        label: path.to_owned(),
        conversation: agent.conversation.clone(),
        models: spending
            .as_ref()
            .map(|spending| spending.models.keys().cloned().collect::<Vec<_>>())
            .filter(|models| !models.is_empty())
            .or_else(|| agent.model().map(|model| vec![model]))
            .unwrap_or_default(),
        spending: spending.and_then(|spending| spending.total),
        warnings: unreadable.len() + session.repairs.len(),
    })
}

/// What one side did.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Stats {
    /// What the side was named on the command line.
    pub label: String,
    /// Messages the user typed.
    pub user_turns: usize,
    /// The model's replies.
    pub model_replies: usize,
    /// Tool calls by tool name.
    pub tool_calls: BTreeMap<String, usize>,
    /// Tool calls whose result says they failed.
    pub failed_tool_calls: usize,
    /// Input tokens, when recorded.
    pub tokens_in: Option<u64>,
    /// Output tokens, when recorded.
    pub tokens_out: Option<u64>,
    /// Cost in USD, when recorded (a lower bound when some calls were not
    /// priced).
    pub cost: Option<f64>,
    /// The models used.
    pub models: Vec<String>,
    /// Logs left out and repairs made while reading the session.
    pub warnings: usize,
}

/// Where the two conversations part.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Divergence {
    /// The index of the first message that differs; those before are shared.
    pub index: usize,
    /// That message on side a, `None` when a ended there.
    pub a: Option<Excerpt>,
    /// That message on side b, `None` when b ended there.
    pub b: Option<Excerpt>,
}

/// A message, shortly.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Excerpt {
    /// `user`, `assistant`, `tool` (tool results) or `system`.
    pub role: String,
    /// The start of its text, on one line.
    pub text: String,
}

/// The files the sides' file tools changed.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Files {
    /// Changed only by side a.
    pub only_a: Vec<String>,
    /// Changed only by side b.
    pub only_b: Vec<String>,
    /// Changed by both.
    pub both: Vec<String>,
}

/// The comparison of two sides.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Comparison {
    /// Messages both sides share before they part.
    pub shared: usize,
    /// Where they part; `None` when the conversations are the same.
    pub divergence: Option<Divergence>,
    /// Side a.
    pub a: Stats,
    /// Side b.
    pub b: Stats,
    /// The files each side changed.
    pub files: Files,
}

/// Compares sides `a` and `b`.
pub fn compare(a: &Side, b: &Side) -> Comparison {
    let (ours, theirs) = (a.conversation.messages(), b.conversation.messages());
    let shared = ours
        .iter()
        .zip(theirs)
        .take_while(|(ours, theirs)| ours == theirs)
        .count();
    let divergence = (ours.len() != theirs.len() || shared < ours.len()).then(|| Divergence {
        index: shared,
        a: ours.get(shared).map(|message| excerpt(message)),
        b: theirs.get(shared).map(|message| excerpt(message)),
    });
    let (files_a, files_b) = (
        changed_files(&a.conversation),
        changed_files(&b.conversation),
    );
    Comparison {
        shared,
        divergence,
        a: stats(a),
        b: stats(b),
        files: Files {
            only_a: files_a.difference(&files_b).cloned().collect(),
            only_b: files_b.difference(&files_a).cloned().collect(),
            both: files_a.intersection(&files_b).cloned().collect(),
        },
    }
}

/// What `side` did.
fn stats(side: &Side) -> Stats {
    let mut stats = Stats {
        label: side.label.clone(),
        user_turns: side
            .conversation
            .entries()
            .filter(|entry| entry.is_users_own())
            .count(),
        models: side.models.clone(),
        warnings: side.warnings,
        tokens_in: side.spending.and_then(|spent| spent.tokens.input_tokens),
        tokens_out: side.spending.and_then(|spent| spent.tokens.output_tokens),
        cost: side.spending.map(|spent| spent.cost),
        ..Stats::default()
    };
    for message in side.conversation.messages() {
        match &**message {
            Message::Assistant(reply) => {
                stats.model_replies += 1;
                for call in reply.tool_calls() {
                    let name = call.function.name.to_string();
                    *stats.tool_calls.entry(name).or_default() += 1;
                }
            }
            Message::User { content } => {
                stats.failed_tool_calls += content
                    .iter()
                    .filter(
                        |item| matches!(item, UserContent::ToolResult(result) if result.is_error),
                    )
                    .count();
            }
            Message::System { .. } => {}
        }
    }
    stats
}

/// The files the file tools' calls of `conversation` changed: each call of
/// `write`, `edit`, `delete` or `move` whose result is no failure, by the
/// paths its arguments name.
fn changed_files(conversation: &Conversation) -> BTreeSet<String> {
    let failed: BTreeSet<_> = conversation
        .messages()
        .iter()
        .filter_map(|message| match &**message {
            Message::User { content } => Some(content),
            _ => None,
        })
        .flatten()
        .filter_map(|item| match item {
            UserContent::ToolResult(result) if result.is_error => Some(result.call.clone()),
            _ => None,
        })
        .collect();
    let calls = conversation
        .messages()
        .iter()
        .filter_map(|message| match &**message {
            Message::Assistant(reply) => Some(reply.tool_calls()),
            _ => None,
        });
    calls
        .flatten()
        .filter(|call| !failed.contains(&call.id))
        .flat_map(paths_changed)
        .collect()
}

/// The paths `call` changes, by its tool's typed arguments; none for a
/// tool that changes no file, or arguments that do not read.
fn paths_changed(call: &ToolCall) -> Vec<String> {
    let args = call.function.arguments_value();
    let read = |paths: Result<Vec<String>, serde_json::Error>| paths.unwrap_or_default();
    match call.function.name.as_str() {
        <Write as Tool>::NAME => {
            read(serde_json::from_value::<WriteArgs>(args).map(|args| vec![args.path]))
        }
        <Edit as Tool>::NAME => {
            read(serde_json::from_value::<EditArgs>(args).map(|args| vec![args.path]))
        }
        <Delete as Tool>::NAME => {
            read(serde_json::from_value::<DeleteArgs>(args).map(|args| vec![args.path]))
        }
        <Move as Tool>::NAME => {
            read(serde_json::from_value::<MoveArgs>(args).map(|args| vec![args.from, args.to]))
        }
        _ => Vec::new(),
    }
}

/// `message`, shortly.
fn excerpt(message: &Message) -> Excerpt {
    let (role, text) = match message {
        Message::System { content } => ("system", content.clone()),
        Message::User { content } => {
            let texts: Vec<&str> = content
                .iter()
                .filter_map(|item| match item {
                    UserContent::Text(text) => Some(text.text()),
                    _ => None,
                })
                .collect();
            if texts.is_empty() {
                let results = content.iter().filter_map(|item| match item {
                    UserContent::ToolResult(result) => {
                        let failed = if result.is_error { " (failed)" } else { "" };
                        Some(format!("result of {}{failed}", result.name))
                    }
                    _ => None,
                });
                ("tool", results.collect::<Vec<_>>().join(", "))
            } else {
                ("user", texts.join(" "))
            }
        }
        Message::Assistant(reply) => {
            let text = reply.text();
            let calls: Vec<String> = reply
                .content
                .iter()
                .filter_map(|part| match part {
                    AssistantContent::ToolCall(call) => {
                        let paths = paths_changed(call);
                        let name = &call.function.name;
                        Some(if paths.is_empty() {
                            format!("calls {name}")
                        } else {
                            format!("calls {name} {}", paths.join(" "))
                        })
                    }
                    _ => None,
                })
                .collect();
            let text = [text]
                .into_iter()
                .chain(calls)
                .filter(|part| !part.is_empty());
            ("assistant", text.collect::<Vec<_>>().join("; "))
        }
    };
    Excerpt {
        role: role.to_owned(),
        text: title_of(&text),
    }
}

/// `comparison` as text for a terminal.
pub fn render(comparison: &Comparison) -> String {
    let mut out = Vec::new();
    match &comparison.divergence {
        None => out.push(format!("identical: {} messages shared", comparison.shared)),
        Some(divergence) => {
            out.push(format!(
                "diverge at message {} ({} shared)",
                divergence.index, comparison.shared
            ));
            for (side, message) in [("a", &divergence.a), ("b", &divergence.b)] {
                out.push(match message {
                    Some(message) => format!("  {side}: [{}] {}", message.role, message.text),
                    None => format!("  {side}: (ends)"),
                });
            }
        }
    }
    for (side, stats) in [("a", &comparison.a), ("b", &comparison.b)] {
        out.push(format!("{side}: {}", stats.label));
        out.push(format!(
            "  {} user turns, {} model replies, {} tool calls ({} failed)",
            stats.user_turns,
            stats.model_replies,
            stats.tool_calls.values().sum::<usize>(),
            stats.failed_tool_calls
        ));
        if !stats.tool_calls.is_empty() {
            let calls = stats
                .tool_calls
                .iter()
                .map(|(name, n)| format!("{name} {n}"));
            out.push(format!("  calls: {}", calls.collect::<Vec<_>>().join(", ")));
        }
        let tokens = |tokens: Option<u64>| tokens.map_or("?".to_owned(), |n| n.to_string());
        let cost = stats
            .cost
            .map_or("?".to_owned(), |cost| format!("${cost:.4}"));
        out.push(format!(
            "  tokens in {} out {}, cost {cost}",
            tokens(stats.tokens_in),
            tokens(stats.tokens_out)
        ));
        if !stats.models.is_empty() {
            out.push(format!("  models: {}", stats.models.join(", ")));
        }
        if stats.warnings > 0 {
            out.push(format!(
                "  warnings: {} (logs left out or repaired)",
                stats.warnings
            ));
        }
    }
    let files = &comparison.files;
    for (what, paths) in [
        ("both", &files.both),
        ("only a", &files.only_a),
        ("only b", &files.only_b),
    ] {
        if !paths.is_empty() {
            out.push(format!("files {what}: {}", paths.join(", ")));
        }
    }
    out.push(String::new());
    out.join("\n")
}

/// Runs `session-compare` with `args` (without the program's name): the
/// report to print.
///
/// # Errors
///
/// A message saying why, for the wrong arguments or a side that does not
/// load ([`load`]).
pub fn run(args: &[String]) -> Result<String, String> {
    let json = args.iter().any(|arg| arg == "--json");
    let sides: Vec<&String> = args.iter().filter(|arg| *arg != "--json").collect();
    let [a, b] = sides.as_slice() else {
        return Err("usage: session-compare [--json] <a> <b>".to_owned());
    };
    let comparison = compare(&load(a)?, &load(b)?);
    if json {
        serde_json::to_string_pretty(&comparison)
            .map(|text| text + "\n")
            .map_err(|error| error.to_string())
    } else {
        Ok(render(&comparison))
    }
}

#[cfg(test)]
mod tests;
