//! OC-09: Edit the conversation in your editor (one-conversation fresh
//! held-out spec).
//!
//! `/edit-history` writes the focused agent's conversation to a Markdown
//! file, one section per message part (`## [n] user`, `## [n] assistant`,
//! `## [n] tool-result <tool> <call id>`), and opens the editor on it with
//! the terminal (rig-tui's `RunInTerminal`). Tool calls are fenced
//! ```` ```call <tool> <id> ```` blocks of JSON arguments; reasoning, images
//! and other parts are placeholder lines that keep the original part.
//!
//! On save the file is parsed, normalized (`pairing::normalize`, which
//! merges the model's turns left side by side and keeps ids) and saved as
//! one change of the conversation (`set_entries`): text edits, deleted
//! sections, changed call arguments.
//! A call and its result go together: deleting one side asks whether to
//! delete both or reopen the editor. A parse error reopens the editor with
//! the error on top; quitting without saving changes nothing.
//! `/edit-history undo` restores the version before the last edit, read
//! back from the session's log, its entries put back with their ids
//! (`restore`), keeping the turns that came after it. Neither runs while
//! the agent's turn does, nor without the terminal view.

#[cfg(test)]
mod tests;

use std::collections::{BTreeSet, HashSet};
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;

use rig_harness::prelude::*;
use rig_harness::rig_core::conversation::Conversation as History;
use rig_harness::rig_core::conversation::pairing::{self, PairingError};
use rig_harness::rig_core::message::{Message, Text, ToolResult as ResultPart, ToolResultContent};
use rig_harness::rig_ecs::session_log::SessionStore;
use rig_harness::rig_memory::{LogRecord, SessionError};
use rig_tui::{RunInTerminal, TerminalExit};
use serde::Deserialize;
use serde_json::{Map, Value};

/// Who an edit is recorded as made by.
const PLUGIN: &str = "edit-history";
/// Who an undo is recorded as made by.
const UNDO: &str = "edit-history-undo";
/// The dialog's choice that deletes both sides of a call.
const DELETE_BOTH: &str = "delete-both";
/// The dialog's choice that reopens the editor.
const REOPEN: &str = "reopen";
/// The title of the dialog asked when one side of a call was deleted.
pub const DIALOG_TITLE: &str = "Delete the tool call and its result together?";

/// What the file says above the first section.
const PREAMBLE: &str = "\
# Edit the conversation, save and quit to apply; quit without saving to change nothing.
# Edit text freely, delete whole sections, change a call's JSON arguments.
# Headers stay as they are and in order; sections cannot be added.
# A call and its result go together. `<<...>>` lines keep that part as it was;
# delete one to drop the part. A text line starting with `## `, `<<`,
# ```` ```call ```` or `\\` is written with a `\\` in front.

";

/// The `[edit-history]` settings table: `editor = "nvim"`; by default
/// `$VISUAL`, else `$EDITOR`, else `vi`.
#[derive(Resource, Deserialize, Default, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct EditHistorySettings {
    /// The editor command, with its arguments; the file goes last.
    pub editor: Option<String>,
}

/// `/edit-history` and `/edit-history undo`; needs the terminal view,
/// which runs the editor.
#[derive(Default)]
pub struct TranscriptInEditorPlugin;

/// There when the terminal view is the run's front: only it runs the
/// editor (`RunInTerminal`), so without it the command refuses.
#[derive(Resource, Default, Clone, Copy, Debug)]
pub struct EditorTerminal;

impl Plugin for TranscriptInEditorPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<EditHistorySettings>("edit-history")
            .add_command(
                "edit-history",
                "Edit the conversation in the editor: /edit-history, /edit-history undo",
                edit_history,
            )
            .add_systems(
                Update,
                (
                    editor_closed.before(AgentLoop),
                    dialog_answered.after(AgentLoop),
                ),
            );
    }

    fn finish(&self, app: &mut App) {
        if app.runs_front(rig_tui::FRONT) {
            app.init_resource::<EditorTerminal>();
        }
    }
}

/// What a section stands for.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Kind {
    System,
    User,
    Assistant,
    Result { tool: String, id: String },
}

/// A section's header: its message, from 1, and its kind.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Header {
    n: usize,
    kind: Kind,
}

impl fmt::Display for Header {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = self.n;
        match &self.kind {
            Kind::System => write!(f, "## [{n}] system"),
            Kind::User => write!(f, "## [{n}] user"),
            Kind::Assistant => write!(f, "## [{n}] assistant"),
            Kind::Result { tool, id } => write!(f, "## [{n}] tool-result {tool} {id}"),
        }
    }
}

/// One item of a section's body.
#[derive(Clone, Debug, PartialEq)]
enum Item {
    Text(String),
    /// A placeholder: the message's part `part` (a result's for a
    /// tool-result section), of `kind`, kept as it was.
    Keep {
        kind: String,
        part: usize,
    },
    Call {
        tool: String,
        id: String,
        args: Map<String, Value>,
    },
}

/// A section of the file; `parts` are the message parts it was written
/// from (none for a section read from an edited file).
#[derive(Clone, Debug, PartialEq)]
struct Section {
    header: Header,
    items: Vec<Item>,
    parts: Vec<usize>,
}

/// The edited conversation, message by original message (`None`: deleted),
/// and the calls left with one side.
#[derive(Clone, Debug)]
struct Built {
    messages: Vec<Option<Message>>,
    orphans: BTreeSet<String>,
}

/// On a `RunInTerminal` of the editor: whose conversation it edits, the
/// file, what the file held when it opened, and the conversation then.
#[derive(Component, Clone, Debug)]
struct Editing {
    agent: Entity,
    path: PathBuf,
    written: String,
    original: Vec<Entry>,
    sections: Vec<Section>,
}

/// On the dialog asked when one side of a call was deleted.
#[derive(Component, Clone, Debug)]
struct Choosing {
    editing: Editing,
    text: String,
    built: Built,
}

// Writing the file.

fn escape(line: &str) -> String {
    let special = ["\\", "## ", "<<", "```call"];
    if special.iter().any(|prefix| line.starts_with(prefix)) {
        format!("\\{line}")
    } else {
        line.to_owned()
    }
}

fn push_text(lines: &mut Vec<String>, text: &str) {
    lines.extend(text.lines().map(escape));
}

fn placeholder(kind: &str, part: usize) -> String {
    format!("<<{kind} #{part}: kept as is; delete this line to drop it>>")
}

fn assistant_kind(block: &AssistantContent) -> &'static str {
    match block {
        AssistantContent::Text(_) => "text",
        AssistantContent::ToolCall(_) => "call",
        AssistantContent::Reasoning(_) => "reasoning",
        AssistantContent::Image(_) => "image",
        _ => "item",
    }
}

fn user_kind(part: &UserContent) -> &'static str {
    match part {
        UserContent::Text(_) => "text",
        UserContent::ToolResult(_) => "result",
        UserContent::Image(_) => "image",
        UserContent::Audio(_) => "audio",
        UserContent::Video(_) => "video",
        UserContent::Document(_) => "document",
    }
}

fn result_kind(content: &ToolResultContent) -> &'static str {
    match content {
        ToolResultContent::Text(_) => "text",
        ToolResultContent::Image(_) => "image",
        ToolResultContent::Json { .. } => "json",
    }
}

/// The file for `messages`, and the parts each of its sections holds.
fn render(messages: &[Message]) -> (String, Vec<Vec<usize>>) {
    let mut out = String::from(PREAMBLE);
    let mut parts_of = Vec::new();
    let mut section = |header: Header, lines: Vec<String>, parts: Vec<usize>| {
        out.push_str(&format!("{header}\n\n"));
        for line in lines {
            out.push_str(&line);
            out.push('\n');
        }
        out.push('\n');
        parts_of.push(parts);
    };
    for (at, message) in messages.iter().enumerate() {
        let n = at + 1;
        match message {
            Message::System { content } => {
                let mut lines = Vec::new();
                push_text(&mut lines, content);
                let kind = Kind::System;
                section(Header { n, kind }, lines, Vec::new());
            }
            Message::Assistant(turn) => {
                let mut lines = Vec::new();
                for (part, block) in turn.content.iter().enumerate() {
                    match block {
                        AssistantContent::Text(text) => push_text(&mut lines, &text.text),
                        AssistantContent::ToolCall(call) => {
                            let name = call.function.name.as_str();
                            lines.push(format!("```call {name} {}", call.id));
                            let args = serde_json::to_string_pretty(&call.function.arguments);
                            lines.extend(args.unwrap_or_default().lines().map(str::to_owned));
                            lines.push("```".to_owned());
                        }
                        other => lines.push(placeholder(assistant_kind(other), part)),
                    }
                }
                let parts = (0..turn.content.len()).collect();
                let kind = Kind::Assistant;
                section(Header { n, kind }, lines, parts);
            }
            Message::User { content } => {
                let (mut lines, mut parts) = (Vec::new(), Vec::new());
                for (part, item) in content.iter().enumerate() {
                    match item {
                        UserContent::ToolResult(result) => {
                            if !parts.is_empty() {
                                let kind = Kind::User;
                                let taken = std::mem::take(&mut parts);
                                section(Header { n, kind }, std::mem::take(&mut lines), taken);
                            }
                            let mut body = Vec::new();
                            for (at, content) in result.content.iter().enumerate() {
                                match content {
                                    ToolResultContent::Text(text) => {
                                        push_text(&mut body, &text.text)
                                    }
                                    other => body.push(placeholder(result_kind(other), at)),
                                }
                            }
                            let tool = result.name.as_str().to_owned();
                            let kind = Kind::Result {
                                tool,
                                id: result.call.to_string(),
                            };
                            section(Header { n, kind }, body, vec![part]);
                        }
                        UserContent::Text(text) => {
                            push_text(&mut lines, &text.text);
                            parts.push(part);
                        }
                        other => {
                            lines.push(placeholder(user_kind(other), part));
                            parts.push(part);
                        }
                    }
                }
                if !parts.is_empty() {
                    section(
                        Header {
                            n,
                            kind: Kind::User,
                        },
                        lines,
                        parts,
                    );
                }
            }
        }
    }
    (out, parts_of)
}

// Reading it back.

fn parse_header(line: &str) -> Option<Header> {
    let (n, rest) = line.strip_prefix("## [")?.split_once("] ")?;
    let n = n.parse().ok().filter(|n| *n > 0)?;
    let words: Vec<&str> = rest.split_whitespace().collect();
    let kind = match words.as_slice() {
        ["system"] => Kind::System,
        ["user"] => Kind::User,
        ["assistant"] => Kind::Assistant,
        ["tool-result", tool, id] => Kind::Result {
            tool: (*tool).to_owned(),
            id: (*id).to_owned(),
        },
        _ => return None,
    };
    Some(Header { n, kind })
}

/// Text lines, without the blank lines around them, as an item.
fn flush(text: &mut Vec<&str>, items: &mut Vec<Item>) {
    let lines = std::mem::take(text);
    let first = lines.iter().position(|line| !line.trim().is_empty());
    let last = lines.iter().rposition(|line| !line.trim().is_empty());
    if let (Some(first), Some(last)) = (first, last) {
        let kept = lines.get(first..=last).unwrap_or_default();
        items.push(Item::Text(kept.join("\n")));
    }
}

fn parse_body(header: &Header, body: &[(usize, &str)]) -> Result<Vec<Item>, String> {
    let (mut items, mut text) = (Vec::new(), Vec::new());
    let mut lines = body.iter();
    while let Some(&(number, line)) = lines.next() {
        if let Some(rest) = line.strip_prefix("```call") {
            if header.kind != Kind::Assistant {
                return Err(format!(
                    "line {number}: a tool call belongs in an assistant section"
                ));
            }
            flush(&mut text, &mut items);
            let words: Vec<&str> = rest.split_whitespace().collect();
            let [tool, id] = words.as_slice() else {
                return Err(format!(
                    "line {number}: write a call as ```call <tool> <id>"
                ));
            };
            let (mut json, mut closed) = (String::new(), false);
            for &(_, line) in lines.by_ref() {
                if line.trim_end() == "```" {
                    closed = true;
                    break;
                }
                json.push_str(line);
                json.push('\n');
            }
            if !closed {
                return Err(format!("line {number}: the call {id} has no closing ```"));
            }
            let args = match serde_json::from_str::<Value>(&json) {
                Ok(Value::Object(args)) => args,
                Ok(_) => {
                    return Err(format!(
                        "line {number}: the arguments of call {id} must be a JSON object"
                    ));
                }
                Err(why) => {
                    return Err(format!(
                        "line {number}: the arguments of call {id} are not valid JSON: {why}"
                    ));
                }
            };
            let (tool, id) = ((*tool).to_owned(), (*id).to_owned());
            items.push(Item::Call { tool, id, args });
        } else if let Some(inner) = line
            .trim_end()
            .strip_prefix("<<")
            .and_then(|it| it.strip_suffix(">>"))
        {
            flush(&mut text, &mut items);
            let head = inner.split(':').next().unwrap_or_default();
            let parsed = head
                .split_once(" #")
                .and_then(|(kind, part)| Some((kind.to_owned(), part.trim().parse().ok()?)));
            let Some((kind, part)) = parsed else {
                return Err(format!(
                    "line {number}: placeholder lines cannot be edited, only deleted"
                ));
            };
            items.push(Item::Keep { kind, part });
        } else {
            text.push(line.strip_prefix('\\').unwrap_or(line));
        }
    }
    flush(&mut text, &mut items);
    Ok(items)
}

/// The sections of `text`.
fn parse(text: &str) -> Result<Vec<Section>, String> {
    let mut sections: Vec<(Header, Vec<(usize, &str)>)> = Vec::new();
    for (at, line) in text.lines().enumerate() {
        let number = at + 1;
        if line.starts_with("## ") {
            let Some(header) = parse_header(line) else {
                return Err(format!(
                    "line {number}: `{line}` is no section header; a text line starting \
                     with `## ` needs a `\\` in front"
                ));
            };
            sections.push((header, Vec::new()));
        } else if let Some((_, body)) = sections.last_mut() {
            body.push((number, line));
        } else if !(line.trim().is_empty() || line.starts_with('#')) {
            return Err(format!(
                "line {number}: text before the first section belongs to no message"
            ));
        }
    }
    sections
        .into_iter()
        .map(|(header, body)| {
            let items = parse_body(&header, &body)?;
            Ok(Section {
                header,
                items,
                parts: Vec::new(),
            })
        })
        .collect()
}

/// For each edited section, the original section it is: the headers must
/// be the original ones, in order.
fn match_sections(original: &[Section], edited: &[Section]) -> Result<Vec<usize>, String> {
    let mut next = 0;
    let mut matched = Vec::with_capacity(edited.len());
    for section in edited {
        let found = original
            .iter()
            .enumerate()
            .skip(next)
            .find(|(_, it)| it.header == section.header);
        let Some((at, _)) = found else {
            return Err(format!(
                "`{}` is not a section of the conversation or is out of order: \
                 sections can be edited or deleted, not added or moved",
                section.header
            ));
        };
        matched.push(at);
        next = at + 1;
    }
    Ok(matched)
}

fn kept<'a, T>(
    parts: &'a [T],
    part: usize,
    kind: &str,
    kind_of: fn(&T) -> &'static str,
    n: usize,
) -> Result<&'a T, String> {
    parts
        .get(part)
        .filter(|it| kind_of(it) == kind)
        .ok_or_else(|| format!("the placeholder <<{kind} #{part}>> is not one of message {n}'s"))
}

/// Message `n` rebuilt from its sections left in the file (each with the
/// original it is).
fn rebuild(
    message: &Message,
    n: usize,
    kept_sections: &[(&Section, &Section)],
) -> Result<Message, String> {
    match message {
        Message::System { content } => {
            let texts: Vec<String> = kept_sections
                .iter()
                .flat_map(|(_, edited)| &edited.items)
                .filter_map(|item| match item {
                    Item::Text(text) => Some(text.clone()),
                    _ => None,
                })
                .collect();
            let content = if texts.is_empty() {
                content.clone()
            } else {
                texts.join("\n\n")
            };
            Ok(Message::System { content })
        }
        Message::Assistant(turn) => {
            let mut content = Vec::new();
            for item in kept_sections.iter().flat_map(|(_, edited)| &edited.items) {
                let block = match item {
                    Item::Text(text) => turn
                        .content
                        .iter()
                        .find(
                            |block| matches!(block, AssistantContent::Text(it) if it.text == *text),
                        )
                        .cloned()
                        .unwrap_or_else(|| AssistantContent::Text(Text::new(text.clone()))),
                    Item::Keep { kind, part } => {
                        kept(&turn.content, *part, kind, assistant_kind, n)?.clone()
                    }
                    Item::Call { tool, id, args } => {
                        let call = turn.tool_calls().find(|call| call.id.to_string() == *id);
                        let Some(call) = call.filter(|call| call.function.name.as_str() == tool)
                        else {
                            return Err(format!(
                                "message {n} has no call {id} of {tool}: only a call's arguments can change"
                            ));
                        };
                        let mut call = call.clone();
                        if call.function.arguments != *args {
                            // Its provider item no longer fits, which the
                            // encoders tell by its fingerprint.
                            call.function.arguments = args.clone();
                            call.function.invalid_arguments = None;
                        }
                        AssistantContent::ToolCall(call)
                    }
                };
                content.push(block);
            }
            Ok(Message::Assistant(turn.clone().with_content(content)))
        }
        Message::User { content } => {
            let mut parts = Vec::new();
            for (original, edited) in kept_sections {
                if original.items == edited.items {
                    parts.extend(
                        original
                            .parts
                            .iter()
                            .filter_map(|at| content.get(*at))
                            .cloned(),
                    );
                    continue;
                }
                if let Kind::Result { .. } = edited.header.kind {
                    let first = original.parts.first().and_then(|at| content.get(*at));
                    let Some(UserContent::ToolResult(result)) = first else {
                        return Err(format!("message {n} lost a result"));
                    };
                    let mut output = Vec::new();
                    for item in &edited.items {
                        output.push(match item {
                            Item::Text(text) => ToolResultContent::text(text.clone()),
                            Item::Keep { kind, part } => {
                                kept(&result.content, *part, kind, result_kind, n)?.clone()
                            }
                            Item::Call { .. } => {
                                return Err(format!("message {n}: no call in a result"));
                            }
                        });
                    }
                    if output.is_empty() {
                        output.push(ToolResultContent::text(String::new()));
                    }
                    parts.push(UserContent::ToolResult(ResultPart {
                        content: output,
                        ..result.clone()
                    }));
                    continue;
                }
                for item in &edited.items {
                    parts.push(match item {
                        Item::Text(text) => content
                            .iter()
                            .find(|part| matches!(part, UserContent::Text(it) if it.text == *text))
                            .cloned()
                            .unwrap_or_else(|| UserContent::Text(Text::new(text.clone()))),
                        Item::Keep { kind, part } => {
                            kept(content, *part, kind, user_kind, n)?.clone()
                        }
                        Item::Call { .. } => {
                            return Err(format!("message {n}: no call in a user section"));
                        }
                    });
                }
            }
            Ok(Message::User { content: parts })
        }
    }
}

/// The call ids `message` makes and those it answers.
fn ids(message: &Message) -> (Vec<String>, Vec<String>) {
    match message {
        Message::Assistant(turn) => (
            turn.tool_calls().map(|call| call.id.to_string()).collect(),
            Vec::new(),
        ),
        Message::User { content } => {
            let results = content.iter().filter_map(|part| match part {
                UserContent::ToolResult(result) => Some(result.call.to_string()),
                _ => None,
            });
            (Vec::new(), results.collect())
        }
        Message::System { .. } => (Vec::new(), Vec::new()),
    }
}

/// The calls and results of `messages`.
fn calls_and_results<'a>(
    messages: impl Iterator<Item = &'a Message>,
) -> (HashSet<String>, HashSet<String>) {
    let (mut calls, mut results) = (HashSet::new(), HashSet::new());
    for message in messages {
        let (made, answered) = ids(message);
        calls.extend(made);
        results.extend(answered);
    }
    (calls, results)
}

/// The conversation `original`, written as `sections`, as the file now
/// says (`edited`).
fn build(original: &[Entry], sections: &[Section], edited: &[Section]) -> Result<Built, String> {
    let matched = match_sections(sections, edited)?;
    let pairs: Vec<(&Section, &Section)> = matched
        .iter()
        .filter_map(|at| sections.get(*at))
        .zip(edited)
        .collect();
    let mut messages = Vec::with_capacity(original.len());
    for (at, entry) in original.iter().enumerate() {
        let n = at + 1;
        let all = sections.iter().filter(|it| it.header.n == n).count();
        let left: Vec<(&Section, &Section)> = pairs
            .iter()
            .filter(|(it, _)| it.header.n == n)
            .copied()
            .collect();
        if left.is_empty() {
            messages.push(None);
        } else if left.len() == all && left.iter().all(|(it, edited)| it.items == edited.items) {
            messages.push(Some((*entry.message).clone()));
        } else {
            messages.push(Some(rebuild(&entry.message, n, &left)?));
        }
    }
    let (calls, results) = calls_and_results(original.iter().map(|entry| &*entry.message));
    let (calls_now, results_now) = calls_and_results(messages.iter().flatten());
    let orphans = calls
        .intersection(&results)
        .filter(|id| calls_now.contains(*id) != results_now.contains(*id))
        .cloned()
        .collect();
    Ok(Built { messages, orphans })
}

/// The entries of the edited conversation, both sides of each orphaned
/// call taken out when `delete_both`, normalized: the model's turns left
/// side by side are one, keeping the first's id and author. A reply whose
/// calls moved, in it or by a merge, gets a new id, so what was kept for a
/// call at its place, such as its run time, is not drawn for the call now
/// there. Refused when that needs more than merging turns and the original
/// paired its calls, and when the edit takes out, moves or merges calls of
/// a reply that repeats a call id, as its calls could only be told apart by
/// order.
fn finish(original: &[Entry], built: &Built, delete_both: bool) -> Result<Vec<Entry>, String> {
    let orphan = |id: String| delete_both && built.orphans.contains(&id);
    let mut entries = Vec::new();
    for (entry, message) in original.iter().zip(&built.messages) {
        let Some(mut message) = message.clone() else {
            continue;
        };
        let empty = match &mut message {
            Message::Assistant(turn) => {
                turn.content.retain(|block| {
                    !matches!(block, AssistantContent::ToolCall(call) if orphan(call.id.to_string()))
                });
                turn.content.is_empty()
            }
            Message::User { content } => {
                content.retain(|part| {
                    !matches!(part, UserContent::ToolResult(result) if orphan(result.call.to_string()))
                });
                content.is_empty()
            }
            Message::System { content } => content.trim().is_empty(),
        };
        let (calls, before) = (ids(&message).0, ids(&entry.message).0);
        // A call that kept its place keeps its reply's id, and what was
        // kept for it at that place.
        let moved = calls.iter().zip(&before).any(|(call, was)| call != was);
        // The file names a call by its id alone, so in a reply that repeats
        // one it can't tell which call an edit took out or moved, and a new
        // id would leave such calls to be paired by order.
        let repeats = |id: &&String| before.iter().filter(|was| was == id).count() > 1;
        let shared = |calls: &[String]| calls.iter().filter(repeats).cloned().collect::<Vec<_>>();
        let changed = moved || shared(&calls) != shared(&before);
        if let Some(id) = shared(&before).first().filter(|_| changed) {
            return Err(format!(
                "a reply asks for two calls with the id `{id}`, so the edit can't keep each call in its place; leave that reply's calls as they are, or take out only calls at its end"
            ));
        }
        if !empty {
            entries.push(Entry {
                message: message.into(),
                id: entry.id.filter(|_| !moved),
                ..entry.clone()
            });
        }
    }
    let normalized = pairing::normalize(entries);
    let worse = normalized
        .repairs
        .iter()
        .find(|repair| !matches!(repair, PairingError::ConsecutiveAssistant { .. }));
    let before: Vec<Message> = original
        .iter()
        .map(|entry| (*entry.message).clone())
        .collect();
    if let (Some(why), Ok(())) = (worse, pairing::validate(&before)) {
        return Err(format!(
            "the edited conversation does not pair its tool calls: {why}"
        ));
    }
    // Merging replies can put a call where another was, under that one's
    // id: such a reply gets a new id, so nothing kept for the old places
    // matches it. One that repeats a call id can't: its calls would be
    // paired by order.
    let mut entries = normalized.entries;
    for entry in &mut entries {
        let calls = ids(&entry.message).0;
        let kept = original
            .iter()
            .find(|was| was.id.is_some() && was.id == entry.id);
        let placed = kept.is_some_and(|was| ids(&was.message).0.starts_with(&calls));
        if entry.id.is_none() || placed {
            continue;
        }
        let repeated = calls
            .iter()
            .find(|id| calls.iter().filter(|it| it == id).count() > 1);
        if let Some(id) = repeated {
            return Err(format!(
                "a reply asks for two calls with the id `{id}`, so the edit can't keep each call in its place; leave that reply and the messages around it as they are"
            ));
        }
        entry.id = None;
    }
    Ok(entries)
}

/// Every entry of `conversation`.
fn entries_of(conversation: &History) -> Vec<Entry> {
    (0..conversation.messages().len())
        .filter_map(|at| conversation.entry(at))
        .collect()
}

/// Makes `conversation` hold `entries`, as one change by `by`; whether
/// anything changed.
fn apply(conversation: &mut History, entries: Vec<Entry>, by: &str) -> bool {
    let (revision, halted) = (conversation.revision(), conversation.halted());
    conversation.set_entries(entries, Author::plugin(by));
    // A user message left last waits for the next one, as before the edit.
    conversation.halt(halted.unwrap_or(Halt::Kept));
    conversation.revision() != revision
}

/// The conversation before the last edit not undone, read from the log
/// `log`, with the messages added after the edit; and how many those are.
/// An edit already reverted otherwise is passed over.
fn undone(
    store: &dyn Store,
    log: &str,
    conversation: &History,
) -> Result<(Vec<Entry>, usize), String> {
    let unreadable = |why: SessionError| format!("could not read the session log: {why}");
    let mut edits = Vec::new();
    for (stamp, record) in history(store, log).map_err(unreadable)? {
        let LogRecord::Change(ChangeRecord { by: Some(by), .. }) = record else {
            continue;
        };
        if by == Author::plugin(PLUGIN) {
            edits.push(stamp.n);
        } else if by == Author::plugin(UNDO) {
            edits.pop();
        }
    }
    let current = entries_of(conversation);
    for n in edits.into_iter().rev() {
        let edited: HashSet<MessageId> = as_of(store, log, n + 1)
            .map_err(unreadable)?
            .entries()
            .map(|entry| entry.id)
            .collect();
        let since: Vec<Entry> = current
            .iter()
            .filter(|entry| entry.id.is_some_and(|id| !edited.contains(&id)))
            .cloned()
            .collect();
        let added = since.len();
        let mut entries = entries_of(&as_of(store, log, n).map_err(unreadable)?);
        entries.extend(since);
        if entries != current {
            return Ok((entries, added));
        }
    }
    Err("no edit to undo".to_owned())
}

// The command and the editor.

fn editor_command(settings: &EditHistorySettings, path: &Path) -> Command {
    let line = settings.editor.clone().or_else(corpus_shared::editor::line);
    corpus_shared::editor::command(line.as_deref(), &[path])
}

/// Writes `text` to the editing's file and opens the editor on it.
fn open(
    mut editing: Editing,
    text: String,
    settings: &EditHistorySettings,
    commands: &mut Commands,
) -> Result<(), String> {
    std::fs::write(&editing.path, &text)
        .map_err(|why| format!("could not write {}: {why}", editing.path.display()))?;
    let command = editor_command(settings, &editing.path);
    editing.written = text;
    commands.spawn((editing, RunInTerminal(command)));
    Ok(())
}

/// Opens the editor again on `text`, with `why` on top.
fn reopen(
    editing: Editing,
    text: &str,
    why: &str,
    settings: &EditHistorySettings,
    commands: &mut Commands,
    notices: &mut MessageWriter<Notice>,
) {
    let agent = editing.agent;
    let body: Vec<&str> = text
        .lines()
        .skip_while(|line| {
            !line.starts_with("## ") && (line.trim().is_empty() || line.starts_with('#'))
        })
        .collect();
    let why = why.replace('\n', " ");
    let text = format!("# ERROR: {why}\n{PREAMBLE}{}\n", body.join("\n"));
    if let Err(failed) = open(editing, text, settings, commands) {
        notices.write(Notice::error(
            agent,
            format!("/edit-history: {failed}; nothing changed."),
        ));
    }
}

/// `/edit-history` or `/edit-history undo`.
fn edit_history(
    In(args): In<CommandArgs>,
    mut agents: Query<(&mut Conversation, Has<ActiveTurn>, Option<&AgentId>)>,
    editing: Query<&Editing>,
    choosing: Query<&Choosing>,
    (settings, scratch, store, terminal): (
        Res<EditHistorySettings>,
        Option<Res<Scratch>>,
        Option<Res<SessionStore>>,
        Option<Res<EditorTerminal>>,
    ),
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let agent = args.agent;
    let Ok((mut conversation, busy, id)) = agents.get_mut(agent) else {
        return;
    };
    let mut refuse = |why: &str| {
        notices.write(Notice::error(agent, format!("/edit-history: {why}")));
    };
    if terminal.is_none() {
        return refuse("only the terminal view runs the editor.");
    }
    if busy {
        return refuse("the agent's turn is running; wait for it to end or interrupt it.");
    }
    let open_already = editing.iter().any(|it| it.agent == agent)
        || choosing.iter().any(|it| it.editing.agent == agent);
    if open_already {
        return refuse("the conversation is open in the editor already.");
    }
    match args.args.as_str() {
        "" => {}
        "undo" => {
            let (Some(store), Some(id)) = (store, id) else {
                return refuse("undo needs the session's log.");
            };
            match undone(&*store.0, &id.0, &conversation) {
                Err(why) => refuse(&why),
                Ok((entries, added)) => {
                    apply(&mut conversation, entries, UNDO);
                    let kept = match added {
                        0 => String::new(),
                        n => format!(" The {n} messages since were kept."),
                    };
                    notices.write(Notice::info(
                        agent,
                        format!("Restored the conversation before the last edit.{kept}"),
                    ));
                }
            }
            return;
        }
        other => {
            return refuse(&format!(
                "`{other}`: use /edit-history or /edit-history undo."
            ));
        }
    }
    let original = entries_of(&conversation);
    let messages: Vec<Message> = original
        .iter()
        .map(|entry| (*entry.message).clone())
        .collect();
    let (text, parts_of) = render(&messages);
    let sections = match parse(&text) {
        Ok(sections) if sections.len() == parts_of.len() => sections
            .into_iter()
            .zip(parts_of)
            .map(|(section, parts)| Section { parts, ..section })
            .collect(),
        _ => return refuse("could not write this conversation as text."),
    };
    let Some(dir) = scratch.and_then(|it| it.0.clone()) else {
        return refuse("there is no session directory to write the file in.");
    };
    let name = id.map_or_else(|| agent.index().to_string(), |id| id.0.clone());
    let path = dir.join(format!("edit-history-{name}.md"));
    if let Err(why) = std::fs::create_dir_all(&dir) {
        return refuse(&format!("could not make {}: {why}", dir.display()));
    }
    let editing = Editing {
        agent,
        path,
        written: String::new(),
        original,
        sections,
    };
    if let Err(why) = open(editing, text, &settings, &mut commands) {
        refuse(&why);
    }
}

/// Applies `built` to the agent's conversation, unless its turn runs or
/// it changed since the editor opened; reopens the editor when it does not
/// pair its calls.
fn commit(
    agents: &mut Query<(&mut Conversation, Has<ActiveTurn>)>,
    (editing, text, built, delete_both): (&Editing, &str, &Built, bool),
    settings: &EditHistorySettings,
    commands: &mut Commands,
    notices: &mut MessageWriter<Notice>,
) {
    let agent = editing.agent;
    let Ok((mut conversation, busy)) = agents.get_mut(agent) else {
        return;
    };
    let refused = if busy {
        Some("the agent's turn started meanwhile")
    } else if entries_of(&conversation) != editing.original {
        Some("the conversation changed while it was open")
    } else {
        None
    };
    if let Some(why) = refused {
        std::fs::remove_file(&editing.path).ok();
        notices.write(Notice::error(
            agent,
            format!("/edit-history: {why}; nothing changed."),
        ));
        return;
    }
    match finish(&editing.original, built, delete_both) {
        Err(why) => reopen(editing.clone(), text, &why, settings, commands, notices),
        Ok(entries) => {
            std::fs::remove_file(&editing.path).ok();
            let count = entries.len();
            let text = if apply(&mut conversation, entries, PLUGIN) {
                format!(
                    "Edited the conversation: {count} messages, {} before. /edit-history undo restores it.",
                    editing.original.len()
                )
            } else {
                "The edit changed nothing.".to_owned()
            };
            notices.write(Notice::info(agent, text));
        }
    }
}

/// Reads the file once the editor closed: applies it, asks about calls
/// left with one side, or reopens the editor on an error.
fn editor_closed(
    done: Query<(Entity, &TerminalExit, &Editing), Added<TerminalExit>>,
    mut agents: Query<(&mut Conversation, Has<ActiveTurn>)>,
    (answerers, settings): (Answerers, Res<EditHistorySettings>),
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    for (run, exit, editing) in &done {
        commands.entity(run).despawn();
        let agent = editing.agent;
        let read = match exit.failure() {
            None => std::fs::read_to_string(&editing.path).map_err(|why| why.to_string()),
            Some(why) => Err(format!("the editor failed ({why})")),
        };
        let text = match read {
            Ok(text) if text != editing.written => text,
            unchanged => {
                std::fs::remove_file(&editing.path).ok();
                let why = unchanged
                    .err()
                    .unwrap_or_else(|| "the file was not saved".to_owned());
                notices.write(Notice::info(
                    agent,
                    format!("/edit-history: {why}; nothing changed."),
                ));
                continue;
            }
        };
        let built =
            parse(&text).and_then(|edited| build(&editing.original, &editing.sections, &edited));
        match built {
            Err(why) => reopen(
                editing.clone(),
                &text,
                &why,
                &settings,
                &mut commands,
                &mut notices,
            ),
            Ok(built) if built.orphans.is_empty() => {
                let commit_args = (editing, text.as_str(), &built, false);
                commit(
                    &mut agents,
                    commit_args,
                    &settings,
                    &mut commands,
                    &mut notices,
                );
            }
            Ok(built) => {
                let calls: Vec<&str> = built.orphans.iter().map(String::as_str).collect();
                let why = format!(
                    "a tool call and its result go together: {}",
                    calls.join(", ")
                );
                if !answerers.can_answer(Some(agent)) {
                    reopen(
                        editing.clone(),
                        &text,
                        &why,
                        &settings,
                        &mut commands,
                        &mut notices,
                    );
                    continue;
                }
                let ask = Ask::new(DIALOG_TITLE)
                    .body(format!("You deleted one side of {why}."))
                    .choice(DELETE_BOTH, "Delete both")
                    .choice(REOPEN, "Reopen the editor");
                let choosing = Choosing {
                    editing: editing.clone(),
                    text,
                    built,
                };
                commands.spawn((ask, ForAgent(agent), choosing));
            }
        }
    }
}

/// Acts on the dialog's answer: deletes both sides, or reopens the
/// editor; dismissed, nothing changes.
fn dialog_answered(
    asks: Query<(&Outcome, &Choosing), (With<Ask>, Added<Outcome>)>,
    mut agents: Query<(&mut Conversation, Has<ActiveTurn>)>,
    settings: Res<EditHistorySettings>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    for (
        outcome,
        Choosing {
            editing,
            text,
            built,
        },
    ) in &asks
    {
        let answer = Answer::of(outcome);
        if answer.as_ref().is_some_and(|it| it.is(DELETE_BOTH)) {
            let commit_args = (editing, text.as_str(), built, true);
            commit(
                &mut agents,
                commit_args,
                &settings,
                &mut commands,
                &mut notices,
            );
        } else if answer.is_some_and(|it| it.is(REOPEN)) {
            let why = "a tool call and its result go together: delete both sections or neither";
            reopen(
                editing.clone(),
                text,
                why,
                &settings,
                &mut commands,
                &mut notices,
            );
        } else {
            std::fs::remove_file(&editing.path).ok();
            notices.write(Notice::info(
                editing.agent,
                "/edit-history: dismissed; nothing changed.",
            ));
        }
    }
}
