//! CT-06: a searchable archive of compacted history.
//!
//! Whenever an agent's conversation is condensed, the messages that leave
//! the model's view stay findable: `history_search {query, kind?, limit?}`
//! searches them (user messages, replies, tool calls and results) and
//! answers short snippets with archive ids and turn numbers, and
//! `history_get {id, context?, from_line?, to_line?}` answers one whole,
//! a large one cut with a note saying how to read the rest by lines, and
//! optionally its neighbours.
//!
//! The archive is the session's own log, read with rig-memory's
//! [`history`]: every message an agent ever had is in its log, so nothing
//! is copied. A condensation is a splice that takes entries out and puts
//! in at least one entry made after all of them (a summary), whichever
//! plugin made it; rig-compaction's are such splices. An agent's archive
//! is what the condensations still standing took out: a condensation
//! stands while its summary is in the conversation, or was itself taken
//! out by one that stands. So:
//!
//! - repeated compactions add only what they took out, and an earlier
//!   summary condensed again is archived with them;
//! - a fork copies the log up to its cut, so it sees the archive as of its
//!   fork point plus its own later compactions; a branch switched away
//!   from takes its summaries, and so its condensations, with it; an
//!   undone compaction's messages are back in view and leave the archive;
//! - the archive survives restarts with the log, and a message deleted or
//!   redacted while the session is stopped (rig-memory's `rewrite`) is
//!   gone from it too, as that rewrites the log in place;
//! - each agent searches its own log; a subagent also searches the log of
//!   the agent that spawned it only with the `share_with_subagents`
//!   setting of the `[compaction_archive]` table.
//!
//! An archived message is shown as the log last had it, but for in-place
//! edits by rig-compaction (its clearing of old tool outputs), so the
//! real output is found, not the placeholder.
//!
//! Each summary gets one more line, [`ARCHIVED_LINE`]: rig-compaction's
//! as a reviewer of its text ([`Summarizing::summary`]), before it
//! replaces anything and so before the request built from it, and any
//! other plugin's once its splice is committed.

use std::collections::{BTreeSet, HashMap, HashSet};

use rig_compaction::Summarizing;
use rig_harness::prelude::*;
use rig_harness::rig_core::conversation::Conversation as Chat;
use rig_harness::rig_core::message::ToolResultContent;
use rig_harness::rig_ecs::session_log::SessionStore;
use rig_harness::rig_memory::{LogRecord, SessionError, summary_author};
use schemars::JsonSchema;
use serde::Deserialize;

/// Who this plugin is, in decisions and splices.
pub const NAME: &str = "compaction-archive";

/// The search tool's name.
pub const SEARCH: &str = "history_search";

/// The tool that reads one archived message.
pub const GET: &str = "history_get";

/// The line every condensation summary gets.
pub const ARCHIVED_LINE: &str =
    "Earlier messages are archived; use `history_search` to look them up.";

/// Characters of an archived message `history_get` answers before it cuts.
pub const MAX_CHARS: usize = 4_000;

/// Characters a snippet shows on each side of the match.
const SNIPPET_CHARS: usize = 80;

/// Characters a neighbour shows.
const NEIGHBOUR_CHARS: usize = 400;

/// Hits a search answers unless it says.
const DEFAULT_LIMIT: usize = 10;

/// Hits a search answers at most.
const MAX_LIMIT: usize = 50;

/// The prefix of an archive id of the agent's own archive.
const OWN: &str = "m";

/// The prefix of an archive id of the parent's archive, when shared.
const PARENT: &str = "p";

/// Archives what condensations take out of the model's view and offers
/// `history_search` and `history_get` over it.
#[derive(Default)]
pub struct CompactionArchivePlugin;

impl Plugin for CompactionArchivePlugin {
    fn build(&self, app: &mut App) {
        let reads = || ToolOptions {
            footprint: Some(Touches::Reads(Scope::Anything)),
        };
        app.add_setting::<ArchiveSettings>("compaction_archive")
            .add_open_tool(
                SEARCH,
                "Search the archived messages that compaction took out of the conversation: \
                 user messages, replies, tool calls and results. Every word of `query` must \
                 appear (case is ignored); `kind` is `user`, `assistant` or `tool`. Answers \
                 snippets with archive ids and turn numbers. Use it to find the exact text of \
                 an error, a command or what the user said from before a compaction, then \
                 `history_get` to read one whole.",
                reads(),
                search,
            )
            .add_open_tool(
                GET,
                "Read one archived message by the id `history_search` gave, with `context` \
                 neighbours on each side; a long one is cut, and `from_line`/`to_line` read \
                 the rest.",
                reads(),
                get,
            )
            .add_systems(
                AgentPass,
                mark_compaction_summary.in_set(ReviewSystems::Act),
            )
            .add_systems(Update, mark_other_summaries.after(AgentLoop));
    }
}

/// The `[compaction_archive]` table of the settings.
#[derive(Resource, Deserialize, Default, Clone, Copy, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct ArchiveSettings {
    /// Whether a subagent also searches the archive of the agent that
    /// spawned it. Off by default.
    pub share_with_subagents: bool,
}

/// What an archived message is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    /// The user's message, or another request.
    User,
    /// A reply with text.
    Assistant,
    /// A tool call without text, or tool results.
    Tool,
}

impl Kind {
    /// The kind of `message`.
    pub fn of(message: &message::Message) -> Self {
        match message {
            message::Message::User { content }
                if content
                    .iter()
                    .any(|item| matches!(item, UserContent::ToolResult(_))) =>
            {
                Self::Tool
            }
            message::Message::User { .. } | message::Message::System { .. } => Self::User,
            message::Message::Assistant(reply)
                if reply.tool_calls().next().is_some() && reply.text().trim().is_empty() =>
            {
                Self::Tool
            }
            message::Message::Assistant(_) => Self::Assistant,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

/// A message a condensation took out of the model's view.
#[derive(Clone, Debug, PartialEq)]
pub struct Archived {
    /// Its id in its conversation.
    pub id: MessageId,
    /// The turn it was made in, counted from 1, when the log says.
    pub turn: Option<u64>,
    /// What it is.
    pub kind: Kind,
    /// The message.
    pub message: message::Message,
}

impl Archived {
    /// Its text, as searched and shown.
    pub fn text(&self) -> String {
        text_of(&self.message)
    }
}

/// The archive of the conversation kept in the log `log` of `store`, now
/// `conversation`, oldest first: every message a condensation still
/// standing took out, as the log last had it.
///
/// # Errors
///
/// When the log cannot be read.
pub fn read_archive(
    store: &dyn Store,
    log: &str,
    conversation: &Chat,
) -> Result<Vec<Archived>, SessionError> {
    let compaction = summary_author();
    let mut latest: HashMap<MessageId, message::Message> = HashMap::new();
    let mut condensed: Vec<(Vec<MessageId>, Vec<MessageId>)> = Vec::new();
    for (_, record) in history(store, log)? {
        let LogRecord::Change(recorded) = record else {
            continue;
        };
        let in_place_by_compaction = recorded.by.as_ref() == Some(&compaction);
        condensed.extend(condensation(&recorded.change));
        match recorded.change {
            Change::Push { entries } => keep_latest(&mut latest, entries, false),
            Change::Splice { with, .. } => keep_latest(&mut latest, with, in_place_by_compaction),
            Change::Reset { state } => {
                let entries = (0..state.messages().len()).filter_map(|at| state.entry(at));
                keep_latest(&mut latest, entries.collect(), false);
            }
            Change::Halt { .. } | Change::Aside { .. } => {}
        }
    }
    let live: HashSet<MessageId> = conversation.entries().map(|entry| entry.id).collect();
    let mut standing = live.clone();
    let mut archived = BTreeSet::new();
    for (summaries, gone) in condensed.into_iter().rev() {
        if summaries.iter().any(|summary| standing.contains(summary)) {
            standing.extend(gone.iter().copied());
            archived.extend(gone.into_iter().filter(|id| !live.contains(id)));
        }
    }
    let turns: Vec<(MessageId, u64)> = conversation
        .turns()
        .map(|(_, record)| (record.start, record.index))
        .collect();
    let archive = archived.into_iter().filter_map(|id| {
        let message = latest.remove(&id)?;
        let turn = turns.iter().rev().find(|(start, _)| *start <= id);
        Some(Archived {
            id,
            turn: turn.map(|(_, index)| index.saturating_add(1)),
            kind: Kind::of(&message),
            message,
        })
    });
    Ok(archive.collect())
}

/// When `change` condenses: the entries it put in that were made after
/// every entry it took out (its summaries), and the entries it took out.
pub fn condensation(change: &Change) -> Option<(Vec<MessageId>, Vec<MessageId>)> {
    let Change::Splice { removed, with, .. } = change else {
        return None;
    };
    let put: HashSet<MessageId> = with.iter().filter_map(|entry| entry.id).collect();
    let newest = removed.iter().max()?;
    let summaries: Vec<MessageId> = with
        .iter()
        .filter_map(|entry| entry.id)
        .filter(|id| id > newest)
        .collect();
    let gone: Vec<MessageId> = removed
        .iter()
        .filter(|id| !put.contains(id))
        .copied()
        .collect();
    (!summaries.is_empty() && !gone.is_empty()).then_some((summaries, gone))
}

/// Keeps each of `entries` as the latest version of its message, but an
/// in-place edit of one already kept when `in_place_skipped`.
fn keep_latest(
    latest: &mut HashMap<MessageId, message::Message>,
    entries: Vec<Entry>,
    in_place_skipped: bool,
) {
    for entry in entries {
        let Some(id) = entry.id else { continue };
        if !(in_place_skipped && latest.contains_key(&id)) {
            latest.insert(id, std::sync::Arc::unwrap_or_clone(entry.message));
        }
    }
}

/// The text of `message`: its text, tool results, and tool calls as
/// `name(arguments)`, one part a line.
pub fn text_of(message: &message::Message) -> String {
    let mut parts: Vec<String> = Vec::new();
    match message {
        message::Message::System { content } => parts.push(content.clone()),
        message::Message::User { content } => {
            for item in content {
                match item {
                    UserContent::Text(text) => parts.push(text.text.clone()),
                    UserContent::ToolResult(result) => {
                        parts.extend(result.content.iter().map(|part| match part {
                            ToolResultContent::Text(text) => text.text.clone(),
                            ToolResultContent::Json { value } => value.to_string(),
                            ToolResultContent::Image(_) => "[image]".to_owned(),
                        }));
                    }
                    UserContent::Image(_)
                    | UserContent::Audio(_)
                    | UserContent::Video(_)
                    | UserContent::Document(_) => parts.push("[attachment]".to_owned()),
                }
            }
        }
        message::Message::Assistant(reply) => {
            for block in &reply.content {
                match block {
                    AssistantContent::Text(text) => parts.push(text.text.clone()),
                    AssistantContent::ToolCall(call) => parts.push(format!(
                        "{}({})",
                        call.function.name.as_str(),
                        serde_json::to_string(&call.function.arguments).unwrap_or_default()
                    )),
                    _ => {}
                }
            }
        }
    }
    parts.join("\n")
}

/// An archived message and whose archive it is in.
struct Scoped {
    prefix: &'static str,
    archived: Archived,
}

impl Scoped {
    /// Its archive id: the prefix and its message's number.
    fn label(&self) -> String {
        let number = serde_json::to_value(self.archived.id)
            .ok()
            .and_then(|value| value.as_u64());
        number.map_or_else(|| "?".to_owned(), |n| format!("{}{n}", self.prefix))
    }

    /// Its id, turn and kind.
    fn header(&self) -> String {
        let turn = self
            .archived
            .turn
            .map_or_else(|| "turn ?".to_owned(), |turn| format!("turn {turn}"));
        format!("[{}] {turn}, {}", self.label(), self.archived.kind.label())
    }
}

/// The arguments of `history_search`.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SearchArgs {
    /// Words that must all appear; case is ignored.
    pub query: String,
    /// Only messages of this kind: `user`, `assistant` or `tool`.
    #[serde(default)]
    pub kind: Option<Kind>,
    /// At most this many hits; 10 unless given.
    #[serde(default)]
    pub limit: Option<usize>,
}

/// The arguments of `history_get`.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GetArgs {
    /// The archive id `history_search` gave, such as `m12`.
    pub id: String,
    /// How many archived messages before and after it to show too.
    #[serde(default)]
    pub context: Option<usize>,
    /// The first line to answer, counted from 1.
    #[serde(default)]
    pub from_line: Option<usize>,
    /// The last line to answer.
    #[serde(default)]
    pub to_line: Option<usize>,
}

/// The agents as the tools read them.
type Agents<'w, 's> = Query<
    'w,
    's,
    (
        &'static AgentId,
        &'static Conversation,
        Option<&'static SpawnedBy>,
    ),
>;

/// What `agent` may search: its archive, and its parent's when shared.
fn archive_of(
    agent: Entity,
    agents: &Agents,
    store: Option<&SessionStore>,
    settings: ArchiveSettings,
) -> Result<Vec<Scoped>, String> {
    let store = store.ok_or("this session is not saved, so nothing is archived")?;
    let read = |agent: Entity, prefix: &'static str| -> Result<Vec<Scoped>, String> {
        let (id, conversation, _) = agents.get(agent).map_err(|failure| failure.to_string())?;
        let archive = read_archive(&*store.0, &id.0, conversation);
        let archive = archive.map_err(|failure| failure.to_string())?;
        let scoped = archive
            .into_iter()
            .map(|archived| Scoped { prefix, archived });
        Ok(scoped.collect())
    };
    let mut archive = read(agent, OWN)?;
    let parent = agents.get(agent).ok().and_then(|(_, _, by)| by);
    if settings.share_with_subagents
        && let Some(&SpawnedBy(parent)) = parent
    {
        archive.extend(read(parent, PARENT)?);
    }
    Ok(archive)
}

/// Decides `call` with `result`.
fn answer(
    call: Entity,
    result: Result<String, String>,
    calls: &mut Query<&mut Decision, ToolCalls>,
) {
    let result = match result {
        Ok(text) => ToolResult::success(text.into()),
        Err(why) => ToolResult::failed(ToolExecutionError::other(why)),
    };
    if let Ok(mut decision) = calls.get_mut(call) {
        decision.decide(Decider::plugin(NAME), result);
    }
}

/// Answers a `history_search` call.
fn search(
    called: On<ToolCalled<SearchArgs>>,
    agents: Agents,
    store: Option<Res<SessionStore>>,
    settings: Res<ArchiveSettings>,
    mut calls: Query<&mut Decision, ToolCalls>,
) {
    let archive = archive_of(called.agent, &agents, store.as_deref(), *settings);
    let result = archive.map(|archive| found(&archive, &called.args));
    answer(called.call, result, &mut calls);
}

/// Answers a `history_get` call.
fn get(
    called: On<ToolCalled<GetArgs>>,
    agents: Agents,
    store: Option<Res<SessionStore>>,
    settings: Res<ArchiveSettings>,
    mut calls: Query<&mut Decision, ToolCalls>,
) {
    let archive = archive_of(called.agent, &agents, store.as_deref(), *settings);
    let result = archive.and_then(|archive| read_one(&archive, &called.args));
    answer(called.call, result, &mut calls);
}

/// The hits of a search, as the model reads them.
fn found(archive: &[Scoped], args: &SearchArgs) -> String {
    let terms: Vec<String> = args
        .query
        .split_whitespace()
        .map(str::to_lowercase)
        .collect();
    let limit = args.limit.unwrap_or(DEFAULT_LIMIT).clamp(1, MAX_LIMIT);
    let hits: Vec<String> = archive
        .iter()
        .filter(|scoped| args.kind.is_none_or(|kind| scoped.archived.kind == kind))
        .filter_map(|scoped| {
            let text = scoped.archived.text();
            let lower = text.to_lowercase();
            let all = !terms.is_empty() && terms.iter().all(|term| lower.contains(term));
            let at = terms.first().and_then(|term| lower.find(term.as_str()));
            let at = at.filter(|_| all)?;
            Some(format!("{}: {}", scoped.header(), snippet(&text, at)))
        })
        .take(limit)
        .collect();
    let query = &args.query;
    if hits.is_empty() {
        return format!(
            "No archived message matches \"{query}\" ({} archived).",
            archive.len()
        );
    }
    let count = match hits.len() {
        1 => "1 archived message matches".to_owned(),
        n => format!("{n} archived messages match"),
    };
    format!(
        "{count} \"{query}\":\n{}\nRead one whole with `{GET}`.",
        hits.join("\n")
    )
}

/// The archived message `args.id`, cut or by lines, with its neighbours.
fn read_one(archive: &[Scoped], args: &GetArgs) -> Result<String, String> {
    let at = archive
        .iter()
        .position(|scoped| scoped.label() == args.id)
        .ok_or_else(|| format!("no archived message `{}`", args.id))?;
    let scoped = archive.get(at).ok_or("no such message")?;
    let text = scoped.archived.text();
    let lines: Vec<&str> = text.lines().collect();
    let total = lines.len();
    let body = match (args.from_line, args.to_line) {
        (None, None) => cut(&text, &lines),
        (from, to) => {
            let from = from.unwrap_or(1).max(1);
            let to = to.unwrap_or(total).min(total);
            let range = lines.get(from.saturating_sub(1)..to);
            let range = range.ok_or_else(|| format!("it has {total} lines"))?;
            format!("{}\n[lines {from}-{to} of {total}]", range.join("\n"))
        }
    };
    let mut parts = Vec::new();
    let context = args.context.unwrap_or(0);
    let near = |scoped: &Scoped| {
        let text = scoped.archived.text();
        let end = boundary(&text, NEIGHBOUR_CHARS);
        let more = if end < text.len() { "…" } else { "" };
        format!(
            "{}:\n{}{more}",
            scoped.header(),
            text.get(..end).unwrap_or_default()
        )
    };
    let same = |other: &&Scoped| other.prefix == scoped.prefix;
    let before = archive
        .get(at.saturating_sub(context)..at)
        .unwrap_or_default();
    parts.extend(before.iter().filter(same).map(near));
    parts.push(format!("{} ({total} lines):\n{body}", scoped.header()));
    let after = archive.get(at.saturating_add(1)..).unwrap_or_default();
    parts.extend(after.iter().take(context).filter(same).map(near));
    Ok(parts.join("\n\n"))
}

/// `text`, whole when it is short, else its first lines within
/// [`MAX_CHARS`] and a note on how to read the rest.
fn cut(text: &str, lines: &[&str]) -> String {
    if text.len() <= MAX_CHARS {
        return text.to_owned();
    }
    let mut used = 0usize;
    let shown = lines
        .iter()
        .take_while(|line| {
            used = used.saturating_add(line.len()).saturating_add(1);
            used <= MAX_CHARS
        })
        .count();
    let total = lines.len();
    if shown == 0 {
        let first = lines.first().copied().unwrap_or_default();
        let first = first.get(..boundary(first, MAX_CHARS)).unwrap_or_default();
        return format!(
            "{first}\n[truncated: line 1 of {total} cut at {MAX_CHARS} characters; ask \
             `{GET}` with from_line and to_line for the rest]"
        );
    }
    let head = lines.get(..shown).unwrap_or_default().join("\n");
    format!(
        "{head}\n[truncated: lines 1-{shown} of {total}; ask `{GET}` with from_line and to_line \
         for the rest]"
    )
}

/// `text` around byte `at`, on one line.
fn snippet(text: &str, at: usize) -> String {
    let start = boundary(text, at.saturating_sub(SNIPPET_CHARS));
    let end = boundary(text, at.saturating_add(SNIPPET_CHARS * 2));
    let inner = text.get(start..end).unwrap_or_default();
    let inner: Vec<&str> = inner.split_whitespace().collect();
    let before = if start > 0 { "…" } else { "" };
    let after = if end < text.len() { "…" } else { "" };
    format!("{before}{}{after}", inner.join(" "))
}

/// The char boundary of `text` at or before byte `at`.
fn boundary(text: &str, at: usize) -> usize {
    let mut at = at.min(text.len());
    while !text.is_char_boundary(at) {
        at = at.saturating_sub(1);
    }
    at
}

/// Puts [`ARCHIVED_LINE`] at the end of entry `id` of `conversation`,
/// unless it says it already, by a splice that keeps its id and author:
/// whether it did.
pub fn mark(conversation: &mut Chat, id: MessageId) -> bool {
    let Some(at) = conversation.index_of(id) else {
        return false;
    };
    let Some(mut entry) = conversation.entry(at) else {
        return false;
    };
    if text_of(&entry.message).contains(ARCHIVED_LINE) {
        return false;
    }
    let line = format!("\n\n{ARCHIVED_LINE}");
    match std::sync::Arc::make_mut(&mut entry.message) {
        message::Message::User { content } => {
            match content.iter_mut().rev().find_map(|item| match item {
                UserContent::Text(text) => Some(text),
                _ => None,
            }) {
                Some(text) => text.text.push_str(&line),
                None => content.push(UserContent::text(ARCHIVED_LINE)),
            }
        }
        message::Message::Assistant(reply) => {
            match reply
                .content
                .iter_mut()
                .rev()
                .find_map(|block| match block {
                    AssistantContent::Text(text) => Some(text),
                    _ => None,
                }) {
                Some(text) => text.text.push_str(&line),
                None => return false,
            }
        }
        message::Message::System { content } => content.push_str(&line),
    }
    conversation.splice(at..=at, [entry], Author::plugin(NAME));
    true
}

/// rig-compaction's summary gets the line as it is reviewed, before it
/// replaces anything.
fn mark_compaction_summary(mut calls: Query<&mut Summarizing, ReadyToSettle>) {
    for mut summarizing in &mut calls {
        if let Some(summary) = summarizing
            .summary
            .as_mut()
            .filter(|summary| !summary.contains(ARCHIVED_LINE))
        {
            summary.push_str("\n\n");
            summary.push_str(ARCHIVED_LINE);
        }
    }
}

/// Another plugin's condensation summary, an entry by a plugin, gets the
/// line once its splice is committed.
fn mark_other_summaries(
    mut committed: MessageReader<ConversationChanged>,
    mut agents: Query<&mut Conversation>,
) {
    for ConversationChanged { agent, change } in committed.read() {
        let Some((summaries, _)) = condensation(&change.change) else {
            continue;
        };
        let Change::Splice { with, .. } = &change.change else {
            continue;
        };
        let by_plugin = |entry: &&Entry| {
            entry.author.as_ref().is_some_and(|author| {
                matches!(author.kind, AuthorKind::Plugin(_)) && *author != Author::plugin(NAME)
            })
        };
        let summary = with
            .iter()
            .filter(by_plugin)
            .find_map(|entry| entry.id.filter(|id| summaries.contains(id)));
        if let (Some(summary), Ok(mut conversation)) = (summary, agents.get_mut(*agent)) {
            mark(&mut conversation, summary);
        }
    }
}

#[cfg(test)]
mod tests;
