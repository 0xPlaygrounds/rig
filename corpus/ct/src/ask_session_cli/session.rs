//! The saved session a question is about: found by name, id, `last` or
//! path, read without writing ([`Session::read`]), and laid out as the
//! numbered messages the agent's tools search and fetch ([`Material`]).

use std::collections::HashMap;
use std::path::Path;

use rig_basics::session_picker::SessionEntry;
use rig_harness::harness_protocol::SessionDir;
use rig_harness::prelude::*;
use rig_harness::rig_core::conversation::pairing::{close_pending_with, pending_calls};
use rig_harness::rig_core::conversation::{Conversation, Entry, MessageId};
use rig_harness::rig_core::message::{AssistantContent, Message, ToolResultContent, UserContent};
use rig_harness::rig_memory::Session;
use rig_harness::rig_tools::shorten;

/// What a tool result still running in the asked session reads as.
const STILL_RUNNING: &str = "Still running when the session was read; no result yet.";
/// Most characters of a search hit's snippet.
const SNIPPET: usize = 200;

/// One message of the asked session, as the tools show it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Item {
    /// The turn it belongs to, by the session's turn records.
    pub turn: Option<u64>,
    /// `user`, `assistant` or `system`.
    pub role: &'static str,
    /// Its text: what was typed or answered, each tool call with its
    /// arguments, each tool result.
    pub text: String,
}

/// The asked session, read once.
#[derive(Resource, Clone, Debug, Default)]
pub struct Material {
    /// What the session was named on the command line.
    pub named: String,
    /// The conversation's entries, which the agent's conversation starts
    /// with, every tool call answered.
    pub history: Vec<Entry>,
    /// Every message the tools reach, numbered by place.
    pub items: Vec<Item>,
}

/// The saved session `named` names: its id or name in `home`, `last` for
/// the newest saved one other than `current`, or the path of its directory
/// or of a file in it.
///
/// # Errors
///
/// When none is found, with why.
pub fn find(
    named: &str,
    home: &Home,
    stores: &StoreOpener,
    current: &Path,
) -> Result<SessionDir, String> {
    if named == "last" {
        let newest = home
            .saved_sessions()
            .into_iter()
            .filter(|(_, dir)| dir.path() != current)
            .filter_map(|(_, dir)| SessionEntry::read(&dir, stores))
            .max_by_key(|entry| entry.updated);
        return newest
            .map(|entry| home.session(&entry.id))
            .ok_or_else(|| "there is no saved session".to_owned());
    }
    let found = home.find(named).map(|id| home.session(&id));
    found.or_else(|why| {
        Path::new(named)
            .ancestors()
            .find(|dir| dir.join("logs").is_dir())
            .map(SessionDir::at)
            .ok_or(why)
    })
}

impl Material {
    /// The main agent's conversation of the session in `dir`, opened
    /// through `stores` and read without writing anything.
    ///
    /// # Errors
    ///
    /// When it cannot be opened or read, or has no agent.
    pub fn read(named: &str, dir: &SessionDir, stores: &StoreOpener) -> Result<Self, String> {
        let unreadable = |why: String| format!("`{named}` cannot be read: {why}");
        let store = stores
            .open(dir)
            .map_err(|why| unreadable(why.to_string()))?;
        let session = Session::read(&*store).map_err(|why| unreadable(why.to_string()))?;
        let agent = session
            .main_agent()
            .ok_or_else(|| unreadable("it has no agent".to_owned()))?;
        Ok(Self::of(named, &agent.conversation))
    }

    /// `conversation` laid out.
    pub fn of(named: &str, conversation: &Conversation) -> Self {
        let starts: HashMap<MessageId, u64> = conversation
            .turns()
            .map(|(_, record)| (record.start, record.index))
            .collect();
        let mut history: Vec<Entry> = (0..conversation.messages().len())
            .filter_map(|at| conversation.entry(at))
            .collect();
        let messages: Vec<Message> = history
            .iter()
            .map(|entry| (*entry.message).clone())
            .collect();
        let running = pending_calls(&messages);
        if !running.is_empty() {
            history.push(Entry::from(close_pending_with(&running, STILL_RUNNING)));
        }
        let items = numbered(&history, &starts);
        Self {
            named: named.to_owned(),
            history,
            items,
        }
    }

    /// Each turn and the message it starts at.
    pub fn turns(&self) -> Vec<(u64, usize)> {
        let mut turns: Vec<(u64, usize)> = Vec::new();
        for (at, item) in self.items.iter().enumerate() {
            if let Some(turn) = item.turn
                && turns.last().is_none_or(|(last, _)| *last != turn)
            {
                turns.push((turn, at));
            }
        }
        turns
    }

    /// The `limit` messages that match most of `query`'s words, best and
    /// then earliest first, each with its number.
    pub fn search(&self, query: &str, limit: usize) -> Vec<(usize, &Item)> {
        let words: Vec<String> = query.split_whitespace().map(str::to_lowercase).collect();
        let mut hits: Vec<(usize, usize, &Item)> = self
            .items
            .iter()
            .enumerate()
            .filter_map(|(at, item)| {
                let text = item.text.to_lowercase();
                let score = words.iter().filter(|word| text.contains(*word)).count();
                (score > 0).then_some((score, at, item))
            })
            .collect();
        hits.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        hits.into_iter()
            .take(limit)
            .map(|(_, at, item)| (at, item))
            .collect()
    }
}

/// A hit's line: its number, turn and role, and its text around
/// the first word of `query` it holds.
pub fn hit_line(at: usize, item: &Item, query: &str) -> String {
    let text = item.text.replace('\n', " ");
    let lower = text.to_lowercase();
    let found = query
        .split_whitespace()
        .filter_map(|word| lower.find(&word.to_lowercase()))
        .min()
        .unwrap_or(0);
    let start = text
        .char_indices()
        .map(|(index, _)| index)
        .take_while(|index| *index + SNIPPET / 2 <= found)
        .last()
        .unwrap_or(0);
    let snippet = shorten(text.get(start..).unwrap_or_default(), SNIPPET);
    format!("{} {snippet}", header(at, item))
}

/// `#12 · turn 4 · assistant`.
pub fn header(at: usize, item: &Item) -> String {
    let turn = item.turn.map_or_else(
        || "before any turn".to_owned(),
        |turn| format!("turn {turn}"),
    );
    format!("#{at} · {turn} · {}:", item.role)
}

/// `entries` as items, each in the turn whose start it is or follows.
fn numbered(entries: &[Entry], starts: &HashMap<MessageId, u64>) -> Vec<Item> {
    let mut turn = None;
    entries
        .iter()
        .map(|entry| {
            if let Some(started) = entry.id.and_then(|id| starts.get(&id)) {
                turn = Some(*started);
            }
            let (role, text) = text_of(&entry.message);
            Item { turn, role, text }
        })
        .collect()
}

/// A message's role and its text, its tool calls and results included.
fn text_of(message: &Message) -> (&'static str, String) {
    match message {
        Message::System { content } => ("system", content.clone()),
        Message::User { content } => {
            let parts = content.iter().map(|part| match part {
                UserContent::Text(text) => text.text.clone(),
                UserContent::ToolResult(result) => {
                    let output: Vec<String> = result
                        .content
                        .iter()
                        .map(|item| match item {
                            ToolResultContent::Text(text) => text.text.clone(),
                            ToolResultContent::Json { value } => value.to_string(),
                            ToolResultContent::Image(_) => "[image]".to_owned(),
                        })
                        .collect();
                    format!("[{} result] {}", result.name.as_str(), output.join("\n"))
                }
                _ => "[attachment]".to_owned(),
            });
            ("user", parts.collect::<Vec<_>>().join("\n"))
        }
        Message::Assistant(reply) => {
            let parts = reply.content.iter().filter_map(|part| match part {
                AssistantContent::Text(text) => Some(text.text.clone()),
                AssistantContent::ToolCall(call) => Some(format!(
                    "[calls {}] {}",
                    call.function.name.as_str(),
                    call.function.raw_arguments()
                )),
                _ => None,
            });
            ("assistant", parts.collect::<Vec<_>>().join("\n"))
        }
    }
}
