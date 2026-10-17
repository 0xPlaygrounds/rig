//! What a session says about itself beside its logs, kept as a file of its
//! store: what a listing shows of it ([`SessionMeta`], in [`META`]).
//!
//! The file changes only by fields that a reader not knowing them skips and
//! a reader missing them takes as empty.

use std::collections::{BTreeMap, BTreeSet};
use std::io;

use rig_core::completion::Message;
use rig_core::conversation::EntryRef;
use rig_core::message::{AssistantContent, UserContent};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{Files, Session, main_order};
use rig_core::now_ms;

/// The file of a session's store that holds its [`SessionMeta`].
pub const META: &str = "meta.json";

/// Most characters of a title made by [`title_of`].
const TITLE_CHARS: usize = 60;

/// What a listing shows of a session, kept in its store: what it is
/// about, the labels plugins give it, and when it last changed. Its logs
/// hold the same facts; this is the cache a listing reads without folding
/// them. Fields a reader does not know are skipped, and missing ones are
/// empty.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct SessionMeta {
    /// What the session is about: the start of the first message the user
    /// typed ([`title_of`]); empty until a turn ended.
    pub title: String,
    /// Plugins' labels by tag, such as a goal or what it cost.
    pub labels: BTreeMap<String, String>,
    /// When it last changed, in milliseconds since the Unix epoch.
    pub updated: u64,
    /// The main agent's model, `vendor/model`; empty when unknown.
    pub model: String,
    /// How many messages the main agent's active line holds.
    pub messages: usize,
    /// Plugins' listed values by key, each copied here whenever it changes,
    /// so a listing shows and filters by them without opening a log; kept
    /// small.
    #[serde(rename = "data", skip_serializing_if = "BTreeMap::is_empty")]
    pub listed: BTreeMap<String, Value>,
}

impl SessionMeta {
    /// The listing kept in `files`, if it has one that reads.
    pub fn read(files: &(impl Files + ?Sized)) -> Option<Self> {
        serde_json::from_slice(&files.read_file(META).ok()?).ok()
    }

    /// Keeps it as the listing in `files`.
    ///
    /// # Errors
    ///
    /// When `files` cannot write it.
    pub fn write(&self, files: &(impl Files + ?Sized)) -> io::Result<()> {
        files.write_file(META, &serde_json::to_vec_pretty(self)?)
    }

    /// Updates this listing from `session`, for a program that writes a
    /// session no rig is running: the main agent's model and message count,
    /// `updated` set to now, and each listed value (keys in `listed`, plus
    /// those already listed), taken from the session's own values, else from
    /// the first agent by [`main_order`] that has one. A value that is gone
    /// is unlisted. The title and labels stay.
    pub fn relist(&mut self, session: &Session, listed: &BTreeSet<String>) {
        let mut agents: Vec<_> = session.agents.iter().filter(|a| !a.retired).collect();
        agents.sort_by(|a, b| {
            main_order(a.parent.is_some(), &a.id).cmp(&main_order(b.parent.is_some(), &b.id))
        });
        let main = agents.first();
        self.messages = main.map_or(0, |main| main.conversation.messages().len());
        self.model = main.and_then(|main| main.model()).unwrap_or_default();
        let keys: BTreeSet<String> = listed.iter().chain(self.listed.keys()).cloned().collect();
        self.listed = keys
            .into_iter()
            .filter_map(|key| {
                let mut holders = agents.iter().map(|agent| &agent.saved);
                let value = session
                    .saved
                    .get(&key)
                    .or_else(|| holders.find_map(|state| state.get(&key)))?;
                Some((key, value.clone()))
            })
            .collect();
        self.updated = now_ms();
    }
}

/// A title or label of `text`, as a listing shows it: on one line, each
/// run of whitespace one space, cut to 60 characters (the last an
/// ellipsis when cut).
///
/// ```
/// use rig_memory::title_of;
///
/// assert_eq!(title_of("  fix\n the   parser "), "fix the parser");
/// assert_eq!(title_of(&"a".repeat(80)).chars().count(), 60);
/// ```
pub fn title_of(text: &str) -> String {
    let line = text.split_whitespace().collect::<Vec<_>>().join(" ");
    match line.char_indices().nth(TITLE_CHARS) {
        None => line,
        Some(_) => {
            let kept = line.chars().take(TITLE_CHARS - 1);
            kept.chain(['…']).collect()
        }
    }
}

/// One line of `entry`: who wrote it, then the start of its text, or the
/// tools it called.
pub fn excerpt(entry: &EntryRef<'_>) -> String {
    let (who, text) = match entry.message {
        Message::System { content } => ("system", content.clone()),
        Message::User { content } => {
            let text: Vec<&str> = content
                .iter()
                .filter_map(|part| match part {
                    UserContent::Text(text) => Some(text.text.as_str()),
                    _ => None,
                })
                .collect();
            match text.is_empty() {
                true => ("tool results", String::new()),
                false => ("user", text.join(" ")),
            }
        }
        Message::Assistant(turn) => {
            let text = turn.text();
            let calls: Vec<&str> = turn
                .content
                .iter()
                .filter_map(|part| match part {
                    AssistantContent::ToolCall(call) => Some(call.function.name.as_str()),
                    _ => None,
                })
                .collect();
            match (text.trim().is_empty(), calls.is_empty()) {
                (true, false) => ("calls", calls.join(", ")),
                _ => ("assistant", text),
            }
        }
    };
    format!("{who}: {}", title_of(&text)).trim_end().to_owned()
}

#[cfg(test)]
mod tests;
