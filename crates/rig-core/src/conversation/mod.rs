//! Conversations: the one model of a conversation in rig, and the rules a
//! list of messages keeps so a model can read it.
//!
//! A [`Conversation`] holds its entries (each a [`Message`] with a
//! [`MessageId`] that survives edits and, when it is not the user's own or
//! the model's, an [`Author`]), its [`Aside`]s, which never reach a model,
//! and why its last user message is unanswered ([`Halt`]). Every edit is a
//! [`Change`], applied by the same code whether it is made live or read
//! back from a log, and kept in the conversation's outbox
//! ([`unsaved`](Conversation::unsaved)) until whoever saves it takes it.
//! [`pairing`] pairs each tool call with its result.
//!
//! ```
//! use rig_core::conversation::{Author, Conversation, Entry, pairing};
//! use rig_core::message::Message;
//!
//! let mut conversation = Conversation::new();
//! let hello = conversation.push(Message::user("Hello"));
//! conversation.push(Message::assistant("Hi"));
//! // A plugin rewrites the greeting; the entry keeps its id.
//! let mut greeting = conversation.entry(0).ok_or("no entry")?;
//! greeting.message = Message::user("Hello there");
//! conversation.splice(0..1, [greeting], Author::plugin("polite"));
//! assert_eq!(conversation.index_of(hello), Some(0));
//!
//! // The outbox folds into the same value.
//! let mut folded = Conversation::new();
//! for change in conversation.take_unsaved() {
//!     folded.apply(&change)?;
//! }
//! assert_eq!(folded, conversation);
//! assert!(pairing::pending_calls(conversation.messages()).is_empty());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::collections::HashSet;
use std::ops::{Bound, RangeBounds};

use serde::{Deserialize, Serialize, Serializer};

use crate::completion::Message;
use crate::message::{ToolCall, UserContent};

mod aside;
mod author;
mod lineage;
pub mod pairing;

pub use aside::{Aside, AsideId};
pub use author::{Author, AuthorKind, RequestId};
pub use lineage::Lineage;

use pairing::PairingError;

/// A conversation: its entries (each a message with an id and, when it is
/// not the user's own or the model's, an author), its asides, why its last
/// user message is unanswered, and the changes made to it that nobody has
/// saved yet. Every edit builds a [`Change`], applies it with the same code
/// a fold ([`apply`](Self::apply)) uses, and keeps it in the outbox
/// ([`unsaved`](Self::unsaved)).
///
/// Two conversations are equal when their entries, saved asides and halt
/// are: the outbox, the revision and unsaved asides are not compared, so a
/// value folded from a log equals the live one.
///
/// Its serde form is its state: entries, saved asides, halt and the next
/// ids, without the outbox, the revision or the lineage. A value read back
/// from it is [`Lineage::DETACHED`].
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(try_from = "State")]
pub struct Conversation {
    /// What a request sends, so [`messages`](Self::messages) is a slice.
    messages: Vec<Message>,
    /// Parallel to `messages`.
    meta: Vec<Meta>,
    /// In the order put.
    #[serde(serialize_with = "saved_asides")]
    asides: Vec<(AsideId, Aside)>,
    #[serde(skip_serializing_if = "Option::is_none")]
    halted: Option<Halt>,
    /// The id the next new entry gets; only rises, but for a reset to another value.
    next_id: u64,
    /// The id the next new aside gets; only rises, but for a reset to another value.
    next_aside: u64,
    /// The changes behind this value: those of the log it continues, then
    /// its own.
    #[serde(skip)]
    revision: u64,
    /// The log it continues.
    #[serde(skip)]
    lineage: Lineage,
    #[serde(skip)]
    unsaved: Vec<Recorded>,
}

/// An entry's id and author, beside its message.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Meta {
    id: MessageId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    author: Option<Author>,
}

/// A conversation as saved, checked before it becomes one.
#[derive(Deserialize)]
struct State {
    #[serde(default)]
    messages: Vec<Message>,
    #[serde(default)]
    meta: Vec<Meta>,
    #[serde(default)]
    asides: Vec<(AsideId, Aside)>,
    #[serde(default)]
    halted: Option<Halt>,
    #[serde(default)]
    next_id: u64,
    #[serde(default)]
    next_aside: u64,
}

impl TryFrom<State> for Conversation {
    type Error = ApplyError;

    fn try_from(state: State) -> Result<Self, ApplyError> {
        if state.meta.len() != state.messages.len() {
            return Err(mismatch(format!(
                "{} messages but {} ids",
                state.messages.len(),
                state.meta.len()
            )));
        }
        let mut ids = HashSet::new();
        if let Some(meta) = state
            .meta
            .iter()
            .find(|meta| meta.id.0 >= state.next_id || !ids.insert(meta.id))
        {
            return Err(mismatch(format!("message id {} reused", meta.id.0)));
        }
        let mut asides = HashSet::new();
        if let Some((id, _)) = state
            .asides
            .iter()
            .find(|(id, _)| id.0 >= state.next_aside || !asides.insert(*id))
        {
            return Err(mismatch(format!("aside id {} reused", id.0)));
        }
        Ok(Self {
            messages: state.messages,
            meta: state.meta,
            asides: state.asides,
            halted: state.halted,
            next_id: state.next_id,
            next_aside: state.next_aside,
            revision: 0,
            lineage: Lineage::DETACHED,
            unsaved: Vec::new(),
        })
    }
}

fn saved_asides<S: Serializer>(
    asides: &[(AsideId, Aside)],
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.collect_seq(asides.iter().filter(|(_, aside)| aside.saved))
}

impl PartialEq for Conversation {
    fn eq(&self, other: &Self) -> bool {
        let saved = |conversation: &Self| {
            let asides = conversation.asides.iter();
            asides
                .filter(|(_, aside)| aside.saved)
                .cloned()
                .collect::<Vec<_>>()
        };
        self.messages == other.messages
            && self.meta == other.meta
            && self.halted == other.halted
            && saved(self) == saved(other)
    }
}

/// A message's identity in its conversation: kept while entries around it
/// are added, cut or replaced, while it is spliced back changed, across a
/// save and a fork. Ids rise in the order messages were first made.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct MessageId(u64);

/// A message as it goes into a conversation: the message, who wrote it,
/// and, for a message taken out with [`Conversation::entry`], its id,
/// which a [`splice`](Conversation::splice) keeps. A recorded change
/// always carries every id.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Entry {
    /// The message.
    pub message: Message,
    /// Who wrote it; `None` for the user's own text, the model's reply and
    /// tool results.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub author: Option<Author>,
    /// Its id in the conversation, if it has one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<MessageId>,
}

impl Entry {
    /// A new entry without an author.
    pub fn new(message: Message) -> Self {
        Self {
            message,
            author: None,
            id: None,
        }
    }

    /// This entry, written by `author`.
    #[must_use]
    pub fn by(self, author: Author) -> Self {
        Self {
            author: Some(author),
            ..self
        }
    }
}

impl From<Message> for Entry {
    fn from(message: Message) -> Self {
        Self::new(message)
    }
}

/// Entry `index` of a conversation, borrowed.
#[derive(Clone, Copy, Debug)]
pub struct EntryRef<'c> {
    /// Where it is.
    pub index: usize,
    /// Its id.
    pub id: MessageId,
    /// Its message.
    pub message: &'c Message,
    /// Who wrote it, when it is not the user's own or the model's.
    pub author: Option<&'c Author>,
}

impl EntryRef<'_> {
    /// Whether it is a user message the user typed: text or attachments,
    /// not only tool results, with no author or [`AuthorKind::User`] with
    /// no title.
    pub fn is_users_own(&self) -> bool {
        let typed = match self.message {
            Message::User { content } => content
                .iter()
                .any(|item| !matches!(item, UserContent::ToolResult(_))),
            _ => false,
        };
        let users = self
            .author
            .is_none_or(|author| author.kind == AuthorKind::User && author.title.is_none());
        typed && users
    }
}

/// The text of the entry that goes between a request the user stopped and
/// the next entry, so the model does not carry the request out. rig pushes
/// it as its own entry, by [`Author::rig`].
pub const STOPPED: &str = "[The request above was stopped by the user before it was answered. \
                           Do not carry it out unless asked again.]";

/// Why the user's last message is left without an answer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Halt {
    /// The user stopped the turn answering it: the next entry pushed goes
    /// after [`STOPPED`].
    Stopped,
    /// It was left for the next turn, such as a note to an idle agent or a
    /// request whose turn failed.
    #[default]
    Kept,
}

/// One edit of a conversation. Its serde form is the conversation part of
/// the saved format.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Change {
    /// Entries added at the end, each with its id.
    Push {
        /// The entries.
        entries: Vec<Entry>,
    },
    /// Entries `from..to`, whose ids were `removed`, replaced by `with`
    /// (each with its id). `removed` doubles as a check: applying it where
    /// the range holds other ids fails.
    Splice {
        /// The first entry replaced.
        from: usize,
        /// The end of the range replaced.
        to: usize,
        /// The ids of entries `from..to` before the splice.
        removed: Vec<MessageId>,
        /// What replaces them.
        with: Vec<Entry>,
    },
    /// The halt set, or with `None` cleared.
    Halt {
        /// The halt.
        halt: Option<Halt>,
    },
    /// Aside `id` put or replaced (`Some`), or removed (`None`).
    Aside {
        /// The aside's id.
        id: AsideId,
        /// The aside.
        aside: Option<Aside>,
    },
    /// The whole value replaced: a component inserted over another, or a
    /// conversation built elsewhere and saved onto a log it was not read
    /// from.
    Reset {
        /// The new value.
        state: Box<Conversation>,
    },
}

/// A change as kept in the outbox and saved: who made it and when.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Recorded {
    /// The change.
    #[serde(flatten)]
    pub change: Change,
    /// Who made it: a push's entry author, a splice's `by`, an aside's
    /// author; `None` for rig's own.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub by: Option<Author>,
    /// Milliseconds since the Unix epoch; the writer stamps it when `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at: Option<u64>,
}

/// What a splice changed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Spliced {
    /// The ids no longer in the conversation.
    pub removed: Vec<MessageId>,
    /// The ids of the entries it put in, kept or new.
    pub changed: Vec<MessageId>,
}

/// Why a recorded change could not be applied: it does not fit the
/// conversation, such as a splice of a range holding other ids.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ApplyError {
    /// The change does not match the conversation.
    #[error("{0}")]
    Mismatch(String),
}

fn mismatch(why: String) -> ApplyError {
    ApplyError::Mismatch(why)
}

/// The first characters of the agent id `id`, enough to tell agents apart.
pub fn short_id(id: &str) -> &str {
    id.get(..8).unwrap_or(id)
}

impl Conversation {
    /// An empty conversation.
    pub fn new() -> Self {
        Self::default()
    }

    /// A conversation of `messages`, each its own entry, recorded so that
    /// saving it writes them (an import, a library user's history).
    pub fn from_messages(messages: impl IntoIterator<Item = Message>) -> Self {
        let mut conversation = Self::new();
        conversation.extend(messages);
        conversation
    }

    // Reading.

    /// The messages, oldest first: what a request sends.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// The entries, oldest first.
    pub fn entries(&self) -> impl DoubleEndedIterator<Item = EntryRef<'_>> + ExactSizeIterator {
        let entries = self.messages.iter().zip(&self.meta).enumerate();
        entries.map(|(index, (message, meta))| EntryRef {
            index,
            id: meta.id,
            message,
            author: meta.author.as_ref(),
        })
    }

    /// Entry `at`, with its author and id, to splice back changed.
    pub fn entry(&self, at: usize) -> Option<Entry> {
        let meta = self.meta.get(at)?;
        Some(Entry {
            message: self.messages.get(at)?.clone(),
            author: meta.author.clone(),
            id: Some(meta.id),
        })
    }

    /// The id of entry `at`.
    pub fn id(&self, at: usize) -> Option<MessageId> {
        self.meta.get(at).map(|meta| meta.id)
    }

    /// Who wrote entry `at`; `None` for the user's own text, the model's
    /// reply, a tool result, or no such entry.
    pub fn author(&self, at: usize) -> Option<&Author> {
        self.meta.get(at)?.author.as_ref()
    }

    /// Where entry `id` is now; `None` once an edit took it out.
    pub fn index_of(&self, id: MessageId) -> Option<usize> {
        self.meta.iter().position(|meta| meta.id == id)
    }

    /// Where a span that began at entry `id` begins now, such as a turn:
    /// its index while it is there; the end for an entry not made yet; 0
    /// for one an edit took out, as a summary then stands for what came
    /// before it.
    pub fn start_of(&self, id: MessageId) -> usize {
        self.index_of(id).unwrap_or(if id.0 >= self.next_id {
            self.messages.len()
        } else {
            0
        })
    }

    /// Where a turn starting now begins: the first of the user messages at
    /// the end, which it answers, else the first entry it adds.
    pub fn turn_start(&self) -> MessageId {
        let trailing = self.messages.iter().rev();
        let trailing = trailing
            .take_while(|message| matches!(message, Message::User { .. }))
            .count();
        let first = self.messages.len().saturating_sub(trailing);
        let answered = (trailing > 0).then(|| self.id(first)).flatten();
        answered.unwrap_or(MessageId(self.next_id))
    }

    /// Why the user's last message is left unanswered, if it was halted.
    /// Cleared when an entry is pushed or a splice reaches the end.
    pub fn halted(&self) -> Option<Halt> {
        self.halted
    }

    /// Whether the model owes an answer: the last message is the user's,
    /// and it was not halted.
    pub fn awaits_model(&self) -> bool {
        self.halted.is_none() && matches!(self.messages.last(), Some(Message::User { .. }))
    }

    /// The tool calls of the last assistant message that nothing answers
    /// yet. See [`pairing::pending_calls`].
    pub fn pending_calls(&self) -> Vec<ToolCall> {
        pairing::pending_calls(&self.messages)
    }

    /// Whether the messages pair every tool call with its result. See
    /// [`pairing::validate`].
    pub fn validate(&self) -> Result<(), PairingError> {
        pairing::validate(&self.messages)
    }

    /// The asides, in the order put.
    pub fn asides(&self) -> impl Iterator<Item = (AsideId, &Aside)> {
        self.asides.iter().map(|(id, aside)| (*id, aside))
    }

    /// Aside `id`.
    pub fn aside(&self, id: AsideId) -> Option<&Aside> {
        let mut asides = self.asides.iter();
        asides
            .find(|(known, _)| *known == id)
            .map(|(_, aside)| aside)
    }

    /// Where `aside` is drawn: before entry `position`, after entry
    /// `position - 1`. While an edit took out the entry it follows, after
    /// the entry that stands for it now ([`start_of`](Self::start_of)).
    pub fn aside_position(&self, aside: &Aside) -> usize {
        let Some(after) = aside.after else {
            return 0;
        };
        match self.index_of(after) {
            Some(index) => index.saturating_add(1),
            None => self
                .messages
                .len()
                .min(self.start_of(after).saturating_add(1)),
        }
    }

    // Editing: each builds a change with every id filled in, applies it and
    // records it.

    /// Pushes `entry`, after a [`STOPPED`] entry by [`Author::rig`] in the
    /// same change when the user stopped the last request; a halt is
    /// cleared. A pushed entry always gets a new id. Its id.
    pub fn push(&mut self, entry: impl Into<Entry>) -> MessageId {
        let entry = entry.into();
        let by = entry.author.clone();
        let mut ids = self.push_entries(vec![entry], by);
        ids.pop().unwrap_or(MessageId(self.next_id))
    }

    /// Pushes each of `messages` as its own entry without an author, in one
    /// change, as [`push`](Self::push) does.
    pub fn extend(&mut self, messages: impl IntoIterator<Item = Message>) {
        let entries = messages.into_iter().map(Entry::new).collect();
        self.push_entries(entries, None);
    }

    fn push_entries(&mut self, entries: Vec<Entry>, by: Option<Author>) -> Vec<MessageId> {
        if entries.is_empty() {
            return Vec::new();
        }
        let stop = (self.halted == Some(Halt::Stopped))
            .then(|| Entry::new(Message::user(STOPPED)).by(Author::rig()));
        let mut next = self.next_id;
        let mut ids = Vec::with_capacity(entries.len());
        let mut fill = |mut entry: Entry| {
            let id = MessageId(next);
            next = next.saturating_add(1);
            entry.id = Some(id);
            (entry, id)
        };
        let stop = stop.map(|entry| fill(entry).0);
        let entries = entries.into_iter().map(|entry| {
            let (entry, id) = fill(entry);
            ids.push(id);
            entry
        });
        let entries = stop.into_iter().chain(entries).collect();
        self.record(Change::Push { entries }, by);
        self.clear_halt();
        ids
    }

    /// Replaces entries in `range`, clamped to the conversation, with
    /// `with`, recorded as made by `by`. An entry of `with` keeps its id
    /// when that id was in the range and no earlier entry of `with` kept
    /// it; any other gets a new id. Nothing merges. A splice reaching the
    /// end clears the halt. An empty range replaced by nothing changes and
    /// records nothing.
    pub fn splice(
        &mut self,
        range: impl RangeBounds<usize>,
        with: impl IntoIterator<Item = Entry>,
        by: Author,
    ) -> Spliced {
        let len = self.messages.len();
        let to = match range.end_bound() {
            Bound::Included(&end) => end.saturating_add(1),
            Bound::Excluded(&end) => end,
            Bound::Unbounded => len,
        }
        .min(len);
        let from = match range.start_bound() {
            Bound::Included(&start) => start,
            Bound::Excluded(&start) => start.saturating_add(1),
            Bound::Unbounded => 0,
        }
        .min(to);
        let mut with: Vec<Entry> = with.into_iter().collect();
        if from == to && with.is_empty() {
            return Spliced::default();
        }
        let replaced = self.meta.get(from..to).unwrap_or_default();
        let removed: Vec<MessageId> = replaced.iter().map(|meta| meta.id).collect();
        let in_range: HashSet<MessageId> = removed.iter().copied().collect();
        let mut kept = HashSet::new();
        let mut next = self.next_id;
        for entry in &mut with {
            let keeps = entry
                .id
                .filter(|id| in_range.contains(id) && kept.insert(*id));
            entry.id = Some(keeps.unwrap_or_else(|| {
                let id = MessageId(next);
                next = next.saturating_add(1);
                id
            }));
        }
        let changed = with.iter().filter_map(|entry| entry.id).collect();
        let gone = removed.iter().filter(|id| !kept.contains(id)).copied();
        let spliced = Spliced {
            removed: gone.collect(),
            changed,
        };
        let change = Change::Splice {
            from,
            to,
            removed,
            with,
        };
        self.record(change, Some(by));
        if to == len {
            self.clear_halt();
        }
        spliced
    }

    /// Leaves the user's last message unanswered, for `halt`. Whether it
    /// was halted now: not when the model owes no answer.
    pub fn halt(&mut self, halt: Halt) -> bool {
        let halts = self.awaits_model();
        if halts {
            self.record(Change::Halt { halt: Some(halt) }, None);
        }
        halts
    }

    /// Asks the model again for an answer to the user's last message, also
    /// a halted one; whether the last message is the user's.
    pub fn resume(&mut self) -> bool {
        self.clear_halt();
        self.awaits_model()
    }

    fn clear_halt(&mut self) {
        if self.halted.is_some() {
            self.record(Change::Halt { halt: None }, None);
        }
    }

    /// Puts `aside` after the asides already put. An unsaved aside is not
    /// recorded. Its id.
    pub fn add_aside(&mut self, aside: Aside) -> AsideId {
        let id = AsideId(self.next_aside);
        if aside.saved {
            let by = Some(aside.author.clone());
            let aside = Some(aside);
            self.record(Change::Aside { id, aside }, by);
        } else {
            self.place_aside(id, aside);
        }
        id
    }

    /// Replaces aside `id` with `aside`, in its place; whether there was
    /// one. Replacing a saved aside with an unsaved one records its
    /// removal.
    pub fn replace_aside(&mut self, id: AsideId, aside: Aside) -> bool {
        let Some(was_saved) = self.aside(id).map(|old| old.saved) else {
            return false;
        };
        if aside.saved {
            let by = Some(aside.author.clone());
            let aside = Some(aside);
            self.record(Change::Aside { id, aside }, by);
        } else {
            // The saved value loses it; the live one keeps it in its place.
            self.place_aside(id, aside);
            if was_saved {
                self.log(Change::Aside { id, aside: None }, None);
            }
        }
        true
    }

    /// Removes aside `id`; it, if there was one.
    pub fn remove_aside(&mut self, id: AsideId) -> Option<Aside> {
        let at = self.asides.iter().position(|(known, _)| *known == id)?;
        let (_, aside) = self.asides.remove(at);
        if aside.saved {
            self.log(Change::Aside { id, aside: None }, None);
        }
        Some(aside)
    }

    /// Puts `aside` as `id` without recording it: in place of the aside
    /// with that id, else last.
    fn place_aside(&mut self, id: AsideId, aside: Aside) {
        self.next_aside = self.next_aside.max(id.0.saturating_add(1));
        match self.asides.iter_mut().find(|(known, _)| *known == id) {
            Some((_, slot)) => *slot = aside,
            None => self.asides.push((id, aside)),
        }
    }

    // The record.

    /// Applies a recorded change verbatim, such as one read from a log or
    /// sent by a process that mirrors this conversation. It is not put in
    /// the outbox. Fails, changing nothing, when it does not fit.
    pub fn apply(&mut self, change: &Recorded) -> Result<(), ApplyError> {
        self.change(&change.change)?;
        self.revision = self.revision.saturating_add(1);
        Ok(())
    }

    /// The changes made that nobody has saved yet, oldest first.
    pub fn unsaved(&self) -> &[Recorded] {
        &self.unsaved
    }

    /// Takes the changes made that nobody has saved yet, to save them.
    pub fn take_unsaved(&mut self) -> Vec<Recorded> {
        std::mem::take(&mut self.unsaved)
    }

    /// The changes behind this value, one per record: those of the log it
    /// continues ([`lineage`](Self::lineage)) when it was last loaded or
    /// saved, then each applied or made since. An unsaved aside counts
    /// none.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// The saved log this value continues: [`Lineage::EMPTY`] when it was
    /// begun here from nothing and never saved, [`Lineage::DETACHED`] when
    /// it was read back from its serde form.
    pub fn lineage(&self) -> Lineage {
        self.lineage
    }

    /// Whether this value's unsaved changes, appended to the log of
    /// `lineage` that holds `changes` changes, give this value: it
    /// continues that log, holding all of its changes and since then only
    /// its unsaved ones. A value begun here from nothing continues any
    /// empty log.
    pub fn continues(&self, lineage: Lineage, changes: u64) -> bool {
        let base = self.revision.saturating_sub(self.unsaved.len() as u64);
        base == changes
            && (self.lineage == lineage || (changes == 0 && self.lineage == Lineage::EMPTY))
    }

    /// Marks this value as holding the `changes` changes of the log of
    /// `lineage`, then its unsaved ones: what a store does once it loaded
    /// the value from that log or saved it there.
    pub fn continue_log(&mut self, lineage: Lineage, changes: u64) {
        self.lineage = lineage;
        self.revision = changes.saturating_add(self.unsaved.len() as u64);
    }

    /// A copy with an empty outbox, for edits never saved (a request's
    /// copy, a branch in memory).
    pub fn view(&self) -> Self {
        Self {
            messages: self.messages.clone(),
            meta: self.meta.clone(),
            asides: self.asides.clone(),
            halted: self.halted,
            next_id: self.next_id,
            next_aside: self.next_aside,
            revision: self.revision,
            lineage: self.lineage,
            unsaved: Vec::new(),
        }
    }

    /// Applies `change`, built live so it fits, and records it.
    fn record(&mut self, change: Change, by: Option<Author>) {
        match self.change(&change) {
            Ok(()) => self.log(change, by),
            Err(error) => tracing::error!(%error, "a conversation edit did not fit; dropped"),
        }
    }

    /// Records `change`, already applied.
    fn log(&mut self, change: Change, by: Option<Author>) {
        self.revision = self.revision.saturating_add(1);
        self.unsaved.push(Recorded {
            change,
            by,
            at: None,
        });
    }

    /// Applies `change`, taking its ids verbatim, after checking it fits.
    fn change(&mut self, change: &Change) -> Result<(), ApplyError> {
        match change {
            Change::Push { entries } => {
                let next = self.new_ids(entries, &HashSet::new())?;
                for entry in entries {
                    self.messages.push(entry.message.clone());
                    let (id, author) = (entry.id.unwrap_or_default(), entry.author.clone());
                    self.meta.push(Meta { id, author });
                }
                self.next_id = next;
            }
            Change::Splice {
                from,
                to,
                removed,
                with,
            } => {
                let (from, to) = (*from, *to);
                let replaced = self.meta.get(from..to).ok_or_else(|| {
                    mismatch(format!(
                        "splice of {from}..{to} in {} entries",
                        self.meta.len()
                    ))
                })?;
                if !replaced.iter().map(|meta| &meta.id).eq(removed) {
                    return Err(mismatch(format!("splice of {from}..{to} names other ids")));
                }
                let next = self.new_ids(with, &removed.iter().copied().collect())?;
                self.messages
                    .splice(from..to, with.iter().map(|entry| entry.message.clone()));
                self.meta.splice(
                    from..to,
                    with.iter().map(|entry| Meta {
                        id: entry.id.unwrap_or_default(),
                        author: entry.author.clone(),
                    }),
                );
                self.next_id = next;
            }
            Change::Halt { halt } => self.halted = *halt,
            Change::Aside {
                id,
                aside: Some(aside),
            } => self.place_aside(*id, aside.clone()),
            Change::Aside { id, aside: None } => {
                let at = self.asides.iter().position(|(known, _)| known == id);
                let at = at.ok_or_else(|| mismatch(format!("no aside {}", id.0)))?;
                self.asides.remove(at);
            }
            // Exactly the new value, its next ids too: the live value goes
            // on numbering from them, so a fold must as well.
            Change::Reset { state } => {
                let (unsaved, revision) = (std::mem::take(&mut self.unsaved), self.revision);
                let lineage = self.lineage;
                *self = state.view();
                self.unsaved = unsaved;
                self.revision = revision;
                self.lineage = lineage;
            }
        }
        Ok(())
    }

    /// Checks that each of `entries` has an id that is either one of
    /// `reused`, at most once, or new: rising from the next id. The next id
    /// after them.
    fn new_ids(&self, entries: &[Entry], reused: &HashSet<MessageId>) -> Result<u64, ApplyError> {
        let mut next = self.next_id;
        let mut seen = HashSet::new();
        for entry in entries {
            let id = entry
                .id
                .ok_or_else(|| mismatch("a recorded entry without an id".to_owned()))?;
            if reused.contains(&id) && seen.insert(id) {
                continue;
            }
            if id.0 < next {
                return Err(mismatch(format!("message id {} reused", id.0)));
            }
            next = id.0.saturating_add(1);
        }
        Ok(next)
    }
}

#[cfg(test)]
mod tests;
