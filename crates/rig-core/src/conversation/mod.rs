//! One conversation model for all of rig: [`Conversation`], its edits as
//! [`Change`]s, [`TurnRecord`]s kept as asides, and [`pairing`] of tool
//! calls with their results.
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
//! greeting.message = Message::user("Hello there").into();
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

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::ops::{Bound, Range, RangeBounds};
use std::str::FromStr;
use std::sync::Arc;

use serde::{Deserialize, Serialize, Serializer};

use crate::completion::Message;
use crate::message::{StopReason, ToolCall, UserContent};

mod aside;
mod author;
mod lineage;
mod note;
pub mod pairing;
mod turn;

pub use aside::{Aside, AsideId};
pub use author::{Author, AuthorKind, RequestId};
pub use lineage::Lineage;
pub use note::Note;
pub use turn::{TURN, TurnEnd, TurnRecord};

use pairing::{Normalized, PairingError};

/// How a text file attached to a user's message starts, followed by its
/// label and `">`.
pub const FILE_OPEN: &str = "<file path=\"";

/// How a text file attached to a user's message ends.
pub const FILE_CLOSE: &str = "</file>";

/// The label and the contents of an attached text file, when `text` is
/// one, so a view can show it without the markup around it.
pub fn attached_file(text: &str) -> Option<(&str, &str)> {
    let rest = text.strip_prefix(FILE_OPEN)?.strip_suffix(FILE_CLOSE)?;
    rest.split_once("\">\n")
}

/// A conversation: its entries (each a message with an id and, when it is
/// not the user's own or the model's, an author), its asides, why its last
/// user message is unanswered, and the changes made to it that nobody has
/// saved yet. Every edit builds a [`Change`], applies it with the same code
/// a fold ([`apply`](Self::apply)) uses, and keeps it in the outbox
/// ([`unsaved`](Self::unsaved)).
///
/// Two conversations are equal when their entries (messages, ids and
/// authors), saved asides and halt are: the outbox, the revision, unsaved
/// asides and when entries were made are not compared, so a value folded
/// from a log equals the live one and two values built alike are equal.
///
/// Its serde form is its state: entries, saved asides, halt and the next
/// ids, without the outbox, the revision or the lineage. A value read back
/// from it is [`Lineage::DETACHED`].
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(try_from = "State")]
pub struct Conversation {
    /// What a request sends, so [`messages`](Self::messages) is a slice;
    /// shared with the requests and records made from it.
    messages: Vec<Arc<Message>>,
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
    unsaved: Vec<ChangeRecord>,
    /// The time edits are made at, set by [`dated`](Self::dated); now when
    /// `None`.
    #[serde(skip)]
    clock: Option<u64>,
}

/// An entry's id, author and when it was made, beside its message. Its
/// serde form carries `made`, so a reset and a value read back keep it;
/// equality leaves it out.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Meta {
    id: MessageId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    author: Option<Author>,
    /// Milliseconds since the Unix epoch: when the entry was first made.
    #[serde(default)]
    made: u64,
}

/// A conversation as saved, checked before it becomes one.
#[derive(Deserialize)]
struct State {
    #[serde(default)]
    messages: Vec<Arc<Message>>,
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
            clock: None,
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
        let saved = |(_, aside): &&(AsideId, Aside)| aside.saved;
        let same = |a: &Meta, b: &Meta| a.id == b.id && a.author == b.author;
        self.messages == other.messages
            && self.meta.len() == other.meta.len()
            && self.meta.iter().zip(&other.meta).all(|(a, b)| same(a, b))
            && self.halted == other.halted
            && self
                .asides
                .iter()
                .filter(saved)
                .eq(other.asides.iter().filter(saved))
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

impl MessageId {
    /// Its number: ids rise from 0 in the order messages were first made.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Its plain form, `m` and its number (`m4`): how commands name a message.
impl fmt::Display for MessageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "m{}", self.0)
    }
}

/// Reads its plain form, `m4`, or the number alone.
impl FromStr for MessageId {
    type Err = ParseIdError;

    fn from_str(text: &str) -> Result<Self, ParseIdError> {
        parse_id(text, 'm').map(Self)
    }
}

/// A [`MessageId`] or [`AsideId`] that could not be read from its plain form.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{0:?} is not an id")]
pub struct ParseIdError(pub String);

/// The number of `text`, written with or without `prefix`.
fn parse_id(text: &str, prefix: char) -> Result<u64, ParseIdError> {
    let number = text.strip_prefix(prefix).unwrap_or(text);
    number.parse().map_err(|_| ParseIdError(text.to_owned()))
}

/// A message's identity beyond its conversation: when it was made and its
/// id. Equal for a message and its copy in a fork (a fork copies ids and
/// made-times), different for one made since, unless both were made in
/// the same millisecond with the same id.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct MessageKey {
    /// When it was first made, in milliseconds since the Unix epoch.
    pub made: u64,
    /// Its id in its conversation.
    pub id: MessageId,
}

/// A message as it goes into a conversation: the message, who wrote it,
/// and, for a message taken out with [`Conversation::entry`], its id,
/// which a [`splice`](Conversation::splice) keeps. A recorded change
/// always carries every id.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Entry {
    /// The message, shared with the conversation it goes into.
    pub message: Arc<Message>,
    /// Who wrote it; `None` for the user's own text, the model's reply and
    /// tool results.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub author: Option<Author>,
    /// Its id in the conversation, if it has one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<MessageId>,
    /// When it was first made, in milliseconds since the Unix epoch: set by
    /// [`Conversation::entry`], kept by an entry put back as it was. A
    /// recorded change carries it only for such an entry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub made: Option<u64>,
}

impl Entry {
    /// A new entry without an author.
    pub fn new(message: impl Into<Arc<Message>>) -> Self {
        Self {
            message: message.into(),
            author: None,
            id: None,
            made: None,
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

impl From<Arc<Message>> for Entry {
    fn from(message: Arc<Message>) -> Self {
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
    /// When it was first made, in milliseconds since the Unix epoch.
    pub made: u64,
}

impl EntryRef<'_> {
    /// Its identity beyond its conversation.
    pub fn key(&self) -> MessageKey {
        MessageKey {
            made: self.made,
            id: self.id,
        }
    }

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

    /// Whether it asks something of the model: a user message with text
    /// or attachments, not only tool results, that is not rig's own (the
    /// [`STOPPED`] note). The user's, an agent's or a plugin's.
    pub fn is_request(&self) -> bool {
        let asks = match self.message {
            Message::User { content } => content
                .iter()
                .any(|item| !matches!(item, UserContent::ToolResult(_))),
            _ => false,
        };
        asks && self
            .author
            .is_none_or(|author| author.kind != AuthorKind::Rig)
    }

    /// Whether it is the [`STOPPED`] note rig puts between a request the
    /// user stopped and what came next.
    pub fn is_stop_marker(&self) -> bool {
        let by_rig = self
            .author
            .is_some_and(|author| author.kind == AuthorKind::Rig);
        by_rig
            && matches!(self.message, Message::User { content }
                if matches!(content.as_slice(), [UserContent::Text(text)] if text.text == STOPPED))
    }
}

/// The text of the entry that goes between a request the user stopped and
/// the next entry, so the model does not carry the request out. rig pushes
/// it as its own entry, by [`Author::rig`].
pub const STOPPED: &str = "[The request above was stopped by the user before it was answered. \
                           Do not carry it out unless asked again.]";

/// Why the user's last message is left without an answer, so a restore
/// does not answer it. How a turn ended is its [`TurnRecord`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Halt {
    /// The user stopped the turn answering it: the next entry pushed goes
    /// after [`STOPPED`].
    Stopped,
    /// It is left for the next turn: a note to an idle agent, a request a
    /// plugin stopped, or another author's message a failed turn left.
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
pub struct ChangeRecord {
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
    /// The ids of the entries it put in, kept, put back or new.
    pub changed: Vec<MessageId>,
    /// The ids [`restore`](Conversation::restore) put back.
    pub restored: Vec<MessageId>,
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
    pub fn from_messages(messages: impl IntoIterator<Item = impl Into<Arc<Message>>>) -> Self {
        let mut conversation = Self::new();
        conversation.extend(messages);
        conversation
    }

    // Reading.

    /// The messages, oldest first: what a request sends.
    pub fn messages(&self) -> &[Arc<Message>] {
        &self.messages
    }

    /// The messages, oldest first, taken out of the conversation: what a
    /// request sends, without a copy.
    pub fn into_messages(self) -> Vec<Arc<Message>> {
        self.messages
    }

    /// The entries, oldest first.
    pub fn entries(&self) -> impl DoubleEndedIterator<Item = EntryRef<'_>> + ExactSizeIterator {
        let entries = self.messages.iter().zip(&self.meta).enumerate();
        entries.map(|(index, (message, meta))| EntryRef {
            index,
            id: meta.id,
            message,
            author: meta.author.as_ref(),
            made: meta.made,
        })
    }

    /// Entry `at`, with its author, id and when it was made, to splice
    /// back changed.
    pub fn entry(&self, at: usize) -> Option<Entry> {
        let meta = self.meta.get(at)?;
        Some(Entry {
            message: self.messages.get(at)?.clone(),
            author: meta.author.clone(),
            id: Some(meta.id),
            made: Some(meta.made),
        })
    }

    /// The id of entry `at`.
    pub fn id(&self, at: usize) -> Option<MessageId> {
        self.meta.get(at).map(|meta| meta.id)
    }

    /// The identity of entry `at` beyond this conversation.
    pub fn key(&self, at: usize) -> Option<MessageKey> {
        let meta = self.meta.get(at)?;
        Some(MessageKey {
            made: meta.made,
            id: meta.id,
        })
    }

    /// Where the entry with identity `key` is now; `None` when it is not
    /// here, also when an entry with its id was made at another time.
    pub fn index_of_key(&self, key: MessageKey) -> Option<usize> {
        let at = self.index_of(key.id)?;
        (self.meta.get(at)?.made == key.made).then_some(at)
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
            .take_while(|message| matches!(&***message, Message::User { .. }))
            .count();
        let first = self.messages.len().saturating_sub(trailing);
        let answered = (trailing > 0).then(|| self.id(first)).flatten();
        answered.unwrap_or(MessageId(self.next_id))
    }

    /// The entries a turn that began at `start` (its
    /// [`turn_start`](Self::turn_start)) answers: the user messages in a
    /// row from where it begins now ([`start_of`](Self::start_of)). After a
    /// stopped turn they begin with that turn's tool results and the
    /// [`STOPPED`] note, before what was sent next.
    pub fn asked(&self, start: MessageId) -> impl Iterator<Item = EntryRef<'_>> {
        let entries = self.entries().skip(self.start_of(start));
        entries.take_while(|entry| matches!(entry.message, Message::User { .. }))
    }

    /// The message that asked for the turn that began at `start`: the
    /// first of the entries it answers ([`asked`](Self::asked)) that is a
    /// request ([`EntryRef::is_request`]).
    pub fn request(&self, start: MessageId) -> Option<EntryRef<'_>> {
        self.asked(start).find(EntryRef::is_request)
    }

    /// Why the user's last message is left unanswered, if it was halted.
    /// Cleared when an entry is pushed or a splice reaches the end.
    pub fn halted(&self) -> Option<Halt> {
        self.halted
    }

    /// Whether the model owes an answer: the last message a model reads is
    /// the user's, and it was not halted. A failed or stopped reply after it
    /// is never read ([`StopReason::is_failure`]), so the answer is still
    /// owed.
    pub fn awaits_model(&self) -> bool {
        let read = self.messages.iter().rev().map(Arc::as_ref).find(|message| {
            !matches!(message, Message::Assistant(turn)
                if turn.stop.as_ref().is_some_and(StopReason::is_failure))
        });
        self.halted.is_none() && matches!(read, Some(Message::User { .. }))
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
    pub fn asides(&self) -> impl DoubleEndedIterator<Item = (AsideId, &Aside)> {
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
    /// cleared. Returns the new entry's id; a pushed entry never reuses one.
    pub fn push(&mut self, entry: impl Into<Entry>) -> MessageId {
        let entry = entry.into();
        let by = entry.author.clone();
        let mut ids = self.push_entries(vec![entry], by);
        ids.pop().unwrap_or(MessageId(self.next_id))
    }

    /// Pushes each of `messages` as its own entry without an author, in one
    /// change, as [`push`](Self::push) does.
    pub fn extend(&mut self, messages: impl IntoIterator<Item = impl Into<Arc<Message>>>) {
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
            entry.made = None;
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
        self.replace(range, with.into_iter().collect(), by, false)
    }

    /// As [`splice`](Self::splice), and an entry of `with` whose id this
    /// conversation made (below the next id), that no entry outside the
    /// range holds and no earlier entry of `with` took, keeps that id and
    /// when it was made ([`Entry::made`], else now): entries put back as
    /// they were, such as a summary's originals or an edit undone. Asides that followed such an entry follow it again.
    /// Only for entries this conversation held: an entry from elsewhere
    /// that carries an id would take one that meant another message here.
    ///
    /// ```
    /// use rig_core::completion::Message;
    /// use rig_core::conversation::{Author, Conversation, Entry};
    ///
    /// let mut conversation = Conversation::from_messages([
    ///     Message::user("one"),
    ///     Message::assistant("two"),
    /// ]);
    /// let ids: Vec<_> = conversation.entries().map(|entry| entry.id).collect();
    /// let originals: Vec<Entry> = (0..2).filter_map(|at| conversation.entry(at)).collect();
    /// let summary = Entry::new(Message::user("a summary"));
    /// conversation.splice(.., [summary], Author::plugin("compact"));
    /// let undone = conversation.restore(.., originals, Author::plugin("undo"));
    /// assert_eq!(undone.restored, ids);
    /// assert!(conversation.entries().map(|entry| entry.id).eq(ids));
    /// ```
    pub fn restore(
        &mut self,
        range: impl RangeBounds<usize>,
        with: impl IntoIterator<Item = Entry>,
        by: Author,
    ) -> Spliced {
        self.replace(range, with.into_iter().collect(), by, true)
    }

    /// Makes the conversation hold `entries`, as made by `by`: one
    /// [`restore`](Self::restore) of the range between the longest start
    /// and end it shares with them (the same messages, ids and authors), so
    /// entries taken from it keep their ids. Nothing when they are equal.
    pub fn set_entries(&mut self, entries: Vec<Entry>, by: Author) -> Spliced {
        let same = |entry: &Entry, at: usize| {
            entry.id.is_some()
                && entry.id == self.id(at)
                && entry.author.as_ref() == self.author(at)
                && Some(&entry.message) == self.messages.get(at)
        };
        let (len, count) = (self.messages.len(), entries.len());
        let start = entries
            .iter()
            .enumerate()
            .take_while(|(at, entry)| same(entry, *at))
            .count();
        let end = entries
            .iter()
            .rev()
            .zip((0..len).rev())
            .take(len.min(count).saturating_sub(start))
            .take_while(|(entry, at)| same(entry, *at))
            .count();
        if start == len && start == count {
            return Spliced::default();
        }
        let with = entries
            .into_iter()
            .skip(start)
            .take(count.saturating_sub(start).saturating_sub(end));
        self.restore(start..len.saturating_sub(end), with, by)
    }

    /// Replaces entries in `range` with `with`: [`splice`](Self::splice),
    /// or with `restores`, [`restore`](Self::restore).
    fn replace(
        &mut self,
        range: impl RangeBounds<usize>,
        mut with: Vec<Entry>,
        by: Author,
        restores: bool,
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
        if from == to && with.is_empty() {
            return Spliced::default();
        }
        let replaced = self.meta.get(from..to).unwrap_or_default();
        let removed: Vec<MessageId> = replaced.iter().map(|meta| meta.id).collect();
        let in_range: HashSet<MessageId> = removed.iter().copied().collect();
        let mut outside = None;
        let mut taken = HashSet::new();
        let mut restored = Vec::new();
        let mut next = self.next_id;
        for entry in &mut with {
            let id = entry.id.filter(|id| !taken.contains(id));
            if let Some(id) = id.filter(|id| in_range.contains(id)) {
                taken.insert(id);
                entry.made = None;
                continue;
            }
            let back = id.filter(|id| {
                restores
                    && id.0 < self.next_id
                    && !self.outside(from..to, &mut outside).contains(id)
            });
            if let Some(id) = back {
                taken.insert(id);
                restored.push(id);
                continue;
            }
            entry.id = Some(MessageId(next));
            entry.made = None;
            next = next.saturating_add(1);
        }
        let changed = with.iter().filter_map(|entry| entry.id).collect();
        let gone = removed.iter().filter(|id| !taken.contains(id)).copied();
        let spliced = Spliced {
            removed: gone.collect(),
            changed,
            restored,
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

    /// The ids of the entries outside `range`, gathered into `outside` the
    /// first time they are asked for.
    fn outside<'s>(
        &self,
        range: Range<usize>,
        outside: &'s mut Option<HashSet<MessageId>>,
    ) -> &'s HashSet<MessageId> {
        outside.get_or_insert_with(|| {
            let before = self.meta.get(..range.start).unwrap_or_default();
            let after = self.meta.get(range.end..).unwrap_or_default();
            before.iter().chain(after).map(|meta| meta.id).collect()
        })
    }

    /// Rewrites entries in place, as made by `by`: `edit` sees each entry,
    /// oldest first, and returns the message to put in its place, or
    /// `None` to leave it. The entries it rewrote keep their ids and
    /// authors, and the whole edit is one change: a [`splice`](Self::splice)
    /// of the entries from the first rewritten to the last. Nothing is
    /// recorded when it rewrote none. Returns the ids of the entries it
    /// rewrote.
    ///
    /// ```
    /// use rig_core::completion::Message;
    /// use rig_core::conversation::{Author, Conversation};
    ///
    /// let mut conversation = Conversation::from_messages([
    ///     Message::user("one"),
    ///     Message::user("two"),
    /// ]);
    /// let before = conversation.id(1);
    /// let changed = conversation.edit_each(Author::plugin("shout"), |entry| {
    ///     (entry.index == 1).then(|| Message::user("TWO"))
    /// });
    /// assert_eq!(changed, before.into_iter().collect::<Vec<_>>());
    /// assert_eq!(conversation.messages().get(1).map(|message| &**message), Some(&Message::user("TWO")));
    /// assert_eq!(conversation.id(1), before);
    /// ```
    pub fn edit_each(
        &mut self,
        by: Author,
        mut edit: impl FnMut(EntryRef<'_>) -> Option<Message>,
    ) -> Vec<MessageId> {
        let rewritten: Vec<(usize, MessageId, Message)> = self
            .entries()
            .filter_map(|entry| {
                let (at, id) = (entry.index, entry.id);
                edit(entry).map(|message| (at, id, message))
            })
            .collect();
        let (Some(&(first, ..)), Some(&(last, ..))) = (rewritten.first(), rewritten.last()) else {
            return Vec::new();
        };
        let ids = rewritten.iter().map(|(_, id, _)| *id).collect();
        let mut rewritten = rewritten.into_iter().peekable();
        let entries: Vec<Entry> = (first..=last)
            .filter_map(|at| match rewritten.next_if(|(next, ..)| *next == at) {
                Some((_, id, message)) => Some(Entry {
                    message: Arc::new(message),
                    author: self.author(at).cloned(),
                    id: Some(id),
                    made: None,
                }),
                None => self.entry(at),
            })
            .collect();
        self.splice(first..=last, entries, by);
        ids
    }

    /// Repairs the entries as [`pairing::normalize`] does, saved with
    /// [`set_entries`](Self::set_entries) as made by `by`, so merged
    /// entries keep the first's id, author and made-time. An aside that
    /// followed an entry merged into another follows that one; one that
    /// followed a dropped entry follows the entry before it (each move
    /// recorded, by `by`). Returns what it repaired.
    pub fn normalize(&mut self, by: Author) -> Normalized {
        let entries = (0..self.messages.len()).filter_map(|at| self.entry(at));
        let normalized = pairing::normalize(entries.collect());
        let into: HashMap<MessageId, MessageId> = normalized
            .merged
            .iter()
            .map(|(kept, gone)| (*gone, *kept))
            .collect();
        let dropped: HashSet<MessageId> = normalized.dropped.iter().copied().collect();
        // Where an aside that followed `id` goes: the entry it went into,
        // or for a dropped one, the entry before it that stays.
        let anchor = |id: MessageId| -> Option<MessageId> {
            let before = self.meta.iter().rev().skip_while(|meta| meta.id != id);
            before
                .map(|meta| meta.id)
                .find_map(|id| match into.get(&id) {
                    Some(kept) => Some(*kept),
                    None => (!dropped.contains(&id)).then_some(id),
                })
        };
        let moves: Vec<(AsideId, Option<MessageId>)> = self
            .asides
            .iter()
            .filter_map(|(at, aside)| {
                let after = aside.after?;
                let moves = into.contains_key(&after) || dropped.contains(&after);
                moves.then(|| (*at, anchor(after)))
            })
            .collect();
        self.set_entries(normalized.entries.clone(), by.clone());
        for (id, after) in moves {
            self.move_aside(id, after, &by);
        }
        normalized
    }

    /// Takes everything a log holds from `from_log`, a value read from that
    /// log: entries, saved asides, halt, next ids, lineage and revision.
    /// This value's unsaved asides stay, with their ids. For a host whose
    /// log was rewritten under the value it holds.
    ///
    /// # Errors
    ///
    /// [`ApplyError::Mismatch`], changing nothing, while either value has
    /// unsaved changes, or when an unsaved aside's id is a saved one's in
    /// `from_log`.
    pub fn reread(&mut self, from_log: Conversation) -> Result<(), ApplyError> {
        if !self.unsaved.is_empty() || !from_log.unsaved.is_empty() {
            return Err(mismatch(
                "a value with unsaved changes is not reread".to_owned(),
            ));
        }
        let mut asides = from_log.asides;
        asides.retain(|(_, aside)| aside.saved);
        let unsaved = self.asides.iter().filter(|(_, aside)| !aside.saved);
        if let Some((id, _)) = unsaved
            .clone()
            .find(|(id, _)| asides.iter().any(|(saved, _)| saved == id))
        {
            return Err(mismatch(format!("aside id {} is saved in the log", id.0)));
        }
        let next_aside = unsaved
            .clone()
            .map(|(id, _)| id.0.saturating_add(1))
            .fold(from_log.next_aside, u64::max);
        asides.extend(unsaved.cloned());
        asides.sort_by_key(|(id, _)| *id);
        *self = Self {
            messages: from_log.messages,
            meta: from_log.meta,
            asides,
            halted: from_log.halted,
            next_id: from_log.next_id,
            next_aside,
            revision: from_log.revision,
            lineage: from_log.lineage,
            unsaved: Vec::new(),
            clock: self.clock,
        };
        Ok(())
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

    fn clear_halt(&mut self) {
        if self.halted.is_some() {
            self.record(Change::Halt { halt: None }, None);
        }
    }

    /// Puts `aside` after the asides already put and returns its id. An
    /// unsaved aside is not recorded.
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

    /// Removes aside `id` and returns it, or `None` if there was none.
    pub fn remove_aside(&mut self, id: AsideId) -> Option<Aside> {
        let at = self.asides.iter().position(|(known, _)| *known == id)?;
        let (_, aside) = self.asides.remove(at);
        if aside.saved {
            self.log(Change::Aside { id, aside: None }, None);
        }
        Some(aside)
    }

    /// Makes aside `id` follow `after`, recorded as made by `by` when it is
    /// saved.
    fn move_aside(&mut self, id: AsideId, after: Option<MessageId>, by: &Author) {
        let Some(aside) = self.aside(id).cloned() else {
            return;
        };
        let aside = aside.after(after);
        if aside.saved {
            let aside = Some(aside);
            self.record(Change::Aside { id, aside }, Some(by.clone()));
        } else {
            self.place_aside(id, aside);
        }
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
    pub fn apply(&mut self, change: &ChangeRecord) -> Result<(), ApplyError> {
        let at = change.at.unwrap_or_else(|| self.now());
        self.change(&change.change, at)?;
        self.revision = self.revision.saturating_add(1);
        Ok(())
    }

    /// Makes the edits of `edit` as made at `at`, in milliseconds since the
    /// Unix epoch, such as messages imported at the times their source
    /// logged them: their records carry that time, which a writer keeps,
    /// and the entries they make were made then. An edit dated inside, by a
    /// nested call, keeps its own time.
    ///
    /// ```
    /// use rig_core::conversation::Conversation;
    /// use rig_core::message::Message;
    ///
    /// let mut conversation = Conversation::new();
    /// conversation.dated(1_790_848_800_000, |conversation| {
    ///     conversation.push(Message::user("Imported"))
    /// });
    /// let at = conversation.unsaved().first().and_then(|recorded| recorded.at);
    /// assert_eq!(at, Some(1_790_848_800_000));
    /// ```
    pub fn dated<R>(&mut self, at: u64, edit: impl FnOnce(&mut Self) -> R) -> R {
        let outer = self.clock.replace(at);
        let made = edit(self);
        self.clock = outer;
        made
    }

    /// The time an edit is made at: the [`dated`](Self::dated) clock, else
    /// now.
    fn now(&self) -> u64 {
        self.clock.unwrap_or_else(crate::now_ms)
    }

    /// The changes made that nobody has saved yet, oldest first.
    pub fn unsaved(&self) -> &[ChangeRecord] {
        &self.unsaved
    }

    /// Takes the changes made that nobody has saved yet, to save them.
    pub fn take_unsaved(&mut self) -> Vec<ChangeRecord> {
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

    /// A copy with an empty outbox, for edits never saved (a preview, a
    /// reset's state).
    pub fn view(&self) -> Self {
        Self {
            asides: self.asides.clone(),
            ..self.request_view()
        }
    }

    /// A [`view`](Self::view) without the asides, for a model request's
    /// copy: a request never sends them, and one is built per model call.
    pub fn request_view(&self) -> Self {
        Self {
            messages: self.messages.clone(),
            meta: self.meta.clone(),
            asides: Vec::new(),
            halted: self.halted,
            next_id: self.next_id,
            next_aside: self.next_aside,
            revision: self.revision,
            lineage: self.lineage,
            unsaved: Vec::new(),
            clock: None,
        }
    }

    /// Applies `change`, built live so it fits, as made now, and records it.
    fn record(&mut self, change: Change, by: Option<Author>) {
        let at = self.now();
        match self.change(&change, at) {
            Ok(()) => self.log_at(change, by, at),
            Err(error) => tracing::error!(%error, "a conversation edit did not fit; dropped"),
        }
    }

    /// Records `change`, already applied, as made now.
    fn log(&mut self, change: Change, by: Option<Author>) {
        let at = self.now();
        self.log_at(change, by, at);
    }

    fn log_at(&mut self, change: Change, by: Option<Author>, at: u64) {
        self.revision = self.revision.saturating_add(1);
        self.unsaved.push(ChangeRecord {
            change,
            by,
            at: Some(at),
        });
    }

    /// Applies `change`, made at `at`, taking its ids verbatim, after
    /// checking it fits. An entry that keeps its id keeps when it was made;
    /// any other was made when its record says, else at `at`.
    fn change(&mut self, change: &Change, at: u64) -> Result<(), ApplyError> {
        let meta = |entry: &Entry, made: u64| Meta {
            id: entry.id.unwrap_or_default(),
            author: entry.author.clone(),
            made,
        };
        match change {
            Change::Push { entries } => {
                let end = self.meta.len();
                let next = self.new_ids(entries, &HashSet::new(), end..end)?;
                for entry in entries {
                    self.messages.push(entry.message.clone());
                    self.meta.push(meta(entry, entry.made.unwrap_or(at)));
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
                let kept: HashMap<MessageId, u64> =
                    replaced.iter().map(|meta| (meta.id, meta.made)).collect();
                let next = self.new_ids(with, &removed.iter().copied().collect(), from..to)?;
                let metas: Vec<Meta> = with
                    .iter()
                    .map(|entry| {
                        let made = entry.id.and_then(|id| kept.get(&id).copied());
                        meta(entry, made.or(entry.made).unwrap_or(at))
                    })
                    .collect();
                self.messages
                    .splice(from..to, with.iter().map(|entry| entry.message.clone()));
                self.meta.splice(from..to, metas);
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

    /// Checks that each of `entries`, which replace `range`, has an id
    /// that is, at most once, one of `reused` or an earlier id no entry
    /// outside `range` holds (put back), or else new: rising from the next
    /// id. Returns the next id after them.
    fn new_ids(
        &self,
        entries: &[Entry],
        reused: &HashSet<MessageId>,
        range: Range<usize>,
    ) -> Result<u64, ApplyError> {
        let mut next = self.next_id;
        let mut seen = HashSet::new();
        let mut outside = None;
        for entry in entries {
            let id = entry
                .id
                .ok_or_else(|| mismatch("a recorded entry without an id".to_owned()))?;
            if id.0 >= next {
                next = id.0.saturating_add(1);
                continue;
            }
            let fits = reused.contains(&id)
                || (id.0 < self.next_id
                    && !self.outside(range.clone(), &mut outside).contains(&id));
            if !fits || !seen.insert(id) {
                return Err(mismatch(format!("message id {} reused", id.0)));
            }
        }
        Ok(next)
    }
}

#[cfg(test)]
mod tests;
