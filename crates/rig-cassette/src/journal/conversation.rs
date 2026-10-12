//! An agent's conversation as its journal keeps it: the messages, each
//! with a [`MessageId`] that survives edits, where each delivered text came
//! from ([`Origin`]), and why the user's last message is left unanswered
//! ([`Halt`]).

use std::collections::HashSet;

#[cfg(feature = "reflect")]
use bevy_reflect::prelude::{Reflect, ReflectDefault, ReflectDeserialize, ReflectSerialize};
use rig_core::completion::Message;
use rig_core::message::UserContent;
use serde::{Deserialize, Serialize};

/// The conversation: every message sent to and received from the model,
/// each with its [`MessageId`], and where each delivered text came from
/// when it is not the user's own. A host that keeps it in a journal
/// changes it only as it logs each change: a message added, or a
/// [`splice`](Self::splice) that cuts, replaces or edits messages after
/// the fact.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "reflect", derive(Reflect))]
#[cfg_attr(
    feature = "reflect",
    reflect(opaque, Default, Clone, Serialize, Deserialize)
)]
pub struct Conversation {
    messages: Vec<Message>,
    /// Each message's id, by index.
    #[serde(default)]
    ids: Vec<MessageId>,
    /// The id the next new message gets.
    #[serde(default)]
    next_id: u64,
    /// The origin of each content item that is not the user's own or the
    /// model's, by message and item index.
    origins: Vec<(usize, usize, Origin)>,
    /// Why the user's last message was left unanswered, until a message
    /// is added or the end is spliced.
    #[serde(default)]
    halted: Option<Halt>,
}

/// A message's identity in its agent's conversation: it stays while
/// messages around it are added, cut or replaced, while any of its content
/// items survive an edit, and across a restore. Ids rise in the order
/// messages were first made.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
#[cfg_attr(feature = "reflect", derive(Reflect))]
#[cfg_attr(feature = "reflect", reflect(Clone, Debug, Default, PartialEq, Hash))]
#[serde(transparent)]
pub struct MessageId(pub(crate) u64);

/// A message as a [`splice`](Conversation::splice) puts it in: the
/// message, where its content items came from (by item index), and, for a
/// message taken from the conversation ([`Conversation::entry`]), its id,
/// which the splice keeps.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Entry {
    /// The message.
    pub message: Message,
    /// The origin of each content item that has one, by item index.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub origins: Vec<(usize, Origin)>,
    /// The id of the message it was taken from, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<MessageId>,
}

impl Entry {
    /// A new message, every content item of it from `origin`.
    pub fn from(message: Message, origin: Origin) -> Self {
        let items = match &message {
            Message::User { content } => content.len(),
            _ => 0,
        };
        let origins = (0..items).map(|item| (item, origin.clone())).collect();
        Self {
            message,
            origins,
            id: None,
        }
    }
}

impl From<Message> for Entry {
    /// A new message without origins.
    fn from(message: Message) -> Self {
        Self {
            message,
            origins: Vec::new(),
            id: None,
        }
    }
}

/// What a splice changed: the ids no longer in the conversation, and
/// those of the messages it made or changed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Spliced {
    /// The ids no longer in the conversation.
    pub removed: Vec<MessageId>,
    /// The ids of the messages the splice made or changed.
    pub changed: Vec<MessageId>,
}

/// The text that goes between a request the user stopped and the next
/// message, which joins it, so the model does not carry the request out.
pub const STOPPED: &str = "[The request above was stopped by the user before it was answered. \
                           Do not carry it out unless asked again.]";

/// Why the user's last message is left without an answer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Halt {
    /// The user stopped the turn answering it.
    Stopped,
    /// It was left for the next turn, such as a note to an idle agent or
    /// a request whose turn failed: the next message joins it.
    #[default]
    Kept,
}

impl Conversation {
    /// The messages, oldest first.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Where content item `item` of message `message` came from, when it
    /// was delivered by an agent or a plugin, or is the user's with a title
    /// or a request id; `None` for the user's plain text, the model's
    /// messages and tool results.
    pub fn origin(&self, message: usize, item: usize) -> Option<&Origin> {
        self.origins
            .iter()
            .find(|(at, index, _)| *at == message && *index == item)
            .map(|(.., origin)| origin)
    }

    /// Why the user's last message is left unanswered, if it was halted:
    /// stopped by the user, or kept for the next turn. Cleared when a
    /// message is added or a splice reaches the end.
    pub fn halted(&self) -> Option<Halt> {
        self.halted
    }

    /// The id of message `at`.
    pub fn id(&self, at: usize) -> Option<MessageId> {
        self.ids.get(at).copied()
    }

    /// Where message `id` is now; `None` once no item of it is left.
    pub fn index_of(&self, id: MessageId) -> Option<usize> {
        self.ids.iter().position(|&known| known == id)
    }

    /// Where a span that began at message `id` begins now, such as a turn:
    /// its index while it is there; the end for a message not made yet; 0
    /// for one an edit took out, as a summary then stands for what came
    /// before it.
    pub fn start_of(&self, id: MessageId) -> usize {
        self.index_of(id).unwrap_or(if id.0 >= self.next_id {
            self.messages.len()
        } else {
            0
        })
    }

    /// Message `at`, with its origins and id, to splice back changed.
    pub fn entry(&self, at: usize) -> Option<Entry> {
        let message = self.messages.get(at)?.clone();
        let origins = self.origins.iter().filter(|(message, ..)| *message == at);
        Some(Entry {
            message,
            origins: origins
                .map(|(_, item, origin)| (*item, origin.clone()))
                .collect(),
            id: self.id(at),
        })
    }

    /// The messages, to load their images in place.
    pub(crate) fn each_message(&mut self) -> std::slice::IterMut<'_, Message> {
        self.messages.iter_mut()
    }

    /// The id the next new message gets.
    pub(crate) fn next_id(&self) -> u64 {
        self.next_id
    }

    /// Where a turn starting now begins: the user's last message, which it
    /// answers, else the first message it adds.
    pub fn turn_start(&self) -> MessageId {
        match (self.messages.last(), self.ids.last()) {
            (Some(Message::User { .. }), Some(&id)) => id,
            _ => MessageId(self.next_id),
        }
    }

    fn new_id(&mut self) -> MessageId {
        let id = MessageId(self.next_id);
        self.next_id = self.next_id.saturating_add(1);
        id
    }

    /// Whether the model owes an answer: the last message is the user's,
    /// and it was not halted.
    pub fn awaits_model(&self) -> bool {
        self.halted.is_none() && matches!(self.messages.last(), Some(Message::User { .. }))
    }

    /// Leaves the user's last message unanswered, for `reason`. Whether it
    /// was halted now: not when the model owes no answer.
    pub fn halt(&mut self, reason: Halt) -> bool {
        let halts = self.awaits_model();
        if halts {
            self.halted = Some(reason);
        }
        halts
    }

    /// Asks the model again for an answer to the user's last message, also
    /// a halted one; whether the last message is the user's.
    pub fn resume(&mut self) -> bool {
        self.halted = None;
        self.awaits_model()
    }

    /// Adds `message`, whose content came from `origin`: a user message
    /// goes into the last message when that is the user's too, such as the
    /// tool results the model waits for, so user and model keep taking
    /// turns. After [`STOPPED`] when that is a request the user stopped.
    pub fn append(&mut self, message: Message, origin: Option<Origin>) {
        let stopped = self.halted.take() == Some(Halt::Stopped);
        let (at, first) = match (self.messages.last(), &message) {
            (Some(Message::User { content }), Message::User { .. }) => (
                self.messages.len().saturating_sub(1),
                content.len() + usize::from(stopped),
            ),
            _ => (self.messages.len(), 0),
        };
        let items = match &message {
            Message::User { content } => content.len(),
            _ => 0,
        };
        if let Some(origin) = origin {
            self.origins
                .extend((first..first + items).map(|item| (at, item, origin.clone())));
        }
        match (self.messages.last_mut(), message) {
            (Some(Message::User { content }), Message::User { content: added }) => {
                if stopped {
                    content.push(UserContent::text(STOPPED));
                }
                content.extend(added);
            }
            (_, message) => {
                self.messages.push(message);
                let id = self.new_id();
                self.ids.push(id);
            }
        }
    }

    /// Replaces messages `from..to`, clamped to the conversation, with
    /// `with`. An entry keeps its id unless that id is in use outside the
    /// range or by an earlier entry; every other one gets a new id. A user
    /// message beside a user message is merged into it, at both seams and
    /// inside `with`, keeping the id that existed before the splice, the
    /// earlier one when both did. The halt goes when the splice reaches
    /// the end.
    pub fn splice(&mut self, from: usize, to: usize, with: Vec<Entry>) -> Spliced {
        let len = self.messages.len();
        let to = to.min(len);
        let from = from.min(to);
        if to == len {
            self.halted = None;
        }
        let existed = self.next_id;
        let old = |id: MessageId| id.0 < existed;
        // The replaced range and a neighbour on each side, which a user
        // message may merge with.
        let start = from.saturating_sub(1);
        let end = to.saturating_add(1).min(len);
        let outside = self.ids.iter().take(from).chain(self.ids.iter().skip(to));
        let mut used: HashSet<MessageId> = outside.copied().collect();
        let neighbour = |at: usize| {
            let entry = self.entry(at)?;
            Some((entry.id?, entry.message, entry.origins, false))
        };
        let before: Vec<_> = (start..from).filter_map(neighbour).collect();
        let after: Vec<_> = (to..end).filter_map(neighbour).collect();
        let mut window = before;
        for entry in with {
            let kept = entry.id.filter(|&id| old(id) && used.insert(id));
            let id = kept.unwrap_or_else(|| self.new_id());
            window.push((id, entry.message, entry.origins, true));
        }
        window.extend(after);
        let mut merged: Vec<(MessageId, Message, Vec<(usize, Origin)>, bool)> =
            Vec::with_capacity(window.len());
        for (id, message, origins, changed) in window {
            if let (
                Some((last_id, Message::User { content }, last_origins, last_changed)),
                Message::User { content: added },
            ) = (merged.last_mut(), &message)
            {
                let offset = content.len();
                let origins = origins.into_iter();
                last_origins.extend(origins.map(|(item, origin)| (item + offset, origin)));
                content.extend(added.iter().cloned());
                if !old(*last_id) && old(id) {
                    *last_id = id;
                }
                *last_changed = true;
                continue;
            }
            merged.push((id, message, origins, changed));
        }
        let kept: HashSet<MessageId> = merged.iter().map(|(id, ..)| *id).collect();
        let replaced = self.ids.get(start..end).unwrap_or_default();
        let removed = replaced.iter().filter(|id| !kept.contains(id)).copied();
        let removed = removed.collect();
        let changed = merged.iter().filter(|(.., changed)| *changed);
        let changed = changed.map(|(id, ..)| *id).collect();
        let grown = merged.len();
        self.origins.retain(|(at, ..)| *at < start || *at >= end);
        for (at, ..) in &mut self.origins {
            if *at >= end {
                *at = *at - (end - start) + grown;
            }
        }
        let mut messages = Vec::with_capacity(grown);
        let mut ids = Vec::with_capacity(grown);
        for (offset, (id, message, origins, _)) in merged.into_iter().enumerate() {
            let at = start + offset;
            let origins = origins.into_iter();
            self.origins
                .extend(origins.map(|(item, origin)| (at, item, origin)));
            messages.push(message);
            ids.push(id);
        }
        self.messages.splice(start..end, messages);
        self.ids.splice(start..end, ids);
        self.origins.sort_by_key(|&(at, item, _)| (at, item));
        Spliced { removed, changed }
    }
}

/// The id of a request to an agent, which the reply to it names, such as
/// the tool call that asked. Unique per sender, so it doubles as an
/// idempotency key.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "reflect", derive(Reflect))]
#[serde(transparent)]
pub struct RequestId(pub String);

/// Who sent a message.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "reflect", derive(Reflect))]
#[serde(rename_all = "snake_case")]
pub enum OriginKind {
    /// The user, typing.
    #[default]
    User,
    /// An agent, such as a subagent answering or an agent handing out a
    /// task.
    Agent,
    /// The plugin named.
    Plugin(String),
}

/// Where a message came from: who sent it, the id of the agent it came
/// from, and the request it answers or makes.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "reflect", derive(Reflect))]
pub struct Origin {
    /// Who sent it.
    #[serde(default)]
    pub kind: OriginKind,
    /// The id of the agent it came from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from: Option<String>,
    /// The request it answers or makes. Kept for correlation, such as a
    /// report to the call that asked; not shown in its header.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request: Option<RequestId>,
    /// What it is about, such as a subagent's task title, shown in its
    /// header.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

impl Origin {
    /// The user's own text.
    pub fn user() -> Self {
        Self::default()
    }

    /// Output of the agent whose id is `from`.
    pub fn agent(from: impl Into<String>, request: Option<RequestId>) -> Self {
        Self {
            kind: OriginKind::Agent,
            from: Some(from.into()),
            request,
            title: None,
        }
    }

    /// Output of the plugin `name`.
    pub fn plugin(name: impl Into<String>) -> Self {
        Self {
            kind: OriginKind::Plugin(name.into()),
            ..Self::default()
        }
    }

    /// This origin about `title`, such as a task's.
    pub fn titled(self, title: impl Into<String>) -> Self {
        Self {
            title: Some(title.into()),
            ..self
        }
    }

    /// Who sent it, for views: such as `agent 1a2b3c4d "Fix the parser"`.
    /// The request id is left out; it stays in [`Origin::request`].
    pub fn label(&self) -> String {
        let who = match &self.kind {
            OriginKind::User => "you".to_owned(),
            OriginKind::Agent => {
                format!("agent {}", self.from.as_deref().map_or("?", short_id))
            }
            OriginKind::Plugin(name) => format!("plugin {name}"),
        };
        match &self.title {
            Some(title) => format!("{who} \"{title}\""),
            None => who,
        }
    }

    /// The line that heads text of this origin for the model, or `None`
    /// for the user's own: it names the sender and says the text is not
    /// the user's words.
    pub fn header(&self) -> Option<String> {
        match self.kind {
            OriginKind::User => None,
            _ => Some(format!(
                "[Output of {}, not the user's words]",
                self.label()
            )),
        }
    }
}

/// The first characters of the agent id `id`, enough to tell agents apart.
pub fn short_id(id: &str) -> &str {
    id.get(..8).unwrap_or(id)
}
