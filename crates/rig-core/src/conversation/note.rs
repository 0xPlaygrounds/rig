//! Notes in a transcript ([`Note`]).

use serde::{Deserialize, Serialize};

use super::{Aside, Author, MessageId};

/// A note in a transcript: an [`Aside`] of kind [`Note::KIND`] that rig's
/// terminal view draws after the entry it follows, with what it shows
/// collapsed, in a line, and what it shows expanded. It never
/// reaches a model, and it is saved, restored and forked with the
/// conversation; one made [`unsaved`](Aside::unsaved) is not saved.
///
/// ```
/// use rig_core::conversation::{Author, Conversation, Note};
/// use rig_core::message::Message;
///
/// let mut conversation = Conversation::new();
/// conversation.push(Message::user("Why tabs?"));
/// let after = conversation.id(0);
/// let note = Note::new("Why tabs?", "The project's style says so.");
/// conversation.add_aside(note.aside(Author::plugin("btw"), after));
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Note {
    /// What it shows collapsed, in a line.
    pub title: String,
    /// What it shows expanded.
    pub text: String,
}

impl Note {
    /// The kind of the asides that hold a note.
    pub const KIND: &'static str = "note";

    /// A note titled `title`, showing `text` expanded.
    pub fn new(title: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            text: text.into(),
        }
    }

    /// The saved aside by `author` that holds this note, following the
    /// entry `after` (`None`: before the first).
    pub fn aside(self, author: Author, after: Option<MessageId>) -> Aside {
        let value = serde_json::json!({ "title": self.title, "text": self.text });
        Aside::new(Self::KIND, author, value).after(after)
    }

    /// The note `aside` holds, when it holds one.
    pub fn of(aside: &Aside) -> Option<Self> {
        if aside.kind != Self::KIND {
            return None;
        }
        Self::deserialize(&aside.value).ok()
    }
}
