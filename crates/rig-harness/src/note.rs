//! A plugin's note in an agent's transcript: an [`Aside`] of kind
//! [`Note::KIND`] in the agent's conversation, which every front that shows
//! the agent draws after the entry it follows. It never reaches a model,
//! and it is saved, restored and forked with the conversation; one made
//! [`unsaved`](Aside::unsaved) lives only as long as the app.
//!
//! ```
//! use rig_harness::note::Note;
//! use rig_harness::prelude::*;
//!
//! /// Shows a side question's answer after the agent's newest message.
//! fn answered(agent: Entity, mut conversations: Query<&mut Conversation>) {
//!     let Ok(mut conversation) = conversations.get_mut(agent) else {
//!         return;
//!     };
//!     let last = conversation.messages().len().checked_sub(1);
//!     let after = last.and_then(|last| conversation.id(last));
//!     let note = Note::new("Why tabs?", "The project's style says so.");
//!     conversation.add_aside(note.aside(Author::plugin("btw"), after));
//! }
//! ```
//!
//! A front draws a note with its author's look; the terminal view lets a
//! plugin choose that look (rig-tui's `add_entry_renderer`).

use rig_core::conversation::{Aside, Author, MessageId};
use serde::{Deserialize, Serialize};

/// A note in an agent's transcript: what it shows collapsed, in a line,
/// and what it shows expanded.
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

#[cfg(test)]
mod tests;
