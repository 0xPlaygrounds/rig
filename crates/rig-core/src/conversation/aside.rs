//! Records kept in a conversation that never reach a model.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use super::{Author, MessageId, ParseIdError, parse_id};

/// Kept with the conversation, saved and forked with it, never sent to a
/// model: a plugin's note in the transcript, the kernel's record of a call
/// in flight, a plugin's own data. Readers take the kinds they know.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Aside {
    /// What it is, chosen by its author: `"note"` (a transcript
    /// [`Note`](super::Note)), `"rig.call"` (the kernel's call-in-flight
    /// record), a plugin's own kind.
    pub kind: String,
    /// The entry it follows; `None` before the first. While an edit took
    /// that entry out it follows [`start_of`](super::Conversation::start_of)
    /// it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after: Option<MessageId>,
    /// Who put it in.
    pub author: Author,
    /// Its data, verbatim.
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub value: serde_json::Value,
    /// Whether it is saved with the conversation (the default). An unsaved
    /// one lives as long as the value: no change records it.
    #[serde(default = "yes", skip_serializing_if = "is_yes")]
    pub saved: bool,
}

fn yes() -> bool {
    true
}

fn is_yes(saved: &bool) -> bool {
    *saved
}

impl Aside {
    /// A saved aside of `kind` by `author`, holding `value`, before the
    /// first entry.
    pub fn new(kind: impl Into<String>, author: Author, value: serde_json::Value) -> Self {
        Self {
            kind: kind.into(),
            after: None,
            author,
            value,
            saved: true,
        }
    }

    /// This aside, following entry `after` (`None`: before the first).
    #[must_use]
    pub fn after(self, after: Option<MessageId>) -> Self {
        Self { after, ..self }
    }

    /// This aside, never saved.
    #[must_use]
    pub fn unsaved(self) -> Self {
        Self {
            saved: false,
            ..self
        }
    }
}

/// An aside's identity in its conversation, kept while it is replaced and
/// across a save. Rises in the order asides were put.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct AsideId(pub(super) u64);

impl AsideId {
    /// Its number: ids rise from 0 in the order asides were put.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Its plain form, `a` and its number (`a4`): how commands name an aside.
impl fmt::Display for AsideId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "a{}", self.0)
    }
}

/// Reads its plain form, `a4`, or the number alone.
impl FromStr for AsideId {
    type Err = ParseIdError;

    fn from_str(text: &str) -> Result<Self, ParseIdError> {
        parse_id(text, 'a').map(Self)
    }
}
