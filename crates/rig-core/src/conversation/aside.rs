//! Records kept in a conversation that never reach a model.

#[cfg(feature = "bevy")]
use bevy_reflect::prelude::{Reflect, ReflectDefault};
use serde::{Deserialize, Serialize};

use super::{Author, MessageId};

/// Kept with the conversation, saved and forked with it, never sent to a
/// model: a plugin's note in the transcript, the kernel's record of a call
/// in flight, a hook's own data. Readers take the kinds they know.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Aside {
    /// What it is, chosen by its author: `"note"` (the harness's transcript
    /// note), `"rig.call"` (the kernel's call-in-flight record), a hook's
    /// own kind.
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
#[cfg_attr(feature = "bevy", derive(Reflect))]
#[cfg_attr(feature = "bevy", reflect(Clone, Debug, Default, PartialEq, Hash))]
#[serde(transparent)]
pub struct AsideId(pub(super) u64);
