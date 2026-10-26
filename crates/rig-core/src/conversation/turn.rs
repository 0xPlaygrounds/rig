//! Turn records: where each turn of a conversation began and how it ended,
//! kept as asides, so a saved session tells a stopped or failed turn from
//! an answered one without the app that ran it.

use serde::{Deserialize, Serialize};

use super::{Aside, AsideId, Author, Conversation, MessageId};
use crate::message::CallId;

/// The kind of an aside holding a [`TurnRecord`].
pub const TURN: &str = "rig.turn";

/// One turn of a conversation, from the message that asked for it to how
/// it ended. Kept as an aside of kind [`TURN`] by [`Author::rig`], put
/// after the last entry when the turn begins and replaced when it ends.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnRecord {
    /// How many turns began before this one.
    pub index: u64,
    /// Where it begins ([`Conversation::turn_start`]):
    /// [`asked`](Conversation::asked) is what it answers,
    /// [`request`](Conversation::request) the message that asked for it.
    pub start: MessageId,
    /// How it ended; `None` while it runs, and when a crash or an exit cut
    /// it short.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end: Option<TurnEnd>,
}

/// How a turn ended.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "how", rename_all = "snake_case")]
pub enum TurnEnd {
    /// The model answered.
    Answered {
        /// The reply entry the turn ended on; `None` when the last entry was
        /// no reply.
        #[serde(default, rename = "by", skip_serializing_if = "Option::is_none")]
        answer: Option<MessageId>,
    },
    /// The turn failed, for `why`.
    Failed {
        /// Why.
        why: String,
    },
    /// The turn was stopped before an answer: by the user when `why` is
    /// `None`.
    Stopped {
        /// Why, when it was not the user.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        why: Option<String>,
        /// The last entry when a stop cut the turn off; `None` when the turn
        /// ended as stopped in its own course, such as a compaction ending it.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        at: Option<MessageId>,
        /// The tool calls the stop answered: those that had no result of
        /// their own.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        calls: Vec<CallId>,
    },
}

impl TurnRecord {
    /// The record `aside` holds, when it is one: of kind [`TURN`], its
    /// value a record.
    pub fn of(aside: &Aside) -> Option<Self> {
        (aside.kind == TURN)
            .then(|| Self::deserialize(&aside.value).ok())
            .flatten()
    }
}

impl Conversation {
    /// The turn records, oldest first, with their asides' ids; the newest
    /// is `turns().next_back()`, found from the end.
    pub fn turns(&self) -> impl DoubleEndedIterator<Item = (AsideId, TurnRecord)> + '_ {
        self.asides()
            .filter_map(|(id, aside)| Some((id, TurnRecord::of(aside)?)))
    }

    /// Records a turn that begins now, numbered after the last record and
    /// starting at [`turn_start`](Self::turn_start). Returns the new record.
    pub fn open_turn(&mut self) -> TurnRecord {
        let last = self.turns().next_back().map(|(_, record)| record.index);
        let record = TurnRecord {
            index: last.map_or(0, |index| index.saturating_add(1)),
            start: self.turn_start(),
            end: None,
        };
        let after = self.entries().next_back().map(|entry| entry.id);
        let value = serde_json::to_value(&record).unwrap_or_default();
        self.add_aside(Aside::new(TURN, Author::rig(), value).after(after));
        record
    }

    /// Ends the last turn record, when it is open, with `end`. Returns
    /// whether it was open.
    pub fn end_turn(&mut self, end: TurnEnd) -> bool {
        let Some((id, record)) = self.turns().next_back() else {
            return false;
        };
        let Some(aside) = self.aside(id).filter(|_| record.end.is_none()).cloned() else {
            return false;
        };
        let ended = TurnRecord {
            end: Some(end),
            ..record
        };
        let value = serde_json::to_value(ended).unwrap_or_default();
        self.replace_aside(id, Aside { value, ..aside })
    }
}
