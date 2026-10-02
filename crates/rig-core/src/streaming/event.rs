//! The events of a completion stream. A completion decoder never builds one:
//! it writes through the reply's part handles
//! ([`Out`](crate::wire::Out)), and the writer emits each part's start, its
//! fragments and its end, in that order. A [`Part`] is a part's position in
//! the response's `choice`; only this crate constructs one, so code outside
//! the writer cannot build an event out of order. A recorded stream reads
//! back through [`Transcript::parse`], the one place a sequence is checked.
//!
//! ```
//! use rig_core::streaming::{PartKind, StreamEvent, Transcript};
//!
//! let transcript = Transcript::parse(serde_json::json!([
//!     {"item": "event", "value": {"event": "start", "part": 0, "kind": "text"}},
//!     {"item": "event", "value": {"event": "text", "part": 0, "text": "Hello"}},
//!     {"item": "event", "value": {"event": "end", "part": 0,
//!         "content": {"type": "text", "text": "Hello"}}},
//! ]))?;
//! assert!(matches!(
//!     transcript.events().next(),
//!     Some(StreamEvent::Start { kind: PartKind::Text, .. })
//! ));
//! # Ok::<(), rig_core::streaming::SequenceError>(())
//! ```

use serde::{Deserialize, Serialize};

use crate::message::AssistantContent;

use super::UnknownPayload;

/// A part's position in the response's `choice`. Only this crate constructs
/// one:
///
/// ```compile_fail,E0423
/// let part = rig_core::streaming::Part(0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct Part(u32);

impl Part {
    pub(crate) const fn new(index: u32) -> Self {
        Self(index)
    }

    /// The part's position in the response's `choice`.
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// What kind of part a [`StreamEvent::Start`] opened.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PartKind {
    /// Answer text.
    Text,
    /// Reasoning.
    Reasoning,
    /// A tool call.
    ToolCall,
    /// An image.
    Image,
    /// A provider item with no canonical meaning.
    Opaque,
}

/// One event of a completion stream: a part starts, grows, or ends with the
/// content it finalized.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum StreamEvent {
    /// A part opened.
    Start {
        /// The part.
        part: Part,
        /// What it holds.
        kind: PartKind,
    },
    /// A text part grew.
    Text {
        /// The part.
        part: Part,
        /// The fragment.
        text: String,
    },
    /// A reasoning part grew.
    Reasoning {
        /// The part.
        part: Part,
        /// The fragment.
        text: String,
    },
    /// A tool call's arguments, as the provider sent them.
    Arguments {
        /// The part.
        part: Part,
        /// The raw JSON arguments.
        json: String,
    },
    /// A part ended with the content it finalized.
    End {
        /// The part.
        part: Part,
        /// The finalized content.
        content: AssistantContent,
    },
}

impl StreamEvent {
    /// The part this event is about.
    pub const fn part(&self) -> Part {
        match self {
            Self::Start { part, .. }
            | Self::Text { part, .. }
            | Self::Reasoning { part, .. }
            | Self::Arguments { part, .. }
            | Self::End { part, .. } => *part,
        }
    }

    /// A stable variant name that exposes no payload to logs.
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Start { .. } => "Start",
            Self::Text { .. } => "Text",
            Self::Reasoning { .. } => "Reasoning",
            Self::Arguments { .. } => "Arguments",
            Self::End { .. } => "End",
        }
    }
}

/// One item of a stream: an event of the operation, or a payload the
/// provider sent that the decoder does not model. An unmodeled payload
/// always reaches the consumer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "item", content = "value", rename_all = "snake_case")]
pub enum Item<E> {
    /// An event of the operation.
    Event(E),
    /// A payload the decoder does not model.
    Unknown(UnknownPayload),
}

/// A completion stream's items read back from their serialized form, in an
/// order the writer could have produced: each part starts once, at a
/// position no part took before, grows only while open and with its own
/// kind of fragment, and ends once. Positions follow the order provider
/// items opened, so a dropped item leaves a gap. Unmodeled payloads may come
/// anywhere.
#[derive(Debug, Clone, Default)]
pub struct Transcript {
    items: Vec<Item<StreamEvent>>,
    /// Each started part's kind, and whether it ended.
    parts: std::collections::BTreeMap<usize, (PartKind, bool)>,
}

impl PartialEq for Transcript {
    fn eq(&self, other: &Self) -> bool {
        self.items == other.items
    }
}

impl Serialize for Transcript {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.items.serialize(serializer)
    }
}

/// Why a serialized event sequence is not one the writer could produce.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SequenceError {
    /// The value is not a list of stream items.
    #[error("not a list of stream items: {0}")]
    Shape(String),
    /// An event names a part that has not started, or a start reuses a
    /// position.
    #[error("item {0} names a part that has not started")]
    UnknownPart(usize),
    /// A part ended twice, or grew after its end.
    #[error("item {0} touches a part that already ended")]
    EndedTwice(usize),
    /// A fragment or end does not match the part's kind.
    #[error("item {0} does not match its part's kind")]
    WrongKind(usize),
    /// A part started and never ended.
    #[error("part {0} never ended")]
    Unclosed(usize),
}

/// The serialized form of one event, before its order is checked.
#[derive(Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
enum RawEvent {
    Start {
        part: u32,
        kind: PartKind,
    },
    Text {
        part: u32,
        text: String,
    },
    Reasoning {
        part: u32,
        text: String,
    },
    Arguments {
        part: u32,
        json: String,
    },
    End {
        part: u32,
        content: AssistantContent,
    },
}

impl Transcript {
    /// Read a serialized item sequence, refusing one the writer could not
    /// have produced. Every part must have ended.
    pub fn parse(value: serde_json::Value) -> Result<Self, SequenceError> {
        let transcript = Self::parse_prefix(value)?;
        transcript.check_closed()?;
        Ok(transcript)
    }

    /// [`Self::parse`] for a stream that stopped early (an error or a
    /// cancellation): parts may still be open at its end.
    pub fn parse_prefix(value: serde_json::Value) -> Result<Self, SequenceError> {
        let raw: Vec<Item<RawEvent>> = serde_json::from_value(value)
            .map_err(|error| SequenceError::Shape(error.to_string()))?;
        let mut transcript = Self::default();
        for item in raw {
            transcript.push(match item {
                Item::Unknown(payload) => Item::Unknown(payload),
                Item::Event(RawEvent::Start { part, kind }) => Item::Event(StreamEvent::Start {
                    part: Part(part),
                    kind,
                }),
                Item::Event(RawEvent::Text { part, text }) => Item::Event(StreamEvent::Text {
                    part: Part(part),
                    text,
                }),
                Item::Event(RawEvent::Reasoning { part, text }) => {
                    Item::Event(StreamEvent::Reasoning {
                        part: Part(part),
                        text,
                    })
                }
                Item::Event(RawEvent::Arguments { part, json }) => {
                    Item::Event(StreamEvent::Arguments {
                        part: Part(part),
                        json,
                    })
                }
                Item::Event(RawEvent::End { part, content }) => Item::Event(StreamEvent::End {
                    part: Part(part),
                    content,
                }),
            })?;
        }
        Ok(transcript)
    }

    /// Append the next item a stream yielded, refusing one the writer could
    /// not have produced after the items so far.
    pub fn push(&mut self, item: Item<StreamEvent>) -> Result<(), SequenceError> {
        let position = self.items.len();
        if let Item::Event(event) = &item {
            let index = event.part().index();
            let kind = match event {
                StreamEvent::Start { kind, .. } => {
                    if self.parts.insert(index, (*kind, false)).is_some() {
                        return Err(SequenceError::UnknownPart(position));
                    }
                    None
                }
                StreamEvent::Text { .. } => Some(PartKind::Text),
                StreamEvent::Reasoning { .. } => Some(PartKind::Reasoning),
                StreamEvent::Arguments { .. } => Some(PartKind::ToolCall),
                StreamEvent::End { content, .. } => Some(match content {
                    AssistantContent::Text(_) => PartKind::Text,
                    AssistantContent::Reasoning(_) => PartKind::Reasoning,
                    AssistantContent::ToolCall(_) => PartKind::ToolCall,
                    AssistantContent::Image(_) => PartKind::Image,
                    AssistantContent::Opaque(_) => PartKind::Opaque,
                }),
            };
            if let Some(kind) = kind {
                match self.parts.get_mut(&index) {
                    None => return Err(SequenceError::UnknownPart(position)),
                    Some((_, true)) => return Err(SequenceError::EndedTwice(position)),
                    Some((open, false)) if *open != kind => {
                        return Err(SequenceError::WrongKind(position));
                    }
                    Some((_, ended)) => *ended = matches!(event, StreamEvent::End { .. }),
                }
            }
        }
        self.items.push(item);
        Ok(())
    }

    fn check_closed(&self) -> Result<(), SequenceError> {
        match self.parts.iter().find(|(_, (_, ended))| !ended) {
            Some((part, _)) => Err(SequenceError::Unclosed(*part)),
            None => Ok(()),
        }
    }

    /// Items a stream yielded, checked as [`Self::push`] checks them: a
    /// stream that stopped early may leave parts open.
    pub fn from_items(items: Vec<Item<StreamEvent>>) -> Result<Self, SequenceError> {
        let mut transcript = Self::default();
        for item in items {
            transcript.push(item)?;
        }
        Ok(transcript)
    }

    /// The number of items.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether there are no items.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// The items, in order.
    pub fn items(&self) -> &[Item<StreamEvent>] {
        &self.items
    }

    /// The items, in order.
    pub fn into_items(self) -> Vec<Item<StreamEvent>> {
        self.items
    }

    /// The events, in order, without the unmodeled payloads.
    pub fn events(&self) -> impl Iterator<Item = &StreamEvent> {
        self.items.iter().filter_map(|item| match item {
            Item::Event(event) => Some(event),
            Item::Unknown(_) => None,
        })
    }
}

impl From<Transcript> for Vec<Item<StreamEvent>> {
    fn from(transcript: Transcript) -> Self {
        transcript.items
    }
}

impl<'de> Deserialize<'de> for Transcript {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(deserializer)?;
        Self::parse_prefix(value).map_err(serde::de::Error::custom)
    }
}

// The stream vocabulary crosses threads on every target: the bus sends it
// over a channel and the effect log records it.
const _: fn() = || {
    fn assert_wire<T: Clone + Send + Sync + 'static + Serialize>() {}
    assert_wire::<StreamEvent>();
    assert_wire::<Transcript>();
    assert_wire::<Item<StreamEvent>>();
};

#[cfg(test)]
mod tests;
