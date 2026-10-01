//! Provider output items kept verbatim beside, or instead of, their
//! canonical form. A [`NativeItem`] is one item exactly as the wire
//! dialect that produced it stated it. It replays only to that dialect, and
//! only to a service its [`Sealed`] issuer opens for.
//!
//! An item with no canonical meaning is its own
//! [`AssistantContent::Native`](super::AssistantContent::Native) block. An
//! item whose canonical block loses something rides that block as its
//! `native` residue, and replays in its place while the block is unedited.
//!
//! ```
//! use rig_core::message::{Issuer, NativeItem, Sealed};
//!
//! let item = NativeItem::new("anthropic.messages", serde_json::json!({"type": "compaction"}));
//! let sealed = Sealed::new(Issuer::from("anthropic"), item);
//! let issuers = [Issuer::from("anthropic")];
//! assert!(sealed.open_native("anthropic.messages", &issuers).is_some());
//! assert!(sealed.open_native("openai.responses", &issuers).is_none());
//! ```

use std::borrow::Cow;

use serde::{Deserialize, Serialize};

use super::{AssistantContent, Image, Issuer, Reasoning, Sealed, Text, ToolCall};

/// One provider output item, verbatim, tagged with the wire dialect whose
/// encoder understands it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NativeItem {
    /// The wire dialect that produced the item, such as `"anthropic.messages"`.
    dialect: Cow<'static, str>,
    /// The item as the provider stated it.
    item: serde_json::Value,
}

impl NativeItem {
    /// `item`, as `dialect` stated it.
    pub fn new(dialect: impl Into<Cow<'static, str>>, item: serde_json::Value) -> Self {
        Self {
            dialect: dialect.into(),
            item,
        }
    }

    /// The dialect that produced the item.
    pub fn dialect(&self) -> &str {
        &self.dialect
    }

    /// The item as the provider stated it.
    pub fn item(&self) -> &serde_json::Value {
        &self.item
    }

    /// The item's `type` tag, when it has one.
    pub fn kind(&self) -> Option<&str> {
        self.item.get("type").and_then(serde_json::Value::as_str)
    }

    /// The item decoded as the dialect's own wire type.
    ///
    /// # Errors
    /// Returns the deserialization error when the item is not a `T`.
    pub fn decode<T: serde::de::DeserializeOwned>(&self) -> Result<T, serde_json::Error> {
        T::deserialize(&self.item)
    }
}

impl NativeItem {
    /// Whether two blocks say the same thing canonically: equal once the
    /// provider items they carry and the issuers of their seals are set
    /// aside. A dialect replays a block's item only while the item still
    /// projects to the block.
    pub fn same_canonical(left: &AssistantContent, right: &AssistantContent) -> bool {
        canonical(left) == canonical(right)
    }
}

/// `block` without the provider items it carries, its reasoning unsealed.
fn canonical(block: &AssistantContent) -> CanonicalBlock {
    match block {
        AssistantContent::Text(text) => CanonicalBlock::Text(Text {
            native: None,
            ..text.clone()
        }),
        AssistantContent::ToolCall(call) => CanonicalBlock::ToolCall(ToolCall {
            native: None,
            ..call.clone()
        }),
        AssistantContent::Reasoning(reasoning) => CanonicalBlock::Reasoning(Reasoning {
            native: None,
            ..reasoning.value().clone()
        }),
        AssistantContent::Image(image) => CanonicalBlock::Image(image.clone()),
        AssistantContent::Native(native) => CanonicalBlock::Native(native.value().clone()),
    }
}

#[derive(PartialEq)]
enum CanonicalBlock {
    Text(Text),
    ToolCall(ToolCall),
    Reasoning(Reasoning),
    Image(Image),
    Native(NativeItem),
}

impl Sealed<NativeItem> {
    /// The item, when it is `dialect`'s and one of `issuers` opens it.
    pub fn open_native(&self, dialect: &str, issuers: &[Issuer]) -> Option<&NativeItem> {
        self.open_for(issuers)
            .filter(|native| native.dialect() == dialect)
    }
}

/// `item` as its dialect replays it. A member that states nothing (null or
/// empty) or that `is_default` calls the dialect's documented default is
/// omitted unless `canonical`, the block's canonical encoding, states that
/// member too. Without a canonical encoding only defaults are omitted, since
/// an unmodelled item may require an empty member. Objects are compared
/// member by member, arrays element by element.
pub fn replay_form(
    item: &serde_json::Value,
    canonical: Option<&serde_json::Value>,
    is_default: fn(&str, &serde_json::Value) -> bool,
) -> serde_json::Value {
    use serde_json::Value;

    fn empty(value: &Value) -> bool {
        match value {
            Value::Null => true,
            Value::String(text) => text.is_empty(),
            Value::Array(items) => items.is_empty(),
            Value::Object(map) => map.is_empty(),
            Value::Bool(_) | Value::Number(_) => false,
        }
    }

    match item {
        Value::Object(members) => Value::Object(
            members
                .iter()
                .filter_map(|(key, value)| {
                    let stated = canonical.and_then(|canonical| canonical.get(key));
                    let omit = stated.is_none()
                        && (is_default(key, value) || (canonical.is_some() && empty(value)));
                    (!omit).then(|| (key.clone(), replay_form(value, stated, is_default)))
                })
                .collect(),
        ),
        Value::Array(items) => Value::Array(
            items
                .iter()
                .enumerate()
                .map(|(at, value)| {
                    let stated = canonical.and_then(|canonical| canonical.get(at));
                    match stated {
                        Some(stated) => replay_form(value, Some(stated), is_default),
                        None => value.clone(),
                    }
                })
                .collect(),
        ),
        other => other.clone(),
    }
}

/// Whether two wire values state the same thing. Null members, empty
/// strings, empty arrays and empty objects count as absent, because wires
/// omit them and restate them interchangeably.
pub fn same_wire_value(left: &serde_json::Value, right: &serde_json::Value) -> bool {
    use serde_json::Value;

    fn absent(value: &Value) -> bool {
        match value {
            Value::Null => true,
            Value::String(text) => text.is_empty(),
            Value::Array(items) => items.is_empty(),
            Value::Object(map) => map.values().all(absent),
            Value::Bool(_) | Value::Number(_) => false,
        }
    }

    match (left, right) {
        (Value::Object(left), Value::Object(right)) => {
            let keys = left.keys().chain(right.keys());
            keys.into_iter()
                .all(|key| match (left.get(key), right.get(key)) {
                    (Some(left), Some(right)) => same_wire_value(left, right),
                    (Some(only), None) | (None, Some(only)) => absent(only),
                    (None, None) => true,
                })
        }
        (Value::Array(left), Value::Array(right)) => {
            left.len() == right.len()
                && left
                    .iter()
                    .zip(right)
                    .all(|(left, right)| same_wire_value(left, right))
        }
        (left, right) => left == right || (absent(left) && absent(right)),
    }
}

#[cfg(test)]
mod tests;
