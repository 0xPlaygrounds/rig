//! The assistant message a message-shaped wire (Chat Completions, Ollama,
//! Cohere) sends for a turn, rebuilt from the turn's blocks as pi's
//! `openai-completions` rebuilds it. The provider's whole message is never
//! sent: each block gives what its own item holds, so a field only a reply
//! carries (generated images, annotations, an answer's audio data) cannot
//! reach a request.

use serde_json::{Map, Value};

use crate::message::{AssistantContent, AssistantMessage, ToolCall};

/// One block's share of the message, in block order.
pub(crate) enum Piece {
    /// Answer text, with its content part while the block holds one.
    Text { text: String, part: Option<Value> },
    /// Reasoning text, under the field it arrived in, or as its content part.
    Reasoning {
        text: String,
        field: Option<String>,
        part: Option<Value>,
    },
    /// A provider item with no canonical meaning: a content part when it has
    /// a `type`, else fields of the message.
    Opaque(Value),
}

/// A turn as the pieces its message is built from.
pub(crate) struct Rebuilt {
    /// Text, reasoning and opaque items, in block order.
    pub(crate) pieces: Vec<Piece>,
    /// Fields block items hold beside their text, verbatim
    /// (`reasoning_details`).
    pub(crate) fields: Map<String, Value>,
    /// Each call, with its item while it is current.
    pub(crate) calls: Vec<(ToolCall, Option<Map<String, Value>>)>,
}

impl Rebuilt {
    /// The pieces of `turn`. Images are left out: no message-shaped wire
    /// takes them back in an assistant message.
    #[deny(clippy::wildcard_enum_match_arm)]
    pub(crate) fn of(turn: &AssistantMessage) -> Self {
        let mut rebuilt = Self {
            pieces: Vec::new(),
            fields: Map::new(),
            calls: Vec::new(),
        };
        for block in &turn.content {
            let item = block.native_item().cloned();
            let part = item.clone().filter(|item| item.get("type").is_some());
            match block {
                AssistantContent::Text(text) => rebuilt.pieces.push(Piece::Text {
                    text: text.text.clone(),
                    part,
                }),
                AssistantContent::Reasoning(reasoning) => {
                    let mut field = None;
                    if let (None, Some(Value::Object(item))) = (&part, item) {
                        for (key, value) in item {
                            match value {
                                Value::String(_) => {
                                    field.get_or_insert(key);
                                }
                                value => {
                                    crate::providers::openai::wire::dto::merge_fields(
                                        &mut rebuilt.fields,
                                        &Map::from_iter([(key, value)]),
                                    );
                                }
                            }
                        }
                    }
                    rebuilt.pieces.push(Piece::Reasoning {
                        text: reasoning.text.clone(),
                        field,
                        part,
                    });
                }
                AssistantContent::ToolCall(call) => {
                    let item = match item {
                        Some(Value::Object(item)) => Some(item),
                        _ => None,
                    };
                    rebuilt.calls.push((call.clone(), item));
                }
                AssistantContent::Opaque(opaque) => {
                    rebuilt.pieces.push(Piece::Opaque(opaque.item.clone()));
                }
                AssistantContent::Image(_) => {}
            }
        }
        rebuilt
    }

    /// The text, joined with nothing between blocks; blank text is left out.
    pub(crate) fn text(&self) -> String {
        self.pieces
            .iter()
            .filter_map(|piece| match piece {
                Piece::Text { text, .. } if !text.trim().is_empty() => Some(text.as_str()),
                Piece::Text { .. } | Piece::Reasoning { .. } | Piece::Opaque(_) => None,
            })
            .collect()
    }

    /// The reasoning that is not a content part, joined with `"\n"` per
    /// field in the order the fields first appear. Reasoning with no field
    /// goes under `default`, or nowhere without one; blank reasoning is left
    /// out.
    pub(crate) fn reasoning(&self, default: Option<&str>) -> Vec<(String, String)> {
        let mut fields: Vec<(String, String)> = Vec::new();
        for piece in &self.pieces {
            let Piece::Reasoning {
                text,
                field,
                part: None,
            } = piece
            else {
                continue;
            };
            let Some(field) = field.as_deref().or(default) else {
                continue;
            };
            if text.trim().is_empty() {
                continue;
            }
            match fields.iter_mut().find(|(name, _)| name == field) {
                Some((_, joined)) => {
                    joined.push('\n');
                    joined.push_str(text);
                }
                None => fields.push((field.to_owned(), text.clone())),
            }
        }
        fields
    }

    /// Whether a block is a content part, so the message's content is a
    /// part array.
    pub(crate) fn has_parts(&self) -> bool {
        self.pieces.iter().any(|piece| match piece {
            Piece::Text { part, .. } | Piece::Reasoning { part, .. } => part.is_some(),
            Piece::Opaque(item) => item.get("type").is_some(),
        })
    }
}

/// `call` as the item a wire sends: its current item, or a function call
/// built from its canonical fields, with `id`, the canonical name and the
/// canonical arguments, as JSON text when `text`. A `custom` call takes its
/// arguments' `input` back as its input.
pub(crate) fn call_item(
    call: &ToolCall,
    item: Option<Map<String, Value>>,
    id: Option<String>,
    text: bool,
) -> Value {
    let mut item = item.unwrap_or_else(|| Map::from_iter([("type".to_owned(), "function".into())]));
    match id {
        Some(id) => item.insert("id".to_owned(), id.into()),
        None => item.shift_remove("id"),
    };
    let custom = item.get("type").and_then(Value::as_str) == Some("custom");
    let (key, value) = if custom {
        let input = match call.function.arguments.get("input") {
            Some(Value::String(input)) => input.clone(),
            Some(input) => input.to_string(),
            None => String::new(),
        };
        ("input", Value::String(input))
    } else {
        let arguments = Value::Object(call.function.arguments.clone());
        let arguments = if text {
            Value::String(arguments.to_string())
        } else {
            arguments
        };
        ("arguments", arguments)
    };
    let slot = item
        .entry(if custom { "custom" } else { "function" })
        .or_insert_with(|| Value::Object(Map::new()));
    if !slot.is_object() {
        *slot = Value::Object(Map::new());
    }
    if let Value::Object(fields) = slot {
        fields.insert("name".to_owned(), call.function.name.to_string().into());
        fields.insert(key.to_owned(), value);
    }
    Value::Object(item)
}

#[cfg(test)]
mod tests;
