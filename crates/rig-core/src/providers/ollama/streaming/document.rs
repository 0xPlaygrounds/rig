//! The native `/api/chat` reassembler: a stream of records rebuilds the body
//! a unary call returns.

use serde_json::{Map, Value};

use super::{ChatDecoder, ChatRecord};
use crate::wire::document::Reassemble;
use crate::wire::{Decoder, WireEvent, WireFrame};

/// The `/api/chat` reply a stream of records adds up to.
///
/// Top-level fields take the last non-null value a record states, so the
/// final `done` record's reason, counts and durations stand. Within
/// `message`, `content` and `thinking` append, `tool_calls` and `images`
/// collect, and other fields take the last non-null value. Top-level
/// `logprobs` collect. A reply that failed in band keeps its `error`
/// beside what arrived before it.
#[derive(Debug, Default)]
pub struct ChatResponse {
    /// The decoder's classifier.
    classifier: ChatDecoder,
    document: Map<String, Value>,
}

/// The `message` fields whose fragments join as text.
const APPENDED: &[&str] = &["content", "thinking"];

/// The fields whose lists join, in `message` and at the top level.
const COLLECTED: &[&str] = &["tool_calls", "images", "logprobs"];

/// Fold `fields` into `into`: text in `appended` joins, lists in
/// [`COLLECTED`] extend, any other non-null value replaces, and `null`
/// only fills an absent key.
fn fold(into: &mut Map<String, Value>, fields: Map<String, Value>, appended: &[&str]) {
    for (key, value) in fields {
        match (into.get_mut(&key), value) {
            (Some(Value::String(text)), Value::String(more))
                if appended.contains(&key.as_str()) =>
            {
                text.push_str(&more);
            }
            (Some(Value::Array(items)), Value::Array(more))
                if COLLECTED.contains(&key.as_str()) =>
            {
                items.extend(more);
            }
            (Some(_), Value::Null) => {}
            (_, value) => {
                into.insert(key, value);
            }
        }
    }
}

impl Reassemble<WireFrame> for ChatResponse {
    fn absorb(&mut self, frame: &WireFrame) {
        // The driver reports what does not classify.
        let WireEvent::Known(ChatRecord(mut fields)) = self.classifier.classify(frame.clone())
        else {
            return;
        };
        if let Some(Value::Object(message)) = fields.shift_remove("message") {
            match self.document.get_mut("message") {
                Some(Value::Object(into)) => fold(into, message, APPENDED),
                _ => {
                    self.document
                        .insert("message".to_owned(), Value::Object(message));
                }
            }
        }
        fold(&mut self.document, fields, &[]);
    }

    fn finish(self) -> Value {
        if self.document.is_empty() {
            Value::Null
        } else {
            Value::Object(self.document)
        }
    }
}
