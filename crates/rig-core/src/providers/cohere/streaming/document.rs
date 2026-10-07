//! The native Chat reassembler: a stream of events rebuilds the chat
//! response a unary call returns.

use std::collections::BTreeMap;

use serde_json::{Map, Value};

use super::ChatDecoder;
use crate::json_utils::Lenient;
use crate::wire::document::Reassemble;
use crate::wire::{Decoder, WireEvent, WireFrame};

/// The native chat response a stream of events adds up to.
///
/// `message-start` gives `id` and the message's skeleton, less an empty
/// `tool_plan`, which a unary reply without a plan leaves out. `content-start`
/// places a content part at its index and `content-delta` appends to its
/// text fields. `tool-plan-delta` appends to `message.tool_plan`.
/// `tool-call-start` places a call at its index and `tool-call-delta`
/// appends to its `function.arguments`. `citation-start` places a citation
/// at its index in `message.citations`. A delta's `logprobs` join the
/// reply's. `message-end` gives `finish_reason`, `usage` and, on a failed
/// reply, `error`. A whole reply is its own document.
#[derive(Debug, Default)]
pub struct ChatResponse {
    /// The decoder's classifier.
    classifier: ChatDecoder,
    /// The top-level fields: `id`, then what `message-end` states.
    top: Map<String, Value>,
    /// The message fields `message-start` states.
    message: Map<String, Value>,
    content: BTreeMap<u64, Map<String, Value>>,
    tool_plan: Option<String>,
    tool_calls: BTreeMap<u64, Map<String, Value>>,
    citations: BTreeMap<u64, Value>,
    logprobs: Vec<Value>,
    /// A whole reply, which is the document as sent.
    whole: Option<Value>,
}

/// Fold `fields` into `into`: a string appends to a string, an object folds
/// into an object, any other non-null value replaces, and `null` only fills
/// an absent key.
fn append(into: &mut Map<String, Value>, fields: &Map<String, Value>) {
    for (key, value) in fields {
        match (into.get_mut(key), value) {
            (Some(Value::String(text)), Value::String(more)) => text.push_str(more),
            (Some(Value::Object(inner)), Value::Object(more)) => append(inner, more),
            (Some(_), Value::Null) => {}
            (_, value) => {
                into.insert(key.clone(), value.clone());
            }
        }
    }
}

/// The entries of `slots` in index order, as a list.
fn listed<T: Into<Value>>(slots: BTreeMap<u64, T>) -> Value {
    Value::Array(slots.into_values().map(Into::into).collect())
}

impl crate::wire::document::Serves<crate::operation::Completion> for ChatResponse {}

impl Reassemble<WireFrame> for ChatResponse {
    fn absorb(&mut self, frame: &WireFrame) {
        let WireEvent::Known(event) = self.classifier.classify(frame.clone()) else {
            return;
        };
        let fields = &event.fields;
        let index = fields.u64("index").unwrap_or(0);
        let stated = |key: &str| fields.at(&format!("/delta/message/{key}"));
        let object = |key: &str| stated(key).and_then(Value::as_object);
        match event.kind() {
            "" => self.whole = Some(fields.clone()),
            "message-start" => {
                if let Some(id) = fields.get("id") {
                    self.top.insert("id".to_owned(), id.clone());
                }
                if let Some(Value::Object(message)) = fields.at("/delta/message") {
                    let mut message = message.clone();
                    // The skeleton states an empty plan; a unary reply with
                    // no plan states none.
                    if let Some(Value::String(plan)) = message.shift_remove("tool_plan")
                        && !plan.is_empty()
                    {
                        self.tool_plan.get_or_insert_default().push_str(&plan);
                    }
                    append(&mut self.message, &message);
                }
            }
            "content-start" => {
                if let Some(part) = object("content") {
                    self.content.insert(index, part.clone());
                }
            }
            "content-delta" => {
                if let Some(delta) = object("content") {
                    append(self.content.entry(index).or_default(), delta);
                }
            }
            "tool-plan-delta" => {
                if let Some(plan) = stated("tool_plan").and_then(Value::as_str) {
                    self.tool_plan.get_or_insert_default().push_str(plan);
                }
            }
            "tool-call-start" => {
                if let Some(call) = object("tool_calls") {
                    self.tool_calls.insert(index, call.clone());
                }
            }
            "tool-call-delta" => {
                if let Some(delta) = object("tool_calls") {
                    append(self.tool_calls.entry(index).or_default(), delta);
                }
            }
            "citation-start" => {
                if let Some(citation) = stated("citations") {
                    self.citations.insert(index, citation.clone());
                }
            }
            "message-end" => {
                if let Some(Value::Object(end)) = fields.get("delta") {
                    append(&mut self.top, end);
                }
            }
            _ => {}
        }
        match fields.get("logprobs") {
            None | Some(Value::Null) => {}
            Some(Value::Array(logprobs)) => self.logprobs.extend(logprobs.iter().cloned()),
            Some(logprobs) => self.logprobs.push(logprobs.clone()),
        }
    }

    fn finish(self) -> Value {
        if let Some(whole) = self.whole {
            return whole;
        }
        let mut message = self.message;
        let lists = [
            ("content", listed(self.content)),
            ("tool_calls", listed(self.tool_calls)),
            ("citations", listed(self.citations)),
        ];
        for (key, list) in lists {
            if list.as_array().is_some_and(|list| !list.is_empty()) {
                message.insert(key.to_owned(), list);
            }
        }
        if let Some(plan) = self.tool_plan.filter(|plan| !plan.is_empty()) {
            message.insert("tool_plan".to_owned(), Value::String(plan));
        }
        let mut document = self.top;
        if !message.is_empty() {
            document.insert("message".to_owned(), Value::Object(message));
        }
        if !self.logprobs.is_empty() {
            document.insert("logprobs".to_owned(), Value::Array(self.logprobs));
        }
        if document.is_empty() {
            Value::Null
        } else {
            Value::Object(document)
        }
    }
}
