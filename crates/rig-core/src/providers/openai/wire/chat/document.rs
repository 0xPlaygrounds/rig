//! The `chat.completion` document a stream of `chat.completion.chunk`s adds
//! up to, for every Chat dialect.
//!
//! Each choice's `delta` folds into its `message`: text fields append, tool
//! calls merge by `index` with their `arguments` appended, `reasoning_details`
//! merge as the decoder merges them, and list fields append. Every other
//! field keeps its last non-null value, and a field that only ever arrived
//! as `null` stays `null`, as the unary body states it.

use serde_json::{Map, Value};

use super::{ChatDecoder, ChatEvent, Quirks, REASONING_DETAILS, merge_details};
use crate::wire::document::Reassemble;
use crate::wire::{Decoder, WireEvent, WireFrame};

/// The unary document's tag.
const OBJECT: &str = "chat.completion";
/// Top-level chunk fields that pad a stream and are no part of a reply.
const STREAM_PADDING: [&str; 2] = ["obfuscation", "p"];
/// Message fields whose streamed fragments append.
const APPENDED_TEXT: [&str; 4] = ["content", "refusal", "reasoning", "reasoning_content"];
/// Message fields whose streamed entries append.
const APPENDED_LISTS: [&str; 2] = ["annotations", "images"];

/// A `chat.completion` being rebuilt from the frames of one reply.
///
/// Chunks fold into the document, and a whole `chat.completion` frame
/// replaces its choices' messages. The `[DONE]` sentinel, frames that do not
/// classify and the in-band error envelope add nothing. A bare-string reply
/// has no document, so it finishes `null`, as does a reply no frame added to.
#[derive(Default)]
pub struct ChatCompletion {
    /// The decoder's classifier for the dialect, so a frame counts exactly
    /// when the decoder reads it.
    classifier: ChatDecoder,
    /// Top-level fields, in arrival order. `choices` holds its place and is
    /// filled at the finish.
    envelope: Map<String, Value>,
    /// Each choice in arrival order.
    choices: Vec<Choice>,
    /// Whether a bare string answered.
    bare: bool,
}

/// One choice being rebuilt.
struct Choice {
    /// Its `index` in the reply.
    index: u64,
    fields: Map<String, Value>,
    /// Whether its last frame carried the whole message, so the choice is
    /// kept as sent.
    whole: bool,
}

impl ChatCompletion {
    /// A reassembler for a dialect with `quirks`.
    pub(super) fn new(quirks: Quirks) -> Self {
        Self {
            classifier: ChatDecoder::new(quirks),
            ..Self::default()
        }
    }

    /// Fold one `chat.completion` or `chat.completion.chunk` object.
    fn frame(&mut self, frame: &Value) {
        for (key, value) in frame.as_object().into_iter().flatten() {
            match key.as_str() {
                "choices" => {
                    self.envelope
                        .entry(key.clone())
                        .or_insert(Value::Array(Vec::new()));
                    for choice in value.as_array().into_iter().flatten() {
                        self.choice(choice);
                    }
                }
                // Per-chunk padding against size side channels (OpenAI's
                // `obfuscation`, Mistral's `p`); a unary body has none.
                key if STREAM_PADDING.contains(&key) => {}
                "object" if value.is_string() => {
                    self.envelope.insert(key.clone(), Value::from(OBJECT));
                }
                _ => keep(&mut self.envelope, key, value),
            }
        }
    }

    /// One choice of a frame: its `delta` into the message, or its whole
    /// `message` in place of it, and its other fields kept.
    fn choice(&mut self, chunk: &Value) {
        let index = chunk.get("index").and_then(Value::as_u64).unwrap_or(0);
        let position = self.choices.iter().position(|choice| choice.index == index);
        let choice = match position.and_then(|position| self.choices.get_mut(position)) {
            Some(choice) => choice,
            None => {
                self.choices.push(Choice {
                    index,
                    fields: Map::new(),
                    whole: false,
                });
                match self.choices.last_mut() {
                    Some(choice) => choice,
                    None => return,
                }
            }
        };
        // A choice that carries its whole message (a whole reply, or a
        // dialect that streams the message so far beside each delta) is
        // kept as sent.
        choice.whole = chunk.get("message").is_some();
        let fields = &mut choice.fields;
        for (key, value) in chunk.as_object().into_iter().flatten() {
            match (key.as_str(), value) {
                ("message", _) => {
                    fields.insert(key.clone(), value.clone());
                }
                ("delta", Value::Object(delta)) if !choice.whole => {
                    if let Value::Object(message) = fields
                        .entry("message")
                        .or_insert_with(|| Value::Object(Map::new()))
                    {
                        merge_delta(message, delta);
                    }
                }
                ("delta", _) if !choice.whole => {}
                ("logprobs", _) if !choice.whole => merge_logprobs(fields, value),
                _ => keep(fields, key, value),
            }
        }
    }
}

impl Reassemble<WireFrame> for ChatCompletion {
    fn absorb(&mut self, frame: &WireFrame) {
        match self.classifier.classify(frame.clone()) {
            WireEvent::Known(ChatEvent::Chunk(frame) | ChatEvent::Whole(frame)) => {
                self.frame(&frame);
            }
            WireEvent::Known(ChatEvent::BareText(_)) => self.bare = true,
            // The sentinel, the error envelope and what does not classify
            // are not part of the document.
            _ => {}
        }
    }

    fn finish(self) -> Value {
        let Self {
            mut envelope,
            mut choices,
            bare,
            ..
        } = self;
        if bare || envelope.is_empty() {
            return Value::Null;
        }
        if envelope.contains_key("choices") {
            // A unary body lists its choices in `index` order; a stream
            // interleaves them.
            choices.sort_by_key(|choice| choice.index);
            let choices = choices
                .into_iter()
                .map(|mut choice| {
                    if let (false, Some(Value::Object(message))) =
                        (choice.whole, choice.fields.get_mut("message"))
                    {
                        finish_message(message);
                    }
                    Value::Object(choice.fields)
                })
                .collect();
            envelope.insert("choices".to_owned(), Value::Array(choices));
        }
        Value::Object(envelope)
    }
}

/// Keep `value` at `key`: a non-null value replaces, and `null` only fills
/// an absent field.
fn keep(map: &mut Map<String, Value>, key: &str, value: &Value) {
    if !value.is_null() || !map.contains_key(key) {
        map.insert(key.to_owned(), value.clone());
    }
}

/// Append `fragment` to the text at `key`; a `null` there starts it.
fn append_text(map: &mut Map<String, Value>, key: &str, fragment: &str) {
    match map.get_mut(key) {
        Some(Value::String(text)) => text.push_str(fragment),
        _ => {
            map.insert(key.to_owned(), Value::from(fragment));
        }
    }
}

/// Fold `part` into `map`: the strings at `appended` keys append, and the
/// other fields are kept.
fn merge_appending(map: &mut Map<String, Value>, part: &Map<String, Value>, appended: &[&str]) {
    for (key, value) in part {
        match value {
            Value::String(fragment) if appended.contains(&key.as_str()) => {
                append_text(map, key, fragment);
            }
            _ => keep(map, key, value),
        }
    }
}

/// Fold one `delta` into the message it continues.
fn merge_delta(message: &mut Map<String, Value>, delta: &Map<String, Value>) {
    for (key, value) in delta {
        match (key.as_str(), value) {
            (text, Value::String(fragment)) if APPENDED_TEXT.contains(&text) => {
                append_text(message, key, fragment);
            }
            ("tool_calls", Value::Array(calls)) => {
                for call in calls {
                    merge_call(message, call);
                }
            }
            (REASONING_DETAILS, Value::Array(details)) => {
                merge_details(message, details.clone());
            }
            (list, Value::Array(entries)) if APPENDED_LISTS.contains(&list) => {
                match message.get_mut(key) {
                    Some(Value::Array(merged)) => merged.extend(entries.iter().cloned()),
                    _ => {
                        message.insert(key.clone(), value.clone());
                    }
                }
            }
            ("audio", Value::Object(audio)) => {
                if let Value::Object(merged) = message
                    .entry(key.clone())
                    .or_insert_with(|| Value::Object(Map::new()))
                {
                    merge_appending(merged, audio, &["data", "transcript"]);
                }
            }
            _ => keep(message, key, value),
        }
    }
}

/// Fold one tool-call fragment into the call at its `index`.
fn merge_call(message: &mut Map<String, Value>, fragment: &Value) {
    let Value::Object(fragment) = fragment else {
        return;
    };
    let index = fragment.get("index").and_then(Value::as_u64).unwrap_or(0);
    if !matches!(message.get("tool_calls"), Some(Value::Array(_))) {
        message.insert("tool_calls".to_owned(), Value::Array(Vec::new()));
    }
    let Some(Value::Array(calls)) = message.get_mut("tool_calls") else {
        return;
    };
    let position = calls
        .iter()
        .position(|call| call.get("index").and_then(Value::as_u64) == Some(index));
    let call = match position.and_then(|position| calls.get_mut(position)) {
        Some(call) => call,
        None => {
            calls.push(Value::Object(Map::from_iter([(
                "index".to_owned(),
                Value::from(index),
            )])));
            match calls.last_mut() {
                Some(call) => call,
                None => return,
            }
        }
    };
    let Value::Object(call) = call else {
        return;
    };
    for (key, value) in fragment {
        match (key.as_str(), value) {
            ("index", _) => {}
            ("function" | "custom", Value::Object(function)) => {
                if let Value::Object(merged) = call
                    .entry(key.clone())
                    .or_insert_with(|| Value::Object(Map::new()))
                {
                    merge_appending(merged, function, &["arguments", "input"]);
                }
            }
            _ => keep(call, key, value),
        }
    }
}

/// Append a chunk's token log probabilities to the choice's.
fn merge_logprobs(choice: &mut Map<String, Value>, logprobs: &Value) {
    let (Some(Value::Object(merged)), Value::Object(logprobs)) =
        (choice.get_mut("logprobs"), logprobs)
    else {
        keep(choice, "logprobs", logprobs);
        return;
    };
    for (key, value) in logprobs {
        match (merged.get_mut(key), value) {
            (Some(Value::Array(tokens)), Value::Array(more)) => tokens.extend(more.iter().cloned()),
            _ => keep(merged, key, value),
        }
    }
}

/// Close a rebuilt message as the unary body states it: calls without
/// their stream `index`, and an empty `content` beside calls or a refusal
/// is `null`.
fn finish_message(message: &mut Map<String, Value>) {
    let mut called = false;
    if let Some(Value::Array(calls)) = message.get_mut("tool_calls") {
        for call in calls.iter_mut().filter_map(Value::as_object_mut) {
            call.shift_remove("index");
        }
        called = !calls.is_empty();
    }
    let refused = message
        .get("refusal")
        .and_then(Value::as_str)
        .is_some_and(|refusal| !refusal.is_empty());
    if (called || refused) && message.get("content").and_then(Value::as_str) == Some("") {
        message.insert("content".to_owned(), Value::Null);
    }
}

#[cfg(test)]
mod tests;
