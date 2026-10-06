//! The Messages reassembler: a streamed reply rebuilt into the `Message` a
//! unary reply is, so one reading of `raw` serves both.

use std::collections::BTreeMap;

use serde_json::{Map, Value, json};

use super::{MessagesDecoder, MessagesEvent};
use crate::wire::document::Reassemble;
use crate::wire::{Decoder, WireEvent, WireFrame};

/// Rebuilds the `Message` a streamed Messages reply adds up to.
///
/// `message_start.message` is the skeleton. Each `content_block_start`
/// places its block at its index in `content`. Text, thinking and signature
/// deltas append; `input_json_delta` fragments are parsed into the block's
/// `input` at `content_block_stop` (`{}` when none streamed); a
/// `citations_delta` appends to `citations`; any other delta merges into
/// its block as the decoder merges it. `message_delta` folds its fields
/// and its cumulative `usage` over the skeleton. A whole `message` frame is
/// the document. A reply that failed or was cut short yields the message
/// so far, and one that stated none yields `Null`.
#[derive(Debug, Default)]
pub struct Message {
    /// The message so far; `None` until a frame states part of it.
    message: Option<Map<String, Value>>,
    /// The input JSON streamed to each block not yet stopped.
    inputs: BTreeMap<usize, String>,
}

impl Message {
    fn message(&mut self) -> &mut Map<String, Value> {
        self.message.get_or_insert_with(Map::new)
    }

    /// The block at `index` in `content`, when one was placed there.
    fn block(&mut self, index: usize) -> Option<&mut Map<String, Value>> {
        self.message
            .as_mut()?
            .get_mut("content")?
            .as_array_mut()?
            .get_mut(index)?
            .as_object_mut()
    }

    /// Place `block` at `index` in `content`, padding any gap with `null`.
    fn place(&mut self, index: usize, block: Map<String, Value>) {
        let content = self
            .message()
            .entry("content")
            .or_insert_with(|| Value::Array(Vec::new()));
        if !content.is_array() {
            *content = Value::Array(Vec::new());
        }
        if let Value::Array(content) = content {
            if content.len() <= index {
                content.resize(index + 1, Value::Null);
            }
            if let Some(slot) = content.get_mut(index) {
                *slot = Value::Object(block);
            }
        }
    }

    fn delta(&mut self, index: usize, delta: Map<String, Value>) {
        let kind = delta
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        // A gateway that skips `content_block_start` still streams the
        // block's text, so the first delta opens it, as the decoder does.
        if self.block(index).is_none() {
            let opened = match kind {
                "text_delta" => json!({"type": "text", "text": ""}),
                "thinking_delta" => json!({"type": "thinking", "thinking": ""}),
                _ => Value::Null,
            };
            if let Value::Object(block) = opened {
                self.place(index, block);
            }
        }
        if kind == "input_json_delta" {
            if let Some(fragment) = delta.get("partial_json").and_then(Value::as_str) {
                self.inputs.entry(index).or_default().push_str(fragment);
            }
            return;
        }
        let Some(block) = self.block(index) else {
            return;
        };
        if kind == "citations_delta" {
            let citation = delta.get("citation").cloned().unwrap_or_default();
            match block.get_mut("citations") {
                Some(Value::Array(citations)) => citations.push(citation),
                _ => {
                    block.insert("citations".to_owned(), Value::Array(vec![citation]));
                }
            }
            return;
        }
        let mut item = Value::Object(std::mem::take(block));
        crate::operation::completion::merge(&mut item, &delta);
        if let Value::Object(item) = item {
            *block = item;
        }
    }

    /// Set the `input` of the block at `index` from the JSON streamed to
    /// it. Input that does not parse leaves the input the block opened
    /// with.
    fn settle(&mut self, index: usize) {
        let Some(json) = self.inputs.remove(&index) else {
            return;
        };
        let Ok(input) = crate::json_utils::parse_tool_arguments(&json) else {
            return;
        };
        if let Some(block) = self.block(index) {
            block.insert("input".to_owned(), input);
        }
    }

    fn message_delta(&mut self, event: &MessagesEvent) {
        if let Some(Value::Object(delta)) = event.fields.get("delta") {
            let message = self.message();
            for (key, value) in delta {
                fold(message, key, value);
            }
        }
        if let Some(Value::Object(terminal)) = event.fields.get("usage") {
            let usage = self
                .message()
                .entry("usage")
                .or_insert_with(|| Value::Object(Map::new()));
            if !usage.is_object() {
                *usage = Value::Object(Map::new());
            }
            if let Value::Object(usage) = usage {
                // Zero input is how some gateways leave the count out of
                // `message_delta`, as the decoder reads it.
                let started = usage
                    .get("input_tokens")
                    .and_then(Value::as_u64)
                    .is_some_and(|tokens| tokens > 0);
                for (key, value) in terminal {
                    if key == "input_tokens" && started && value.as_u64() == Some(0) {
                        continue;
                    }
                    fold(usage, key, value);
                }
            }
        }
    }
}

/// The shared folding rule: a non-null value replaces, and `null` only
/// fills an absent key.
fn fold(target: &mut Map<String, Value>, key: &str, value: &Value) {
    if !value.is_null() || !target.contains_key(key) {
        target.insert(key.to_owned(), value.clone());
    }
}

impl Reassemble<WireFrame> for Message {
    fn absorb(&mut self, frame: &WireFrame) {
        // The driver reports what does not classify.
        let WireEvent::Known(event) = MessagesDecoder::default().classify(frame.clone()) else {
            return;
        };
        match event.kind() {
            "message" => {
                if let Value::Object(message) = event.fields {
                    self.message = Some(message);
                    self.inputs.clear();
                }
            }
            "message_start" => {
                if let Some(Value::Object(start)) = event.fields.get("message") {
                    let message = self.message();
                    for (key, value) in start {
                        fold(message, key, value);
                    }
                }
            }
            "content_block_start" => {
                if let (Ok(index), Ok(block)) = (event.index(), event.item("content_block")) {
                    self.inputs.remove(&index);
                    self.place(index, block);
                }
            }
            "content_block_delta" => {
                if let (Ok(index), Ok(delta)) = (event.index(), event.item("delta")) {
                    self.delta(index, delta);
                }
            }
            "content_block_stop" => {
                if let Ok(index) = event.index() {
                    self.settle(index);
                }
            }
            "message_delta" => self.message_delta(&event),
            // `message_stop`, `ping` and `error` add nothing: a failed
            // reply keeps the message so far.
            _ => {}
        }
    }

    fn finish(mut self) -> Value {
        let open: Vec<usize> = self.inputs.keys().copied().collect();
        for index in open {
            self.settle(index);
        }
        self.message.map_or(Value::Null, Value::Object)
    }
}

#[cfg(test)]
mod tests;
