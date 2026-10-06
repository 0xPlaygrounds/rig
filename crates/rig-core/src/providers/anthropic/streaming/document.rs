//! The Messages reassembler.
//!
//! Interim: [`TerminalRecord`] rebuilds the streamed `raw` this wire wrote
//! before replies were reassembled, so no recorded value moves with the
//! mechanism. The Anthropic family replaces it with the `Message`
//! reassembler of TYPED_OPTIONS.md section 9.

use serde_json::{Value, json};

use super::{MessagesDecoder, object};
use crate::json_utils::Lenient;
use crate::wire::document::Reassemble;
use crate::wire::{Decoder, WireEvent, WireFrame};

/// Interim Messages reassembler: the terminal record of a streamed reply
/// (`usage` with `message_start`'s counters under the terminal ones,
/// `stop_reason`, `stop_sequence`, `message_id` and `model`), taken at the
/// `message_delta` that states the stop reason. A whole message, or a reply
/// that failed in band before that delta, rebuilds nothing.
#[derive(Debug, Default)]
pub struct TerminalRecord {
    /// The decoder's own metadata fold and classifier, fed the same frames.
    metadata: MessagesDecoder,
    record: Option<Value>,
    failed: bool,
}

impl Reassemble<WireFrame> for TerminalRecord {
    fn absorb(&mut self, frame: &WireFrame) {
        if self.record.is_some() || self.failed {
            return;
        }
        // The driver reports what does not classify.
        let WireEvent::Known(event) = self.metadata.classify(frame.clone()) else {
            return;
        };
        match event.kind() {
            "message_start" => {
                if let Some(message) = event.fields.get("message").filter(|m| m.is_object()) {
                    self.metadata.metadata(message);
                }
            }
            "message_delta" => {
                let delta = event.fields.get("delta");
                let Some(reason) = delta.and_then(|delta| delta.str("stop_reason")) else {
                    return;
                };
                let usage = self.metadata.terminal(event.fields.get("usage"));
                // The stop sequence rides the same `message_delta` as the
                // stop reason: `message_start` always opens with `null`.
                let stop_sequence = delta.and_then(|delta| delta.str("stop_sequence"));
                self.record = Some(Value::Object(object([
                    ("usage", Some(usage.record())),
                    ("stop_reason", Some(json!(reason))),
                    (
                        "stop_sequence",
                        stop_sequence.map(|sequence| json!(sequence)),
                    ),
                    (
                        "message_id",
                        self.metadata.message_id.clone().map(Value::String),
                    ),
                    (
                        "model",
                        self.metadata.response_model.clone().map(Value::String),
                    ),
                ])));
            }
            "message" | "error" => self.failed = true,
            _ => {}
        }
    }

    fn finish(self) -> Value {
        self.record.unwrap_or(Value::Null)
    }
}
