//! The native `/api/chat` reassembler.
//!
//! Interim: [`TerminalRecord`] rebuilds the streamed `raw` this wire wrote
//! before replies were reassembled, so no recorded value moves with the
//! mechanism. The Ollama family replaces it with the chat-response
//! reassembler of TYPED_OPTIONS.md section 9.

use serde_json::Value;

use super::{ChatDecoder, ChatRecord};
use crate::json_utils::Lenient;
use crate::wire::document::Reassemble;
use crate::wire::{Decoder, WireEvent, WireFrame};

/// Interim native chat reassembler: the record that ended the reply
/// (`done: true`), as sent. A reply that failed in band first rebuilds
/// nothing.
#[derive(Debug, Default)]
pub struct TerminalRecord {
    /// The decoder's classifier.
    classifier: ChatDecoder,
    record: Option<Value>,
    failed: bool,
}

impl Reassemble<WireFrame> for TerminalRecord {
    fn absorb(&mut self, frame: &WireFrame) {
        if self.record.is_some() || self.failed {
            return;
        }
        // The driver reports what does not classify.
        let WireEvent::Known(ChatRecord(fields)) = self.classifier.classify(frame.clone()) else {
            return;
        };
        let record = Value::Object(fields);
        if record.get("error").is_some_and(|error| !error.is_null()) {
            self.failed = true;
        } else if record.bool("done") == Some(true) {
            self.record = Some(record);
        }
    }

    fn finish(self) -> Value {
        self.record.unwrap_or(Value::Null)
    }
}
