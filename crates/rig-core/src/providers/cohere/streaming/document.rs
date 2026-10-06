//! The native Chat reassembler.
//!
//! Interim: [`TerminalRecord`] rebuilds the streamed `raw` this wire wrote
//! before replies were reassembled, so no recorded value moves with the
//! mechanism. The Cohere family replaces it with the chat-response
//! reassembler of TYPED_OPTIONS.md section 9.

use serde_json::Value;

use super::ChatDecoder;
use crate::wire::document::Reassemble;
use crate::wire::{Decoder, WireEvent, WireFrame};

/// Interim native Chat reassembler: the `message-end` event as sent. A
/// whole reply, or a stream that never sent `message-end`, rebuilds
/// nothing.
#[derive(Debug, Default)]
pub struct TerminalRecord {
    /// The decoder's classifier.
    classifier: ChatDecoder,
    record: Option<Value>,
}

impl Reassemble<WireFrame> for TerminalRecord {
    fn absorb(&mut self, frame: &WireFrame) {
        if self.record.is_some() {
            return;
        }
        if let WireEvent::Known(event) = self.classifier.classify(frame.clone())
            && event.kind() == "message-end"
        {
            self.record = Some(event.fields);
        }
    }

    fn finish(self) -> Value {
        self.record.unwrap_or(Value::Null)
    }
}
