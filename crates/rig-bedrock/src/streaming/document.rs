//! The Converse reassembler.
//!
//! Interim: [`TerminalRecord`] rebuilds the streamed `raw` this wire wrote
//! before replies were reassembled, so no recorded value moves with the
//! mechanism. Its family replaces it with the `ConverseOutput`
//! reassembler of TYPED_OPTIONS.md section 9.

use serde_json::{Map, Value};

use super::member;
use crate::completion::ConverseFrame;
use rig_core::wire::document::Reassemble;

/// Interim Converse reassembler: the stream's `messageStart`,
/// `messageStop` and `metadata` payloads, by type, once `metadata` ended
/// the reply. A whole reply, or a stream that did not reach `metadata`,
/// rebuilds nothing.
#[derive(Debug, Default)]
pub struct TerminalRecord {
    record: Map<String, Value>,
    ended: bool,
}

impl Reassemble<ConverseFrame> for TerminalRecord {
    fn absorb(&mut self, frame: &ConverseFrame) {
        let ConverseFrame::Event(event) = frame else {
            return;
        };
        if self.ended {
            return;
        }
        if let Some((kind @ ("messageStart" | "messageStop" | "metadata"), payload)) = member(event)
        {
            self.record.insert(kind.to_owned(), payload.clone());
            self.ended = kind == "metadata";
        }
    }

    fn finish(self) -> Value {
        if self.ended {
            Value::Object(self.record)
        } else {
            Value::Null
        }
    }
}
