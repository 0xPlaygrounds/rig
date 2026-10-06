//! The Responses reassembler.
//!
//! Interim: [`TerminalRecord`] rebuilds the streamed `raw` this wire wrote
//! before replies were reassembled, so no recorded value moves with the
//! mechanism. The Responses family replaces it with the `Response`
//! reassembler of TYPED_OPTIONS.md section 9.

use serde_json::{Value, json};

use super::{ResponsesEvent, classify_responses_payload};
use crate::wire::WireFrame;
use crate::wire::document::Reassemble;

/// Interim Responses reassembler: the response a `response.completed` or
/// `response.incomplete` event carries (marked `incomplete` for the
/// latter), or a whole response body. A reply that failed in band first
/// rebuilds nothing.
#[derive(Debug, Default)]
pub struct TerminalRecord {
    record: Option<Value>,
    failed: bool,
}

impl Reassemble<WireFrame> for TerminalRecord {
    fn absorb(&mut self, frame: &WireFrame) {
        if self.record.is_some() || self.failed {
            return;
        }
        match classify_responses_payload(&frame.as_str()) {
            crate::wire::WireEvent::Known(ResponsesEvent::Frame { kind, frame, .. }) => {
                match kind.as_str() {
                    "response.completed" | "response.incomplete" => {
                        let mut response = frame
                            .get("response")
                            .filter(|response| response.is_object())
                            .cloned()
                            .unwrap_or_else(|| json!({}));
                        if let (true, Some(fields)) =
                            (kind == "response.incomplete", response.as_object_mut())
                        {
                            fields.insert("status".to_owned(), json!("incomplete"));
                        }
                        self.record = Some(response);
                    }
                    "response.failed" => self.failed = true,
                    _ => {}
                }
            }
            crate::wire::WireEvent::Known(ResponsesEvent::Whole(body)) => {
                self.record = Some(body);
            }
            crate::wire::WireEvent::Known(ResponsesEvent::Failure(_)) => self.failed = true,
            // The driver reports what does not classify.
            _ => {}
        }
    }

    fn finish(self) -> Value {
        self.record.unwrap_or(Value::Null)
    }
}
