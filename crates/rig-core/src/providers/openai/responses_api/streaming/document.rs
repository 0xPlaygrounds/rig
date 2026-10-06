//! The Responses reassembler: a streamed reply's `raw` is the `Response`
//! object a unary reply's body is.

use std::collections::BTreeMap;

use serde_json::{Map, Value, json};

use super::{ResponsesEvent, classify_responses_payload};
use crate::json_utils::Lenient;
use crate::wire::document::Reassemble;
use crate::wire::{WireEvent, WireFrame};

/// Rebuilds the `Response` of a Responses reply from its events. The
/// response the latest lifecycle event carries is the document, and a
/// terminal one (`response.completed`, `.incomplete` or `.failed`) is
/// final, with the status its event names. When that response states no
/// `output`, as the ChatGPT backend's terminal does, the output is the
/// items `response.output_item.done` stated, by output index. Fields a
/// lifecycle event states beside its `response` (Copilot's `copilot_usage`)
/// are fields of the document, as the unary body states them. A whole
/// response body, as a WebSocket `response.done` carries, is the document
/// as it stands.
#[derive(Debug, Default)]
pub struct Response {
    /// The response object stated so far.
    response: Option<Value>,
    /// Whether `response` is the reply's last word.
    terminal: bool,
    /// The output items the stream stated, by output index: as added, then
    /// as done.
    items: BTreeMap<usize, Value>,
    /// Fields lifecycle events stated beside their `response`, the latest
    /// non-null value of each.
    beside: Map<String, Value>,
}

impl Response {
    /// Keep `item` at the frame's output index, or after every item so far
    /// when it names none. An item added after its index is done keeps the
    /// done one.
    fn item(&mut self, frame: &Value, done: bool) {
        let Some(item) = frame.get("item").filter(|item| item.is_object()) else {
            return;
        };
        let index = frame
            .u64("output_index")
            .and_then(|index| usize::try_from(index).ok())
            .unwrap_or_else(|| {
                self.items
                    .last_key_value()
                    .map_or(0, |(last, _)| last.saturating_add(1))
            });
        if done {
            self.items.insert(index, item.clone());
        } else {
            self.items.entry(index).or_insert_with(|| item.clone());
        }
    }

    /// The response a lifecycle event of `kind` carries. A terminal one
    /// states its status by its event, whatever its body says.
    fn lifecycle(&mut self, kind: &str, frame: &Value) {
        if self.terminal {
            return;
        }
        for (key, value) in frame.as_object().into_iter().flatten() {
            let event_field = matches!(key.as_str(), "type" | "sequence_number" | "response");
            if !event_field && (!value.is_null() || !self.beside.contains_key(key)) {
                self.beside.insert(key.clone(), value.clone());
            }
        }
        let status = match kind {
            "response.completed" => Some("completed"),
            "response.incomplete" => Some("incomplete"),
            "response.failed" => Some("failed"),
            _ => None,
        };
        let mut response = frame
            .get("response")
            .filter(|response| response.is_object())
            .cloned()
            .unwrap_or_else(|| json!({}));
        if let Some(status) = status {
            self.terminal = true;
            if let Some(fields) = response.as_object_mut() {
                fields.insert("status".to_owned(), json!(status));
            }
        }
        self.response = Some(response);
    }
}

impl crate::wire::document::Serves<crate::operation::Completion> for Response {}

impl Reassemble<WireFrame> for Response {
    fn absorb(&mut self, frame: &WireFrame) {
        match classify_responses_payload(&frame.as_str()) {
            WireEvent::Known(ResponsesEvent::Frame { kind, frame, .. }) => match kind.as_str() {
                "response.output_item.added" => self.item(&frame, false),
                "response.output_item.done" => self.item(&frame, true),
                kind if super::is_lifecycle_event(kind) => self.lifecycle(kind, &frame),
                _ => {}
            },
            WireEvent::Known(ResponsesEvent::Whole(body)) if !self.terminal => {
                self.terminal = true;
                self.response = Some(body);
            }
            // An `error` event ends the reply with the response so far; the
            // driver reports what does not classify.
            _ => {}
        }
    }

    fn finish(self) -> Value {
        let Some(mut response) = self.response.or_else(|| {
            (!self.items.is_empty()).then(|| json!({ "object": "response", "output": [] }))
        }) else {
            return Value::Null;
        };
        if let Some(fields) = response.as_object_mut() {
            for (key, value) in self.beside {
                fields.entry(key).or_insert(value);
            }
        }
        if let Some(fields) = response.as_object_mut()
            && fields
                .get("output")
                .is_none_or(|output| output.as_array().is_none_or(Vec::is_empty))
            && !self.items.is_empty()
        {
            fields.insert(
                "output".to_owned(),
                Value::Array(self.items.into_values().collect()),
            );
        }
        response
    }
}

#[cfg(test)]
mod tests;
