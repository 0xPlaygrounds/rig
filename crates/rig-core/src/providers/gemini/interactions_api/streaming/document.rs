//! The Interactions reassembler: a streamed interaction's events rebuilt
//! into the interaction resource the unary reply returns, its `steps`
//! included, which `interaction.completed` leaves out.
//!
//! ```
//! use rig_core::providers::gemini::interactions_api::streaming::InteractionsDecoder;
//! use rig_core::providers::gemini::interactions_api::streaming::document::Interaction;
//! use rig_core::wire::document::Reassemble;
//! use rig_core::wire::{Decoder, WireEvent, WireFrame};
//!
//! let decoder = InteractionsDecoder::default();
//! let mut document = Interaction::default();
//! for frame in [
//!     r#"{"event_type":"step.start","index":0,"step":{"type":"model_output"}}"#,
//!     r#"{"event_type":"step.delta","index":0,"delta":{"type":"text","text":"42"}}"#,
//!     r#"{"event_type":"interaction.completed","interaction":{"id":"v1_a","status":"completed"}}"#,
//! ] {
//!     if let WireEvent::Known(event) = decoder.classify(WireFrame::Text(frame.to_owned())) {
//!         document.absorb(&event);
//!     }
//! }
//! let document = document.finish();
//! assert_eq!(document["steps"][0]["content"][0]["text"], "42");
//! assert_eq!(document["status"], "completed");
//! ```

use std::collections::BTreeMap;

use serde_json::{Map, Value, json};

use super::{InteractionsEvent, SseEvent};
use crate::json_utils::Lenient;
use crate::operation::completion::merge;
use crate::wire::document::{Reassemble, slot};

/// The interaction resource a stream of Interactions events adds up to.
///
/// `interaction.created`, `interaction.status_update` and
/// `interaction.completed` set the resource's fields: a non-null value
/// replaces, and `null` only fills an absent field. The steps are rebuilt
/// by index as the decoder folds them: `step.start` states a step, a
/// `text` or `text_annotation_delta` delta extends a model output's last
/// text item and any other content delta is an item of its own, a text
/// `thought_summary` extends the summary's last text item,
/// `arguments_delta` fragments become the call's `arguments` at
/// `step.stop`, and any other delta merges into its step. Steps the completed interaction states itself win. A whole
/// interaction frame is the document as sent.
#[derive(Default)]
pub struct Interaction {
    document: Map<String, Value>,
    steps: BTreeMap<usize, Value>,
    /// The argument JSON each call step streamed so far.
    arguments: BTreeMap<usize, String>,
    whole: Option<Map<String, Value>>,
}

impl Interaction {
    /// Set the resource fields `interaction` states.
    fn fields(&mut self, interaction: Map<String, Value>) {
        for (key, value) in interaction {
            if key == "steps" && value.as_array().is_none_or(Vec::is_empty) {
                continue;
            }
            if !value.is_null() || !self.document.contains_key(&key) {
                self.document.insert(key, value);
            }
        }
    }

    /// The step at `index`, opened as the kind `delta` implies when no
    /// `step.start` stated it.
    fn step(&mut self, index: usize, delta: &str) -> &mut Value {
        self.steps.entry(index).or_insert_with(|| {
            let kind = match delta {
                "text" | "text_annotation_delta" | "image" | "audio" | "document" | "video" => {
                    "model_output"
                }
                "thought_summary" | "thought_signature" => "thought",
                "arguments_delta" => "function_call",
                other => other,
            };
            json!({ "type": kind })
        })
    }

    /// Apply one delta to the step at `index`.
    fn delta(&mut self, index: usize, delta: &Map<String, Value>) {
        let kind = delta
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let step = self.step(index, kind);
        let own = step.str("type").unwrap_or_default().to_owned();
        match (own.as_str(), kind) {
            ("model_output", _) => {
                let Some(content) = list(step, "content") else {
                    return;
                };
                match content.last_mut() {
                    Some(item)
                        if matches!(kind, "text" | "text_annotation_delta")
                            && item.str("type") == Some("text") =>
                    {
                        merge(item, delta);
                    }
                    _ => content.push(Value::Object(delta.clone())),
                }
            }
            ("thought", "thought_summary") => {
                let content = delta.get("content").cloned().unwrap_or_default();
                let Some(summary) = list(step, "summary") else {
                    return;
                };
                match (summary.last_mut(), content.as_object()) {
                    (Some(item), Some(fields))
                        if item.str("type") == Some("text")
                            && content.str("type") == Some("text") =>
                    {
                        merge(item, fields);
                    }
                    _ => summary.push(content),
                }
            }
            ("function_call", "arguments_delta") => {
                let arguments = self.arguments.entry(index).or_default();
                match delta.get("arguments") {
                    Some(Value::String(fragment)) => arguments.push_str(fragment),
                    Some(Value::Null) | None => {}
                    Some(other) => arguments.push_str(&other.to_string()),
                }
            }
            // A delta that restates its step replaces the fields it names;
            // any other merges.
            (own, kind) if own == kind => {
                if let Some(step) = step.as_object_mut() {
                    step.extend(delta.clone());
                }
            }
            _ => merge(step, delta),
        }
    }

    /// The step at `index` is complete: its streamed arguments, when they
    /// parse, supersede the ones its start announced.
    fn stop(&mut self, index: usize) {
        let Some(arguments) = self.arguments.remove(&index) else {
            return;
        };
        if let (false, Ok(arguments), Some(Value::Object(step))) = (
            arguments.is_empty(),
            crate::json_utils::parse_tool_arguments(&arguments),
            self.steps.get_mut(&index),
        ) {
            step.insert("arguments".to_owned(), arguments);
        }
    }
}

/// The list at `key` of `step`, made empty when it is absent or not a
/// list. `None` when `step` is not an object.
fn list<'a>(step: &'a mut Value, key: &str) -> Option<&'a mut Vec<Value>> {
    slot(step.as_object_mut()?, key, Value::Array).as_array_mut()
}

impl crate::wire::document::Serves<crate::operation::Completion> for Interaction {}

impl Reassemble<InteractionsEvent> for Interaction {
    fn absorb(&mut self, event: &InteractionsEvent) {
        let SseEvent { event_type, fields } = match event {
            InteractionsEvent::Whole(interaction) => {
                self.whole = Some(interaction.clone());
                return;
            }
            InteractionsEvent::Sse(event) => event,
        };
        let index = fields
            .get("index")
            .and_then(Value::as_u64)
            .and_then(|index| usize::try_from(index).ok());
        match (event_type.as_str(), index) {
            ("interaction.created" | "interaction.completed", _) => {
                if let Some(Value::Object(interaction)) = fields.get("interaction") {
                    self.fields(interaction.clone());
                }
            }
            ("interaction.status_update", _) => {
                if let Some(status) = fields.get("status") {
                    self.fields(Map::from_iter([("status".to_owned(), status.clone())]));
                }
            }
            ("step.start", Some(index)) => {
                if let Some(step @ Value::Object(_)) = fields.get("step") {
                    self.steps.insert(index, step.clone());
                }
            }
            ("step.delta", Some(index)) => {
                if let Some(Value::Object(delta)) = fields.get("delta") {
                    self.delta(index, delta);
                }
            }
            ("step.stop", Some(index)) => self.stop(index),
            _ => {}
        }
    }

    fn finish(mut self) -> Value {
        if let Some(whole) = self.whole {
            return Value::Object(whole);
        }
        if !self.steps.is_empty() && !self.document.contains_key("steps") {
            let steps = std::mem::take(&mut self.steps).into_values().collect();
            self.document
                .insert("steps".to_owned(), Value::Array(steps));
        }
        if self.document.is_empty() {
            Value::Null
        } else {
            Value::Object(self.document)
        }
    }
}

#[cfg(test)]
mod tests;
