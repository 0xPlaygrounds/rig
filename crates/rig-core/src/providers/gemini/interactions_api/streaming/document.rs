//! The Interactions reassembler: a streamed interaction's events rebuilt
//! into the interaction resource the unary reply returns, its `steps`
//! included, which `interaction.completed` leaves out.
//!
//! ```
//! use rig_core::providers::gemini::interactions_api::streaming::document::Interaction;
//! use rig_core::wire::WireFrame;
//! use rig_core::wire::document::Reassemble;
//!
//! let mut document = Interaction::default();
//! for event in [
//!     r#"{"event_type":"step.start","index":0,"step":{"type":"model_output"}}"#,
//!     r#"{"event_type":"step.delta","index":0,"delta":{"type":"text","text":"42"}}"#,
//!     r#"{"event_type":"interaction.completed","interaction":{"id":"v1_a","status":"completed"}}"#,
//! ] {
//!     document.absorb(&WireFrame::Text(event.to_owned()));
//! }
//! let document = document.finish();
//! assert_eq!(document["steps"][0]["content"][0]["text"], "42");
//! assert_eq!(document["status"], "completed");
//! ```

use std::collections::BTreeMap;

use serde_json::{Map, Value, json};

use super::{InteractionsDecoder, InteractionsEvent, SseEvent};
use crate::json_utils::Lenient;
use crate::operation::completion::merge;
use crate::wire::document::Reassemble;
use crate::wire::{Decoder, WireEvent, WireFrame};

/// The interaction resource a stream of Interactions events adds up to.
///
/// `interaction.created`, `interaction.status_update` and
/// `interaction.completed` set the resource's fields: a non-null value
/// replaces, and `null` only fills an absent field. The steps are rebuilt
/// by index as the decoder folds them: `step.start` states a step, a
/// `text` delta extends a model output's last text item and any other
/// content delta is an item of its own, a text `thought_summary` extends
/// the summary's last text item, `arguments_delta` fragments become the
/// call's `arguments` at `step.stop`, and any other delta merges into its
/// step. Steps the completed interaction states itself win. A whole
/// interaction frame is the document as sent.
#[derive(Default)]
pub struct Interaction {
    /// The decoder's classifier.
    classifier: InteractionsDecoder,
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
                "text" | "image" | "audio" | "document" | "video" => "model_output",
                "thought_summary" | "thought_signature" => "thought",
                "arguments_delta" => "function_call",
                other => other,
            };
            json!({ "type": kind })
        })
    }

    /// Apply one delta to the step at `index`.
    fn delta(&mut self, index: usize, mut delta: Map<String, Value>) {
        let kind = delta
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let step = self.step(index, &kind);
        let own = step.str("type").unwrap_or_default().to_owned();
        match (own.as_str(), kind.as_str()) {
            ("model_output", _) => {
                let Some(content) = list(step, "content") else {
                    return;
                };
                match content.last_mut() {
                    Some(item) if kind == "text" && item.str("type") == Some("text") => {
                        delta.shift_remove("type");
                        merge(item, &delta);
                    }
                    _ => content.push(Value::Object(delta)),
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
                let fragment = match delta.get("arguments") {
                    Some(Value::String(fragment)) => fragment.clone(),
                    Some(Value::Null) | None => String::new(),
                    Some(other) => other.to_string(),
                };
                self.arguments.entry(index).or_default().push_str(&fragment);
            }
            // A delta that restates its step replaces the fields it names;
            // any other merges.
            (own, kind) if own == kind => {
                if let Some(step) = step.as_object_mut() {
                    step.extend(delta);
                }
            }
            _ => merge(step, &delta),
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
    let slot = step
        .as_object_mut()?
        .entry(key)
        .or_insert_with(|| Value::Array(Vec::new()));
    if !slot.is_array() {
        *slot = Value::Array(Vec::new());
    }
    slot.as_array_mut()
}

impl Reassemble<WireFrame> for Interaction {
    fn absorb(&mut self, frame: &WireFrame) {
        // The driver reports what does not classify.
        let WireEvent::Known(event) = self.classifier.classify(frame.clone()) else {
            return;
        };
        let SseEvent { event_type, fields } = match event {
            InteractionsEvent::Whole(interaction) => {
                self.whole = Some(interaction);
                return;
            }
            InteractionsEvent::Sse(event) => event,
        };
        let index = fields
            .get("index")
            .and_then(Value::as_u64)
            .and_then(|index| usize::try_from(index).ok());
        let mut fields = fields;
        match (event_type.as_str(), index) {
            ("interaction.created" | "interaction.completed", _) => {
                if let Some(Value::Object(interaction)) = fields.shift_remove("interaction") {
                    self.fields(interaction);
                }
            }
            ("interaction.status_update", _) => {
                if let Some(status) = fields.shift_remove("status") {
                    self.fields(Map::from_iter([("status".to_owned(), status)]));
                }
            }
            ("step.start", Some(index)) => {
                if let Some(step @ Value::Object(_)) = fields.shift_remove("step") {
                    self.steps.insert(index, step);
                }
            }
            ("step.delta", Some(index)) => {
                if let Some(Value::Object(delta)) = fields.shift_remove("delta") {
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
