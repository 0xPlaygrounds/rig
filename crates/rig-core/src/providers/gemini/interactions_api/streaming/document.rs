//! The Interactions reassembler.
//!
//! Interim: [`TerminalRecord`] rebuilds the streamed `raw` this wire wrote
//! before replies were reassembled, so no recorded value moves with the
//! mechanism. The Gemini family replaces it with the interaction-resource
//! reassembler of TYPED_OPTIONS.md section 9.

use serde_json::{Map, Value};

use super::{InteractionsDecoder, InteractionsEvent, SseEvent};
use crate::json_utils::Lenient;
use crate::wire::document::Reassemble;
use crate::wire::{Decoder, WireEvent, WireFrame};

/// Interim Interactions reassembler: `{usage, interaction, model_version}`
/// for the interaction an `interaction.completed` event or a whole frame
/// states, `model_version` being its model or agent, else the one
/// `interaction.created` named. A reply that failed in band first rebuilds
/// nothing.
#[derive(Default)]
pub struct TerminalRecord {
    /// The decoder's classifier.
    classifier: InteractionsDecoder,
    /// The model `interaction.created` named.
    model: Option<String>,
    record: Option<Value>,
    failed: bool,
}

impl TerminalRecord {
    fn complete(&mut self, interaction: Map<String, Value>) {
        let interaction = Value::Object(interaction);
        let field = |key: &str| interaction.str(key).map(str::to_owned);
        let model = field("model")
            .or_else(|| field("agent"))
            .or_else(|| self.model.take());
        let usage = interaction.get("usage").cloned().unwrap_or(Value::Null);
        let mut record = Map::from_iter([
            ("usage".to_owned(), usage),
            ("interaction".to_owned(), interaction),
        ]);
        if let Some(model) = model {
            record.insert("model_version".to_owned(), Value::String(model));
        }
        self.record = Some(Value::Object(record));
    }
}

impl Reassemble<WireFrame> for TerminalRecord {
    fn absorb(&mut self, frame: &WireFrame) {
        if self.record.is_some() || self.failed {
            return;
        }
        // The driver reports what does not classify.
        let WireEvent::Known(event) = self.classifier.classify(frame.clone()) else {
            return;
        };
        let event = match event {
            InteractionsEvent::Whole(interaction) => {
                self.complete(interaction);
                return;
            }
            InteractionsEvent::Sse(event) => event,
        };
        let SseEvent { event_type, fields } = event;
        let fields = Value::Object(fields);
        match event_type.as_str() {
            "interaction.created" => {
                let field = |key: &str| {
                    fields
                        .at(&format!("/interaction/{key}"))
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                };
                self.model = field("model").or_else(|| field("agent"));
            }
            "interaction.completed" => {
                let interaction = fields.obj("interaction").cloned().unwrap_or_default();
                self.complete(interaction);
            }
            "error" => self.failed = true,
            _ => {}
        }
    }

    fn finish(self) -> Value {
        self.record.unwrap_or(Value::Null)
    }
}
