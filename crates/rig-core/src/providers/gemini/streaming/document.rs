//! The GenerateContent reassembler.
//!
//! Interim: [`TerminalRecord`] rebuilds the streamed `raw` this wire wrote
//! before replies were reassembled, so no recorded value moves with the
//! mechanism. The Gemini family replaces it with the
//! `GenerateContentResponse` reassembler of TYPED_OPTIONS.md section 9.

use serde_json::{Map, Value, json};

use super::super::completion::blocked_prompt_error;
use super::{GenerateContentChunk, GenerateContentDecoder};
use crate::json_utils::Lenient;
use crate::wire::document::Reassemble;
use crate::wire::{Decoder, WireEvent, WireFrame};

/// Interim GenerateContent reassembler: the summary record of a reply
/// (`usage_metadata`, the Gemini `finish_reason`, `finish_message`,
/// `model_version` and `response_id`), with the latest value of each. A
/// reply with no finish reason, or one that failed in band, rebuilds
/// nothing.
#[derive(Debug, Default)]
pub struct TerminalRecord {
    /// The decoder's classifier.
    classifier: GenerateContentDecoder,
    finish: Option<String>,
    finish_message: Option<String>,
    usage: Option<Value>,
    model_version: Option<String>,
    response_id: Option<String>,
    /// Whether the decoder ended the reply at a finish reason that is not
    /// a stop, or failed.
    closed: bool,
    failed: bool,
}

impl TerminalRecord {
    /// One chunk, as the decoder reads its metadata.
    fn chunk(&mut self, data: &Value) {
        if let Some(id) = data.str("responseId").filter(|id| !id.is_empty()) {
            self.response_id = Some(id.to_owned());
        }
        if let Some(model) = data.str("modelVersion").filter(|model| !model.is_empty()) {
            self.model_version = Some(model.to_owned());
        }
        if let Some(usage) = data.get("usageMetadata") {
            self.usage = Some(usage.clone());
        }
        if data.at("/error").is_some()
            || data
                .get("promptFeedback")
                .and_then(blocked_prompt_error)
                .is_some()
        {
            self.failed = true;
            return;
        }
        let candidate = match data
            .get("candidates")
            .map(|candidates| (candidates, candidates.get(0)))
        {
            None | Some((Value::Null, _) | (Value::Array(_), None)) => return,
            Some((Value::Array(_), Some(candidate @ Value::Object(_)))) => candidate,
            Some(_) => {
                self.failed = true;
                return;
            }
        };
        match candidate.get("finishReason") {
            Some(Value::String(name)) => self.finish = Some(name.clone()),
            Some(Value::Number(number)) => self.finish = Some(format!("FINISH_REASON_{number}")),
            _ => {}
        }
        if let Some(message) = candidate.str("finishMessage") {
            self.finish_message = Some(message.to_owned());
        }
        let parts = match candidate.get("content") {
            None | Some(Value::Null) => None,
            Some(content @ Value::Object(_)) => content.get("parts"),
            Some(_) => {
                self.failed = true;
                return;
            }
        };
        if !matches!(parts, None | Some(Value::Null | Value::Array(_))) {
            self.failed = true;
            return;
        }
        // The decoder ends the reply at once on a finish reason that is
        // neither a stop nor the token limit.
        use crate::completion::FinishReason::{Length, Stop};
        let reason = self
            .finish
            .as_deref()
            .map(super::super::completion::map_google_finish_reason);
        self.closed = !matches!(
            (reason, candidate.get("finishReason")),
            (None | Some(Stop | Length), _) | (_, None)
        );
    }
}

impl Reassemble<WireFrame> for TerminalRecord {
    fn absorb(&mut self, frame: &WireFrame) {
        if self.closed || self.failed {
            return;
        }
        // The driver reports what does not classify.
        if let WireEvent::Known(GenerateContentChunk(data)) =
            self.classifier.classify(frame.clone())
        {
            self.chunk(&Value::Object(data));
        }
    }

    fn finish(self) -> Value {
        if self.failed {
            return Value::Null;
        }
        let Some(reason) = self.finish else {
            return Value::Null;
        };
        let summary = json!({
            "usage_metadata": self.usage.unwrap_or_else(|| Value::Object(Map::new())),
            "finish_reason": reason,
            "finish_message": self.finish_message,
            "model_version": self.model_version,
            "response_id": self.response_id,
        });
        let Value::Object(mut summary) = summary else {
            return Value::Null;
        };
        summary.retain(|_, value| !value.is_null());
        Value::Object(summary)
    }
}
