//! The gRPC GenerateContent reassembler.
//!
//! Interim: [`TerminalRecord`] rebuilds the streamed `raw` this wire wrote
//! before replies were reassembled, so no recorded value moves with the
//! mechanism. The Gemini family replaces it with the REST wire's
//! `GenerateContentResponse` reassembler over each chunk's REST JSON
//! (TYPED_OPTIONS.md section 9).

use serde_json::{Map, Value};

use crate::proto::GenerateContentResponse;
use crate::rest::to_rest;
use rig_core::wire::document::Reassemble;

/// `response` as its REST JSON object, the document a unary reply reports.
/// `None` when it does not transcode, which fails its decoding too.
pub(crate) fn rest_document(response: &GenerateContentResponse) -> Option<Value> {
    match to_rest(response).ok()? {
        chunk @ Value::Object(_) => Some(chunk),
        _ => Some(Value::Object(Map::new())),
    }
}

/// Interim gRPC reassembler: the REST JSON of the latest chunk whose first
/// candidate states a finish reason. A stream with none rebuilds nothing.
#[derive(Debug, Default)]
pub struct TerminalRecord(Option<Value>);

impl Reassemble<GenerateContentResponse> for TerminalRecord {
    fn absorb(&mut self, frame: &GenerateContentResponse) {
        if frame
            .candidates
            .first()
            .is_some_and(|candidate| candidate.finish_reason != 0)
            && let Some(chunk) = rest_document(frame)
        {
            self.0 = Some(chunk);
        }
    }

    fn finish(self) -> Value {
        self.0.unwrap_or(Value::Null)
    }
}
