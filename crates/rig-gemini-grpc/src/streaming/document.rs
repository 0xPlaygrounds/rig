//! The gRPC GenerateContent document: each reply chunk's REST JSON, which
//! a unary reply reports whole and the REST wire's
//! [`GenerateContentResponse`](rig_core::providers::gemini::streaming::document::GenerateContentResponse)
//! rebuilds from a stream's chunks.

use serde_json::{Map, Value};

use crate::proto::GenerateContentResponse;
use crate::rest::to_rest;
use rig_core::error::ProviderError;
use rig_core::providers::gemini::streaming::document::GenerateContentResponse as Document;
use rig_core::wire::document::Reassemble;

/// `response` as its REST JSON object, the document a unary reply reports.
/// A message that does not transcode fails its decoding too.
pub(crate) fn rest_document(response: &GenerateContentResponse) -> Result<Value, ProviderError> {
    Ok(match to_rest(response)? {
        chunk @ Value::Object(_) => chunk,
        _ => Value::Object(Map::new()),
    })
}

impl Reassemble<GenerateContentResponse> for Document {
    fn absorb(&mut self, frame: &GenerateContentResponse) {
        if let Ok(Value::Object(chunk)) = rest_document(frame) {
            self.chunk(chunk);
        }
    }

    fn finish(self) -> Value {
        self.document()
    }
}
