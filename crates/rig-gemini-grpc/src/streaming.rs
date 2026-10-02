//! Decodes Gemini protobuf replies into Rig completion events.
//!
//! ```
//! use rig_core::wire::Wire;
//! use rig_gemini_grpc::completion::{GEMINI_2_5_FLASH, GenerateContent};
//!
//! let decoder = GenerateContent::new(GEMINI_2_5_FLASH).decoder();
//! # let _ = decoder;
//! ```

use serde_json::Value;

use rig_core::error::ProviderError;
use rig_core::operation::Completion;
use rig_core::providers::gemini::streaming::{GenerateContentChunk, GenerateContentDecoder};
use rig_core::wire::{Decoder, Flow, Out, WireEvent};

use super::proto;
use super::rest::to_rest;

/// The Gemini gRPC wire's decoder over `GenerateContentResponse`s: a unary
/// reply is one of them, a stream sends several. Each is read as its
/// canonical proto3 JSON, which is the REST chunk, by the REST wire's
/// decoder, so a block's provider item is its part in REST JSON, which the
/// encoder transcodes back.
#[derive(Debug, Default)]
pub struct GrpcAdapter(GenerateContentDecoder);

impl<'id> Decoder<'id, Completion, proto::GenerateContentResponse> for GrpcAdapter {
    type Event = proto::GenerateContentResponse;

    fn classify(&self, frame: proto::GenerateContentResponse) -> WireEvent<Self::Event> {
        // Tonic decodes the frames; every one is a reply chunk.
        WireEvent::Known(frame)
    }

    fn decode(
        &mut self,
        response: proto::GenerateContentResponse,
        out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        let chunk = match to_rest(&response)? {
            Value::Object(chunk) => chunk,
            _ => serde_json::Map::new(),
        };
        // The latest response carrying a finish reason is the raw record.
        if response
            .candidates
            .first()
            .is_some_and(|candidate| candidate.finish_reason != 0)
        {
            self.0.keep_raw(Value::Object(chunk.clone()));
        }
        Decoder::<'id, Completion>::decode(&mut self.0, GenerateContentChunk(chunk), out)
    }

    fn eof(&mut self, out: Out<'id, Completion>) -> Result<Flow, ProviderError> {
        Decoder::<'id, Completion>::eof(&mut self.0, out)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests;
