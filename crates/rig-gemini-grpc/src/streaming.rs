//! Decodes Gemini protobuf replies into Rig completion events.
//!
//! ```
//! use rig_core::wire::{Mode, Wire};
//! use rig_gemini_grpc::completion::{GEMINI_2_5_FLASH, GenerateContent};
//!
//! let decoder = GenerateContent::new(GEMINI_2_5_FLASH).decoder(Mode::Streaming);
//! # let _ = decoder;
//! ```

use serde_json::{Map, Value};

use rig_core::driver::warn_unmodeled;
use rig_core::operation::{AdapterOutput, Completion};
use rig_core::providers::internal::chunk_lifecycle::{ChunkParts, MintedReasoningLifecycle};
use rig_core::providers::internal::wire::{self, TypedEvent, WireEvent};
use rig_core::streaming;

use super::completion::{
    GrpcFrame, encode_optional_base64 as encode_signature, prost_struct_to_json,
};
use super::proto;

/// The Gemini gRPC typed wire as a [`Decoder`](rig_core::wire::Decoder) over
/// protobuf frames: the chunk carrying a finish reason is the terminal, and
/// the per-stream state is the thought block's lifecycle plus the tool-key
/// minter. A whole unary reply replays as the events a stream sends.
pub struct GrpcAdapter {
    /// Derives signed reasoning boundaries for thought parts without wire IDs.
    reasoning: MintedReasoningLifecycle,
    /// Mints a distinct identity for each call lacking a wire ID.
    tool_ids: streaming::SyntheticIds,
    /// Suppresses further output after a tool-protocol terminal failure.
    failed: bool,
    /// The unary reply's document, for the response's `raw`.
    document: Option<Value>,
}

impl Default for GrpcAdapter {
    fn default() -> Self {
        Self {
            reasoning: MintedReasoningLifecycle::new(streaming::MintKind::Reasoning),
            tool_ids: streaming::SyntheticIds::tool(),
            failed: false,
            document: None,
        }
    }
}

impl rig_core::wire::Decoder<Completion, GrpcFrame> for GrpcAdapter {
    type Event = GrpcFrame;

    fn classify(&self, frame: GrpcFrame) -> WireEvent<Self::Event> {
        // Tonic handles frame decoding; unknown oneof values are handled per
        // part during interpretation.
        wire::classify_typed_event(TypedEvent::Modeled(frame))
    }

    fn interpret(&mut self, frame: Self::Event, out: &mut AdapterOutput) {
        if self.failed {
            return;
        }
        let resp = match frame {
            GrpcFrame::Chunk(chunk) => chunk,
            GrpcFrame::Whole(response) => return self.whole(*response, out),
        };

        let mut is_final = false;

        if let Some(candidate) = resp.candidates.first() {
            // Enum default is 0 = FINISH_REASON_UNSPECIFIED.
            if candidate.finish_reason != 0 {
                is_final = true;
            }

            // Protocol failures must not be followed by a successful terminal record.
            if let Some(err) = super::completion::tool_protocol_finish_reason_error(
                candidate.finish_reason,
                candidate.finish_message.as_deref(),
            ) {
                self.failed = true;
                out.push(Err(err));
                return;
            }

            if let Some(content) = candidate.content.as_ref() {
                // Parts within one response are distinct parts; text chunks
                // across stream responses continue one part. A signed part
                // keeps its signature on it and ends there.
                let mut previous_text = false;
                for part in &content.parts {
                    let text =
                        !part.thought && matches!(part.data, Some(proto::part::Data::Text(_)));
                    let signed = text && !part.thought_signature.is_empty();
                    let empty = matches!(&part.data, Some(proto::part::Data::Text(text)) if text.is_empty());
                    if text && (previous_text || (signed && empty)) {
                        out.end_active_text();
                    }
                    previous_text = text;
                    let parts = self.interpret_part(part);
                    self.reasoning.emit_chunk(parts, out);
                    if signed {
                        out.end_active_text();
                    }
                }
            }
        }

        // Only a provider finish reason establishes completion; synthesizing a
        // terminal at EOF would hide truncation.
        if is_final {
            match serde_json::to_value(&resp) {
                Ok(raw) => out.final_record(terminal_record(&resp, raw)),
                Err(err) => out.error(err.into()),
            }
        }
    }

    fn finish(&mut self, _out: &mut AdapterOutput) {
        // EOF without a finish reason is truncation: no terminal record.
    }

    fn is_finished(&self) -> bool {
        // Stop reading after an emitted protocol failure rather than drain the transport.
        self.failed
    }

    fn document(&self) -> Option<Value> {
        self.document.clone()
    }
}

impl GrpcAdapter {
    /// Replay a whole unary reply as the events a stream sends for it.
    fn whole(&mut self, response: proto::GenerateContentResponse, out: &mut AdapterOutput) {
        // The provider's own document, captured before the reply is consumed
        // into normalized content.
        match serde_json::to_value(&response) {
            Ok(document) => self.document = Some(document),
            Err(error) => return out.error(error.into()),
        }
        match super::completion::assistant_content(&response) {
            Ok(choice) => out.content(&choice),
            Err(error) => return out.error(error),
        }
        out.final_record(terminal_record(&response, Value::Null));
    }

    /// Converts a protobuf part into content for shared lifecycle derivation.
    fn interpret_part(&mut self, part: &proto::Part) -> ChunkParts {
        match &part.data {
            // A thought part's signature closes the thinking block: the shared
            // accumulator signs the accumulated deltas, using the same base64
            // encoding as the unary path.
            Some(proto::part::Data::Text(text)) if part.thought => ChunkParts {
                reasoning: Some(text.clone()),
                reasoning_signature: encode_signature(&part.thought_signature),
                ..ChunkParts::default()
            },
            // A signature on answer text returns on that text part.
            Some(proto::part::Data::Text(text)) => ChunkParts {
                text: Some(text.clone()),
                text_meta: encode_signature(&part.thought_signature).and_then(|signature| {
                    rig_core::providers::gemini::text_signature_extras(
                        rig_core::providers::gemini::GEMINI_TEXT_EXTRAS_KEY,
                        signature,
                    )
                }),
                ..ChunkParts::default()
            },
            Some(proto::part::Data::FunctionCall(function_call)) => {
                let args_json = function_call
                    .args
                    .as_ref()
                    .map_or_else(|| Value::Object(Map::new()), prost_struct_to_json);

                // Preserve wire identity or mint a distinct local key; tool names
                // cannot distinguish repeated calls and are never identifiers.
                let key = match streaming::non_empty_id(function_call.id.clone()) {
                    Some(wire_id) => streaming::BlockId::wire(wire_id),
                    None => self.tool_ids.mint(),
                };

                // Keep call_id unset for this single-ID protocol so downstream
                // normalization does not infer a separate item identity.
                let mut end = streaming::ToolCallEnd::whole(function_call.name.clone(), args_json)
                    // A signature on a function-call part belongs to the
                    // call, not to the thought block.
                    .with_signature(encode_signature(&part.thought_signature));
                end.tool_id = key.wire_str().map(str::to_owned);

                // A whole call is its start and its authoritative end.
                ChunkParts {
                    tool_events: vec![
                        streaming::StreamEvent::BlockStart {
                            id: key.clone(),
                            kind: streaming::BlockKind::ToolCall,
                        },
                        streaming::StreamEvent::BlockEnd {
                            id: key,
                            end: streaming::BlockClose::ToolCall(end),
                            block: None,
                        },
                    ],
                    ..ChunkParts::default()
                }
            }
            None => {
                // Missing or unknown oneof data uses the shared redacted warning policy.
                warn_unmodeled("gemini_grpc_part", part);
                ChunkParts::default()
            }
            Some(_) => ChunkParts::default(),
        }
    }
}

/// Map the terminal `GenerateContentResponse` onto rig's
/// [`streaming::StreamFinal`], serializing the native record onto
/// [`streaming::StreamFinal::raw`].
fn terminal_record(
    response: &proto::GenerateContentResponse,
    raw: Value,
) -> streaming::StreamFinal {
    let usage = super::completion::map_usage(response.usage_metadata.as_ref());
    let finish_reason = response
        .candidates
        .first()
        .and_then(|candidate| super::completion::map_finish_reason(candidate.finish_reason));

    streaming::StreamFinal::new(super::completion::PROVIDER_NAME, usage, raw)
    .with_optional_finish_reason(finish_reason)
    .with_optional_response_id(Some(response.response_id.clone()).filter(|id| !id.is_empty()))
    .with_optional_model(Some(response.model_version.clone()).filter(|model| !model.is_empty()))
    .with_reasoning_issuer(super::completion::REASONING_ISSUER)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests;
