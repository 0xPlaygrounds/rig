//! Decodes Gemini protobuf replies into Rig completion events.
//!
//! ```
//! use rig_core::wire::Wire;
//! use rig_gemini_grpc::completion::{GEMINI_2_5_FLASH, GenerateContent};
//!
//! let decoder = GenerateContent::new(GEMINI_2_5_FLASH).decoder();
//! # let _ = decoder;
//! ```

use base64::Engine as _;
use serde_json::{Map, Value, json};

use rig_core::driver::warn_unmodeled;
use rig_core::error::ProviderError;
use rig_core::operation::Completion;
use rig_core::providers::gemini::streaming::{GenerateContentChunk, GenerateContentDecoder};
use rig_core::wire::{Decoder, Flow, Out, WireEvent};

use super::completion::{encode_optional_base64, prost_struct_to_json, rest_usage};
use super::proto;

/// The Gemini gRPC wire's decoder over `GenerateContentResponse`s: a unary
/// reply is one of them, a stream sends several. Each is restated as the
/// REST chunk it transcodes to and read by the REST wire's decoder, so a
/// block's provider item is its part in REST JSON, which the encoder
/// transcodes back.
#[derive(Debug, Default)]
pub struct GrpcAdapter(GenerateContentDecoder);

impl<'id> Decoder<'id, Completion, proto::GenerateContentResponse> for GrpcAdapter {
    type Event = proto::GenerateContentResponse;

    fn classify(&self, frame: proto::GenerateContentResponse) -> WireEvent<Self::Event> {
        // Tonic handles frame decoding; unknown oneof values are handled per
        // part.
        WireEvent::Known(frame)
    }

    fn decode(
        &mut self,
        response: proto::GenerateContentResponse,
        out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        // The latest response carrying a finish reason is the raw record.
        if response
            .candidates
            .first()
            .is_some_and(|candidate| candidate.finish_reason != 0)
        {
            self.0.keep_raw(serde_json::to_value(&response)?);
        }
        Decoder::<'id, Completion>::decode(&mut self.0, rest_chunk(response), out)
    }

    fn eof(&mut self, out: Out<'id, Completion>) -> Result<Flow, ProviderError> {
        Decoder::<'id, Completion>::eof(&mut self.0, out)
    }
}

/// `response` as the REST chunk it transcodes to: its parts, finish reason,
/// safety ratings and citations. An unspecified finish reason is left out,
/// as the REST API does.
fn rest_chunk(response: proto::GenerateContentResponse) -> GenerateContentChunk {
    let candidates = response
        .candidates
        .into_iter()
        .map(|candidate| {
            let mut fields = Map::new();
            if let Some(content) = candidate.content {
                let parts: Vec<Value> = content.parts.iter().filter_map(rest_part).collect();
                fields.insert(
                    "content".to_owned(),
                    json!({ "parts": parts, "role": content.role }),
                );
            }
            if candidate.finish_reason != 0 {
                let reason = proto::candidate::FinishReason::try_from(candidate.finish_reason)
                    .map_or_else(
                        |_| format!("FINISH_REASON_{}", candidate.finish_reason),
                        |reason| reason.as_str_name().to_owned(),
                    );
                fields.insert("finishReason".to_owned(), reason.into());
            }
            if let Some(message) = candidate.finish_message {
                fields.insert("finishMessage".to_owned(), message.into());
            }
            if let Some(index) = candidate.index {
                fields.insert("index".to_owned(), index.into());
            }
            if !candidate.safety_ratings.is_empty() {
                let ratings: Vec<Value> = candidate
                    .safety_ratings
                    .iter()
                    .map(|rating| {
                        json!({
                            "category": rating.category().as_str_name(),
                            "probability": rating.probability().as_str_name(),
                            "blocked": rating.blocked,
                        })
                    })
                    .collect();
                fields.insert("safetyRatings".to_owned(), ratings.into());
            }
            if let Some(citations) = candidate.citation_metadata {
                let sources: Vec<Value> = citations
                    .citation_sources
                    .into_iter()
                    .map(|source| {
                        json!({
                            "startIndex": source.start_index,
                            "endIndex": source.end_index,
                            "uri": source.uri,
                            "license": source.license,
                        })
                    })
                    .collect();
                fields.insert(
                    "citationMetadata".to_owned(),
                    json!({ "citationSources": sources }),
                );
            }
            fields
        })
        .collect();
    GenerateContentChunk {
        response_id: response.response_id,
        candidates,
        usage_metadata: response.usage_metadata.as_ref().map(rest_usage),
        model_version: Some(response.model_version).filter(|model| !model.is_empty()),
        ..GenerateContentChunk::default()
    }
}

/// One protobuf part in its REST JSON form. A part whose data this proto
/// does not model is skipped with the shared redacted warning.
fn rest_part(part: &proto::Part) -> Option<Value> {
    use proto::part::Data;
    let base64 = |bytes: &[u8]| base64::engine::general_purpose::STANDARD.encode(bytes);
    let (kind, data) = match part.data.as_ref() {
        Some(Data::Text(text)) => ("text", json!(text)),
        Some(Data::InlineData(blob)) => (
            "inlineData",
            json!({ "mimeType": blob.mime_type, "data": base64(&blob.data) }),
        ),
        Some(Data::FileData(file)) => (
            "fileData",
            json!({ "mimeType": file.mime_type, "fileUri": file.file_uri }),
        ),
        Some(Data::FunctionCall(call)) => {
            let mut function_call = json!({
                "name": call.name,
                "args": call.args.as_ref().map_or_else(|| json!({}), prost_struct_to_json),
            });
            if !call.id.is_empty()
                && let Some(function_call) = function_call.as_object_mut()
            {
                function_call.insert("id".to_owned(), json!(call.id));
            }
            ("functionCall", function_call)
        }
        Some(Data::FunctionResponse(response)) => (
            "functionResponse",
            json!({
                "name": response.name,
                "response": response.response.as_ref().map(prost_struct_to_json),
                "id": response.id,
            }),
        ),
        Some(Data::ExecutableCode(code)) => (
            "executableCode",
            json!({ "language": code.language, "code": code.code }),
        ),
        Some(Data::CodeExecutionResult(result)) => (
            "codeExecutionResult",
            json!({ "outcome": result.outcome, "output": result.output }),
        ),
        None => {
            warn_unmodeled("gemini_grpc_part", part);
            return None;
        }
    };
    let mut rest = Map::from_iter([(kind.to_owned(), data)]);
    if part.thought {
        rest.insert("thought".to_owned(), true.into());
    }
    if let Some(signature) = encode_optional_base64(&part.thought_signature) {
        rest.insert("thoughtSignature".to_owned(), signature.into());
    }
    Some(Value::Object(rest))
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests;
