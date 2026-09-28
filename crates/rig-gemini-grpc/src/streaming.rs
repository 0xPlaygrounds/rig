//! Decodes Gemini protobuf replies into Rig completion events.
//!
//! ```
//! use rig_core::wire::Wire;
//! use rig_gemini_grpc::completion::{GEMINI_2_5_FLASH, GenerateContent};
//!
//! let decoder = GenerateContent::new(GEMINI_2_5_FLASH).decoder();
//! # let _ = decoder;
//! ```

use serde_json::{Map, Value};

use rig_core::driver::warn_unmodeled;
use rig_core::error::ProviderError;
use rig_core::message::NativePart;
use rig_core::operation::{Completion, Finish};
use rig_core::providers::gemini::api;
use rig_core::providers::gemini::edge::{self, Unit};
use rig_core::wire::{Flow, Out, WireEvent};

use super::completion::{encode_optional_base64 as encode_signature, prost_struct_to_json};
use super::proto;

/// The Gemini gRPC wire's decoder over `GenerateContentResponse`s: a unary
/// reply is one of them, a stream sends several. Like the REST wire, the
/// reply ends at EOF once a finish reason arrived, since a hosted-tool round
/// can report one before more content.
pub struct GrpcAdapter<'id> {
    /// The edge's writer: thoughts, text and calls in wire order.
    writer: edge::Writer<'id>,
    /// The latest response carrying a finish reason: the reply's end and
    /// its raw record.
    last: Option<proto::GenerateContentResponse>,
}

impl Default for GrpcAdapter<'_> {
    fn default() -> Self {
        Self {
            writer: edge::Writer::new(),
            last: None,
        }
    }
}

impl<'id> rig_core::wire::Decoder<'id, Completion, proto::GenerateContentResponse>
    for GrpcAdapter<'id>
{
    type Event = proto::GenerateContentResponse;

    fn classify(&self, frame: proto::GenerateContentResponse) -> WireEvent<Self::Event> {
        // Tonic handles frame decoding; unknown oneof values are handled per
        // part.
        WireEvent::Known(frame)
    }

    fn decode(
        &mut self,
        resp: proto::GenerateContentResponse,
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        out.issued_by(super::completion::REASONING_ISSUER);
        let Some(candidate) = resp.candidates.first() else {
            return Ok(Flow::More);
        };
        // Protocol failures must not be followed by a successful end.
        if let Some(err) = super::completion::tool_protocol_finish_reason_error(
            candidate.finish_reason,
            candidate.finish_message.as_deref(),
        ) {
            return Err(err);
        }
        if let Some(content) = candidate.content.as_ref() {
            // Parts within one response are distinct parts; text chunks
            // across stream responses continue one part.
            let mut previous_text = false;
            for part in &content.parts {
                let Some(unit) = unit(part)? else {
                    // Missing or unknown oneof data uses the shared redacted warning policy.
                    warn_unmodeled("gemini_grpc_part", part);
                    continue;
                };
                let text = matches!(unit, Unit::Text { .. });
                if text && previous_text {
                    self.writer.split_text(&mut out);
                }
                previous_text = text;
                let answers = !matches!(
                    part.data,
                    Some(
                        proto::part::Data::ExecutableCode(_)
                            | proto::part::Data::CodeExecutionResult(_)
                    )
                );
                self.writer.unit(unit, answers, &mut out)?;
            }
        }
        // Enum default is 0 = FINISH_REASON_UNSPECIFIED. The last one wins.
        if candidate.finish_reason != 0 {
            self.last = Some(resp);
        }
        Ok(Flow::More)
    }

    /// Without a provider finish reason the reply did not end; an empty
    /// reply needs a truncating finish reason to explain it.
    fn eof(&mut self, mut out: Out<'id, Completion>) -> Result<Flow, ProviderError> {
        let Some(last) = self.last.take() else {
            return Err(ProviderError::Truncated);
        };
        let finish_reason = last
            .candidates
            .first()
            .and_then(|candidate| super::completion::map_finish_reason(candidate.finish_reason));
        let cut_short = finish_reason
            .as_ref()
            .is_some_and(rig_core::completion::FinishReason::truncated_output);
        if !self.writer.delivered() && !cut_short {
            return Err(ProviderError::Response(
                rig_core::message::EMPTY_RESPONSE_ERROR.to_owned(),
            ));
        }
        self.writer.close(&mut out);
        out.raw(serde_json::to_value(&last)?);
        Ok(out.end(
            Finish::new(super::completion::map_usage(last.usage_metadata.as_ref()))
                .with_optional_reason(finish_reason)
                .with_optional_response_id(Some(last.response_id).filter(|id| !id.is_empty()))
                .with_optional_model(Some(last.model_version).filter(|model| !model.is_empty())),
        ))
    }
}

/// The edge unit of one protobuf part, or `None` for a part with no data.
fn unit(part: &proto::Part) -> Result<Option<Unit>, ProviderError> {
    let signature = encode_signature(&part.thought_signature);
    Ok(Some(match &part.data {
        Some(proto::part::Data::Text(text)) if part.thought => Unit::Thought {
            text: text.clone(),
            signature,
        },
        Some(proto::part::Data::Text(text)) => Unit::Text {
            text: text.clone(),
            signature,
        },
        Some(proto::part::Data::FunctionCall(call)) => Unit::Call {
            id: Some(call.id.clone()).filter(|id| !id.is_empty()),
            name: call.name.clone(),
            args: match call.args.as_ref().map(prost_struct_to_json) {
                Some(Value::Object(args)) => args,
                _ => Map::new(),
            },
            signature,
        },
        Some(data) => {
            // A part rig has no type for keeps Google's own JSON shape.
            let mut native = api::Part {
                thought: part.thought.then_some(true),
                thought_signature: signature,
                ..Default::default()
            };
            match data {
                proto::part::Data::InlineData(blob) => {
                    native.inline_data = Some(api::Blob {
                        mime_type: Some(blob.mime_type.clone()),
                        data: encode_signature(&blob.data),
                        ..Default::default()
                    });
                }
                proto::part::Data::FileData(file) => {
                    native.file_data = Some(api::FileData {
                        mime_type: Some(file.mime_type.clone()),
                        file_uri: Some(file.file_uri.clone()),
                        ..Default::default()
                    });
                }
                proto::part::Data::ExecutableCode(code) => {
                    native.executable_code = Some(api::ExecutableCode {
                        language: Some(code.language.clone().into()),
                        code: Some(code.code.clone()),
                        ..Default::default()
                    });
                }
                proto::part::Data::CodeExecutionResult(result) => {
                    native.code_execution_result = Some(api::CodeExecutionResult {
                        outcome: Some(result.outcome.clone().into()),
                        output: Some(result.output.clone()),
                        ..Default::default()
                    });
                }
                proto::part::Data::FunctionResponse(response) => {
                    native.function_response = Some(api::FunctionResponse {
                        name: Some(response.name.clone()),
                        id: Some(response.id.clone()).filter(|id| !id.is_empty()),
                        response: match response.response.as_ref().map(prost_struct_to_json) {
                            Some(Value::Object(map)) => Some(map),
                            _ => None,
                        },
                        ..Default::default()
                    });
                }
                proto::part::Data::Text(_) | proto::part::Data::FunctionCall(_) => {
                    return Ok(None);
                }
            }
            Unit::Native(NativePart::new(
                api::PART_SCHEMA,
                serde_json::value::to_raw_value(&native)?,
            ))
        }
        None => return Ok(None),
    }))
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests;
