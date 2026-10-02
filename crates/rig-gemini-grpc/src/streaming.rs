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
use serde_json::{Map, Value};

use rig_core::driver::warn_unmodeled;
use rig_core::error::ProviderError;
use rig_core::message::{self, CallId, MimeType, ToolCall, ToolFunction, ToolName};
use rig_core::operation::{Completion, Finish};
use rig_core::wire::{Flow, Out, WireEvent};

use super::completion::{encode_optional_base64 as encode_signature, prost_struct_to_json};
use super::proto;

/// The Gemini gRPC wire's decoder over `GenerateContentResponse`s: a unary
/// reply is one of them, a stream sends several. Like the REST wire, the
/// reply ends at EOF once a finish reason arrived, since a hosted-tool round
/// can report one before more content.
#[cfg(any())]
pub struct GrpcAdapter<'id> {
    /// Thought boundaries inferred from content transitions and signatures.
    thoughts: Thoughts<'id>,
    /// The answer text part text chunks extend.
    text: Option<TextPart<'id>>,
    /// The latest response carrying a finish reason: the reply's end and
    /// its raw record.
    last: Option<proto::GenerateContentResponse>,
    /// At least one part mapped to assistant content.
    delivered: bool,
}

#[cfg(any())]
impl Default for GrpcAdapter<'_> {
    fn default() -> Self {
        Self {
            thoughts: Thoughts::new(),
            text: None,
            last: None,
            delivered: false,
        }
    }
}

#[cfg(any())]
impl<'id> rig_core::wire::Decoder<'id, Completion, proto::GenerateContentResponse>
    for GrpcAdapter<'id>
{
    type Event = proto::GenerateContentResponse;

    fn classify(&self, frame: proto::GenerateContentResponse) -> WireEvent<Self::Event> {
        // Tonic handles frame decoding; unknown oneof values are handled per
        // part.
        WireEvent::Known(frame)
    }

    #[cfg(any())]
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
                let text = !part.thought && matches!(part.data, Some(proto::part::Data::Text(_)));
                if text && previous_text {
                    out.close_open_text(&mut self.text);
                }
                previous_text = text;
                self.interpret_part(part, &mut out)?;
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
    #[cfg(any())]
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
        if !self.delivered && !cut_short {
            return Err(ProviderError::Response(
                rig_core::message::EMPTY_RESPONSE_ERROR.to_owned(),
            ));
        }
        out.close_open_text(&mut self.text);
        self.thoughts.close(&mut out, None);
        out.raw(serde_json::to_value(&last)?);
        Ok(out.end(Finish {
            usage: super::completion::map_usage(last.usage_metadata.as_ref()),
            reason: finish_reason,
            response_id: Some(last.response_id),
            model: Some(last.model_version),
            ..Finish::default()
        }))
    }
}

#[cfg(any())]
impl<'id> GrpcAdapter<'id> {
    /// Write one protobuf part.
    #[cfg(any())]
    fn interpret_part(
        &mut self,
        part: &proto::Part,
        out: &mut Out<'id, Completion>,
    ) -> Result<(), ProviderError> {
        match &part.data {
            // A thought part's signature closes the thinking block, using the
            // same base64 encoding the request side decodes.
            Some(proto::part::Data::Text(text)) if part.thought => {
                self.delivered = true;
                if !text.is_empty() {
                    out.close_open_text(&mut self.text);
                }
                self.thoughts.fragment(out, text);
                if let Some(signature) = encode_signature(&part.thought_signature) {
                    self.thoughts.signature(out, signature);
                }
            }
            // A signature on answer text returns on that text part; a signed
            // part ends there.
            Some(proto::part::Data::Text(text)) => {
                self.delivered = true;
                let signature = encode_signature(&part.thought_signature);
                let signed = signature.is_some();
                if signed && text.is_empty() {
                    out.close_open_text(&mut self.text);
                }
                let params = signature.and_then(|signature| {
                    rig_core::providers::gemini::text_signature_extras(
                        rig_core::providers::gemini::GEMINI_TEXT_EXTRAS_KEY,
                        signature,
                    )
                });
                if !text.is_empty() || params.is_some() {
                    self.thoughts.boundary();
                    let part = out.extend_text(&mut self.text, text);
                    if let Some(params) = params {
                        out.text_params(part, params);
                    }
                }
                if signed {
                    out.close_open_text(&mut self.text);
                }
            }
            Some(proto::part::Data::FunctionCall(function_call)) => {
                self.delivered = true;
                self.thoughts.boundary();
                out.close_open_text(&mut self.text);
                let name = ToolName::new(function_call.name.clone()).map_err(|error| {
                    ProviderError::Response(format!("Gemini returned a function call: {error}"))
                })?;
                let arguments = function_call
                    .args
                    .as_ref()
                    .map_or_else(|| Value::Object(Map::new()), prost_struct_to_json);
                // Rig issues an id for a call the provider sent without one.
                out.tool_call(
                    ToolCall::new(
                        CallId::from_wire(function_call.id.clone()),
                        ToolFunction::new(name, arguments),
                    )
                    // A signature on a function-call part belongs to the call.
                    .with_signature(encode_signature(&part.thought_signature)),
                )?;
            }
            Some(proto::part::Data::InlineData(inline_data)) => {
                self.delivered = true;
                self.thoughts.boundary();
                out.close_open_text(&mut self.text);
                let media_type = message::MediaType::from_mime_type(&inline_data.mime_type);
                let Some(message::MediaType::Image(media_type)) = media_type else {
                    return Err(ProviderError::Response(format!(
                        "Unsupported media type {media_type:?}"
                    )));
                };
                out.content(message::AssistantContent::image_base64(
                    base64::engine::general_purpose::STANDARD.encode(&inline_data.data),
                    Some(media_type),
                    Some(message::ImageDetail::default()),
                ))?;
            }
            None => {
                // Missing or unknown oneof data uses the shared redacted warning policy.
                warn_unmodeled("gemini_grpc_part", part);
            }
            Some(_) => {}
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
#[cfg(any())]
mod tests;
