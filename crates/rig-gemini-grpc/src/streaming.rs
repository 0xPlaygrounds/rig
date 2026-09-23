//! Normalizes Gemini protobuf streams into Rig completion events.
//!
//! ```
//! use rig_gemini_grpc::streaming::stream_from_events;
//!
//! let response = stream_from_events(futures::stream::empty());
//! ```

use async_stream::stream;
use futures::StreamExt;
use serde_json::{Map, Value};

use rig_core::completion::CompletionRequest;
use rig_core::driver::{run_wire_stream, warn_unmodeled};
use rig_core::error::ProviderError;
use rig_core::operation::{AdapterOutput, Completion};
use rig_core::providers::internal::chunk_lifecycle::{ChunkParts, MintedReasoningLifecycle};
use rig_core::providers::internal::wire::{self, TypedEvent, WireEvent};
use rig_core::streaming;
use rig_core::wasm_compat::WasmCompatSend;

use super::Client;
use super::completion::{encode_optional_base64 as encode_signature, prost_struct_to_json};
use super::proto;

/// The Gemini gRPC typed wire as a [`Decoder`](rig_core::wire::Decoder) over
/// protobuf frames: the chunk carrying a finish reason is the terminal, and
/// the per-stream state is the thought block's lifecycle plus the tool-key
/// minter.
struct GrpcAdapter {
    /// Derives signed reasoning boundaries for thought parts without wire IDs.
    reasoning: MintedReasoningLifecycle,
    /// Mints a distinct identity for each call lacking a wire ID.
    tool_ids: streaming::SyntheticIds,
    /// Suppresses further output after a tool-protocol terminal failure.
    failed: bool,
}

impl Default for GrpcAdapter {
    fn default() -> Self {
        Self {
            reasoning: MintedReasoningLifecycle::new(streaming::MintKind::Reasoning),
            tool_ids: streaming::SyntheticIds::tool(),
            failed: false,
        }
    }
}

impl rig_core::wire::Decoder<Completion, proto::GenerateContentResponse> for GrpcAdapter {
    type Event = proto::GenerateContentResponse;

    fn classify(&self, frame: proto::GenerateContentResponse) -> WireEvent<Self::Event> {
        // Tonic handles frame decoding; unknown oneof values are handled per
        // part during interpretation.
        wire::classify_typed_event(TypedEvent::Modeled(frame))
    }

    fn interpret(&mut self, resp: Self::Event, out: &mut AdapterOutput) {
        if self.failed {
            return;
        }

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
            match terminal_record(&resp) {
                Ok(record) => out.final_record(record),
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
}

impl GrpcAdapter {
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
) -> Result<streaming::StreamFinal, serde_json::Error> {
    let usage = super::completion::map_usage(response.usage_metadata.as_ref());
    let finish_reason = response
        .candidates
        .first()
        .and_then(|candidate| super::completion::map_finish_reason(candidate.finish_reason));

    Ok(streaming::StreamFinal::new(
        super::completion::PROVIDER_NAME,
        usage,
        serde_json::to_value(response)?,
    )
    .with_optional_finish_reason(finish_reason)
    .with_optional_response_id(Some(response.response_id.clone()).filter(|id| !id.is_empty()))
    .with_optional_model(Some(response.model_version.clone()).filter(|model| !model.is_empty()))
    .with_reasoning_issuer(super::completion::REASONING_ISSUER))
}

/// Normalizes typed protobuf events through the shared completion driver.
/// No gRPC transport is required; input errors propagate through the stream.
pub fn stream_from_events(
    events: impl futures::Stream<Item = Result<proto::GenerateContentResponse, ProviderError>>
    + WasmCompatSend
    + 'static,
) -> streaming::StreamingCompletionResponse {
    streaming::StreamingCompletionResponse::stream(
        super::completion::PROVIDER_NAME,
        run_wire_stream(events, GrpcAdapter::default()),
    )
    .with_reasoning_issuer(super::completion::REASONING_ISSUER)
}

/// Open a stream normalized to rig's [`streaming::StreamFinal`] terminal
/// record; the adapter maps Gemini's own protobuf terminal onto
/// [`streaming::StreamFinal::raw`].
pub(crate) async fn stream(
    client: Client,
    model: String,
    completion_request: CompletionRequest,
) -> Result<streaming::StreamingCompletionResponse, ProviderError> {
    let request = super::completion::create_grpc_request(&model, completion_request)?;

    let mut grpc_client = client
        .grpc_client()
        .map_err(|e| ProviderError::Provider(e.to_string()))?;

    let mut response_stream = grpc_client
        .stream_generate_content(request)
        .await
        .map_err(|status| super::completion::rpc_error(&status))?
        .into_inner();

    // Stop receiving after a tonic failure; successfully received messages
    // are classified by the shared driver.
    let transport = stream! {
        while let Some(item) = response_stream.next().await {
            match item {
                Ok(resp) => yield Ok(resp),
                Err(status) => {
                    yield Err(super::completion::rpc_error(&status));
                    break;
                }
            }
        }
    };

    Ok(streaming::StreamingCompletionResponse::stream(
        super::completion::PROVIDER_NAME,
        run_wire_stream(transport, GrpcAdapter::default()),
    )
    .with_reasoning_issuer(super::completion::REASONING_ISSUER))
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests;
