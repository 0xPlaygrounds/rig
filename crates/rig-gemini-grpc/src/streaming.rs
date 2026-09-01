// ================================================================
//! Google Gemini gRPC Streaming Integration
// ================================================================

use async_stream::stream;
use futures::StreamExt;
use serde_json::{Map, Value};

use rig_core::completion::{CompletionError, CompletionRequest};
use rig_core::providers::internal::adapter::{
    AdapterOutput, WireAdapter, run_wire_stream, warn_unmodeled,
};
use rig_core::providers::internal::chunk_lifecycle::{ChunkParts, MintedReasoningLifecycle};
use rig_core::providers::internal::wire::{self, TypedEvent, WireEvent};
use rig_core::streaming;
use rig_core::wasm_compat::WasmCompatSend;

use super::Client;
use super::GenerateContentResponse;
use super::completion::{encode_optional_base64 as encode_signature, prost_struct_to_json};
use super::proto;

pub type StreamingCompletionResponse = GenerateContentResponse;

/// The Gemini gRPC typed wire as a [`WireAdapter`]: the chunk carrying a
/// finish reason is the terminal, and the per-stream state is the thought
/// block's lifecycle plus the tool-key minter.
struct GrpcAdapter {
    /// Owns the constant-key thought lifecycle. Thought parts carry no wire id
    /// and this wire announces no block boundaries, so the shared derivation
    /// emits the signed close and the synthesized boundary end — the same
    /// helper the REST wire uses, so both Gemini surfaces agree.
    reasoning: MintedReasoningLifecycle,
    /// Per-stream minter for id-less tool-call keys — a fresh key per call, so
    /// two id-less calls in one turn never collide on one identity.
    tool_ids: streaming::SyntheticIds,
    /// A tool-protocol finish reason ended the turn; later frames are dead —
    /// the provider aborted, and interpreting more output (or a terminal)
    /// would dress the failure up as a completed turn. Mirrors the REST
    /// adapter's identically named latch.
    failed: bool,
}

impl Default for GrpcAdapter {
    fn default() -> Self {
        Self {
            reasoning: MintedReasoningLifecycle::new(REASONING_ID),
            tool_ids: streaming::SyntheticIds::tool(),
            failed: false,
        }
    }
}

/// Gemini thought parts carry no id or block boundaries; a per-stream constant
/// minted identity keeps every thought delta merging into one item, and the
/// core accumulator's minted-id boundary splits items around other output.
/// Minted, so it can never reach a request.
const REASONING_ID: streaming::StreamPartId =
    streaming::StreamPartId::minted(streaming::MintKind::Reasoning, 0);

impl WireAdapter for GrpcAdapter {
    type Frame = proto::GenerateContentResponse;
    type Event = proto::GenerateContentResponse;
    type Response = StreamingCompletionResponse;

    fn classify(&self, frame: Self::Frame) -> WireEvent<Self::Event> {
        // prost/tonic already deserialized the frame, and a gRPC decode
        // failure surfaces as a transport `Status` error, so every frame is a
        // modeled event here. The wire's unknown-variant signal is per-part
        // (a `part.data` oneof decoding to `None`) — sub-frame granularity,
        // so `interpret` applies the warn-and-skip policy there.
        wire::classify_typed_event(TypedEvent::Modeled(frame))
    }

    fn interpret(&mut self, resp: Self::Event, out: &mut AdapterOutput<Self::Response>) {
        if self.failed {
            return;
        }

        let mut is_final = false;

        if let Some(candidate) = resp.candidates.first() {
            // Enum default is 0 = FINISH_REASON_UNSPECIFIED.
            if candidate.finish_reason != 0 {
                is_final = true;
            }

            // A tool-protocol abort is a failed turn, not a finished one:
            // push the error and stop, exactly as the REST adapter does, so
            // no terminal record follows to report the turn as complete.
            if let Some(err) = super::completion::tool_protocol_finish_reason_error(
                candidate.finish_reason,
                candidate.finish_message.as_deref(),
            ) {
                self.failed = true;
                out.push(Err(err));
                return;
            }

            if let Some(content) = candidate.content.as_ref() {
                for part in &content.parts {
                    let parts = self.interpret_part(part);
                    self.reasoning.emit_chunk(parts, out);
                }
            }
        }

        // Only a chunk carrying a genuine finish reason counts as the provider
        // completing the turn. A stream that reached EOF without one was
        // truncated, and synthesizing a terminal record from the last content
        // chunk (or a default) would report a successful completion for a turn
        // the provider never finished.
        if is_final {
            out.push(Ok(streaming::RawStreamingChoice::FinalResponse(resp)));
        }
    }

    fn finish(&mut self, _out: &mut AdapterOutput<Self::Response>) {
        // EOF without a finish reason is truncation: no terminal record.
    }

    fn is_finished(&self) -> bool {
        // A tool-protocol terminal failure is the wire's own in-band
        // terminal: `interpret` already pushed the `Err` and gates itself on
        // `failed`, so the driver must stop reading rather than drain the
        // rest of the transport.
        self.failed
    }
}

impl GrpcAdapter {
    /// Declare what one protobuf part carried; the shared lifecycle derives
    /// the event sequence, so this adapter holds no boundary bookkeeping of
    /// its own (the REST wire's `interpret_part` has the same shape).
    fn interpret_part(&mut self, part: &proto::Part) -> ChunkParts<StreamingCompletionResponse> {
        match &part.data {
            // A thought part's signature closes the thinking block: the shared
            // accumulator signs the accumulated deltas, using the same base64
            // encoding as the unary path.
            Some(proto::part::Data::Text(text)) if part.thought => ChunkParts {
                reasoning: Some(text.clone()),
                reasoning_signature: encode_signature(&part.thought_signature),
                ..ChunkParts::default()
            },
            // A trailing non-thought part can carry the signature of the
            // already-closed thought block, and one lifecycle end signs the
            // right part in every case (#2258 B4); the text after it closes a
            // still-open block through the derived boundary end.
            Some(proto::part::Data::Text(text)) => ChunkParts {
                reasoning_signature: encode_signature(&part.thought_signature),
                text: Some(text.clone()),
                ..ChunkParts::default()
            },
            Some(proto::part::Data::FunctionCall(function_call)) => {
                let args_json = function_call
                    .args
                    .as_ref()
                    .map_or_else(|| Value::Object(Map::new()), prost_struct_to_json);

                // The wire's id when present; never the tool name — a
                // name-as-id would collide two calls to the same tool in one
                // turn. An id-less call keys the stream by a minted identity,
                // counted up per stream so two id-less calls stay distinct,
                // and its durable id stays absent.
                let key = match streaming::WireId::new(function_call.id.clone()) {
                    Some(wire_id) => streaming::StreamPartId::wire(wire_id.into_string()),
                    None => self.tool_ids.mint(),
                };

                // Gemini is a single-identifier wire: the id above travels as
                // the part identity and `call_id` stays unset — setting both
                // from one id would take the dual-wire arm downstream and
                // fabricate an item id the wire never issued.
                let tool_call = streaming::RawStreamingToolCall::new(
                    key,
                    function_call.name.clone(),
                    args_json,
                )
                // A signature on a function-call part belongs to the
                // call, not to the thought block.
                .with_signature(encode_signature(&part.thought_signature));

                ChunkParts {
                    tool_events: vec![streaming::RawStreamingChoice::ToolCall(tool_call)],
                    ..ChunkParts::default()
                }
            }
            None => {
                // A oneof decoding to `None` is prost's unknown-variant
                // signal: a part kind this client does not model. Warn-and-skip
                // through the shared redaction policy, mirroring the driver's
                // `Unknown` policy at part granularity.
                warn_unmodeled("gemini_grpc_part", part);
                ChunkParts::default()
            }
            Some(_) => ChunkParts::default(),
        }
    }
}

/// Drive already-typed `GenerateContentResponse` events through the full
/// shared pipeline — driver policy, canonical grammar, terminal
/// normalization.
///
/// The events-first conformance seam: the adapter is a pure
/// `(state, event) → events` function, so grammar scenarios feed protobuf
/// events directly with no gRPC transport.
pub fn stream_from_events(
    events: impl futures::Stream<Item = Result<proto::GenerateContentResponse, CompletionError>>
    + WasmCompatSend
    + 'static,
) -> streaming::StreamingCompletionResponse {
    let raw = run_wire_stream(events, GrpcAdapter::default());
    streaming::StreamingCompletionResponse::stream(
        super::completion::PROVIDER_NAME,
        normalize_grpc_stream(raw),
    )
}

/// Open a stream whose terminal record stays Gemini's own protobuf response.
pub(crate) async fn raw_stream(
    client: Client,
    model: String,
    completion_request: CompletionRequest,
) -> Result<streaming::RawStreamingResult<StreamingCompletionResponse>, CompletionError> {
    let request = super::completion::create_grpc_request(&model, completion_request)?;

    let mut grpc_client = client
        .grpc_client()
        .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

    let mut response_stream = grpc_client
        .stream_generate_content(request)
        .await
        .map_err(|status| super::completion::rpc_error(&status))?
        .into_inner();

    // Transport layer: gRPC messages only — a `Status` error is a transport
    // error; classification and policy live in the shared driver.
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

    Ok(Box::pin(run_wire_stream(transport, GrpcAdapter::default())))
}

/// Open a stream normalized to rig's [`streaming::StreamFinal`] terminal
/// record. Delegates to [`raw_stream`] — one RPC either way.
pub(crate) async fn stream(
    client: Client,
    model: String,
    completion_request: CompletionRequest,
) -> Result<streaming::StreamingCompletionResponse, CompletionError> {
    let raw = raw_stream(client, model, completion_request).await?;

    Ok(streaming::StreamingCompletionResponse::stream(
        super::completion::PROVIDER_NAME,
        normalize_grpc_stream(raw),
    ))
}

/// Normalize the provider-native terminal record into rig's
/// [`streaming::StreamFinal`].
fn normalize_grpc_stream(
    raw: streaming::RawStreamingResult<StreamingCompletionResponse>,
) -> streaming::StreamingResult {
    streaming::normalize_stream(raw, |response| {
        let usage = super::completion::map_usage(response.usage_metadata.as_ref());
        let finish_reason = response
            .candidates
            .first()
            .and_then(|candidate| super::completion::map_finish_reason(candidate.finish_reason));

        Ok(
            streaming::StreamFinal::new(super::completion::PROVIDER_NAME, usage)
                .with_optional_finish_reason(finish_reason)
                .with_optional_response_id(
                    Some(response.response_id.clone()).filter(|id| !id.is_empty()),
                )
                .with_optional_model(
                    Some(response.model_version).filter(|model| !model.is_empty()),
                ),
        )
    })
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests;
