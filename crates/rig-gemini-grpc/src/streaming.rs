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
                    .map(prost_struct_to_json)
                    .unwrap_or_else(|| Value::Object(Map::new()));

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
    let request = super::completion::create_grpc_request(model, completion_request)?;

    let mut grpc_client = client
        .grpc_client()
        .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

    let mut response_stream = grpc_client
        .stream_generate_content(request)
        .await
        .map_err(super::completion::rpc_error)?
        .into_inner();

    // Transport layer: gRPC messages only — a `Status` error is a transport
    // error; classification and policy live in the shared driver.
    let transport = stream! {
        while let Some(item) = response_stream.next().await {
            match item {
                Ok(resp) => yield Ok(resp),
                Err(status) => {
                    yield Err(super::completion::rpc_error(status));
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
                    Some(response.model_version.clone()).filter(|model| !model.is_empty()),
                ),
        )
    })
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use base64::Engine as _;
    use rig_core::message::{Reasoning, ReasoningContent};
    use rig_core::streaming::StreamedAssistantContent;

    fn thought_part(text: &str, signature: &[u8]) -> proto::Part {
        proto::Part {
            data: Some(proto::part::Data::Text(text.to_string())),
            thought: true,
            thought_signature: signature.to_vec(),
            ..Default::default()
        }
    }

    fn response(parts: Vec<proto::Part>, finish_reason: i32) -> proto::GenerateContentResponse {
        proto::GenerateContentResponse {
            candidates: vec![proto::Candidate {
                content: Some(proto::Content {
                    parts,
                    role: "model".to_string(),
                }),
                finish_reason,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    /// Drive protobuf events through the full normalized path and collect the
    /// Reasoning blocks the consumer sees.
    async fn reasoning_blocks(events: Vec<proto::GenerateContentResponse>) -> Vec<Reasoning> {
        let mut stream = stream_from_events(futures::stream::iter(events.into_iter().map(Ok)));
        let mut blocks = Vec::new();
        while let Some(item) = stream.next().await {
            if let StreamedAssistantContent::Reasoning { reasoning, .. } =
                item.expect("stream item should be ok")
            {
                blocks.push(reasoning);
            }
        }
        blocks
    }

    // Streaming parity with the unary conversion (completion.rs
    // `Reasoning::new_with_signature` + base64): a signed thought part must
    // reach the normalized stream as a completed signed Reasoning block that
    // restates the accumulated thought text.
    #[tokio::test]
    async fn signed_thought_part_restates_accumulated_text_with_signature() {
        let signature_bytes = b"opaque-signature".as_slice();
        let events = vec![
            response(vec![thought_part("think1 ", b"")], 0),
            response(
                vec![thought_part("think2", signature_bytes)],
                proto::candidate::FinishReason::Stop as i32,
            ),
        ];

        let blocks = reasoning_blocks(events).await;
        let signed = blocks
            .last()
            .expect("the signed part must yield a Reasoning block");
        assert_eq!(
            signed.content,
            vec![ReasoningContent::Text {
                text: "think1 think2".to_string(),
                // The expected encoding is the unary path's: standard base64
                // over the wire's signature bytes.
                signature: Some(base64::engine::general_purpose::STANDARD.encode(signature_bytes)),
            }]
        );
    }

    // The wire's real signed shape: the signature rides a trailing EMPTY
    // thought part. It must still emit a signed block so the signature
    // survives into chat history (signature-only case).
    #[tokio::test]
    async fn signature_on_empty_trailer_part_still_carries_the_signature() {
        let signature_bytes = b"trailer-signature".as_slice();
        let events = vec![
            response(vec![thought_part("thinking...", b"")], 0),
            response(
                vec![thought_part("", signature_bytes)],
                proto::candidate::FinishReason::Stop as i32,
            ),
        ];

        let blocks = reasoning_blocks(events).await;
        let signed = blocks
            .last()
            .expect("the signed trailer must yield a Reasoning block");
        assert_eq!(
            signed.content,
            vec![ReasoningContent::Text {
                text: "thinking...".to_string(),
                signature: Some(base64::engine::general_purpose::STANDARD.encode(signature_bytes)),
            }]
        );
    }

    // Signature with no thought text anywhere in the stream: the signed block
    // still surfaces (empty text) rather than dropping the signature.
    #[tokio::test]
    async fn signature_without_any_thought_text_still_surfaces() {
        let signature_bytes = b"lone-signature".as_slice();
        let events = vec![response(
            vec![thought_part("", signature_bytes)],
            proto::candidate::FinishReason::Stop as i32,
        )];

        let blocks = reasoning_blocks(events).await;
        let signed = blocks
            .last()
            .expect("a lone signature must yield a Reasoning block");
        assert_eq!(
            signed.content,
            vec![ReasoningContent::Text {
                text: String::new(),
                signature: Some(base64::engine::general_purpose::STANDARD.encode(signature_bytes)),
            }]
        );
    }

    fn function_call_part(name: &str, id: &str) -> proto::Part {
        proto::Part {
            data: Some(proto::part::Data::FunctionCall(proto::FunctionCall {
                name: name.to_string(),
                args: None,
                id: id.to_string(),
            })),
            ..Default::default()
        }
    }

    // Two id-less calls to the same tool in one turn are two distinct calls,
    // correlated by order rather than by the tool name.
    //
    // That is all this pins. The per-stream minter also gives each call its own
    // stream key now, where the fixed `Tool.for_wire_index(0)` key gave both the
    // same one — but no assertion here can tell the two apart: a whole tool call
    // is emitted immediately with a freshly generated `internal_call_id`, so the
    // shared key never collided anything downstream. It was a latent identity
    // bug, not an observable one, and pinning it would mean asserting on
    // internal keys.
    #[tokio::test]
    async fn two_id_less_function_calls_stay_distinct() {
        let events = vec![response(
            vec![
                function_call_part("get_weather", ""),
                function_call_part("get_weather", ""),
            ],
            proto::candidate::FinishReason::Stop as i32,
        )];

        let mut stream = stream_from_events(futures::stream::iter(events.into_iter().map(Ok)));
        let mut correlators = Vec::new();
        while let Some(item) = stream.next().await {
            if let StreamedAssistantContent::ToolCall {
                tool_call,
                internal_call_id,
            } = item.expect("stream item should be ok")
            {
                assert_eq!(tool_call.function.name, "get_weather");
                correlators.push(internal_call_id);
            }
        }

        assert_eq!(correlators.len(), 2, "correlators: {correlators:?}");
        assert_ne!(
            correlators.first(),
            correlators.last(),
            "each call needs its own correlator"
        );
    }

    // ---- #2258 H4: tool-protocol finish reasons must fail the turn ----

    fn failed_response(
        reason: proto::candidate::FinishReason,
        finish_message: Option<&str>,
    ) -> proto::GenerateContentResponse {
        proto::GenerateContentResponse {
            candidates: vec![proto::Candidate {
                content: Some(proto::Content {
                    parts: vec![],
                    role: "model".to_string(),
                }),
                finish_reason: reason as i32,
                finish_message: finish_message.map(str::to_owned),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    struct Drained {
        errors: Vec<String>,
        reached_terminal: bool,
        text: String,
    }

    async fn drain(events: Vec<proto::GenerateContentResponse>) -> Drained {
        let mut stream = stream_from_events(futures::stream::iter(events.into_iter().map(Ok)));
        let mut drained = Drained {
            errors: Vec::new(),
            reached_terminal: false,
            text: String::new(),
        };

        while let Some(item) = stream.next().await {
            match item {
                Ok(StreamedAssistantContent::Final(_)) => drained.reached_terminal = true,
                Ok(StreamedAssistantContent::Text(text)) => drained.text.push_str(&text.text),
                Ok(_) => {}
                Err(error) => drained.errors.push(error.to_string()),
            }
        }

        drained
    }

    // The gRPC surface only set `is_final` on a nonzero finish reason, so an
    // aborted tool protocol read as a completed turn. It must now fail, as
    // the REST surface always has.
    #[tokio::test]
    async fn malformed_function_call_fails_the_stream_with_no_terminal() {
        let drained = drain(vec![failed_response(
            proto::candidate::FinishReason::MalformedFunctionCall,
            Some("could not parse the function call"),
        )])
        .await;

        assert_eq!(drained.errors.len(), 1, "errors: {:?}", drained.errors);
        let error = drained.errors.first().expect("one error");
        assert!(
            error.contains("MALFORMED_FUNCTION_CALL")
                && error.contains("could not parse the function call"),
            "error should name the reason and carry finish_message: {error}"
        );
        assert!(
            !drained.reached_terminal,
            "a failed turn must not synthesize a terminal record"
        );
    }

    #[tokio::test]
    async fn unexpected_and_too_many_tool_calls_also_fail_the_stream() {
        for reason in [
            proto::candidate::FinishReason::UnexpectedToolCall,
            proto::candidate::FinishReason::TooManyToolCalls,
        ] {
            let drained = drain(vec![failed_response(reason, None)]).await;
            assert_eq!(
                drained.errors.len(),
                1,
                "{} should fail the stream",
                reason.as_str_name()
            );
            assert!(!drained.reached_terminal);
        }
    }

    // Everything after the in-band failure is dead: the adapter latches
    // `failed` and reports `is_finished`, so a later genuine terminal cannot
    // dress the aborted turn up as complete.
    #[tokio::test]
    async fn frames_after_a_tool_protocol_failure_are_not_interpreted() {
        let drained = drain(vec![
            failed_response(proto::candidate::FinishReason::MalformedFunctionCall, None),
            response(
                vec![proto::Part {
                    data: Some(proto::part::Data::Text("recovered?".to_string())),
                    ..Default::default()
                }],
                proto::candidate::FinishReason::Stop as i32,
            ),
        ])
        .await;

        assert_eq!(drained.errors.len(), 1, "errors: {:?}", drained.errors);
        assert!(drained.text.is_empty(), "text: {:?}", drained.text);
        assert!(!drained.reached_terminal);
    }

    // Ordinary terminals are untouched by the new gate.
    #[tokio::test]
    async fn non_tool_protocol_finish_reasons_still_complete_the_turn() {
        let drained = drain(vec![response(
            vec![proto::Part {
                data: Some(proto::part::Data::Text("done".to_string())),
                ..Default::default()
            }],
            proto::candidate::FinishReason::Stop as i32,
        )])
        .await;

        assert!(drained.errors.is_empty(), "errors: {:?}", drained.errors);
        assert_eq!(drained.text, "done");
        assert!(drained.reached_terminal);
    }

    // The unary path routes through the same helper, so the two surfaces
    // report an aborted tool protocol with the same message.
    #[test]
    fn unary_and_streaming_report_the_same_tool_protocol_error() {
        let response = failed_response(
            proto::candidate::FinishReason::TooManyToolCalls,
            Some("budget exhausted"),
        );

        let expected = super::super::completion::tool_protocol_finish_reason_error(
            proto::candidate::FinishReason::TooManyToolCalls as i32,
            Some("budget exhausted"),
        )
        .expect("the helper must produce an error")
        .to_string();

        match rig_core::completion::CompletionResponse::try_from(response) {
            Err(err) => assert_eq!(err.to_string(), expected),
            Ok(_) => panic!("the unary path must fail on a tool-protocol finish reason"),
        }
    }

    // The streaming path maps both the initial `stream_generate_content` RPC
    // failure and any per-item iteration error through `rpc_error`. Pin that the
    // mapping preserves the provider's status text and exposes no HTTP status.
    #[test]
    fn stream_rpc_error_preserves_status_text_without_http_status() {
        let status = tonic::Status::unavailable("boom");
        let expected = status.to_string();

        let err = super::super::completion::rpc_error(status);

        assert_eq!(err.provider_response_body(), Some(expected.as_str()));
        assert_eq!(err.provider_response_status(), None);
    }
}
