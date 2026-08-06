// ================================================================
//! Google Gemini gRPC Streaming Integration
// ================================================================

use async_stream::stream;
use base64::Engine as _;
use futures::StreamExt;
use serde_json::{Map, Value};

use rig_core::completion::{CompletionError, CompletionRequest};
use rig_core::providers::internal::adapter::{AdapterOutput, WireAdapter, run_wire_stream};
use rig_core::providers::internal::wire::{self, TypedEvent, WireEvent};
use rig_core::streaming;
use rig_core::wasm_compat::WasmCompatSend;

use super::Client;
use super::GenerateContentResponse;
use super::proto;

pub type StreamingCompletionResponse = GenerateContentResponse;

/// The Gemini gRPC typed wire as a [`WireAdapter`]: the chunk carrying a
/// finish reason is the terminal, and the only per-stream state is the open
/// thinking block's accumulated text (for signed restatement).
#[derive(Default)]
struct GrpcAdapter {
    /// Thought text since the last boundary (signed emission, visible text,
    /// or tool call). The grammar requires a full `Reasoning` block to be
    /// the block's *completed* form, but Gemini attaches
    /// `thought_signature` to a single part — so the adapter restates the
    /// accumulated text, mirroring the REST wire's `thoughtSignature`
    /// handling. Reset on non-thought output to mirror the accumulator's
    /// minted-id boundary.
    thought_buffer: String,
    /// A tool-protocol finish reason ended the turn; later frames are dead —
    /// the provider aborted, and interpreting more output (or a terminal)
    /// would dress the failure up as a completed turn. Mirrors the REST
    /// adapter's identically named latch.
    failed: bool,
}

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
                    match &part.data {
                        Some(proto::part::Data::Text(text)) => {
                            if part.thought {
                                self.thought_buffer.push_str(text);
                                if let Some(signature) = encode_signature(&part.thought_signature) {
                                    // The signature closes the thinking
                                    // block: emit the completed signed
                                    // `Reasoning` — the full accumulated
                                    // text restated with the signature —
                                    // superseding the deltas it restates
                                    // (same base64 encoding as the unary
                                    // path's `Reasoning::new_with_signature`
                                    // conversion). A signature on an empty
                                    // trailer part still emits, so it
                                    // survives into chat history.
                                    out.push(Ok(streaming::RawStreamingChoice::Reasoning {
                                        // Thought parts carry no wire id or
                                        // block boundaries; a per-stream
                                        // constant minted identity merges
                                        // them into one item.
                                        id: rig_core::streaming::StreamPartId::Minted {
                                            kind: rig_core::streaming::MintKind::Reasoning,
                                            index: 0,
                                        },
                                        provider_id: None,
                                        content: rig_core::message::ReasoningContent::Text {
                                            text: std::mem::take(&mut self.thought_buffer),
                                            signature: Some(signature),
                                        },
                                    }));
                                } else if !text.is_empty() {
                                    out.push(Ok(streaming::RawStreamingChoice::ReasoningDelta {
                                        // Thought parts carry no wire id or
                                        // block boundaries; a per-stream
                                        // constant minted identity merges
                                        // them into one item.
                                        id: rig_core::streaming::StreamPartId::Minted {
                                            kind: rig_core::streaming::MintKind::Reasoning,
                                            index: 0,
                                        },
                                        provider_id: None,
                                        reasoning: text.clone(),
                                    }));
                                }
                            } else {
                                // A trailing non-thought part can carry the
                                // signature of the already-closed thought
                                // block; it is lifecycle metadata, not new
                                // reasoning, and must not be dropped
                                // (#2258 B4).
                                if let Some(signature) = encode_signature(&part.thought_signature) {
                                    out.push(Ok(
                                        streaming::RawStreamingChoice::ReasoningSignature {
                                            id: rig_core::streaming::StreamPartId::Minted {
                                                kind: rig_core::streaming::MintKind::Reasoning,
                                                index: 0,
                                            },
                                            signature,
                                        },
                                    ));
                                }
                                // Non-thought output closes the open
                                // reasoning item (accumulator minted-id
                                // boundary).
                                self.thought_buffer.clear();
                                if !text.is_empty() {
                                    out.push(Ok(streaming::RawStreamingChoice::Message(
                                        text.clone(),
                                    )));
                                }
                            }
                        }
                        Some(proto::part::Data::FunctionCall(function_call)) => {
                            // Non-thought output closes the open reasoning
                            // item (accumulator minted-id boundary).
                            self.thought_buffer.clear();
                            let args_json = function_call
                                .args
                                .as_ref()
                                .map(prost_struct_to_json)
                                .unwrap_or_else(|| Value::Object(Map::new()));

                            // The wire's id when present; never the tool
                            // name — a name-as-id would collide two calls to
                            // the same tool in one turn. An id-less call
                            // keys the stream by a minted identity and its
                            // durable id stays absent.
                            let tool_id = if function_call.id.is_empty() {
                                rig_core::streaming::MintKind::Tool.for_wire_index(0)
                            } else {
                                rig_core::streaming::StreamPartId::wire(function_call.id.clone())
                            };

                            let mut tool_call = streaming::RawStreamingToolCall::new(
                                tool_id,
                                function_call.name.clone(),
                                args_json,
                            )
                            .with_signature(encode_signature(&part.thought_signature));

                            if !function_call.id.is_empty() {
                                tool_call = tool_call.with_call_id(function_call.id.clone());
                            }

                            out.push(Ok(streaming::RawStreamingChoice::ToolCall(tool_call)));
                        }
                        None => {
                            // A oneof decoding to `None` is prost's
                            // unknown-variant signal: a part kind this client
                            // does not model. Warn-and-skip, mirroring the
                            // driver's `Unknown` policy at part granularity.
                            tracing::warn!("skipping unrecognized gRPC content part");
                        }
                        Some(_) => {}
                    }
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
        let usage = response
            .usage_metadata
            .as_ref()
            .map(|usage| rig_core::completion::Usage {
                input_tokens: usage.prompt_token_count as u64,
                output_tokens: usage.candidates_token_count as u64,
                total_tokens: usage.total_token_count as u64,
                cached_input_tokens: usage.cached_content_token_count as u64,
                cache_creation_input_tokens: 0,
                tool_use_prompt_tokens: 0,
                reasoning_tokens: 0,
            })
            .unwrap_or_default();
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

fn encode_signature(bytes: &[u8]) -> Option<String> {
    if bytes.is_empty() {
        None
    } else {
        Some(base64::engine::general_purpose::STANDARD.encode(bytes))
    }
}

fn prost_struct_to_json(st: &proto::Struct) -> Value {
    let mut out = Map::with_capacity(st.fields.len());
    for (k, v) in &st.fields {
        out.insert(k.clone(), prost_value_to_json(v));
    }
    Value::Object(out)
}

fn prost_value_to_json(v: &proto::Value) -> Value {
    match &v.kind {
        None | Some(proto::value::Kind::NullValue(_)) => Value::Null,
        Some(proto::value::Kind::BoolValue(b)) => Value::Bool(*b),
        Some(proto::value::Kind::NumberValue(n)) => serde_json::Number::from_f64(*n)
            .map(Value::Number)
            .unwrap_or(Value::Null),
        Some(proto::value::Kind::StringValue(s)) => Value::String(s.clone()),
        Some(proto::value::Kind::StructValue(st)) => prost_struct_to_json(st),
        Some(proto::value::Kind::ListValue(list)) => {
            Value::Array(list.values.iter().map(prost_value_to_json).collect())
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
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
            if let StreamedAssistantContent::Reasoning(reasoning) =
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
