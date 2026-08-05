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
        let mut is_final = false;

        if let Some(candidate) = resp.candidates.first() {
            // Enum default is 0 = FINISH_REASON_UNSPECIFIED.
            if candidate.finish_reason != 0 {
                is_final = true;
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
                                        // constant merges them into one item.
                                        id: "reasoning-0".to_string(),
                                        content: rig_core::message::ReasoningContent::Text {
                                            text: std::mem::take(&mut self.thought_buffer),
                                            signature: Some(signature),
                                        },
                                    }));
                                } else if !text.is_empty() {
                                    out.push(Ok(streaming::RawStreamingChoice::ReasoningDelta {
                                        // Thought parts carry no wire id or
                                        // block boundaries; a per-stream
                                        // constant merges them into one item.
                                        id: "reasoning-0".to_string(),
                                        reasoning: text.clone(),
                                    }));
                                }
                            } else {
                                // Non-thought output closes the open
                                // reasoning item (accumulator minted-id
                                // boundary).
                                self.thought_buffer.clear();
                                out.push(Ok(streaming::RawStreamingChoice::Message(text.clone())));
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

                            let tool_id = if function_call.id.is_empty() {
                                function_call.name.clone()
                            } else {
                                function_call.id.clone()
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
