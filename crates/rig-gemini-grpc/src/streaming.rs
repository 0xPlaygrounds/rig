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

/// The Gemini gRPC typed wire as a [`WireAdapter`]: stateless — parts map
/// one-to-one onto grammar events, and the chunk carrying a finish reason is
/// the terminal.
struct GrpcAdapter;

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
                                out.push(Ok(streaming::RawStreamingChoice::ReasoningDelta {
                                    // Thought parts carry no wire id or block
                                    // boundaries; a per-stream constant merges
                                    // them into one item.
                                    id: "reasoning-0".to_string(),
                                    reasoning: text.clone(),
                                }));
                            } else {
                                out.push(Ok(streaming::RawStreamingChoice::Message(text.clone())));
                            }
                        }
                        Some(proto::part::Data::FunctionCall(function_call)) => {
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
    let raw = run_wire_stream(events, GrpcAdapter);
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

    Ok(Box::pin(run_wire_stream(transport, GrpcAdapter)))
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
