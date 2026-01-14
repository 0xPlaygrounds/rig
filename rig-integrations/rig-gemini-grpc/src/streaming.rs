// ================================================================
//! Google Gemini gRPC Streaming Integration
// ================================================================

use async_stream::stream;
use base64::Engine as _;
use futures::StreamExt;
use serde_json::{Map, Value};

use rig::completion::{CompletionError, CompletionRequest};
use rig::streaming;

use super::Client;
use super::GenerateContentResponse;
use super::proto;

pub type StreamingCompletionResponse = GenerateContentResponse;

pub(crate) async fn stream(
    client: Client,
    model: String,
    completion_request: CompletionRequest,
) -> Result<streaming::StreamingCompletionResponse<StreamingCompletionResponse>, CompletionError> {
    let request = super::completion::create_grpc_request(model, completion_request)?;

    let mut grpc_client = client
        .grpc_client()
        .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

    let mut response_stream = grpc_client
        .stream_generate_content(request)
        .await
        .map_err(|e| CompletionError::ProviderError(e.to_string()))?
        .into_inner();

    let stream = stream! {
        let mut last_resp: Option<StreamingCompletionResponse> = None;
        let mut final_resp: Option<StreamingCompletionResponse> = None;

        while let Some(item) = response_stream.next().await {
            match item {
                Ok(resp) => {
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
                                            yield Ok(streaming::RawStreamingChoice::ReasoningDelta {
                                                id: None,
                                                reasoning: text.clone(),
                                            });
                                        } else {
                                            yield Ok(streaming::RawStreamingChoice::Message(text.clone()));
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

                                        yield Ok(streaming::RawStreamingChoice::ToolCall(tool_call));
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }

                    if is_final {
                        final_resp = Some(resp);
                        break;
                    } else {
                        last_resp = Some(resp);
                    }
                }
                Err(status) => {
                    yield Err(CompletionError::ProviderError(status.to_string()));
                    return;
                }
            }
        }

        let resp = final_resp.or(last_resp).unwrap_or_default();
        yield Ok(streaming::RawStreamingChoice::FinalResponse(resp));
    };

    Ok(streaming::StreamingCompletionResponse::stream(Box::pin(
        stream,
    )))
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
