// ================================================================
//! Google Gemini gRPC Completion Integration
// ================================================================

/// `gemini-2.5-flash` completion model
pub const GEMINI_2_5_FLASH: &str = "gemini-2.5-flash";
/// `gemini-2.0-flash-lite` completion model
pub const GEMINI_2_0_FLASH_LITE: &str = "gemini-2.0-flash-lite";
/// `gemini-2.0-flash` completion model
pub const GEMINI_2_0_FLASH: &str = "gemini-2.0-flash";

use rig::OneOrMany;
use rig::completion::{self, CompletionError, CompletionRequest};
use rig::message::{self, MimeType, Reasoning};
use rig::telemetry::ProviderResponseExt;
use base64::Engine as _;
use std::convert::TryFrom;

use super::Client;
use super::proto::{self, GenerateContentRequest, GenerateContentResponse};

// =================================================================
// Rig Implementation Types
// =================================================================

#[derive(Clone, Debug)]
pub struct CompletionModel {
    pub(crate) client: Client,
    pub model: String,
}

impl CompletionModel {
    pub fn new(client: Client, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }
}

impl completion::CompletionModel for CompletionModel {
    type Response = GenerateContentResponse;
    type StreamingResponse = super::streaming::StreamingCompletionResponse;
    type Client = super::Client;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model)
    }

    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<GenerateContentResponse>, CompletionError> {
        let request = create_grpc_request(self.model.clone(), completion_request)?;

        let mut grpc_client = self
            .client
            .grpc_client()
            .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

        let response = grpc_client
            .generate_content(request)
            .await
            .map_err(|e| CompletionError::ProviderError(e.to_string()))?
            .into_inner();

        response.try_into()
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<
        rig::streaming::StreamingCompletionResponse<Self::StreamingResponse>,
        CompletionError,
    > {
        super::streaming::stream(self.client.clone(), self.model.clone(), request).await
    }
}

// Helper function to create gRPC request from Rig's CompletionRequest
pub(crate) fn create_grpc_request(
    model: String,
    completion_request: CompletionRequest,
) -> Result<GenerateContentRequest, CompletionError> {
    let mut contents = Vec::new();

    // Convert chat history to gRPC Content messages
    for msg in completion_request.chat_history {
        contents.push(rig_message_to_grpc_content(msg)?);
    }

    // Handle system instruction (preamble)
    let system_instruction = completion_request.preamble.map(|preamble| proto::Content {
        parts: vec![proto::Part {
            data: Some(proto::part::Data::Text(preamble)),
            thought: false,
            thought_signature: Vec::new(),
            part_metadata: None,
        }],
        role: "model".to_string(),
    });

    // Handle generation config
    let generation_config =
        if completion_request.temperature.is_some() || completion_request.max_tokens.is_some() {
            Some(proto::GenerationConfig {
                temperature: completion_request.temperature.map(|t| t as f32),
                max_output_tokens: completion_request.max_tokens.map(|t| t as i32),
                ..Default::default()
            })
        } else {
            None
        };

    // Handle tools (functions)
    let tools = if !completion_request.tools.is_empty() {
        let function_declarations = completion_request
            .tools
            .into_iter()
            .map(|tool| proto::FunctionDeclaration {
                name: tool.name,
                description: tool.description,
                ..Default::default()
            })
            .collect();

        vec![proto::Tool {
            function_declarations,
            code_execution: None,
        }]
    } else {
        vec![]
    };

    Ok(GenerateContentRequest {
        model: format!("models/{}", model),
        contents,
        tools,
        safety_settings: vec![],
        generation_config,
        tool_config: None,
        system_instruction,
        cached_content: String::new(),
    })
}

// Convert Rig message to gRPC Content
fn rig_message_to_grpc_content(msg: message::Message) -> Result<proto::Content, CompletionError> {
    match msg {
        message::Message::User { content } => {
            let parts = content
                .into_iter()
                .map(rig_user_content_to_grpc_part)
                .collect::<Result<Vec<_>, _>>()?;

            Ok(proto::Content {
                parts,
                role: "user".to_string(),
            })
        }
        message::Message::Assistant { content, .. } => {
            let parts = content
                .into_iter()
                .map(rig_assistant_content_to_grpc_part)
                .collect::<Result<Vec<_>, _>>()?;

            Ok(proto::Content {
                parts,
                role: "model".to_string(),
            })
        }
    }
}

// Convert Rig UserContent to gRPC Part
fn rig_user_content_to_grpc_part(
    content: message::UserContent,
) -> Result<proto::Part, CompletionError> {
    match content {
        message::UserContent::Text(message::Text { text }) => Ok(proto::Part {
            data: Some(proto::part::Data::Text(text)),
            thought: false,
            thought_signature: Vec::new(),
            part_metadata: None,
        }),
        message::UserContent::ToolResult(result) => {
            let response_text = match &result.content.first() {
                message::ToolResultContent::Text(t) => t.text.clone(),
                message::ToolResultContent::Image(_) => {
                    return Err(CompletionError::RequestError(
                        "Tool result content must be text".into(),
                    ));
                }
            };

            let result_value: serde_json::Value = serde_json::from_str(&response_text)
                .unwrap_or_else(|_| serde_json::json!(response_text));

            let response_struct =
                json_to_prost_struct(serde_json::json!({ "result": result_value }))?;

            Ok(proto::Part {
                data: Some(proto::part::Data::FunctionResponse(
                    proto::FunctionResponse {
                        name: result.id,
                        response: Some(response_struct),
                        id: result.call_id.unwrap_or_default(),
                    },
                )),
                thought: false,
                thought_signature: Vec::new(),
                part_metadata: None,
            })
        }
        message::UserContent::Image(img) => {
            let Some(media_type) = img.media_type else {
                return Err(CompletionError::RequestError(
                    "Media type for image is required for Gemini".into(),
                ));
            };

            match media_type {
                message::ImageMediaType::JPEG
                | message::ImageMediaType::PNG
                | message::ImageMediaType::WEBP
                | message::ImageMediaType::HEIC
                | message::ImageMediaType::HEIF => {}
                _ => {
                    return Err(CompletionError::RequestError(
                        format!("Unsupported image media type {media_type:?}").into(),
                    ));
                }
            }

            let mime_type = media_type.to_mime_type().to_string();

            let data = match img.data {
                message::DocumentSourceKind::Url(file_uri) => {
                    return Ok(proto::Part {
                        data: Some(proto::part::Data::FileData(proto::FileData {
                            mime_type,
                            file_uri,
                        })),
                        thought: false,
                        thought_signature: Vec::new(),
                        part_metadata: None,
                    });
                }
                message::DocumentSourceKind::Raw(bytes) => bytes,
                message::DocumentSourceKind::Base64(data) | message::DocumentSourceKind::String(data) => {
                    decode_base64_bytes(&data)?
                }
                message::DocumentSourceKind::Unknown => {
                    return Err(CompletionError::RequestError("Image content has no body".into()));
                }
                _ => {
                    return Err(CompletionError::RequestError("Unsupported document source kind".into()));
                }
            };

            Ok(proto::Part {
                data: Some(proto::part::Data::InlineData(proto::Blob { mime_type, data })),
                thought: false,
                thought_signature: Vec::new(),
                part_metadata: None,
            })
        }
        _ => Err(CompletionError::RequestError(
            "Unsupported user content type".into(),
        )),
    }
}

// Convert Rig AssistantContent to gRPC Part
fn rig_assistant_content_to_grpc_part(
    content: message::AssistantContent,
) -> Result<proto::Part, CompletionError> {
    match content {
        message::AssistantContent::Text(message::Text { text }) => Ok(proto::Part {
            data: Some(proto::part::Data::Text(text)),
            thought: false,
            thought_signature: Vec::new(),
            part_metadata: None,
        }),
        message::AssistantContent::ToolCall(tool_call) => {
            let args = json_to_prost_struct(tool_call.function.arguments)?;

            Ok(proto::Part {
                data: Some(proto::part::Data::FunctionCall(proto::FunctionCall {
                    name: tool_call.function.name,
                    args: Some(args),
                    id: tool_call.call_id.unwrap_or(tool_call.id),
                })),
                thought: false,
                thought_signature: decode_optional_base64(tool_call.signature)?,
                part_metadata: None,
            })
        }
        message::AssistantContent::Reasoning(reasoning) => Ok(proto::Part {
            data: Some(proto::part::Data::Text(reasoning.reasoning.join("\n"))),
            thought: true,
            thought_signature: decode_optional_base64(reasoning.signature)?,
            part_metadata: None,
        }),
        _ => Err(CompletionError::RequestError(
            "Unsupported assistant content type".into(),
        )),
    }
}

// Convert gRPC GenerateContentResponse to Rig CompletionResponse
impl TryFrom<GenerateContentResponse> for completion::CompletionResponse<GenerateContentResponse> {
    type Error = CompletionError;

    fn try_from(response: GenerateContentResponse) -> Result<Self, Self::Error> {
        let candidate = response.candidates.first().ok_or_else(|| {
            CompletionError::ResponseError("No response candidates in response".into())
        })?;

        let content_ref = candidate.content.as_ref().ok_or_else(|| {
            CompletionError::ResponseError(format!(
                "Gemini candidate missing content (finish_reason={})",
                candidate.finish_reason
            ))
        })?;

        let mut assistant_contents = Vec::new();

        for part in &content_ref.parts {
            let assistant_content = match &part.data {
                Some(proto::part::Data::Text(text)) => {
                    if part.thought {
                        completion::AssistantContent::Reasoning(Reasoning::new(text).with_signature(
                            encode_optional_base64(&part.thought_signature),
                        ))
                    } else {
                        completion::AssistantContent::text(text)
                    }
                }
                Some(proto::part::Data::InlineData(inline_data)) => {
                    let mime_type = message::MediaType::from_mime_type(&inline_data.mime_type);
                    match mime_type {
                        Some(message::MediaType::Image(media_type)) => {
                            let b64 = base64::engine::general_purpose::STANDARD.encode(&inline_data.data);
                            completion::AssistantContent::image_base64(
                                b64,
                                Some(media_type),
                                Some(message::ImageDetail::default()),
                            )
                        }
                        _ => {
                            return Err(CompletionError::ResponseError(format!(
                                "Unsupported media type {mime_type:?}"
                            )));
                        }
                    }
                }
                Some(proto::part::Data::FunctionCall(function_call)) => {
                    let args = function_call
                        .args
                        .as_ref()
                        .map(prost_struct_to_json)
                        .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));

                    let mut tool_call = message::ToolCall::new(
                        if function_call.id.is_empty() {
                            function_call.name.clone()
                        } else {
                            function_call.id.clone()
                        },
                        message::ToolFunction::new(function_call.name.clone(), args),
                    );

                    if !function_call.id.is_empty() {
                        tool_call = tool_call.with_call_id(function_call.id.clone());
                    }

                    tool_call =
                        tool_call.with_signature(encode_optional_base64(&part.thought_signature));

                    completion::AssistantContent::ToolCall(tool_call)
                }
                _ => {
                    return Err(CompletionError::ResponseError(
                        "Response did not contain a message or tool call".into(),
                    ));
                }
            };

            assistant_contents.push(assistant_content);
        }

        let choice = OneOrMany::many(assistant_contents).map_err(|_| {
            CompletionError::ResponseError(
                "Response contained no message or tool call (empty)".to_owned(),
            )
        })?;

        let usage = response
            .usage_metadata
            .as_ref()
            .map(|usage| completion::Usage {
                input_tokens: usage.prompt_token_count as u64,
                output_tokens: usage.candidates_token_count as u64,
                total_tokens: usage.total_token_count as u64,
            })
            .unwrap_or_default();

        Ok(completion::CompletionResponse {
            choice,
            usage,
            raw_response: response,
        })
    }
}

// Implement ProviderResponseExt for telemetry
impl ProviderResponseExt for GenerateContentResponse {
    type OutputMessage = proto::Candidate;
    type Usage = proto::UsageMetadata;

    fn get_response_id(&self) -> Option<String> {
        if self.response_id.is_empty() {
            None
        } else {
            Some(self.response_id.clone())
        }
    }

    fn get_response_model_name(&self) -> Option<String> {
        if self.model_version.is_empty() {
            None
        } else {
            Some(self.model_version.clone())
        }
    }

    fn get_output_messages(&self) -> Vec<Self::OutputMessage> {
        self.candidates.clone()
    }

    fn get_text_response(&self) -> Option<String> {
        self.candidates.first().and_then(|c| {
            c.content.as_ref().and_then(|content| {
                let text: Vec<String> = content
                    .parts
                    .iter()
                    .filter_map(|part| {
                        if let Some(proto::part::Data::Text(text)) = &part.data {
                            Some(text.clone())
                        } else {
                            None
                        }
                    })
                    .collect();

                if text.is_empty() {
                    None
                } else {
                    Some(text.join("\n"))
                }
            })
        })
    }

    fn get_usage(&self) -> Option<Self::Usage> {
        self.usage_metadata.clone()
    }
}

fn decode_base64_bytes(input: &str) -> Result<Vec<u8>, CompletionError> {
    let data = input.trim();

    // Allow `data:<mime>;base64,<data>` inputs.
    let data = if let Some(rest) = data.strip_prefix("data:") {
        rest.split_once(',').map(|(_, b64)| b64).unwrap_or(data)
    } else {
        data
    };

    let mut last_err: Option<String> = None;

    for engine in [
        &base64::engine::general_purpose::STANDARD,
        &base64::engine::general_purpose::URL_SAFE,
        &base64::engine::general_purpose::STANDARD_NO_PAD,
        &base64::engine::general_purpose::URL_SAFE_NO_PAD,
    ] {
        match engine.decode(data) {
            Ok(bytes) => return Ok(bytes),
            Err(err) => last_err = Some(err.to_string()),
        }
    }

    let err = last_err.unwrap_or_else(|| "unknown base64 decode error".to_string());
    Err(CompletionError::RequestError(format!("Invalid base64 data: {err}").into()))
}

fn decode_optional_base64(sig: Option<String>) -> Result<Vec<u8>, CompletionError> {
    let Some(sig) = sig else { return Ok(Vec::new()); };
    decode_base64_bytes(&sig)
}

fn encode_optional_base64(bytes: &[u8]) -> Option<String> {
    if bytes.is_empty() {
        None
    } else {
        Some(base64::engine::general_purpose::STANDARD.encode(bytes))
    }
}

fn json_to_prost_struct(value: serde_json::Value) -> Result<proto::Struct, CompletionError> {
    match value {
        serde_json::Value::Object(map) => Ok(proto::Struct {
            fields: map
                .into_iter()
                .map(|(k, v)| (k, json_to_prost_value(v)))
                .collect(),
        }),
        _ => Err(CompletionError::RequestError(
            "Expected a JSON object for google.protobuf.Struct".into(),
        )),
    }
}

fn json_to_prost_value(value: serde_json::Value) -> proto::Value {
    match value {
        serde_json::Value::Null => proto::Value {
            kind: Some(proto::value::Kind::NullValue(
                proto::NullValue::NullValue as i32,
            )),
        },
        serde_json::Value::Bool(b) => proto::Value {
            kind: Some(proto::value::Kind::BoolValue(b)),
        },
        serde_json::Value::Number(n) => proto::Value {
            kind: Some(proto::value::Kind::NumberValue(
                n.as_f64().unwrap_or_default(),
            )),
        },
        serde_json::Value::String(s) => proto::Value {
            kind: Some(proto::value::Kind::StringValue(s)),
        },
        serde_json::Value::Array(items) => proto::Value {
            kind: Some(proto::value::Kind::ListValue(proto::ListValue {
                values: items.into_iter().map(json_to_prost_value).collect(),
            })),
        },
        serde_json::Value::Object(map) => proto::Value {
            kind: Some(proto::value::Kind::StructValue(proto::Struct {
                fields: map
                    .into_iter()
                    .map(|(k, v)| (k, json_to_prost_value(v)))
                    .collect(),
            })),
        },
    }
}

fn prost_struct_to_json(st: &proto::Struct) -> serde_json::Value {
    let mut out = serde_json::Map::with_capacity(st.fields.len());
    for (k, v) in &st.fields {
        out.insert(k.clone(), prost_value_to_json(v));
    }
    serde_json::Value::Object(out)
}

fn prost_value_to_json(v: &proto::Value) -> serde_json::Value {
    match &v.kind {
        None | Some(proto::value::Kind::NullValue(_)) => serde_json::Value::Null,
        Some(proto::value::Kind::BoolValue(b)) => serde_json::Value::Bool(*b),
        Some(proto::value::Kind::NumberValue(n)) => serde_json::Number::from_f64(*n)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        Some(proto::value::Kind::StringValue(s)) => serde_json::Value::String(s.clone()),
        Some(proto::value::Kind::StructValue(st)) => prost_struct_to_json(st),
        Some(proto::value::Kind::ListValue(list)) => {
            serde_json::Value::Array(list.values.iter().map(prost_value_to_json).collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_base64_bytes_accepts_url_safe_with_padding() {
        assert!(matches!(
            decode_base64_bytes("_-wgVQA="),
            Ok(bytes) if bytes == vec![0xFF, 0xEC, 0x20, 0x55, 0x00]
        ));
    }

    #[test]
    fn test_decode_base64_bytes_accepts_url_safe_no_pad() {
        assert!(matches!(
            decode_base64_bytes("_-wgVQA"),
            Ok(bytes) if bytes == vec![0xFF, 0xEC, 0x20, 0x55, 0x00]
        ));
    }

    #[test]
    fn test_decode_base64_bytes_accepts_standard_no_pad() {
        assert!(matches!(
            decode_base64_bytes("Zg"),
            Ok(bytes) if bytes == b"f".to_vec()
        ));
    }

    #[test]
    fn test_decode_base64_bytes_accepts_data_uri_prefix() {
        assert!(matches!(
            decode_base64_bytes("data:text/plain;base64,Zm9v"),
            Ok(bytes) if bytes == b"foo".to_vec()
        ));
    }
}
