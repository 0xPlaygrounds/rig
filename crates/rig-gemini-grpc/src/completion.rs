// ================================================================
//! Google Gemini gRPC Completion Integration
// ================================================================

/// `gemini-2.5-flash` completion model
pub const GEMINI_2_5_FLASH: &str = "gemini-2.5-flash";
/// `gemini-2.0-flash-lite` completion model
pub const GEMINI_2_0_FLASH_LITE: &str = "gemini-2.0-flash-lite";
/// `gemini-2.0-flash` completion model
pub const GEMINI_2_0_FLASH: &str = "gemini-2.0-flash";

use base64::Engine as _;
use rig_core::completion::{self, CompletionError, CompletionRequest};
use rig_core::message::{self, MimeType, Reasoning};
use rig_core::providers::gemini::completion::attach_trailing_signature;
use rig_core::providers::gemini::completion::gemini_api_types::{
    Schema as GeminiSchema, map_google_finish_reason, tool_parameters_to_schema,
};
use rig_core::telemetry::ProviderResponseExt;
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

/// Stable descriptor name reported on normalized responses from this provider.
pub const PROVIDER_NAME: &str = "gemini-grpc";

/// Map Gemini's protobuf `finishReason` onto rig's normalized vocabulary.
///
/// The wire value is a prost enum discriminant; `as_str_name` recovers the
/// SCREAMING_SNAKE proto spelling the shared Google table keys on, and a
/// discriminant this proto does not model keeps its numeric identity so a
/// reason Google adds later surfaces rather than reading as a natural stop.
pub fn map_finish_reason(reason: i32) -> Option<completion::FinishReason> {
    use proto::candidate::FinishReason as Wire;

    let Ok(reason) = Wire::try_from(reason) else {
        return Some(completion::FinishReason::Other(format!(
            "FINISH_REASON_{reason}"
        )));
    };

    map_google_finish_reason(reason.as_str_name())
}

/// Turn a tool-protocol terminal `finishReason` into an error, mirroring the
/// REST wire's `function_call_finish_reason_error`.
///
/// These reasons mean the turn ABORTED inside the tool protocol: the model
/// emitted a call the API could not parse, called a tool that was not
/// offered, or exceeded the per-turn call budget. The candidate that carries
/// them has no usable tool call, so reporting the turn as merely "finished
/// for some other reason" lets an agent loop read an aborted turn as a
/// complete one. The REST surface has always failed here; the gRPC surface
/// must not diverge.
///
/// Only the reasons this proto models are matched — REST's
/// `MISSING_THOUGHT_SIGNATURE` / `MALFORMED_RESPONSE` have no protobuf
/// discriminant in `v1beta`, so an unmapped value cannot masquerade as one.
pub fn tool_protocol_finish_reason_error(
    reason: i32,
    finish_message: Option<&str>,
) -> Option<CompletionError> {
    use proto::candidate::FinishReason as Wire;

    let reason = Wire::try_from(reason).ok()?;
    match reason {
        Wire::MalformedFunctionCall | Wire::UnexpectedToolCall | Wire::TooManyToolCalls => {
            let message = finish_message.unwrap_or("no finish message provided");
            Some(CompletionError::ResponseError(format!(
                "Gemini stopped with finish_reason={}: {message}",
                reason.as_str_name()
            )))
        }
        _ => None,
    }
}

impl CompletionModel {
    /// Execute a completion and return Gemini's own protobuf response.
    ///
    /// This is the escape hatch for fields rig does not normalize;
    /// [`completion::CompletionModel::completion`] calls it and maps the
    /// result, so there is exactly one RPC either way.
    pub async fn raw_completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<GenerateContentResponse, CompletionError> {
        let request = create_grpc_request(&self.model, completion_request)?;

        let mut grpc_client = self
            .client
            .grpc_client()
            .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

        let response = grpc_client
            .generate_content(request)
            .await
            .map_err(|status| rpc_error(&status))?
            .into_inner();

        Ok(response)
    }

    /// Open a stream whose terminal record stays Gemini's own protobuf
    /// response.
    pub async fn raw_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<
        rig_core::streaming::RawStreamingResult<super::streaming::StreamingCompletionResponse>,
        CompletionError,
    > {
        super::streaming::raw_stream(self.client.clone(), self.model.clone(), request).await
    }
}

impl completion::CompletionModel for CompletionModel {
    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse, CompletionError> {
        // Capture before `try_into` consumes the raw value.
        let raw = self.raw_completion(completion_request).await?;
        let captured = serde_json::to_value(&raw)?;
        let response: completion::CompletionResponse = raw.try_into()?;
        Ok(response.with_raw(captured))
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<rig_core::streaming::StreamingCompletionResponse, CompletionError> {
        super::streaming::stream(self.client.clone(), self.model.clone(), request).await
    }
}

/// Build a non-thought `proto::Part` around the given data payload.
pub(crate) fn data_part(data: proto::part::Data) -> proto::Part {
    proto::Part {
        data: Some(data),
        thought: false,
        thought_signature: Vec::new(),
        part_metadata: None,
    }
}

/// Build a plain (non-thought) text `proto::Part`.
pub(crate) fn text_part(text: String) -> proto::Part {
    data_part(proto::part::Data::Text(text))
}

// Map a failed gRPC call into a `CompletionError` that preserves the provider's
// error payload verbatim. gRPC is a non-HTTP transport, so there is no
// `http::StatusCode`; the body is preserved via `from_provider_body` (status:
// None) rather than a Rig-prefixed `ProviderError` diagnostic. Note: tonic does
// not distinguish a server-returned gRPC error from a transport/connection
// failure, so a pure connection error is also preserved here rather than gated
// out as a Rig diagnostic the way Bedrock's typed service errors are.
pub(crate) fn rpc_error(status: &tonic::Status) -> CompletionError {
    CompletionError::from_provider_body(status.to_string())
}

// Helper function to create gRPC request from Rig's CompletionRequest
pub(crate) fn create_grpc_request(
    model: &str,
    completion_request: CompletionRequest,
) -> Result<GenerateContentRequest, CompletionError> {
    let CompletionRequest {
        model: _,
        chat_history,
        documents: _,
        tools,
        temperature,
        max_tokens,
        tool_choice: _,
        additional_params: _,
        output_schema: _,
        record_telemetry_content: _,
    } = completion_request;

    let (history_system, mut chat_history) = split_system_messages_from_history(chat_history);
    // functionResponse.name keys the replay: cross-provider ingested
    // results arrive with an empty name and their call carries it.
    rig_core::providers::internal::resolve_empty_tool_result_names(&mut chat_history);
    let mut contents = Vec::new();

    // Convert chat history to gRPC Content messages
    for msg in chat_history {
        contents.push(rig_message_to_grpc_content(msg)?);
    }

    let mut system_parts = Vec::new();
    for content in history_system {
        if !content.is_empty() {
            system_parts.push(text_part(content));
        }
    }
    let system_instruction = if system_parts.is_empty() {
        None
    } else {
        Some(proto::Content {
            parts: system_parts,
            role: "model".to_string(),
        })
    };

    // Handle generation config
    let generation_config = if temperature.is_some() || max_tokens.is_some() {
        Some(proto::GenerationConfig {
            temperature: temperature.map(|t| t as f32),
            max_output_tokens: max_tokens.map(|t| t as i32),
            ..Default::default()
        })
    } else {
        None
    };

    // Handle tools (functions)
    let tools = if !tools.is_empty() {
        let function_declarations = tools
            .into_iter()
            .map(|tool| {
                Ok(proto::FunctionDeclaration {
                    name: tool.name,
                    description: tool.description,
                    parameters: tool_parameters_to_proto_schema(&tool.parameters)?,
                    ..Default::default()
                })
            })
            .collect::<Result<Vec<_>, CompletionError>>()?;

        vec![proto::Tool {
            function_declarations,
            code_execution: None,
        }]
    } else {
        vec![]
    };

    Ok(GenerateContentRequest {
        model: format!("models/{model}"),
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
        message::Message::System { .. } => Err(CompletionError::RequestError(
            "System messages must be sent via Gemini gRPC system_instruction".into(),
        )),
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

use rig_core::providers::gemini::completion::split_system_messages_from_history;

// Convert Rig UserContent to gRPC Part
fn rig_user_content_to_grpc_part(
    content: message::UserContent,
) -> Result<proto::Part, CompletionError> {
    match content {
        message::UserContent::Text(message::Text { text, .. }) => Ok(text_part(text)),
        message::UserContent::ToolResult(result) => {
            let mut values = result
                .content
                .into_iter()
                .map(|content| match content {
                    message::ToolResultContent::Text(t) => Ok(serde_json::Value::String(t.text)),
                    message::ToolResultContent::Json { value } => Ok(value),
                    message::ToolResultContent::Image(_) => Err(CompletionError::RequestError(
                        "Gemini gRPC does not support images in tool results".into(),
                    )),
                })
                .collect::<Result<Vec<_>, _>>()?;
            let result_value = if values.len() == 1 {
                values.remove(0)
            } else {
                serde_json::Value::Array(values)
            };

            let response_struct =
                json_to_prost_struct(serde_json::json!({ "result": result_value }))?;

            // `FunctionResponse.name` is the executed function's name —
            // required data on the result. Only a provider-issued id may
            // travel back on the wire (the proto field is optional-empty).
            Ok(data_part(proto::part::Data::FunctionResponse(
                proto::FunctionResponse {
                    name: result.name,
                    response: Some(response_struct),
                    id: result
                        .provider
                        .map(|provider| provider.call_id)
                        .unwrap_or_default(),
                },
            )))
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
                    return Ok(data_part(proto::part::Data::FileData(proto::FileData {
                        mime_type,
                        file_uri,
                    })));
                }
                message::DocumentSourceKind::Raw(bytes) => bytes,
                message::DocumentSourceKind::Base64(data)
                | message::DocumentSourceKind::String(data) => decode_base64_bytes(&data)?,
                message::DocumentSourceKind::Unknown => {
                    return Err(CompletionError::RequestError(
                        "Image content has no body".into(),
                    ));
                }
                _ => {
                    return Err(CompletionError::RequestError(
                        "Unsupported document source kind".into(),
                    ));
                }
            };

            Ok(data_part(proto::part::Data::InlineData(proto::Blob {
                mime_type,
                data,
            })))
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
        message::AssistantContent::Text(message::Text { text, .. }) => Ok(text_part(text)),
        message::AssistantContent::ToolCall(tool_call) => {
            let args = json_to_prost_struct(tool_call.function.arguments)?;

            Ok(proto::Part {
                thought_signature: decode_optional_base64(tool_call.signature)?,
                ..data_part(proto::part::Data::FunctionCall(proto::FunctionCall {
                    name: tool_call.function.name,
                    args: Some(args),
                    // Only a provider-issued id may travel back on the
                    // wire; minted correlation handles stay internal.
                    id: tool_call
                        .provider
                        .map(|provider| provider.call_id)
                        .unwrap_or_default(),
                }))
            })
        }
        message::AssistantContent::Reasoning(reasoning) => Ok(proto::Part {
            data: Some(proto::part::Data::Text(reasoning.display_text())),
            thought: true,
            thought_signature: decode_optional_base64(
                reasoning
                    .first_signature()
                    .map(std::string::ToString::to_string),
            )?,
            part_metadata: None,
        }),
        _ => Err(CompletionError::RequestError(
            "Unsupported assistant content type".into(),
        )),
    }
}

// Convert gRPC GenerateContentResponse to Rig CompletionResponse
impl TryFrom<GenerateContentResponse> for completion::CompletionResponse {
    type Error = CompletionError;

    fn try_from(response: GenerateContentResponse) -> Result<Self, Self::Error> {
        let candidate = response.candidates.first().ok_or_else(|| {
            CompletionError::ResponseError("No response candidates in response".into())
        })?;

        // Same helper (and therefore the same message) as the streaming path,
        // so a tool-protocol abort reads identically on both surfaces.
        if let Some(err) = tool_protocol_finish_reason_error(
            candidate.finish_reason,
            candidate.finish_message.as_deref(),
        ) {
            return Err(err);
        }

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
                        completion::AssistantContent::Reasoning(Reasoning::new_with_signature(
                            text,
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
                            let b64 =
                                base64::engine::general_purpose::STANDARD.encode(&inline_data.data);
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
                    let args = function_call.args.as_ref().map_or(
                        serde_json::Value::Object(serde_json::Map::new()),
                        prost_struct_to_json,
                    );

                    // An id-less call mints its correlation handle —
                    // never name-as-id, which collides two same-tool calls.
                    let tool_call = message::ToolCall::from_wire(
                        function_call.id.clone(),
                        message::ToolFunction::new(function_call.name.clone(), args),
                    )
                    .with_signature(encode_optional_base64(&part.thought_signature));

                    completion::AssistantContent::ToolCall(tool_call)
                }
                _ => {
                    return Err(CompletionError::ResponseError(
                        "Response did not contain a message or tool call".into(),
                    ));
                }
            };

            assistant_contents.push(assistant_content);

            // The wire hangs a `thoughtSignature` on a trailing part carrying
            // no `thought` flag, and this crate's own streaming adapter keeps
            // it (`streaming.rs`, the non-thought text arm) while this mapper
            // dropped it — the same blocking/streaming asymmetry the REST wire
            // had. One shared rule places it on both transports.
            if !part.thought
                && matches!(part.data, Some(proto::part::Data::Text(_)))
                && let Some(signature) = encode_optional_base64(&part.thought_signature)
            {
                attach_trailing_signature(&mut assistant_contents, signature);
            }
        }

        let choice = rig_core::message::require_non_empty_response(assistant_contents)?;

        let usage = map_usage(response.usage_metadata.as_ref());

        let finish_reason = response
            .candidates
            .first()
            .and_then(|candidate| map_finish_reason(candidate.finish_reason));
        let model = Some(response.model_version.clone()).filter(|model| !model.is_empty());
        Ok(
            completion::CompletionResponse::new(choice, usage, PROVIDER_NAME)
                .with_optional_finish_reason(finish_reason)
                .with_optional_response_id(
                    Some(response.response_id.clone()).filter(|id| !id.is_empty()),
                )
                .with_optional_model(model),
        )
    }
}

// Implement ProviderResponseExt for telemetry
impl ProviderResponseExt for GenerateContentResponse {
    type Usage = proto::UsageMetadata;

    fn response_id(&self) -> Option<&str> {
        if self.response_id.is_empty() {
            None
        } else {
            Some(self.response_id.as_str())
        }
    }

    fn response_model_name(&self) -> Option<&str> {
        if self.model_version.is_empty() {
            None
        } else {
            Some(self.model_version.as_str())
        }
    }

    fn text_response(&self) -> Option<String> {
        self.candidates.first().and_then(|c| {
            c.content.as_ref().and_then(|content| {
                let text: Vec<String> = content
                    .parts
                    .iter()
                    // `thought` marks the model's chain-of-thought, which the
                    // completion mapper above routes to `Reasoning`. A reader
                    // that wants the response *text* must skip it, or it
                    // reports reasoning as the answer — the same defect the
                    // REST wire carried.
                    .filter(|part| !part.thought)
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

    fn usage(&self) -> Option<Self::Usage> {
        self.usage_metadata
    }
}

fn decode_base64_bytes(input: &str) -> Result<Vec<u8>, CompletionError> {
    let data = input.trim();

    // Allow `data:<mime>;base64,<data>` inputs.
    let data = if let Some(rest) = data.strip_prefix("data:") {
        rest.split_once(',').map_or(data, |(_, b64)| b64)
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
    Err(CompletionError::RequestError(
        format!("Invalid base64 data: {err}").into(),
    ))
}

fn decode_optional_base64(sig: Option<String>) -> Result<Vec<u8>, CompletionError> {
    let Some(sig) = sig else {
        return Ok(Vec::new());
    };
    decode_base64_bytes(&sig)
}

/// Map Gemini's `UsageMetadata` onto rig's normalized `Usage`.
///
/// Known gap (unchanged here): `tool_use_prompt_token_count` and
/// `thoughts_token_count` are not yet surfaced, so `tool_use_prompt_tokens`
/// and `reasoning_tokens` read as 0.
pub(crate) fn map_usage(usage: Option<&proto::UsageMetadata>) -> completion::Usage {
    usage
        .map(|usage| completion::Usage {
            input_tokens: usage.prompt_token_count as u64,
            output_tokens: usage.candidates_token_count as u64,
            total_tokens: usage.total_token_count as u64,
            cached_input_tokens: usage.cached_content_token_count as u64,
            cache_creation_input_tokens: 0,
            tool_use_prompt_tokens: 0,
            reasoning_tokens: 0,
        })
        .unwrap_or_default()
}

pub(crate) fn encode_optional_base64(bytes: &[u8]) -> Option<String> {
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

pub(crate) fn prost_struct_to_json(st: &proto::Struct) -> serde_json::Value {
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
            .map_or(serde_json::Value::Null, serde_json::Value::Number),
        Some(proto::value::Kind::StringValue(s)) => serde_json::Value::String(s.clone()),
        Some(proto::value::Kind::StructValue(st)) => prost_struct_to_json(st),
        Some(proto::value::Kind::ListValue(list)) => {
            serde_json::Value::Array(list.values.iter().map(prost_value_to_json).collect())
        }
    }
}

// Convert the JSON Schema carried by `ToolDefinition.parameters` into the typed
// `proto::Schema` expected by `FunctionDeclaration.parameters`.
//
// Without this, every tool was sent to Gemini with `parameters = None`, which
// caused the model to invoke tools with no argument shape (issue #1710).
//
// An empty object schema (`{"type": "object", "properties": {}}`, the default
// when a tool takes no arguments) is mapped to `None` rather than a vacuous
// schema, matching the convention used by `rig-core::providers::gemini`.
fn tool_parameters_to_proto_schema(
    value: &serde_json::Value,
) -> Result<Option<proto::Schema>, CompletionError> {
    tool_parameters_to_schema(value.clone()).map(|schema| schema.map(gemini_schema_to_proto_schema))
}

fn gemini_schema_to_proto_schema(schema: GeminiSchema) -> proto::Schema {
    proto::Schema {
        r#type: json_type_to_proto_type(&schema.r#type) as i32,
        format: schema.format.unwrap_or_default(),
        description: schema.description.unwrap_or_default(),
        nullable: schema.nullable.unwrap_or(false),
        r#enum: schema.r#enum.unwrap_or_default(),
        items: schema
            .items
            .map(|items| Box::new(gemini_schema_to_proto_schema(*items))),
        properties: schema
            .properties
            .unwrap_or_default()
            .into_iter()
            .map(|(name, schema)| (name, gemini_schema_to_proto_schema(schema)))
            .collect(),
        required: schema.required.unwrap_or_default(),
    }
}

fn json_type_to_proto_type(t: &str) -> proto::Type {
    match t {
        "string" => proto::Type::String,
        "number" => proto::Type::Number,
        "integer" => proto::Type::Integer,
        "boolean" => proto::Type::Boolean,
        "array" => proto::Type::Array,
        "object" => proto::Type::Object,
        "null" => proto::Type::Null,
        _ => proto::Type::Unspecified,
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests;
