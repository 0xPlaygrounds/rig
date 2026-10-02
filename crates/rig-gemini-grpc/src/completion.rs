//! The Gemini `GenerateContent` completion wire over gRPC.
//!
//! ```no_run
//! use rig_core::Model;
//! use rig_gemini_grpc::{GeminiGrpc, completion::{GEMINI_2_5_FLASH, GenerateContent}};
//!
//! # async fn example() -> Result<(), rig_gemini_grpc::GeminiGrpcError> {
//! let model = GeminiGrpc::new("API_KEY").await?.completion(GEMINI_2_5_FLASH);
//! # let _ = model;
//! # Ok(())
//! # }
//! ```

/// `gemini-2.5-flash` completion model
pub const GEMINI_2_5_FLASH: &str = "gemini-2.5-flash";
/// `gemini-2.0-flash-lite` completion model
pub const GEMINI_2_0_FLASH_LITE: &str = "gemini-2.0-flash-lite";
/// `gemini-2.0-flash` completion model
pub const GEMINI_2_0_FLASH: &str = "gemini-2.0-flash";

use base64::Engine as _;
use futures::StreamExt;
use rig_core::completion::CompletionRequest;
use rig_core::driver::{Exchange, Opened, Opening, Transport};
use rig_core::error::EncodeError;
use rig_core::error::ProviderError;
use rig_core::message;
use rig_core::operation::Completion;
use rig_core::providers::gemini::completion::gemini_api_types::{
    Blob, Content, FileData, Part, PartKind, Role, Schema as GeminiSchema, UsageMetadata,
    tool_parameters_to_schema,
};
use rig_core::providers::gemini::completion::{contents, split_system_messages_from_history};
use rig_core::wire::{Descriptor, Mode, Wire};

use super::GeminiGrpc;
use super::proto::{self, GenerateContentRequest, GenerateContentResponse};

/// The `GenerateContent` endpoint for one model: `GenerateContent` for a
/// unary call, `StreamGenerateContent` for a streamed one.
#[derive(Clone, Debug, PartialEq)]
pub struct GenerateContent {
    pub model: String,
}

impl GenerateContent {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
        }
    }
}

impl Wire for GenerateContent {
    type Op = Completion;
    type Payload = GenerateContentRequest;
    type Frame = GenerateContentResponse;
    type Decoder<'id> = crate::streaming::GrpcAdapter;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(PROVIDER_NAME)
            .model(self.model.as_str())
            .replay(self)
    }

    fn encode(
        &self,
        request: CompletionRequest,
        _mode: Mode,
    ) -> Result<GenerateContentRequest, EncodeError> {
        let model = request.model.clone().unwrap_or_else(|| self.model.clone());
        create_grpc_request(&model, request)
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        crate::streaming::GrpcAdapter::default()
    }
}

impl rig_core::completion::ReplayTarget for GenerateContent {
    fn api(&self) -> rig_core::message::Api {
        rig_core::message::Api::from_static("gemini.generate_content")
    }

    fn provider(&self) -> &str {
        PROVIDER_NAME
    }

    fn model(&self) -> &str {
        &self.model
    }

    /// The gRPC transcode carries no images inside function responses.
    fn accepts(&self, _model: &str) -> rig_core::completion::Accepts {
        rig_core::completion::Accepts {
            tool_result_images: false,
            ..rig_core::completion::Accepts::ALL
        }
    }

    fn normalize_tool_call_id(
        &self,
        id: &str,
        model: &str,
        _source: Option<&message::Origin>,
    ) -> String {
        rig_core::providers::gemini::completion::normalize_tool_call_id(model, id)
    }
}

impl Transport<GenerateContent> for GeminiGrpc {
    fn send(
        &self,
        request: GenerateContentRequest,
        exchange: Exchange,
    ) -> Opening<GenerateContentResponse> {
        let mode = exchange.mode;
        let mut client = self.grpc_client();
        Opening::new(async move {
            Ok(match mode {
                Mode::Unary => match client.generate_content(request).await {
                    Ok(response) => Opened::new(futures::stream::iter([Ok(response.into_inner())])),
                    Err(status) => Opened::failed(rpc_error(&status)),
                },
                Mode::Streaming => match client.stream_generate_content(request).await {
                    Ok(response) => {
                        let mut chunks = response.into_inner();
                        // Stop receiving after a tonic failure.
                        Opened::new(async_stream::stream! {
                            while let Some(item) = chunks.next().await {
                                match item {
                                    Ok(chunk) => yield Ok(chunk),
                                    Err(status) => {
                                        yield Err(rpc_error(&status));
                                        break;
                                    }
                                }
                            }
                        })
                    }
                    Err(status) => Opened::failed(rpc_error(&status)),
                },
            })
        })
    }
}

/// Stable descriptor name reported on normalized responses from this provider.
pub const PROVIDER_NAME: &str = "gemini-grpc";

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

/// Preserves tonic status display text with RPC code and retry classification.
/// Transport failures use the same provider-body representation.
pub(crate) fn rpc_error(status: &tonic::Status) -> ProviderError {
    ProviderError::from_provider_body(status.to_string())
        .with_provider_code(Some(grpc_code_name(status.code())))
        .with_transient(Some(transient_grpc_code(status.code())))
}

/// The gRPC status code's canonical name (`UNAVAILABLE`): the code the
/// provider answered with, kept apart from the message so a report can
/// key on it.
pub(crate) fn grpc_code_name(code: tonic::Code) -> String {
    format!("{code:?}")
        .chars()
        .fold(String::new(), |mut name, c| {
            if c.is_ascii_uppercase() && !name.is_empty() {
                name.push('_');
            }
            name.push(c.to_ascii_uppercase());
            name
        })
}

/// Recognizes transient gRPC codes; other codes are non-transient.
pub(crate) fn transient_grpc_code(code: tonic::Code) -> bool {
    matches!(
        code,
        tonic::Code::Unavailable
            | tonic::Code::ResourceExhausted
            | tonic::Code::DeadlineExceeded
            | tonic::Code::Aborted
    )
}

pub(crate) fn create_grpc_request(
    model: &str,
    completion_request: CompletionRequest,
) -> Result<GenerateContentRequest, EncodeError> {
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

    let (history_system, chat_history) = split_system_messages_from_history(chat_history);
    let chat_history = chat_history.into_iter().map(encode_raw_images).collect();
    let contents = contents(chat_history, model)?
        .into_iter()
        .map(|content| grpc_content(serde_json::from_value(content)?))
        .collect::<Result<Vec<_>, _>>()?;

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

    let generation_config = if temperature.is_some() || max_tokens.is_some() {
        Some(proto::GenerationConfig {
            temperature: temperature.map(|t| t as f32),
            max_output_tokens: max_tokens.map(|t| t as i32),
            ..Default::default()
        })
    } else {
        None
    };

    let tools = if !tools.is_empty() {
        let function_declarations = tools
            .into_iter()
            .map(|tool| {
                Ok(proto::FunctionDeclaration {
                    name: tool.name.into(),
                    description: tool.description,
                    parameters: tool_parameters_to_proto_schema(&tool.parameters)?,
                    ..Default::default()
                })
            })
            .collect::<Result<Vec<_>, EncodeError>>()?;

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

/// Encodes raw image bytes as base64, the form the shared Gemini conversion
/// takes, so gRPC keeps accepting them.
fn encode_raw_images(mut message: message::Message) -> message::Message {
    if let message::Message::User { content } = &mut message {
        for item in content {
            if let message::UserContent::Image(image) = item
                && let message::DocumentSourceKind::Raw(bytes) = &image.data
            {
                let data = base64::engine::general_purpose::STANDARD.encode(bytes);
                image.data = message::DocumentSourceKind::Base64(data);
            }
        }
    }
    message
}

/// Transcodes content built by the shared Gemini conversion into its
/// protobuf encoding.
fn grpc_content(content: Content) -> Result<proto::Content, EncodeError> {
    let role = match content.role {
        Some(Role::Model) => "model",
        Some(Role::User) | None => "user",
    };
    Ok(proto::Content {
        parts: content
            .parts
            .into_iter()
            .map(grpc_part)
            .collect::<Result<Vec<_>, _>>()?,
        role: role.to_string(),
    })
}

/// Transcodes one Gemini part. Rejects what the gRPC proto cannot carry:
/// media inside a function response and part metadata.
fn grpc_part(part: Part) -> Result<proto::Part, EncodeError> {
    if part
        .additional_params
        .as_ref()
        .is_some_and(|extra| extra.as_object().is_none_or(|extra| !extra.is_empty()))
    {
        return Err(EncodeError::request(
            "Gemini gRPC does not support part metadata",
        ));
    }
    let data = match part.part {
        PartKind::Text(text) => proto::part::Data::Text(text),
        PartKind::InlineData(Blob { mime_type, data }) => {
            proto::part::Data::InlineData(proto::Blob {
                mime_type,
                data: decode_base64_bytes(&data)?,
            })
        }
        PartKind::FileData(FileData {
            mime_type,
            file_uri,
        }) => proto::part::Data::FileData(proto::FileData {
            mime_type: mime_type.unwrap_or_default(),
            file_uri,
        }),
        PartKind::FunctionCall(call) => proto::part::Data::FunctionCall(proto::FunctionCall {
            name: call.name,
            args: Some(json_to_prost_struct(call.args)?),
            id: call.id.unwrap_or_default(),
        }),
        PartKind::FunctionResponse(response) => {
            if response.parts.is_some() {
                return Err(EncodeError::request(
                    "Gemini gRPC does not support images in tool results",
                ));
            }
            proto::part::Data::FunctionResponse(proto::FunctionResponse {
                name: response.name,
                response: response.response.map(json_to_prost_struct).transpose()?,
                id: response.id.unwrap_or_default(),
            })
        }
        PartKind::ExecutableCode(code) => {
            proto::part::Data::ExecutableCode(proto::ExecutableCode {
                language: proto::executable_code::Language::from_str_name(&wire_name(
                    &code.language,
                )?)
                .unwrap_or_default() as i32,
                code: code.code,
            })
        }
        PartKind::CodeExecutionResult(result) => {
            proto::part::Data::CodeExecutionResult(proto::CodeExecutionResult {
                outcome: proto::code_execution_result::Outcome::from_str_name(&wire_name(
                    &result.outcome,
                )?)
                .unwrap_or_default() as i32,
                output: result.output.unwrap_or_default(),
            })
        }
    };
    Ok(proto::Part {
        data: Some(data),
        thought: part.thought.unwrap_or(false),
        thought_signature: decode_optional_base64(part.thought_signature)?,
        part_metadata: None,
    })
}

/// A REST enum value's wire spelling, which the trimmed proto carries as a
/// string.
fn wire_name(value: &impl serde::Serialize) -> Result<String, EncodeError> {
    match serde_json::to_value(value)? {
        serde_json::Value::String(name) => Ok(name),
        other => Err(EncodeError::request(format!(
            "expected a Gemini enum spelling, got {other}"
        ))),
    }
}

fn decode_base64_bytes(input: &str) -> Result<Vec<u8>, EncodeError> {
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
    Err(EncodeError::request(format!("Invalid base64 data: {err}")))
}

fn decode_optional_base64(sig: Option<String>) -> Result<Vec<u8>, EncodeError> {
    let Some(sig) = sig else {
        return Ok(Vec::new());
    };
    decode_base64_bytes(&sig)
}

/// Gemini's protobuf `UsageMetadata` as the REST usage it transcodes to.
/// Proto3 cannot tell an unsent count from zero, so every optional count
/// is reported.
pub(crate) fn rest_usage(usage: &proto::UsageMetadata) -> UsageMetadata {
    UsageMetadata {
        prompt_token_count: usage.prompt_token_count,
        cached_content_token_count: Some(usage.cached_content_token_count),
        candidates_token_count: Some(usage.candidates_token_count),
        total_token_count: usage.total_token_count,
        thoughts_token_count: Some(usage.thoughts_token_count),
        tool_use_prompt_token_count: Some(usage.tool_use_prompt_token_count),
        ..UsageMetadata::default()
    }
}

pub(crate) fn encode_optional_base64(bytes: &[u8]) -> Option<String> {
    if bytes.is_empty() {
        None
    } else {
        Some(base64::engine::general_purpose::STANDARD.encode(bytes))
    }
}

fn json_to_prost_struct(value: serde_json::Value) -> Result<proto::Struct, EncodeError> {
    match value {
        serde_json::Value::Object(map) => Ok(proto::Struct {
            fields: map
                .into_iter()
                .map(|(k, v)| (k, json_to_prost_value(v)))
                .collect(),
        }),
        _ => Err(EncodeError::request(
            "Expected a JSON object for google.protobuf.Struct",
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

/// Converts tool parameters to protobuf schema through the shared Gemini conversion.
/// Empty object schemas map to `None`.
fn tool_parameters_to_proto_schema(
    value: &serde_json::Value,
) -> Result<Option<proto::Schema>, EncodeError> {
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
#[allow(clippy::expect_used, clippy::unwrap_used)]
pub(crate) mod tests;
