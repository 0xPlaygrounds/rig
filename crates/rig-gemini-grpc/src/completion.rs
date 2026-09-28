//! The Gemini `GenerateContent` completion wire over gRPC.
//!
//! ```no_run
//! use rig_core::Model;
//! use rig_gemini_grpc::{GeminiGrpc, completion::{GEMINI_3_8_FLASH, GenerateContent}};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//! let model = GeminiGrpc::new("API_KEY").await?.completion(GEMINI_3_8_FLASH);
//! # let _ = model;
//! # Ok(())
//! # }
//! ```

/// `gemini-3.8-flash` completion model
pub const GEMINI_3_8_FLASH: &str = rig_core::providers::gemini::GEMINI_3_8_FLASH;
/// `gemini-2.5-flash` completion model
pub const GEMINI_2_5_FLASH: &str = "gemini-2.5-flash";
/// `gemini-2.0-flash-lite` completion model
pub const GEMINI_2_0_FLASH_LITE: &str = "gemini-2.0-flash-lite";
/// `gemini-2.0-flash` completion model
pub const GEMINI_2_0_FLASH: &str = "gemini-2.0-flash";

use base64::Engine as _;
use futures::StreamExt;
use rig_core::completion::{self, CompletionRequest};
use rig_core::driver::{Exchange, Opened, Opening, Transport};
use rig_core::error::EncodeError;
use rig_core::error::ProviderError;
use rig_core::message;
use rig_core::operation::Completion;
use rig_core::providers::gemini::api;
use rig_core::providers::gemini::edge::{self, Source, Unit};
use rig_core::wire::{Descriptor, Mode, Wire};
use std::convert::TryFrom;

use super::GeminiGrpc;
use super::proto::{self, GenerateContentRequest, GenerateContentResponse};
use super::streaming::GrpcAdapter;

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
    type Decoder<'id> = GrpcAdapter<'id>;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(PROVIDER_NAME).model(self.model.as_str())
    }

    /// The Gemini service issues this wire's reasoning, over gRPC or REST,
    /// so that is the reasoning a request may replay.
    fn encode(
        &self,
        request: CompletionRequest,
        _mode: Mode,
    ) -> Result<GenerateContentRequest, EncodeError> {
        create_grpc_request(&self.model, request.replayable_to(&[ISSUER])?)
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        GrpcAdapter::default()
    }
}

impl Transport<GenerateContent> for GeminiGrpc {
    fn send(
        &self,
        request: GenerateContentRequest,
        exchange: Exchange,
    ) -> Opening<GenerateContentResponse> {
        let mode = exchange.mode;
        let mut client = match self.grpc_client() {
            Ok(client) => client,
            Err(error) => return Opening::failed(ProviderError::Provider(error.to_string())),
        };
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

/// The issuer this transport's reasoning records: the Gemini API service,
/// which also serves the REST transport, so thought signatures move between
/// the two.
pub const REASONING_ISSUER: &str = rig_core::providers::gemini::PROVIDER_NAME;

/// [`REASONING_ISSUER`], the only issuer whose reasoning this wire replays.
const ISSUER: message::Issuer = message::Issuer::from_static(REASONING_ISSUER);

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

    edge::finish_reason(reason.as_str_name())
}

/// Returns a response error for malformed calls, unexpected calls, or exceeded
/// tool-call limits, including the supplied finish message. Other discriminants
/// return `None`.
pub fn tool_protocol_finish_reason_error(
    reason: i32,
    finish_message: Option<&str>,
) -> Option<ProviderError> {
    use proto::candidate::FinishReason as Wire;

    let reason = Wire::try_from(reason).ok()?;
    match reason {
        Wire::MalformedFunctionCall | Wire::UnexpectedToolCall | Wire::TooManyToolCalls => {
            edge::tool_protocol_error(reason.as_str_name(), finish_message)
        }
        _ => None,
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
    if completion_request.additional_params.is_some() {
        return Err(EncodeError::request(
            "additional_params is not read by Gemini gRPC",
        ));
    }
    let chat_history = completion_request.chat_history_with_documents();
    let CompletionRequest {
        tools,
        temperature,
        max_tokens,
        ..
    } = completion_request;

    let mut system_parts = Vec::new();
    let mut contents = Vec::new();
    for message in chat_history {
        match message {
            message::Message::System { content } => {
                if !content.is_empty() {
                    system_parts.push(text_part(content));
                }
            }
            message::Message::User { content } => {
                let parts = content
                    .into_iter()
                    .map(|content| unit_part(edge::user_unit(content)?))
                    .collect::<Result<Vec<_>, _>>()?;
                contents.push(proto::Content {
                    parts,
                    role: "user".to_string(),
                });
            }
            message::Message::Assistant { content, .. } => {
                let mut parts = Vec::new();
                for content in content {
                    for unit in edge::assistant_units(content, &ISSUER)? {
                        parts.push(unit_part(unit)?);
                    }
                }
                if !parts.is_empty() {
                    contents.push(proto::Content {
                        parts,
                        role: "model".to_string(),
                    });
                }
            }
        }
    }
    let system_instruction = (!system_parts.is_empty()).then(|| proto::Content {
        parts: system_parts,
        role: String::new(),
    });

    let generation_config = if temperature.is_some() || max_tokens.is_some() {
        Some(proto::GenerationConfig {
            temperature: temperature.map(|t| t as f32),
            max_output_tokens: max_tokens.map(|t| t as i32),
            ..Default::default()
        })
    } else {
        None
    };

    let tools = if tools.is_empty() {
        Vec::new()
    } else {
        // The schema goes as written, in `parameters_json_schema`.
        let function_declarations = tools
            .into_iter()
            .map(|tool| proto::FunctionDeclaration {
                name: tool.name,
                description: tool.description,
                parameters_json_schema: (!tool.parameters.is_null())
                    .then(|| json_to_prost_value(tool.parameters)),
                ..Default::default()
            })
            .collect();
        vec![proto::Tool {
            function_declarations,
            code_execution: None,
        }]
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

/// The protobuf part for one edge unit.
fn unit_part(unit: Unit) -> Result<proto::Part, EncodeError> {
    Ok(match unit {
        Unit::Text { text, signature } => proto::Part {
            thought_signature: decode_optional_base64(signature)?,
            ..text_part(text)
        },
        Unit::Thought { text, signature } => proto::Part {
            data: Some(proto::part::Data::Text(text)),
            thought: true,
            thought_signature: decode_optional_base64(signature)?,
            part_metadata: None,
        },
        Unit::Call {
            id,
            name,
            args,
            signature,
        } => proto::Part {
            thought_signature: decode_optional_base64(signature)?,
            ..data_part(proto::part::Data::FunctionCall(proto::FunctionCall {
                name,
                args: Some(json_to_prost_struct(serde_json::Value::Object(args))?),
                id: id.unwrap_or_default(),
            }))
        },
        Unit::Result {
            id,
            name,
            response,
            media,
        } => {
            if !media.is_empty() {
                return Err(EncodeError::request(
                    "Gemini gRPC does not support media in tool results",
                ));
            }
            data_part(proto::part::Data::FunctionResponse(
                proto::FunctionResponse {
                    name,
                    response: Some(json_to_prost_struct(serde_json::Value::Object(
                        response.unwrap_or_default(),
                    ))?),
                    id: id.unwrap_or_default(),
                },
            ))
        }
        Unit::Media(media) => {
            let mime_type = media
                .mime_type
                .ok_or_else(|| EncodeError::request("Gemini gRPC media needs a media type"))?;
            match media.source {
                Source::Inline(data) => data_part(proto::part::Data::InlineData(proto::Blob {
                    mime_type,
                    data: decode_base64_bytes(&data)?,
                })),
                Source::Uri(file_uri) => data_part(proto::part::Data::FileData(proto::FileData {
                    mime_type,
                    file_uri,
                })),
            }
        }
        Unit::Native(native) => native_part(native)?,
    })
}

/// The protobuf part for a native GenerateContent part. The gRPC schema
/// carries fewer part kinds than REST; one it cannot carry is refused.
fn native_part(native: message::NativePart) -> Result<proto::Part, EncodeError> {
    if native.schema != api::PART_SCHEMA {
        return Err(EncodeError::request(format!(
            "a native `{}` part cannot be sent over Gemini gRPC",
            native.schema
        )));
    }
    let part: api::Part = serde_json::from_str(native.json())?;
    let signature = decode_optional_base64(part.thought_signature.clone())?;
    let data = if let Some(code) = part.executable_code {
        proto::part::Data::ExecutableCode(proto::ExecutableCode {
            language: code
                .language
                .map(|language| language.as_str().to_owned())
                .unwrap_or_default(),
            code: code.code.unwrap_or_default(),
        })
    } else if let Some(result) = part.code_execution_result {
        proto::part::Data::CodeExecutionResult(proto::CodeExecutionResult {
            outcome: result
                .outcome
                .map(|outcome| outcome.as_str().to_owned())
                .unwrap_or_default(),
            output: result.output.unwrap_or_default(),
        })
    } else if let Some(blob) = part.inline_data {
        proto::part::Data::InlineData(proto::Blob {
            mime_type: blob.mime_type.unwrap_or_default(),
            data: decode_base64_bytes(blob.data.as_deref().unwrap_or_default())?,
        })
    } else if let Some(file) = part.file_data {
        proto::part::Data::FileData(proto::FileData {
            mime_type: file.mime_type.unwrap_or_default(),
            file_uri: file.file_uri.unwrap_or_default(),
        })
    } else {
        return Err(EncodeError::request(
            "Gemini gRPC cannot carry this native part",
        ));
    };
    Ok(proto::Part {
        data: Some(data),
        thought: part.thought.unwrap_or(false),
        thought_signature: signature,
        part_metadata: None,
    })
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

/// Rig's usage for Gemini's counts, by the edge's one rule.
pub(crate) fn map_usage(usage: Option<&proto::UsageMetadata>) -> completion::Usage {
    usage
        .map(|usage| {
            let count = |value: i32| u64::try_from(value).ok();
            edge::usage(edge::Counts {
                prompt: count(usage.prompt_token_count),
                cached: count(usage.cached_content_token_count),
                candidates: count(usage.candidates_token_count),
                ..edge::Counts::default()
            })
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

pub(crate) fn json_to_prost_value(value: serde_json::Value) -> proto::Value {
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

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
pub(crate) mod tests;
