//! GenerateContent through the [`edge`]: the request body built
//! from rig's request and the model's settings, and the decoder for a whole
//! `generateContent` body or `streamGenerateContent` chunks.

use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;

use super::api;
use super::edge::{self, Dialect, Encoded, Media, Source, Unit};
use super::prefix::CachedPrefix;
use crate::completion::{CompletionRequest, ToolDefinition};
use crate::error::{EncodeError, ProviderError};
use crate::message::{MediaDetail, Message};
use crate::operation::{Completion, Finish};
use crate::providers::internal::wire;
use crate::wire::{Decoder, Flow, Out, WireEvent, WireFrame};

/// GenerateContent's parts.
pub(crate) struct GenerateContentDialect;

impl Dialect for GenerateContentDialect {
    const SCHEMA: &'static str = api::PART_SCHEMA;
    type Part = api::Part;

    fn units(raw: &RawValue) -> Result<Vec<Unit>, serde_json::Error> {
        Ok(vec![Self::unit(raw)?])
    }

    fn part(unit: Unit) -> Result<Encoded<api::Part>, EncodeError> {
        Self::encode(unit)
    }
}

impl GenerateContentDialect {
    /// The one unit of a part.
    fn unit(raw: &RawValue) -> Result<Unit, serde_json::Error> {
        let part: api::Part = serde_json::from_str(raw.get())?;
        // Only a part whose every field has a home in rig's model is
        // normalized; anything else stays native.
        let bare = api::Part {
            text: None,
            thought: None,
            thought_signature: None,
            function_call: None,
            ..part.clone()
        };
        if bare != api::Part::default() {
            return Ok(edge::native(raw, Self::SCHEMA));
        }
        let api::Part {
            text,
            thought,
            thought_signature: signature,
            function_call,
            ..
        } = part;
        Ok(match (text, thought, function_call) {
            (Some(text), None, None) => Unit::Text { text, signature },
            (Some(text), Some(true), None) => Unit::Thought { text, signature },
            (None, None, Some(call)) if call.unmodeled.is_empty() && call.name.is_some() => {
                Unit::Call {
                    id: call.id,
                    name: call.name.unwrap_or_default(),
                    args: call.args.unwrap_or_default(),
                    signature,
                }
            }
            _ => edge::native(raw, Self::SCHEMA),
        })
    }

    fn encode(unit: Unit) -> Result<Encoded<api::Part>, EncodeError> {
        Ok(Encoded::Typed(match unit {
            Unit::Text { text, signature } => api::Part {
                text: Some(text),
                thought_signature: signature,
                ..Default::default()
            },
            Unit::Thought { text, signature } => api::Part {
                text: Some(text),
                thought: Some(true),
                thought_signature: signature,
                ..Default::default()
            },
            Unit::Call {
                id,
                name,
                args,
                signature,
            } => api::Part {
                function_call: Some(api::FunctionCall {
                    id,
                    name: Some(name),
                    args: Some(args),
                    ..Default::default()
                }),
                thought_signature: signature,
                ..Default::default()
            },
            Unit::Result {
                id,
                name,
                response,
                media,
            } => api::Part {
                function_response: Some(api::FunctionResponse {
                    id,
                    name: Some(name),
                    // Gemini requires a response object.
                    response: Some(response.unwrap_or_default()),
                    parts: media
                        .into_iter()
                        .map(function_response_part)
                        .collect::<Result<_, _>>()?,
                    ..Default::default()
                }),
                ..Default::default()
            },
            Unit::Media(media) => media_part(media),
            Unit::Native(native) => {
                return Ok(Encoded::Raw(edge::native_raw(native, Self::SCHEMA)?));
            }
        }))
    }
}

fn function_response_part(media: Media) -> Result<api::FunctionResponsePart, EncodeError> {
    let Source::Inline(data) = media.source else {
        return Err(EncodeError::request(
            "a Gemini function response carries inline media only",
        ));
    };
    Ok(api::FunctionResponsePart {
        inline_data: Some(api::FunctionResponseBlob {
            data: Some(data),
            mime_type: media.mime_type,
            ..Default::default()
        }),
        ..Default::default()
    })
}

fn media_part(media: Media) -> api::Part {
    let (inline_data, file_data) = match media.source {
        Source::Inline(data) => (
            Some(api::Blob {
                data: Some(data),
                mime_type: media.mime_type,
                ..Default::default()
            }),
            None,
        ),
        Source::Uri(uri) => (
            None,
            Some(api::FileData {
                file_uri: Some(uri),
                mime_type: media.mime_type,
                ..Default::default()
            }),
        ),
    };
    api::Part {
        inline_data,
        file_data,
        media_resolution: media.detail.and_then(resolution),
        ..Default::default()
    }
}

/// Gemini's per-part resolution for a detail level. `Auto` leaves the
/// model's default.
fn resolution(detail: MediaDetail) -> Option<api::PartMediaResolution> {
    let level = match detail {
        MediaDetail::Auto => return None,
        MediaDetail::Low => api::MediaResolutionLevel::Low,
        MediaDetail::Medium => api::MediaResolutionLevel::Medium,
        MediaDetail::High => api::MediaResolutionLevel::High,
        MediaDetail::UltraHigh => api::MediaResolutionLevel::UltraHigh,
    };
    Some(api::PartMediaResolution {
        level: Some(level),
        ..Default::default()
    })
}

/// A `generateContent` body. Contents are written part by part so a native
/// part goes out as the exact JSON Gemini sent.
#[derive(Debug, Serialize)]
pub(crate) struct Body {
    contents: Vec<Turn>,
    #[serde(flatten)]
    request: api::GenerateContentRequest,
}

/// One turn of a request's contents.
#[derive(Debug, Serialize)]
struct Turn {
    role: &'static str,
    parts: Vec<Encoded<api::Part>>,
}

impl Body {
    /// The request as Google's schema reads it, native parts included.
    pub(crate) fn to_api(&self) -> Result<api::GenerateContentRequest, serde_json::Error> {
        serde_json::from_slice(&serde_json::to_vec(self)?)
    }
}

/// The body GenerateContent sends for `request` with the model's `settings`,
/// reading its prefix from `cached` when set.
pub(crate) fn body(
    request: CompletionRequest,
    settings: &api::RequestSettings,
    cached: Option<&CachedPrefix>,
) -> Result<Body, EncodeError> {
    if request.additional_params.is_some() {
        return Err(EncodeError::request(
            "additional_params is not read by Gemini; use GenerateContent::settings",
        ));
    }
    let chat_history = request.chat_history_with_documents();
    let CompletionRequest {
        tools,
        temperature,
        max_tokens,
        tool_choice,
        output_schema,
        ..
    } = request;

    let mut system = Vec::new();
    let mut contents = Vec::new();
    for message in chat_history {
        match message {
            Message::System { content } => system.push(api::Part {
                text: Some(content),
                ..Default::default()
            }),
            Message::User { content } => contents.push(Turn {
                role: "user",
                parts: content
                    .into_iter()
                    .map(|content| GenerateContentDialect::part(edge::user_unit(content)?))
                    .collect::<Result<_, _>>()?,
            }),
            Message::Assistant { content, .. } => {
                let mut parts = Vec::new();
                for content in content {
                    for unit in edge::assistant_units(content, &super::ISSUER)? {
                        parts.push(GenerateContentDialect::part(unit)?);
                    }
                }
                if !parts.is_empty() {
                    contents.push(Turn {
                        role: "model",
                        parts,
                    });
                }
            }
        }
    }

    let mut request: api::GenerateContentRequest = convert(settings)?;
    request.system_instruction = (!system.is_empty()).then(|| api::Content {
        parts: system,
        ..Default::default()
    });

    let mut generation: api::GenerationConfig = convert(&settings.generation_config)?;
    generation.temperature = temperature;
    generation.max_output_tokens = max_tokens
        .map(i32::try_from)
        .transpose()
        .map_err(|_| EncodeError::request("max_tokens is larger than Gemini's maxOutputTokens"))?;
    if let Some(schema) = output_schema {
        generation.response_mime_type = Some("application/json".to_owned());
        generation.response_json_schema = Some(schema.to_value());
    }
    request.generation_config =
        (generation != api::GenerationConfig::default()).then_some(generation);

    request.tools = tool_list(&tools, &settings.tools)?;
    let mut tool_config: api::ToolConfig = convert(&settings.tool_config)?;
    tool_config.function_calling_config = tool_choice.map(function_calling);
    tool_config.include_server_side_tool_invocations = mixes_tools(&request.tools).then_some(true);
    request.tool_config = (tool_config != api::ToolConfig::default()).then_some(tool_config);

    if let Some(prefix) = cached {
        prefix.strip(&mut request)?;
    }
    Ok(Body { contents, request })
}

/// Function declarations first, then the hosted tools, as one `tools` list.
pub(crate) fn tool_list(
    tools: &[ToolDefinition],
    hosted: &[api::HostedTool],
) -> Result<Vec<api::Tool>, EncodeError> {
    let mut list = Vec::new();
    if !tools.is_empty() {
        list.push(api::Tool {
            function_declarations: tools.iter().map(function_declaration).collect(),
            ..Default::default()
        });
    }
    for tool in hosted {
        list.push(convert(tool)?);
    }
    Ok(list)
}

/// Whether `tools` mixes function declarations with hosted tools, which
/// Gemini accepts only with `includeServerSideToolInvocations`.
pub(crate) fn mixes_tools(tools: &[api::Tool]) -> bool {
    let declares = tools
        .iter()
        .any(|tool| !tool.function_declarations.is_empty());
    let hosts = tools.iter().any(|tool| {
        *tool
            != api::Tool {
                function_declarations: tool.function_declarations.clone(),
                ..Default::default()
            }
    });
    declares && hosts
}

/// The declaration for `tool`, its JSON Schema sent as written.
pub(crate) fn function_declaration(tool: &ToolDefinition) -> api::FunctionDeclaration {
    api::FunctionDeclaration {
        name: Some(tool.name.clone()),
        description: Some(tool.description.clone()),
        parameters_json_schema: (!tool.parameters.is_null()).then(|| tool.parameters.clone()),
        ..Default::default()
    }
}

fn function_calling(choice: crate::message::ToolChoice) -> api::FunctionCallingConfig {
    use crate::message::ToolChoice;
    let (mode, allowed_function_names) = match choice {
        ToolChoice::Auto => (api::FunctionCallingMode::Auto, Vec::new()),
        ToolChoice::None => (api::FunctionCallingMode::None, Vec::new()),
        ToolChoice::Required => (api::FunctionCallingMode::Any, Vec::new()),
        ToolChoice::Specific { function_names } => (api::FunctionCallingMode::Any, function_names),
    };
    api::FunctionCallingConfig {
        mode: Some(mode),
        allowed_function_names,
        ..Default::default()
    }
}

/// A settings template as the mirror it was generated from. Both share
/// Google's field names, so the conversion is exact.
pub(crate) fn convert<A: Serialize, B: serde::de::DeserializeOwned>(
    value: &A,
) -> Result<B, EncodeError> {
    Ok(serde_json::from_value(serde_json::to_value(value)?)?)
}

/// One `generateContent` body or `streamGenerateContent` chunk: Google's
/// document, and each part's JSON as it arrived.
#[derive(Debug)]
pub struct Document {
    response: api::GenerateContentResponse,
    parts: Vec<Box<RawValue>>,
    error: Option<serde_json::Value>,
}

impl<'de> Deserialize<'de> for Document {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Raw {
            #[serde(default)]
            error: Option<serde_json::Value>,
            #[serde(default)]
            candidates: Vec<RawCandidate>,
        }
        #[derive(Deserialize)]
        struct RawCandidate {
            #[serde(default)]
            content: Option<RawContent>,
        }
        #[derive(Deserialize)]
        struct RawContent {
            #[serde(default)]
            parts: Vec<Box<RawValue>>,
        }
        use serde::de::Error;
        let text = Box::<RawValue>::deserialize(deserializer)?;
        let raw: Raw = serde_json::from_str(text.get()).map_err(D::Error::custom)?;
        let response = serde_json::from_str(text.get()).map_err(D::Error::custom)?;
        let parts = raw
            .candidates
            .into_iter()
            .next()
            .and_then(|candidate| candidate.content)
            .map(|content| content.parts)
            .unwrap_or_default();
        Ok(Self {
            response,
            parts,
            error: raw.error,
        })
    }
}

/// What a genuine body or chunk carries; a frame with any of these keys
/// must decode.
const KEYS: &[&str] = &["candidates", "usageMetadata", "promptFeedback", "error"];

/// Decodes GenerateContent replies, a whole body or a stream of chunks. The
/// reply ends at EOF, since a hosted-tool round can report a finish reason
/// before more content arrives.
pub struct GenerateContentDecoder<'id> {
    writer: edge::Writer<'id>,
    usage: Option<api::UsageMetadata>,
    finish: Option<api::FinishReason>,
    finish_message: Option<String>,
    model_version: Option<String>,
    response_id: Option<String>,
}

impl GenerateContentDecoder<'_> {
    pub(crate) fn new() -> Self {
        Self {
            writer: edge::Writer::new(),
            usage: None,
            finish: None,
            finish_message: None,
            model_version: None,
            response_id: None,
        }
    }
}

impl<'id> Decoder<'id, Completion> for GenerateContentDecoder<'id> {
    type Event = Document;

    fn classify(&self, frame: WireFrame) -> WireEvent<Document> {
        // A frame with only a response id is metadata, not unknown content.
        if super::streaming::is_analysis_only(&frame) {
            return wire::classify_marker_keyed_frame(&frame.as_str(), &["responseId"]);
        }
        wire::classify_marker_keyed_frame(&frame.as_str(), KEYS)
    }

    fn decode(
        &mut self,
        document: Document,
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        let Document {
            response,
            parts,
            error,
        } = document;
        let span = tracing::Span::current();
        if let Some(id) = response.response_id.filter(|id| !id.is_empty()) {
            span.record("gen_ai.response.id", id.as_str());
            self.response_id = Some(id);
        }
        if let Some(model) = response.model_version {
            span.record("gen_ai.response.model", model.as_str());
            self.model_version = Some(model);
        }
        if response.usage_metadata.is_some() {
            self.usage = response.usage_metadata;
        }
        if let Some(error) = error {
            // An in-band abort: its code is an HTTP status.
            let status = error
                .get("code")
                .and_then(serde_json::Value::as_u64)
                .and_then(|code| u16::try_from(code).ok())
                .and_then(|code| http::StatusCode::from_u16(code).ok())
                .filter(|status| status.is_client_error() || status.is_server_error());
            let body = serde_json::json!({ "error": error }).to_string();
            return Err(match status {
                Some(status) => ProviderError::from_http_response(status, body),
                None => ProviderError::from_provider_body(body),
            });
        }
        if let Some(feedback) = &response.prompt_feedback
            && let Some(reason) = &feedback.block_reason
        {
            let ratings: Vec<_> = feedback
                .safety_ratings
                .iter()
                .map(|rating| {
                    (
                        rating
                            .category
                            .as_ref()
                            .map_or("", api::HarmCategory::as_str)
                            .to_owned(),
                        rating
                            .probability
                            .as_ref()
                            .map_or("", api::HarmProbability::as_str)
                            .to_owned(),
                    )
                })
                .collect();
            if let Some(blocked) = edge::blocked(reason.as_str(), &ratings) {
                return Err(blocked);
            }
        }
        let Some(candidate) = response.candidates.into_iter().next() else {
            return Ok(Flow::More);
        };
        if let Some(message) = candidate.finish_message {
            self.finish_message = Some(message);
        }
        if let Some(reason) = candidate.finish_reason {
            if let Some(error) =
                edge::tool_protocol_error(reason.as_str(), self.finish_message.as_deref())
            {
                return Err(error);
            }
            // The last reason wins: a hosted-tool round can report one early.
            self.finish = Some(reason);
        }
        let typed = candidate
            .content
            .map(|content| content.parts)
            .unwrap_or_default();
        // Parts of one document are distinct; text across chunks is one part.
        let mut previous_text = false;
        for (raw, part) in parts.iter().zip(typed) {
            let unit = GenerateContentDialect::unit(raw)?;
            let text = matches!(unit, Unit::Text { .. });
            if text && previous_text {
                self.writer.split_text(&mut out);
            }
            previous_text = text;
            let hosted = part.executable_code.is_some()
                || part.code_execution_result.is_some()
                || part.tool_call.is_some()
                || part.tool_response.is_some();
            self.writer.unit(unit, !hosted, &mut out)?;
        }
        Ok(Flow::More)
    }

    fn eof(&mut self, mut out: Out<'id, Completion>) -> Result<Flow, ProviderError> {
        let Some(finish) = self.finish.take() else {
            return Err(ProviderError::Truncated);
        };
        let reason = edge::finish_reason(finish.as_str());
        let cut_short = reason
            .as_ref()
            .is_some_and(crate::completion::FinishReason::truncated_output);
        if !self.writer.delivered() && !cut_short {
            return Err(ProviderError::Response(
                crate::message::EMPTY_RESPONSE_ERROR.to_owned(),
            ));
        }
        self.writer.close(&mut out);
        let usage = self
            .usage
            .as_ref()
            .map(counts)
            .map(edge::usage)
            .unwrap_or_default();
        out.raw(super::streaming::summary(
            self.usage.take(),
            Some(finish),
            self.finish_message.take(),
            self.model_version.clone(),
            self.response_id.clone(),
        ));
        Ok(out.end(
            Finish::new(usage)
                .with_optional_reason(reason)
                .with_optional_response_id(self.response_id.take())
                .with_optional_model(self.model_version.take()),
        ))
    }
}

/// GenerateContent's counts under the edge's names.
pub(crate) fn counts(usage: &api::UsageMetadata) -> edge::Counts {
    edge::Counts {
        prompt: edge::count(usage.prompt_token_count),
        tool_use_prompt: edge::count(usage.tool_use_prompt_token_count),
        cached: edge::count(usage.cached_content_token_count),
        candidates: edge::count(usage.candidates_token_count),
        thoughts: edge::count(usage.thoughts_token_count),
    }
}

#[cfg(test)]
mod tests;
