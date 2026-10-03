//! Responses endpoint encoding and observation for configured OpenAI dialects.
//!
//! ```
//! use rig_core::providers::openai::OpenAI;
//! let wire = OpenAI::new("key").responses("gpt-5.2");
//! ```

use crate::completion::{self, ProviderCapabilities};
use crate::error::EncodeError;
use crate::observe::ObservedError;
use crate::operation::Completion;
use crate::providers::openai::wire::{OpenAIConfig, ResponsesContract};
use crate::wire::{
    AdapterEvent, AdapterUsage, AdapterVerdict, Body, Capabilities, Descriptor, Encoded, Framing,
    Mode, ObservationSink, Wire,
};
use serde::{Deserialize, Serialize};

use super::streaming::ResponsesDecoder;
use super::{
    CompletionRequest, Include, ResponsesRequestParams, ResponsesToolDefinition,
    SystemInstructionsPlacement,
};

/// The Responses wire: `POST /responses`, SSE when streamed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Responses {
    /// The provider this wire speaks to.
    pub provider: OpenAIConfig,
    /// The model to address.
    pub model: String,
    /// Tools added to every request from this wire.
    pub tools: Vec<ResponsesToolDefinition>,
    /// Whether Rig-generated tools request OpenAI's strict validation.
    pub strict_tools: bool,
    /// Where this wire puts Rig's system instructions. Defaults to the
    /// dialect's placement.
    pub system_instructions: SystemInstructionsPlacement,
}

impl Responses {
    pub(crate) fn encode_with_headers(
        &self,
        request: completion::CompletionRequest,
        mode: Mode,
        headers: impl FnOnce(
            &OpenAIConfig,
            &completion::CompletionRequest,
            http::request::Builder,
        ) -> http::request::Builder,
    ) -> Result<Encoded, EncodeError> {
        let quirks = &self.provider.dialect.quirks.responses;
        // The codex gateway only ever answers with an event stream, and
        // names no content type on it. It is asked for one whatever the
        // caller wanted: the reply is framed the same way either way, and
        // the driver folds it.
        let codex = quirks.contract == ResponsesContract::Codex;
        let streaming = matches!(mode, Mode::Streaming) || codex;
        let builder = headers(
            &self.provider,
            &request,
            http::Request::post(self.provider.uri(quirks.path, None)),
        );
        let request = self.responses_request(request, streaming)?;
        crate::providers::internal::trace_json(
            crate::providers::internal::LogTarget::Completions,
            "Responses completion request",
            &request,
        );
        let body = serde_json::to_vec(&request)?;

        let request = builder
            .header(http::header::CONTENT_TYPE, "application/json")
            .body(Body::Bytes(body))?;

        let framing = if streaming {
            Framing::Sse
        } else {
            Framing::Whole
        };
        let encoded = Encoded::new(request, framing)
            .with_request_id_header(self.provider.dialect.request_id_header)
            .with_route(Some(self.provider.dialect.quirks.responses.path))
            .with_projection(project_payload);
        Ok(if codex {
            encoded.with_relaxed_content_type()
        } else {
            encoded
        })
    }

    /// Create a wire with the provider's instruction placement and dialect's
    /// strict-tool default, without additional tools.
    pub fn new(provider: OpenAIConfig, model: impl Into<String>) -> Self {
        Self {
            system_instructions: provider.system_instructions_placement(),
            strict_tools: provider.dialect.quirks.responses.strict_tools_by_default,
            provider,
            model: model.into(),
            tools: Vec::new(),
        }
    }

    /// Sanitize function schemas for strict mode and send `strict: true`.
    pub fn with_strict_tools(mut self) -> Self {
        self.strict_tools = true;
        self
    }

    /// Add a tool to every request from this wire.
    pub fn with_tool(mut self, tool: impl Into<ResponsesToolDefinition>) -> Self {
        self.tools.push(tool.into());
        self
    }

    /// Add tools to every request from this wire.
    pub fn with_tools<I, Tool>(mut self, tools: I) -> Self
    where
        I: IntoIterator<Item = Tool>,
        Tool: Into<ResponsesToolDefinition>,
    {
        self.tools.extend(tools.into_iter().map(Into::into));
        self
    }

    /// Put Rig's system instructions somewhere other than the dialect's
    /// default placement.
    pub fn with_system_instructions_placement(
        mut self,
        placement: SystemInstructionsPlacement,
    ) -> Self {
        self.system_instructions = placement;
        self
    }

    /// Send Rig's system instructions as `system` messages in `input`, for a
    /// backend that rejects or ignores top-level `instructions`.
    pub fn with_system_instructions_as_messages(self) -> Self {
        self.with_system_instructions_placement(SystemInstructionsPlacement::InputSystemMessages)
    }

    /// The Responses request this wire sends, before serialization.
    pub(crate) fn responses_request(
        &self,
        request: completion::CompletionRequest,
        streaming: bool,
    ) -> Result<CompletionRequest, EncodeError> {
        let quirks = &self.provider.dialect.quirks.responses;
        let mut request = CompletionRequest::try_from(ResponsesRequestParams {
            model: self.model.clone(),
            request,
            system_instructions_placement: self.system_instructions,
        })?;
        request.tools.extend(self.tools.clone());
        if self.strict_tools {
            request.tools = request
                .tools
                .into_iter()
                .map(ResponsesToolDefinition::normalize)
                .collect();
        }
        if let Some(instructions) = &self.provider.instructions {
            request.instructions = Some(merge_instructions(
                instructions,
                request.instructions.as_deref(),
            ));
        }
        if quirks.contract == ResponsesContract::Codex {
            // The codex gateway takes the turn and the tools; sampling,
            // storage, metadata and structured output are not its to accept,
            // and `store: false` is the one value it wants stated.
            request.temperature = None;
            request.max_output_tokens = None;
            request.additional_parameters.background = None;
            request.additional_parameters.metadata.clear();
            request.additional_parameters.parallel_tool_calls = None;
            request.additional_parameters.service_tier = None;
            request.additional_parameters.store = Some(false);
            request.additional_parameters.text = None;
            request.additional_parameters.top_p = None;
            request.additional_parameters.user = None;
            // Reasoning items replay across turns only with their encrypted
            // payload, and this gateway stores nothing.
            let include = request
                .additional_parameters
                .include
                .get_or_insert_with(Vec::new);
            if !include
                .iter()
                .any(|item| matches!(item, Include::ReasoningEncryptedContent))
            {
                include.push(Include::ReasoningEncryptedContent);
            }
        }
        request.stream = streaming.then_some(true);
        Ok(request)
    }
}

/// Merge a gateway's own instructions ahead of the caller's preamble,
/// without repeating them when the preamble already carries them.
fn merge_instructions(instructions: &str, existing: Option<&str>) -> String {
    match existing.map(str::trim).filter(|value| !value.is_empty()) {
        Some(existing) if existing.contains(instructions) => existing.to_owned(),
        Some(existing) => format!("{instructions}\n\n{existing}"),
        None => instructions.to_owned(),
    }
}

impl Wire for Responses {
    type Op = Completion;
    type Payload = crate::wire::Encoded;
    type Frame = crate::wire::WireFrame;
    type Decoder<'id> = ResponsesDecoder;

    /// The xAI contract does not compose native structured output with tools.
    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(self.provider.dialect.name)
            .model(self.model.as_str())
            .capabilities(Capabilities::completion(
                ProviderCapabilities::default().with_native_output_tool_composition(
                    self.provider.dialect.quirks.responses.contract != ResponsesContract::Xai,
                ),
            ))
            .replay(self)
    }

    fn encode(
        &self,
        request: completion::CompletionRequest,
        mode: Mode,
    ) -> Result<Encoded, EncodeError> {
        self.encode_with_headers(request, mode, OpenAIConfig::completion_headers)
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        ResponsesDecoder::new()
    }
}

impl crate::completion::ReplayTarget for Responses {
    fn api(&self) -> crate::message::Api {
        crate::message::Api::from_static("openai.responses")
    }

    fn provider(&self) -> &str {
        self.provider.dialect.name
    }

    fn model(&self) -> &str {
        &self.model
    }

    /// Responses reads images in user input and in function outputs, never
    /// in assistant messages, and only on a model with vision input. Every
    /// documented model calls tools except `o1-mini` and `o1-preview`.
    fn accepts(&self, model: &str) -> crate::completion::Accepts {
        let images = reads_images(self.provider.dialect.quirks.responses.contract, model);
        let model = model.to_ascii_lowercase();
        crate::completion::Accepts {
            user_images: images,
            assistant_images: false,
            tool_result_images: images,
            tools: !(model.starts_with("o1-mini") || model.starts_with("o1-preview")),
        }
    }

    /// Responses carries user and tool-result images as data URLs, URLs or
    /// file ids, documents as files or text, and no audio, video or
    /// assistant image. File ids need a dialect that resolves them, and
    /// xAI's `input_image` takes none.
    fn encodes(&self, _model: &str, media: crate::completion::Media<'_>) -> bool {
        use crate::completion::{Media, Place};
        use crate::message::DocumentSourceKind;
        let quirks = &self.provider.dialect.quirks;
        let file_ids = quirks.accepts_file_ids;
        match media {
            Media::Image(_, Place::Assistant) | Media::Audio(_) | Media::Video(_) => false,
            Media::Image(image, _) => {
                super::image_input(image).is_some()
                    && (!matches!(image.data, DocumentSourceKind::FileId(_))
                        || file_ids && quirks.responses.contract != ResponsesContract::Xai)
            }
            Media::Document(document) => {
                super::document_input(document).is_some()
                    && (file_ids || !matches!(document.data, DocumentSourceKind::FileId(_)))
            }
        }
    }

    /// A request naming `previous_response_id` continues a response the
    /// provider stores, which holds the calls its first results answer.
    fn continues_stored(&self, request: &completion::CompletionRequest) -> bool {
        request
            .additional_params
            .as_ref()
            .and_then(|params| params.get("previous_response_id"))
            .is_some_and(|id| !id.is_null())
    }

    /// pi's `normalizeIdPart`: characters outside `[a-zA-Z0-9_-]` become
    /// `_`, the id is cut to 64 characters and loses its trailing `_`.
    fn normalize_tool_call_id(
        &self,
        id: &str,
        _model: &str,
        _source: Option<&crate::message::Origin>,
    ) -> String {
        let sanitized: String = id
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .take(64)
            .collect();
        sanitized.trim_end_matches('_').to_owned()
    }
}

/// Whether `model` reads images, by its vendor's documented text-only
/// models. An unknown model reads them.
fn reads_images(contract: ResponsesContract, model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    let model = model.rsplit('/').next().unwrap_or_default();
    let text_only = match contract {
        ResponsesContract::Xai => {
            (model.starts_with("grok-2") && !model.contains("vision"))
                || model.starts_with("grok-3")
                || model.starts_with("grok-code")
        }
        ResponsesContract::OpenAi | ResponsesContract::Codex => {
            matches!(model, "gpt-4" | "gpt-4-0613" | "gpt-4-0314")
                || model.starts_with("gpt-4-32k")
                || model.starts_with("gpt-3.5")
                || model.starts_with("o1-mini")
                || model.starts_with("o1-preview")
                || model.starts_with("o3-mini")
                || model.starts_with("gpt-5.3-codex-spark")
        }
    };
    !text_only
}

#[derive(Default, Deserialize)]
struct TokenDetails {
    #[serde(default, deserialize_with = "crate::observe::lenient_count")]
    cached_tokens: Option<u64>,
    #[serde(default, deserialize_with = "crate::observe::lenient_count")]
    cache_write_tokens: Option<u64>,
    #[serde(default, deserialize_with = "crate::observe::lenient_count")]
    reasoning_tokens: Option<u64>,
}

/// A Responses `usage` object, read leniently: a counter that is absent or
/// not a count is unreported.
#[derive(Default, Deserialize)]
struct Usage {
    #[serde(default, deserialize_with = "crate::observe::lenient_count")]
    input_tokens: Option<u64>,
    #[serde(default, deserialize_with = "crate::observe::lenient_count")]
    output_tokens: Option<u64>,
    #[serde(default, deserialize_with = "crate::observe::lenient_count")]
    total_tokens: Option<u64>,
    #[serde(default, deserialize_with = "lenient_details")]
    input_tokens_details: Option<TokenDetails>,
    #[serde(default, deserialize_with = "lenient_details")]
    output_tokens_details: Option<TokenDetails>,
}

/// A details object, or none when it is not one.
fn lenient_details<'de, D>(deserializer: D) -> Result<Option<TokenDetails>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    Ok(serde_json::from_value(value).ok())
}

/// The usage a Responses `usage` value reports; anything but an object
/// reports none.
pub(crate) fn usage_of(usage: &serde_json::Value) -> completion::Usage {
    let usage = Usage::deserialize(usage).unwrap_or_default();
    let input = usage.input_tokens_details.unwrap_or_default();
    let output = usage.output_tokens_details.unwrap_or_default();
    completion::Usage {
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        total_tokens: usage.total_tokens,
        cached_input_tokens: input.cached_tokens,
        cache_creation_input_tokens: input.cache_write_tokens,
        reasoning_tokens: output.reasoning_tokens,
        ..completion::Usage::default()
    }
}

#[derive(Deserialize)]
struct IncompleteDetails {
    reason: Option<String>,
}

/// The response object, whether it arrived nested under a stream event's
/// `response` or as the unary reply itself. Every field is optional: this is
/// the observation parse, and a payload that carries none of them projects
/// nothing rather than failing.
#[derive(Default, Deserialize)]
struct ResponseObject {
    id: Option<String>,
    model: Option<String>,
    status: Option<String>,
    incomplete_details: Option<IncompleteDetails>,
    usage: Option<serde_json::Value>,
    error: Option<ObservedError>,
}

#[derive(Deserialize)]
struct Payload {
    #[serde(rename = "type")]
    kind: Option<String>,
    response: Option<ResponseObject>,
    /// The unary reply *is* the response object, so the same fields are
    /// read at the top level rather than declared a second time.
    #[serde(flatten)]
    unwrapped: ResponseObject,
    // The stream `error` event's own fields.
    code: Option<serde_json::Value>,
    message: Option<String>,
}

/// The facts a Responses payload carries before normalization discards
/// them: the verdict, the model, the response id, the usage and any error
/// envelope.
///
/// The unary reply is the response object itself; a stream event wraps that
/// object under `response` (`response.created`, `.completed`, `.failed`,
/// `.incomplete`) or, for `error`, carries the envelope's fields itself.
pub(crate) fn project_payload(payload: &[u8], sink: &mut ObservationSink<'_>) {
    let Ok(payload) = serde_json::from_slice::<Payload>(payload) else {
        return;
    };
    if payload.kind.as_deref() == Some("error") {
        // The event carries its envelope either nested under `error` or as
        // its own top-level fields; the nested form names the error type.
        payload
            .unwrapped
            .error
            .unwrap_or(ObservedError {
                code: payload.code,
                kind: None,
                message: payload.message,
            })
            .emit(sink);
        return;
    }
    let object = payload.response.unwrap_or(payload.unwrapped);
    if let Some(usage) = object.usage.filter(serde_json::Value::is_object) {
        let usage = usage_of(&usage);
        sink.emit(AdapterEvent::Usage {
            usage: AdapterUsage {
                input_tokens: usage.input_tokens,
                output_tokens: usage.output_tokens,
                total_tokens: usage.total_tokens,
                cached_input_tokens: usage.cached_input_tokens,
                reasoning_tokens: usage.reasoning_tokens,
                tool_input_tokens: None,
            },
        });
    }
    // `status` is the provider's verdict; `in_progress` on a stream's
    // opening event is not one yet, so it is left out of the projection.
    let finish_reason = object
        .status
        .filter(|status| status != "in_progress" && status != "queued");
    let verdict = AdapterVerdict {
        finish_reason: finish_reason.map(|value| sink.scrub(&value)),
        block_reason: None,
        detail: object
            .incomplete_details
            .and_then(|details| details.reason)
            .map(|value| sink.scrub(&value)),
        model: object.model.map(|value| sink.scrub(&value)),
    };
    let response_id = object.id.map(|value| sink.scrub(&value));
    sink.provider(verdict, response_id);
    if let Some(error) = object.error {
        error.emit(sink);
    }
}

#[cfg(test)]
mod tests;
