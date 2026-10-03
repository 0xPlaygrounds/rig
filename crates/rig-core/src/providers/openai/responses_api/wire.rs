//! Responses endpoint encoding and observation for configured OpenAI dialects.
//!
//! ```
//! use rig_core::providers::openai::OpenAI;
//! let wire = OpenAI::new("key").responses("gpt-5.2");
//! ```

use crate::completion::{self, ProviderCapabilities};
use crate::error::EncodeError;
use crate::json_utils::Lenient;
use crate::observe::ObservedError;
use crate::operation::Completion;
use crate::providers::openai::wire::OpenAIConfig;
pub(crate) use crate::providers::openai::wire::ResponsesContract;
use crate::wire::{
    AdapterEvent, AdapterUsage, AdapterVerdict, Body, Capabilities, Descriptor, Encoded, Framing,
    Mode, ObservationSink, Wire,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::streaming::{ResponsesDecoder, usage_of};
use super::{ResponsesToolDefinition, SystemInstructionsPlacement};

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

    /// The wire's own tools are declared beside the request's.
    fn declares_tools(&self, request: &completion::CompletionRequest) -> bool {
        !self.tools.is_empty() || crate::completion::history::declares_tools(request)
    }

    /// A call item names its id in `call_id`.
    fn call_id_slot(&self) -> Option<&'static str> {
        Some("/call_id")
    }

    /// An edited block keeps its item's `id` and `type`, a message its
    /// `phase`, and reasoning its ciphertext, so the items after it stay
    /// paired (pi's text signature is `{id, phase}`).
    fn identity(&self, item: &Value) -> Map<String, Value> {
        let keys: &[&str] = match item.str("type") {
            Some("message") => &["type", "id", "phase"],
            Some("reasoning") => &["type", "id", "encrypted_content"],
            Some("function_call" | "custom_tool_call") => &["type", "id"],
            _ => &[],
        };
        keys.iter()
            .filter_map(|key| Some(((*key).to_owned(), item.get(*key)?.clone())))
            .collect()
    }

    /// A reasoning item goes only with the item it preceded.
    fn needs_next(&self, item: &Value) -> bool {
        item.str("type") == Some("reasoning")
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
                super::image_part(image).is_some()
                    && (!matches!(image.data, DocumentSourceKind::FileId(_))
                        || file_ids && quirks.responses.contract != ResponsesContract::Xai)
            }
            Media::Document(document) => {
                super::document_part(document).is_some()
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
        let legal = |c: char| c.is_ascii_alphanumeric() || c == '_' || c == '-';
        let sanitized: String = id.replace(|c| !legal(c), "_").chars().take(64).collect();
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

/// The facts a Responses payload carries before normalization discards
/// them: the verdict, the model, the response id, the usage and any error
/// envelope.
///
/// The unary reply is the response object itself; a stream event wraps that
/// object under `response` (`response.created`, `.completed`, `.failed`,
/// `.incomplete`) or, for `error`, carries the envelope's fields itself.
pub(crate) fn project_payload(payload: &[u8], sink: &mut ObservationSink<'_>) {
    let Ok(payload) = serde_json::from_slice::<Value>(payload) else {
        return;
    };
    let envelope = |error: &Value| ObservedError {
        code: error.get("code").filter(|code| !code.is_null()).cloned(),
        kind: error
            .str("type")
            .or_else(|| error.str("status"))
            .map(str::to_owned),
        message: error.str("message").map(str::to_owned),
    };
    if payload.str("type") == Some("error") {
        // The event carries its envelope either nested under `error` or as
        // its own top-level fields; the nested form names the error type.
        match payload.get("error").filter(|error| error.is_object()) {
            Some(error) => envelope(error),
            None => ObservedError {
                kind: None,
                ..envelope(&payload)
            },
        }
        .emit(sink);
        return;
    }
    let object = payload
        .get("response")
        .filter(|response| response.is_object())
        .unwrap_or(&payload);
    if let Some(usage) = object.get("usage").filter(|usage| usage.is_object()) {
        let usage = usage_of(usage);
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
    let verdict = AdapterVerdict {
        finish_reason: object
            .str("status")
            .filter(|status| *status != "in_progress" && *status != "queued")
            .map(|value| sink.scrub(value)),
        block_reason: None,
        detail: object
            .at("/incomplete_details/reason")
            .and_then(Value::as_str)
            .map(|value| sink.scrub(value)),
        model: object.str("model").map(|value| sink.scrub(value)),
    };
    let response_id = object.str("id").map(|value| sink.scrub(value));
    sink.provider(verdict, response_id);
    if let Some(error) = object.get("error").filter(|error| error.is_object()) {
        envelope(error).emit(sink);
    }
}

#[cfg(test)]
mod tests;
