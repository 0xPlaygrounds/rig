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
use crate::providers::openai::wire::{OpenAI, ResponsesContract};
use crate::wire::{
    AdapterEvent, AdapterUsage, AdapterVerdict, Body, Encoded, Framing, Mode, ObservationSink, Wire,
};
use serde::{Deserialize, Serialize};

use super::streaming::{ResponsesDecoder, ResponsesStreamOptions};
use super::{
    CompletionRequest, Include, ResponsesRequestParams, ResponsesToolDefinition,
    SystemInstructionsPlacement,
};

/// The Responses wire: `POST /responses`, SSE when streamed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Responses {
    /// The provider this wire speaks to.
    pub provider: OpenAI,
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
            &OpenAI,
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
            .with_request_id_header(self.provider.dialect.request_id_header);
        Ok(if codex {
            encoded.with_relaxed_content_type()
        } else {
            encoded
        })
    }

    /// Create a wire with the provider's instruction placement and dialect's
    /// strict-tool default, without additional tools.
    pub fn new(provider: OpenAI, model: impl Into<String>) -> Self {
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
    type Decoder = ResponsesDecoder;

    fn name(&self) -> &str {
        self.provider.dialect.name
    }

    fn model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn replay_issuers(&self, model: Option<&str>) -> Vec<String> {
        crate::providers::openai::wire::replay_issuers(
            &self.provider.dialect,
            model.unwrap_or(&self.model),
        )
    }

    fn route(&self) -> Option<&str> {
        Some(self.provider.dialect.quirks.responses.path)
    }

    fn encode(
        &self,
        request: completion::CompletionRequest,
        mode: Mode,
    ) -> Result<Encoded, EncodeError> {
        self.encode_with_headers(request, mode, OpenAI::completion_headers)
    }

    fn decoder(&self, _mode: Mode) -> ResponsesDecoder {
        let quirks = &self.provider.dialect.quirks.responses;
        let options = if quirks.contract == ResponsesContract::Xai {
            // xAI answers a 200 with its error envelope, and the same
            // gateway publishes a finished call at its `output_item.done`.
            ResponsesStreamOptions::strict_with_immediate_tool_calls()
        } else {
            ResponsesStreamOptions::strict()
        };
        let mut decoder = ResponsesDecoder::new(self.provider.dialect.name, options);
        if quirks.contract == ResponsesContract::Codex {
            // The codex gateway's replayed frames may omit their envelope
            // bookkeeping; elsewhere an envelope-less frame is a defect
            // worth surfacing rather than salvaging.
            decoder = decoder.with_envelope_repair();
        }
        if self.provider.dialect.quirks.upstream_reasoning_issuer {
            decoder = decoder.with_upstream_reasoning_issuer();
        }
        decoder
    }

    fn capabilities(&self) -> ProviderCapabilities {
        // The xAI contract does not compose native structured output with tools.
        ProviderCapabilities::default().with_native_output_tool_composition(
            self.provider.dialect.quirks.responses.contract != ResponsesContract::Xai,
        )
    }
}

/// Normalize a whole Responses body through the decoder and completion fold.
/// Return serialization, decoder, or fold errors without performing I/O.
#[cfg(any(test, feature = "websocket"))]
pub(crate) fn fold_body(
    provider: &str,
    response: super::CompletionResponse,
) -> Result<completion::CompletionResponse, crate::error::ProviderError> {
    use super::streaming::ResponsesEvent;
    use crate::operation::AdapterOutput;
    use crate::wire::{Decoder, Fold, Operation, Reply, Sink};

    let reply = Reply {
        provider: provider.to_owned(),
        raw: serde_json::to_value(&response)?,
        provider_request_id: response.provider_request_id.clone(),
    };
    let mut decoder = ResponsesDecoder::new(provider, ResponsesStreamOptions::strict());
    let mut out = AdapterOutput::new();
    decoder.interpret(ResponsesEvent::Whole(Box::new(response)), &mut out);

    let mut fold = <Completion as Operation>::Fold::default();
    for item in Sink::<Completion>::drain(&mut out) {
        fold.absorb(item?)?;
    }
    fold.finish(reply)
}

#[derive(Default, Deserialize)]
struct TokenDetails {
    #[serde(default, deserialize_with = "crate::observe::lenient_count")]
    cached_tokens: Option<u64>,
    #[serde(default, deserialize_with = "crate::observe::lenient_count")]
    reasoning_tokens: Option<u64>,
}

#[derive(Deserialize)]
struct Usage {
    #[serde(default, deserialize_with = "crate::observe::lenient_count")]
    input_tokens: Option<u64>,
    #[serde(default, deserialize_with = "crate::observe::lenient_count")]
    output_tokens: Option<u64>,
    #[serde(default, deserialize_with = "crate::observe::lenient_count")]
    total_tokens: Option<u64>,
    #[serde(default)]
    input_tokens_details: Option<TokenDetails>,
    #[serde(default)]
    output_tokens_details: Option<TokenDetails>,
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
    usage: Option<Usage>,
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
pub(crate) fn project_payload(payload: &[u8], sink: &mut dyn ObservationSink) {
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
    if let Some(usage) = object.usage {
        sink.emit(AdapterEvent::Usage {
            usage: AdapterUsage {
                input_tokens: usage.input_tokens,
                output_tokens: usage.output_tokens,
                total_tokens: usage.total_tokens,
                cached_input_tokens: usage.input_tokens_details.and_then(|d| d.cached_tokens),
                reasoning_tokens: usage.output_tokens_details.and_then(|d| d.reasoning_tokens),
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
