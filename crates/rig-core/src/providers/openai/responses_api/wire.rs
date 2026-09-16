//! The OpenAI Responses endpoint as data: one wire on the shared
//! [`OpenAI`] configuration, N dialects.
//!
//! [`Responses`] is the wire; [`OpenAI`] is the configuration a host stores,
//! and [`OpenAI::responses`] builds this wire from it. A gateway that speaks
//! this format differs from OpenAI by *data* — which request shape it
//! accepts, whether it answers every request with an event stream, whether
//! a 200 can carry its error envelope — so it is the
//! [`responses`](crate::providers::openai::wire::Quirks::responses) field of
//! a [`Dialect`](crate::providers::openai::wire::Dialect) constant, never a
//! type or a trait.
//!
//! Dialects: [`OPENAI`](crate::providers::openai::wire::OPENAI),
//! [`chatgpt::DIALECT`](crate::providers::chatgpt::DIALECT),
//! [`xai::DIALECT`](crate::providers::xai::DIALECT) and
//! [`copilot::wire::DIALECT`](crate::providers::copilot::wire::DIALECT).

use crate::completion::{self, CompletionError, ProviderCapabilities};
use crate::operation::Completion;
use crate::providers::openai::wire::{OpenAI, RequestShape};
use crate::wire::{
    AdapterErrorEnvelope, AdapterEvent, AdapterUsage, AdapterVerdict, Body, Encoded, Framing, Mode,
    ObservationSink, Wire,
};
use serde::{Deserialize, Deserializer, Serialize};

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
    /// The Responses wire for `model` on `provider`, with the dialect's
    /// defaults.
    pub fn new(provider: OpenAI, model: impl Into<String>) -> Self {
        Self {
            system_instructions: provider.dialect.quirks.responses.system_instructions,
            provider,
            model: model.into(),
            tools: Vec::new(),
            strict_tools: false,
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

    /// The credential and identity headers every request to this dialect
    /// carries, shared by [`Wire::encode`] and the websocket handshake.
    pub(crate) fn headers(&self, builder: http::request::Builder) -> http::request::Builder {
        let mut builder = builder.header(
            http::header::AUTHORIZATION,
            format!("Bearer {}", self.provider.api_key.expose()),
        );
        if let Some(identity) = &self.provider.identity {
            builder = builder
                .header("originator", &identity.originator)
                .header(http::header::USER_AGENT, &identity.user_agent);
        }
        if self
            .provider
            .dialect
            .quirks
            .identity
            .is_some_and(|identity| identity.session_ids)
        {
            // A fresh per-request correlator, minted in the provider that
            // asks for it — which is where the record-replay guard
            // (`tests/core/no_random_ids.rs`) pins the one call site.
            builder = builder.header("session_id", crate::providers::chatgpt::session_id());
        }
        if let Some(account_id) = &self.provider.account_id {
            builder = builder.header("ChatGPT-Account-Id", account_id);
        }
        builder
    }

    /// The Responses request this wire sends, before serialization.
    pub(crate) fn responses_request(
        &self,
        request: completion::CompletionRequest,
        streaming: bool,
    ) -> Result<CompletionRequest, CompletionError> {
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
        if quirks.codex_parameter_subset {
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

    fn route(&self) -> Option<&str> {
        Some(self.provider.dialect.quirks.responses.path)
    }

    fn encode(
        &self,
        request: completion::CompletionRequest,
        mode: Mode,
    ) -> Result<Encoded, CompletionError> {
        let quirks = &self.provider.dialect.quirks.responses;
        // A gateway that only ever answers with an event stream is asked for
        // one whatever the caller wanted: the reply is framed the same way
        // either way, and the driver folds it.
        let streaming = matches!(mode, Mode::Streaming) || quirks.always_streams;
        let body = match quirks.request {
            RequestShape::Responses => {
                let request = self.responses_request(request, streaming)?;
                crate::providers::internal::trace_json(
                    crate::providers::internal::LogTarget::Completions,
                    "Responses completion request",
                    &request,
                );
                serde_json::to_vec(&request)?
            }
            // xAI's own input shape; its types live with the provider that
            // needs them.
            RequestShape::Xai => {
                let (_, body) = crate::providers::xai::api::create_completion_request(
                    self.model.clone(),
                    request,
                    &self.tools,
                    self.strict_tools,
                    streaming,
                )?;
                crate::providers::internal::trace_json(
                    crate::providers::internal::LogTarget::Completions,
                    "Responses completion request",
                    &body,
                );
                serde_json::to_vec(&body)?
            }
        };

        let request = self
            .headers(http::Request::post(format!(
                "{}{}",
                self.provider.base_url.trim_end_matches('/'),
                quirks.path
            )))
            .header(http::header::CONTENT_TYPE, "application/json")
            .body(Body::Bytes(body))
            .map_err(|error| CompletionError::ResponseError(error.to_string()))?;

        let framing = if streaming {
            Framing::Sse
        } else {
            Framing::Whole
        };
        let encoded = Encoded::new(request, framing)
            .with_request_id_header(self.provider.dialect.request_id_header);
        Ok(if quirks.relaxed_content_type {
            encoded.with_relaxed_content_type()
        } else {
            encoded
        })
    }

    fn decoder(&self) -> ResponsesDecoder {
        let quirks = &self.provider.dialect.quirks.responses;
        let options = if quirks.error_envelope_in_success {
            // The same gateway that answers 200 with an envelope publishes a
            // finished call at its `output_item.done`.
            ResponsesStreamOptions::strict_with_immediate_tool_calls()
        } else {
            ResponsesStreamOptions::strict()
        };
        let mut decoder = ResponsesDecoder::new(self.provider.dialect.name, options);
        if quirks.repair_envelope_less_frames {
            decoder = decoder.with_envelope_repair();
        }
        decoder
    }

    fn capabilities(&self) -> ProviderCapabilities {
        // The Responses API constrains only the final assistant message via
        // `text.format`; tools are still called across turns, so native
        // structured output composes with tool calls (issue #1928) — except
        // on a gateway that says otherwise.
        ProviderCapabilities::default().with_native_output_tool_composition(
            self.provider
                .dialect
                .quirks
                .responses
                .native_output_with_tools,
        )
    }
}

/// Fold one whole Responses body into the normalized response.
///
/// The ONE interpreter: the decoder's unary variant synthesizes the events
/// the stream sends, and the operation's own fold turns those into the
/// response — the same two steps [`crate::driver::call`] runs, without a
/// socket. Only the websocket session needs that: every other transport
/// reaches the same fold through the driver.
#[cfg(any(test, feature = "websocket"))]
pub(crate) fn fold_body(
    provider: &str,
    response: super::CompletionResponse,
) -> Result<completion::CompletionResponse, CompletionError> {
    use super::streaming::ResponsesEvent;
    use crate::providers::internal::adapter::AdapterOutput;
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

// ── observation ────────────────────────────────────────────────────────

/// A count the provider may report as something other than a number.
fn count<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<u64>, D::Error> {
    Ok(serde_json::Value::deserialize(deserializer)?.as_u64())
}

#[derive(Default, Deserialize)]
struct TokenDetails {
    #[serde(default, deserialize_with = "count")]
    cached_tokens: Option<u64>,
    #[serde(default, deserialize_with = "count")]
    reasoning_tokens: Option<u64>,
}

/// The error envelope this wire reports: `{"error": {code, message, type}}`;
/// the stream's `error` event carries the same fields at the top.
#[derive(Deserialize)]
struct Envelope {
    code: Option<serde_json::Value>,
    #[serde(rename = "type")]
    kind: Option<String>,
    message: Option<String>,
}

#[derive(Deserialize)]
struct Usage {
    #[serde(default, deserialize_with = "count")]
    input_tokens: Option<u64>,
    #[serde(default, deserialize_with = "count")]
    output_tokens: Option<u64>,
    #[serde(default, deserialize_with = "count")]
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

#[derive(Deserialize)]
struct ResponseObject {
    id: Option<String>,
    model: Option<String>,
    status: Option<String>,
    incomplete_details: Option<IncompleteDetails>,
    usage: Option<Usage>,
    error: Option<Envelope>,
}

#[derive(Deserialize)]
struct Payload {
    #[serde(rename = "type")]
    kind: Option<String>,
    response: Option<ResponseObject>,
    // The unary reply's own fields, and the stream `error` event's.
    id: Option<String>,
    model: Option<String>,
    status: Option<String>,
    incomplete_details: Option<IncompleteDetails>,
    usage: Option<Usage>,
    error: Option<Envelope>,
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
        let error = payload.error.unwrap_or(Envelope {
            code: payload.code,
            kind: None,
            message: payload.message,
        });
        envelope(sink, error);
        return;
    }
    let object = payload.response.unwrap_or(ResponseObject {
        id: payload.id,
        model: payload.model,
        status: payload.status,
        incomplete_details: payload.incomplete_details,
        usage: payload.usage,
        error: payload.error,
    });
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
        envelope(sink, error);
    }
}

fn envelope(sink: &mut dyn ObservationSink, error: Envelope) {
    let code = error.code.map(|code| match code {
        serde_json::Value::String(code) => sink.scrub(&code),
        serde_json::Value::Number(code) => code.to_string(),
        _ => "[invalid]".to_owned(),
    });
    sink.emit(AdapterEvent::ErrorEnvelope {
        error: AdapterErrorEnvelope {
            code,
            status: error.kind.map(|value| sink.scrub(&value)),
            message: error.message.map(|value| sink.scrub(&value)),
        },
    });
}

#[cfg(test)]
mod tests;
