//! The OpenAI Responses endpoint as data: one config struct, one wire, N
//! dialects.
//!
//! [`Responses`] is the wire; [`ResponsesApi`] is the shared configuration a
//! host stores. A gateway that speaks this format differs from OpenAI by
//! *data* — its name, base URL and environment variables, which request
//! shape it accepts, whether it answers every request with an event stream,
//! whether a 200 can carry its error envelope — so it is a [`Dialect`]
//! constant, never a type or a trait.
//!
//! Dialects: [`OPENAI`] here, [`chatgpt::DIALECT`](crate::providers::chatgpt::DIALECT)
//! and [`xai::DIALECT`](crate::providers::xai::DIALECT) with their providers.

use crate::client::env::{self, EnvError};
use crate::completion::{self, CompletionError, ProviderCapabilities};
use crate::operation::Completion;
use crate::providers::internal::adapter::AdapterOutput;
use crate::wire::{
    AdapterErrorEnvelope, AdapterEvent, AdapterUsage, AdapterVerdict, Body, Decoder, Encoded, Fold,
    Framing, HasCompletion, Mode, ObservationSink, Operation, Reply, Secret, Sink, Wire,
};
use serde::{Deserialize, Deserializer, Serialize};

use super::streaming::{ResponsesDecoder, ResponsesEvent, ResponsesStreamOptions};
use super::{
    CompletionRequest, CompletionResponse, Include, ResponsesRequestParams,
    ResponsesToolDefinition, SystemInstructionsPlacement,
};

/// How a Responses-format provider differs from OpenAI: data only.
///
/// Serialized by `name` and deserialized by looking that name up in
/// [`Dialect::by_name`]: a dialect is an *identity*, not a payload, so a
/// host storing a wire cannot reconstitute one with somebody else's base
/// URL. An unknown name is an error rather than a silent default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dialect {
    /// The provider descriptor name, as records and telemetry spell it.
    pub name: &'static str,
    /// The default base URL.
    pub base_url: &'static str,
    /// The environment variable carrying the credential.
    pub api_key_env: &'static str,
    /// The environment variable overriding the base URL, when the provider
    /// documents one.
    pub base_url_env: Option<&'static str>,
    /// The reply header carrying the provider's transport request id.
    pub request_id_header: Option<&'static str>,
    /// What this gateway does differently.
    pub quirks: Quirks,
}

/// The behaviours a Responses-format gateway varies in.
///
/// Every field has a caller: each is something a recorded cassette shows one
/// of the three dialects doing and the others not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Quirks {
    /// The endpoint path, appended to the base URL.
    pub path: &'static str,
    /// Where Rig's system instructions go in the request.
    pub system_instructions: SystemInstructionsPlacement,
    /// Which request body this gateway accepts.
    pub request: RequestShape,
    /// A second environment variable naming the base URL, kept because the
    /// provider documents both spellings.
    pub base_url_env_alias: Option<&'static str>,
    /// The environment variable naming the account a credential belongs to,
    /// sent as `ChatGPT-Account-Id`.
    pub account_id_env: Option<&'static str>,
    /// Instructions this gateway expects every turn to carry, merged ahead
    /// of the caller's preamble.
    pub default_instructions: Option<&'static str>,
    /// The environment variable overriding [`Self::default_instructions`].
    pub instructions_env: Option<&'static str>,
    /// The caller identity this gateway requires on every request.
    pub identity: Option<Identity>,
    /// The gateway answers with an event stream whether or not a stream was
    /// asked for, so a unary call reads a replayed SSE body.
    pub always_streams: bool,
    /// The gateway's streamed reply may name no content type at all.
    pub relaxed_content_type: bool,
    /// The gateway accepts only the codex parameter subset: no sampling
    /// controls, no storage, no metadata, no structured output.
    pub codex_parameter_subset: bool,
    /// The gateway answers a success with its error envelope as the whole
    /// body, and publishes a finished tool call at `output_item.done`. The
    /// stream's own `error` event is not this: that is protocol on every
    /// dialect and the decoder always reads it.
    pub error_envelope_in_success: bool,
    /// The gateway's replayed frames may omit their envelope bookkeeping
    /// (`sequence_number`, `output_index`, …), which the typed decode
    /// salvages. Off elsewhere: on a gateway whose frames do carry
    /// envelopes, an envelope-less frame is a defect worth surfacing.
    pub repair_envelope_less_frames: bool,
    /// Native structured output composes with tool calls.
    pub native_output_with_tools: bool,
}

/// Which request body a dialect accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequestShape {
    /// OpenAI's own Responses request.
    Responses,
    /// xAI's input shape, whose types live with that provider.
    Xai,
}

/// The caller identity a gateway requires on every request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Identity {
    /// The `originator` header's default value.
    pub originator: &'static str,
    /// The environment variable overriding `originator`.
    pub originator_env: &'static str,
    /// The environment variable overriding `user-agent`.
    pub user_agent_env: &'static str,
    /// Whether every request carries a fresh `session_id` header.
    pub session_ids: bool,
}

/// OpenAI itself.
pub const OPENAI: Dialect = Dialect {
    name: "openai",
    base_url: "https://api.openai.com/v1",
    api_key_env: "OPENAI_API_KEY",
    base_url_env: Some("OPENAI_BASE_URL"),
    request_id_header: Some("x-request-id"),
    quirks: Quirks {
        path: "/responses",
        system_instructions: SystemInstructionsPlacement::Instructions,
        request: RequestShape::Responses,
        base_url_env_alias: None,
        account_id_env: None,
        default_instructions: None,
        instructions_env: None,
        identity: None,
        always_streams: false,
        relaxed_content_type: false,
        codex_parameter_subset: false,
        error_envelope_in_success: false,
        repair_envelope_less_frames: false,
        native_output_with_tools: true,
    },
};

impl Dialect {
    /// The dialect this crate ships under `name`.
    pub fn by_name(name: &str) -> Option<Self> {
        [
            OPENAI,
            crate::providers::chatgpt::DIALECT,
            crate::providers::xai::DIALECT,
        ]
        .into_iter()
        .find(|dialect| dialect.name == name)
    }
}

impl Serialize for Dialect {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.name)
    }
}

impl<'de> Deserialize<'de> for Dialect {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let name = String::deserialize(deserializer)?;
        Self::by_name(&name).ok_or_else(|| {
            serde::de::Error::custom(format!("`{name}` is not a Responses-format provider"))
        })
    }
}

/// The user agent a gateway that asks for one is told: the crate, the host,
/// and who is calling.
pub(crate) fn default_user_agent(originator: &str) -> String {
    format!(
        "rig/{} ({} {}; {originator})",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS,
        std::env::consts::ARCH,
    )
}

/// The identity a gateway requires on every request, resolved.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CallerIdentity {
    /// The `originator` header.
    pub originator: String,
    /// The `user-agent` header.
    pub user_agent: String,
}

/// The shared configuration of a Responses-format provider.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponsesApi {
    /// The credential, sent as `Authorization: Bearer`.
    pub api_key: Secret,
    /// The API root the endpoint path is appended to.
    pub base_url: String,
    /// The account the credential belongs to, when the gateway asks which
    /// (`ChatGPT-Account-Id`).
    pub account_id: Option<String>,
    /// Instructions merged ahead of every turn's preamble, when the gateway
    /// expects some.
    pub instructions: Option<String>,
    /// The caller identity, when the gateway requires one.
    pub identity: Option<CallerIdentity>,
    /// Which Responses-format provider this is.
    pub dialect: Dialect,
}

impl ResponsesApi {
    /// OpenAI itself, with default settings.
    pub fn new(api_key: impl Into<Secret>) -> Self {
        Self::with_dialect(api_key, &OPENAI)
    }

    /// A Responses-format provider with its dialect's default settings.
    pub fn with_dialect(api_key: impl Into<Secret>, dialect: &Dialect) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: dialect.base_url.to_owned(),
            account_id: None,
            instructions: dialect.quirks.default_instructions.map(str::to_owned),
            identity: dialect.quirks.identity.map(|identity| CallerIdentity {
                originator: identity.originator.to_owned(),
                user_agent: default_user_agent(identity.originator),
            }),
            dialect: *dialect,
        }
    }

    /// OpenAI from `OPENAI_API_KEY` and `OPENAI_BASE_URL`.
    pub fn from_env() -> Result<Self, EnvError> {
        Self::from_env_with(&OPENAI)
    }

    /// A Responses-format provider from the variables its dialect names.
    pub fn from_env_with(dialect: &Dialect) -> Result<Self, EnvError> {
        let mut provider = Self::with_dialect(env::required(dialect.api_key_env)?, dialect);
        let quirks = &dialect.quirks;
        for name in [dialect.base_url_env, quirks.base_url_env_alias]
            .into_iter()
            .flatten()
        {
            if let Some(base_url) = env::optional(name)? {
                provider.base_url = base_url;
                break;
            }
        }
        if let Some(name) = quirks.account_id_env {
            provider.account_id = env::optional(name)?;
        }
        if let Some(name) = quirks.instructions_env
            && let Some(instructions) = env::optional(name)?
            && !instructions.trim().is_empty()
        {
            provider.instructions = Some(instructions);
        }
        if let (Some(identity), Some(resolved)) = (quirks.identity, provider.identity.as_mut()) {
            if let Some(originator) =
                env::optional(identity.originator_env)?.filter(|value| !value.is_empty())
            {
                resolved.originator = originator;
                resolved.user_agent = default_user_agent(&resolved.originator);
            }
            if let Some(user_agent) =
                env::optional(identity.user_agent_env)?.filter(|value| !value.is_empty())
            {
                resolved.user_agent = user_agent;
            }
        }
        Ok(provider)
    }

    /// Point the wire at another API root.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Name the account the credential belongs to.
    pub fn with_account_id(mut self, account_id: impl Into<String>) -> Self {
        self.account_id = Some(account_id.into());
        self
    }

    /// Merge these instructions ahead of every turn's preamble.
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// The Responses wire for `model`.
    pub fn responses(&self, model: impl Into<String>) -> Responses {
        Responses {
            system_instructions: self.dialect.quirks.system_instructions,
            provider: self.clone(),
            model: model.into(),
            tools: Vec::new(),
            strict_tools: false,
        }
    }
}

impl HasCompletion for ResponsesApi {
    type Wire = Responses;

    fn completion(&self, model: impl Into<String>) -> Responses {
        self.responses(model)
    }
}

/// The Responses wire: `POST /responses`, SSE when streamed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Responses {
    /// The provider this wire speaks to.
    pub provider: ResponsesApi,
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
        let quirks = &self.provider.dialect.quirks;
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

    fn encode(
        &self,
        request: completion::CompletionRequest,
        mode: Mode,
    ) -> Result<Encoded, CompletionError> {
        let quirks = &self.provider.dialect.quirks;
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
        let quirks = &self.provider.dialect.quirks;
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
            self.provider.dialect.quirks.native_output_with_tools,
        )
    }
}

/// Fold one whole Responses body into the normalized response.
///
/// The ONE interpreter: the decoder's unary variant synthesizes the events
/// the stream sends, and the operation's own fold turns those into the
/// response — the same two steps [`crate::driver::call`] runs, without a
/// socket.
pub(crate) fn fold_body(
    provider: &str,
    response: CompletionResponse,
) -> Result<completion::CompletionResponse, CompletionError> {
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
