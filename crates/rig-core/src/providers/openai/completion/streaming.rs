use crate::telemetry::{CompletionOperation, CompletionSpanBuilder};
use http::Request;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::completion::{CompletionError, CompletionRequest};
use crate::http_client::HttpClientExt;
use crate::json_utils::{self, merge};
use crate::providers::internal::openai_chat_completions_compatible::{
    self, CompatibleChoiceData, CompatibleChunk, CompatibleFinishReason, CompatibleStreamProfile,
    CompatibleTerminal, CompatibleToolCallChunk,
};
use crate::providers::internal::wire;
use crate::providers::openai::completion::{
    CompletionModelOptions, GenericCompletionModel, OpenAICompatibleProvider, Usage,
};
use crate::streaming::{self, RawStreamingResult, StreamFinal};

// ================================================================
// OpenAI Completion Streaming API
// ================================================================
#[derive(Default, Deserialize, Debug)]
pub(crate) struct StreamingFunction {
    pub(crate) name: Option<String>,
    #[serde(
        default,
        deserialize_with = "crate::json_utils::deserialize_json_string_or_value"
    )]
    pub(crate) arguments: Option<String>,
}

#[derive(Deserialize, Debug)]
pub(crate) struct StreamingToolCall {
    // Optional in several compatible dialects (e.g. Mistral); missing means
    // a single in-flight tool call.
    #[serde(default)]
    pub(crate) index: usize,
    pub(crate) id: Option<String>,
    #[serde(default, deserialize_with = "json_utils::null_or_default")]
    pub(crate) function: StreamingFunction,
}

impl From<&StreamingToolCall> for CompatibleToolCallChunk {
    fn from(value: &StreamingToolCall) -> Self {
        Self {
            index: value.index,
            id: value.id.clone(),
            name: value.function.name.clone(),
            arguments: value.function.arguments.clone(),
        }
    }
}

fn deserialize_delta_content<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    // Some compatible providers (e.g. Mistral's reasoning models) stream
    // delta content as an array of content parts rather than a string.
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    Ok(value.and_then(|value| match value {
        serde_json::Value::String(text) => Some(text),
        serde_json::Value::Array(parts) => {
            let text = crate::providers::openai::completion::joined_text_parts(&parts);
            (!text.is_empty()).then_some(text)
        }
        _ => None,
    }))
}

#[derive(Deserialize, Debug, Default)]
struct StreamingDelta {
    #[serde(default, deserialize_with = "deserialize_delta_content")]
    content: Option<String>,
    /// A structured-output refusal streams here, on its own key, with
    /// `content` held at `null` for the whole turn — the same sibling-of-
    /// `content` spelling the unary path sees. Its deltas are the turn's
    /// visible text, so they join the text stream (see [`delta_text`]).
    #[serde(default)]
    refusal: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    // Not part of the official OpenAI API; some compatible providers (e.g.
    // Groq) send the same payload under `reasoning`. A separate field rather
    // than a serde alias so a delta carrying BOTH keys is not a
    // duplicate-field error that drops the whole chunk.
    #[serde(default)]
    reasoning: Option<String>,
    #[serde(default, deserialize_with = "json_utils::null_or_default")]
    tool_calls: Vec<StreamingToolCall>,
    #[serde(default, deserialize_with = "json_utils::null_or_default")]
    reasoning_details: Vec<serde_json::Value>,
}

#[derive(Deserialize, Debug, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    ToolCalls,
    Stop,
    ContentFilter,
    Length,
    #[serde(untagged)]
    Other(String), // This will handle the deprecated function_call
}

impl FinishReason {
    /// This reason in the provider's own wire spelling.
    ///
    /// Round-tripping through the wire form keeps `map_openai_finish_reason`
    /// the single place the OpenAI-compatible vocabulary is interpreted, so the
    /// streaming and unary paths cannot drift — including on the deprecated
    /// `function_call` spelling, which this enum captures in
    /// [`FinishReason::Other`].
    fn as_wire(&self) -> &str {
        match self {
            Self::ToolCalls => "tool_calls",
            Self::Stop => "stop",
            Self::ContentFilter => "content_filter",
            Self::Length => "length",
            Self::Other(other) => other,
        }
    }
}

/// Normalize a streamed OpenAI-compatible `finish_reason` field.
///
/// A missing value — or an empty one, as some gateways send — is reported as
/// [`CompatibleFinishReason::Absent`]; anything outside the normalized
/// vocabulary is preserved verbatim in
/// [`crate::completion::FinishReason::Other`].
#[cfg(test)]
pub(crate) fn map_finish_reason(reason: Option<&FinishReason>) -> CompatibleFinishReason {
    CompatibleFinishReason::from_wire(reason.map(FinishReason::as_wire))
}

/// The visible text a delta carries: its `content`, or — when `content` has
/// none — its `refusal`.
///
/// A refusal turn streams `"content": null` beside the refusal deltas (and
/// opens with an empty `"refusal": ""`), so preferring non-empty content keeps
/// ordinary turns byte-identical while letting a refusal reach the caller
/// instead of vanishing. An empty `content` string with no refusal to fall
/// back on stays exactly as it was.
fn delta_text(delta: &StreamingDelta) -> Option<String> {
    match delta.content.as_deref() {
        Some(content) if !content.is_empty() => delta.content.clone(),
        content => delta
            .refusal
            .clone()
            .filter(|refusal| !refusal.is_empty())
            .or_else(|| content.map(str::to_owned)),
    }
}

#[derive(Deserialize, Debug)]
struct StreamingChoice {
    // Defaulted because a choice on the wire is not guaranteed to carry a
    // delta: Azure prepends a `prompt_filter_results` chunk (delta-less
    // choice) to every stream when content filtering is enabled. An empty
    // delta with no finish reason is a no-op frame, matching how the
    // reference SDKs treat it (skip at consumption, never an error).
    #[serde(default)]
    delta: StreamingDelta,
    finish_reason: Option<FinishReason>,
    /// Upstream provider spelling forwarded by gateways such as OpenRouter.
    /// Direct providers omit it; their profile's default mapper ignores it.
    native_finish_reason: Option<String>,
    /// Which candidate this delta belongs to when the caller asked for
    /// `n > 1`. Optional because providers streaming a single candidate may
    /// omit it; absent is read as candidate 0.
    #[serde(default)]
    index: Option<usize>,
    /// Per-token probabilities for this chunk. Kept as provider metadata:
    /// OpenAI-compatible services extend the object independently, while the
    /// raw terminal response must retain every chunk rather than choosing a
    /// provider-specific token schema here.
    #[serde(
        default,
        deserialize_with = "crate::message::optional_additional_params"
    )]
    logprobs: Option<crate::message::AdditionalParams>,
}

#[derive(Deserialize, Debug)]
struct StreamingCompletionChunk<U = Usage> {
    id: Option<String>,
    model: Option<String>,
    choices: Vec<StreamingChoice>,
    usage: Option<U>,
    /// Provider-specific top-level chunk fields. Chat-completions-compatible
    /// services add fields independently (`service_tier`, `provider`, and
    /// similar metadata), and `raw_stream` must not erase them merely because
    /// the shared wire shape does not know their names yet.
    #[serde(flatten)]
    additional_params: serde_json::Map<String, serde_json::Value>,
}

/// Final streaming response. `U` is the provider's streaming usage payload
/// ([`Usage`] for OpenAI itself; providers with richer usage accounting, e.g.
/// Mistral and DeepSeek, substitute their own via
/// [`OpenAICompatibleProvider::StreamingUsage`]).
///
/// This is the provider-native terminal record yielded by
/// [`GenericCompletionModel::raw_stream`]. The normalized path maps it into a
/// [`StreamFinal`] exactly once, through
/// [`normalize_stream`](crate::streaming::normalize_stream).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamingCompletionResponse<U = Usage> {
    /// Usage reported on the stream's terminal event.
    pub usage: U,
    /// Why the model stopped generating, when the stream reported it.
    ///
    /// Normalized out of the OpenAI-compatible `finish_reason` vocabulary, with
    /// unrecognized values preserved verbatim. The `Stop` -> `ToolCalls`
    /// upgrade is deliberately *not* applied here: it belongs to
    /// [`normalize_stream`](crate::streaming::normalize_stream), the only place
    /// that sees which tool calls the stream actually emitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<crate::completion::FinishReason>,
    /// Provider-assigned response identifier, when the stream emitted one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    /// Provider-reported model identifier, when the stream emitted one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// The transport request id from the SSE connection's `x-request-id`
    /// response header — not part of any stream frame; stamped by the
    /// transport. `None` when the provider did not report one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_request_id: Option<String>,
    /// Token log probabilities accumulated from all primary-choice chunks.
    ///
    /// This stays provider-native on [`GenericCompletionModel::raw_stream`]:
    /// normalized completions do not currently model log probabilities, just
    /// as the blocking normalized path omits `Choice::logprobs` while its raw
    /// response retains them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<serde_json::Value>,
    /// Provider-specific top-level fields accumulated from the stream's
    /// chunks, such as OpenAI's `service_tier` and `system_fingerprint` or
    /// OpenRouter's routed `provider`.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "crate::message::optional_additional_params"
    )]
    pub additional_params: Option<crate::message::AdditionalParams>,
}

impl<U> StreamingCompletionResponse<U> {
    /// Create a terminal record carrying `usage`; the optional metadata starts
    /// unset.
    pub fn new(usage: U) -> Self {
        Self {
            usage,
            finish_reason: None,
            response_id: None,
            model: None,
            provider_request_id: None,
            logprobs: None,
            additional_params: None,
        }
    }

    /// Build the terminal record from the shared streaming layer's terminal
    /// state.
    pub(crate) fn from_terminal(terminal: CompatibleTerminal<U>) -> Self {
        Self {
            usage: terminal.usage,
            finish_reason: terminal.finish_reason,
            response_id: terminal.response_id,
            model: terminal.model,
            // Stamped by the transport layer; the shared chunk accumulator
            // never sees connection headers.
            provider_request_id: None,
            logprobs: terminal.logprobs.map(Into::into),
            additional_params: terminal.additional_params,
        }
    }
}

/// Normalize an OpenAI-compatible streaming terminal record.
///
/// As on the unary path, the provider descriptor name is an *input* rather than
/// a constant: this terminal record is shared by every OpenAI-compatible
/// provider, so baking in `"openai"` here would mislabel Groq, Together,
/// DeepSeek and the rest.
impl<U> From<(&str, StreamingCompletionResponse<U>)> for StreamFinal
where
    U: Into<crate::completion::Usage>,
{
    fn from((provider, response): (&str, StreamingCompletionResponse<U>)) -> Self {
        StreamFinal::new(provider, response.usage.into())
            .with_optional_finish_reason(response.finish_reason)
            .with_optional_response_id(response.response_id)
            .with_optional_provider_request_id(response.provider_request_id)
            .with_optional_model(response.model)
    }
}

impl<Ext, H> GenericCompletionModel<Ext, H>
where
    crate::client::Client<Ext, H>: HttpClientExt + Clone + 'static,
    Ext: crate::client::Provider
        + OpenAICompatibleProvider
        + Clone
        + crate::wasm_compat::WasmCompatSend
        + 'static,
{
    /// Open a chat-completions stream whose terminal record stays
    /// provider-native.
    ///
    /// This is the escape hatch for provider-specific terminal fields rig does
    /// not normalize. It shares the request builder, transport, telemetry, and
    /// error handling with
    /// [`CompletionModel::stream`](crate::completion::CompletionModel::stream),
    /// which calls it and normalizes the terminal record — one network request
    /// either way.
    pub async fn raw_stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<RawStreamingResult<StreamingCompletionResponse<Ext::StreamingUsage>>, CompletionError>
    {
        let preamble = completion_request.system_instructions().map(str::to_owned);
        let record_telemetry_content = completion_request.record_telemetry_content;
        let options = CompletionModelOptions {
            strict_tools: self.strict_tools,
            tool_result_array_content: self.tool_result_array_content,
            prompt_caching: self.prompt_caching,
        };
        let mut request = self.client.ext().build_completion_request(
            self.model.clone(),
            completion_request,
            options,
        )?;
        self.client.ext().prepare_request(&mut request)?;

        // Deliberately the configured model, not the per-request override:
        // Azure's deployment URL is pinned to the model handle.
        let path = self.client.ext().completion_path(&self.model);
        let resolved_model = request.model.clone();
        let modern_output_cap = self.sends_modern_output_cap(&request.model);
        let mut request_as_json =
            crate::providers::openai::completion::request_body(&request, modern_output_cap)?;

        // `merge` is shallow, so include_usage is inserted into any
        // caller-supplied stream_options rather than merged over it: the
        // caller's keys survive and the usage chunk is still requested.
        if Ext::STREAM_INCLUDE_USAGE {
            match request_as_json.get_mut("stream_options") {
                Some(serde_json::Value::Object(options)) => {
                    options
                        .entry("include_usage")
                        .or_insert(serde_json::Value::Bool(true));
                }
                Some(_) => {}
                None => {
                    request_as_json = merge(
                        request_as_json,
                        json!({"stream_options": {"include_usage": true}}),
                    );
                }
            }
        }
        request_as_json = merge(request_as_json, json!({"stream": true}));
        self.client
            .ext()
            .finalize_request_body_with_options(&mut request_as_json, options)?;

        crate::providers::internal::trace_json(
            crate::providers::internal::LogTarget::Completions,
            "OpenAI Chat Completions streaming completion request",
            &request_as_json,
        );

        let req_body = serde_json::to_vec(&request_as_json)?;

        let req = self
            .client
            .post(&path)?
            .body(req_body)
            .map_err(|e| CompletionError::HttpError(e.into()))?;

        let span = CompletionSpanBuilder::new(
            Ext::PROVIDER_NAME,
            &resolved_model,
            CompletionOperation::Chat,
        )
        .system_instructions(preamble.as_deref(), record_telemetry_content)
        .build();

        let client = self.client.clone();

        tracing::Instrument::instrument(
            openai_chat_completions_compatible::send_compatible_raw_streaming_request(
                client,
                req,
                Ext::REQUEST_ID_HEADER,
                OpenAICompatibleProfile::<Ext, Ext::StreamingUsage> {
                    provider: self.client.ext().clone(),
                    emits_complete_single_chunk_tool_calls:
                        Ext::EMITS_COMPLETE_SINGLE_CHUNK_TOOL_CALLS,
                    usage: std::marker::PhantomData,
                },
            ),
            span,
        )
        .await
    }

    /// Open a chat-completions stream with a normalized terminal record.
    ///
    /// Delegates to [`raw_stream`](Self::raw_stream) and maps only its terminal
    /// record; every incremental event passes through untouched.
    pub(crate) async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<streaming::StreamingCompletionResponse, CompletionError> {
        let stream = self.raw_stream(completion_request).await?;

        Ok(streaming::StreamingCompletionResponse::stream(
            Ext::PROVIDER_NAME,
            streaming::normalize_stream(stream, |response| {
                Ok((Ext::PROVIDER_NAME, response).into())
            }),
        ))
    }
}

#[derive(Clone, Copy, Default)]
struct OpenAICompatibleProfile<Ext = crate::providers::openai::OpenAICompletionsExt, U = Usage> {
    provider: Ext,
    emits_complete_single_chunk_tool_calls: bool,
    usage: std::marker::PhantomData<U>,
}

impl<Ext, U> CompatibleStreamProfile for OpenAICompatibleProfile<Ext, U>
where
    Ext: OpenAICompatibleProvider + Clone + crate::wasm_compat::WasmCompatSend,
    U: Clone
        + Default
        + Into<crate::completion::Usage>
        + serde::de::DeserializeOwned
        + crate::wasm_compat::WasmCompatSend
        + 'static,
{
    type Usage = U;
    type Detail = serde_json::Value;
    type FinalResponse = StreamingCompletionResponse<Self::Usage>;

    fn stamp_request_id(response: &mut Self::FinalResponse, request_id: String) {
        response.provider_request_id = Some(request_id);
    }

    fn classify_chunk(
        &self,
        data: &str,
    ) -> wire::WireEvent<CompatibleChunk<Self::Usage, Self::Detail>> {
        // Classification only — the unknown/corrupt policy (warn-skip vs.
        // in-band `Err` item) lives in the shared driver, not here.
        wire::classify_chat_completions_frame::<StreamingCompletionChunk<U>>(data).map(|data| {
            // `n > 1` streams as interleaved chunks distinguished only by
            // `choices[].index`. Taking each *chunk's* first choice would
            // concatenate every candidate into one garbled answer, while the
            // blocking path answers the same request from candidate 0 alone;
            // selecting by index keeps the two transports agreeing.
            let primary = data
                .choices
                .iter()
                .position(|choice| choice.index.is_none_or(|index| index == 0))
                .and_then(|position| data.choices.get(position))
                .map(std::slice::from_ref)
                .unwrap_or_default();

            openai_chat_completions_compatible::normalize_first_choice_chunk(
                data.id,
                data.model,
                data.usage,
                crate::message::AdditionalParams::new(data.additional_params),
                primary,
                |choice| CompatibleChoiceData {
                    // The shared mapping also folds `function_call` — the
                    // deprecated pre-tools finish reason some compatible
                    // providers still emit — onto `ToolCalls`.
                    finish_reason: match self.provider.map_streaming_finish_reason(
                        choice.finish_reason.as_ref().map(FinishReason::as_wire),
                        choice.native_finish_reason.as_deref(),
                    ) {
                        Some(reason) => CompatibleFinishReason::Reported(reason),
                        None => CompatibleFinishReason::Absent,
                    },
                    text: delta_text(&choice.delta),
                    reasoning: choice
                        .delta
                        .reasoning_content
                        .clone()
                        .or_else(|| choice.delta.reasoning.clone()),
                    tool_calls: openai_chat_completions_compatible::tool_call_chunks(
                        &choice.delta.tool_calls,
                    ),
                    details: choice.delta.reasoning_details.clone(),
                    logprobs: choice.logprobs.clone(),
                },
            )
        })
    }

    fn build_final_response(
        &self,
        terminal: CompatibleTerminal<Self::Usage>,
    ) -> Self::FinalResponse {
        StreamingCompletionResponse::from_terminal(terminal)
    }

    fn detail_reasoning(
        &self,
        detail: &Self::Detail,
    ) -> Option<(
        crate::streaming::StreamPartId,
        Option<crate::streaming::WireId>,
        crate::message::ReasoningContent,
    )> {
        self.provider.streaming_detail_reasoning(detail)
    }

    fn reasoning_signature(&self, detail: &Self::Detail) -> Option<String> {
        self.provider.streaming_reasoning_signature(detail)
    }

    fn decorate_tool_call(
        &self,
        detail: &Self::Detail,
    ) -> Option<crate::streaming::ToolCallDecoration> {
        self.provider.decorate_streaming_tool_call(detail)
    }

    fn uses_distinct_tool_call_eviction(&self) -> bool {
        true
    }

    fn emits_complete_single_chunk_tool_calls(&self) -> bool {
        self.emits_complete_single_chunk_tool_calls
    }
}

/// Send an OpenAI chat-completions streaming request, keeping the terminal
/// record provider-native.
pub(crate) async fn send_compatible_raw_streaming_request<T>(
    http_client: T,
    req: Request<Vec<u8>>,
) -> Result<RawStreamingResult<StreamingCompletionResponse<Usage>>, CompletionError>
where
    T: HttpClientExt + Clone + 'static,
{
    openai_chat_completions_compatible::send_compatible_raw_streaming_request(
        http_client,
        req,
        <crate::providers::openai::OpenAICompletionsExt as OpenAICompatibleProvider>::REQUEST_ID_HEADER,
        OpenAICompatibleProfile::<crate::providers::openai::OpenAICompletionsExt, Usage>::default(),
    )
    .await
}

/// Send an OpenAI chat-completions streaming request and normalize its terminal
/// record.
///
/// `provider` is the descriptor name to attribute the stream to. It is a
/// parameter rather than a constant because this helper is public and the
/// chat-completions wire shape is shared: hardcoding `"openai"` would label
/// every out-of-tree compatible provider's stream as OpenAI's.
pub async fn send_compatible_streaming_request<T>(
    http_client: T,
    req: Request<Vec<u8>>,
    provider: impl Into<String>,
) -> Result<streaming::StreamingCompletionResponse, CompletionError>
where
    T: HttpClientExt + Clone + 'static,
{
    let provider = provider.into();
    let stream = send_compatible_raw_streaming_request(http_client, req).await?;

    let mapper_provider = provider.clone();
    Ok(streaming::StreamingCompletionResponse::stream(
        provider,
        streaming::normalize_stream(stream, move |response| {
            Ok((mapper_provider.as_str(), response).into())
        }),
    ))
}

#[cfg(test)]
mod tests;
