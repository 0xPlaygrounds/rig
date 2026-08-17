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
        let preamble = completion_request.preamble.clone();
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
mod tests {
    use super::*;
    use crate::completion::FinishReason as NormalizedFinishReason;
    use crate::providers::internal::openai_chat_completions_compatible::test_support::{
        assert_zero_arg_tool_call_is_emitted, sse_bytes_from_data_lines,
    };

    fn streaming_request() -> http::Request<Vec<u8>> {
        http::Request::builder()
            .method("POST")
            .uri("http://localhost/v1/chat/completions")
            .body(Vec::new())
            .unwrap()
    }

    #[test]
    fn test_finish_reason_mapping_covers_every_wire_value() {
        for (wire, expected) in [
            (FinishReason::Stop, NormalizedFinishReason::Stop),
            (FinishReason::Length, NormalizedFinishReason::Length),
            (FinishReason::ToolCalls, NormalizedFinishReason::ToolCalls),
            (
                FinishReason::ContentFilter,
                NormalizedFinishReason::ContentFilter,
            ),
            // The deprecated pre-tools spelling still means a tool call.
            (
                FinishReason::Other("function_call".to_string()),
                NormalizedFinishReason::ToolCalls,
            ),
            // Some gateways report the token limit under OpenAI's older name.
            (
                FinishReason::Other("max_tokens".to_string()),
                NormalizedFinishReason::Length,
            ),
        ] {
            assert_eq!(
                map_finish_reason(Some(&wire)),
                CompatibleFinishReason::Reported(expected),
                "unexpected mapping for {wire:?}"
            );
        }
    }

    #[test]
    fn test_unknown_finish_reason_is_preserved_verbatim() {
        let wire = FinishReason::Other("GUARDRAIL_INTERVENED".to_string());

        assert_eq!(
            map_finish_reason(Some(&wire)),
            CompatibleFinishReason::Reported(NormalizedFinishReason::Other(
                "GUARDRAIL_INTERVENED".to_string()
            )),
            "an unrecognized reason must survive in the provider's own spelling"
        );
    }

    #[test]
    fn test_missing_or_empty_finish_reason_is_absent() {
        assert_eq!(map_finish_reason(None), CompatibleFinishReason::Absent);
        assert_eq!(
            map_finish_reason(Some(&FinishReason::Other(String::new()))),
            CompatibleFinishReason::Absent,
            "an empty finish_reason must not read as a provider-reported reason"
        );
    }

    /// One `choices[].delta` object, decoded from the wire.
    fn delta(wire: serde_json::Value) -> StreamingDelta {
        serde_json::from_value(wire).expect("delta should decode")
    }

    /// Replay `chunks` as an OpenAI chat-completions SSE body, returning the
    /// visible text the stream produced and its terminal record.
    async fn collect_openai_stream(
        chunks: &[&str],
    ) -> (String, Option<crate::streaming::StreamFinal>) {
        use crate::test_utils::MockStreamingClient;
        use futures::StreamExt;

        let client = MockStreamingClient {
            sse_bytes: sse_bytes_from_data_lines(
                chunks.iter().copied().chain(std::iter::once("[DONE]")),
            ),
        };
        let mut stream = send_compatible_streaming_request(client, streaming_request(), "openai")
            .await
            .expect("stream should open");

        let mut text = String::new();
        let mut terminal = None;
        while let Some(chunk) = stream.next().await {
            match chunk.expect("stream item") {
                streaming::StreamedAssistantContent::Text(chunk) => text.push_str(&chunk.text),
                streaming::StreamedAssistantContent::Final(final_record) => {
                    terminal = Some(final_record);
                }
                _ => {}
            }
        }

        (text, terminal)
    }

    /// Replay Chat Completions chunks without normalizing the terminal, so
    /// provider-native metadata can be asserted directly.
    async fn collect_openai_raw_terminal(chunks: &[&str]) -> Option<StreamingCompletionResponse> {
        use crate::test_utils::MockStreamingClient;
        use futures::StreamExt;

        let client = MockStreamingClient {
            sse_bytes: sse_bytes_from_data_lines(
                chunks.iter().copied().chain(std::iter::once("[DONE]")),
            ),
        };
        let mut stream = send_compatible_raw_streaming_request(client, streaming_request())
            .await
            .expect("raw stream should open");

        let mut terminal = None;
        while let Some(chunk) = stream.next().await {
            if let streaming::RawStreamingChoice::FinalResponse(response) =
                chunk.expect("stream item")
            {
                terminal = Some(response);
            }
        }
        terminal
    }

    /// Log probabilities are distributed across token chunks. The raw
    /// terminal must reconstruct both documented arrays in arrival order,
    /// including nested top-token arrays, instead of retaining only the last
    /// chunk or dropping the field entirely.
    #[tokio::test]
    async fn raw_terminal_accumulates_streamed_logprobs() {
        let chunks = [
            r#"{"choices":[{"index":0,"delta":{"reasoning_content":"why"},"finish_reason":null,"logprobs":{"reasoning_content":[{"token":"why","top_logprobs":[{"token":"why"}]}]}}]}"#,
            r#"{"choices":[{"index":0,"delta":{"content":"co"},"finish_reason":null,"logprobs":{"content":[{"token":"co","top_logprobs":[{"token":"co"}]}]}}]}"#,
            r#"{"choices":[{"index":0,"delta":{"content":"balt"},"finish_reason":null,"logprobs":{"content":[{"token":"balt","top_logprobs":[{"token":"balt"}]}]}}]}"#,
            r#"{"choices":[{"index":0,"delta":{},"finish_reason":"stop","logprobs":null}]}"#,
        ];

        let terminal = collect_openai_raw_terminal(&chunks)
            .await
            .expect("stream should terminate");
        assert_eq!(
            terminal.logprobs,
            Some(json!({
                "reasoning_content": [{
                    "token": "why",
                    "top_logprobs": [{"token": "why"}]
                }],
                "content": [
                    {"token": "co", "top_logprobs": [{"token": "co"}]},
                    {"token": "balt", "top_logprobs": [{"token": "balt"}]}
                ]
            }))
        );
    }

    /// Top-level metadata is not part of a choice, but it is still native
    /// response data. Compatible providers add keys independently, so the raw
    /// terminal preserves and merges both familiar and previously unknown
    /// fields instead of requiring a shared-wire release for each new key.
    #[tokio::test]
    async fn raw_terminal_retains_top_level_chunk_metadata() {
        let chunks = [
            r#"{"id":"chatcmpl-1","model":"gpt-test","object":"chat.completion.chunk","created":17,"system_fingerprint":"fp_one","service_tier":"default","provider":"OpenAI","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":null}]}"#,
            r#"{"id":"chatcmpl-1","model":"gpt-test","object":"chat.completion.chunk","created":17,"system_fingerprint":"fp_one","service_tier":"priority","provider":"OpenAI","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#,
        ];

        let terminal = collect_openai_raw_terminal(&chunks)
            .await
            .expect("stream should terminate");
        let params = terminal
            .additional_params
            .expect("top-level metadata should survive");

        assert_eq!(params["object"], "chat.completion.chunk");
        assert_eq!(params["created"], 17);
        assert_eq!(params["system_fingerprint"], "fp_one");
        assert_eq!(params["service_tier"], "priority");
        assert_eq!(params["provider"], "OpenAI");
    }

    /// Empty and null probability objects are both documented absence shapes
    /// for optional provider metadata. This is a synthetic wire test because
    /// a live model cannot be instructed to choose the empty-object spelling.
    #[test]
    fn empty_and_null_streamed_logprobs_canonicalize_to_absence() {
        for logprobs in [serde_json::Value::Null, json!({})] {
            let chunk = json!({
                "choices": [{
                    "index": 0,
                    "delta": {"content": "hi"},
                    "finish_reason": null,
                    "logprobs": logprobs
                }]
            });
            let decoded = serde_json::from_value::<StreamingCompletionChunk<Usage>>(chunk)
                .expect("an empty optional metadata shape should decode");
            assert!(
                decoded
                    .choices
                    .first()
                    .expect("the fixture has one choice")
                    .logprobs
                    .is_none()
            );
        }
    }

    /// The compatibility allowance is limited to object-or-null metadata;
    /// accepting other JSON kinds would hide a malformed provider response.
    #[test]
    fn non_object_streamed_logprobs_remain_loud() {
        for logprobs in [json!([]), json!("invalid"), json!(42)] {
            let chunk = json!({
                "choices": [{
                    "index": 0,
                    "delta": {"content": "hi"},
                    "finish_reason": null,
                    "logprobs": logprobs
                }]
            });
            assert!(
                serde_json::from_value::<StreamingCompletionChunk<Usage>>(chunk).is_err(),
                "non-object logprobs must not be silently discarded"
            );
        }
    }

    /// The refusal shape the wire actually sends: `content` held at `null` for
    /// the whole turn while the refusal arrives on its own key. Rig modeled no
    /// `refusal` field at all, so every one of these deltas was visible-text-less
    /// and a refused turn streamed nothing.
    #[test]
    fn delta_text_takes_the_refusal_when_content_is_null() {
        assert_eq!(
            delta_text(&delta(json!({ "content": null, "refusal": "I'm" }))),
            Some("I'm".to_string())
        );
        assert_eq!(
            delta_text(&delta(json!({ "refusal": " sorry" }))),
            Some(" sorry".to_string())
        );
    }

    /// The turn's opening delta carries `"refusal": ""` beside the assistant
    /// role; an empty refusal is not text.
    #[test]
    fn delta_text_ignores_the_opening_empty_refusal() {
        assert_eq!(
            delta_text(&delta(
                json!({ "role": "assistant", "content": null, "refusal": "" })
            )),
            None
        );
    }

    /// Ordinary content deltas are untouched, including the empty-string form
    /// some gateways send.
    #[test]
    fn delta_text_prefers_content_and_leaves_it_unchanged() {
        assert_eq!(
            delta_text(&delta(json!({ "content": "hello" }))),
            Some("hello".to_string())
        );
        assert_eq!(
            delta_text(&delta(json!({ "content": "" }))),
            Some(String::new())
        );
        assert_eq!(delta_text(&delta(json!({}))), None);
    }

    /// A delta carrying both keys is not a shape OpenAI has been observed to
    /// send; within a delta, content wins so the visible answer is never
    /// displaced.
    ///
    /// This rule is per-delta, and deliberately so — a stream cannot know
    /// whether text arrives later without buffering the turn. The unary
    /// path's `assistant_refusal_fallback` is a *whole-message* rule, so on a
    /// hypothetical turn that mixed text and a refusal across deltas the two
    /// would differ: blocking would report only the text, streaming both in
    /// arrival order. Recorded here rather than claimed away; no observed
    /// turn mixes them, because a refusal turn holds `content` at `null` for
    /// its whole length.
    #[test]
    fn delta_text_prefers_content_over_a_simultaneous_refusal() {
        assert_eq!(
            delta_text(&delta(json!({ "content": "answer", "refusal": "no" }))),
            Some("answer".to_string())
        );
        assert_eq!(
            delta_text(&delta(json!({ "content": "", "refusal": "no" }))),
            Some("no".to_string()),
            "an empty content string must not suppress a real refusal"
        );
    }

    /// The whole refusal turn, assembled: the deltas concatenate into the same
    /// text the blocking path reports, and the terminal is a clean `stop`.
    #[tokio::test]
    async fn refusal_only_stream_delivers_the_refusal_text() {
        let chunks = [
            r#"{"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":null,"refusal":""},"finish_reason":null}]}"#,
            r#"{"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"delta":{"refusal":"I'm sorry"},"finish_reason":null}]}"#,
            r#"{"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"delta":{"refusal":", I can't help."},"finish_reason":null}]}"#,
            r#"{"id":"chatcmpl-1","model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
            r#"{"id":"chatcmpl-1","model":"gpt-4o","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":8,"total_tokens":18}}"#,
        ];

        let (text, terminal) = collect_openai_stream(&chunks).await;

        assert_eq!(text, "I'm sorry, I can't help.");
        let terminal = terminal.expect("a refusal turn still ends with a terminal record");
        assert_eq!(terminal.finish_reason, Some(NormalizedFinishReason::Stop));
        assert_eq!(terminal.usage.output_tokens, 8);
    }

    #[test]
    fn test_streaming_function_deserialization() {
        let json = r#"{"name": "get_weather", "arguments": "{\"location\":\"Paris\"}"}"#;
        let function: StreamingFunction = serde_json::from_str(json).unwrap();
        assert_eq!(function.name, Some("get_weather".to_string()));
        assert_eq!(
            function.arguments.as_ref().unwrap(),
            r#"{"location":"Paris"}"#
        );
    }

    #[test]
    fn test_streaming_function_object_arguments() {
        // Some OpenAI-compatible gateways send `arguments` as a JSON object
        // instead of the spec-mandated JSON-encoded string. Accept it by
        // re-serializing to the string form rather than dropping the chunk.
        let json = r#"{"name": "list_dir", "arguments": {}}"#;
        let function: StreamingFunction = serde_json::from_str(json).unwrap();
        assert_eq!(function.name, Some("list_dir".to_string()));
        assert_eq!(function.arguments.as_ref().unwrap(), "{}");

        let json = r#"{"name": "get_weather", "arguments": {"city": "London"}}"#;
        let function: StreamingFunction = serde_json::from_str(json).unwrap();
        assert_eq!(function.arguments.as_ref().unwrap(), r#"{"city":"London"}"#);
    }

    #[test]
    fn test_streaming_function_null_arguments() {
        let json = r#"{"name": "list_dir", "arguments": null}"#;
        let function: StreamingFunction = serde_json::from_str(json).unwrap();
        assert!(function.arguments.is_none());

        let json = r#"{"name": "list_dir"}"#;
        let function: StreamingFunction = serde_json::from_str(json).unwrap();
        assert!(function.arguments.is_none());
    }

    #[test]
    fn test_streaming_tool_call_deserialization() {
        let json = r#"{
            "index": 0,
            "id": "call_abc123",
            "function": {
                "name": "get_weather",
                "arguments": "{\"city\":\"London\"}"
            }
        }"#;
        let tool_call: StreamingToolCall = serde_json::from_str(json).unwrap();
        assert_eq!(tool_call.index, 0);
        assert_eq!(tool_call.id, Some("call_abc123".to_string()));
        assert_eq!(tool_call.function.name, Some("get_weather".to_string()));
    }

    #[test]
    fn test_streaming_tool_call_partial_deserialization() {
        // Partial tool calls have no name and partial arguments
        let json = r#"{
            "index": 0,
            "id": null,
            "function": {
                "name": null,
                "arguments": "Paris"
            }
        }"#;
        let tool_call: StreamingToolCall = serde_json::from_str(json).unwrap();
        assert_eq!(tool_call.index, 0);
        assert!(tool_call.id.is_none());
        assert!(tool_call.function.name.is_none());
        assert_eq!(tool_call.function.arguments.as_ref().unwrap(), "Paris");
    }

    #[test]
    fn test_streaming_tool_call_missing_function_deserialization() {
        let json = r#"{
            "index": 0,
            "id": "call_abc123"
        }"#;
        let tool_call: StreamingToolCall = serde_json::from_str(json).unwrap();
        assert_eq!(tool_call.index, 0);
        assert_eq!(tool_call.id, Some("call_abc123".to_string()));
        assert!(tool_call.function.name.is_none());
        assert!(tool_call.function.arguments.is_none());
    }

    #[test]
    fn test_streaming_tool_call_null_function_deserialization() {
        let json = r#"{
            "index": 0,
            "id": "call_abc123",
            "function": null
        }"#;
        let tool_call: StreamingToolCall = serde_json::from_str(json).unwrap();
        assert_eq!(tool_call.index, 0);
        assert_eq!(tool_call.id, Some("call_abc123".to_string()));
        assert!(tool_call.function.name.is_none());
        assert!(tool_call.function.arguments.is_none());
    }

    #[test]
    fn test_streaming_delta_with_tool_calls() {
        let json = r#"{
            "content": null,
            "tool_calls": [{
                "index": 0,
                "id": "call_xyz",
                "function": {
                    "name": "search",
                    "arguments": ""
                }
            }]
        }"#;
        let delta: StreamingDelta = serde_json::from_str(json).unwrap();
        assert!(delta.content.is_none());
        assert_eq!(delta.tool_calls.len(), 1);
        assert_eq!(delta.tool_calls[0].id, Some("call_xyz".to_string()));
    }

    #[test]
    fn test_streaming_delta_with_null_tool_calls() {
        let json = r#"{
            "content": "Hello",
            "tool_calls": null
        }"#;
        let delta: StreamingDelta = serde_json::from_str(json).unwrap();
        assert_eq!(delta.content, Some("Hello".to_string()));
        assert!(delta.tool_calls.is_empty());
    }

    #[test]
    fn test_streaming_chunk_deserialization() {
        let json = r#"{
            "choices": [{
                "delta": {
                    "content": "Hello",
                    "tool_calls": []
                }
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }"#;
        let chunk: StreamingCompletionChunk = serde_json::from_str(json).unwrap();
        assert_eq!(chunk.choices.len(), 1);
        assert_eq!(chunk.choices[0].delta.content, Some("Hello".to_string()));
        assert!(chunk.usage.is_some());
    }

    #[test]
    fn test_streaming_chunk_with_multiple_tool_call_deltas() {
        // Simulates multiple partial tool call chunks arriving
        let json_start = r#"{
            "choices": [{
                "delta": {
                    "content": null,
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_123",
                        "function": {
                            "name": "get_weather",
                            "arguments": ""
                        }
                    }]
                }
            }],
            "usage": null
        }"#;

        let json_chunk1 = r#"{
            "choices": [{
                "delta": {
                    "content": null,
                    "tool_calls": [{
                        "index": 0,
                        "id": null,
                        "function": {
                            "name": null,
                            "arguments": "{\"loc"
                        }
                    }]
                }
            }],
            "usage": null
        }"#;

        let json_chunk2 = r#"{
            "choices": [{
                "delta": {
                    "content": null,
                    "tool_calls": [{
                        "index": 0,
                        "id": null,
                        "function": {
                            "name": null,
                            "arguments": "ation\":\"NYC\"}"
                        }
                    }]
                }
            }],
            "usage": null
        }"#;

        // Verify each chunk deserializes correctly
        let start_chunk: StreamingCompletionChunk = serde_json::from_str(json_start).unwrap();
        assert_eq!(start_chunk.choices[0].delta.tool_calls.len(), 1);
        assert_eq!(
            start_chunk.choices[0].delta.tool_calls[0]
                .function
                .name
                .as_ref()
                .unwrap(),
            "get_weather"
        );

        let chunk1: StreamingCompletionChunk = serde_json::from_str(json_chunk1).unwrap();
        assert_eq!(chunk1.choices[0].delta.tool_calls.len(), 1);
        assert_eq!(
            chunk1.choices[0].delta.tool_calls[0]
                .function
                .arguments
                .as_ref()
                .unwrap(),
            "{\"loc"
        );

        let chunk2: StreamingCompletionChunk = serde_json::from_str(json_chunk2).unwrap();
        assert_eq!(chunk2.choices[0].delta.tool_calls.len(), 1);
        assert_eq!(
            chunk2.choices[0].delta.tool_calls[0]
                .function
                .arguments
                .as_ref()
                .unwrap(),
            "ation\":\"NYC\"}"
        );
    }

    #[tokio::test]
    async fn test_streaming_usage_only_chunk_is_not_ignored() {
        use crate::test_utils::MockStreamingClient;
        use futures::StreamExt;

        // Some providers emit a final "usage-only" chunk where `choices` is empty.
        let client = MockStreamingClient {
            sse_bytes: sse_bytes_from_data_lines([
                "{\"choices\":[{\"delta\":{\"content\":\"Hello\",\"tool_calls\":[]}}],\"usage\":null}",
                "{\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15}}",
                "[DONE]",
            ]),
        };

        let mut stream = send_compatible_streaming_request(client, streaming_request(), "openai")
            .await
            .unwrap();

        let mut final_usage = None;
        while let Some(chunk) = stream.next().await {
            if let streaming::StreamedAssistantContent::Final(res) = chunk.unwrap() {
                final_usage = Some(res.usage);
                break;
            }
        }

        let usage = final_usage.expect("expected a final response with usage");
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.total_tokens, 15);
    }

    #[tokio::test]
    async fn test_streaming_final_record_carries_provider_metadata() {
        use crate::test_utils::MockStreamingClient;
        use futures::StreamExt;

        let client = MockStreamingClient {
            sse_bytes: sse_bytes_from_data_lines([
                "{\"id\":\"chatcmpl-42\",\"model\":\"gpt-5.2-2026-01-01\",\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}],\"usage\":null}",
                "{\"id\":\"chatcmpl-42\",\"model\":\"gpt-5.2-2026-01-01\",\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}],\"usage\":null}",
                "[DONE]",
            ]),
        };

        let mut stream = send_compatible_streaming_request(client, streaming_request(), "openai")
            .await
            .unwrap();

        let mut final_response = None;
        while let Some(chunk) = stream.next().await {
            if let streaming::StreamedAssistantContent::Final(res) = chunk.unwrap() {
                final_response = Some(res);
                break;
            }
        }

        let res = final_response.expect("expected a final response");
        assert_eq!(res.provider, "openai");
        assert_eq!(res.response_id.as_deref(), Some("chatcmpl-42"));
        assert_eq!(res.message_id, None);
        assert_eq!(res.model.as_deref(), Some("gpt-5.2-2026-01-01"));
        assert_eq!(res.finish_reason, Some(NormalizedFinishReason::Length));
    }

    #[tokio::test]
    async fn test_streaming_unknown_finish_reason_reaches_the_final_record() {
        use crate::test_utils::MockStreamingClient;
        use futures::StreamExt;

        let client = MockStreamingClient {
            sse_bytes: sse_bytes_from_data_lines([
                "{\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}],\"usage\":null}",
                "{\"choices\":[{\"delta\":{},\"finish_reason\":\"GUARDRAIL_INTERVENED\"}],\"usage\":null}",
                "[DONE]",
            ]),
        };

        let mut stream = send_compatible_streaming_request(client, streaming_request(), "openai")
            .await
            .unwrap();

        let mut final_response = None;
        while let Some(chunk) = stream.next().await {
            if let streaming::StreamedAssistantContent::Final(res) = chunk.unwrap() {
                final_response = Some(res);
                break;
            }
        }

        let res = final_response.expect("expected a final response");
        assert_eq!(
            res.finish_reason,
            Some(NormalizedFinishReason::Other(
                "GUARDRAIL_INTERVENED".to_string()
            ))
        );
    }

    /// A `stop` reported on a turn that streamed a tool call must surface as
    /// `ToolCalls`. The provider mapper deliberately does not do this — the
    /// upgrade belongs to `normalize_stream`, which sees the emitted tool
    /// calls — so this pins the wiring rather than the mapping.
    #[tokio::test]
    async fn test_stop_finish_reason_upgrades_to_tool_calls() {
        use crate::test_utils::MockStreamingClient;
        use futures::StreamExt;

        let client = MockStreamingClient {
            sse_bytes: sse_bytes_from_data_lines([
                "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"ping\",\"arguments\":\"{}\"}}]},\"finish_reason\":null}],\"usage\":null}",
                "{\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":null}",
                "[DONE]",
            ]),
        };

        let mut stream = send_compatible_streaming_request(client, streaming_request(), "openai")
            .await
            .unwrap();

        let mut saw_tool_call = false;
        let mut final_response = None;
        while let Some(chunk) = stream.next().await {
            match chunk.unwrap() {
                streaming::StreamedAssistantContent::ToolCall { .. } => saw_tool_call = true,
                streaming::StreamedAssistantContent::Final(res) => final_response = Some(res),
                _ => {}
            }
        }

        assert!(saw_tool_call, "expected the tool call to be emitted");
        let res = final_response.expect("expected a final response");
        assert_eq!(res.finish_reason, Some(NormalizedFinishReason::ToolCalls));
    }

    #[tokio::test]
    async fn test_streaming_reasoning_content_and_text_chunks_are_incremental() {
        use crate::test_utils::MockStreamingClient;
        use futures::StreamExt;

        let client = MockStreamingClient {
            sse_bytes: sse_bytes_from_data_lines([
                "{\"id\":\"cmpl-1\",\"model\":\"Qwen/Qwen3-4B\",\"choices\":[{\"delta\":{\"reasoning_content\":\"think \",\"tool_calls\":[]},\"finish_reason\":null}],\"usage\":null}",
                "{\"id\":\"cmpl-1\",\"model\":\"Qwen/Qwen3-4B\",\"choices\":[{\"delta\":{\"reasoning_content\":\"more\",\"tool_calls\":[]},\"finish_reason\":null}],\"usage\":null}",
                "{\"id\":\"cmpl-1\",\"model\":\"Qwen/Qwen3-4B\",\"choices\":[{\"delta\":{\"content\":\"hel\",\"tool_calls\":[]},\"finish_reason\":null}],\"usage\":null}",
                "{\"id\":\"cmpl-1\",\"model\":\"Qwen/Qwen3-4B\",\"choices\":[{\"delta\":{\"content\":\"lo\",\"tool_calls\":[]},\"finish_reason\":\"stop\"}],\"usage\":null}",
                "{\"choices\":[],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":6,\"total_tokens\":10}}",
                "[DONE]",
            ]),
        };

        let mut stream = send_compatible_streaming_request(client, streaming_request(), "openai")
            .await
            .unwrap();

        let mut reasoning_chunks = Vec::new();
        let mut text_chunks = Vec::new();
        let mut final_response = None;

        while let Some(chunk) = stream.next().await {
            match chunk.unwrap() {
                streaming::StreamedAssistantContent::ReasoningDelta { reasoning, .. } => {
                    reasoning_chunks.push(reasoning)
                }
                streaming::StreamedAssistantContent::Text(text) => text_chunks.push(text.text),
                streaming::StreamedAssistantContent::Final(response) => {
                    final_response = Some(response)
                }
                _ => {}
            }
        }

        assert_eq!(
            reasoning_chunks,
            vec!["think ".to_string(), "more".to_string()]
        );
        assert_eq!(text_chunks, vec!["hel".to_string(), "lo".to_string()]);

        let response = final_response.expect("expected final usage");
        assert_eq!(response.usage.input_tokens, 4);
        assert_eq!(response.usage.output_tokens, 6);
        assert_eq!(response.usage.total_tokens, 10);
        assert_eq!(response.finish_reason, Some(NormalizedFinishReason::Stop));
    }

    #[tokio::test]
    async fn test_streaming_cached_input_tokens_populated() {
        use crate::streaming::RawStreamingChoice;
        use crate::test_utils::MockStreamingClient;
        use futures::StreamExt;

        // Usage chunk includes prompt_tokens_details with cached_tokens.
        let client = MockStreamingClient {
            sse_bytes: sse_bytes_from_data_lines([
                "{\"choices\":[{\"delta\":{\"content\":\"Hi\",\"tool_calls\":[]}}],\"usage\":null}",
                "{\"choices\":[],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":10,\"total_tokens\":110,\"prompt_tokens_details\":{\"cached_tokens\":80}}}",
                "[DONE]",
            ]),
        };

        // The raw stream keeps the provider's own usage payload, so this
        // asserts both halves: what the provider reported and what it
        // normalizes into.
        let mut stream = send_compatible_raw_streaming_request(client, streaming_request())
            .await
            .unwrap();

        let mut final_response = None;
        while let Some(chunk) = stream.next().await {
            if let RawStreamingChoice::FinalResponse(res) = chunk.unwrap() {
                final_response = Some(res);
                break;
            }
        }

        let res = final_response.expect("expected a final response");

        // Verify provider-level usage has the cached_tokens
        assert_eq!(
            res.usage
                .prompt_tokens_details
                .as_ref()
                .unwrap()
                .cached_tokens,
            80
        );

        // Verify core Usage also has cached_input_tokens
        let core_usage = crate::completion::Usage::from(res.usage);
        assert_eq!(core_usage.cached_input_tokens, 80);
        assert_eq!(core_usage.input_tokens, 100);
        assert_eq!(core_usage.total_tokens, 110);
    }

    /// Reproduces the bug where a proxy/gateway sends multiple parallel tool
    /// calls all sharing `index: 0` but with distinct `id` values.  Without
    /// the fix, rig merges both calls into one corrupted entry.
    #[tokio::test]
    async fn test_duplicate_index_different_id_tool_calls() {
        use crate::test_utils::MockStreamingClient;
        use futures::StreamExt;

        // Simulate a gateway that sends two tool calls both at index 0.
        // First tool call: id="call_aaa", name="command", args={"cmd":"ls"}
        // Second tool call: id="call_bbb", name="git", args={"action":"log"}
        let client = MockStreamingClient {
            sse_bytes: sse_bytes_from_data_lines([
                "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_aaa\",\"function\":{\"name\":\"command\",\"arguments\":\"\"}}]},\"finish_reason\":null}],\"usage\":null}",
                "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":null,\"function\":{\"name\":null,\"arguments\":\"{\\\"cmd\\\"\"}}]},\"finish_reason\":null}],\"usage\":null}",
                "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":null,\"function\":{\"name\":null,\"arguments\":\":\\\"ls\\\"}\"}}]},\"finish_reason\":null}],\"usage\":null}",
                "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_bbb\",\"function\":{\"name\":\"git\",\"arguments\":\"\"}}]},\"finish_reason\":null}],\"usage\":null}",
                "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":null,\"function\":{\"name\":null,\"arguments\":\"{\\\"action\\\"\"}}]},\"finish_reason\":null}],\"usage\":null}",
                "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":null,\"function\":{\"name\":null,\"arguments\":\":\\\"log\\\"}\"}}]},\"finish_reason\":null}],\"usage\":null}",
                "{\"choices\":[{\"delta\":{\"tool_calls\":[]},\"finish_reason\":\"tool_calls\"}],\"usage\":null}",
                "{\"choices\":[],\"usage\":{\"prompt_tokens\":20,\"completion_tokens\":10,\"total_tokens\":30}}",
                "[DONE]",
            ]),
        };

        let mut stream = send_compatible_streaming_request(client, streaming_request(), "openai")
            .await
            .unwrap();

        let mut collected_tool_calls = Vec::new();
        while let Some(chunk) = stream.next().await {
            if let streaming::StreamedAssistantContent::ToolCall {
                tool_call,
                internal_call_id: _,
            } = chunk.unwrap()
            {
                collected_tool_calls.push(tool_call);
            }
        }

        assert_eq!(
            collected_tool_calls.len(),
            2,
            "expected 2 separate tool calls, got {collected_tool_calls:?}"
        );

        assert_eq!(collected_tool_calls[0].id, "call_aaa");
        assert_eq!(collected_tool_calls[0].function.name, "command");
        assert_eq!(
            collected_tool_calls[0].function.arguments,
            serde_json::json!({"cmd": "ls"})
        );

        assert_eq!(collected_tool_calls[1].id, "call_bbb");
        assert_eq!(collected_tool_calls[1].function.name, "git");
        assert_eq!(
            collected_tool_calls[1].function.arguments,
            serde_json::json!({"action": "log"})
        );
    }

    #[tokio::test]
    async fn test_tool_call_id_chunk_without_function_is_preserved() {
        use crate::test_utils::MockStreamingClient;
        use futures::StreamExt;

        let client = MockStreamingClient {
            sse_bytes: sse_bytes_from_data_lines([
                "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_abc123\"}]},\"finish_reason\":null}],\"usage\":null}",
                "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":null,\"function\":{\"name\":\"lookup\",\"arguments\":\"\"}}]},\"finish_reason\":null}],\"usage\":null}",
                "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":null,\"function\":{\"name\":null,\"arguments\":\"{\\\"id\\\":1}\"}}]},\"finish_reason\":null}],\"usage\":null}",
                "{\"choices\":[{\"delta\":{\"tool_calls\":[]},\"finish_reason\":\"tool_calls\"}],\"usage\":null}",
                "[DONE]",
            ]),
        };

        let mut stream = send_compatible_streaming_request(client, streaming_request(), "openai")
            .await
            .unwrap();

        let mut collected_tool_calls = Vec::new();
        while let Some(chunk) = stream.next().await {
            if let streaming::StreamedAssistantContent::ToolCall {
                tool_call,
                internal_call_id: _,
            } = chunk.unwrap()
            {
                collected_tool_calls.push(tool_call);
            }
        }

        assert_eq!(
            collected_tool_calls.len(),
            1,
            "expected id-only chunk to be retained for later tool-call deltas"
        );
        assert_eq!(collected_tool_calls[0].id, "call_abc123");
        assert_eq!(collected_tool_calls[0].function.name, "lookup");
        assert_eq!(
            collected_tool_calls[0].function.arguments,
            serde_json::json!({"id": 1})
        );
    }

    /// Reproduces the bug where a provider (e.g. GLM-4 via OpenAI-compatible
    /// endpoint) sends a unique `id` on every SSE delta chunk for the same
    /// logical tool call.  Without the fix, each chunk triggers an eviction,
    /// yielding incomplete fragments as "completed" tool calls.
    #[tokio::test]
    async fn test_unique_id_per_chunk_single_tool_call() {
        use crate::test_utils::MockStreamingClient;
        use futures::StreamExt;

        // Each chunk carries a different id but they all represent delta
        // fragments of the SAME tool call at index 0.
        let client = MockStreamingClient {
            sse_bytes: sse_bytes_from_data_lines([
                "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"chatcmpl-tool-aaa\",\"function\":{\"name\":\"web_search\",\"arguments\":\"null\"}}]},\"finish_reason\":null}],\"usage\":null}",
                "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"chatcmpl-tool-bbb\",\"function\":{\"name\":\"\",\"arguments\":\"{\\\"query\\\": \\\"META\"}}]},\"finish_reason\":null}],\"usage\":null}",
                "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"chatcmpl-tool-ccc\",\"function\":{\"name\":\"\",\"arguments\":\" Platforms news\\\"}\"}}]},\"finish_reason\":null}],\"usage\":null}",
                "{\"choices\":[{\"delta\":{\"tool_calls\":[]},\"finish_reason\":\"tool_calls\"}],\"usage\":null}",
                "{\"choices\":[],\"usage\":{\"prompt_tokens\":15,\"completion_tokens\":8,\"total_tokens\":23}}",
                "[DONE]",
            ]),
        };

        let mut stream = send_compatible_streaming_request(client, streaming_request(), "openai")
            .await
            .unwrap();

        let mut collected_tool_calls = Vec::new();
        while let Some(chunk) = stream.next().await {
            if let streaming::StreamedAssistantContent::ToolCall {
                tool_call,
                internal_call_id: _,
            } = chunk.unwrap()
            {
                collected_tool_calls.push(tool_call);
            }
        }

        assert_eq!(
            collected_tool_calls.len(),
            1,
            "expected 1 tool call (all chunks are fragments of the same call), got {collected_tool_calls:?}"
        );

        assert_eq!(collected_tool_calls[0].function.name, "web_search");
        // The arguments should be the fully accumulated string, not fragments
        let args_str = match &collected_tool_calls[0].function.arguments {
            serde_json::Value::String(s) => s.clone(),
            v => v.to_string(),
        };
        assert!(
            args_str.contains("META Platforms news"),
            "expected accumulated arguments containing the full query, got: {args_str}"
        );
    }

    #[tokio::test]
    async fn test_zero_arg_tool_call_normalized_on_finish_reason() {
        use crate::test_utils::MockStreamingClient;

        let client = MockStreamingClient {
            sse_bytes: sse_bytes_from_data_lines([
                "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_123\",\"function\":{\"name\":\"ping\",\"arguments\":\"\"}}]},\"finish_reason\":null}],\"usage\":null}",
                "{\"choices\":[{\"delta\":{\"tool_calls\":[]},\"finish_reason\":\"tool_calls\"}],\"usage\":null}",
                "[DONE]",
            ]),
        };

        let stream = send_compatible_streaming_request(client, streaming_request(), "openai")
            .await
            .unwrap();

        assert_zero_arg_tool_call_is_emitted(stream, "call_123", "ping", true).await;
    }

    #[tokio::test]
    async fn test_zero_arg_tool_call_is_preserved_at_eof() {
        use crate::test_utils::MockStreamingClient;

        let client = MockStreamingClient {
            sse_bytes: sse_bytes_from_data_lines([
                "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_123\",\"function\":{\"name\":\"ping\",\"arguments\":\"\"}}]},\"finish_reason\":null}],\"usage\":null}",
            ]),
        };

        let stream = send_compatible_streaming_request(client, streaming_request(), "openai")
            .await
            .unwrap();

        // The tool call was fully delivered, so it is still flushed at EOF —
        // but the stream reached EOF without `[DONE]` or a finish reason, so
        // no terminal record is synthesized for the truncated turn.
        assert_zero_arg_tool_call_is_emitted(stream, "call_123", "ping", false).await;
    }

    /// The default OpenAI profile must not let a stream end silently: corrupt
    /// frames surface as error items, and a bare `[DONE]` with no successfully
    /// decoded frame yields no terminal record. Unknown-shaped events (no
    /// `object`/`choices`) stay skippable for forward compatibility.
    #[tokio::test]
    async fn test_default_profile_surfaces_unparseable_frames_as_errors() {
        use crate::test_utils::MockStreamingClient;
        use futures::StreamExt;

        let client = MockStreamingClient {
            sse_bytes: sse_bytes_from_data_lines([
                // Not JSON at all.
                "{bad",
                // Recognizable chat chunk with a schema defect.
                "{\"object\":\"chat.completion.chunk\",\"choices\":\"nope\"}",
                // Unknown event shape: skipped, not an error.
                "{\"type\":\"ping\"}",
                "[DONE]",
            ]),
        };

        let mut stream = send_compatible_streaming_request(client, streaming_request(), "openai")
            .await
            .unwrap();

        let mut error_count = 0;
        let mut saw_final = false;
        let mut unknown = None;
        while let Some(item) = stream.next().await {
            match item {
                Ok(streaming::StreamedAssistantContent::Final(_)) => saw_final = true,
                // The unknown-shaped event skips the semantic path but
                // surfaces verbatim on the raw passthrough channel.
                Ok(streaming::StreamedAssistantContent::Unknown(value)) => unknown = Some(value),
                Ok(other) => panic!("unexpected stream item: {other:?}"),
                Err(_) => error_count += 1,
            }
        }
        assert_eq!(unknown, Some(serde_json::json!({"type": "ping"}).into()));

        assert_eq!(
            error_count, 2,
            "each corrupt frame must surface as an error item"
        );
        assert!(
            !saw_final,
            "a stream with no successfully decoded frame must not emit a terminal record"
        );
        assert!(stream.response.is_none());
    }

    #[tokio::test]
    async fn azure_content_filter_prelude_chunk_is_a_no_op_not_an_error() {
        use crate::test_utils::MockStreamingClient;
        use futures::StreamExt;

        // Azure prepends a delta-less choice carrying `prompt_filter_results`
        // to every stream when content filtering is enabled. It must parse as
        // a no-op frame, never surface as an error item.
        let client = MockStreamingClient {
            sse_bytes: sse_bytes_from_data_lines([
                r#"{"id":"","object":"","choices":[{"prompt_index":0,"content_filter_results":{"hate":{"filtered":false,"severity":"safe"}}}]}"#,
                r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"delta":{"content":"hi"},"finish_reason":null}]}"#,
                r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}"#,
                "[DONE]",
            ]),
        };

        let mut stream = send_compatible_streaming_request(client, streaming_request(), "openai")
            .await
            .unwrap();

        let mut texts = Vec::new();
        let mut saw_final = false;
        while let Some(item) = stream.next().await {
            match item {
                Ok(streaming::StreamedAssistantContent::Text(text)) => texts.push(text.text),
                Ok(streaming::StreamedAssistantContent::Final(_)) => saw_final = true,
                Ok(_) => {}
                Err(error) => panic!("the filter prelude chunk must not error: {error}"),
            }
        }

        assert_eq!(texts, ["hi"]);
        assert!(saw_final, "the genuine terminal must still arrive");
    }

    /// Raw-capture tests for the streaming terminal, through
    /// [`send_compatible_streaming_request`] — the shared helper every
    /// OpenAI-compatible stream (and every out-of-tree compatible provider)
    /// funnels through, so the terminal it produces is the whole streaming
    /// capture story for this wire shape.
    mod raw_capture {
        use super::*;
        use crate::test_utils::MockStreamingClient;
        use futures::StreamExt;

        /// A stream whose terminal carries metadata that only the
        /// provider-native terminal keeps (`service_tier`, `system_fingerprint`
        /// under `additional_params`, plus usage and `finish_reason`).
        const CHUNKS: [&str; 3] = [
            "{\"id\":\"chatcmpl-raw-7\",\"model\":\"gpt-4o-mini-2024-07-18\",\"service_tier\":\"default\",\"system_fingerprint\":\"fp_stream\",\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}],\"usage\":null}",
            "{\"id\":\"chatcmpl-raw-7\",\"model\":\"gpt-4o-mini-2024-07-18\",\"service_tier\":\"default\",\"system_fingerprint\":\"fp_stream\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":null}",
            "{\"id\":\"chatcmpl-raw-7\",\"model\":\"gpt-4o-mini-2024-07-18\",\"service_tier\":\"default\",\"system_fingerprint\":\"fp_stream\",\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":1,\"total_tokens\":4}}",
        ];

        async fn terminal() -> streaming::StreamFinal {
            let client = MockStreamingClient {
                sse_bytes: sse_bytes_from_data_lines(
                    CHUNKS.iter().copied().chain(std::iter::once("[DONE]")),
                ),
            };
            let mut stream =
                send_compatible_streaming_request(client, streaming_request(), "openai")
                    .await
                    .expect("stream should open");

            let mut terminal = None;
            while let Some(item) = stream.next().await {
                if let streaming::StreamedAssistantContent::Final(record) =
                    item.expect("stream item")
                {
                    terminal = Some(record);
                }
            }
            terminal.expect("the stream must end with a terminal record")
        }

        /// The load-bearing streaming property: the terminal's `raw` is the
        /// provider-native terminal record — it deserializes back into
        /// [`StreamingCompletionResponse`] and re-serializes identically — and
        /// re-normalizing that capture reproduces every normalized field.
        /// Also reads terminal-only metadata off the capture.
        #[tokio::test]
        async fn terminal_captures_raw_that_round_trips_into_the_terminal_type() {
            let record = terminal().await;

            let raw = &record.raw;
            let typed: StreamingCompletionResponse =
                serde_json::from_value(raw.clone()).expect("raw must deserialize");
            assert_eq!(
                serde_json::to_value(&typed).expect("re-serialize"),
                *raw,
                "the capture must be exactly what the terminal type serializes to"
            );
            assert_eq!(typed.response_id.as_deref(), Some("chatcmpl-raw-7"));
            assert_eq!(raw["additional_params"]["service_tier"], "default");
            assert_eq!(raw["additional_params"]["system_fingerprint"], "fp_stream");

            let renormalized: streaming::StreamFinal = ("openai", typed).into();
            assert_eq!(record.identity(), renormalized.identity());
            assert_eq!(record.finish_reason, renormalized.finish_reason);
            assert_eq!(record.model, renormalized.model);
            assert_eq!(record.usage, renormalized.usage);
            assert_eq!(record.finish_reason, Some(NormalizedFinishReason::Stop));
            assert_eq!(record.model.as_deref(), Some("gpt-4o-mini-2024-07-18"));
            assert_eq!(record.usage.total_tokens, 4);
        }
    }
}
