use crate::telemetry::{CompletionOperation, CompletionSpanBuilder};
use http::Request;
use serde::Deserialize;
use serde_json::json;

use crate::completion::{CompletionError, CompletionRequest};
use crate::http_client::HttpClientExt;
use crate::json_utils::merge;
use crate::providers::internal::openai_chat_completions_compatible::{
    self, CompatibleChoiceData, CompatibleChunk, CompatibleFinishReason, CompatibleStreamProfile,
    CompatibleTerminal,
};
use crate::providers::internal::wire;
use crate::providers::openai::completion::{
    CompletionModelOptions, GenericCompletionModel, OpenAICompatibleProvider, Usage,
};
use crate::streaming::{self, StreamFinal};

// ================================================================
// OpenAI Completion Streaming API
// ================================================================
// The wire's reply DTOs, its `finish_reason` vocabulary and its terminal
// record live with the wire that decodes them
// (`providers::openai::wire::dto`). They are re-exported here so the
// public paths `openai::FinishReason` and
// `openai::StreamingCompletionResponse` are unchanged while there is one
// definition of each.
pub use crate::providers::openai::wire::dto::{FinishReason, StreamingCompletionResponse};
use crate::providers::openai::wire::dto::{StreamingDelta, delta_text};

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
    /// similar metadata), and the terminal record must not erase them merely
    /// because the shared wire shape does not know their names yet.
    #[serde(flatten)]
    additional_params: serde_json::Map<String, serde_json::Value>,
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
    /// Open a chat-completions stream with observation context owned by this
    /// invocation.
    pub(crate) async fn stream_observed(
        &self,
        completion_request: CompletionRequest,
        observation: Option<crate::observe::AdapterContext>,
    ) -> Result<streaming::StreamingCompletionResponse, CompletionError> {
        let preamble = completion_request.system_instructions().map(str::to_owned);
        let record_telemetry_content = completion_request.record_telemetry_content;
        let options = CompletionModelOptions {
            strict_tools: self.strict_tools,
            tool_result_array_content: self.tool_result_array_content,
            prompt_caching: self.prompt_caching,
        };
        let mut request = self.client.provider().build_completion_request(
            self.model.clone(),
            completion_request,
            options,
        )?;
        self.client.provider().prepare_request(&mut request)?;

        // Deliberately the configured model, not the per-request override:
        // Azure's deployment URL is pinned to the model handle.
        let path = self.client.provider().completion_path(&self.model);
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
            .provider()
            .finalize_request_body_with_options(&mut request_as_json, options)?;

        crate::providers::internal::trace_json(
            crate::providers::internal::LogTarget::Completions,
            "OpenAI Chat Completions streaming completion request",
            &request_as_json,
        );

        let req_body = serde_json::to_vec(&request_as_json)?;

        let mut req = self
            .client
            .post(&path)?
            .body(req_body)
            .map_err(|e| CompletionError::HttpError(e.into()))?;
        if let Some(observation) = observation {
            crate::providers::openai::observation::attach_chat(
                observation,
                &mut req,
                "/chat/completions",
            );
        }

        let span = CompletionSpanBuilder::new(
            Ext::PROVIDER_NAME,
            &resolved_model,
            CompletionOperation::Chat,
        )
        .system_instructions(preamble.as_deref(), record_telemetry_content)
        .build();

        let client = self.client.clone();

        let stream = tracing::Instrument::instrument(
            openai_chat_completions_compatible::send_compatible_raw_streaming_request(
                client,
                req,
                Ext::REQUEST_ID_HEADER,
                Ext::PROVIDER_NAME.to_owned(),
                OpenAICompatibleProfile::<Ext, Ext::StreamingUsage> {
                    provider: self.client.provider().clone(),
                    emits_complete_single_chunk_tool_calls:
                        Ext::EMITS_COMPLETE_SINGLE_CHUNK_TOOL_CALLS,
                    usage: std::marker::PhantomData,
                },
            ),
            span,
        )
        .await?;

        Ok(streaming::StreamingCompletionResponse::stream(
            Ext::PROVIDER_NAME,
            stream,
        ))
    }
}

#[derive(Clone, Copy, Default)]
struct OpenAICompatibleProfile<Ext = crate::providers::openai::OpenAICompletions, U = Usage> {
    provider: Ext,
    emits_complete_single_chunk_tool_calls: bool,
    usage: std::marker::PhantomData<U>,
}

impl<Ext, U> CompatibleStreamProfile for OpenAICompatibleProfile<Ext, U>
where
    Ext: OpenAICompatibleProvider + Clone + crate::wasm_compat::WasmCompatSend,
    U: Clone
        + Into<crate::completion::Usage>
        + serde::Serialize
        + serde::de::DeserializeOwned
        + crate::wasm_compat::WasmCompatSend
        + 'static,
{
    type Usage = U;
    type Detail = serde_json::Value;

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

    fn final_record(
        &self,
        provider: &str,
        terminal: CompatibleTerminal<Self::Usage>,
    ) -> Result<StreamFinal, CompletionError> {
        let native = StreamingCompletionResponse::from_terminal(terminal);
        // The provider's own terminal record rides along serialized — the
        // same capture every unary `completion` performs before `normalize`.
        let raw = serde_json::to_value(&native)?;
        Ok(native.into_stream_final(provider).with_raw(raw))
    }

    fn detail_reasoning(
        &self,
        detail: &Self::Detail,
    ) -> Option<(
        crate::streaming::BlockId,
        Option<String>,
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

/// Send an OpenAI chat-completions streaming request under the OpenAI
/// profile, attributed to `provider`.
pub(crate) async fn send_compatible_raw_streaming_request<T>(
    http_client: T,
    req: Request<Vec<u8>>,
    provider: String,
) -> Result<streaming::StreamingResult, CompletionError>
where
    T: HttpClientExt + Clone + 'static,
{
    openai_chat_completions_compatible::send_compatible_raw_streaming_request(
        http_client,
        req,
        <crate::providers::openai::OpenAICompletions as OpenAICompatibleProvider>::REQUEST_ID_HEADER,
        provider,
        OpenAICompatibleProfile::<crate::providers::openai::OpenAICompletions, Usage>::default(),
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
    let stream = send_compatible_raw_streaming_request(http_client, req, provider.clone()).await?;
    Ok(streaming::StreamingCompletionResponse::stream(
        provider, stream,
    ))
}

#[cfg(test)]
mod tests;
