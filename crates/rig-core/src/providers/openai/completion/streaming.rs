use crate::telemetry::{CompletionOperation, CompletionSpanBuilder};
use http::Request;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::{Level, enabled};

use crate::completion::{
    CompletionError, CompletionRequest, CompletionTerminalMetadata, GetCompletionMetadata,
};
use crate::http_client::HttpClientExt;
use crate::json_utils::{self, merge};
use crate::providers::internal::openai_chat_completions_compatible::{
    self, CompatibleChoiceData, CompatibleChunk, CompatibleFinishReason, CompatibleStreamProfile,
    CompatibleToolCallChunk,
};
use crate::providers::openai::completion::{
    CompletionModelOptions, GenericCompletionModel, OpenAICompatibleProvider, Usage,
};
use crate::streaming;

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

#[derive(Deserialize, Debug)]
struct StreamingDelta {
    #[serde(default, deserialize_with = "deserialize_delta_content")]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    // Not part of the official OpenAI API; some compatible providers (e.g.
    // Groq) send the same payload under `reasoning`. A separate field rather
    // than a serde alias so a delta carrying BOTH keys is not a
    // duplicate-field error that drops the whole chunk.
    #[serde(default)]
    reasoning: Option<String>,
    #[serde(default, deserialize_with = "json_utils::null_or_vec")]
    tool_calls: Vec<StreamingToolCall>,
    #[serde(default, deserialize_with = "json_utils::null_or_vec")]
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
    fn as_raw(&self) -> &str {
        match self {
            Self::ToolCalls => "tool_calls",
            Self::Stop => "stop",
            Self::ContentFilter => "content_filter",
            Self::Length => "length",
            Self::Other(reason) => reason,
        }
    }
}

#[derive(Deserialize, Debug)]
struct StreamingChoice {
    delta: StreamingDelta,
    finish_reason: Option<FinishReason>,
    #[serde(default)]
    native_finish_reason: Option<String>,
    #[serde(default)]
    error: Option<serde_json::Value>,
}

#[derive(Deserialize, Debug)]
struct StreamingCompletionChunk<U = Usage> {
    id: Option<String>,
    model: Option<String>,
    #[serde(default)]
    choices: Vec<StreamingChoice>,
    usage: Option<U>,
    #[serde(default)]
    error: Option<serde_json::Value>,
}

/// Final streaming response. `U` is the provider's streaming usage payload
/// ([`Usage`] for OpenAI itself; providers with richer usage accounting, e.g.
/// Mistral and DeepSeek, substitute their own via
/// [`OpenAICompatibleProvider::StreamingUsage`].
#[derive(Clone, Serialize, Deserialize)]
pub struct StreamingCompletionResponse<U = Usage> {
    pub usage: U,
    pub terminal_metadata: Option<CompletionTerminalMetadata>,
}

impl<U> GetCompletionMetadata for StreamingCompletionResponse<U>
where
    U: GetCompletionMetadata,
{
    fn token_usage(&self) -> crate::completion::Usage {
        self.usage.token_usage()
    }

    fn terminal_metadata(&self) -> Option<CompletionTerminalMetadata> {
        self.terminal_metadata.clone()
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
    pub(crate) async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<
        streaming::StreamingCompletionResponse<StreamingCompletionResponse<Ext::StreamingUsage>>,
        CompletionError,
    > {
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
        let mut request_as_json = serde_json::to_value(request)?;

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

        if enabled!(Level::TRACE) {
            tracing::trace!(
                target: "rig::completions",
                "OpenAI Chat Completions streaming completion request: {}",
                serde_json::to_string_pretty(&request_as_json)?
            );
        }

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
            openai_chat_completions_compatible::send_compatible_streaming_request(
                client,
                req,
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
        + GetCompletionMetadata
        + serde::de::DeserializeOwned
        + crate::wasm_compat::WasmCompatSend
        + Unpin
        + 'static,
{
    type Usage = U;
    type Detail = serde_json::Value;
    type FinalResponse = StreamingCompletionResponse<U>;

    fn normalize_chunk(
        &self,
        data: &str,
    ) -> Result<Option<CompatibleChunk<Self::Usage, Self::Detail>>, CompletionError> {
        let raw_data = data;
        let data = match serde_json::from_str::<StreamingCompletionChunk<U>>(data) {
            Ok(data) => data,
            Err(error) => {
                tracing::error!(?error, message = data, "Failed to parse SSE message");
                return Ok(None);
            }
        };

        if data.error.is_some()
            || data.choices.first().is_some_and(|choice| {
                choice.error.is_some()
                    || choice.finish_reason.as_ref().map(FinishReason::as_raw) == Some("error")
            })
        {
            return Err(crate::provider_response::completion_error_from_body(
                raw_data,
            ));
        }

        Ok(Some(
            openai_chat_completions_compatible::normalize_first_choice_chunk(
                data.id,
                data.model,
                data.usage,
                &data.choices,
                |choice| CompatibleChoiceData {
                    // `function_call` is the deprecated pre-tools finish reason
                    // some compatible providers still emit for tool calls.
                    finish_reason: match &choice.finish_reason {
                        Some(FinishReason::ToolCalls) => CompatibleFinishReason::ToolCalls,
                        Some(FinishReason::Other(other)) if other == "function_call" => {
                            CompatibleFinishReason::ToolCalls
                        }
                        _ => CompatibleFinishReason::Other,
                    },
                    terminal_metadata: self.provider.streaming_terminal_metadata(
                        choice.finish_reason.as_ref().map(FinishReason::as_raw),
                        choice.native_finish_reason.as_deref(),
                    ),
                    text: choice.delta.content.clone(),
                    reasoning: choice
                        .delta
                        .reasoning_content
                        .clone()
                        .or_else(|| choice.delta.reasoning.clone()),
                    tool_calls: openai_chat_completions_compatible::tool_call_chunks(
                        &choice.delta.tool_calls,
                    ),
                    details: choice.delta.reasoning_details.clone(),
                },
            ),
        ))
    }

    fn build_final_response(
        &self,
        usage: Self::Usage,
        terminal_metadata: Option<CompletionTerminalMetadata>,
    ) -> Self::FinalResponse {
        StreamingCompletionResponse {
            usage,
            terminal_metadata,
        }
    }

    fn decorate_tool_call(
        &self,
        detail: &Self::Detail,
        tool_calls: &mut std::collections::HashMap<usize, crate::streaming::RawStreamingToolCall>,
    ) {
        self.provider
            .decorate_streaming_tool_call(detail, tool_calls);
    }

    fn uses_distinct_tool_call_eviction(&self) -> bool {
        true
    }

    fn emits_complete_single_chunk_tool_calls(&self) -> bool {
        self.emits_complete_single_chunk_tool_calls
    }
}

pub async fn send_compatible_streaming_request<T>(
    http_client: T,
    req: Request<Vec<u8>>,
) -> Result<streaming::StreamingCompletionResponse<StreamingCompletionResponse>, CompletionError>
where
    T: HttpClientExt + Clone + 'static,
{
    openai_chat_completions_compatible::send_compatible_streaming_request(
        http_client,
        req,
        OpenAICompatibleProfile::<crate::providers::openai::OpenAICompletionsExt, Usage>::default(),
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::internal::openai_chat_completions_compatible::test_support::{
        assert_zero_arg_tool_call_is_emitted, sse_bytes_from_data_lines,
    };

    #[test]
    fn openrouter_streaming_preserves_native_reason_and_rejects_error_envelopes() {
        let profile = OpenAICompatibleProfile::<
            crate::providers::openrouter::OpenRouterExt,
            crate::providers::openrouter::Usage,
        >::default();

        let normalized = profile
            .normalize_chunk(
                r#"{"choices":[{"delta":{"content":"done"},"finish_reason":"stop","native_finish_reason":"STOP"}]}"#,
            )
            .expect("terminal chunk should normalize")
            .expect("terminal chunk should not be ignored");
        let metadata = normalized
            .choice
            .and_then(|choice| choice.terminal_metadata)
            .expect("terminal metadata");
        assert_eq!(
            metadata.reason(),
            crate::completion::CompletionFinishReason::Stop
        );
        assert_eq!(metadata.raw_reason(), Some("STOP"));

        for raw in [
            r#"{"error":{"code":429,"message":"rate limited"}}"#,
            r#"{"choices":[{"delta":{"content":"partial"},"finish_reason":"error","error":{"message":"upstream failed"}}]}"#,
        ] {
            let error = profile
                .normalize_chunk(raw)
                .expect_err("error chunk must terminate the stream");
            assert_eq!(error.provider_response_body(), Some(raw));
        }
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

        let req = http::Request::builder()
            .method("POST")
            .uri("http://localhost/v1/chat/completions")
            .body(Vec::new())
            .unwrap();

        let mut stream = send_compatible_streaming_request(client, req)
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
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.total_tokens, 15);
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

        let req = http::Request::builder()
            .method("POST")
            .uri("http://localhost/v1/chat/completions")
            .body(Vec::new())
            .unwrap();

        let mut stream = send_compatible_streaming_request(client, req)
            .await
            .unwrap();

        let mut reasoning_chunks = Vec::new();
        let mut text_chunks = Vec::new();
        let mut final_usage = None;

        while let Some(chunk) = stream.next().await {
            match chunk.unwrap() {
                streaming::StreamedAssistantContent::ReasoningDelta { reasoning, .. } => {
                    reasoning_chunks.push(reasoning)
                }
                streaming::StreamedAssistantContent::Text(text) => text_chunks.push(text.text),
                streaming::StreamedAssistantContent::Final(response) => {
                    final_usage = Some(response.usage)
                }
                _ => {}
            }
        }

        assert_eq!(
            reasoning_chunks,
            vec!["think ".to_string(), "more".to_string()]
        );
        assert_eq!(text_chunks, vec!["hel".to_string(), "lo".to_string()]);

        let usage = final_usage.expect("expected final usage");
        assert_eq!(usage.prompt_tokens, 4);
        assert_eq!(usage.total_tokens, 10);
        let token_usage = usage.token_usage();
        assert_eq!(token_usage.output_tokens, 6);
    }

    #[tokio::test]
    async fn test_streaming_cached_input_tokens_populated() {
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

        let req = http::Request::builder()
            .method("POST")
            .uri("http://localhost/v1/chat/completions")
            .body(Vec::new())
            .unwrap();

        let mut stream = send_compatible_streaming_request(client, req)
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

        // Verify provider-level usage has the cached_tokens
        assert_eq!(
            res.usage
                .prompt_tokens_details
                .as_ref()
                .unwrap()
                .cached_tokens,
            80
        );

        // Verify core Usage also has cached_input_tokens via GetCompletionMetadata
        let core_usage = res.token_usage();
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

        let req = http::Request::builder()
            .method("POST")
            .uri("http://localhost/v1/chat/completions")
            .body(Vec::new())
            .unwrap();

        let mut stream = send_compatible_streaming_request(client, req)
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

        let req = http::Request::builder()
            .method("POST")
            .uri("http://localhost/v1/chat/completions")
            .body(Vec::new())
            .unwrap();

        let mut stream = send_compatible_streaming_request(client, req)
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

        let req = http::Request::builder()
            .method("POST")
            .uri("http://localhost/v1/chat/completions")
            .body(Vec::new())
            .unwrap();

        let mut stream = send_compatible_streaming_request(client, req)
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

        let req = http::Request::builder()
            .method("POST")
            .uri("http://localhost/v1/chat/completions")
            .body(Vec::new())
            .unwrap();

        let stream = send_compatible_streaming_request(client, req)
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

        let req = http::Request::builder()
            .method("POST")
            .uri("http://localhost/v1/chat/completions")
            .body(Vec::new())
            .unwrap();

        let stream = send_compatible_streaming_request(client, req)
            .await
            .unwrap();

        assert_zero_arg_tool_call_is_emitted(stream, "call_123", "ping", true).await;
    }
}
