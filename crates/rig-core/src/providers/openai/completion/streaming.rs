use serde::Deserialize;

use crate::json_utils;
use crate::providers::internal::openai_chat_completions_compatible::{
    self, ChatCompletionsDialect, CompatibleChoice, CompatibleFinishReason,
    CompatibleToolCallChunk, NormalizedCompatibleChunk,
};

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

#[derive(Deserialize, Debug)]
struct StreamingChoice {
    delta: StreamingDelta,
    finish_reason: Option<FinishReason>,
}

#[derive(Deserialize, Debug)]
struct StreamingCompletionChunk {
    id: Option<String>,
    model: Option<String>,
    choices: Vec<StreamingChoice>,
    /// Left as a raw value so the dialect's own wire usage type parses it (see
    /// [`normalize_wire_usage`]); a malformed payload skips the whole chunk,
    /// exactly as a typed field would.
    usage: Option<serde_json::Value>,
}

/// Parse one OpenAI Chat Completions SSE `data` payload for `dialect`. Pure.
///
/// This is the sans-IO chunk normalizer for all 17 OpenAI-compatible dialects:
/// the wire schema is shared, and the only variation — which `usage` payload to
/// parse — is a match on plain dialect data.
pub(crate) fn normalize_chat_completions_chunk(
    data: &str,
    dialect: ChatCompletionsDialect,
) -> NormalizedCompatibleChunk {
    let chunk = match serde_json::from_str::<StreamingCompletionChunk>(data) {
        Ok(chunk) => chunk,
        Err(error) => {
            tracing::error!(?error, message = data, "Failed to parse SSE message");
            return Ok(None);
        }
    };

    let usage = match openai_chat_completions_compatible::normalize_wire_usage(
        chunk.usage,
        dialect.usage,
    ) {
        Ok(usage) => usage,
        Err(error) => {
            tracing::error!(?error, message = data, "Failed to parse SSE message");
            return Ok(None);
        }
    };

    Ok(Some(
        openai_chat_completions_compatible::first_choice_chunk(
            chunk.id,
            chunk.model,
            usage,
            &chunk.choices,
            |choice| CompatibleChoice {
                // `function_call` is the deprecated pre-tools finish reason
                // some compatible providers still emit for tool calls.
                finish_reason: match &choice.finish_reason {
                    Some(FinishReason::ToolCalls) => CompatibleFinishReason::ToolCalls,
                    Some(FinishReason::Other(other)) if other == "function_call" => {
                        CompatibleFinishReason::ToolCalls
                    }
                    _ => CompatibleFinishReason::Other,
                },
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::internal::openai_chat_completions_compatible::ChunkNormalizer;
    use crate::providers::internal::openai_chat_completions_compatible::test_support::{
        assert_zero_arg_tool_call_is_emitted, sse_bytes_from_data_lines,
    };
    use crate::streaming;
    use crate::test_utils::MockStreamingClient;

    /// Drive the OpenAI chat-completions dialect over scripted SSE bytes.
    ///
    /// The sans-IO half of `functions::open_stream`: same
    /// `drive_compatible_stream` + `STREAM_DIALECT`, with the transport edge
    /// (`boxed_event_source`) fed a canned event source instead of a live one.
    fn drive_openai_sse(
        sse_bytes: impl Into<bytes::Bytes>,
    ) -> streaming::StreamingCompletionResponse {
        let client = MockStreamingClient {
            sse_bytes: sse_bytes.into(),
        };
        let req = http::Request::builder()
            .method("POST")
            .uri("http://localhost/v1/chat/completions")
            .body(Vec::new())
            .expect("request should build");
        openai_chat_completions_compatible::drive_compatible_stream(
            crate::http_client::sse::boxed_event_source(client, req, false),
            ChunkNormalizer::ChatCompletions(crate::providers::openai::functions::STREAM_DIALECT),
        )
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
        use futures::StreamExt;

        // Some providers emit a final "usage-only" chunk where `choices` is empty.
        let mut stream = drive_openai_sse(sse_bytes_from_data_lines([
            "{\"choices\":[{\"delta\":{\"content\":\"Hello\",\"tool_calls\":[]}}],\"usage\":null}",
            "{\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15}}",
            "[DONE]",
        ]));

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
    async fn test_streaming_reasoning_content_and_text_chunks_are_incremental() {
        use futures::StreamExt;

        let mut stream = drive_openai_sse(sse_bytes_from_data_lines([
            "{\"id\":\"cmpl-1\",\"model\":\"Qwen/Qwen3-4B\",\"choices\":[{\"delta\":{\"reasoning_content\":\"think \",\"tool_calls\":[]},\"finish_reason\":null}],\"usage\":null}",
            "{\"id\":\"cmpl-1\",\"model\":\"Qwen/Qwen3-4B\",\"choices\":[{\"delta\":{\"reasoning_content\":\"more\",\"tool_calls\":[]},\"finish_reason\":null}],\"usage\":null}",
            "{\"id\":\"cmpl-1\",\"model\":\"Qwen/Qwen3-4B\",\"choices\":[{\"delta\":{\"content\":\"hel\",\"tool_calls\":[]},\"finish_reason\":null}],\"usage\":null}",
            "{\"id\":\"cmpl-1\",\"model\":\"Qwen/Qwen3-4B\",\"choices\":[{\"delta\":{\"content\":\"lo\",\"tool_calls\":[]},\"finish_reason\":\"stop\"}],\"usage\":null}",
            "{\"choices\":[],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":6,\"total_tokens\":10}}",
            "[DONE]",
        ]));

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
        assert_eq!(usage.input_tokens, 4);
        assert_eq!(usage.total_tokens, 10);
        assert_eq!(usage.output_tokens, 6);
    }

    #[tokio::test]
    async fn test_streaming_cached_input_tokens_populated() {
        use futures::StreamExt;

        // Usage chunk includes prompt_tokens_details with cached_tokens.
        let mut stream = drive_openai_sse(sse_bytes_from_data_lines([
            "{\"choices\":[{\"delta\":{\"content\":\"Hi\",\"tool_calls\":[]}}],\"usage\":null}",
            "{\"choices\":[],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":10,\"total_tokens\":110,\"prompt_tokens_details\":{\"cached_tokens\":80}}}",
            "[DONE]",
        ]));

        let mut final_response = None;
        while let Some(chunk) = stream.next().await {
            if let streaming::StreamedAssistantContent::Final(res) = chunk.unwrap() {
                final_response = Some(res);
                break;
            }
        }

        let res = final_response.expect("expected a final response");

        // The normalized final carries cached_input_tokens converted from the
        // wire's prompt_tokens_details.cached_tokens — the same arithmetic the
        // deleted GetTokenUsage impl performed.
        assert_eq!(res.usage.cached_input_tokens, 80);
        assert_eq!(res.usage.input_tokens, 100);
        assert_eq!(res.usage.total_tokens, 110);
    }

    /// Reproduces the bug where a proxy/gateway sends multiple parallel tool
    /// calls all sharing `index: 0` but with distinct `id` values.  Without
    /// the fix, rig merges both calls into one corrupted entry.
    #[tokio::test]
    async fn test_duplicate_index_different_id_tool_calls() {
        use futures::StreamExt;

        // Simulate a gateway that sends two tool calls both at index 0.
        // First tool call: id="call_aaa", name="command", args={"cmd":"ls"}
        // Second tool call: id="call_bbb", name="git", args={"action":"log"}
        let mut stream = drive_openai_sse(sse_bytes_from_data_lines([
            "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_aaa\",\"function\":{\"name\":\"command\",\"arguments\":\"\"}}]},\"finish_reason\":null}],\"usage\":null}",
            "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":null,\"function\":{\"name\":null,\"arguments\":\"{\\\"cmd\\\"\"}}]},\"finish_reason\":null}],\"usage\":null}",
            "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":null,\"function\":{\"name\":null,\"arguments\":\":\\\"ls\\\"}\"}}]},\"finish_reason\":null}],\"usage\":null}",
            "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_bbb\",\"function\":{\"name\":\"git\",\"arguments\":\"\"}}]},\"finish_reason\":null}],\"usage\":null}",
            "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":null,\"function\":{\"name\":null,\"arguments\":\"{\\\"action\\\"\"}}]},\"finish_reason\":null}],\"usage\":null}",
            "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":null,\"function\":{\"name\":null,\"arguments\":\":\\\"log\\\"}\"}}]},\"finish_reason\":null}],\"usage\":null}",
            "{\"choices\":[{\"delta\":{\"tool_calls\":[]},\"finish_reason\":\"tool_calls\"}],\"usage\":null}",
            "{\"choices\":[],\"usage\":{\"prompt_tokens\":20,\"completion_tokens\":10,\"total_tokens\":30}}",
            "[DONE]",
        ]));

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
        use futures::StreamExt;

        let mut stream = drive_openai_sse(sse_bytes_from_data_lines([
            "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_abc123\"}]},\"finish_reason\":null}],\"usage\":null}",
            "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":null,\"function\":{\"name\":\"lookup\",\"arguments\":\"\"}}]},\"finish_reason\":null}],\"usage\":null}",
            "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":null,\"function\":{\"name\":null,\"arguments\":\"{\\\"id\\\":1}\"}}]},\"finish_reason\":null}],\"usage\":null}",
            "{\"choices\":[{\"delta\":{\"tool_calls\":[]},\"finish_reason\":\"tool_calls\"}],\"usage\":null}",
            "[DONE]",
        ]));

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
        use futures::StreamExt;

        // Each chunk carries a different id but they all represent delta
        // fragments of the SAME tool call at index 0.
        let mut stream = drive_openai_sse(sse_bytes_from_data_lines([
            "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"chatcmpl-tool-aaa\",\"function\":{\"name\":\"web_search\",\"arguments\":\"null\"}}]},\"finish_reason\":null}],\"usage\":null}",
            "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"chatcmpl-tool-bbb\",\"function\":{\"name\":\"\",\"arguments\":\"{\\\"query\\\": \\\"META\"}}]},\"finish_reason\":null}],\"usage\":null}",
            "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"chatcmpl-tool-ccc\",\"function\":{\"name\":\"\",\"arguments\":\" Platforms news\\\"}\"}}]},\"finish_reason\":null}],\"usage\":null}",
            "{\"choices\":[{\"delta\":{\"tool_calls\":[]},\"finish_reason\":\"tool_calls\"}],\"usage\":null}",
            "{\"choices\":[],\"usage\":{\"prompt_tokens\":15,\"completion_tokens\":8,\"total_tokens\":23}}",
            "[DONE]",
        ]));

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
        let stream = drive_openai_sse(sse_bytes_from_data_lines([
            "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_123\",\"function\":{\"name\":\"ping\",\"arguments\":\"\"}}]},\"finish_reason\":null}],\"usage\":null}",
            "{\"choices\":[{\"delta\":{\"tool_calls\":[]},\"finish_reason\":\"tool_calls\"}],\"usage\":null}",
            "[DONE]",
        ]));

        assert_zero_arg_tool_call_is_emitted(stream, "call_123", "ping", true).await;
    }

    #[tokio::test]
    async fn test_zero_arg_tool_call_is_preserved_at_eof() {
        let stream = drive_openai_sse(sse_bytes_from_data_lines([
            "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_123\",\"function\":{\"name\":\"ping\",\"arguments\":\"\"}}]},\"finish_reason\":null}],\"usage\":null}",
        ]));

        assert_zero_arg_tool_call_is_emitted(stream, "call_123", "ping", true).await;
    }
}
