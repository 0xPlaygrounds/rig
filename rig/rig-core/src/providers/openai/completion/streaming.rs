use http::Request;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::{Level, enabled, info_span};

use crate::completion::{CompletionError, CompletionRequest, GetTokenUsage};
use crate::http_client::HttpClientExt;
use crate::json_utils::{self, merge};
use crate::providers::internal::chat_compatible::{
    self, CompatibleChoice, CompatibleChunk, CompatibleFinishReason, CompatibleStreamProfile,
    CompatibleToolCallChunk,
};
use crate::providers::openai::completion::{GenericCompletionModel, OpenAIRequestParams, Usage};
use crate::streaming;

// ================================================================
// OpenAI Completion Streaming API
// ================================================================
#[derive(Deserialize, Debug)]
pub(crate) struct StreamingFunction {
    pub(crate) name: Option<String>,
    pub(crate) arguments: Option<String>,
}

#[derive(Deserialize, Debug)]
pub(crate) struct StreamingToolCall {
    pub(crate) index: usize,
    pub(crate) id: Option<String>,
    pub(crate) function: StreamingFunction,
}

#[derive(Deserialize, Debug)]
struct StreamingDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>, // This is not part of the official OpenAI API
    #[serde(default, deserialize_with = "json_utils::null_or_vec")]
    tool_calls: Vec<StreamingToolCall>,
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
    usage: Option<Usage>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct StreamingCompletionResponse {
    pub usage: Usage,
}

impl GetTokenUsage for StreamingCompletionResponse {
    fn token_usage(&self) -> Option<crate::completion::Usage> {
        let mut usage = crate::completion::Usage::new();
        usage.input_tokens = self.usage.prompt_tokens as u64;
        usage.output_tokens = self.usage.total_tokens as u64 - self.usage.prompt_tokens as u64;
        usage.total_tokens = self.usage.total_tokens as u64;
        usage.cached_input_tokens = self
            .usage
            .prompt_tokens_details
            .as_ref()
            .map_or(0, |d| d.cached_tokens as u64);
        Some(usage)
    }
}

impl<Ext, H> GenericCompletionModel<Ext, H>
where
    crate::client::Client<Ext, H>: HttpClientExt + Clone + 'static,
    Ext: crate::client::Provider + Clone + 'static,
{
    pub(crate) async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<streaming::StreamingCompletionResponse<StreamingCompletionResponse>, CompletionError>
    {
        let request = super::CompletionRequest::try_from(OpenAIRequestParams {
            model: self.model.clone(),
            request: completion_request,
            strict_tools: self.strict_tools,
            tool_result_array_content: self.tool_result_array_content,
        })?;
        let request_messages = serde_json::to_string(&request.messages)
            .expect("Converting to JSON from a Rust struct shouldn't fail");
        let mut request_as_json = serde_json::to_value(request).expect("this should never fail");

        request_as_json = merge(
            request_as_json,
            json!({"stream": true, "stream_options": {"include_usage": true}}),
        );

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
            .post("/chat/completions")?
            .body(req_body)
            .map_err(|e| CompletionError::HttpError(e.into()))?;

        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat",
                gen_ai.operation.name = "chat",
                gen_ai.provider.name = "openai",
                gen_ai.request.model = self.model,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.usage.cached_tokens = tracing::field::Empty,
                gen_ai.input.messages = request_messages,
                gen_ai.output.messages = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        let client = self.client.clone();

        tracing::Instrument::instrument(send_compatible_streaming_request(client, req), span).await
    }
}

#[derive(Clone, Copy)]
struct OpenAICompatibleProfile;

impl CompatibleStreamProfile for OpenAICompatibleProfile {
    type Usage = Usage;
    type Detail = ();
    type FinalResponse = StreamingCompletionResponse;

    fn normalize_chunk(
        &self,
        data: &str,
    ) -> Result<Option<CompatibleChunk<Self::Usage, Self::Detail>>, CompletionError> {
        let data = match serde_json::from_str::<StreamingCompletionChunk>(data) {
            Ok(data) => data,
            Err(error) => {
                tracing::error!(?error, message = data, "Failed to parse SSE message");
                return Ok(None);
            }
        };

        let choice = data.choices.first().map(|choice| CompatibleChoice {
            finish_reason: if choice.finish_reason == Some(FinishReason::ToolCalls) {
                CompatibleFinishReason::ToolCalls
            } else {
                CompatibleFinishReason::Other
            },
            text: choice.delta.content.clone(),
            reasoning: choice.delta.reasoning_content.clone(),
            tool_calls: choice
                .delta
                .tool_calls
                .iter()
                .map(|tool_call| CompatibleToolCallChunk {
                    index: tool_call.index,
                    id: tool_call.id.clone(),
                    name: tool_call.function.name.clone(),
                    arguments: tool_call.function.arguments.clone(),
                })
                .collect(),
            details: Vec::new(),
        });

        Ok(Some(CompatibleChunk {
            response_id: data.id,
            response_model: data.model,
            choice,
            usage: data.usage,
        }))
    }

    fn build_final_response(&self, usage: Self::Usage) -> Self::FinalResponse {
        StreamingCompletionResponse { usage }
    }

    fn uses_distinct_tool_call_eviction(&self) -> bool {
        true
    }
}

pub async fn send_compatible_streaming_request<T>(
    http_client: T,
    req: Request<Vec<u8>>,
) -> Result<streaming::StreamingCompletionResponse<StreamingCompletionResponse>, CompletionError>
where
    T: HttpClientExt + Clone + 'static,
{
    chat_compatible::send_compatible_streaming_request(http_client, req, OpenAICompatibleProfile)
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::internal::chat_compatible::test_support::assert_zero_arg_tool_call_is_emitted;

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
        use crate::http_client::mock::MockStreamingClient;
        use bytes::Bytes;
        use futures::StreamExt;

        // Some providers emit a final "usage-only" chunk where `choices` is empty.
        let sse = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\",\"tool_calls\":[]}}],\"usage\":null}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15}}\n\n",
            "data: [DONE]\n\n",
        );

        let client = MockStreamingClient {
            sse_bytes: Bytes::from(sse),
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
    async fn test_streaming_cached_input_tokens_populated() {
        use crate::http_client::mock::MockStreamingClient;
        use bytes::Bytes;
        use futures::StreamExt;

        // Usage chunk includes prompt_tokens_details with cached_tokens.
        let sse = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\",\"tool_calls\":[]}}],\"usage\":null}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":10,\"total_tokens\":110,\"prompt_tokens_details\":{\"cached_tokens\":80}}}\n\n",
            "data: [DONE]\n\n",
        );

        let client = MockStreamingClient {
            sse_bytes: Bytes::from(sse),
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

        // Verify core Usage also has cached_input_tokens via GetTokenUsage
        let core_usage = res.token_usage().expect("token_usage should return Some");
        assert_eq!(core_usage.cached_input_tokens, 80);
        assert_eq!(core_usage.input_tokens, 100);
        assert_eq!(core_usage.total_tokens, 110);
    }

    /// Reproduces the bug where a proxy/gateway sends multiple parallel tool
    /// calls all sharing `index: 0` but with distinct `id` values.  Without
    /// the fix, rig merges both calls into one corrupted entry.
    #[tokio::test]
    async fn test_duplicate_index_different_id_tool_calls() {
        use crate::http_client::mock::MockStreamingClient;
        use bytes::Bytes;
        use futures::StreamExt;

        // Simulate a gateway that sends two tool calls both at index 0.
        // First tool call: id="call_aaa", name="command", args={"cmd":"ls"}
        // Second tool call: id="call_bbb", name="git", args={"action":"log"}
        let sse = concat!(
            // First tool call starts
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_aaa\",\"function\":{\"name\":\"command\",\"arguments\":\"\"}}]},\"finish_reason\":null}],\"usage\":null}\n\n",
            // First tool call argument chunks
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":null,\"function\":{\"name\":null,\"arguments\":\"{\\\"cmd\\\"\"}}]},\"finish_reason\":null}],\"usage\":null}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":null,\"function\":{\"name\":null,\"arguments\":\":\\\"ls\\\"}\"}}]},\"finish_reason\":null}],\"usage\":null}\n\n",
            // Second tool call starts AT THE SAME index 0 but with a NEW id
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_bbb\",\"function\":{\"name\":\"git\",\"arguments\":\"\"}}]},\"finish_reason\":null}],\"usage\":null}\n\n",
            // Second tool call argument chunks
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":null,\"function\":{\"name\":null,\"arguments\":\"{\\\"action\\\"\"}}]},\"finish_reason\":null}],\"usage\":null}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":null,\"function\":{\"name\":null,\"arguments\":\":\\\"log\\\"}\"}}]},\"finish_reason\":null}],\"usage\":null}\n\n",
            // Finish with tool_calls reason
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[]},\"finish_reason\":\"tool_calls\"}],\"usage\":null}\n\n",
            // Usage chunk
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":20,\"completion_tokens\":10,\"total_tokens\":30}}\n\n",
            "data: [DONE]\n\n",
        );

        let client = MockStreamingClient {
            sse_bytes: Bytes::from(sse),
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

    /// Reproduces the bug where a provider (e.g. GLM-4 via OpenAI-compatible
    /// endpoint) sends a unique `id` on every SSE delta chunk for the same
    /// logical tool call.  Without the fix, each chunk triggers an eviction,
    /// yielding incomplete fragments as "completed" tool calls.
    #[tokio::test]
    async fn test_unique_id_per_chunk_single_tool_call() {
        use crate::http_client::mock::MockStreamingClient;
        use bytes::Bytes;
        use futures::StreamExt;

        // Each chunk carries a different id but they all represent delta
        // fragments of the SAME tool call at index 0.
        let sse = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"chatcmpl-tool-aaa\",\"function\":{\"name\":\"web_search\",\"arguments\":\"null\"}}]},\"finish_reason\":null}],\"usage\":null}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"chatcmpl-tool-bbb\",\"function\":{\"name\":\"\",\"arguments\":\"{\\\"query\\\": \\\"META\"}}]},\"finish_reason\":null}],\"usage\":null}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"chatcmpl-tool-ccc\",\"function\":{\"name\":\"\",\"arguments\":\" Platforms news\\\"}\"}}]},\"finish_reason\":null}],\"usage\":null}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[]},\"finish_reason\":\"tool_calls\"}],\"usage\":null}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":15,\"completion_tokens\":8,\"total_tokens\":23}}\n\n",
            "data: [DONE]\n\n",
        );

        let client = MockStreamingClient {
            sse_bytes: Bytes::from(sse),
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
        use crate::http_client::mock::MockStreamingClient;
        use bytes::Bytes;

        let sse = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_123\",\"function\":{\"name\":\"ping\",\"arguments\":\"\"}}]},\"finish_reason\":null}],\"usage\":null}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[]},\"finish_reason\":\"tool_calls\"}],\"usage\":null}\n\n",
            "data: [DONE]\n\n",
        );

        let client = MockStreamingClient {
            sse_bytes: Bytes::from(sse),
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
        use crate::http_client::mock::MockStreamingClient;
        use bytes::Bytes;

        let sse = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_123\",\"function\":{\"name\":\"ping\",\"arguments\":\"\"}}]},\"finish_reason\":null}],\"usage\":null}\n\n";

        let client = MockStreamingClient {
            sse_bytes: Bytes::from(sse),
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
