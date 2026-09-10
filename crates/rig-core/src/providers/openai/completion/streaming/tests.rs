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
async fn collect_openai_stream(chunks: &[&str]) -> (String, Option<crate::streaming::StreamFinal>) {
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
        if let streaming::RawStreamingChoice::FinalResponse(response) = chunk.expect("stream item")
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
                reasoning_chunks.push(reasoning);
            }
            streaming::StreamedAssistantContent::Text(text) => text_chunks.push(text.text),
            streaming::StreamedAssistantContent::Final(response) => {
                final_response = Some(response);
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
        let mut stream = send_compatible_streaming_request(client, streaming_request(), "openai")
            .await
            .expect("stream should open");

        let mut terminal = None;
        while let Some(item) = stream.next().await {
            if let streaming::StreamedAssistantContent::Final(record) = item.expect("stream item") {
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
