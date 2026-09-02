use super::{
    ChatApiErrorResponse, Client, CompletionRoute, CopilotIntent, TEXT_EMBEDDING_3_SMALL,
    base_url_from_token, default_headers, env_api_key, env_base_url, env_github_access_token,
    route_for_model,
};
use crate::client::CompletionClient;
use crate::completion::CompletionModel;
use crate::http_client;
use crate::message::AssistantContent;
use crate::providers::internal::openai_chat_completions_compatible::test_support::{
    sse_bytes_from_data_lines, sse_bytes_from_json_events,
};
use crate::providers::openai;
use crate::streaming::{BlockClose, Delta, StreamEvent};
use crate::test_utils::MockStreamingClient;
use crate::test_utils::{RecordingHttpClient, SequencedStreamingHttpClient};
use futures::StreamExt;
use std::collections::HashMap;

fn env_map(entries: &[(&str, &str)]) -> HashMap<String, String> {
    entries
        .iter()
        .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
        .collect()
}

fn minimal_chat_response() -> &'static str {
    r#"{
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "hello"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 4,
                "total_tokens": 7
            }
        }"#
}

fn minimal_responses_response() -> &'static str {
    r#"{
            "id": "resp_123",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "error": null,
            "incomplete_details": null,
            "instructions": null,
            "max_output_tokens": null,
            "model": "gpt-5.3-codex",
            "usage": {
                "input_tokens": 4,
                "input_tokens_details": {
                    "cached_tokens": 0
                },
                "output_tokens": 3,
                "output_tokens_details": {
                    "reasoning_tokens": 0
                },
                "total_tokens": 7
            },
            "output": [{
                "type": "message",
                "id": "msg_123",
                "role": "assistant",
                "status": "completed",
                "content": [{
                    "type": "output_text",
                    "text": "hello"
                }]
            }],
            "tools": []
        }"#
}

fn minimal_embeddings_response() -> &'static str {
    r#"{
            "data": [
                {
                    "embedding": [0.1, 0.2, 0.3]
                },
                {
                    "embedding": [0.4, 0.5, 0.6]
                }
            ]
        }"#
}

#[test]
fn deserialize_standard_openai_response() {
    let json = r#"{
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello!"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }"#;

    let response: openai::completion::CompletionResponse =
        serde_json::from_str(json).expect("standard OpenAI response should deserialize");
    assert_eq!(response.id, "chatcmpl-abc123");
    assert_eq!(response.object, "chat.completion");
    assert_eq!(response.created, 1700000000);
    assert_eq!(response.model, "gpt-4o");
    assert_eq!(response.choices.len(), 1);
    assert_eq!(response.choices[0].finish_reason, "stop");
}

#[test]
fn deserialize_copilot_response_without_object_and_created() {
    let response: openai::completion::CompletionResponse =
        serde_json::from_str(minimal_chat_response()).expect("Copilot response should deserialize");

    assert_eq!(response.id, "chatcmpl-123");
    assert_eq!(response.object, "");
    assert_eq!(response.created, 0);
    assert_eq!(response.model, "gpt-4o");
    assert_eq!(response.choices.len(), 1);
}

#[test]
fn deserialize_copilot_response_without_finish_reason() {
    let json = r#"{
            "id": "chatcmpl-claude-001",
            "model": "claude-3.5-sonnet",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Here is my analysis."
                }
            }],
            "usage": {
                "prompt_tokens": 50,
                "total_tokens": 80
            }
        }"#;

    let response: openai::completion::CompletionResponse =
        serde_json::from_str(json).expect("Claude-via-Copilot response should deserialize");

    assert_eq!(response.model, "claude-3.5-sonnet");
    assert_eq!(response.choices[0].finish_reason, "");
    assert_eq!(response.choices[0].index, 0);
}

#[test]
fn error_response_with_message_field() {
    let json = r#"{"message": "rate limit exceeded"}"#;
    let err: ChatApiErrorResponse = serde_json::from_str(json).expect("message-shaped error");

    assert_eq!(err.error_message(), "rate limit exceeded");
}

#[test]
fn error_response_with_error_field() {
    let json = r#"{"error": "model not found"}"#;
    let err: ChatApiErrorResponse = serde_json::from_str(json).expect("error-shaped error");

    assert_eq!(err.error_message(), "model not found");
}

#[test]
fn routes_codex_models_to_responses() {
    assert_eq!(route_for_model("gpt-5.3-codex"), CompletionRoute::Responses);
    assert_eq!(
        route_for_model("gpt-5.1-CODEX-mini"),
        CompletionRoute::Responses
    );
    assert_eq!(route_for_model("gpt-5.2"), CompletionRoute::ChatCompletions);
    assert_eq!(
        route_for_model("claude-sonnet-4.5"),
        CompletionRoute::ChatCompletions
    );
}

#[test]
fn copilot_intent_headers_use_panel_by_default_and_edits_when_requested() {
    let panel_headers = default_headers("token", "user", false, CopilotIntent::default());
    assert_eq!(
        panel_headers
            .iter()
            .find(|(name, _)| *name == "openai-intent")
            .map(|(_, value)| value.as_str()),
        Some("conversation-panel")
    );

    let edits_headers = default_headers("token", "user", false, CopilotIntent::Edits);
    assert_eq!(
        edits_headers
            .iter()
            .find(|(name, _)| *name == "openai-intent")
            .map(|(_, value)| value.as_str()),
        Some("conversation-edits")
    );
}

#[test]
fn copilot_completion_model_intent_builders_update_intent() {
    let client = Client::builder()
        .api_key("copilot-token")
        .http_client(crate::test_utils::RecordingHttpClient::new(""))
        .build()
        .expect("build client");

    let default_model = client.completion_model("gpt-4o");
    assert_eq!(default_model.intent.as_header(), "conversation-panel");

    let edits_model = client
        .completion_model("gpt-4o")
        .with_intent(CopilotIntent::Edits);
    assert_eq!(edits_model.intent.as_header(), "conversation-edits");

    let panel_model = client
        .completion_model("gpt-4o")
        .with_edits_intent()
        .with_panel_intent();
    assert_eq!(panel_model.intent.as_header(), "conversation-panel");
}

#[test]
fn base_url_from_token_derives_api_endpoint() {
    assert_eq!(
        base_url_from_token("tid=1;proxy-ep=proxy.individual.githubcopilot.com;exp=2").as_deref(),
        Some("https://api.individual.githubcopilot.com")
    );
    assert_eq!(
        base_url_from_token("tid=1;proxy-ep=https://proxy.individual.githubcopilot.com;exp=2")
            .as_deref(),
        Some("https://api.individual.githubcopilot.com")
    );
    assert_eq!(base_url_from_token("tid=1;exp=2"), None);
}

#[test]
fn base_url_from_token_rejects_unsafe_or_non_copilot_endpoints() {
    assert_eq!(
        base_url_from_token("tid=1;proxy-ep=http://proxy.individual.githubcopilot.com;exp=2"),
        None
    );
    assert_eq!(
        base_url_from_token("tid=1;proxy-ep=https://evil.example.com;exp=2"),
        None
    );
    assert_eq!(base_url_from_token("tid=1;proxy-ep=://bad;exp=2"), None);
    assert_eq!(base_url_from_token("tid=1;proxy-ep=;exp=2"), None);
    assert_eq!(
        base_url_from_token("tid=1;proxy-ep=https://proxy.individual.githubcopilot.com/base;exp=2"),
        None
    );
}

#[tokio::test]
async fn api_key_with_proxy_endpoint_overrides_base_url() {
    let http_client = RecordingHttpClient::new(minimal_chat_response());
    let client = Client::builder()
        .api_key("tid=1;proxy-ep=proxy.individual.githubcopilot.com;exp=2")
        .http_client(http_client.clone())
        .build()
        .expect("build client");
    let model = client.completion_model("gpt-4o");
    let request = model.completion_request("hello").build();

    let _response = model.completion(request).await.expect("chat completion");

    let requests = http_client.requests();
    assert_eq!(requests.len(), 1);
    assert!(
        requests[0]
            .uri
            .starts_with("https://api.individual.githubcopilot.com"),
        "expected proxy-derived base URL, got {}",
        requests[0].uri
    );
}

#[tokio::test]
async fn explicit_base_url_wins_over_token_proxy_endpoint() {
    let http_client = RecordingHttpClient::new(minimal_chat_response());
    let client = Client::builder()
        .api_key("tid=1;proxy-ep=proxy.individual.githubcopilot.com;exp=2")
        .base_url("https://custom.example.com")
        .http_client(http_client.clone())
        .build()
        .expect("build client");
    let model = client.completion_model("gpt-4o");
    let request = model.completion_request("hello").build();

    let _response = model.completion(request).await.expect("chat completion");

    let requests = http_client.requests();
    assert_eq!(requests.len(), 1);
    assert!(
        requests[0].uri.starts_with("https://custom.example.com"),
        "expected explicit base URL, got {}",
        requests[0].uri
    );
}

#[tokio::test]
async fn completion_model_edits_intent_sets_request_header() {
    let http_client = RecordingHttpClient::new(minimal_chat_response());
    let client = Client::builder()
        .api_key("copilot-token")
        .http_client(http_client.clone())
        .build()
        .expect("build client");
    let model = client.completion_model("gpt-4o").with_edits_intent();
    let request = model.completion_request("hello").build();

    let _response = model.completion(request).await.expect("chat completion");

    let requests = http_client.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(
        requests[0]
            .headers
            .get("openai-intent")
            .and_then(|value| value.to_str().ok()),
        Some("conversation-edits")
    );
}

#[tokio::test]
async fn completion_model_routes_chat_requests_to_chat_completions() {
    let http_client = RecordingHttpClient::new(minimal_chat_response());
    let client = Client::builder()
        .api_key("copilot-token")
        .http_client(http_client.clone())
        .build()
        .expect("build client");
    let model = client.completion_model("gpt-4o");
    let request = model.completion_request("hello").build();

    let _response = model.completion(request).await.expect("chat completion");

    let requests = http_client.requests();
    assert_eq!(requests.len(), 1);
    assert!(requests[0].uri.ends_with("/chat/completions"));
    assert!(String::from_utf8_lossy(&requests[0].body).contains("\"model\":\"gpt-4o\""));
}

#[tokio::test]
async fn completion_model_routes_codex_requests_to_responses() {
    let http_client = RecordingHttpClient::new(minimal_responses_response());
    let client = Client::builder()
        .api_key("copilot-token")
        .http_client(http_client.clone())
        .build()
        .expect("build client");
    let model = client.completion_model("gpt-5.3-codex");
    let request = model.completion_request("hello").build();

    let _response = model
        .completion(request)
        .await
        .expect("responses completion");

    let requests = http_client.requests();
    assert_eq!(requests.len(), 1);
    assert!(requests[0].uri.ends_with("/responses"));
    assert!(String::from_utf8_lossy(&requests[0].body).contains("\"model\":\"gpt-5.3-codex\""));
}

#[tokio::test]
async fn embeddings_accept_minimal_copilot_response_shape() {
    use crate::client::EmbeddingsClient;
    use crate::embeddings::EmbeddingModel as _;

    let http_client = RecordingHttpClient::new(minimal_embeddings_response());
    let client = Client::builder()
        .api_key("copilot-token")
        .http_client(http_client.clone())
        .build()
        .expect("build client");
    let model = client.embedding_model(TEXT_EMBEDDING_3_SMALL);

    let embeddings = model
        .embed_texts(["one".to_string(), "two".to_string()])
        .await
        .expect("embeddings should deserialize");

    assert_eq!(embeddings.len(), 2);
    assert_eq!(embeddings[0].vec, vec![0.1, 0.2, 0.3]);
    assert_eq!(embeddings[1].vec, vec![0.4, 0.5, 0.6]);

    let requests = http_client.requests();
    assert_eq!(requests.len(), 1);
    assert!(requests[0].uri.ends_with("/embeddings"));
    assert!(
        String::from_utf8_lossy(&requests[0].body).contains("\"model\":\"text-embedding-3-small\"")
    );
}

#[tokio::test]
async fn responses_stream_terminates_after_terminal_error() {
    let tool_call_done = serde_json::json!({
        "type": "response.output_item.done",
        "output_index": 0,
        "sequence_number": 1,
        "item": {
            "type": "function_call",
            "id": "fc_123",
            "arguments": "{}",
            "call_id": "call_123",
            "name": "example_tool",
            "status": "completed"
        }
    });
    let failed = serde_json::json!({
        "type": "response.failed",
        "sequence_number": 2,
        "response": {
            "id": "resp_123",
            "object": "response",
            "created_at": 1700000000,
            "status": "failed",
            "error": {
                "code": "server_error",
                "message": "Copilot response stream failed"
            },
            "incomplete_details": null,
            "instructions": null,
            "max_output_tokens": null,
            "model": "gpt-5.3-codex",
            "usage": null,
            "output": [],
            "tools": []
        }
    });
    let http_client = MockStreamingClient {
        sse_bytes: sse_bytes_from_json_events(&[tool_call_done, failed]),
    };
    let client = Client::builder()
        .api_key("copilot-token")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.completion_model("gpt-5.3-codex");
    let request = model.completion_request("hello").build();
    let mut stream = model.stream(request).await.expect("stream should start");

    // The fully-delivered tool call is content, so it is flushed *before*
    // the terminal error: consumers that stop at the first `Err` still
    // see the completed work.
    let mut flushed_tool_call = false;
    let err = loop {
        match stream
            .next()
            .await
            .expect("fully-delivered tool call should be flushed before the error")
        {
            Ok(StreamEvent::BlockEnd {
                block: Some(AssistantContent::ToolCall(tool_call)),
                ..
            }) => {
                assert_eq!(tool_call.function.name, "example_tool");
                flushed_tool_call = true;
            }
            Ok(StreamEvent::BlockStart { .. } | StreamEvent::BlockDelta { .. }) => {}
            Ok(item) => panic!("expected the flushed tool call, got {item:?}"),
            Err(err) => break err,
        }
    };
    assert!(
        flushed_tool_call,
        "the flushed tool call must precede the terminal error"
    );
    // The terminal `response.failed` event carries the provider's error
    // payload, so the full raw event JSON is preserved for inspection
    // (status: None — the error arrived over an already-established stream),
    // matching the OpenAI Responses SSE path.
    assert!(matches!(
        err,
        crate::completion::CompletionError::ProviderResponse(_)
    ));
    assert_eq!(err.provider_response_status(), None);
    let json = err
        .provider_response_json()
        .expect("preserved body should parse as JSON")
        .expect("preserved body should not be empty");
    let response_error = json
        .get("response")
        .and_then(|response| response.get("error"))
        .expect("preserved body should retain the provider error object");
    assert_eq!(
        response_error.get("code").and_then(|value| value.as_str()),
        Some("server_error")
    );
    assert_eq!(
        response_error
            .get("message")
            .and_then(|value| value.as_str()),
        Some("Copilot response stream failed")
    );
    assert!(
        stream.next().await.is_none(),
        "responses stream should end without a terminal record after a terminal error"
    );
}

#[tokio::test]
async fn responses_stream_object_less_failed_still_attaches_the_raw_event() {
    // #2258 F4 decision: the old Copilot code kept a deliberate two-tier
    // shape — `response.failed` WITHOUT an error object surfaced as a
    // `ProviderError` with `provider_response_body() == None`. The shared
    // Responses interpreter unifies this: the raw event body is ALWAYS
    // attached, error object or not, so callers can inspect what the
    // provider actually sent. Documented in MIGRATING.
    let failed = serde_json::json!({
        "type": "response.failed",
        "sequence_number": 1,
        "response": {
            "id": "resp_123",
            "object": "response",
            "created_at": 1700000000,
            "status": "failed",
            "error": null,
            "incomplete_details": null,
            "instructions": null,
            "max_output_tokens": null,
            "model": "gpt-5.3-codex",
            "usage": null,
            "output": [],
            "tools": []
        }
    });
    let http_client = MockStreamingClient {
        sse_bytes: sse_bytes_from_json_events(&[failed]),
    };
    let client = Client::builder()
        .api_key("copilot-token")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.completion_model("gpt-5.3-codex");
    let request = model.completion_request("hello").build();
    let mut stream = model.stream(request).await.expect("stream should start");

    let err = match stream.next().await.expect("stream should yield an item") {
        Ok(item) => panic!("stream should surface a provider error, got {item:?}"),
        Err(err) => err,
    };
    assert!(matches!(
        err,
        crate::completion::CompletionError::ProviderResponse(_)
    ));
    assert_eq!(err.provider_response_status(), None);
    assert!(
        err.provider_response_body()
            .is_some_and(|body| body.contains("response.failed")),
        "an object-less response.failed must still carry the raw event body"
    );
    assert!(
        stream.next().await.is_none(),
        "responses stream should end after the terminal error"
    );
}

#[tokio::test]
async fn responses_stream_incomplete_is_a_terminal_with_partial_content() {
    // The content exists only in the delta; the terminal
    // `response.incomplete` body has an empty `output`.
    let text_delta = serde_json::json!({
        "type": "response.output_text.delta",
        "content_index": 0,
        "delta": "partial",
        "item_id": "msg_1",
        "logprobs": [],
        "output_index": 0,
        "sequence_number": 1
    });
    let incomplete = serde_json::json!({
        "type": "response.incomplete",
        "sequence_number": 2,
        "response": {
            "id": "resp_123",
            "object": "response",
            "created_at": 1700000000,
            "status": "incomplete",
            "error": null,
            "incomplete_details": { "reason": "max_output_tokens" },
            "instructions": null,
            "max_output_tokens": null,
            "model": "gpt-5.3-codex",
            "usage": { "input_tokens": 1, "output_tokens": 2, "total_tokens": 3 },
            "output": [],
            "tools": []
        }
    });
    let http_client = MockStreamingClient {
        sse_bytes: sse_bytes_from_json_events(&[text_delta, incomplete]),
    };
    let client = Client::builder()
        .api_key("copilot-token")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.completion_model("gpt-5.3-codex");
    let request = model.completion_request("hello").build();
    let mut stream = model.stream(request).await.expect("stream should start");

    let mut text = String::new();
    let mut terminal = None;
    while let Some(item) = stream.next().await {
        match item.expect("incomplete turn should not surface an error") {
            StreamEvent::BlockDelta {
                delta: Delta::Text { text: chunk },
                ..
            } => text.push_str(&chunk),
            StreamEvent::Final(final_response) => terminal = Some(final_response),
            StreamEvent::BlockStart { .. }
            | StreamEvent::BlockEnd {
                end: BlockClose::Text,
                ..
            } => {}
            other => panic!("unexpected stream item: {other:?}"),
        }
    }

    assert_eq!(text, "partial");
    let terminal = terminal.expect("incomplete turn should emit a terminal record");
    assert_eq!(
        terminal.finish_reason,
        Some(crate::completion::FinishReason::Length)
    );
    assert_eq!(terminal.usage.input_tokens, 1);
    assert_eq!(terminal.usage.output_tokens, 2);
    assert_eq!(terminal.usage.total_tokens, 3);
}

#[tokio::test]
async fn chat_stream_surfaces_malformed_frame_and_still_completes() {
    let http_client = MockStreamingClient {
        sse_bytes: sse_bytes_from_data_lines([
            "{\"choices\":[{\"delta\":{\"content\":\"hello\"},\"finish_reason\":null}],\"usage\":null}",
            "{not valid json",
            "{\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":null}",
            "[DONE]",
        ]),
    };
    let client = Client::builder()
        .api_key("copilot-token")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.completion_model("gpt-4o");
    let request = model.completion_request("hello").build();
    let mut stream = model.stream(request).await.expect("stream should start");

    let mut text = String::new();
    let mut saw_error = false;
    let mut terminal = None;
    while let Some(item) = stream.next().await {
        match item {
            Ok(StreamEvent::BlockDelta {
                delta: Delta::Text { text: chunk },
                ..
            }) => text.push_str(&chunk),
            Ok(
                StreamEvent::BlockStart { .. }
                | StreamEvent::BlockEnd {
                    end: BlockClose::Text,
                    ..
                },
            ) => {}
            Ok(StreamEvent::Final(final_response)) => {
                terminal = Some(final_response);
            }
            Ok(other) => panic!("unexpected stream item: {other:?}"),
            Err(err) => {
                assert!(
                    matches!(err, crate::completion::CompletionError::JsonError(_)),
                    "expected a JSON parse error item, got {err:?}"
                );
                saw_error = true;
            }
        }
    }

    // The malformed frame is surfaced as an error item, and the content
    // and genuine terminal on either side of it both still arrive.
    assert_eq!(text, "hello");
    assert!(saw_error, "malformed frame should surface an error item");
    let terminal = terminal.expect("stream should still emit its terminal record");
    assert_eq!(
        terminal.finish_reason,
        Some(crate::completion::FinishReason::Stop)
    );
}

#[tokio::test]
async fn chat_stream_surfaces_recognizable_chunk_with_malformed_field() {
    // The frame is recognizably a chat completion chunk (it has
    // `choices`), but the payload fails the full parse — a data-level
    // defect surfaced as an error item, not a skippable unknown event.
    let http_client = MockStreamingClient {
        sse_bytes: sse_bytes_from_data_lines([
            "{\"object\":\"chat.completion.chunk\",\"choices\":42}",
            "{\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":null}",
            "[DONE]",
        ]),
    };
    let client = Client::builder()
        .api_key("copilot-token")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.completion_model("gpt-4o");
    let request = model.completion_request("hello").build();
    let mut stream = model.stream(request).await.expect("stream should start");

    let mut saw_error = false;
    let mut terminal = None;
    while let Some(item) = stream.next().await {
        match item {
            Ok(StreamEvent::Final(final_response)) => {
                terminal = Some(final_response);
            }
            Ok(other) => panic!("unexpected stream item: {other:?}"),
            Err(err) => {
                assert!(
                    matches!(err, crate::completion::CompletionError::JsonError(_)),
                    "expected a JSON parse error item, got {err:?}"
                );
                saw_error = true;
            }
        }
    }

    assert!(
        saw_error,
        "a recognizable chunk with a malformed field should surface an error item"
    );
    let terminal = terminal.expect("stream should still emit its terminal record");
    assert_eq!(
        terminal.finish_reason,
        Some(crate::completion::FinishReason::Stop)
    );
}

#[tokio::test]
async fn chat_stream_skips_unrecognized_event_and_still_completes() {
    // Valid JSON that is not recognizably a chat completion chunk (no
    // `choices`, no `"object": "chat.completion.chunk"`) is an event this
    // client doesn't know yet — skipped semantically for forward
    // compatibility, surfaced verbatim on the raw passthrough channel.
    let http_client = MockStreamingClient {
        sse_bytes: sse_bytes_from_data_lines([
            "{\"type\":\"copilot.heartbeat\",\"payload\":{}}",
            "{\"choices\":[{\"delta\":{\"content\":\"hello\"},\"finish_reason\":null}],\"usage\":null}",
            "{\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":null}",
            "[DONE]",
        ]),
    };
    let client = Client::builder()
        .api_key("copilot-token")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.completion_model("gpt-4o");
    let request = model.completion_request("hello").build();
    let mut stream = model.stream(request).await.expect("stream should start");

    let mut text = String::new();
    let mut terminal = None;
    let mut unknown = None;
    while let Some(item) = stream.next().await {
        match item.expect("unrecognized events must not surface errors") {
            StreamEvent::BlockDelta {
                delta: Delta::Text { text: chunk },
                ..
            } => text.push_str(&chunk),
            StreamEvent::Final(final_response) => terminal = Some(final_response),
            StreamEvent::Unknown(value) => unknown = Some(value),
            StreamEvent::BlockStart { .. }
            | StreamEvent::BlockEnd {
                end: BlockClose::Text,
                ..
            } => {}
            other => panic!("unexpected stream item: {other:?}"),
        }
    }

    assert_eq!(text, "hello");
    assert_eq!(
        unknown,
        Some(serde_json::json!({"type": "copilot.heartbeat", "payload": {}}).into()),
        "the unrecognized frame must surface verbatim on the raw channel"
    );
    let terminal = terminal.expect("stream should still emit its terminal record");
    assert_eq!(
        terminal.finish_reason,
        Some(crate::completion::FinishReason::Stop)
    );
}

#[tokio::test]
async fn responses_stream_preserves_reasoning_metadata_on_final_response() {
    let metadata = serde_json::json!({
        "context": "all_turns",
        "effort": "ultra",
        "summary": null,
        "future_control": true
    });
    let completed = serde_json::json!({
        "type": "response.completed",
        "sequence_number": 1,
        "response": {
            "id": "resp_123",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "error": null,
            "incomplete_details": null,
            "instructions": null,
            "max_output_tokens": null,
            "model": "gpt-5.3-codex",
            "reasoning": metadata.clone(),
            "usage": null,
            "output": [],
            "tools": []
        }
    });
    let http_client = MockStreamingClient {
        sse_bytes: sse_bytes_from_json_events(&[completed]),
    };
    let client = Client::builder()
        .api_key("copilot-token")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.completion_model("gpt-5.3-codex");
    let request = model.completion_request("hello").build();
    // Reasoning metadata is Copilot's own terminal payload, not part of
    // the normalized `StreamFinal`, so this reads it back off the terminal
    // record's `raw` — the provider's own type, serialized.
    let mut stream = model.stream(request).await.expect("stream should start");

    while let Some(item) = stream.next().await {
        if let crate::streaming::StreamEvent::Final(record) =
            item.expect("completed stream should not error")
        {
            let response: crate::providers::openai::responses_api::streaming::StreamingCompletionResponse =
                serde_json::from_value(record.raw).expect("raw terminal is the Responses record");
            assert_eq!(response.reasoning_context.as_deref(), Some("all_turns"));
            assert_eq!(response.reasoning_metadata.as_ref(), metadata.as_object());
            return;
        }
    }

    panic!("responses stream should yield a final response");
}

#[tokio::test]
async fn chat_stream_terminates_after_transport_error() {
    let chunks = vec![
        Ok(sse_bytes_from_data_lines([
            "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_123\",\"function\":{\"name\":\"ping\",\"arguments\":\"\"}}]},\"finish_reason\":null}],\"usage\":null}",
        ])),
        Err(http_client::Error::InvalidStatusCode(
            http::StatusCode::BAD_GATEWAY,
        )),
    ];

    let http_client = SequencedStreamingHttpClient::new(chunks);
    let client = Client::builder()
        .api_key("copilot-token")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.completion_model("gpt-4o");
    let request = model.completion_request("hello").build();
    let mut stream = model.stream(request).await.expect("stream should start");

    // The fully-delivered tool call is content, so it is flushed *before*
    // the terminal error: consumers that stop at the first `Err` still
    // see the completed work.
    let mut saw_error = false;
    let mut saw_tool_call = false;
    while let Some(item) = stream.next().await {
        match item {
            Ok(StreamEvent::BlockStart { .. } | StreamEvent::BlockDelta { .. }) => {
                assert!(!saw_error, "deltas should precede the terminal error");
            }
            Ok(StreamEvent::BlockEnd {
                block: Some(AssistantContent::ToolCall(tool_call)),
                ..
            }) => {
                assert!(
                    !saw_error,
                    "flushed tool call should precede the terminal error"
                );
                assert_eq!(tool_call.function.name, "ping");
                saw_tool_call = true;
            }
            Err(err) => {
                assert_eq!(
                    err.to_string(),
                    "HttpError: Invalid status code: 502 Bad Gateway"
                );
                assert_eq!(
                    err.provider_response_status(),
                    Some(http::StatusCode::BAD_GATEWAY)
                );
                assert_eq!(err.provider_response_body(), None);
                saw_error = true;
            }
            Ok(other) => panic!("unexpected stream item: {other:?}"),
        }
    }

    assert!(
        saw_tool_call,
        "fully-delivered tool call should be flushed before the error"
    );
    assert!(saw_error, "stream should surface the transport error");
    assert!(
        stream.next().await.is_none(),
        "chat stream should end without a terminal record after a transport error"
    );
}

#[test]
fn env_api_key_prefers_github_prefixed_vars() {
    let env = env_map(&[
        ("COPILOT_API_KEY", "copilot-key"),
        ("GITHUB_COPILOT_API_KEY", "github-key"),
        ("GITHUB_TOKEN", "bootstrap-token"),
    ]);
    let get = |name: &str| env.get(name).cloned();

    assert_eq!(env_api_key(&get).as_deref(), Some("github-key"));
}

#[test]
fn env_github_access_token_prefers_explicit_bootstrap_var() {
    let env = env_map(&[
        ("COPILOT_GITHUB_ACCESS_TOKEN", "explicit-bootstrap"),
        ("GITHUB_TOKEN", "fallback-bootstrap"),
    ]);
    let get = |name: &str| env.get(name).cloned();

    assert_eq!(
        env_github_access_token(&get).as_deref(),
        Some("explicit-bootstrap")
    );
}

#[test]
fn env_base_url_prefers_github_prefixed_vars() {
    let env = env_map(&[
        ("COPILOT_BASE_URL", "https://copilot.example"),
        ("GITHUB_COPILOT_API_BASE", "https://github.example"),
    ]);
    let get = |name: &str| env.get(name).cloned();

    assert_eq!(
        env_base_url(&get).as_deref(),
        Some("https://github.example")
    );
}

#[test]
fn env_without_api_key_falls_back_to_oauth() {
    let env = env_map(&[("COPILOT_BASE_URL", "https://copilot.example")]);
    let get = |name: &str| env.get(name).cloned();

    assert!(env_api_key(&get).is_none());
    assert!(env_github_access_token(&get).is_none());
    assert_eq!(
        env_base_url(&get).as_deref(),
        Some("https://copilot.example")
    );
}

#[test]
fn env_github_token_is_not_treated_as_copilot_api_key() {
    let env = env_map(&[("GITHUB_TOKEN", "bootstrap-token")]);
    let get = |name: &str| env.get(name).cloned();

    assert!(env_api_key(&get).is_none());
    assert_eq!(
        env_github_access_token(&get).as_deref(),
        Some("bootstrap-token")
    );
}
