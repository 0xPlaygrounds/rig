use super::*;
use crate::providers::gemini::completion::gemini_api_types::TrafficType;
use serde_json::json;

#[test]
fn test_deserialize_stream_response_with_single_text_part() {
    let json_data = json!({
        "candidates": [{
            "content": {
                "parts": [
                    {"text": "Hello, world!"}
                ],
                "role": "model"
            },
            "finishReason": "STOP",
            "index": 0
        }],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "totalTokenCount": 15
        }
    });

    let response: StreamGenerateContentResponse = serde_json::from_value(json_data).unwrap();
    assert_eq!(response.candidates.len(), 1);
    assert!(matches!(
        response.candidates[0].finish_reason,
        Some(FinishReason::Stop)
    ));
    let content = response.candidates[0]
        .content
        .as_ref()
        .expect("candidate should contain content");
    assert_eq!(content.parts.len(), 1);

    if let Part {
        part: PartKind::Text(text),
        ..
    } = &content.parts[0]
    {
        assert_eq!(text, "Hello, world!");
    } else {
        panic!("Expected text part");
    }
}

#[test]
fn test_streaming_tool_protocol_finish_reason_returns_response_error() {
    for (finish_reason, reason_name, finish_message) in [
        (
            "MALFORMED_FUNCTION_CALL",
            "MalformedFunctionCall",
            "malformed function call: default_api",
        ),
        (
            "UNEXPECTED_TOOL_CALL",
            "UnexpectedToolCall",
            "unexpected tool call: default_api",
        ),
        (
            "MISSING_THOUGHT_SIGNATURE",
            "MissingThoughtSignature",
            "missing thought signature for tool call",
        ),
        (
            "TOO_MANY_TOOL_CALLS",
            "TooManyToolCalls",
            "too many tool calls in response",
        ),
        (
            "MALFORMED_RESPONSE",
            "MalformedResponse",
            "malformed response from provider",
        ),
    ] {
        let json_data = json!({
            "candidates": [{
                "finishReason": finish_reason,
                "finishMessage": finish_message,
                "index": 0
            }]
        });

        let response: StreamGenerateContentResponse = serde_json::from_value(json_data).unwrap();
        let candidate = response
            .candidates
            .first()
            .expect("expected terminal candidate");
        let err = tool_protocol_finish_reason_error(candidate)
            .expect("tool protocol finish reason should be an error");

        assert!(matches!(
            err,
            CompletionError::ResponseError(message)
                if message.contains(reason_name)
                    && message.contains(finish_message)
        ));
    }
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[tokio::test]
async fn tool_protocol_failure_ends_the_stream_without_draining_later_frames() {
    use crate::client::CompletionClient;
    use crate::completion::CompletionModel as _;
    use crate::providers::gemini::Client;
    use crate::streaming::{Delta, StreamEvent};
    use crate::test_utils::MockStreamingClient;
    use futures::StreamExt;

    // A tool-protocol terminal failure, then more frames: a well-formed
    // text chunk, an unknown frame, and a terminal `finishReason` chunk.
    // The failure must be the LAST item the consumer sees — the driver
    // stops reading (`is_finished`), so nothing after it is interpreted
    // or passed through as `Unknown`.
    let frames = [
        r#"{"candidates":[{"content":{"parts":[{"text":"hi"}],"role":"model"},"index":0}]}"#,
        r#"{"candidates":[{"finishReason":"MALFORMED_FUNCTION_CALL","finishMessage":"malformed function call","index":0}]}"#,
        r#"{"candidates":[{"content":{"parts":[{"text":"dead"}],"role":"model"},"index":0}]}"#,
        r#"{"someFutureField":{"x":1}}"#,
        r#"{"candidates":[{"content":{"parts":[],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}"#,
    ];
    let sse_bytes = bytes::Bytes::from(
        frames
            .iter()
            .map(|frame| format!("data: {frame}\n\n"))
            .collect::<String>(),
    );

    let client = Client::builder()
        .api_key("test-key")
        .http_client(MockStreamingClient { sse_bytes })
        .build()
        .expect("build client");
    let model = client.completion_model("gemini-2.5-flash");
    let request = model.completion_request("hello").build();
    let mut stream = crate::completion::CompletionModel::stream(&model, request)
        .await
        .expect("stream should open");

    let mut texts = Vec::new();
    let mut saw_error = false;
    let mut items_after_error = 0usize;
    while let Some(item) = stream.next().await {
        if saw_error {
            items_after_error += 1;
        }
        match item {
            Ok(StreamEvent::BlockDelta {
                delta: Delta::Text { text },
                ..
            }) => texts.push(text),
            Ok(_) => {}
            Err(_) => saw_error = true,
        }
    }

    assert_eq!(texts, ["hi"]);
    assert!(
        saw_error,
        "the tool-protocol failure must reach the consumer"
    );
    assert_eq!(
        items_after_error, 0,
        "the in-band failure must end the stream: no later text, Unknown passthrough, or terminal"
    );
    assert!(stream.response.is_none());
}

#[test]
fn test_deserialize_stream_response_with_usage_only_chunk() {
    let json_data = json!({
        "responseId": "response-123",
        "modelVersion": "gemini-2.0-flash-001",
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "totalTokenCount": 15
        }
    });

    let response: StreamGenerateContentResponse = serde_json::from_value(json_data).unwrap();
    assert_eq!(response.response_id.as_deref(), Some("response-123"));
    assert_eq!(
        response.model_version.as_deref(),
        Some("gemini-2.0-flash-001")
    );
    assert!(response.candidates.is_empty());

    let usage = response
        .usage_metadata
        .as_ref()
        .map(crate::completion::Usage::from)
        .unwrap();
    assert_eq!(usage.input_tokens, 10);
    assert_eq!(usage.output_tokens, 5);
    assert_eq!(usage.total_tokens, 15);
}

#[test]
fn test_deserialize_stream_response_with_multiple_text_parts() {
    let json_data = json!({
        "candidates": [{
            "content": {
                "parts": [
                    {"text": "Hello, "},
                    {"text": "world!"},
                    {"text": " How are you?"}
                ],
                "role": "model"
            },
            "finishReason": "STOP",
            "index": 0
        }],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 8,
            "totalTokenCount": 18
        }
    });

    let response: StreamGenerateContentResponse = serde_json::from_value(json_data).unwrap();
    assert_eq!(response.candidates.len(), 1);
    let content = response.candidates[0]
        .content
        .as_ref()
        .expect("candidate should contain content");
    assert_eq!(content.parts.len(), 3);

    // Verify all three text parts are present
    for (i, expected_text) in ["Hello, ", "world!", " How are you?"].iter().enumerate() {
        if let Part {
            part: PartKind::Text(text),
            ..
        } = &content.parts[i]
        {
            assert_eq!(text, expected_text);
        } else {
            panic!("Expected text part at index {i}");
        }
    }
}

#[test]
fn test_deserialize_stream_response_with_multiple_tool_calls() {
    let json_data = json!({
        "candidates": [{
            "content": {
                "parts": [
                    {
                        "functionCall": {
                            "name": "get_weather",
                            "args": {"city": "San Francisco"},
                            "id": "call-weather"
                        }
                    },
                    {
                        "functionCall": {
                            "name": "get_temperature",
                            "args": {"location": "New York"},
                            "id": "call-temperature"
                        }
                    }
                ],
                "role": "model"
            },
            "finishReason": "STOP",
            "index": 0
        }],
        "usageMetadata": {
            "promptTokenCount": 50,
            "candidatesTokenCount": 20,
            "totalTokenCount": 70
        }
    });

    let response: StreamGenerateContentResponse = serde_json::from_value(json_data).unwrap();
    let content = response.candidates[0]
        .content
        .as_ref()
        .expect("candidate should contain content");
    assert_eq!(content.parts.len(), 2);

    // Verify first tool call
    if let Part {
        part: PartKind::FunctionCall(call),
        ..
    } = &content.parts[0]
    {
        assert_eq!(call.name, "get_weather");
        assert_eq!(call.id.as_deref(), Some("call-weather"));
    } else {
        panic!("Expected function call at index 0");
    }

    // Verify second tool call
    if let Part {
        part: PartKind::FunctionCall(call),
        ..
    } = &content.parts[1]
    {
        assert_eq!(call.name, "get_temperature");
        assert_eq!(call.id.as_deref(), Some("call-temperature"));
    } else {
        panic!("Expected function call at index 1");
    }
}

#[test]
fn test_deserialize_stream_response_with_mixed_parts() {
    let json_data = json!({
        "candidates": [{
            "content": {
                "parts": [
                    {
                        "text": "Let me think about this...",
                        "thought": true
                    },
                    {
                        "text": "Here's my response: "
                    },
                    {
                        "functionCall": {
                            "name": "search",
                            "args": {"query": "rust async"}
                        }
                    },
                    {
                        "text": "I found the answer!"
                    }
                ],
                "role": "model"
            },
            "finishReason": "STOP",
            "index": 0
        }],
        "usageMetadata": {
            "promptTokenCount": 100,
            "candidatesTokenCount": 50,
            "thoughtsTokenCount": 15,
            "totalTokenCount": 165
        }
    });

    let response: StreamGenerateContentResponse = serde_json::from_value(json_data).unwrap();
    let content = response.candidates[0]
        .content
        .as_ref()
        .expect("candidate should contain content");
    let parts = &content.parts;
    assert_eq!(parts.len(), 4);

    // Verify reasoning (thought) part
    if let Part {
        part: PartKind::Text(text),
        thought: Some(true),
        ..
    } = &parts[0]
    {
        assert_eq!(text, "Let me think about this...");
    } else {
        panic!("Expected thought part at index 0");
    }

    // Verify regular text
    if let Part {
        part: PartKind::Text(text),
        thought,
        ..
    } = &parts[1]
    {
        assert_eq!(text, "Here's my response: ");
        assert!(thought.is_none() || thought == &Some(false));
    } else {
        panic!("Expected text part at index 1");
    }

    // Verify tool call
    if let Part {
        part: PartKind::FunctionCall(call),
        ..
    } = &parts[2]
    {
        assert_eq!(call.name, "search");
    } else {
        panic!("Expected function call at index 2");
    }

    // Verify final text
    if let Part {
        part: PartKind::Text(text),
        ..
    } = &parts[3]
    {
        assert_eq!(text, "I found the answer!");
    } else {
        panic!("Expected text part at index 3");
    }
}

#[test]
fn test_deserialize_stream_response_with_empty_parts() {
    let json_data = json!({
        "candidates": [{
            "content": {
                "parts": [],
                "role": "model"
            },
            "finishReason": "STOP",
            "index": 0
        }],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 0,
            "totalTokenCount": 10
        }
    });

    let response: StreamGenerateContentResponse = serde_json::from_value(json_data).unwrap();
    let content = response.candidates[0]
        .content
        .as_ref()
        .expect("candidate should contain content");
    assert_eq!(content.parts.len(), 0);
}

#[test]
fn test_partial_usage_token_calculation() {
    let usage = PartialUsage {
        total_token_count: 100,
        cached_content_token_count: Some(20),
        candidates_token_count: Some(30),
        thoughts_token_count: Some(10),
        prompt_token_count: 40,
        prompt_tokens_details: None,
        cache_tokens_details: None,
        candidates_tokens_details: None,
        tool_use_prompt_token_count: Some(12),
        tool_use_prompt_tokens_details: None,
        traffic_type: None,
    };

    let token_usage = crate::completion::Usage::from(&usage);
    assert_eq!(token_usage.input_tokens, 40);
    assert_eq!(token_usage.cached_input_tokens, 20);
    assert_eq!(token_usage.output_tokens, 30);
    assert_eq!(token_usage.reasoning_tokens, 10);
    assert_eq!(token_usage.tool_use_prompt_tokens, 12);
    assert_eq!(token_usage.total_tokens, 100);
}

#[test]
fn test_partial_usage_with_missing_counts() {
    let usage = PartialUsage {
        total_token_count: 50,
        cached_content_token_count: None,
        candidates_token_count: Some(30),
        thoughts_token_count: None,
        prompt_token_count: 20,
        prompt_tokens_details: None,
        cache_tokens_details: None,
        candidates_tokens_details: None,
        tool_use_prompt_token_count: None,
        tool_use_prompt_tokens_details: None,
        traffic_type: None,
    };

    let token_usage = crate::completion::Usage::from(&usage);
    assert_eq!(token_usage.input_tokens, 20);
    assert_eq!(token_usage.cached_input_tokens, 0);
    assert_eq!(token_usage.output_tokens, 30);
    assert_eq!(token_usage.reasoning_tokens, 0);
    assert_eq!(token_usage.total_tokens, 50);
}

#[test]
fn test_partial_usage_deserializes_without_total_token_count() {
    // Gemini's proto3-JSON encoding omits fields whose value is the default (0),
    // so `totalTokenCount` is absent on short/empty/blocked generations.
    let usage: PartialUsage =
        serde_json::from_str(r#"{"promptTokenCount": 12}"#).expect("should deserialize");
    assert_eq!(usage.total_token_count, 0);
    assert_eq!(usage.prompt_token_count, 12);
}

#[test]
fn test_streaming_completion_response_has_finish_reason_and_model_version() {
    use super::super::completion::gemini_api_types::FinishReason;

    let response = StreamingCompletionResponse {
        usage_metadata: PartialUsage::default(),
        finish_reason: Some(FinishReason::Stop),
        finish_message: None,
        model_version: Some("gemini-2.5-pro-preview-05-06".to_string()),
        response_id: None,
    };

    assert!(matches!(response.finish_reason, Some(FinishReason::Stop)));
    assert_eq!(
        response.model_version.as_deref(),
        Some("gemini-2.5-pro-preview-05-06")
    );

    let json = serde_json::to_string(&response).unwrap();
    let deserialized: StreamingCompletionResponse = serde_json::from_str(&json).unwrap();
    assert!(matches!(
        deserialized.finish_reason,
        Some(FinishReason::Stop)
    ));
    assert_eq!(
        deserialized.model_version.as_deref(),
        Some("gemini-2.5-pro-preview-05-06")
    );
}

#[test]
fn test_streaming_completion_response_token_usage() {
    let response = StreamingCompletionResponse {
        usage_metadata: PartialUsage {
            total_token_count: 150,
            cached_content_token_count: None,
            candidates_token_count: Some(75),
            thoughts_token_count: None,
            prompt_token_count: 75,
            prompt_tokens_details: None,
            cache_tokens_details: None,
            candidates_tokens_details: None,
            tool_use_prompt_token_count: None,
            tool_use_prompt_tokens_details: None,
            traffic_type: None,
        },
        finish_reason: Some(FinishReason::Stop),
        finish_message: None,
        model_version: Some("gemini-2.0-flash-001".to_string()),
        response_id: None,
    };

    let token_usage = crate::completion::Usage::from(&response);
    assert_eq!(token_usage.input_tokens, 75);
    assert_eq!(token_usage.output_tokens, 75);
    assert_eq!(token_usage.reasoning_tokens, 0);
    assert_eq!(token_usage.cached_input_tokens, 0);
    assert_eq!(token_usage.total_tokens, 150);
    assert!(matches!(response.finish_reason, Some(FinishReason::Stop)));
    assert_eq!(
        response.model_version.as_deref(),
        Some("gemini-2.0-flash-001")
    );
}

#[test]
fn test_partial_usage_serde_roundtrip_with_all_optional_fields() {
    let json_data = serde_json::json!({
        "promptTokenCount": 100,
        "cachedContentTokenCount": 25,
        "candidatesTokenCount": 50,
        "thoughtsTokenCount": 15,
        "totalTokenCount": 190,
        "promptTokensDetails": [
            { "modality": "TEXT", "tokenCount": 80 },
            { "modality": "IMAGE", "tokenCount": 20 }
        ],
        "cacheTokensDetails": [
            { "modality": "TEXT", "tokenCount": 25 }
        ],
        "candidatesTokensDetails": [
            { "modality": "TEXT", "tokenCount": 50 }
        ],
        "toolUsePromptTokenCount": 12,
        "toolUsePromptTokensDetails": [
            { "modality": "TEXT", "tokenCount": 12 }
        ],
        "trafficType": "PROVISIONED_THROUGHPUT"
    });

    let usage: PartialUsage = serde_json::from_value(json_data).unwrap();
    assert_eq!(usage.prompt_token_count, 100);
    assert_eq!(usage.cached_content_token_count, Some(25));
    assert_eq!(usage.candidates_token_count, Some(50));
    assert_eq!(usage.thoughts_token_count, Some(15));
    assert_eq!(usage.total_token_count, 190);
    assert!(usage.prompt_tokens_details.is_some());
    assert_eq!(usage.prompt_tokens_details.as_ref().unwrap().len(), 2);
    assert!(usage.cache_tokens_details.is_some());
    assert!(usage.candidates_tokens_details.is_some());
    assert_eq!(usage.tool_use_prompt_token_count, Some(12));
    assert!(usage.tool_use_prompt_tokens_details.is_some());
    assert!(matches!(
        usage.traffic_type,
        Some(TrafficType::ProvisionedThroughput)
    ));

    let token_usage = crate::completion::Usage::from(&usage);
    assert_eq!(token_usage.input_tokens, 100);
    assert_eq!(token_usage.cached_input_tokens, 25);
    assert_eq!(token_usage.output_tokens, 50);
    assert_eq!(token_usage.reasoning_tokens, 15);
    assert_eq!(token_usage.tool_use_prompt_tokens, 12);
    assert_eq!(token_usage.total_tokens, 190);
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
mod terminal_emission {
    use crate::client::CompletionClient;
    use crate::completion::CompletionModel as _;
    use crate::providers::gemini::Client;
    use crate::streaming::{Delta, StreamEvent};
    use crate::test_utils::MockStreamingClient;
    use futures::StreamExt;

    const CONTENT_CHUNK: &str = r#"{"candidates":[{"content":{"parts":[{"text":"hi"}],"role":"model"}}],"responseId":"resp-1","modelVersion":"gemini-2.5-pro"}"#;
    const TERMINAL_CHUNK: &str = r#"{"candidates":[{"content":{"parts":[{"text":"!"}],"role":"model"},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":2,"totalTokenCount":7},"responseId":"resp-1","modelVersion":"gemini-2.5-pro"}"#;

    fn sse(frames: &[&str]) -> bytes::Bytes {
        bytes::Bytes::from(
            frames
                .iter()
                .map(|frame| format!("data: {frame}\n\n"))
                .collect::<String>(),
        )
    }

    async fn collect(
        sse_bytes: bytes::Bytes,
    ) -> (
        Vec<String>,
        bool,
        bool,
        crate::streaming::StreamingCompletionResponse,
    ) {
        let client = Client::builder()
            .api_key("test-key")
            .http_client(MockStreamingClient { sse_bytes })
            .build()
            .expect("build client");
        let model = client
            .completion_model(crate::providers::gemini::completion::GEMINI_2_5_PRO_PREVIEW_06_05);
        let request = model.completion_request("hello").build();
        let mut stream = crate::completion::CompletionModel::stream(&model, request)
            .await
            .expect("stream should open");

        let mut texts = Vec::new();
        let mut saw_error = false;
        let mut saw_terminal = false;
        while let Some(item) = stream.next().await {
            match item {
                Ok(StreamEvent::BlockDelta {
                    delta: Delta::Text { text },
                    ..
                }) => texts.push(text),
                Ok(StreamEvent::Final(_)) => saw_terminal = true,
                Ok(_) => {}
                Err(_) => saw_error = true,
            }
        }
        (texts, saw_error, saw_terminal, stream)
    }

    #[tokio::test]
    async fn a_signature_with_no_thought_text_still_emits_a_signed_block() {
        // gRPC and Interactions emit signature-only blocks; the REST
        // wire must not diverge — the signature is replay-required
        // provider state even when no thought text accumulated.
        const SIGNATURE_ONLY_CHUNK: &str = r#"{"candidates":[{"content":{"parts":[{"text":"","thought":true,"thoughtSignature":"sig-only"}],"role":"model"},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":1,"totalTokenCount":4}}"#;

        let client = Client::builder()
            .api_key("test-key")
            .http_client(MockStreamingClient {
                sse_bytes: sse(&[SIGNATURE_ONLY_CHUNK]),
            })
            .build()
            .expect("build client");
        let model = client
            .completion_model(crate::providers::gemini::completion::GEMINI_2_5_PRO_PREVIEW_06_05);
        let request = model.completion_request("hello").build();
        let mut stream = crate::completion::CompletionModel::stream(&model, request)
            .await
            .expect("stream should open");

        let mut signed = None;
        while let Some(item) = stream.next().await {
            if let StreamEvent::BlockEnd {
                block: Some(crate::message::AssistantContent::Reasoning(reasoning)),
                ..
            } = item.expect("stream item should be Ok")
            {
                signed = Some(reasoning);
            }
        }
        let signed = signed.expect("signature-only block must be emitted");
        assert!(signed.content.iter().any(|content| matches!(
            content,
            crate::message::ReasoningContent::Text { signature: Some(sig), .. } if sig == "sig-only"
        )));
    }

    #[tokio::test]
    async fn truncated_stream_yields_content_but_no_terminal_record() {
        let (texts, saw_error, saw_terminal, stream) = collect(sse(&[CONTENT_CHUNK])).await;

        assert_eq!(texts, ["hi"]);
        assert!(!saw_error);
        assert!(
            !saw_terminal,
            "EOF without a finishReason chunk must not synthesize a terminal record"
        );
        assert!(stream.response.is_none());
    }

    #[tokio::test]
    async fn errored_stream_forwards_the_error_and_no_terminal_record() {
        use crate::test_utils::SequencedStreamingHttpClient;

        // A transport failure after some content must reach the consumer
        // and must not be papered over with a synthesized terminal record.
        let client = Client::builder()
            .api_key("test-key")
            .http_client(SequencedStreamingHttpClient::new(vec![
                Ok(sse(&[CONTENT_CHUNK])),
                Err(crate::http_client::Error::InvalidStatusCodeWithMessage(
                    http::StatusCode::BAD_GATEWAY,
                    "connection reset".to_string(),
                )),
            ]))
            .build()
            .expect("build client");
        let model = client
            .completion_model(crate::providers::gemini::completion::GEMINI_2_5_PRO_PREVIEW_06_05);
        let request = model.completion_request("hello").build();
        let mut stream = crate::completion::CompletionModel::stream(&model, request)
            .await
            .expect("stream should open");

        let mut texts = Vec::new();
        let mut saw_error = false;
        let mut saw_terminal = false;
        while let Some(item) = stream.next().await {
            match item {
                Ok(StreamEvent::BlockDelta {
                    delta: Delta::Text { text },
                    ..
                }) => texts.push(text),
                Ok(StreamEvent::Final(_)) => saw_terminal = true,
                Ok(_) => {}
                Err(_) => saw_error = true,
            }
        }

        assert_eq!(texts, ["hi"]);
        assert!(saw_error, "the transport failure must reach the consumer");
        assert!(
            !saw_terminal,
            "a failed stream must not synthesize a terminal record"
        );
        assert!(stream.response.is_none());
    }

    #[tokio::test]
    async fn malformed_frame_then_eof_yields_error_and_no_terminal_record() {
        let (texts, saw_error, saw_terminal, stream) =
            collect(sse(&[CONTENT_CHUNK, "{not json"])).await;

        assert_eq!(texts, ["hi"]);
        assert!(saw_error, "the malformed frame must reach the consumer");
        assert!(
            !saw_terminal,
            "a parse error followed by EOF must not read as a completed turn"
        );
        assert!(stream.response.is_none());
    }

    #[tokio::test]
    async fn malformed_frame_then_real_terminal_still_completes_the_stream() {
        let (texts, saw_error, saw_terminal, stream) =
            collect(sse(&[CONTENT_CHUNK, "{not json", TERMINAL_CHUNK])).await;

        assert_eq!(texts, ["hi", "!"]);
        assert!(saw_error, "the malformed frame must reach the consumer");
        assert!(
            saw_terminal,
            "a genuine finishReason chunk after a parse error still completes the stream"
        );
        let terminal = stream.response.expect("terminal record");
        assert_eq!(
            terminal.finish_reason,
            Some(crate::completion::FinishReason::Stop)
        );
        assert_eq!(terminal.response_id.as_deref(), Some("resp-1"));
    }
}
