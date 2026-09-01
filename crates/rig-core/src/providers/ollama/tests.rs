use super::*;
use serde_json::json;

// The NDJSON wire has no discriminator, so its classify has exactly two
// outcomes: the response shape or corrupt.
#[test]
fn classify_ndjson_line_is_known_or_corrupt() {
    let line = json!({
        "model": "llama3.2",
        "created_at": "2024-01-01T00:00:00Z",
        "message": {"role": "assistant", "content": "hi"},
        "done": false,
    })
    .to_string();
    assert!(matches!(
        internal::wire::classify_untyped_line::<CompletionResponse>(line.as_bytes()),
        internal::wire::WireEvent::Known(_)
    ));
    assert!(matches!(
        internal::wire::classify_untyped_line::<CompletionResponse>(b"{not json"),
        internal::wire::WireEvent::Corrupt(_)
    ));
    assert!(matches!(
        internal::wire::classify_untyped_line::<CompletionResponse>(br#"{"done": 42}"#),
        internal::wire::WireEvent::Corrupt(_)
    ));
}

#[test]
fn splits_legacy_reasoning_with_or_without_opening_marker() {
    assert_eq!(
        split_legacy_thinking("<think>private reasoning</think>\n\nvisible answer", false),
        (Some("private reasoning"), "visible answer")
    );
    assert_eq!(
        split_legacy_thinking("private reasoning\n</think>\n\nvisible answer", true),
        (Some("private reasoning"), "visible answer")
    );
}

#[test]
fn leaves_unterminated_or_inline_reasoning_markers_visible() {
    assert_eq!(
        split_legacy_thinking("<think>unterminated", true),
        (None, "<think>unterminated")
    );
    assert_eq!(
        split_legacy_thinking("The literal marker is <think>.", true),
        (None, "The literal marker is <think>.")
    );
    assert_eq!(
        split_legacy_thinking("  visible indentation", true),
        (None, "  visible indentation")
    );
    assert_eq!(
        split_legacy_thinking("The closing token </think> is XML-like.", true),
        (None, "The closing token </think> is XML-like.")
    );
    assert_eq!(
        split_legacy_thinking("Example:\n</think>\nis a closing tag.", true),
        (None, "Example:\n</think>\nis a closing tag.")
    );
}

// Test deserialization and conversion for the /api/chat endpoint.
#[tokio::test]
async fn test_chat_completion() {
    // Sample JSON response from /api/chat (non-streaming) based on Ollama docs.
    let sample_chat_response = json!({
        "model": "llama3.2",
        "created_at": "2023-08-04T19:22:45.499127Z",
        "message": {
            "role": "assistant",
            "content": "The sky is blue because of Rayleigh scattering.",
            "images": null,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_current_weather",
                        "arguments": {
                            "location": "San Francisco, CA",
                            "format": "celsius"
                        }
                    }
                }
            ]
        },
        "done": true,
        "total_duration": 8000000000u64,
        "load_duration": 6000000u64,
        "prompt_eval_count": 61u64,
        "prompt_eval_duration": 400000000u64,
        "eval_count": 468u64,
        "eval_duration": 7700000000u64
    });
    let sample_text = sample_chat_response.to_string();

    let chat_resp: CompletionResponse =
        serde_json::from_str(&sample_text).expect("Invalid JSON structure");
    let conv: completion::CompletionResponse = chat_resp.try_into().unwrap();
    assert!(
        !conv.choice.is_empty(),
        "Expected non-empty choice in chat response"
    );
}

#[test]
fn done_reason_maps_documented_values_and_preserves_the_rest() {
    assert_eq!(map_done_reason("stop"), completion::FinishReason::Stop);
    assert_eq!(map_done_reason("length"), completion::FinishReason::Length);
    // Ollama's operational reasons have no normalized equivalent, so they
    // are carried through verbatim rather than read as a natural stop.
    assert_eq!(
        map_done_reason("load"),
        completion::FinishReason::Other("load".to_owned())
    );
    assert_eq!(
        map_done_reason("unload"),
        completion::FinishReason::Other("unload".to_owned())
    );
}

#[test]
fn response_metadata_is_normalized() {
    let response: CompletionResponse = serde_json::from_value(json!({
        "model": "llama3.2",
        "created_at": "2023-08-04T19:22:45.499127Z",
        "message": {"role": "assistant", "content": "Hi!", "tool_calls": []},
        "done": true,
        "done_reason": "length",
        "prompt_eval_count": 12u64,
        "eval_count": 3u64
    }))
    .expect("fixture should deserialize");

    let normalized: completion::CompletionResponse =
        response.try_into().expect("normalization should succeed");

    assert_eq!(normalized.provider, PROVIDER_NAME);
    assert_eq!(normalized.model.as_deref(), Some("llama3.2"));
    assert_eq!(
        normalized.finish_reason(),
        Some(completion::FinishReason::Length)
    );
    // Ollama assigns no message identifier.
    assert_eq!(normalized.message_id, None);
    assert_eq!(normalized.usage.input_tokens, 12);
    assert_eq!(normalized.usage.output_tokens, 3);
    assert_eq!(normalized.usage.total_tokens, 15);
}

// A `done_reason` of `stop` on a turn that actually called a tool must be
// upgraded by the response builder's reconciliation.
#[test]
fn tool_call_turn_upgrades_a_plain_stop_to_tool_calls() {
    let response: CompletionResponse = serde_json::from_value(json!({
        "model": "qwen3:4b",
        "created_at": "2023-08-04T19:22:45.499127Z",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": "get_weather", "arguments": {"location": "Berlin"}}}
            ]
        },
        "done": true,
        "done_reason": "stop"
    }))
    .expect("fixture should deserialize");

    let normalized: completion::CompletionResponse =
        response.try_into().expect("normalization should succeed");

    assert_eq!(
        normalized.finish_reason(),
        Some(completion::FinishReason::ToolCalls)
    );
}

#[test]
fn streaming_terminal_record_is_normalized() {
    let terminal = StreamingCompletionResponse {
        model: "llama3.2".to_string(),
        done_reason: Some("dragons".to_string()),
        total_duration: None,
        load_duration: None,
        prompt_eval_count: Some(7),
        prompt_eval_duration: None,
        eval_count: Some(5),
        eval_duration: None,
    };

    let final_record = StreamFinal::from(terminal);
    assert_eq!(final_record.provider, PROVIDER_NAME);
    assert_eq!(final_record.model.as_deref(), Some("llama3.2"));
    assert_eq!(
        final_record.finish_reason,
        Some(completion::FinishReason::Other("dragons".to_owned()))
    );
    assert_eq!(final_record.usage.total_tokens, 12);
}

// Test conversion from provider Message to completion Message.
#[test]
fn test_message_conversion() {
    // Construct a provider Message (User variant with String content).
    let provider_msg = Message::User {
        content: "Test message".to_owned(),
        images: None,
        name: None,
    };
    // Convert it into a completion::Message.
    let comp_msg: crate::completion::Message = provider_msg.into();
    match comp_msg {
        crate::completion::Message::User { content } => {
            let first_content = content.first();
            // The expected type is crate::completion::message::UserContent::Text wrapping a Text struct.
            match first_content {
                Some(crate::completion::message::UserContent::Text(text_struct)) => {
                    assert_eq!(text_struct.text, "Test message");
                }
                _ => panic!("Expected text content in conversion"),
            }
        }
        _ => panic!("Conversion from provider Message to completion Message failed"),
    }
}

#[test]
fn empty_assistant_history_converts_to_empty_content_not_a_sentinel() {
    // A content-less Ollama assistant message converts to genuinely empty
    // message content — no fabricated `Text("")` block. Pinned because the
    // consequence is deliberate: such a message cannot be replayed through
    // the request boundary, and callers ingesting raw Ollama history
    // filter it rather than rig inventing content (see the `From` doc).
    let provider_msg = Message::Assistant {
        content: String::new(),
        thinking: None,
        images: None,
        name: None,
        tool_calls: Vec::new(),
    };
    let comp_msg: crate::completion::Message = provider_msg.into();
    match comp_msg {
        crate::completion::Message::Assistant { content, .. } => {
            assert!(content.is_empty(), "expected empty content: {content:?}");
        }
        other => panic!("expected an assistant message, got {other:?}"),
    }

    // A non-empty body still converts to exactly one text block.
    let provider_msg = Message::Assistant {
        content: "hello".to_owned(),
        thinking: None,
        images: None,
        name: None,
        tool_calls: Vec::new(),
    };
    let comp_msg: crate::completion::Message = provider_msg.into();
    match comp_msg {
        crate::completion::Message::Assistant { content, .. } => {
            assert!(
                matches!(
                    content.as_slice(),
                    [crate::completion::message::AssistantContent::Text(text)]
                        if text.text == "hello"
                ),
                "unexpected content: {content:?}"
            );
        }
        other => panic!("expected an assistant message, got {other:?}"),
    }
}

#[test]
fn mixed_user_content_preserves_message_order() {
    use crate::message::{Message as RigMessage, ToolResultContent, UserContent};

    let message = RigMessage::User {
        content: vec![
            UserContent::text("before"),
            UserContent::tool_result(
                "",
                "lookup",
                vec![ToolResultContent::json(json!({ "ok": true }))],
            ),
            UserContent::text("after"),
        ],
    };

    let messages = Vec::<Message>::try_from(message).expect("mixed content should convert");
    assert_eq!(messages.len(), 3);
    assert!(matches!(
        &messages[0],
        Message::User { content, .. } if content == "before"
    ));
    assert!(matches!(
        &messages[1],
        Message::ToolResult { name, content }
            if name == "lookup" && content == r#"{"ok":true}"#
    ));
    assert!(matches!(
        &messages[2],
        Message::User { content, .. } if content == "after"
    ));
}

#[test]
fn unsupported_user_content_returns_a_conversion_error() {
    use crate::message::{ImageMediaType, Message as RigMessage, UserContent};

    let message = RigMessage::User {
        content: vec![UserContent::image_url(
            "https://example.com/image.png",
            Some(ImageMediaType::PNG),
            None,
        )],
    };

    let error = Vec::<Message>::try_from(message).expect_err("URL image should be rejected");
    assert!(error.to_string().contains("base64"));
}

// Test conversion of internal tool definition to Ollama's ToolDefinition format.
#[test]
fn test_tool_definition_conversion() {
    // Internal tool definition from the completion module.
    let internal_tool = crate::completion::ToolDefinition {
        name: "get_current_weather".to_owned(),
        description: "Get the current weather for a location".to_owned(),
        parameters: json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location to get the weather for, e.g. San Francisco, CA"
                },
                "format": {
                    "type": "string",
                    "description": "The format to return the weather in, e.g. 'celsius' or 'fahrenheit'",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["location", "format"]
        }),
    };
    // Convert internal tool to Ollama's tool definition.
    let ollama_tool: ToolDefinition = internal_tool.into();
    assert_eq!(ollama_tool.type_field, "function");
    assert_eq!(ollama_tool.function.name, "get_current_weather");
    assert_eq!(
        ollama_tool.function.description,
        "Get the current weather for a location"
    );
    // Check JSON fields in parameters.
    let params = &ollama_tool.function.parameters;
    assert_eq!(params["properties"]["location"]["type"], "string");
}

// Test deserialization of chat response with thinking content
#[tokio::test]
async fn test_chat_completion_with_thinking() {
    let sample_response = json!({
        "model": "qwen-thinking",
        "created_at": "2023-08-04T19:22:45.499127Z",
        "message": {
            "role": "assistant",
            "content": "The answer is 42.",
            "thinking": "Let me think about this carefully. The question asks for the meaning of life...",
            "images": null,
            "tool_calls": []
        },
        "done": true,
        "total_duration": 8000000000u64,
        "load_duration": 6000000u64,
        "prompt_eval_count": 61u64,
        "prompt_eval_duration": 400000000u64,
        "eval_count": 468u64,
        "eval_duration": 7700000000u64
    });

    let chat_resp: CompletionResponse =
        serde_json::from_value(sample_response).expect("Failed to deserialize");

    // Verify thinking field is present
    if let Message::Assistant {
        thinking, content, ..
    } = &chat_resp.message
    {
        assert_eq!(
            thinking.as_ref().unwrap(),
            "Let me think about this carefully. The question asks for the meaning of life..."
        );
        assert_eq!(content, "The answer is 42.");
    } else {
        panic!("Expected Assistant message");
    }
}

// Test deserialization of chat response without thinking content
#[tokio::test]
async fn test_chat_completion_without_thinking() {
    let sample_response = json!({
        "model": "llama3.2",
        "created_at": "2023-08-04T19:22:45.499127Z",
        "message": {
            "role": "assistant",
            "content": "Hello!",
            "images": null,
            "tool_calls": []
        },
        "done": true,
        "total_duration": 8000000000u64,
        "load_duration": 6000000u64,
        "prompt_eval_count": 10u64,
        "prompt_eval_duration": 400000000u64,
        "eval_count": 5u64,
        "eval_duration": 7700000000u64
    });

    let chat_resp: CompletionResponse =
        serde_json::from_value(sample_response).expect("Failed to deserialize");

    // Verify thinking field is None when not provided
    if let Message::Assistant {
        thinking, content, ..
    } = &chat_resp.message
    {
        assert!(thinking.is_none());
        assert_eq!(content, "Hello!");
    } else {
        panic!("Expected Assistant message");
    }
}

// Test deserialization of streaming response with thinking content
#[test]
fn test_streaming_response_with_thinking() {
    let sample_chunk = json!({
        "model": "qwen-thinking",
        "created_at": "2023-08-04T19:22:45.499127Z",
        "message": {
            "role": "assistant",
            "content": "",
            "thinking": "Analyzing the problem...",
            "images": null,
            "tool_calls": []
        },
        "done": false
    });

    let chunk: CompletionResponse =
        serde_json::from_value(sample_chunk).expect("Failed to deserialize");

    if let Message::Assistant {
        thinking, content, ..
    } = &chunk.message
    {
        assert_eq!(thinking.as_ref().unwrap(), "Analyzing the problem...");
        assert_eq!(content, "");
    } else {
        panic!("Expected Assistant message");
    }
}

// Test message conversion with thinking content
#[test]
fn test_message_conversion_with_thinking() {
    // Create an internal message with reasoning content
    let reasoning_content = crate::message::Reasoning::new("Step 1: Consider the problem");

    let internal_msg = crate::message::Message::Assistant {
        id: None,
        content: vec![
            crate::message::AssistantContent::Reasoning(reasoning_content),
            crate::message::AssistantContent::Text(crate::message::Text::new(
                "The answer is X".to_string(),
            )),
        ],
    };

    // Convert to provider Message
    let provider_msgs: Vec<Message> = internal_msg.try_into().unwrap();
    assert_eq!(provider_msgs.len(), 1);

    if let Message::Assistant {
        thinking, content, ..
    } = &provider_msgs[0]
    {
        assert_eq!(thinking.as_ref().unwrap(), "Step 1: Consider the problem");
        assert_eq!(content, "The answer is X");
    } else {
        panic!("Expected Assistant message with thinking");
    }
}

/// A user-supplied ollama-format assistant message carrying a
/// daemon-issued call id keeps it through conversion — the same id
/// policy as the unary decode (preserve when present, absent mints).
#[test]
fn wire_message_conversion_preserves_the_daemon_tool_call_id() {
    let wire = Message::Assistant {
        content: String::new(),
        thinking: None,
        images: None,
        name: None,
        tool_calls: vec![ToolCall {
            id: Some("call_abc".to_owned()),
            r#type: ToolType::default(),
            function: Function {
                name: "get_weather".to_owned(),
                arguments: json!({}),
            },
        }],
    };

    let converted: crate::completion::Message = wire.into();
    let crate::completion::Message::Assistant { content, .. } = converted else {
        panic!("Expected Assistant message");
    };
    let ids: Vec<String> = content
        .iter()
        .filter_map(|item| match item {
            crate::message::AssistantContent::ToolCall(call) => Some(call.id.as_str().to_owned()),
            _ => None,
        })
        .collect();
    assert_eq!(ids, vec!["call_abc".to_owned()]);
}

/// Regression test for issue #1926: a non-streaming `/api/chat` response that
/// carries `thinking` alongside `tool_calls` (the shape qwen3 thinking models
/// emit on a tool-call turn) must surface the reasoning as an
/// `AssistantContent::Reasoning` in `choice` — otherwise it never enters
/// agent history and is never echoed back to Ollama, degrading multi-turn
/// tool-call accuracy. Before the fix `choice` contained only the `ToolCall`.
#[tokio::test]
async fn nonstreaming_response_preserves_thinking_as_reasoning() {
    let sample_response = json!({
        "model": "qwen3:4b",
        "created_at": "2023-08-04T19:22:45.499127Z",
        "message": {
            "role": "assistant",
            "content": "",
            "thinking": "The user asked for the weather in Berlin. I should call get_weather with location=Berlin.",
            "images": null,
            "tool_calls": [
                { "type": "function", "function": { "name": "get_weather", "arguments": { "location": "Berlin" } } }
            ]
        },
        "done": true,
        "done_reason": "stop",
        "total_duration": 8000000000u64,
        "load_duration": 6000000u64,
        "prompt_eval_count": 61u64,
        "prompt_eval_duration": 400000000u64,
        "eval_count": 468u64,
        "eval_duration": 7700000000u64
    });

    let raw: CompletionResponse =
        serde_json::from_value(sample_response).expect("deserialize ollama response");
    let completed: completion::CompletionResponse =
        raw.try_into().expect("convert to completion response");

    let reasoning = completed.choice.iter().find_map(|c| match c {
        completion::AssistantContent::Reasoning(r) => Some(r.clone()),
        _ => None,
    });
    let has_tool_call = completed
        .choice
        .iter()
        .any(|c| matches!(c, completion::AssistantContent::ToolCall(_)));

    assert!(has_tool_call, "tool call should survive the conversion");
    let reasoning = reasoning.expect(
        "non-streaming response must surface `thinking` as AssistantContent::Reasoning (issue #1926)",
    );
    assert_eq!(
        reasoning.display_text(),
        "The user asked for the weather in Berlin. I should call get_weather with location=Berlin.",
    );
}

// Test empty thinking content is handled correctly
#[test]
fn test_empty_thinking_content() {
    let sample_response = json!({
        "model": "llama3.2",
        "created_at": "2023-08-04T19:22:45.499127Z",
        "message": {
            "role": "assistant",
            "content": "Response",
            "thinking": "",
            "images": null,
            "tool_calls": []
        },
        "done": true,
        "total_duration": 8000000000u64,
        "load_duration": 6000000u64,
        "prompt_eval_count": 10u64,
        "prompt_eval_duration": 400000000u64,
        "eval_count": 5u64,
        "eval_duration": 7700000000u64
    });

    let chat_resp: CompletionResponse =
        serde_json::from_value(sample_response).expect("Failed to deserialize");

    if let Message::Assistant {
        thinking, content, ..
    } = &chat_resp.message
    {
        // Empty string should still deserialize as Some("")
        assert_eq!(thinking.as_ref().unwrap(), "");
        assert_eq!(content, "Response");
    } else {
        panic!("Expected Assistant message");
    }
}

// Test thinking with tool calls
#[test]
fn test_thinking_with_tool_calls() {
    let sample_response = json!({
        "model": "qwen-thinking",
        "created_at": "2023-08-04T19:22:45.499127Z",
        "message": {
            "role": "assistant",
            "content": "Let me check the weather.",
            "thinking": "User wants weather info, I should use the weather tool",
            "images": null,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {
                            "location": "San Francisco"
                        }
                    }
                }
            ]
        },
        "done": true,
        "total_duration": 8000000000u64,
        "load_duration": 6000000u64,
        "prompt_eval_count": 30u64,
        "prompt_eval_duration": 400000000u64,
        "eval_count": 50u64,
        "eval_duration": 7700000000u64
    });

    let chat_resp: CompletionResponse =
        serde_json::from_value(sample_response).expect("Failed to deserialize");

    if let Message::Assistant {
        thinking,
        content,
        tool_calls,
        ..
    } = &chat_resp.message
    {
        assert_eq!(
            thinking.as_ref().unwrap(),
            "User wants weather info, I should use the weather tool"
        );
        assert_eq!(content, "Let me check the weather.");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].function.name, "get_weather");
    } else {
        panic!("Expected Assistant message with thinking and tool calls");
    }
}

// Test that `think` and `keep_alive` are extracted as top-level params, not in `options`
#[test]
fn test_completion_request_with_think_param() {
    use crate::completion::Message as CompletionMessage;
    use crate::message::{Text, UserContent};

    // Create a CompletionRequest with "think": true, "keep_alive", and "num_ctx" in additional_params
    let completion_request = CompletionRequest {
        model: None,
        chat_history: vec![
            CompletionMessage::system("You are a helpful assistant."),
            CompletionMessage::User {
                content: vec![UserContent::Text(Text::new("What is 2 + 2?".to_string()))],
            },
        ],
        documents: vec![],
        tools: vec![],
        temperature: Some(0.7),
        max_tokens: Some(1024),
        tool_choice: None,
        additional_params: Some(json!({
            "think": true,
            "keep_alive": "-1m",
            "num_ctx": 4096
        })),
        output_schema: None,
        record_telemetry_content: false,
    };

    // Convert to OllamaCompletionRequest
    let ollama_request = OllamaCompletionRequest::try_from(("qwen3:8b", completion_request))
        .expect("Failed to create Ollama request");

    // Serialize to JSON
    let serialized = serde_json::to_value(&ollama_request).expect("Failed to serialize request");

    // Assert equality with expected JSON
    // - "tools" is skipped when empty (skip_serializing_if)
    // - "think" should be a top-level boolean, NOT in options
    // - "keep_alive" should be a top-level string, NOT in options
    // - "num_ctx" should be in options (it's a model parameter)
    let expected = json!({
        "model": "qwen3:8b",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "What is 2 + 2?"
            }
        ],
        "stream": false,
        "think": true,
        "keep_alive": "-1m",
        "options": {
            "temperature": 0.7,
            "num_predict": 1024,
            "num_ctx": 4096
        }
    });

    assert_eq!(serialized, expected);
}

// Test that `think` and `keep_alive` are extracted as top-level params, not in `options`
#[test]
fn test_completion_request_with_level_low_think_param() {
    use crate::completion::Message as CompletionMessage;
    use crate::message::{Text, UserContent};

    // Create a CompletionRequest with "think": true, "keep_alive", and "num_ctx" in additional_params
    let completion_request = CompletionRequest {
        model: None,
        chat_history: vec![
            CompletionMessage::system("You are a helpful assistant."),
            CompletionMessage::User {
                content: vec![UserContent::Text(Text::new("What is 2 + 2?".to_string()))],
            },
        ],
        documents: vec![],
        tools: vec![],
        temperature: Some(0.7),
        max_tokens: Some(1024),
        tool_choice: None,
        additional_params: Some(json!({
            "think": "low",
            "keep_alive": "-1m",
            "num_ctx": 4096
        })),
        output_schema: None,
        record_telemetry_content: false,
    };

    // Convert to OllamaCompletionRequest
    let ollama_request = OllamaCompletionRequest::try_from(("qwen3:8b", completion_request))
        .expect("Failed to create Ollama request");

    // Serialize to JSON
    let serialized = serde_json::to_value(&ollama_request).expect("Failed to serialize request");

    // Assert equality with expected JSON
    // - "tools" is skipped when empty (skip_serializing_if)
    // - "think" should be a top-level boolean, NOT in options
    // - "keep_alive" should be a top-level string, NOT in options
    // - "num_ctx" should be in options (it's a model parameter)
    let expected = json!({
        "model": "qwen3:8b",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "What is 2 + 2?"
            }
        ],
        "stream": false,
        "think": "low",
        "keep_alive": "-1m",
        "options": {
            "temperature": 0.7,
            "num_predict": 1024,
            "num_ctx": 4096
        }
    });

    assert_eq!(serialized, expected);
}

// Test that `think` and `keep_alive` are extracted as top-level params, not in `options`
#[test]
fn test_completion_request_with_level_medium_think_param() {
    use crate::completion::Message as CompletionMessage;
    use crate::message::{Text, UserContent};

    // Create a CompletionRequest with "think": true, "keep_alive", and "num_ctx" in additional_params
    let completion_request = CompletionRequest {
        model: None,
        chat_history: vec![
            CompletionMessage::system("You are a helpful assistant."),
            CompletionMessage::User {
                content: vec![UserContent::Text(Text::new("What is 2 + 2?".to_string()))],
            },
        ],
        documents: vec![],
        tools: vec![],
        temperature: Some(0.7),
        max_tokens: Some(1024),
        tool_choice: None,
        additional_params: Some(json!({
            "think": "medium",
            "keep_alive": "-1m",
            "num_ctx": 4096
        })),
        output_schema: None,
        record_telemetry_content: false,
    };

    // Convert to OllamaCompletionRequest
    let ollama_request = OllamaCompletionRequest::try_from(("qwen3:8b", completion_request))
        .expect("Failed to create Ollama request");

    // Serialize to JSON
    let serialized = serde_json::to_value(&ollama_request).expect("Failed to serialize request");

    // Assert equality with expected JSON
    // - "tools" is skipped when empty (skip_serializing_if)
    // - "think" should be a top-level boolean, NOT in options
    // - "keep_alive" should be a top-level string, NOT in options
    // - "num_ctx" should be in options (it's a model parameter)
    let expected = json!({
        "model": "qwen3:8b",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "What is 2 + 2?"
            }
        ],
        "stream": false,
        "think": "medium",
        "keep_alive": "-1m",
        "options": {
            "temperature": 0.7,
            "num_predict": 1024,
            "num_ctx": 4096
        }
    });

    assert_eq!(serialized, expected);
}

// Test that `think` and `keep_alive` are extracted as top-level params, not in `options`
#[test]
fn test_completion_request_with_level_high_think_param() {
    use crate::completion::Message as CompletionMessage;
    use crate::message::{Text, UserContent};

    // Create a CompletionRequest with "think": true, "keep_alive", and "num_ctx" in additional_params
    let completion_request = CompletionRequest {
        model: None,
        chat_history: vec![
            CompletionMessage::system("You are a helpful assistant."),
            CompletionMessage::User {
                content: vec![UserContent::Text(Text::new("What is 2 + 2?".to_string()))],
            },
        ],
        documents: vec![],
        tools: vec![],
        temperature: Some(0.7),
        max_tokens: Some(1024),
        tool_choice: None,
        additional_params: Some(json!({
            "think": "high",
            "keep_alive": "-1m",
            "num_ctx": 4096
        })),
        output_schema: None,
        record_telemetry_content: false,
    };

    // Convert to OllamaCompletionRequest
    let ollama_request = OllamaCompletionRequest::try_from(("qwen3:8b", completion_request))
        .expect("Failed to create Ollama request");

    // Serialize to JSON
    let serialized = serde_json::to_value(&ollama_request).expect("Failed to serialize request");

    // Assert equality with expected JSON
    // - "tools" is skipped when empty (skip_serializing_if)
    // - "think" should be a top-level boolean, NOT in options
    // - "keep_alive" should be a top-level string, NOT in options
    // - "num_ctx" should be in options (it's a model parameter)
    let expected = json!({
        "model": "qwen3:8b",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "What is 2 + 2?"
            }
        ],
        "stream": false,
        "think": "high",
        "keep_alive": "-1m",
        "options": {
            "temperature": 0.7,
            "num_predict": 1024,
            "num_ctx": 4096
        }
    });

    assert_eq!(serialized, expected);
}

// Test that `think` and `keep_alive` are extracted as top-level params, not in `options`
#[test]
fn test_completion_request_with_level_invalid_think_param() {
    use crate::completion::Message as CompletionMessage;
    use crate::message::{Text, UserContent};

    // Create a CompletionRequest with "think": true, "keep_alive", and "num_ctx" in additional_params
    let completion_request = CompletionRequest {
        model: None,
        chat_history: vec![
            CompletionMessage::system("You are a helpful assistant."),
            CompletionMessage::User {
                content: vec![UserContent::Text(Text::new("What is 2 + 2?".to_string()))],
            },
        ],
        documents: vec![],
        tools: vec![],
        temperature: Some(0.7),
        max_tokens: Some(1024),
        tool_choice: None,
        additional_params: Some(json!({
            "think": "invalid",
            "keep_alive": "-1m",
            "num_ctx": 4096
        })),
        output_schema: None,
        record_telemetry_content: false,
    };

    // Convert to OllamaCompletionRequest
    let ollama_request = OllamaCompletionRequest::try_from(("qwen3:8b", completion_request));

    assert!(ollama_request.is_err());
}

// Test that `think` is omitted when not specified, so Ollama applies the
// model's default thinking behavior (issue #1970)
#[test]
fn test_completion_request_with_think_omitted_by_default() {
    use crate::completion::Message as CompletionMessage;
    use crate::message::{Text, UserContent};

    // Create a CompletionRequest WITHOUT "think" in additional_params
    let completion_request = CompletionRequest {
        model: None,
        chat_history: vec![
            CompletionMessage::system("You are a helpful assistant."),
            CompletionMessage::User {
                content: vec![UserContent::Text(Text::new("Hello!".to_string()))],
            },
        ],
        documents: vec![],
        tools: vec![],
        temperature: Some(0.5),
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    };

    // Convert to OllamaCompletionRequest
    let ollama_request = OllamaCompletionRequest::try_from(("llama3.2", completion_request))
        .expect("Failed to create Ollama request");

    // Serialize to JSON
    let serialized = serde_json::to_value(&ollama_request).expect("Failed to serialize request");

    // Assert that "think" is absent (so Ollama uses the model default) and
    // "keep_alive" is not present
    let expected = json!({
        "model": "llama3.2",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Hello!"
            }
        ],
        "stream": false,
        "options": {
            "temperature": 0.5
        }
    });

    assert_eq!(serialized, expected);
}

// The native API takes the token limit as `options.num_predict`; an
// explicit `num_predict` in `additional_params` wins over
// `CompletionRequest::max_tokens`.
#[test]
fn test_completion_request_num_predict_from_additional_params_wins() {
    use crate::completion::Message as CompletionMessage;
    use crate::message::{Text, UserContent};

    let completion_request = CompletionRequest {
        model: None,
        chat_history: vec![CompletionMessage::User {
            content: vec![UserContent::Text(Text::new("Hello!".to_string()))],
        }],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: Some(1024),
        tool_choice: None,
        additional_params: Some(json!({ "num_predict": 42 })),
        output_schema: None,
        record_telemetry_content: false,
    };

    let ollama_request = OllamaCompletionRequest::try_from(("llama3.2", completion_request))
        .expect("Failed to create Ollama request");
    let serialized = serde_json::to_value(&ollama_request).expect("Failed to serialize request");

    assert_eq!(serialized["options"], json!({ "num_predict": 42 }));
    assert_eq!(serialized.get("max_tokens"), None);
}

// The plain path: `max_tokens` with no `additional_params` at all, which
// skips the merge and serializes `base_options` directly. Every other
// `max_tokens` test also sets `additional_params`, so without this one the
// branch the fix exists for is never exercised.
#[test]
fn test_completion_request_num_predict_without_additional_params() {
    use crate::completion::Message as CompletionMessage;
    use crate::message::{Text, UserContent};

    let completion_request = CompletionRequest {
        model: None,
        chat_history: vec![CompletionMessage::User {
            content: vec![UserContent::Text(Text::new("Hello!".to_string()))],
        }],
        documents: vec![],
        tools: vec![],
        temperature: Some(0.7),
        max_tokens: Some(1024),
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    };

    let ollama_request = OllamaCompletionRequest::try_from(("llama3.2", completion_request))
        .expect("Failed to create Ollama request");
    let serialized = serde_json::to_value(&ollama_request).expect("Failed to serialize request");

    assert_eq!(
        serialized["options"],
        json!({ "temperature": 0.7, "num_predict": 1024 })
    );
    // Neither belongs at the top level of a native `/api/chat` payload.
    assert_eq!(serialized.get("max_tokens"), None);
    assert_eq!(serialized.get("temperature"), None);
}

// With nothing to put in it, `options` is an empty object rather than
// carrying `"temperature": null` as it did when temperature was seeded
// unconditionally.
#[test]
fn test_completion_request_options_omit_unset_parameters() {
    use crate::completion::Message as CompletionMessage;
    use crate::message::{Text, UserContent};

    let completion_request = CompletionRequest {
        model: None,
        chat_history: vec![CompletionMessage::User {
            content: vec![UserContent::Text(Text::new("Hello!".to_string()))],
        }],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    };

    let ollama_request = OllamaCompletionRequest::try_from(("llama3.2", completion_request))
        .expect("Failed to create Ollama request");
    let serialized = serde_json::to_value(&ollama_request).expect("Failed to serialize request");

    assert_eq!(serialized["options"], json!({}));
}

#[test]
fn test_completion_request_with_output_schema() {
    use crate::completion::Message as CompletionMessage;
    use crate::message::{Text, UserContent};

    let schema: schemars::Schema = serde_json::from_value(json!({
        "type": "object",
        "properties": {
            "age": { "type": "integer" },
            "available": { "type": "boolean" }
        },
        "required": ["age", "available"]
    }))
    .expect("Failed to parse schema");

    let completion_request = CompletionRequest {
        model: Some("llama3.1".to_string()),
        chat_history: vec![CompletionMessage::User {
            content: vec![UserContent::Text(Text::new(
                "How old is Ollama?".to_string(),
            ))],
        }],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: Some(schema),
        record_telemetry_content: false,
    };

    let ollama_request = OllamaCompletionRequest::try_from(("llama3.1", completion_request))
        .expect("Failed to create Ollama request");

    let serialized = serde_json::to_value(&ollama_request).expect("Failed to serialize request");

    let format = serialized
        .get("format")
        .expect("format field should be present");
    assert_eq!(
        *format,
        json!({
            "type": "object",
            "properties": {
                "age": { "type": "integer" },
                "available": { "type": "boolean" }
            },
            "required": ["age", "available"]
        })
    );
}

#[test]
fn test_completion_request_without_output_schema() {
    use crate::completion::Message as CompletionMessage;
    use crate::message::{Text, UserContent};

    let completion_request = CompletionRequest {
        model: Some("llama3.1".to_string()),
        chat_history: vec![CompletionMessage::User {
            content: vec![UserContent::Text(Text::new("Hello!".to_string()))],
        }],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    };

    let ollama_request = OllamaCompletionRequest::try_from(("llama3.1", completion_request))
        .expect("Failed to create Ollama request");

    let serialized = serde_json::to_value(&ollama_request).expect("Failed to serialize request");

    assert!(
        serialized.get("format").is_none(),
        "format field should be absent when output_schema is None"
    );
}

#[test]
fn test_client_initialization() {
    let _client = crate::providers::ollama::Client::new_with(
        Nothing,
        crate::test_utils::RecordingHttpClient::new(""),
    )
    .expect("Client::new() failed");
    let _client_from_builder = crate::providers::ollama::Client::builder()
        .api_key(Nothing)
        .http_client(crate::test_utils::RecordingHttpClient::new(""))
        .build()
        .expect("Client::builder() failed");
}

#[test]
fn ndjson_buffer_returns_complete_lines_in_single_chunk() {
    let mut buf = NdjsonBuffer::new();
    let lines = buf.decode(b"{\"a\":1}\n{\"b\":2}\n");
    assert_eq!(lines, vec![b"{\"a\":1}".to_vec(), b"{\"b\":2}".to_vec()]);
}

#[test]
fn ndjson_buffer_reassembles_line_split_across_chunks() {
    let mut buf = NdjsonBuffer::new();

    assert!(buf.decode(b"{\"model\":\"llama\",\"mes").is_empty());

    let lines = buf.decode(b"sage\":\"hi\"}\n{\"done\"");
    assert_eq!(
        lines,
        vec![b"{\"model\":\"llama\",\"message\":\"hi\"}".to_vec()]
    );

    let lines = buf.decode(b":true}\n");
    assert_eq!(lines, vec![b"{\"done\":true}".to_vec()]);
}

#[test]
fn ndjson_buffer_skips_blank_lines() {
    let mut buf = NdjsonBuffer::new();
    let lines = buf.decode(b"\n{\"a\":1}\n\n");
    assert_eq!(lines, vec![b"{\"a\":1}".to_vec()]);
}

#[test]
fn ndjson_buffer_retains_unterminated_trailing_data() {
    let mut buf = NdjsonBuffer::new();
    let lines = buf.decode(b"{\"a\":1}\n{\"b\":2");
    assert_eq!(lines, vec![b"{\"a\":1}".to_vec()]);
    let lines = buf.decode(b"}\n");
    assert_eq!(lines, vec![b"{\"b\":2}".to_vec()]);
}

#[test]
fn ndjson_buffer_handles_empty_chunk() {
    let mut buf = NdjsonBuffer::new();
    assert!(buf.decode(b"").is_empty());

    buf.decode(b"{\"a\":1");
    assert!(buf.decode(b"").is_empty());

    let lines = buf.decode(b"}\n");
    assert_eq!(lines, vec![b"{\"a\":1}".to_vec()]);
}

#[test]
fn ndjson_buffer_handles_multi_byte_utf8_split_across_chunks() {
    // `\n` (0x0A) cannot appear inside any UTF-8 continuation byte, so a
    // byte-wise newline scan is always safe — but verify explicitly that a
    // multi-byte sequence reassembles correctly when split across chunks.
    let mut buf = NdjsonBuffer::new();
    assert!(buf.decode(&[0xd0]).is_empty());
    assert!(buf.decode(&[0xb8, 0xd0, 0xb7, 0xd0]).is_empty());
    assert!(
        buf.decode(&[
            0xb2, 0xd0, 0xb5, 0xd1, 0x81, 0xd1, 0x82, 0xd0, 0xbd, 0xd0, 0xb8
        ])
        .is_empty()
    );

    let lines = buf.decode(b"\n");
    assert_eq!(lines.len(), 1);
    assert_eq!(std::str::from_utf8(&lines[0]).unwrap(), "известни");
}

#[test]
fn ndjson_buffer_yields_parseable_chunks_when_split_arbitrarily() {
    let original = concat!(
        "{\"model\":\"llama3.2\",\"message\":{\"role\":\"assistant\",\"content\":\"hi\"},\"done\":false}\n",
        "{\"model\":\"llama3.2\",\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true}\n",
    );

    let mut buf = NdjsonBuffer::new();
    let mut received = Vec::new();
    for byte in original.as_bytes() {
        for line in buf.decode(std::slice::from_ref(byte)) {
            let parsed: serde_json::Value =
                serde_json::from_slice(&line).expect("each drained line must be valid JSON");
            received.push(parsed);
        }
    }

    assert_eq!(received.len(), 2);
    assert_eq!(received[0]["message"]["content"], "hi");
    assert_eq!(received[1]["done"], true);
}

// Proves a truncated NDJSON stream — content chunks then EOF without a
// `done: true` record — delivers its content but never a synthesized
// terminal record.
#[tokio::test]
async fn truncated_stream_does_not_synthesize_a_terminal_record() {
    use crate::client::CompletionClient;
    use crate::completion::CompletionModel;
    use crate::streaming::StreamedAssistantContent;
    use crate::test_utils::MockStreamingClient;
    use futures::StreamExt;

    let ndjson = concat!(
        r#"{"model":"llama3.2","created_at":"2023-08-04T19:22:45.499127Z","message":{"role":"assistant","content":"hi"},"done":false}"#,
        "\n",
    );
    let client = Client::builder()
        .api_key("test-key")
        .http_client(MockStreamingClient {
            sse_bytes: bytes::Bytes::from(ndjson),
        })
        .build()
        .expect("build client");
    let model = client.completion_model(LLAMA3_2);
    let request = model.completion_request("hello").build();

    let mut stream = model.stream(request).await.expect("stream should open");

    let mut texts = Vec::new();
    let mut saw_terminal = false;
    while let Some(item) = stream.next().await {
        match item.expect("stream item should be Ok") {
            StreamedAssistantContent::Text(text) => texts.push(text.text),
            StreamedAssistantContent::Final(_) => saw_terminal = true,
            _ => {}
        }
    }

    assert_eq!(texts, ["hi"]);
    assert!(
        !saw_terminal,
        "EOF without a done record must not synthesize a terminal record"
    );
    assert!(stream.response.is_none());
}

// Proves a malformed NDJSON line between valid lines surfaces as an
// `Err` item while the stream keeps consuming: the following content and
// the `done: true` record still arrive.
#[tokio::test]
async fn malformed_line_is_surfaced_and_the_terminal_still_arrives() {
    use crate::client::CompletionClient;
    use crate::completion::CompletionModel;
    use crate::streaming::StreamedAssistantContent;
    use crate::test_utils::MockStreamingClient;
    use futures::StreamExt;

    let ndjson = concat!(
        r#"{"model":"llama3.2","created_at":"2023-08-04T19:22:45.499127Z","message":{"role":"assistant","content":"hi"},"done":false}"#,
        "\n",
        "{not json\n",
        r#"{"model":"llama3.2","created_at":"2023-08-04T19:22:46.499127Z","message":{"role":"assistant","content":" there"},"done":false}"#,
        "\n",
        r#"{"model":"llama3.2","created_at":"2023-08-04T19:22:47.499127Z","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":10,"eval_count":4}"#,
        "\n",
    );
    let client = Client::builder()
        .api_key("test-key")
        .http_client(MockStreamingClient {
            sse_bytes: bytes::Bytes::from(ndjson),
        })
        .build()
        .expect("build client");
    let model = client.completion_model(LLAMA3_2);
    let request = model.completion_request("hello").build();

    let mut stream = model.stream(request).await.expect("stream should open");

    let mut texts = Vec::new();
    let mut saw_error = false;
    let mut terminal = None;
    while let Some(item) = stream.next().await {
        match item {
            Ok(StreamedAssistantContent::Text(text)) => texts.push(text.text),
            Ok(StreamedAssistantContent::Final(final_response)) => {
                terminal = Some(final_response);
            }
            Ok(_) => {}
            Err(_) => saw_error = true,
        }
    }

    assert_eq!(texts, ["hi", " there"]);
    assert!(saw_error, "the malformed line must reach the consumer");
    let terminal = terminal.expect("the genuine done record must still arrive");
    assert_eq!(terminal.usage.input_tokens, 10);
    assert_eq!(terminal.usage.output_tokens, 4);
}

// Proves the `done: true` record ends the stream: a content line that
// arrives after it is never yielded — only the pre-done content and the
// terminal record reach the consumer.
#[tokio::test]
async fn content_after_the_done_record_is_not_yielded() {
    use crate::client::CompletionClient;
    use crate::completion::CompletionModel;
    use crate::streaming::StreamedAssistantContent;
    use crate::test_utils::MockStreamingClient;
    use futures::StreamExt;

    let ndjson = concat!(
        r#"{"model":"llama3.2","created_at":"2023-08-04T19:22:45.499127Z","message":{"role":"assistant","content":"hi"},"done":false}"#,
        "\n",
        r#"{"model":"llama3.2","created_at":"2023-08-04T19:22:46.499127Z","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":10,"eval_count":4}"#,
        "\n",
        r#"{"model":"llama3.2","created_at":"2023-08-04T19:22:47.499127Z","message":{"role":"assistant","content":"stray"},"done":false}"#,
        "\n",
    );
    let client = Client::builder()
        .api_key("test-key")
        .http_client(MockStreamingClient {
            sse_bytes: bytes::Bytes::from(ndjson),
        })
        .build()
        .expect("build client");
    let model = client.completion_model(LLAMA3_2);
    let request = model.completion_request("hello").build();

    let mut stream = model.stream(request).await.expect("stream should open");

    let mut texts = Vec::new();
    let mut terminal = None;
    while let Some(item) = stream.next().await {
        match item.expect("stream item should be Ok") {
            StreamedAssistantContent::Text(text) => texts.push(text.text),
            StreamedAssistantContent::Final(final_response) => {
                assert!(
                    terminal.is_none(),
                    "the terminal record must be yielded exactly once"
                );
                terminal = Some(final_response);
            }
            other => panic!("unexpected stream item: {other:?}"),
        }
    }

    assert_eq!(
        texts,
        ["hi"],
        "content after the done record must not be yielded"
    );
    let terminal = terminal.expect("the done record must yield the terminal record");
    assert_eq!(terminal.usage.input_tokens, 10);
    assert_eq!(terminal.usage.output_tokens, 4);
}

// Proves a non-success HTTP response from `/api/chat` preserves the
// provider's status + body through the `provider_response_*` helpers
// (issue #1931).
#[tokio::test]
async fn completion_non_success_preserves_status_and_body() {
    use crate::client::CompletionClient;
    use crate::completion::CompletionModel;
    use crate::test_utils::RecordingHttpClient;

    let body = r#"{"error":"model not found"}"#;
    let http_client =
        RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
    let client = Client::builder()
        .api_key("test-key")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.completion_model(LLAMA3_2);
    let request = model.completion_request("hello").build();

    let error = model
        .completion(request)
        .await
        .expect_err("should fail with non-success status");

    assert!(matches!(error, CompletionError::HttpError(_)));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::SERVICE_UNAVAILABLE)
    );
    assert_eq!(error.provider_response_body(), Some(body));
}

// Proves a non-success HTTP response from `/api/embed` preserves the
// provider's status + body through the `provider_response_*` helpers
// (issue #1931).
#[tokio::test]
async fn embeddings_non_success_preserves_status_and_body() {
    use crate::client::EmbeddingsClient;
    use crate::embeddings::EmbeddingModel;
    use crate::test_utils::RecordingHttpClient;

    let body = r#"{"error":"model not found"}"#;
    let http_client =
        RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
    let client = Client::builder()
        .api_key("test-key")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.embedding_model(ALL_MINILM);

    let error = model
        .embed_texts(vec!["hello".to_string()])
        .await
        .expect_err("should fail with non-success status");

    assert!(matches!(error, EmbeddingError::HttpError(_)));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::SERVICE_UNAVAILABLE)
    );
    assert_eq!(error.provider_response_body(), Some(body));
}

/// Raw-capture tests: the `TryFrom` shape, driven end to end through
/// `CompletionModel::completion` over the recording mock transport. Ollama
/// has no request-id contract, so there is nothing transport-side to
/// reattach; the capture is the `/api/chat` body exactly as `raw_completion`
/// parses it. The body carries the timing fields (`total_duration`,
/// `eval_duration`, ...) rig never normalizes, so the capture can be shown
/// to answer more than the normalized response does.
mod raw_capture {
    use super::*;
    use crate::client::CompletionClient;
    use crate::completion::CompletionModel as _;
    use crate::test_utils::RecordingHttpClient;

    const BODY: &str = r#"{
            "model": "llama3.2",
            "created_at": "2023-08-04T19:22:45.499127Z",
            "message": {"role": "assistant", "content": "hello"},
            "done": true,
            "done_reason": "stop",
            "total_duration": 5043500667,
            "load_duration": 5025959,
            "prompt_eval_count": 26,
            "prompt_eval_duration": 325953000,
            "eval_count": 5,
            "eval_duration": 4709213000
        }"#;

    fn model() -> CompletionModel<RecordingHttpClient> {
        let client = Client::builder()
            .api_key("test-key")
            .http_client(RecordingHttpClient::new(BODY))
            .build()
            .expect("build client");
        client.completion_model(LLAMA3_2)
    }

    /// The load-bearing capture property: `raw` is Ollama's
    /// `CompletionResponse` as rig parsed it — it deserializes back into
    /// that type and re-serializes to the identical value — and
    /// re-normalizing that capture through the same `TryFrom` reproduces
    /// every normalized field. Also reads `total_duration` and
    /// `eval_duration` off the capture, which the normalized response
    /// provably lacks.
    #[tokio::test]
    async fn completion_captures_raw_that_round_trips_into_the_wire_type() {
        let model = model();

        let response = model
            .completion(model.completion_request("hello").build())
            .await
            .expect("completion");

        let raw = &response.raw;
        let typed: CompletionResponse =
            serde_json::from_value(raw.clone()).expect("raw must deserialize");
        assert_eq!(
            serde_json::to_value(&typed).expect("re-serialize"),
            *raw,
            "the capture must be exactly what the wire type serializes to"
        );
        assert_eq!(typed.total_duration, Some(5_043_500_667));
        assert_eq!(typed.eval_duration, Some(4_709_213_000));
        assert_eq!(raw["total_duration"], 5_043_500_667_u64);
        assert_eq!(typed.done_reason.as_deref(), Some("stop"));

        let renormalized: completion::CompletionResponse =
            typed.try_into().expect("re-normalize the capture");
        assert_eq!(response.identity(), renormalized.identity());
        assert_eq!(response.finish_reason(), renormalized.finish_reason());
        assert_eq!(response.model, renormalized.model);
        assert_eq!(response.usage, renormalized.usage);
        assert_eq!(response.choice, renormalized.choice);
        assert_eq!(
            response.finish_reason(),
            Some(completion::FinishReason::Stop)
        );
        assert_eq!(response.model.as_deref(), Some("llama3.2"));
        assert_eq!(response.usage.total_tokens, 31);
    }
}
