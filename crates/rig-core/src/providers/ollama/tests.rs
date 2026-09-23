use super::*;
use crate::error::ProviderError;
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

/// Fold one `/api/chat` reply body through the bound chat wire, the way a
/// caller's `completion()` does.
async fn unary(body: serde_json::Value) -> Result<completion::CompletionResponse, ProviderError> {
    use crate::completion::CompletionModel as _;
    let model = ollama_model(crate::test_utils::RecordingHttpClient::new(
        body.to_string(),
    ));
    model
        .completion(model.completion_request("hello").build())
        .await
}

// A non-streaming `/api/chat` reply carrying both text and a tool call
// (shape from the Ollama docs) folds to a choice holding both.
#[tokio::test]
async fn test_chat_completion() {
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

    let conv = unary(sample_chat_response)
        .await
        .expect("the reply decodes");
    assert!(
        conv.choice
            .iter()
            .any(|c| matches!(c, completion::AssistantContent::Text(t) if t.text == "The sky is blue because of Rayleigh scattering.")),
        "the text survives: {:?}",
        conv.choice
    );
    assert!(
        conv.choice.iter().any(|c| matches!(
            c,
            completion::AssistantContent::ToolCall(call) if call.function.name == "get_current_weather"
        )),
        "the tool call survives: {:?}",
        conv.choice
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

#[tokio::test]
async fn response_metadata_is_normalized() {
    let normalized = unary(json!({
        "model": "llama3.2",
        "created_at": "2023-08-04T19:22:45.499127Z",
        "message": {"role": "assistant", "content": "Hi!", "tool_calls": []},
        "done": true,
        "done_reason": "length",
        "prompt_eval_count": 12u64,
        "eval_count": 3u64
    }))
    .await
    .expect("normalization should succeed");

    assert_eq!(normalized.provider, PROVIDER_NAME);
    assert_eq!(normalized.model.as_deref(), Some("llama3.2"));
    assert_eq!(
        normalized.finish_reason(),
        Some(completion::FinishReason::Length)
    );
    // Ollama assigns no message identifier.
    assert_eq!(normalized.message_id, None);
    assert_eq!(normalized.usage.input_tokens, Some(12));
    assert_eq!(normalized.usage.output_tokens, Some(3));
    assert_eq!(normalized.usage.total_tokens, Some(15));
}

// A `done_reason` of `stop` on a turn that actually called a tool must be
// upgraded by the response builder's reconciliation.
#[tokio::test]
async fn tool_call_turn_upgrades_a_plain_stop_to_tool_calls() {
    let normalized = unary(json!({
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
    .await
    .expect("normalization should succeed");

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

    let raw = serde_json::to_value(&terminal).expect("serialize terminal");
    let final_record = stream_final(terminal, raw.clone());
    assert_eq!(final_record.raw, raw);
    assert_eq!(final_record.provider, PROVIDER_NAME);
    assert_eq!(final_record.model.as_deref(), Some("llama3.2"));
    assert_eq!(
        final_record.finish_reason,
        Some(completion::FinishReason::Other("dragons".to_owned()))
    );
    assert_eq!(final_record.usage.total_tokens, Some(12));
}

#[test]
fn mixed_user_content_preserves_message_order() {
    use crate::message::{Message as RigMessage, ToolResultContent, UserContent};

    let message = RigMessage::User {
        content: vec![
            UserContent::text("before"),
            UserContent::tool_result(
                "call-not-the-tool-name",
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
        Message::ToolResult { name, content, .. }
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

    let completed = unary(sample_response)
        .await
        .expect("convert to completion response");

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

/// The chat wire bound to `http_client`: the model every case below drives.
fn ollama_model<H: Clone>(http_client: H) -> crate::driver::Bound<Chat, H> {
    crate::driver::Bound::new(Ollama::new(), http_client).completion(LLAMA3_2)
}

// Proves a truncated NDJSON stream — content chunks then EOF without a
// `done: true` record — delivers its content but never a synthesized
// terminal record.
#[tokio::test]
async fn truncated_stream_does_not_synthesize_a_terminal_record() {
    use crate::completion::CompletionModel;
    use crate::streaming::{Delta, StreamEvent};
    use crate::test_utils::MockStreamingClient;
    use futures::StreamExt;

    let ndjson = concat!(
        r#"{"model":"llama3.2","created_at":"2023-08-04T19:22:45.499127Z","message":{"role":"assistant","content":"hi"},"done":false}"#,
        "\n",
    );
    let model = ollama_model(MockStreamingClient {
        sse_bytes: bytes::Bytes::from(ndjson),
    });
    let request = model.completion_request("hello").build();

    let mut stream = model.stream(request).await.expect("stream should open");

    let mut texts = Vec::new();
    let mut saw_terminal = false;
    while let Some(item) = stream.next().await {
        match item.expect("stream item should be Ok") {
            StreamEvent::BlockDelta {
                delta: Delta::Text { text },
                ..
            } => texts.push(text),
            StreamEvent::Final(_) => saw_terminal = true,
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
    use crate::completion::CompletionModel;
    use crate::streaming::{Delta, StreamEvent};
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
    let model = ollama_model(MockStreamingClient {
        sse_bytes: bytes::Bytes::from(ndjson),
    });
    let request = model.completion_request("hello").build();

    let mut stream = model.stream(request).await.expect("stream should open");

    let mut texts = Vec::new();
    let mut saw_error = false;
    let mut terminal = None;
    while let Some(item) = stream.next().await {
        match item {
            Ok(StreamEvent::BlockDelta {
                delta: Delta::Text { text },
                ..
            }) => texts.push(text),
            Ok(StreamEvent::Final(final_response)) => {
                terminal = Some(final_response);
            }
            Ok(_) => {}
            Err(_) => saw_error = true,
        }
    }

    assert_eq!(texts, ["hi", " there"]);
    assert!(saw_error, "the malformed line must reach the consumer");
    let terminal = terminal.expect("the genuine done record must still arrive");
    assert_eq!(terminal.usage.input_tokens, Some(10));
    assert_eq!(terminal.usage.output_tokens, Some(4));
}

// Proves the `done: true` record ends the stream: a content line that
// arrives after it is never yielded — only the pre-done content and the
// terminal record reach the consumer.
#[tokio::test]
async fn content_after_the_done_record_is_not_yielded() {
    use crate::completion::CompletionModel;
    use crate::streaming::{Delta, StreamEvent};
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
    let model = ollama_model(MockStreamingClient {
        sse_bytes: bytes::Bytes::from(ndjson),
    });
    let request = model.completion_request("hello").build();

    let mut stream = model.stream(request).await.expect("stream should open");

    let mut texts = Vec::new();
    let mut terminal = None;
    while let Some(item) = stream.next().await {
        match item.expect("stream item should be Ok") {
            StreamEvent::BlockDelta {
                delta: Delta::Text { text },
                ..
            } => texts.push(text),
            StreamEvent::Final(final_response) => {
                assert!(
                    terminal.is_none(),
                    "the terminal record must be yielded exactly once"
                );
                terminal = Some(final_response);
            }
            // The text block's minted start/end bracket the deltas.
            StreamEvent::BlockStart { .. } | StreamEvent::BlockEnd { .. } => {}
            other => panic!("unexpected stream item: {other:?}"),
        }
    }

    assert_eq!(
        texts,
        ["hi"],
        "content after the done record must not be yielded"
    );
    let terminal = terminal.expect("the done record must yield the terminal record");
    assert_eq!(terminal.usage.input_tokens, Some(10));
    assert_eq!(terminal.usage.output_tokens, Some(4));
}

// Proves a non-success HTTP response from `/api/chat` preserves the
// provider's status + body through the `provider_response_*` helpers
// (issue #1931).
#[tokio::test]
async fn completion_non_success_preserves_status_and_body() {
    use crate::completion::CompletionModel;
    use crate::test_utils::RecordingHttpClient;

    let body = r#"{"error":"model not found"}"#;
    let http_client =
        RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
    let model = ollama_model(http_client);
    let request = model.completion_request("hello").build();

    let error = model
        .completion(request)
        .await
        .expect_err("should fail with non-success status");

    assert!(matches!(error, ProviderError::ProviderResponse(_)));
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
    use crate::embeddings::EmbeddingModel;
    use crate::test_utils::RecordingHttpClient;

    let body = r#"{"error":"model not found"}"#;
    let http_client =
        RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
    let model = crate::driver::Bound::new(Ollama::new(), http_client).embedding(ALL_MINILM, None);

    let error = model
        .embed_texts(vec!["hello".to_string()])
        .await
        .expect_err("should fail with non-success status");

    assert!(matches!(error, ProviderError::ProviderResponse(_)));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::SERVICE_UNAVAILABLE)
    );
    assert_eq!(error.provider_response_body(), Some(body));
}

/// Raw-capture tests: the `/api/chat` reply driven end to end through
/// `CompletionModel::completion` on the bound chat wire over the recording
/// mock transport. Ollama has no request-id contract, so there is nothing
/// transport-side to reattach; `CompletionResponse::raw` is the `/api/chat`
/// body verbatim. The body carries the timing fields (`total_duration`,
/// `eval_duration`, ...) rig never normalizes, so the capture can be shown
/// to answer more than the normalized response does.
mod raw_capture {
    use super::*;
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

    fn model() -> crate::driver::Bound<Chat, RecordingHttpClient> {
        ollama_model(RecordingHttpClient::new(BODY))
    }

    /// The load-bearing capture property: `raw` is the `/api/chat` reply
    /// **verbatim** — the body's own document, not a re-serialization of
    /// the typed parse — it still deserializes back into Ollama's
    /// `CompletionResponse` with every field the body carried intact, and
    /// folding that capture back through the same wire reproduces every
    /// normalized field. Also reads `total_duration` and `eval_duration`
    /// off the capture, which the normalized response provably lacks.
    ///
    /// Compared against the body rather than against
    /// `to_value(&typed)`: that equality was the *old* contract, where
    /// `raw` was precisely that serialization, so it asserted nothing —
    /// and it is false of a verbatim capture the moment the DTO spells
    /// out a default the body omitted. Ollama's assistant message has
    /// exactly one such field, `tool_calls`, which is the lone member of
    /// that variant without `skip_serializing_if` because the recorded
    /// request bodies carry it. So the re-serialization is compared to
    /// the body with that one default filled in: any *other* drift — a
    /// dropped timing field, a reshaped message — still fails.
    #[tokio::test]
    async fn completion_captures_raw_that_round_trips_into_the_wire_type() {
        let model = model();

        let response = model
            .completion(model.completion_request("hello").build())
            .await
            .expect("completion");

        let raw = &response.raw;
        assert_eq!(
            *raw,
            serde_json::from_str::<serde_json::Value>(BODY).expect("the recorded body is JSON"),
            "the capture is the reply document verbatim"
        );
        let typed: CompletionResponse =
            serde_json::from_value(raw.clone()).expect("raw must deserialize");
        let reserialized = serde_json::to_value(&typed).expect("re-serialize");
        let mut with_defaults = raw.clone();
        with_defaults["message"]["tool_calls"] = serde_json::json!([]);
        assert_eq!(
            reserialized, with_defaults,
            "the wire type round-trips the whole document, up to the empty \
             `tool_calls` the body omitted and the DTO always writes"
        );
        assert_eq!(typed.total_duration, Some(5_043_500_667));
        assert_eq!(typed.eval_duration, Some(4_709_213_000));
        assert_eq!(raw["total_duration"], 5_043_500_667_u64);
        assert_eq!(typed.done_reason.as_deref(), Some("stop"));

        let renormalized = unary(raw.clone()).await.expect("re-fold the capture");
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
        assert_eq!(response.usage.total_tokens, Some(31));
    }
}

/// Synthetic wire values test absent-ID and explicit-ID collisions deterministically;
/// recordings cannot reliably force a provider to emit these boundary combinations.
#[tokio::test]
async fn missing_tool_ids_are_distinct_stable_and_collision_free_in_responses() {
    let wire = json!({
        "model": "test", "created_at": "2024-01-01T00:00:00Z", "done": true,
        "message": {"role":"assistant", "content":"", "tool_calls":[
            {"function":{"name":"same","arguments":{"value":1}}},
            {"id":"tool-0","function":{"name":"same","arguments":{"value":2}}},
            {"function":{"name":"same","arguments":{"value":3}}}
        ]}
    });
    let normalize = || async { unary(wire.clone()).await.unwrap() };
    let first = normalize().await;
    assert_eq!(
        serde_json::to_value(&first.choice).unwrap(),
        serde_json::to_value(normalize().await.choice).unwrap()
    );
    let calls: Vec<_> = first
        .choice
        .iter()
        .filter_map(|item| match item {
            crate::message::AssistantContent::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect();
    assert_eq!(
        calls
            .iter()
            .map(|call| &call.id)
            .collect::<std::collections::HashSet<_>>()
            .len(),
        3
    );
    assert!(calls[0].provider.is_none());
    assert_eq!(calls[1].id.explicit(), Some("tool-0"));
    assert!(calls[2].provider.is_none());
}

/// A provider-issued call id is replayed on both legs; a locally minted
/// handle stays off the wire because the slot is optional and results
/// correlate by `tool_name`.
#[test]
fn daemon_issued_call_ids_replay_and_minted_handles_do_not() {
    use crate::message::{
        AssistantContent, Message as RigMessage, ProviderCallId, ToolCall, ToolCallId,
        ToolFunction, ToolResult, ToolResultContent, UserContent,
    };

    let call = |provider: Option<ProviderCallId>| RigMessage::Assistant {
        id: None,
        content: vec![AssistantContent::ToolCall(ToolCall {
            id: ToolCallId::new("handle").expect("non-empty"),
            provider,
            function: ToolFunction {
                name: "add".to_owned(),
                arguments: serde_json::json!({"x": 1}),
            },
            signature: None,
            additional_params: None,
        })],
    };
    let result = |provider: Option<ProviderCallId>| RigMessage::User {
        content: vec![UserContent::ToolResult(ToolResult {
            call: ToolCallId::new("handle").expect("non-empty"),
            provider,
            name: "add".to_owned(),
            content: vec![ToolResultContent::text("2")],
        })],
    };

    let issued = ProviderCallId::new("call_daemon_1");
    let assistant = Vec::<Message>::try_from(call(issued.clone())).expect("converts");
    let tool = Vec::<Message>::try_from(result(issued)).expect("converts");
    let assistant = serde_json::to_value(&assistant[0]).expect("serializes");
    let tool = serde_json::to_value(&tool[0]).expect("serializes");
    assert_eq!(assistant["tool_calls"][0]["id"], "call_daemon_1");
    assert_eq!(tool["tool_call_id"], "call_daemon_1");
    assert_eq!(tool["tool_name"], "add");

    let assistant = Vec::<Message>::try_from(call(None)).expect("converts");
    let tool = Vec::<Message>::try_from(result(None)).expect("converts");
    let assistant = serde_json::to_value(&assistant[0]).expect("serializes");
    let tool = serde_json::to_value(&tool[0]).expect("serializes");
    assert!(assistant["tool_calls"][0].get("id").is_none());
    assert!(tool.get("tool_call_id").is_none());
}
