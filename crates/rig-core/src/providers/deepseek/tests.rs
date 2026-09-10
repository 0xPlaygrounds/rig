use super::*;
use crate::client::ModelListingClient;
use crate::completion::NormalizeCompletionResponse;
use crate::completion::{
    CompletionRequestBuilder, FinishReason, ToolDefinition as RigToolDefinition,
};
use crate::message::ToolChoice as RigToolChoice;
use crate::model::ModelListingError;
use crate::providers::openai::completion::{
    CompletionRequest as OpenAICompletionRequest, OpenAICompatibleProvider, OpenAIRequestParams,
};
use crate::test_utils::{MockCompletionModel, RecordingHttpClient};

/// Normalize a DeepSeek wire response the way the shared completion path
/// does, threading DeepSeek's own descriptor name through the conversion.
fn normalized(response: CompletionResponse) -> crate::completion::CompletionResponse {
    response
        .normalize(DeepSeek::PROVIDER_NAME)
        .expect("DeepSeek response should convert")
}

fn finalized_body(request: crate::completion::CompletionRequest) -> serde_json::Value {
    let request = OpenAICompletionRequest::try_from(OpenAIRequestParams {
        model: "deepseek-v4-flash".to_string(),
        request,
        strict_tools: false,
        tool_result_array_content: false,
        supports_response_format: DeepSeek::SUPPORTS_RESPONSE_FORMAT,
        supports_image_tool_results: false,
        supports_tools: true,
    })
    .expect("request should convert");
    let mut body = serde_json::to_value(request).expect("request should serialize");
    DeepSeek
        .finalize_request_body(&mut body)
        .expect("finalize should succeed");
    body
}

#[test]
fn test_deserialize_vec_choice() {
    let data = r#"[{
            "finish_reason": "stop",
            "index": 0,
            "logprobs": null,
            "message":{"role":"assistant","content":"Hello, world!"}
            }]"#;

    let choices: Vec<Choice> = serde_json::from_str(data).unwrap();
    assert_eq!(choices.len(), 1);
    match &choices.first().unwrap().message {
        Message::Assistant { content, .. } => assert_eq!(content, "Hello, world!"),
    }
}

#[test]
fn test_deserialize_deepseek_response() {
    let data = r#"{
            "choices":[{
                "finish_reason": "stop",
                "index": 0,
                "logprobs": null,
                "message":{"role":"assistant","content":"Hello, world!"}
            }],
            "usage": {
                "completion_tokens": 0,
                "prompt_tokens": 0,
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 0,
                "total_tokens": 0
            }
        }"#;

    let jd = &mut serde_json::Deserializer::from_str(data);
    let result: Result<CompletionResponse, _> = serde_path_to_error::deserialize(jd);
    match result {
        Ok(response) => match &response.choices.first().unwrap().message {
            Message::Assistant { content, .. } => assert_eq!(content, "Hello, world!"),
        },
        Err(err) => {
            panic!("Deserialization error at {}: {}", err.path(), err);
        }
    }
}

#[test]
fn deepseek_request_serializes_specific_tool_choice_as_chat_completions_object() {
    let request = CompletionRequestBuilder::new(MockCompletionModel::default(), "Use a tool.")
        .tool(RigToolDefinition {
            name: "alpha".to_string(),
            description: "Alpha tool".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        })
        .tool(RigToolDefinition {
            name: "beta".to_string(),
            description: "Beta tool".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        })
        .tool_choice(RigToolChoice::Specific {
            function_names: vec!["beta".to_string()],
        })
        .additional_params(serde_json::json!({"thinking": {"type": "disabled"}}))
        .build();

    let body = finalized_body(request);

    assert_eq!(
        body["tool_choice"],
        serde_json::json!({"type": "function", "function": {"name": "beta"}})
    );
}

#[test]
fn deepseek_request_suppresses_required_tool_choice_when_thinking_is_not_disabled() {
    let request = CompletionRequestBuilder::new(MockCompletionModel::default(), "Use a tool.")
        .tool(RigToolDefinition {
            name: "alpha".to_string(),
            description: "Alpha tool".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        })
        .tool_choice(RigToolChoice::Required)
        .build();

    let body = finalized_body(request);

    assert!(
        body.as_object()
            .expect("body is object")
            .contains_key("tool_choice"),
        "suppressed tool_choice should stay present as an explicit null"
    );
    assert_eq!(body["tool_choice"], serde_json::Value::Null);
}

#[test]
fn deepseek_request_flattens_message_content_to_strings() {
    let request = CompletionRequestBuilder::new(MockCompletionModel::default(), "Hello!")
        .preamble("You are helpful.".to_string())
        .build();

    let body = finalized_body(request);

    assert_eq!(body["messages"][0]["role"], "system");
    assert_eq!(body["messages"][0]["content"], "You are helpful.");
    assert_eq!(body["messages"][1]["role"], "user");
    assert_eq!(body["messages"][1]["content"], "Hello!");
}

#[test]
fn deepseek_finalize_joins_user_parts_with_newline_and_concats_assistant_parts() {
    let mut body = serde_json::json!({
        "model": "deepseek-v4-flash",
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "first part"},
                {"type": "text", "text": "second part"}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": " world"}
            ]}
        ]
    });

    DeepSeek
        .finalize_request_body(&mut body)
        .expect("finalize should succeed");

    assert_eq!(body["messages"][0]["content"], "first part\nsecond part");
    assert_eq!(body["messages"][1]["content"], "Hello world");
}

#[test]
fn deepseek_finalize_adds_tool_call_index_to_assistant_history() {
    let mut body = serde_json::json!({
        "model": "deepseek-v4-flash",
        "messages": [{
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "subtract", "arguments": "{\"x\":2,\"y\":5}"}
            }]
        }]
    });

    DeepSeek
        .finalize_request_body(&mut body)
        .expect("finalize should succeed");

    assert_eq!(body["messages"][0]["tool_calls"][0]["index"], 0);
}

#[test]
fn deepseek_response_preserves_metadata_and_reasoning_token_usage() {
    let raw: CompletionResponse = serde_json::from_value(serde_json::json!({
        "id": "chatcmpl_123",
        "object": "chat.completion",
        "model": "deepseek-v4-flash",
        "system_fingerprint": "fp_123",
        "choices": [{
            "finish_reason": "stop",
            "index": 0,
            "logprobs": null,
            "message": {
                "role": "assistant",
                "content": "done",
                "reasoning_content": "thinking"
            }
        }],
        "usage": {
            "completion_tokens": 8,
            "completion_tokens_details": { "reasoning_tokens": 5 },
            "prompt_tokens": 10,
            "prompt_tokens_details": { "cached_tokens": 3 },
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 10,
            "total_tokens": 18
        }
    }))
    .expect("fixture should deserialize");

    let converted = normalized(raw.clone());

    assert_eq!(raw.id.as_deref(), Some("chatcmpl_123"));
    assert_eq!(raw.model.as_deref(), Some("deepseek-v4-flash"));
    assert_eq!(raw.system_fingerprint.as_deref(), Some("fp_123"));
    assert_eq!(converted.provider, "deepseek");
    assert_eq!(converted.response_id.as_deref(), Some("chatcmpl_123"));
    assert_eq!(converted.message_id, None);
    assert_eq!(converted.model.as_deref(), Some("deepseek-v4-flash"));
    assert_eq!(converted.finish_reason(), Some(FinishReason::Stop));
    assert_eq!(converted.usage.input_tokens, 10);
    assert_eq!(converted.usage.cached_input_tokens, 3);
    assert_eq!(converted.usage.output_tokens, 8);
    assert_eq!(converted.usage.reasoning_tokens, 5);
}

fn response_with_finish_reason(finish_reason: &str) -> CompletionResponse {
    serde_json::from_value(serde_json::json!({
        "id": "chatcmpl_finish",
        "model": "deepseek-v4-flash",
        "choices": [{
            "finish_reason": finish_reason,
            "index": 0,
            "logprobs": null,
            "message": {"role": "assistant", "content": "done"}
        }],
        "usage": {
            "completion_tokens": 1,
            "prompt_tokens": 1,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 1,
            "total_tokens": 2
        }
    }))
    .expect("fixture should deserialize")
}

#[test]
fn deepseek_finish_reasons_normalize_and_preserve_unknowns() {
    for (wire, expected) in [
        ("stop", FinishReason::Stop),
        ("length", FinishReason::Length),
        ("max_tokens", FinishReason::Length),
        ("tool_calls", FinishReason::ToolCalls),
        ("function_call", FinishReason::ToolCalls),
        ("content_filter", FinishReason::ContentFilter),
        // Anything DeepSeek invents survives verbatim rather than reading
        // as a natural stop.
        (
            "insufficient_system_resource",
            FinishReason::Other("insufficient_system_resource".to_owned()),
        ),
    ] {
        let converted = normalized(response_with_finish_reason(wire));

        assert_eq!(converted.finish_reason(), Some(expected), "wire: {wire}");
    }
}

/// Build a one-choice DeepSeek turn out of its three assistant slots.
fn assistant_turn(
    finish_reason: &str,
    content: &str,
    reasoning_content: Option<&str>,
    tool_arguments: &[&str],
) -> CompletionResponse {
    let mut message = serde_json::json!({
        "role": "assistant",
        "content": content,
    });
    if let Some(reasoning_content) = reasoning_content {
        message["reasoning_content"] = serde_json::Value::String(reasoning_content.to_owned());
    }
    if !tool_arguments.is_empty() {
        message["tool_calls"] = tool_arguments
            .iter()
            .enumerate()
            .map(|(index, arguments)| {
                serde_json::json!({
                    "id": format!("call_{index}"),
                    "index": index,
                    "type": "function",
                    "function": {"name": format!("tool_{index}"), "arguments": arguments},
                })
            })
            .collect();
    }

    serde_json::from_value(serde_json::json!({
        "id": "chatcmpl_truncated",
        "model": "deepseek-v4-flash",
        "choices": [{
            "finish_reason": finish_reason,
            "index": 0,
            "logprobs": null,
            "message": message,
        }],
        "usage": {
            "completion_tokens": 24,
            "prompt_tokens": 372,
            "prompt_cache_hit_tokens": 256,
            "prompt_cache_miss_tokens": 116,
            "total_tokens": 396
        }
    }))
    .expect("fixture should deserialize")
}

fn block_kinds(choice: &[crate::completion::AssistantContent]) -> Vec<&'static str> {
    choice
        .iter()
        .map(|content| match content {
            crate::completion::AssistantContent::Text(_) => "text",
            crate::completion::AssistantContent::ToolCall(_) => "tool_call",
            crate::completion::AssistantContent::Reasoning(_) => "reasoning",
            crate::completion::AssistantContent::Image(_) => "image",
        })
        .collect()
}

/// DeepSeek emits the tool call anyway when `max_tokens` runs out mid
/// arguments -- live turns capped at 24/32/48/64 tokens returned
/// `finish_reason: "length"` with `arguments` cut off partway through the
/// object. Parsing strictly took the whole response down with it: text,
/// usage, id, model and finish reason all went with the unusable call.
#[test]
fn deepseek_truncated_tool_arguments_do_not_destroy_the_response() {
    // The 24-token budget's recorded `arguments`, verbatim; the text is
    // added on top so the assertions below can show it survives too (the
    // recorded 24-token turn itself came back with `content: ""`).
    let raw = assistant_turn("length", "Acknowledged.", None, &[r#"{"summary": "#]);

    assert!(
        match &raw.choices[0].message {
            Message::Assistant { tool_calls, .. } => tool_calls.is_empty(),
        },
        "the unusable call is dropped at decode, not surfaced as a sentinel"
    );

    let converted = normalized(raw);

    assert_eq!(converted.finish_reason(), Some(FinishReason::Length));
    assert_eq!(converted.response_id.as_deref(), Some("chatcmpl_truncated"));
    assert_eq!(converted.model.as_deref(), Some("deepseek-v4-flash"));
    assert_eq!(converted.usage.total_tokens, 396);
    assert_eq!(converted.usage.cached_input_tokens, 256);
    assert_eq!(block_kinds(&converted.choice), vec!["text"]);
}

/// The unusable call is dropped, exactly as the streaming path drops it,
/// while a complete sibling in the same turn survives.
#[test]
fn deepseek_parallel_calls_drop_only_the_truncated_one() {
    let converted = normalized(assistant_turn(
        "length",
        "",
        None,
        &[r#"{"team": "platform"}"#, r#"{"summary": "Log this"#],
    ));

    assert_eq!(block_kinds(&converted.choice), vec!["tool_call"]);
    let crate::completion::AssistantContent::ToolCall(call) = &converted.choice[0] else {
        panic!("expected the complete call to survive");
    };
    assert_eq!(call.function.name, "tool_0");
    assert_eq!(
        call.function.arguments,
        serde_json::json!({"team": "platform"})
    );
}

/// The tolerant parse must not weaken a complete payload, and must keep
/// reading an empty one as a parameterless invocation.
#[test]
fn deepseek_complete_and_empty_tool_arguments_are_unaffected() {
    let complete = normalized(assistant_turn(
        "tool_calls",
        "",
        None,
        &[r#"{"summary": "done"}"#],
    ));
    let crate::completion::AssistantContent::ToolCall(call) = &complete.choice[0] else {
        panic!("expected a tool call");
    };
    assert_eq!(
        call.function.arguments,
        serde_json::json!({"summary": "done"})
    );

    let empty = normalized(assistant_turn("tool_calls", "", None, &[""]));
    let crate::completion::AssistantContent::ToolCall(call) = &empty.choice[0] else {
        panic!("expected a parameterless tool call");
    };
    assert_eq!(call.function.arguments, serde_json::json!({}));

    let truncated_empty = normalized(assistant_turn("length", "", None, &[""]));
    assert!(
        truncated_empty.choice.is_empty(),
        "an output-length turn with no argument tokens must not dispatch a tool"
    );
}

/// DeepSeek documents that an ordinary function call may contain invalid
/// JSON. Without an outer `length` signal that is a provider response
/// defect and must not disappear from the native response.
#[test]
fn deepseek_malformed_completed_tool_call_is_loud() {
    let response = serde_json::json!({
        "id": "chatcmpl-malformed",
        "model": "deepseek-v4-flash",
        "choices": [{
            "finish_reason": "tool_calls",
            "index": 0,
            "logprobs": null,
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_0",
                    "index": 0,
                    "type": "function",
                    "function": {"name": "page", "arguments": "{\"team\":"}
                }]
            }
        }],
        "usage": {
            "completion_tokens": 1,
            "prompt_tokens": 1,
            "total_tokens": 2
        }
    });

    assert!(
        serde_json::from_value::<CompletionResponse>(response).is_err(),
        "a completed malformed call must not be rewritten away"
    );
}

/// The full blocking block-order enumeration: reasoning present/absent x
/// text present/absent x zero/one/two tool calls. Reasoning leads the
/// choice on every shape, which is the order DeepSeek's own stream emits
/// (`reasoning_content` deltas before the first `content` delta and before
/// the tool call) and the order the shared canonical chunk lifecycle
/// imposes. Appending it last made the two transports disagree about
/// identical wire bytes.
#[test]
fn deepseek_reasoning_leads_the_choice_on_every_turn_shape() {
    for reasoning in [None, Some("thinking")] {
        for text in ["", "spoken"] {
            for calls in [
                &[][..],
                &[r#"{"x":1}"#][..],
                &[r#"{"x":1}"#, r#"{"y":2}"#][..],
            ] {
                let finish_reason = if calls.is_empty() {
                    "stop"
                } else {
                    "tool_calls"
                };
                let raw = assistant_turn(finish_reason, text, reasoning, calls);
                // A turn with nothing in it at all is a provider defect the
                // shared skeleton rejects; it is not an ordering shape.
                if reasoning.is_none() && text.is_empty() && calls.is_empty() {
                    continue;
                }
                let kinds = block_kinds(&normalized(raw).choice);

                let mut expected = Vec::new();
                if reasoning.is_some() {
                    expected.push("reasoning");
                }
                if !text.is_empty() {
                    expected.push("text");
                }
                expected.extend(std::iter::repeat_n("tool_call", calls.len()));

                assert_eq!(
                    kinds,
                    expected,
                    "reasoning={reasoning:?} text={text:?} calls={}",
                    calls.len()
                );
            }
        }
    }
}

#[test]
fn deepseek_stop_finish_reason_upgrades_when_the_turn_called_a_tool() {
    let raw: CompletionResponse = serde_json::from_value(serde_json::json!({
        "id": "chatcmpl_tool",
        "model": "deepseek-v4-flash",
        "choices": [{
            "finish_reason": "stop",
            "index": 0,
            "logprobs": null,
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "index": 0,
                    "type": "function",
                    "function": {"name": "subtract", "arguments": "{\"x\":2,\"y\":5}"}
                }]
            }
        }],
        "usage": {
            "completion_tokens": 1,
            "prompt_tokens": 1,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 1,
            "total_tokens": 2
        }
    }))
    .expect("fixture should deserialize");

    assert_eq!(
        normalized(raw).finish_reason(),
        Some(FinishReason::ToolCalls)
    );
}

#[test]
fn test_deserialize_example_response() {
    let data = r#"
        {
            "id": "e45f6c68-9d9e-43de-beb4-4f402b850feb",
            "object": "chat.completion",
            "created": 0,
            "model": "deepseek-chat",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Why don’t skeletons fight each other?  \nBecause they don’t have the guts! 😄"
                    },
                    "logprobs": null,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 13,
                "completion_tokens": 32,
                "total_tokens": 45,
                "prompt_tokens_details": {
                    "cached_tokens": 0
                },
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 13
            },
            "system_fingerprint": "fp_4b6881f2c5"
        }
        "#;
    let jd = &mut serde_json::Deserializer::from_str(data);
    let result: Result<CompletionResponse, _> = serde_path_to_error::deserialize(jd);

    match result {
        Ok(response) => match &response.choices.first().unwrap().message {
            Message::Assistant { content, .. } => assert_eq!(
                content,
                "Why don’t skeletons fight each other?  \nBecause they don’t have the guts! 😄"
            ),
        },
        Err(err) => {
            panic!("Deserialization error at {}: {}", err.path(), err);
        }
    }
}

#[test]
fn test_serialize_deserialize_tool_call_message() {
    let tool_call_choice_json = r#"
            {
              "finish_reason": "tool_calls",
              "index": 0,
              "logprobs": null,
              "message": {
                "content": "",
                "role": "assistant",
                "tool_calls": [
                  {
                    "function": {
                      "arguments": "{\"x\":2,\"y\":5}",
                      "name": "subtract"
                    },
                    "id": "call_0_2b4a85ee-b04a-40ad-a16b-a405caf6e65b",
                    "index": 0,
                    "type": "function"
                  }
                ]
              }
            }
        "#;

    let choice: Choice =
        serde_json::from_str(tool_call_choice_json).expect("choice should deserialize");
    match &choice.message {
        Message::Assistant { tool_calls, .. } => {
            assert_eq!(tool_calls.len(), 1);
            let call = tool_calls.first().expect("one tool call");
            assert_eq!(call.function.name, "subtract");
            assert_eq!(call.index, 0);
        }
    }

    let serialized = serde_json::to_value(&choice).expect("choice should serialize");
    assert_eq!(
        serialized["message"]["tool_calls"][0]["function"]["name"],
        "subtract"
    );
}

#[test]
fn test_client_initialization() {
    let _client = crate::providers::deepseek::Client::new_with(
        "dummy-key",
        crate::test_utils::RecordingHttpClient::new(""),
    )
    .expect("Client::new() failed");
    let _client_from_builder = crate::providers::deepseek::Client::builder()
        .api_key("dummy-key")
        .http_client(crate::test_utils::RecordingHttpClient::new(""))
        .build()
        .expect("Client::builder() failed");
}

#[test]
fn test_deserialize_list_models_response() {
    let data = r#"{
            "object": "list",
            "data": [
                {"id": "deepseek-chat", "object": "model", "owned_by": "deepseek"},
                {"id": "deepseek-reasoner", "object": "model", "owned_by": "deepseek"}
            ]
        }"#;

    let response: crate::providers::internal::model_listing::DataEnvelope<
        crate::providers::internal::model_listing::ListModelEntry,
    > = serde_json::from_str(data).expect("list models response should deserialize");
    assert_eq!(response.data.len(), 2);
    assert_eq!(response.data[0].id, "deepseek-chat");
    assert_eq!(response.data[0].owned_by.as_deref(), Some("deepseek"));
}

#[tokio::test]
async fn test_list_models_uses_models_endpoint() {
    let response_body = r#"{
            "object": "list",
            "data": [
                {
                    "id": "deepseek-v4-flash",
                    "object": "model",
                    "owned_by": "deepseek"
                },
                {
                    "id": "deepseek-v4-pro",
                    "object": "model",
                    "owned_by": "deepseek"
                }
            ]
        }"#;

    let http_client = RecordingHttpClient::new(response_body);
    let client = Client::builder()
        .api_key("dummy-key")
        .http_client(http_client.clone())
        .build()
        .expect("client should build");

    let models = client
        .list_models()
        .await
        .expect("list_models should succeed");

    assert_eq!(models.len(), 2);
    assert_eq!(models.data[0].id, "deepseek-v4-flash");
    assert_eq!(models.data[0].r#type, None);
    assert_eq!(models.data[0].owned_by.as_deref(), Some("deepseek"));
    let requests = http_client.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].uri, "https://api.deepseek.com/models");
}

#[tokio::test]
async fn test_list_models_preserves_api_error_context() {
    let http_client = RecordingHttpClient::with_error(
        http::StatusCode::UNAUTHORIZED,
        r#"{"error":{"message":"invalid api key"}}"#,
    );
    let client = Client::builder()
        .api_key("dummy-key")
        .http_client(http_client)
        .build()
        .expect("client should build");

    let error = client
        .list_models()
        .await
        .expect_err("list_models should fail");

    match error {
        ModelListingError::ApiError {
            status_code,
            message,
        } => {
            assert_eq!(status_code, 401);
            assert!(message.contains("provider=DeepSeek"));
            assert!(message.contains("path=/models"));
            assert!(message.contains("invalid api key"));
        }
        other => panic!("expected api error, got {other:?}"),
    }
}
