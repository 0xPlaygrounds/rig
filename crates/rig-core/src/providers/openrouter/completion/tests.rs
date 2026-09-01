use super::*;
use crate::completion::NormalizeCompletionResponse;
use crate::message::{AudioMediaType, ImageDetail, VideoMediaType};
use serde_json::json;

#[test]
fn openrouter_client_constructs_a_completion_model() {
    // Also a compile guard: it instantiates the shared chat-completions
    // model over `OpenRouterExt`, which is what proves this provider's
    // response conversion satisfies the normalization bound.
    use crate::client::CompletionClient;

    let client = crate::providers::openrouter::Client::new_with(
        "dummy-key",
        crate::test_utils::RecordingHttpClient::new(""),
    )
    .expect("Client::new() failed");
    let model = client.completion_model(GEMINI_FLASH_2_0);

    assert_eq!(model.model, GEMINI_FLASH_2_0);
}

#[test]
fn mixed_user_content_preserves_order_around_tool_results() {
    let content = vec![
        message::UserContent::text("before"),
        message::UserContent::tool_result_with_call_id(
            "result-id",
            "call-id".to_string(),
            "tool",
            vec![message::ToolResultContent::text("tool output")],
        ),
        message::UserContent::text("after"),
    ];

    let messages = user_contents_to_messages(content).expect("message conversion");

    assert!(matches!(
        messages.as_slice(),
        [
            Message::User { content: before, .. },
            Message::ToolResult { tool_call_id, .. },
            Message::User { content: after, .. },
        ] if matches!(before.first(), Some(UserContent::Text { text }) if text == "before")
            && tool_call_id == "call-id"
            && matches!(after.first(), Some(UserContent::Text { text }) if text == "after")
    ));
}

#[test]
fn test_openrouter_request_uses_request_model_override() {
    let request = CompletionRequest {
        model: Some("google/gemini-2.5-flash".to_string()),
        chat_history: vec!["Hello".into()],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    };

    let openrouter_request = OpenrouterCompletionRequest::try_from(("openai/gpt-4o-mini", request))
        .expect("request conversion should succeed");
    let serialized =
        serde_json::to_value(openrouter_request).expect("serialization should succeed");

    assert_eq!(serialized["model"], "google/gemini-2.5-flash");
}

/// The caller's `max_tokens` must reach the serialized request body —
/// OpenRouter accepts `max_tokens` like OpenAI, and dropping it silently
/// removed the caller's output cap.
#[test]
fn openrouter_request_carries_caller_max_tokens() {
    let request = CompletionRequest {
        model: None,
        chat_history: vec!["Hello".into()],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: Some(512),
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    };

    let openrouter_request = OpenrouterCompletionRequest::try_from(OpenRouterRequestParams {
        model: "openai/gpt-4o-mini",
        request,
        strict_tools: false,
    })
    .expect("request conversion should succeed");
    let serialized =
        serde_json::to_value(openrouter_request).expect("serialization should succeed");

    assert_eq!(serialized["max_tokens"], 512);
}

#[test]
fn openrouter_params_include_direct_request_documents() {
    let request = CompletionRequest {
        model: None,
        chat_history: vec![crate::message::Message::user("What is glarb-glarb?")],
        documents: vec![crate::completion::request::Document {
            id: "doc_1".to_string(),
            text: "Definition of glarb-glarb: an ancient tool.".to_string(),
            additional_props: Default::default(),
        }],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    };

    let request = OpenrouterCompletionRequest::try_from(OpenRouterRequestParams {
        model: "openai/gpt-4o-mini",
        request,
        strict_tools: false,
    })
    .expect("request conversion should succeed");
    let serialized = serde_json::to_value(request).expect("serialization should succeed");

    assert!(
        serialized["messages"].to_string().contains("glarb-glarb"),
        "direct request documents should be normalized through public params"
    );
}

#[test]
fn test_openrouter_request_uses_default_model_when_override_unset() {
    let request = CompletionRequest {
        model: None,
        chat_history: vec!["Hello".into()],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    };

    let openrouter_request = OpenrouterCompletionRequest::try_from(("openai/gpt-4o-mini", request))
        .expect("request conversion should succeed");
    let serialized =
        serde_json::to_value(openrouter_request).expect("serialization should succeed");

    assert_eq!(serialized["model"], "openai/gpt-4o-mini");
}

#[test]
fn final_request_body_serializes_assistant_reasoning_under_openrouter_key() {
    // Reasoning replay normally flows through `reasoning_details`; the
    // plain string field must nevertheless hit the wire under
    // OpenRouter's `reasoning` key, not the shared `reasoning_content`.
    let request = OpenrouterCompletionRequest {
        model: "openai/gpt-4o".to_string(),
        messages: vec![Message::Assistant {
            content: vec![],
            reasoning: Some("thinking it through".to_string()),
            refusal: None,
            audio: None,
            name: None,
            tool_calls: vec![],
            reasoning_details: vec![],
            images: vec![],
        }],
        temperature: None,
        max_tokens: None,
        tools: vec![],
        tool_choice: None,
        additional_params: None,
    };

    let body = final_request_body(&request, false).expect("body should serialize");

    assert_eq!(
        body["messages"][0]["reasoning"],
        serde_json::json!("thinking it through")
    );
    assert!(
        body["messages"][0].get("reasoning_content").is_none(),
        "OpenRouter's assistant reasoning key is `reasoning`, not `reasoning_content`"
    );
}

#[test]
fn test_openrouter_request_maps_output_schema_to_response_format() {
    let schema: schemars::Schema = serde_json::from_value(json!({
        "title": "WeatherResponse",
        "type": "object",
        "properties": {
            "city": { "type": "string" },
            "weather": { "type": "string" }
        }
    }))
    .expect("schema should deserialize");

    let request = CompletionRequest {
        model: None,
        chat_history: vec!["Hello".into()],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: Some(schema),
        record_telemetry_content: false,
    };

    let openrouter_request = OpenrouterCompletionRequest::try_from(("openai/gpt-4o-mini", request))
        .expect("request conversion should succeed");
    let serialized =
        serde_json::to_value(openrouter_request).expect("serialization should succeed");

    assert_eq!(
        serialized["response_format"],
        json!({
            "type": "json_schema",
            "json_schema": {
                "name": "WeatherResponse",
                "strict": true,
                "schema": {
                    "title": "WeatherResponse",
                    "type": "object",
                    "properties": {
                        "city": { "type": "string" },
                        "weather": { "type": "string" }
                    },
                    "additionalProperties": false,
                    "required": ["city", "weather"]
                }
            }
        })
    );
}

#[test]
fn test_openrouter_request_merges_output_schema_with_provider_preferences() {
    let schema: schemars::Schema = serde_json::from_value(json!({
        "type": "object",
        "properties": {
            "answer": { "type": "string" }
        }
    }))
    .expect("schema should deserialize");

    let request = CompletionRequest {
        model: None,
        chat_history: vec!["Hello".into()],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: Some(
            ProviderPreferences::new()
                .require_parameters(true)
                .to_json(),
        ),
        output_schema: Some(schema),
        record_telemetry_content: false,
    };

    let openrouter_request = OpenrouterCompletionRequest::try_from(("openai/gpt-4o-mini", request))
        .expect("request conversion should succeed");
    let serialized =
        serde_json::to_value(openrouter_request).expect("serialization should succeed");

    assert_eq!(serialized["provider"]["require_parameters"], true);
    assert_eq!(serialized["response_format"]["type"], "json_schema");
    assert_eq!(
        serialized["response_format"]["json_schema"]["name"],
        "response_schema"
    );
    assert_eq!(
        serialized["response_format"]["json_schema"]["schema"]["additionalProperties"],
        false
    );
}

#[test]
fn test_completion_response_deserialization_gemini_flash() {
    // Real response from OpenRouter with google/gemini-2.5-flash
    let json = json!({
        "id": "gen-AAAAAAAAAA-AAAAAAAAAAAAAAAAAAAA",
        "provider": "Google",
        "model": "google/gemini-2.5-flash",
        "object": "chat.completion",
        "created": 1765971703u64,
        "choices": [{
            "logprobs": null,
            "finish_reason": "stop",
            "native_finish_reason": "STOP",
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "CONTENT",
                "refusal": null,
                "reasoning": null
            }
        }],
        "usage": {
            "prompt_tokens": 669,
            "completion_tokens": 5,
            "total_tokens": 674
        }
    });

    let response: CompletionResponse = serde_json::from_value(json).unwrap();
    assert_eq!(response.id, "gen-AAAAAAAAAA-AAAAAAAAAAAAAAAAAAAA");
    assert_eq!(response.model, "google/gemini-2.5-flash");
    assert_eq!(response.choices.len(), 1);
    assert_eq!(response.choices[0].finish_reason, Some("stop".to_string()));
    assert_eq!(response.choices[0].logprobs, None);
    let serialized = serde_json::to_value(&response).unwrap();
    assert!(
        serialized["choices"][0].get("logprobs").is_none(),
        "an absent optional native field stays absent when serialized"
    );
}

#[test]
fn raw_completion_choice_retains_logprobs() {
    let logprobs = json!({
        "content": [{
            "token": "cobalt",
            "logprob": -0.01,
            "bytes": [99],
            "top_logprobs": []
        }],
        "refusal": null
    });
    let response: CompletionResponse = serde_json::from_value(json!({
        "id": "gen-logprobs",
        "object": "chat.completion",
        "created": 1,
        "model": "openai/gpt-4o-mini",
        "system_fingerprint": null,
        "choices": [{
            "index": 0,
            "native_finish_reason": "stop",
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "cobalt"},
            "logprobs": logprobs
        }],
        "usage": null
    }))
    .expect("OpenRouter's documented probability object should decode");

    assert_eq!(response.choices[0].logprobs, Some(logprobs));
}

#[test]
fn test_completion_response_usage_prefers_reported_completion_tokens() {
    let json = json!({
        "id": "gen-usage-divergent",
        "object": "chat.completion",
        "created": 1,
        "model": "anthropic/claude-3.5-sonnet",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop"
        }],
        // Divergent accounting: total != prompt + completion.
        "usage": {"prompt_tokens": 500, "completion_tokens": 10, "total_tokens": 505}
    });

    let response: CompletionResponse = serde_json::from_value(json).unwrap();
    let converted = response.normalize(PROVIDER_NAME).unwrap();
    assert_eq!(converted.usage.output_tokens, 10);
}

#[test]
fn test_completion_response_usage_falls_back_when_completion_tokens_missing() {
    let json = json!({
        "id": "gen-usage-omitted",
        "object": "chat.completion",
        "created": 1,
        "model": "some/gateway-model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 100, "total_tokens": 110}
    });

    let response: CompletionResponse = serde_json::from_value(json).unwrap();
    let converted = response.normalize(PROVIDER_NAME).unwrap();
    assert_eq!(converted.usage.output_tokens, 10);
}

#[test]
fn test_completion_response_maps_cache_token_accounting() {
    let json = json!({
        "id": "gen-cache-test",
        "object": "chat.completion",
        "created": 1,
        "model": "anthropic/claude-3.5-sonnet",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "Hi"
            }
        }],
        "usage": {
            "prompt_tokens": 500,
            "completion_tokens": 10,
            "total_tokens": 510,
            "prompt_tokens_details": {
                "cached_tokens": 400,
                "cache_write_tokens": 50
            }
        }
    });

    let response: CompletionResponse = serde_json::from_value(json).unwrap();
    let converted = response.normalize(PROVIDER_NAME).unwrap();

    assert_eq!(converted.usage.input_tokens, 500);
    assert_eq!(converted.usage.output_tokens, 10);
    assert_eq!(converted.usage.cached_input_tokens, 400);
    assert_eq!(converted.usage.cache_creation_input_tokens, 50);
}

#[test]
fn test_completion_response_cache_tokens_absent_defaults_to_zero() {
    let json = json!({
        "id": "gen-no-cache",
        "object": "chat.completion",
        "created": 1,
        "model": "openai/gpt-4o",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "Hi"
            }
        }],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "total_tokens": 110
        }
    });

    let response: CompletionResponse = serde_json::from_value(json).unwrap();
    let converted = response.normalize(PROVIDER_NAME).unwrap();

    assert_eq!(converted.usage.cached_input_tokens, 0);
    assert_eq!(converted.usage.cache_creation_input_tokens, 0);
}

#[test]
fn test_completion_response_deserialization_gemini_model_role() {
    let json = json!({
        "id": "gen-BBBBBBBBBB-BBBBBBBBBBBBBBBBBBBB",
        "provider": "Google",
        "model": "google/gemini-2.5-pro-exp-03-25:free",
        "object": "chat.completion",
        "created": 1743780565u64,
        "choices": [{
            "logprobs": null,
            "finish_reason": "stop",
            "native_finish_reason": "STOP",
            "index": 0,
            "message": {
                "role": "model",
                "content": "CONTENT",
                "refusal": null,
                "reasoning": null
            }
        }],
        "usage": {
            "prompt_tokens": 669,
            "completion_tokens": 5,
            "total_tokens": 674
        }
    });

    let response: CompletionResponse = serde_json::from_value(json).unwrap();
    let converted = response.normalize(PROVIDER_NAME).unwrap();

    // The normalized response carries the model OpenRouter reported, which
    // is routinely not the one that was requested.
    assert_eq!(
        converted.model.as_deref(),
        Some("google/gemini-2.5-pro-exp-03-25:free")
    );
    assert_eq!(converted.provider, "openrouter");
    assert!(matches!(
        converted.choice.first(),
        Some(completion::AssistantContent::Text(text)) if text.text == "CONTENT"
    ));
}

#[test]
fn openrouter_finish_reasons_map_and_preserve_unknown_values() {
    use crate::completion::FinishReason;

    let choice = |finish_reason: Option<&str>, native: Option<&str>| Choice {
        index: 0,
        native_finish_reason: native.map(str::to_string),
        message: Message::Assistant {
            content: vec![],
            reasoning: None,
            refusal: None,
            audio: None,
            name: None,
            tool_calls: vec![],
            reasoning_details: vec![],
            images: vec![],
        },
        finish_reason: finish_reason.map(str::to_string),
        logprobs: None,
    };

    assert_eq!(
        map_finish_reason(&choice(Some("stop"), Some("STOP"))),
        Some(FinishReason::Stop)
    );
    assert_eq!(
        map_finish_reason(&choice(Some("length"), None)),
        Some(FinishReason::Length)
    );
    assert_eq!(
        map_finish_reason(&choice(Some("tool_calls"), None)),
        Some(FinishReason::ToolCalls)
    );
    assert_eq!(
        map_finish_reason(&choice(Some("content_filter"), None)),
        Some(FinishReason::ContentFilter)
    );
    assert_eq!(
        map_finish_reason(&choice(None, Some("completed"))),
        Some(FinishReason::Stop)
    );
    assert_eq!(
        map_finish_reason(&choice(None, Some("max_output_tokens"))),
        Some(FinishReason::Length)
    );
    // A reason OpenRouter could not translate survives verbatim rather
    // than reading as a natural stop.
    assert_eq!(
        map_finish_reason(&choice(Some("error"), None)),
        Some(FinishReason::Other("error".to_string()))
    );
    // No normalized reason: the upstream provider's own spelling is
    // reported, in its own casing.
    assert_eq!(
        map_finish_reason(&choice(None, Some("MALFORMED_FUNCTION_CALL"))),
        Some(FinishReason::Other("MALFORMED_FUNCTION_CALL".to_string()))
    );
    assert_eq!(map_finish_reason(&choice(None, None)), None);
}

#[test]
fn openrouter_stop_with_tool_call_reports_tool_calls() {
    // OpenRouter gateways routinely report a plain `stop` on a turn that
    // carried tool calls; the normalized response upgrades it.
    let json = json!({
        "id": "gen-tool",
        "object": "chat.completion",
        "created": 1,
        "model": "anthropic/claude-3.5-sonnet",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"}
                }]
            }
        }]
    });

    let response: CompletionResponse = serde_json::from_value(json).unwrap();
    let converted = response.normalize(PROVIDER_NAME).unwrap();

    assert_eq!(
        converted.finish_reason(),
        Some(crate::completion::FinishReason::ToolCalls)
    );
}

/// The shared choice decoder tolerates the truncated JSON a
/// `max_tokens`-capped turn emits only under `finish_reason: length`, and
/// every normalizer built on it drops the unusable call rather than losing
/// the turn. Reproduced live on
/// DeepSeek (rig#2354) at 24/32/48/64-token budgets; the same wire type
/// backs OpenRouter, so the same turn shape is pinned here.
#[test]
fn openrouter_truncated_tool_arguments_do_not_destroy_the_response() {
    let json = json!({
        "id": "gen-truncated",
        "object": "chat.completion",
        "created": 1,
        "model": "deepseek/deepseek-chat",
        "choices": [{
            "index": 0,
            "finish_reason": "length",
            "message": {
                "role": "assistant",
                "content": "Acknowledged.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "page", "arguments": "{\"team\":\"platform\"}"}
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "file_report", "arguments": "{\"summary\": "}
                    }
                ]
            }
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 24, "total_tokens": 34}
    });

    let response: CompletionResponse = serde_json::from_value(json).unwrap();
    let converted = response.normalize(PROVIDER_NAME).unwrap();

    assert_eq!(
        converted.finish_reason(),
        Some(crate::completion::FinishReason::Length)
    );
    let names = converted
        .choice
        .iter()
        .filter_map(|content| match content {
            completion::AssistantContent::ToolCall(call) => Some(call.function.name.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(names, vec!["page"], "only the truncated call is dropped");
    assert!(
        converted.choice.iter().any(|content| matches!(
            content,
            completion::AssistantContent::Text(text) if text.text == "Acknowledged."
        )),
        "the turn's text survives: {:?}",
        converted.choice
    );
    assert_eq!(converted.usage.total_tokens, 34);
}

#[test]
fn openrouter_native_length_fallback_tolerates_truncated_tool_arguments() {
    let json = json!({
        "id": "gen-native-truncated",
        "object": "chat.completion",
        "created": 1,
        "model": "anthropic/claude-haiku-4.5",
        "choices": [{
            "index": 0,
            "finish_reason": null,
            "native_finish_reason": "max_output_tokens",
            "message": {
                "role": "assistant",
                "content": "still useful",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{\"q\":"}
                }]
            }
        }]
    });

    let response: CompletionResponse = serde_json::from_value(json)
        .expect("the native terminal reason should authorize narrow truncation tolerance");
    let converted = response.normalize(PROVIDER_NAME).unwrap();

    assert_eq!(
        converted.finish_reason(),
        Some(crate::completion::FinishReason::Length)
    );
    assert!(
        converted
            .choice
            .iter()
            .all(|content| !matches!(content, completion::AssistantContent::ToolCall(_)))
    );
    assert!(matches!(
        converted.choice.first(),
        Some(completion::AssistantContent::Text(text)) if text.text == "still useful"
    ));
}

#[test]
fn openrouter_length_preserves_an_empty_turn_after_dropping_its_only_call() {
    for (finish_reason, native_finish_reason) in
        [(Some("length"), None), (None, Some("max_output_tokens"))]
    {
        let response: CompletionResponse = serde_json::from_value(json!({
            "id": "gen-empty-truncated",
            "object": "chat.completion",
            "created": 1,
            "model": "openai/gpt-4.1-mini",
            "choices": [{
                "index": 0,
                "finish_reason": finish_reason,
                "native_finish_reason": native_finish_reason,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": ""}
                    }]
                }
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}
        }))
        .expect("outer length should permit dropping the incomplete call");
        let converted = response
            .normalize(PROVIDER_NAME)
            .expect("an empty truncated turn still carries its diagnostic");

        assert!(converted.choice.is_empty());
        assert_eq!(
            converted.finish_reason(),
            Some(crate::completion::FinishReason::Length)
        );
        assert_eq!(converted.usage.total_tokens, 11);
        assert_eq!(
            converted.response_id.as_deref(),
            Some("gen-empty-truncated")
        );
    }
}

#[test]
fn openrouter_content_filter_preserves_an_empty_turn() {
    let response: CompletionResponse = serde_json::from_value(json!({
        "id": "gen-filtered",
        "object": "chat.completion",
        "created": 1,
        "model": "openai/gpt-4.1-mini",
        "choices": [{
            "index": 0,
            "finish_reason": "content_filter",
            "message": {"role": "assistant", "content": null}
        }]
    }))
    .unwrap();
    let converted = response.normalize(PROVIDER_NAME).unwrap();

    assert!(converted.choice.is_empty());
    assert_eq!(
        converted.finish_reason(),
        Some(crate::completion::FinishReason::ContentFilter)
    );
}

#[test]
fn openrouter_malformed_completed_tool_arguments_remain_loud() {
    let json = json!({
        "id": "gen-malformed",
        "object": "chat.completion",
        "created": 1,
        "model": "deepseek/deepseek-chat",
        "choices": [{
            "index": 0,
            "finish_reason": "tool_calls",
            "native_finish_reason": "max_output_tokens",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{\"q\":"}
                }]
            }
        }]
    });

    assert!(
        serde_json::from_value::<CompletionResponse>(json).is_err(),
        "only an outer output-length reason authorizes truncation tolerance"
    );
}

#[tokio::test]
async fn streaming_native_length_fallback_drops_partial_tool_call() {
    use crate::client::CompletionClient;
    use crate::completion::CompletionModel as _;
    use crate::providers::internal::openai_chat_completions_compatible::test_support::sse_bytes_from_data_lines;
    use crate::streaming::StreamedAssistantContent;
    use crate::test_utils::MockStreamingClient;
    use futures::StreamExt;

    let http_client = MockStreamingClient {
        sse_bytes: sse_bytes_from_data_lines([
            r#"{"id":"gen-native-truncated","model":"anthropic/claude-haiku-4.5","choices":[{"index":0,"delta":{"role":"assistant","content":"still useful","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\":"}}]},"finish_reason":null,"native_finish_reason":null}]}"#,
            r#"{"id":"gen-native-truncated","model":"anthropic/claude-haiku-4.5","choices":[{"index":0,"delta":{},"finish_reason":null,"native_finish_reason":"max_output_tokens"}]}"#,
            "[DONE]",
        ]),
    };
    let client = crate::providers::openrouter::Client::builder()
        .api_key("dummy-key")
        .http_client(http_client)
        .build()
        .expect("client should build");
    let model = client.completion_model("anthropic/claude-haiku-4.5");
    let request = model.completion_request("lookup").build();
    let mut stream = model.stream(request).await.expect("stream should start");

    let mut terminal = None;
    let mut saw_tool_call = false;
    while let Some(item) = stream.next().await {
        match item.expect("native max_tokens truncation is tolerated") {
            StreamedAssistantContent::ToolCall { .. } => saw_tool_call = true,
            StreamedAssistantContent::Final(final_record) => terminal = Some(final_record),
            _ => {}
        }
    }

    assert!(
        !saw_tool_call,
        "the partial call must not become executable"
    );
    assert_eq!(
        terminal.and_then(|record| record.finish_reason),
        Some(crate::completion::FinishReason::Length)
    );
}

#[tokio::test]
async fn streaming_normalized_reason_wins_over_native_length() {
    use crate::client::CompletionClient;
    use crate::completion::CompletionModel as _;
    use crate::providers::internal::openai_chat_completions_compatible::test_support::sse_bytes_from_data_lines;
    use crate::streaming::StreamedAssistantContent;
    use crate::test_utils::MockStreamingClient;
    use futures::StreamExt;

    let http_client = MockStreamingClient {
        sse_bytes: sse_bytes_from_data_lines([
            r#"{"id":"gen-normalized-wins","model":"anthropic/claude-haiku-4.5","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\":"}}]},"finish_reason":null,"native_finish_reason":null}]}"#,
            r#"{"id":"gen-normalized-wins","model":"anthropic/claude-haiku-4.5","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls","native_finish_reason":"max_output_tokens"}]}"#,
            "[DONE]",
        ]),
    };
    let client = crate::providers::openrouter::Client::builder()
        .api_key("dummy-key")
        .http_client(http_client)
        .build()
        .expect("client should build");
    let model = client.completion_model("anthropic/claude-haiku-4.5");
    let request = model.completion_request("lookup").build();
    let mut stream = model.stream(request).await.expect("stream should start");

    let mut terminal = None;
    let mut errors = Vec::new();
    while let Some(item) = stream.next().await {
        match item {
            Ok(StreamedAssistantContent::Final(final_record)) => terminal = Some(final_record),
            Ok(_) => {}
            Err(error) => errors.push(error.to_string()),
        }
    }

    assert_eq!(errors.len(), 1, "the completed malformed call stays loud");
    assert!(errors[0].contains("malformed JSON input"), "{}", errors[0]);
    assert_eq!(
        terminal.and_then(|record| record.finish_reason),
        Some(crate::completion::FinishReason::ToolCalls)
    );
}

#[test]
fn test_message_assistant_without_reasoning_details() {
    // Verify that missing reasoning_details field doesn't cause deserialization failure
    let json = json!({
        "role": "assistant",
        "content": "Hello world",
        "refusal": null,
        "reasoning": null
    });

    let message: Message = serde_json::from_value(json).unwrap();
    match message {
        Message::Assistant {
            content,
            reasoning_details,
            ..
        } => {
            assert_eq!(content.len(), 1);
            assert!(reasoning_details.is_empty());
        }
        _ => panic!("Expected Assistant message"),
    }
}

#[test]
fn test_data_collection_serialization() {
    assert_eq!(
        serde_json::to_string(&DataCollection::Allow).unwrap(),
        r#""allow""#
    );
    assert_eq!(
        serde_json::to_string(&DataCollection::Deny).unwrap(),
        r#""deny""#
    );
}

#[test]
fn test_data_collection_default() {
    assert_eq!(DataCollection::default(), DataCollection::Allow);
}

#[test]
fn test_quantization_serialization() {
    assert_eq!(
        serde_json::to_string(&Quantization::Int4).unwrap(),
        r#""int4""#
    );
    assert_eq!(
        serde_json::to_string(&Quantization::Int8).unwrap(),
        r#""int8""#
    );
    assert_eq!(
        serde_json::to_string(&Quantization::Fp16).unwrap(),
        r#""fp16""#
    );
    assert_eq!(
        serde_json::to_string(&Quantization::Bf16).unwrap(),
        r#""bf16""#
    );
    assert_eq!(
        serde_json::to_string(&Quantization::Fp32).unwrap(),
        r#""fp32""#
    );
    assert_eq!(
        serde_json::to_string(&Quantization::Fp8).unwrap(),
        r#""fp8""#
    );
    assert_eq!(
        serde_json::to_string(&Quantization::Unknown).unwrap(),
        r#""unknown""#
    );
}

#[test]
fn test_provider_sort_strategy_serialization() {
    assert_eq!(
        serde_json::to_string(&ProviderSortStrategy::Price).unwrap(),
        r#""price""#
    );
    assert_eq!(
        serde_json::to_string(&ProviderSortStrategy::Throughput).unwrap(),
        r#""throughput""#
    );
    assert_eq!(
        serde_json::to_string(&ProviderSortStrategy::Latency).unwrap(),
        r#""latency""#
    );
}

#[test]
fn test_sort_partition_serialization() {
    assert_eq!(
        serde_json::to_string(&SortPartition::Model).unwrap(),
        r#""model""#
    );
    assert_eq!(
        serde_json::to_string(&SortPartition::None).unwrap(),
        r#""none""#
    );
}

#[test]
fn test_provider_sort_simple() {
    let sort = ProviderSort::Simple(ProviderSortStrategy::Latency);
    let json = serde_json::to_value(&sort).unwrap();
    assert_eq!(json, "latency");
}

#[test]
fn test_provider_sort_complex() {
    let sort = ProviderSort::Complex(
        ProviderSortConfig::new(ProviderSortStrategy::Price).partition(SortPartition::None),
    );
    let json = serde_json::to_value(&sort).unwrap();
    assert_eq!(json["by"], "price");
    assert_eq!(json["partition"], "none");
}

#[test]
fn test_provider_sort_complex_without_partition() {
    let sort = ProviderSort::Complex(ProviderSortConfig::new(ProviderSortStrategy::Throughput));
    let json = serde_json::to_value(&sort).unwrap();
    assert_eq!(json["by"], "throughput");
    assert!(json.get("partition").is_none());
}

#[test]
fn test_provider_sort_from_strategy() {
    let sort: ProviderSort = ProviderSortStrategy::Price.into();
    assert_eq!(sort, ProviderSort::Simple(ProviderSortStrategy::Price));
}

#[test]
fn test_provider_sort_from_config() {
    let config = ProviderSortConfig::new(ProviderSortStrategy::Latency);
    let sort: ProviderSort = config.into();
    match sort {
        ProviderSort::Complex(c) => assert_eq!(c.by, ProviderSortStrategy::Latency),
        _ => panic!("Expected Complex variant"),
    }
}

#[test]
fn test_percentile_thresholds_builder() {
    let thresholds = PercentileThresholds::new()
        .p50(10.0)
        .p75(25.0)
        .p90(50.0)
        .p99(100.0);

    assert_eq!(thresholds.p50, Some(10.0));
    assert_eq!(thresholds.p75, Some(25.0));
    assert_eq!(thresholds.p90, Some(50.0));
    assert_eq!(thresholds.p99, Some(100.0));
}

#[test]
fn test_percentile_thresholds_default() {
    let thresholds = PercentileThresholds::default();
    assert_eq!(thresholds.p50, None);
    assert_eq!(thresholds.p75, None);
    assert_eq!(thresholds.p90, None);
    assert_eq!(thresholds.p99, None);
}

#[test]
fn test_throughput_threshold_simple() {
    let threshold = ThroughputThreshold::Simple(50.0);
    let json = serde_json::to_value(&threshold).unwrap();
    assert_eq!(json, 50.0);
}

#[test]
fn test_throughput_threshold_percentile() {
    let threshold = ThroughputThreshold::Percentile(PercentileThresholds::new().p90(50.0));
    let json = serde_json::to_value(&threshold).unwrap();
    assert_eq!(json["p90"], 50.0);
}

#[test]
fn test_latency_threshold_simple() {
    let threshold = LatencyThreshold::Simple(0.5);
    let json = serde_json::to_value(&threshold).unwrap();
    assert_eq!(json, 0.5);
}

#[test]
fn test_latency_threshold_percentile() {
    let threshold = LatencyThreshold::Percentile(PercentileThresholds::new().p50(0.1).p99(1.0));
    let json = serde_json::to_value(&threshold).unwrap();
    assert_eq!(json["p50"], 0.1);
    assert_eq!(json["p99"], 1.0);
}

#[test]
fn test_max_price_builder() {
    let price = MaxPrice::new().prompt(0.001).completion(0.002);

    assert_eq!(price.prompt, Some(0.001));
    assert_eq!(price.completion, Some(0.002));
    assert_eq!(price.request, None);
    assert_eq!(price.image, None);
}

#[test]
fn test_max_price_all_fields() {
    let price = MaxPrice::new()
        .prompt(0.001)
        .completion(0.002)
        .request(0.01)
        .image(0.05);

    let json = serde_json::to_value(&price).unwrap();
    assert_eq!(json["prompt"], 0.001);
    assert_eq!(json["completion"], 0.002);
    assert_eq!(json["request"], 0.01);
    assert_eq!(json["image"], 0.05);
}

#[test]
fn test_max_price_default() {
    let price = MaxPrice::default();
    assert_eq!(price.prompt, None);
    assert_eq!(price.completion, None);
    assert_eq!(price.request, None);
    assert_eq!(price.image, None);
}

#[test]
fn test_provider_preferences_default() {
    let prefs = ProviderPreferences::default();
    assert!(prefs.order.is_none());
    assert!(prefs.only.is_none());
    assert!(prefs.ignore.is_none());
    assert!(prefs.allow_fallbacks.is_none());
    assert!(prefs.require_parameters.is_none());
    assert!(prefs.data_collection.is_none());
    assert!(prefs.zdr.is_none());
    assert!(prefs.sort.is_none());
    assert!(prefs.preferred_min_throughput.is_none());
    assert!(prefs.preferred_max_latency.is_none());
    assert!(prefs.max_price.is_none());
    assert!(prefs.quantizations.is_none());
}

#[test]
fn test_provider_preferences_order_with_fallbacks() {
    let prefs = ProviderPreferences::new()
        .order(["anthropic", "openai"])
        .allow_fallbacks(true);

    let json = prefs.to_json();
    let provider = &json["provider"];

    assert_eq!(provider["order"], json!(["anthropic", "openai"]));
    assert_eq!(provider["allow_fallbacks"], true);
}

#[test]
fn test_provider_preferences_only_allowlist() {
    let prefs = ProviderPreferences::new()
        .only(["azure", "together"])
        .allow_fallbacks(false);

    let json = prefs.to_json();
    let provider = &json["provider"];

    assert_eq!(provider["only"], json!(["azure", "together"]));
    assert_eq!(provider["allow_fallbacks"], false);
}

#[test]
fn test_provider_preferences_ignore() {
    let prefs = ProviderPreferences::new().ignore(["deepinfra"]);

    let json = prefs.to_json();
    let provider = &json["provider"];

    assert_eq!(provider["ignore"], json!(["deepinfra"]));
}

#[test]
fn test_provider_preferences_sort_latency() {
    let prefs = ProviderPreferences::new().sort(ProviderSortStrategy::Latency);

    let json = prefs.to_json();
    let provider = &json["provider"];

    assert_eq!(provider["sort"], "latency");
}

#[test]
fn test_provider_preferences_price_with_throughput() {
    let prefs = ProviderPreferences::new()
        .sort(ProviderSortStrategy::Price)
        .preferred_min_throughput(ThroughputThreshold::Percentile(
            PercentileThresholds::new().p90(50.0),
        ));

    let json = prefs.to_json();
    let provider = &json["provider"];

    assert_eq!(provider["sort"], "price");
    assert_eq!(provider["preferred_min_throughput"]["p90"], 50.0);
}

#[test]
fn test_provider_preferences_require_parameters() {
    let prefs = ProviderPreferences::new().require_parameters(true);

    let json = prefs.to_json();
    let provider = &json["provider"];

    assert_eq!(provider["require_parameters"], true);
}

#[test]
fn test_provider_preferences_data_policy_and_zdr() {
    let prefs = ProviderPreferences::new()
        .data_collection(DataCollection::Deny)
        .zdr(true);

    let json = prefs.to_json();
    let provider = &json["provider"];

    assert_eq!(provider["data_collection"], "deny");
    assert_eq!(provider["zdr"], true);
}

#[test]
fn test_provider_preferences_quantizations() {
    let prefs = ProviderPreferences::new().quantizations([Quantization::Int8, Quantization::Fp16]);

    let json = prefs.to_json();
    let provider = &json["provider"];

    assert_eq!(provider["quantizations"], json!(["int8", "fp16"]));
}

#[test]
fn test_provider_preferences_convenience_methods() {
    let prefs = ProviderPreferences::new().zero_data_retention().fastest();

    assert_eq!(prefs.zdr, Some(true));
    assert_eq!(
        prefs.sort,
        Some(ProviderSort::Simple(ProviderSortStrategy::Throughput))
    );

    let prefs2 = ProviderPreferences::new().cheapest();
    assert_eq!(
        prefs2.sort,
        Some(ProviderSort::Simple(ProviderSortStrategy::Price))
    );

    let prefs3 = ProviderPreferences::new().lowest_latency();
    assert_eq!(
        prefs3.sort,
        Some(ProviderSort::Simple(ProviderSortStrategy::Latency))
    );
}

#[test]
fn test_provider_preferences_serialization_skips_none() {
    let prefs = ProviderPreferences::new().sort(ProviderSortStrategy::Price);

    let json = serde_json::to_value(&prefs).unwrap();

    assert_eq!(json["sort"], "price");
    assert!(json.get("order").is_none());
    assert!(json.get("only").is_none());
    assert!(json.get("ignore").is_none());
    assert!(json.get("zdr").is_none());
}

#[test]
fn test_provider_preferences_deserialization() {
    let json = json!({
        "order": ["anthropic", "openai"],
        "sort": "throughput",
        "data_collection": "deny",
        "zdr": true,
        "quantizations": ["int8", "fp16"]
    });

    let prefs: ProviderPreferences = serde_json::from_value(json).unwrap();

    assert_eq!(
        prefs.order,
        Some(vec!["anthropic".to_string(), "openai".to_string()])
    );
    assert_eq!(
        prefs.sort,
        Some(ProviderSort::Simple(ProviderSortStrategy::Throughput))
    );
    assert_eq!(prefs.data_collection, Some(DataCollection::Deny));
    assert_eq!(prefs.zdr, Some(true));
    assert_eq!(
        prefs.quantizations,
        Some(vec![Quantization::Int8, Quantization::Fp16])
    );
}

#[test]
fn test_provider_preferences_deserialization_complex_sort() {
    let json = json!({
        "sort": {
            "by": "latency",
            "partition": "model"
        }
    });

    let prefs: ProviderPreferences = serde_json::from_value(json).unwrap();

    match prefs.sort {
        Some(ProviderSort::Complex(config)) => {
            assert_eq!(config.by, ProviderSortStrategy::Latency);
            assert_eq!(config.partition, Some(SortPartition::Model));
        }
        _ => panic!("Expected Complex sort variant"),
    }
}

#[test]
fn test_provider_preferences_full_integration() {
    let prefs = ProviderPreferences::new()
        .order(["anthropic", "openai"])
        .only(["anthropic", "openai", "google"])
        .sort(ProviderSortStrategy::Throughput)
        .data_collection(DataCollection::Deny)
        .zdr(true)
        .quantizations([Quantization::Int8])
        .allow_fallbacks(false);

    let json = prefs.to_json();

    assert!(json.get("provider").is_some());
    let provider = &json["provider"];
    assert_eq!(provider["order"], json!(["anthropic", "openai"]));
    assert_eq!(provider["only"], json!(["anthropic", "openai", "google"]));
    assert_eq!(provider["sort"], "throughput");
    assert_eq!(provider["data_collection"], "deny");
    assert_eq!(provider["zdr"], true);
    assert_eq!(provider["quantizations"], json!(["int8"]));
    assert_eq!(provider["allow_fallbacks"], false);
}

#[test]
fn test_provider_preferences_max_price() {
    let prefs =
        ProviderPreferences::new().max_price(MaxPrice::new().prompt(0.001).completion(0.002));

    let json = prefs.to_json();
    let provider = &json["provider"];

    assert_eq!(provider["max_price"]["prompt"], 0.001);
    assert_eq!(provider["max_price"]["completion"], 0.002);
}

#[test]
fn test_provider_preferences_preferred_max_latency() {
    let prefs = ProviderPreferences::new().preferred_max_latency(LatencyThreshold::Simple(0.5));

    let json = prefs.to_json();
    let provider = &json["provider"];

    assert_eq!(provider["preferred_max_latency"], 0.5);
}

#[test]
fn test_provider_preferences_empty_arrays() {
    let prefs = ProviderPreferences::new()
        .order(Vec::<String>::new())
        .quantizations(Vec::<Quantization>::new());

    let json = prefs.to_json();
    let provider = &json["provider"];

    assert_eq!(provider["order"], json!([]));
    assert_eq!(provider["quantizations"], json!([]));
}

// ================================================================
// File Support Tests
// ================================================================

#[test]
fn test_user_content_text_serialization() {
    let content = UserContent::Text {
        text: "Hello, world!".to_string(),
    };
    let json = serde_json::to_value(&content).unwrap();

    assert_eq!(json["type"], "text");
    assert_eq!(json["text"], "Hello, world!");
}

#[test]
fn test_user_content_image_url_serialization() {
    let content = UserContent::Image {
        image_url: ImageUrl {
            url: "https://example.com/image.png".to_string(),
            detail: None,
        },
    };
    let json = serde_json::to_value(&content).unwrap();

    assert_eq!(json["type"], "image_url");
    assert_eq!(json["image_url"]["url"], "https://example.com/image.png");
    assert!(json["image_url"].get("detail").is_none());
}

#[test]
fn test_user_content_image_url_with_detail_serialization() {
    let content = UserContent::Image {
        image_url: ImageUrl {
            url: "https://example.com/image.png".to_string(),
            detail: Some(ImageDetail::High),
        },
    };
    let json = serde_json::to_value(&content).unwrap();

    assert_eq!(json["type"], "image_url");
    assert_eq!(json["image_url"]["url"], "https://example.com/image.png");
    assert_eq!(json["image_url"]["detail"], "high");
}

#[test]
fn test_user_content_image_base64_serialization() {
    let content = UserContent::Image {
        image_url: ImageUrl {
            url: "data:image/png;base64,SGVsbG8=".to_string(),
            detail: Some(ImageDetail::Low),
        },
    };
    let json = serde_json::to_value(&content).unwrap();

    assert_eq!(json["type"], "image_url");
    assert_eq!(json["image_url"]["url"], "data:image/png;base64,SGVsbG8=");
    assert_eq!(json["image_url"]["detail"], "low");
}

#[test]
fn test_user_content_file_url_serialization() {
    let content = UserContent::File {
        file: FileData {
            file_data: Some("https://example.com/doc.pdf".to_string()),
            file_id: None,
            filename: Some("document.pdf".to_string()),
        },
    };
    let json = serde_json::to_value(&content).unwrap();

    assert_eq!(json["type"], "file");
    assert_eq!(json["file"]["file_data"], "https://example.com/doc.pdf");
    assert_eq!(json["file"]["filename"], "document.pdf");
}

#[test]
fn test_user_content_file_base64_serialization() {
    let content = UserContent::File {
        file: FileData {
            file_data: Some("data:application/pdf;base64,JVBERi0xLjQ=".to_string()),
            file_id: None,
            filename: Some("report.pdf".to_string()),
        },
    };
    let json = serde_json::to_value(&content).unwrap();

    assert_eq!(json["type"], "file");
    assert_eq!(
        json["file"]["file_data"],
        "data:application/pdf;base64,JVBERi0xLjQ="
    );
    assert_eq!(json["file"]["filename"], "report.pdf");
}

#[test]
fn test_user_content_text_deserialization() {
    let json = json!({
        "type": "text",
        "text": "Hello!"
    });

    let content: UserContent = serde_json::from_value(json).unwrap();
    assert_eq!(
        content,
        UserContent::Text {
            text: "Hello!".to_string()
        }
    );
}

#[test]
fn test_user_content_image_url_deserialization() {
    let json = json!({
        "type": "image_url",
        "image_url": {
            "url": "https://example.com/img.jpg",
            "detail": "high"
        }
    });

    let content: UserContent = serde_json::from_value(json).unwrap();
    match content {
        UserContent::Image { image_url } => {
            assert_eq!(image_url.url, "https://example.com/img.jpg");
            assert_eq!(image_url.detail, Some(ImageDetail::High));
        }
        _ => panic!("Expected Image variant"),
    }
}

#[test]
fn test_user_content_file_deserialization() {
    let json = json!({
        "type": "file",
        "file": {
            "filename": "doc.pdf",
            "file_data": "https://example.com/doc.pdf"
        }
    });

    let content: UserContent = serde_json::from_value(json).unwrap();
    match content {
        UserContent::File { file } => {
            assert_eq!(file.filename, Some("doc.pdf".to_string()));
            assert_eq!(
                file.file_data,
                Some("https://example.com/doc.pdf".to_string())
            );
        }
        _ => panic!("Expected File variant"),
    }
}

#[test]
fn test_message_user_with_text_serialization() {
    let message = Message::User {
        content: vec![UserContent::Text {
            text: "Hello".to_string(),
        }],
        name: None,
    };
    let json = serde_json::to_value(&message).unwrap();

    // Single text content should be serialized as a plain string
    assert_eq!(json["role"], "user");
    assert_eq!(json["content"], "Hello");
}

#[test]
fn test_message_user_with_mixed_content_serialization() {
    let message = Message::User {
        content: vec![
            UserContent::Text {
                text: "Check this image:".to_string(),
            },
            UserContent::Image {
                image_url: ImageUrl {
                    url: "https://example.com/img.png".to_string(),
                    detail: None,
                },
            },
        ],
        name: None,
    };
    let json = serde_json::to_value(&message).unwrap();

    assert_eq!(json["role"], "user");
    let content = json["content"].as_array().unwrap();
    assert_eq!(content.len(), 2);
    assert_eq!(content[0]["type"], "text");
    assert_eq!(content[1]["type"], "image_url");
}

#[test]
fn test_message_user_with_file_serialization() {
    let message = Message::User {
        content: vec![
            UserContent::Text {
                text: "Analyze this PDF:".to_string(),
            },
            UserContent::File {
                file: FileData {
                    file_data: Some("https://example.com/doc.pdf".to_string()),
                    file_id: None,
                    filename: Some("document.pdf".to_string()),
                },
            },
        ],
        name: None,
    };
    let json = serde_json::to_value(&message).unwrap();

    assert_eq!(json["role"], "user");
    let content = json["content"].as_array().unwrap();
    assert_eq!(content.len(), 2);
    assert_eq!(content[0]["type"], "text");
    assert_eq!(content[1]["type"], "file");
    assert_eq!(
        content[1]["file"]["file_data"],
        "https://example.com/doc.pdf"
    );
}

#[test]
fn test_user_content_from_rig_text() {
    let rig_content = message::UserContent::Text(message::Text::new("Hello".to_string()));
    let openrouter_content: UserContent = user_content_to_openai(rig_content).unwrap();

    assert_eq!(
        openrouter_content,
        UserContent::Text {
            text: "Hello".to_string()
        }
    );
}

#[test]
fn test_user_content_from_rig_image_url() {
    let rig_content = message::UserContent::Image(message::Image {
        data: DocumentSourceKind::Url("https://example.com/img.png".to_string()),
        media_type: Some(message::ImageMediaType::PNG),
        detail: Some(ImageDetail::High),
        additional_params: None,
    });
    let openrouter_content: UserContent = user_content_to_openai(rig_content).unwrap();

    match openrouter_content {
        UserContent::Image { image_url } => {
            assert_eq!(image_url.url, "https://example.com/img.png");
            assert_eq!(image_url.detail, Some(ImageDetail::High));
        }
        _ => panic!("Expected Image variant"),
    }
}

#[test]
fn test_user_content_from_rig_image_base64() {
    let rig_content = message::UserContent::Image(message::Image {
        data: DocumentSourceKind::Base64("SGVsbG8=".to_string()),
        media_type: Some(message::ImageMediaType::JPEG),
        detail: Some(ImageDetail::Low),
        additional_params: None,
    });
    let openrouter_content: UserContent = user_content_to_openai(rig_content).unwrap();

    match openrouter_content {
        UserContent::Image { image_url } => {
            assert_eq!(image_url.url, "data:image/jpeg;base64,SGVsbG8=");
            assert_eq!(image_url.detail, Some(ImageDetail::Low));
        }
        _ => panic!("Expected Image variant"),
    }
}

#[test]
fn test_user_content_from_rig_document_url() {
    let rig_content = message::UserContent::Document(message::Document {
        data: DocumentSourceKind::Url("https://example.com/doc.pdf".to_string()),
        media_type: Some(DocumentMediaType::PDF),
        additional_params: None,
    });
    let openrouter_content: UserContent = user_content_to_openai(rig_content).unwrap();

    match openrouter_content {
        UserContent::File { file } => {
            assert_eq!(
                file.file_data,
                Some("https://example.com/doc.pdf".to_string())
            );
            assert_eq!(file.filename, Some("document.pdf".to_string()));
        }
        _ => panic!("Expected File variant"),
    }
}

#[test]
fn test_user_content_from_rig_document_base64() {
    let rig_content = message::UserContent::Document(message::Document {
        data: DocumentSourceKind::Base64("JVBERi0xLjQ=".to_string()),
        media_type: Some(DocumentMediaType::PDF),
        additional_params: None,
    });
    let openrouter_content: UserContent = user_content_to_openai(rig_content).unwrap();

    match openrouter_content {
        UserContent::File { file } => {
            assert_eq!(
                file.file_data,
                Some("data:application/pdf;base64,JVBERi0xLjQ=".to_string())
            );
            assert_eq!(file.filename, Some("document.pdf".to_string()));
        }
        _ => panic!("Expected File variant"),
    }
}

#[test]
fn test_user_content_from_rig_document_file_id() {
    let rig_content = message::UserContent::Document(message::Document {
        data: DocumentSourceKind::FileId("file_abc".to_string()),
        media_type: None,
        additional_params: None,
    });

    let result: Result<UserContent, _> = user_content_to_openai(rig_content);
    assert!(matches!(
        result,
        Err(message::MessageError::ConversionError(message))
            if message.contains("Provider file IDs are not supported")
    ));
}

#[test]
fn test_openai_file_id_content_round_trips_through_rig_to_openrouter_error() {
    let openai_content = openai::UserContent::File {
        file: openai::FileData {
            file_data: None,
            file_id: Some("file_abc".to_string()),
            filename: None,
        },
    };
    let rig_content: message::UserContent = openai_content.into();

    let result: Result<UserContent, _> = user_content_to_openai(rig_content);
    assert!(matches!(
        result,
        Err(message::MessageError::ConversionError(message))
            if message.contains("Provider file IDs are not supported")
    ));
}

#[test]
fn test_user_content_from_rig_document_string_becomes_text() {
    let rig_content = message::UserContent::Document(message::Document {
        data: DocumentSourceKind::String("Plain text document content".to_string()),
        media_type: Some(DocumentMediaType::TXT),
        additional_params: None,
    });
    let openrouter_content: UserContent = user_content_to_openai(rig_content).unwrap();

    assert_eq!(
        openrouter_content,
        UserContent::Text {
            text: "Plain text document content".to_string()
        }
    );
}

#[test]
fn test_completion_response_with_reasoning_details_maps_to_typed_reasoning() {
    let json = json!({
        "id": "resp_123",
        "object": "chat.completion",
        "created": 1,
        "model": "openrouter/test-model",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "hello",
                "reasoning": null,
                "reasoning_details": [
                    {"type":"reasoning.summary","id":"rs_1","summary":"s1"},
                    {"type":"reasoning.text","id":"rs_1","text":"t1","signature":"sig_1"},
                    {"type":"reasoning.encrypted","id":"rs_1","data":"enc_1"}
                ],
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"}
                }]
            }
        }]
    });

    let response: CompletionResponse = serde_json::from_value(json).unwrap();
    let converted = response.normalize(PROVIDER_NAME).unwrap();
    let items: Vec<completion::AssistantContent> = converted.choice.into_iter().collect();

    assert_eq!(items.len(), 3, "reasoning, text, then tool call");
    assert!(matches!(
        &items[0],
        completion::AssistantContent::Reasoning(message::Reasoning { id: Some(id), content })
            if id == "rs_1" && content.len() == 3
    ));
    assert!(matches!(
        &items[1],
        completion::AssistantContent::Text(text) if text.text == "hello"
    ));
    assert!(matches!(
        &items[2],
        completion::AssistantContent::ToolCall(call) if call.function.name == "lookup"
    ));
}

/// Encrypted `reasoning_details` on the streaming wire must reach the
/// aggregated choice and replay on the next turn.
///
/// The SSE below mirrors the recorded OpenRouter shape
/// (`tests/cassettes/openrouter/streaming_tools/raw_stream_decorates_reasoning_tool_call_metadata.yaml`):
/// the detail arrives with `reasoning: null` and an `rs_*` id of its own,
/// one chunk *before* the `call_*` tool call opens. Routed through
/// tool-call decoration those two id namespaces never match, so the blob
/// was dropped on every streaming turn while the non-streaming path kept
/// it.
#[tokio::test]
async fn streaming_encrypted_reasoning_detail_reaches_the_choice_and_replays() {
    use crate::client::CompletionClient;
    use crate::completion::CompletionModel as _;
    use crate::providers::internal::openai_chat_completions_compatible::test_support::sse_bytes_from_data_lines;
    use crate::streaming::StreamedAssistantContent;
    use crate::test_utils::MockStreamingClient;
    use futures::StreamExt;

    let http_client = MockStreamingClient {
        sse_bytes: sse_bytes_from_data_lines([
            r#"{"id":"chatcmpl-1","model":"openai/o4-mini","choices":[{"index":0,"delta":{"role":"assistant","content":"","reasoning":null,"reasoning_details":[{"type":"reasoning.encrypted","id":"rs_1","format":"openai-responses-v1","index":0,"data":"enc_blob"}]},"finish_reason":null}]}"#,
            r#"{"id":"chatcmpl-1","model":"openai/o4-mini","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}"#,
            r#"{"id":"chatcmpl-1","model":"openai/o4-mini","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":\"Tokyo\"}"}}]},"finish_reason":null}]}"#,
            r#"{"id":"chatcmpl-1","model":"openai/o4-mini","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
            "[DONE]",
        ]),
    };

    let client = crate::providers::openrouter::Client::builder()
        .api_key("dummy-key")
        .http_client(http_client)
        .build()
        .expect("client should build");
    let model = client.completion_model("openai/o4-mini");
    let request = model.completion_request("weather?").build();
    let mut stream = model.stream(request).await.expect("stream should start");

    let mut events: Vec<&'static str> = Vec::new();
    let mut streamed_tool_calls = Vec::new();
    while let Some(chunk) = stream.next().await {
        match chunk.expect("stream item should be ok") {
            StreamedAssistantContent::Reasoning { reasoning, .. } => {
                assert_eq!(reasoning.id.as_deref(), Some("rs_1"));
                assert!(matches!(
                    reasoning.content.first(),
                    Some(message::ReasoningContent::Encrypted(data)) if data == "enc_blob"
                ));
                events.push("reasoning");
            }
            StreamedAssistantContent::ToolCall { tool_call, .. } => {
                streamed_tool_calls.push(tool_call);
                events.push("tool_call");
            }
            _ => {}
        }
    }

    // Wire order: the reasoning block precedes the tool call it was
    // recorded before.
    assert_eq!(events, vec!["reasoning", "tool_call"]);

    // The tool call is *not* where the blob lives: decoration by the
    // detail's own id could never match the call's id.
    let tool_call = streamed_tool_calls.first().expect("streamed tool call");
    assert_eq!(tool_call.id, "call_1");
    assert!(tool_call.signature.is_none());
    assert!(tool_call.additional_params.is_none());

    // (a) the encrypted block reaches the aggregated choice ...
    let choice: Vec<message::AssistantContent> = stream.choice.clone().into_iter().collect();
    assert!(
        choice.iter().any(|content| matches!(
            content,
            message::AssistantContent::Reasoning(message::Reasoning { id: Some(id), content })
                if id == "rs_1"
                    && matches!(
                        content.first(),
                        Some(message::ReasoningContent::Encrypted(data)) if data == "enc_blob"
                    )
        )),
        "encrypted reasoning must reach the aggregated choice: {choice:#?}"
    );

    // ... and (b) replays into the next turn's request messages.
    let messages =
        assistant_contents_to_messages(stream.choice.clone()).expect("history conversion");
    let Message::Assistant {
        reasoning_details, ..
    } = messages.first().expect("assistant message")
    else {
        panic!("Expected assistant message");
    };
    assert!(
        reasoning_details.iter().any(|detail| matches!(
            detail,
            ReasoningDetails::Encrypted { id: Some(id), data, .. }
                if id == "rs_1" && data == "enc_blob"
        )),
        "encrypted reasoning must replay as a reasoning_details entry: {reasoning_details:#?}"
    );
}

/// Anthropic-routed OpenRouter streams put the replay-required signature
/// in a final `reasoning.text` detail with no text of its own. The shared
/// `delta.reasoning` field carries the preceding plaintext, so the detail
/// must close and sign that same block before the tool call is emitted.
#[tokio::test]
async fn streaming_anthropic_reasoning_signature_reaches_choice_and_replays() {
    use crate::client::CompletionClient;
    use crate::completion::CompletionModel as _;
    use crate::providers::internal::openai_chat_completions_compatible::test_support::sse_bytes_from_data_lines;
    use crate::test_utils::MockStreamingClient;
    use futures::StreamExt;

    let http_client = MockStreamingClient {
        sse_bytes: sse_bytes_from_data_lines([
            r#"{"id":"chatcmpl-1","model":"anthropic/claude-haiku-4.5","choices":[{"index":0,"delta":{"role":"assistant","content":"","reasoning":"think first","reasoning_details":[{"type":"reasoning.text","format":"anthropic-claude-v1","index":0,"text":"think first"}]},"finish_reason":null,"native_finish_reason":null}]}"#,
            r#"{"id":"chatcmpl-1","model":"anthropic/claude-haiku-4.5","choices":[{"index":0,"delta":{"role":"assistant","content":"","reasoning_details":[{"type":"reasoning.text","format":"anthropic-claude-v1","index":0,"signature":"sig-live-shape"}]},"finish_reason":null,"native_finish_reason":null}]}"#,
            r#"{"id":"chatcmpl-1","model":"anthropic/claude-haiku-4.5","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"toolu_1","type":"function","function":{"name":"lookup","arguments":"{}"}}]},"finish_reason":null,"native_finish_reason":null}]}"#,
            r#"{"id":"chatcmpl-1","model":"anthropic/claude-haiku-4.5","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls","native_finish_reason":"tool_use"}]}"#,
            "[DONE]",
        ]),
    };

    let client = crate::providers::openrouter::Client::builder()
        .api_key("dummy-key")
        .http_client(http_client)
        .build()
        .expect("client should build");
    let model = client.completion_model("anthropic/claude-haiku-4.5");
    let request = model.completion_request("lookup").build();
    let mut stream = model.stream(request).await.expect("stream should start");
    while let Some(item) = stream.next().await {
        item.expect("signed reasoning stream item");
    }

    let choice = stream.choice.clone().into_iter().collect::<Vec<_>>();
    assert!(matches!(
        choice.first(),
        Some(message::AssistantContent::Reasoning(message::Reasoning { content, .. }))
            if matches!(
                content.first(),
                Some(message::ReasoningContent::Text { text, signature: Some(signature) })
                    if text == "think first" && signature == "sig-live-shape"
            )
    ));
    assert!(matches!(
        choice.get(1),
        Some(message::AssistantContent::ToolCall(call)) if call.function.name == "lookup"
    ));

    let messages = assistant_contents_to_messages(choice).expect("history conversion");
    let Message::Assistant {
        reasoning_details, ..
    } = messages.first().expect("assistant message")
    else {
        panic!("expected assistant history message");
    };
    assert!(matches!(
        reasoning_details.first(),
        Some(ReasoningDetails::Text {
            text: Some(text),
            signature: Some(signature),
            ..
        }) if text == "think first" && signature == "sig-live-shape"
    ));
}

/// An id-less encrypted detail must not clobber the reasoning text
/// accumulating under the wire's constant minted key.
///
/// The shared compat adapter keys `reasoning` text deltas by
/// `Minted { Reasoning, 0 }`; the id-less encrypted detail arrives as a
/// whole block while that part is still open. Keyed identically, the
/// whole block would *restate* — replace — the open text part
/// (pre-fix, all accumulated reasoning text was lost). Keyed as
/// `EncryptedReasoning` it is a sibling: both parts reach the
/// aggregated choice.
#[tokio::test]
async fn id_less_encrypted_detail_does_not_replace_open_reasoning_text() {
    use crate::client::CompletionClient;
    use crate::completion::CompletionModel as _;
    use crate::providers::internal::openai_chat_completions_compatible::test_support::sse_bytes_from_data_lines;
    use crate::test_utils::MockStreamingClient;
    use futures::StreamExt;

    let http_client = MockStreamingClient {
        sse_bytes: sse_bytes_from_data_lines([
            r#"{"id":"chatcmpl-1","model":"openai/o4-mini","choices":[{"index":0,"delta":{"role":"assistant","content":"","reasoning":"deep "},"finish_reason":null}]}"#,
            r#"{"id":"chatcmpl-1","model":"openai/o4-mini","choices":[{"index":0,"delta":{"reasoning":"thought"},"finish_reason":null}]}"#,
            r#"{"id":"chatcmpl-1","model":"openai/o4-mini","choices":[{"index":0,"delta":{"reasoning":null,"reasoning_details":[{"type":"reasoning.encrypted","id":null,"format":"openai-responses-v1","index":0,"data":"enc_blob"}]},"finish_reason":null}]}"#,
            r#"{"id":"chatcmpl-1","model":"openai/o4-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
            "[DONE]",
        ]),
    };

    let client = crate::providers::openrouter::Client::builder()
        .api_key("dummy-key")
        .http_client(http_client)
        .build()
        .expect("client should build");
    let model = client.completion_model("openai/o4-mini");
    let request = model.completion_request("weather?").build();
    let mut stream = model.stream(request).await.expect("stream should start");
    while stream.next().await.is_some() {}

    let choice: Vec<message::AssistantContent> = stream.choice.clone().into_iter().collect();
    assert!(
        choice.iter().any(|content| matches!(
            content,
            message::AssistantContent::Reasoning(message::Reasoning { content, .. })
                if matches!(
                    content.first(),
                    Some(message::ReasoningContent::Text { text, .. }) if text == "deep thought"
                )
        )),
        "the accumulated reasoning text must survive the encrypted detail: {choice:#?}"
    );
    assert!(
        choice.iter().any(|content| matches!(
            content,
            message::AssistantContent::Reasoning(message::Reasoning { id: None, content })
                if matches!(
                    content.first(),
                    Some(message::ReasoningContent::Encrypted(data)) if data == "enc_blob"
                )
        )),
        "the encrypted blob must reach the choice as its own part: {choice:#?}"
    );
}

/// An encrypted detail the wire sends without an id streams under its
/// own minted `EncryptedReasoning` key; it must still replay, and with
/// a null wire id rather than an empty string.
#[test]
fn id_less_encrypted_reasoning_replays_with_a_null_wire_id() {
    use crate::providers::openai::completion::OpenAICompatibleProvider as _;

    let detail = json!({
        "type": "reasoning.encrypted",
        "id": null,
        "format": null,
        "index": 0,
        "data": "enc_blob",
    });
    let (id, provider_id, content) = OpenRouterExt
        .streaming_detail_reasoning(&detail)
        .expect("encrypted detail should map to reasoning");
    // 84a43e9e #4, closed: an id-less detail keys accumulation by a
    // minted (opaque) key and carries NO durable handle — a fabricated
    // "wire" empty id is unrepresentable, so no serializer needs an
    // empty-string filter.
    assert!(
        id.is_minted(),
        "id-less details key by a minted key: {id:?}"
    );
    assert!(
        provider_id.is_none(),
        "absence is None, never a fabricated id"
    );
    assert!(matches!(
        content,
        message::ReasoningContent::Encrypted(ref data) if data == "enc_blob"
    ));

    let messages = assistant_contents_to_messages(vec![message::AssistantContent::Reasoning(
        message::Reasoning {
            id: provider_id.map(|id| id.into_string()),
            content: vec![content],
        },
    )])
    .unwrap();
    let Message::Assistant {
        reasoning_details, ..
    } = messages.first().expect("assistant message")
    else {
        panic!("Expected assistant message");
    };
    assert!(matches!(
        reasoning_details.first(),
        Some(ReasoningDetails::Encrypted { id: None, data, .. }) if data == "enc_blob"
    ));
}

#[test]
fn test_assistant_reasoning_emits_openrouter_reasoning_details() {
    let reasoning = message::Reasoning {
        id: Some("rs_2".to_string()),
        content: vec![
            message::ReasoningContent::Text {
                text: "step".to_string(),
                signature: Some("sig_step".to_string()),
            },
            message::ReasoningContent::Summary("summary".to_string()),
            message::ReasoningContent::Encrypted("enc_blob".to_string()),
        ],
    };

    let messages =
        assistant_contents_to_messages(vec![message::AssistantContent::Reasoning(reasoning)])
            .unwrap();
    let Message::Assistant {
        reasoning,
        reasoning_details,
        ..
    } = messages.first().expect("assistant message")
    else {
        panic!("Expected assistant message");
    };

    assert!(reasoning.is_none());
    assert_eq!(reasoning_details.len(), 3);
    assert!(matches!(
        reasoning_details.first(),
        Some(ReasoningDetails::Text {
            id: Some(id),
            text: Some(text),
            signature: Some(signature),
            ..
        }) if id == "rs_2" && text == "step" && signature == "sig_step"
    ));
}

#[test]
fn test_tool_call_signature_without_params_uses_wire_id_for_encrypted_detail() {
    let tool_call = message::ToolCall::from_wire(
        "call_wire",
        message::ToolFunction {
            name: "lookup".to_string(),
            arguments: json!({}),
        },
    )
    .with_signature(Some("sig-data".to_string()));

    let messages =
        assistant_contents_to_messages(vec![message::AssistantContent::ToolCall(tool_call)])
            .unwrap();

    let Message::Assistant {
        reasoning_details, ..
    } = messages.first().expect("assistant message")
    else {
        panic!("Expected assistant message");
    };

    assert!(matches!(
        reasoning_details.first(),
        Some(ReasoningDetails::Encrypted {
            id: Some(id),
            data,
            ..
        }) if id == "call_wire" && data == "sig-data"
    ));
}

#[test]
fn test_tool_call_minimal_params_fall_back_to_wire_id() {
    let tool_call = message::ToolCall::from_wire(
        "call_wire",
        message::ToolFunction {
            name: "lookup".to_string(),
            arguments: json!({}),
        },
    )
    .with_signature(Some("sig-data".to_string()))
    // Minimal params carrying only a format: the detail id must
    // still correlate with the wire tool-call id.
    .with_additional_params(Some(json!({"format": "anthropic"})));

    let messages =
        assistant_contents_to_messages(vec![message::AssistantContent::ToolCall(tool_call)])
            .unwrap();

    let Message::Assistant {
        reasoning_details, ..
    } = messages.first().expect("assistant message")
    else {
        panic!("Expected assistant message");
    };

    assert!(matches!(
        reasoning_details.first(),
        Some(ReasoningDetails::Encrypted {
            id: Some(id),
            format,
            data,
            ..
        }) if id == "call_wire" && data == "sig-data" && format.as_deref() == Some("anthropic")
    ));
}

#[test]
fn test_assistant_redacted_reasoning_emits_encrypted_detail_not_text() {
    let reasoning = message::Reasoning {
        id: Some("rs_redacted".to_string()),
        content: vec![message::ReasoningContent::Redacted {
            data: "opaque-redacted-data".to_string(),
        }],
    };

    let messages =
        assistant_contents_to_messages(vec![message::AssistantContent::Reasoning(reasoning)])
            .unwrap();

    let Message::Assistant {
        reasoning_details,
        reasoning,
        ..
    } = messages.first().expect("assistant message")
    else {
        panic!("Expected assistant message");
    };

    assert!(reasoning.is_none());
    assert_eq!(reasoning_details.len(), 1);
    assert!(matches!(
        reasoning_details.first(),
        Some(ReasoningDetails::Encrypted {
            id: Some(id),
            data,
            ..
        }) if id == "rs_redacted" && data == "opaque-redacted-data"
    ));
}

#[test]
fn test_completion_response_reasoning_details_respects_index_ordering() {
    let json = json!({
        "id": "resp_ordering",
        "object": "chat.completion",
        "created": 1,
        "model": "openrouter/test-model",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "hello",
                "reasoning": null,
                "reasoning_details": [
                    {"type":"reasoning.summary","id":"rs_order","index":1,"summary":"second"},
                    {"type":"reasoning.summary","id":"rs_order","index":0,"summary":"first"}
                ]
            }
        }]
    });

    let response: CompletionResponse = serde_json::from_value(json).unwrap();
    let converted = response.normalize(PROVIDER_NAME).unwrap();
    let items: Vec<completion::AssistantContent> = converted.choice.into_iter().collect();
    let reasoning_blocks: Vec<_> = items
        .into_iter()
        .filter_map(|item| match item {
            completion::AssistantContent::Reasoning(reasoning) => Some(reasoning),
            _ => None,
        })
        .collect();

    assert_eq!(reasoning_blocks.len(), 1);
    assert_eq!(reasoning_blocks[0].id.as_deref(), Some("rs_order"));
    assert_eq!(
        reasoning_blocks[0].content,
        vec![
            message::ReasoningContent::Summary("first".to_string()),
            message::ReasoningContent::Summary("second".to_string()),
        ]
    );
}

#[test]
fn test_user_content_from_rig_image_missing_media_type_error() {
    let rig_content = message::UserContent::Image(message::Image {
        data: DocumentSourceKind::Base64("SGVsbG8=".to_string()),
        media_type: None, // Missing media type
        detail: None,
        additional_params: None,
    });
    let result: Result<UserContent, _> = user_content_to_openai(rig_content);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("media type required"));
}

#[test]
fn test_user_content_from_rig_image_raw_bytes_error() {
    let rig_content = message::UserContent::Image(message::Image {
        data: DocumentSourceKind::Raw(vec![1, 2, 3]),
        media_type: Some(message::ImageMediaType::PNG),
        detail: None,
        additional_params: None,
    });
    let result: Result<UserContent, _> = user_content_to_openai(rig_content);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("base64"));
}

#[test]
fn test_user_content_from_rig_video_url() {
    let rig_content = message::UserContent::Video(message::Video {
        data: DocumentSourceKind::Url("https://example.com/video.mp4".to_string()),
        media_type: Some(message::VideoMediaType::MP4),
        additional_params: None,
    });
    let openrouter_content: UserContent = user_content_to_openai(rig_content).unwrap();

    match openrouter_content {
        UserContent::Video { video_url } => {
            assert_eq!(video_url.url, "https://example.com/video.mp4");
        }
        _ => panic!("Expected Video variant"),
    }
}

#[test]
fn test_user_content_from_rig_video_base64() {
    let rig_content = message::UserContent::Video(message::Video {
        data: DocumentSourceKind::Base64("SGVsbG8=".to_string()),
        media_type: Some(message::VideoMediaType::MP4),
        additional_params: None,
    });
    let openrouter_content: UserContent = user_content_to_openai(rig_content).unwrap();

    match openrouter_content {
        UserContent::Video { video_url } => {
            assert_eq!(video_url.url, "data:video/mp4;base64,SGVsbG8=");
        }
        _ => panic!("Expected Video variant"),
    }
}

#[test]
fn test_user_content_from_rig_video_base64_missing_media_type_error() {
    let rig_content = message::UserContent::Video(message::Video {
        data: DocumentSourceKind::Base64("SGVsbG8=".to_string()),
        media_type: None,
        additional_params: None,
    });
    let result: Result<UserContent, _> = user_content_to_openai(rig_content);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("media type"));
}

#[test]
fn test_user_content_from_rig_video_raw_bytes_error() {
    let rig_content = message::UserContent::Video(message::Video {
        data: DocumentSourceKind::Raw(vec![1, 2, 3]),
        media_type: Some(message::VideoMediaType::MP4),
        additional_params: None,
    });
    let result: Result<UserContent, _> = user_content_to_openai(rig_content);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("base64"));
}

#[test]
fn test_user_content_from_rig_audio_base64() {
    let rig_content = message::UserContent::Audio(message::Audio {
        data: DocumentSourceKind::Base64("audiodata".to_string()),
        media_type: Some(message::AudioMediaType::MP3),
        additional_params: None,
    });
    let openrouter_content: UserContent = user_content_to_openai(rig_content).unwrap();

    match openrouter_content {
        UserContent::Audio { input_audio } => {
            assert_eq!(input_audio.data, "audiodata");
            assert_eq!(input_audio.format, message::AudioMediaType::MP3);
        }
        _ => panic!("Expected Audio variant"),
    }
}

#[test]
fn test_user_content_from_rig_audio_missing_media_type_error() {
    let rig_content = message::UserContent::Audio(message::Audio {
        data: DocumentSourceKind::Base64("audiodata".to_string()),
        media_type: None, // missing media type
        additional_params: None,
    });
    let result: Result<UserContent, _> = user_content_to_openai(rig_content);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("media type required"));
}

#[test]
fn test_user_content_from_rig_audio_url_error() {
    let rig_content = message::UserContent::Audio(message::Audio {
        data: DocumentSourceKind::Url("https://example.com/audio.wav".to_string()),
        media_type: Some(message::AudioMediaType::WAV),
        additional_params: None,
    });
    let result: Result<UserContent, _> = user_content_to_openai(rig_content);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("base64"));
}

#[test]
fn test_user_content_from_rig_audio_raw_bytes_error() {
    let rig_content = message::UserContent::Audio(message::Audio {
        data: DocumentSourceKind::Raw(vec![1, 2, 3]),
        media_type: Some(message::AudioMediaType::WAV),
        additional_params: None,
    });
    let result: Result<UserContent, _> = user_content_to_openai(rig_content);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("base64"));
}

#[test]
fn test_user_content_from_rig_video_file_id_error() {
    let rig_content = message::UserContent::Video(message::Video {
        data: DocumentSourceKind::FileId("file-123".to_string()),
        media_type: Some(message::VideoMediaType::MP4),
        additional_params: None,
    });
    let result: Result<UserContent, _> = user_content_to_openai(rig_content);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        err.to_string()
            .contains("File IDs are not supported for video")
    );
}

#[test]
fn test_user_content_from_rig_audio_file_id_error() {
    let rig_content = message::UserContent::Audio(message::Audio {
        data: DocumentSourceKind::FileId("file-123".to_string()),
        media_type: Some(message::AudioMediaType::MP3),
        additional_params: None,
    });
    let result: Result<UserContent, _> = user_content_to_openai(rig_content);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        err.to_string()
            .contains("File IDs are not supported for audio")
    );
}

#[test]
fn test_video_helper_converts_to_data_uri() {
    // `UserContent::video(..)` carries base64 data and should become a
    // `video_url` data URI.
    let rig_content = message::UserContent::video("SGVsbG8=", Some(message::VideoMediaType::MP4));
    let openrouter_content: UserContent = user_content_to_openai(rig_content).unwrap();

    match openrouter_content {
        UserContent::Video { video_url } => {
            assert_eq!(video_url.url, "data:video/mp4;base64,SGVsbG8=");
        }
        _ => panic!("Expected Video variant"),
    }
}

#[test]
fn test_video_url_helper_passes_url_through() {
    // `UserContent::video_url(..)` passes the URL through unchanged and does
    // not require a media type.
    let rig_content = message::UserContent::video_url("https://example.com/video.mp4", None);
    let openrouter_content: UserContent = user_content_to_openai(rig_content).unwrap();

    match openrouter_content {
        UserContent::Video { video_url } => {
            assert_eq!(video_url.url, "https://example.com/video.mp4");
        }
        _ => panic!("Expected Video variant"),
    }
}

#[test]
fn test_video_raw_helper_errors() {
    // `UserContent::video_raw(..)` carries raw bytes, which OpenRouter cannot
    // accept; the caller must base64-encode first.
    let rig_content =
        message::UserContent::video_raw(vec![1, 2, 3], Some(message::VideoMediaType::MP4));
    let result: Result<UserContent, _> = user_content_to_openai(rig_content);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("base64"));
}

#[test]
fn test_message_conversion_with_pdf() {
    let rig_message = message::Message::User {
        content: vec![
            message::UserContent::Text(message::Text::new("Summarize this document".to_string())),
            message::UserContent::Document(message::Document {
                data: DocumentSourceKind::Url("https://example.com/paper.pdf".to_string()),
                media_type: Some(DocumentMediaType::PDF),
                additional_params: None,
            }),
        ],
    };

    let openrouter_messages: Vec<Message> = messages_from_rig_message(rig_message).unwrap();
    assert_eq!(openrouter_messages.len(), 1);

    match &openrouter_messages[0] {
        Message::User { content, .. } => {
            assert_eq!(content.len(), 2);

            // First should be text
            match content.first() {
                Some(UserContent::Text { text, .. }) => {
                    assert_eq!(text, "Summarize this document");
                }
                _ => panic!("Expected Text"),
            }
        }
        _ => panic!("Expected User message"),
    }
}

#[test]
fn test_user_content_from_string() {
    let content: UserContent = "Hello".into();
    assert_eq!(
        content,
        UserContent::Text {
            text: "Hello".to_string()
        }
    );

    let content: UserContent = String::from("World").into();
    assert_eq!(
        content,
        UserContent::Text {
            text: "World".to_string()
        }
    );
}

#[test]
fn test_completion_response_reasoning_details_with_multiple_ids_stay_separate() {
    let json = json!({
        "id": "resp_multi_id",
        "object": "chat.completion",
        "created": 1,
        "model": "openrouter/test-model",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "hello",
                "reasoning": null,
                "reasoning_details": [
                    {"type":"reasoning.summary","id":"rs_a","summary":"a1"},
                    {"type":"reasoning.summary","id":"rs_b","summary":"b1"},
                    {"type":"reasoning.summary","id":"rs_a","summary":"a2"}
                ]
            }
        }]
    });

    let response: CompletionResponse = serde_json::from_value(json).unwrap();
    let converted = response.normalize(PROVIDER_NAME).unwrap();
    let items: Vec<completion::AssistantContent> = converted.choice.into_iter().collect();
    let reasoning_blocks: Vec<_> = items
        .into_iter()
        .filter_map(|item| match item {
            completion::AssistantContent::Reasoning(reasoning) => Some(reasoning),
            _ => None,
        })
        .collect();

    assert_eq!(reasoning_blocks.len(), 2);
    assert_eq!(reasoning_blocks[0].id.as_deref(), Some("rs_a"));
    assert_eq!(
        reasoning_blocks[0].content,
        vec![
            message::ReasoningContent::Summary("a1".to_string()),
            message::ReasoningContent::Summary("a2".to_string()),
        ]
    );
    assert_eq!(reasoning_blocks[1].id.as_deref(), Some("rs_b"));
    assert_eq!(
        reasoning_blocks[1].content,
        vec![message::ReasoningContent::Summary("b1".to_string())]
    );
}

#[test]
fn test_user_content_audio_serialization() {
    let content = UserContent::Audio {
        input_audio: openai::InputAudio {
            data: "SGVsbG8=".to_string(),
            format: AudioMediaType::WAV,
        },
    };
    let json = serde_json::to_value(&content).unwrap();

    assert_eq!(json["type"], "input_audio");
    assert_eq!(json["input_audio"]["data"], "SGVsbG8=");
    assert_eq!(json["input_audio"]["format"], "wav");
}

#[test]
fn test_user_content_audio_deserialization() {
    let json = json!({
        "type": "input_audio",
        "input_audio": {
            "data": "SGVsbG8=",
            "format": "wav"
        }
    });

    let content: UserContent = serde_json::from_value(json).unwrap();
    match content {
        UserContent::Audio { input_audio } => {
            assert_eq!(input_audio.data, "SGVsbG8=");
            assert_eq!(input_audio.format, AudioMediaType::WAV);
        }
        _ => panic!("Expected Audio variant"),
    }
}

#[test]
fn test_message_user_with_audio_serialization() {
    let msg = Message::User {
        content: vec![
            UserContent::Text {
                text: "Transcribe this audio:".to_string(),
            },
            UserContent::Audio {
                input_audio: openai::InputAudio {
                    data: "SGVsbG8=".to_string(),
                    format: AudioMediaType::MP3,
                },
            },
        ],
        name: None,
    };
    let json = serde_json::to_value(&msg).unwrap();

    assert_eq!(json["role"], "user");
    let content = json["content"].as_array().unwrap();
    assert_eq!(content.len(), 2);
    assert_eq!(content[0]["type"], "text");
    assert_eq!(content[1]["type"], "input_audio");
    assert_eq!(content[1]["input_audio"]["data"], "SGVsbG8=");
    assert_eq!(content[1]["input_audio"]["format"], "mp3");
}

#[test]
fn test_user_content_video_url_serialization() {
    let content = UserContent::Video {
        video_url: VideoUrl {
            url: "https://example.com/video.mp4".to_string(),
        },
    };
    let json = serde_json::to_value(&content).unwrap();

    assert_eq!(json["type"], "video_url");
    assert_eq!(json["video_url"]["url"], "https://example.com/video.mp4");
}

#[test]
fn test_user_content_video_base64_serialization() {
    let content = UserContent::Video {
        video_url: VideoUrl {
            url: format!(
                "data:{};base64,SGVsbG8=",
                VideoMediaType::MP4.to_mime_type()
            ),
        },
    };
    let json = serde_json::to_value(&content).unwrap();

    assert_eq!(json["type"], "video_url");
    assert_eq!(json["video_url"]["url"], "data:video/mp4;base64,SGVsbG8=");
}

#[test]
fn test_user_content_video_url_deserialization() {
    let json = json!({
        "type": "video_url",
        "video_url": {
            "url": "https://example.com/video.mp4"
        }
    });

    let content: UserContent = serde_json::from_value(json).unwrap();
    match content {
        UserContent::Video { video_url } => {
            assert_eq!(video_url.url, "https://example.com/video.mp4");
        }
        _ => panic!("Expected Video variant"),
    }
}

#[test]
fn test_message_user_with_video_serialization() {
    let msg = Message::User {
        content: vec![
            UserContent::Text {
                text: "Describe this video:".to_string(),
            },
            UserContent::Video {
                video_url: VideoUrl {
                    url: "https://example.com/video.mp4".to_string(),
                },
            },
        ],
        name: None,
    };
    let json = serde_json::to_value(&msg).unwrap();

    assert_eq!(json["role"], "user");
    let content = json["content"].as_array().unwrap();
    assert_eq!(content.len(), 2);
    assert_eq!(content[0]["type"], "text");
    assert_eq!(content[1]["type"], "video_url");
    assert_eq!(
        content[1]["video_url"]["url"],
        "https://example.com/video.mp4"
    );
}

#[test]
fn test_user_content_video_url_no_media_type_needed() {
    let rig_content = message::UserContent::Video(message::Video {
        data: DocumentSourceKind::Url("https://example.com/video.mp4".to_string()),
        media_type: None,
        additional_params: None,
    });
    let openrouter_content: UserContent = user_content_to_openai(rig_content).unwrap();

    match openrouter_content {
        UserContent::Video { video_url } => {
            assert_eq!(video_url.url, "https://example.com/video.mp4");
        }
        _ => panic!("Expected Video variant"),
    }
}

fn prompt_caching_completion_request() -> CompletionRequest {
    CompletionRequest {
        model: None,
        chat_history: vec![
            crate::message::Message::system("You are a helpful assistant."),
            crate::message::Message::user("Hello"),
        ],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    }
}

#[test]
fn test_final_request_body_applies_prompt_caching_to_converted_completion_request() {
    let request = OpenrouterCompletionRequest::try_from(OpenRouterRequestParams {
        model: "anthropic/claude-3.5-sonnet",
        request: prompt_caching_completion_request(),
        strict_tools: false,
    })
    .expect("request conversion should succeed");

    let body = final_request_body(&request, true).expect("request body should serialize");
    let system_block = &body["messages"][0]["content"][0];

    assert_eq!(system_block["type"], "text");
    assert_eq!(system_block["text"], "You are a helpful assistant.");
    assert_eq!(system_block["cache_control"]["type"], "ephemeral");

    let body = final_request_body(&request, false).expect("request body should serialize");
    assert!(
        body["messages"][0]["content"][0]
            .get("cache_control")
            .is_none(),
        "prompt caching should be opt-in"
    );
}

#[test]
fn test_final_request_body_preserves_stream_flag_when_prompt_caching_enabled() {
    let mut request = OpenrouterCompletionRequest::try_from(OpenRouterRequestParams {
        model: "anthropic/claude-3.5-sonnet",
        request: prompt_caching_completion_request(),
        strict_tools: false,
    })
    .expect("request conversion should succeed");
    request.additional_params = Some(json!({ "stream": true }));

    let body = final_request_body(&request, true).expect("request body should serialize");

    assert_eq!(body["stream"], true);
    assert_eq!(
        body["messages"][0]["content"][0]["cache_control"]["type"],
        "ephemeral"
    );
}

#[test]
fn test_apply_prompt_caching_string_system_message() {
    let mut body = json!({
        "model": "anthropic/claude-3.5-sonnet",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"}
        ]
    });

    apply_prompt_caching(&mut body);

    let system_content = &body["messages"][0]["content"];
    assert!(
        system_content.is_array(),
        "system content should be an array after caching"
    );
    let block = &system_content[0];
    assert_eq!(block["type"], "text");
    assert_eq!(block["text"], "You are a helpful assistant.");
    assert_eq!(block["cache_control"]["type"], "ephemeral");

    // User message should be unchanged.
    assert_eq!(body["messages"][1]["content"], "Hello");
}

#[test]
fn test_apply_prompt_caching_array_system_message_marks_last_block() {
    let mut body = json!({
        "model": "anthropic/claude-3.5-sonnet",
        "messages": [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Part 1. "},
                    {"type": "text", "text": "Part 2."}
                ]
            }
        ]
    });

    apply_prompt_caching(&mut body);

    let system_content = &body["messages"][0]["content"];
    assert!(system_content.is_array());
    // Both blocks are preserved; only the last one gets cache_control.
    assert_eq!(system_content.as_array().unwrap().len(), 2);
    assert_eq!(system_content[0]["text"], "Part 1. ");
    assert!(system_content[0].get("cache_control").is_none());
    assert_eq!(system_content[1]["text"], "Part 2.");
    assert_eq!(system_content[1]["cache_control"]["type"], "ephemeral");
}

#[test]
fn test_apply_prompt_caching_preserves_non_text_blocks() {
    let mut body = json!({
        "model": "anthropic/claude-3.5-sonnet",
        "messages": [
            {
                "role": "system",
                "content": [
                    {"type": "image", "source": {"type": "url", "url": "https://example.com/img.png"}},
                    {"type": "text", "text": "Describe the image."}
                ]
            }
        ]
    });

    apply_prompt_caching(&mut body);

    let system_content = &body["messages"][0]["content"];
    assert_eq!(system_content.as_array().unwrap().len(), 2);
    // Non-text block is preserved unchanged.
    assert_eq!(system_content[0]["type"], "image");
    assert!(system_content[0].get("cache_control").is_none());
    // Text block (last) receives the cache boundary.
    assert_eq!(system_content[1]["type"], "text");
    assert_eq!(system_content[1]["cache_control"]["type"], "ephemeral");
}

#[test]
fn test_apply_prompt_caching_no_system_message_is_noop() {
    let mut body = json!({
        "model": "openai/gpt-4o",
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    });

    let body_before = body.clone();
    apply_prompt_caching(&mut body);
    assert_eq!(
        body, body_before,
        "body should be unchanged when no system message exists"
    );
}

#[test]
fn test_completion_response_extracts_generated_images() {
    let json = json!({
        "id": "resp_img",
        "object": "chat.completion",
        "created": 1,
        "model": "google/gemini-flash-image-preview",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "Here is your image.",
                "images": [
                    {"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgo="}}
                ]
            }
        }]
    });

    let response: CompletionResponse = serde_json::from_value(json).unwrap();
    let converted = response.normalize(PROVIDER_NAME).unwrap();
    let items: Vec<completion::AssistantContent> = converted.choice.into_iter().collect();
    assert_eq!(items.len(), 2);

    assert!(items.iter().any(|item| matches!(
        item,
        completion::AssistantContent::Text(t) if t.text == "Here is your image."
    )));
    assert!(items.iter().any(|item| matches!(
        item,
        completion::AssistantContent::Image(message::Image {
            data: message::DocumentSourceKind::Base64(b64),
            media_type: Some(message::ImageMediaType::PNG),
            additional_params: Some(_),
            ..
        }) if b64 == "iVBORw0KGgo="
    )));
    assert!(
        items.iter().any(|item| matches!(
            item,
            completion::AssistantContent::Image(image)
                if is_openrouter_response_image(image)
        )),
        "generated images should be marked as OpenRouter response-only artifacts"
    );
}

#[test]
fn test_completion_response_extracts_generated_images_url() {
    let json = json!({
        "id": "resp_img_url",
        "object": "chat.completion",
        "created": 1,
        "model": "google/gemini-flash-image-preview",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "Here is your image.",
                "images": [
                    {"type":"image_url","image_url":{"url":"https://example.com/generated.png"}}
                ]
            }
        }]
    });

    let response: CompletionResponse = serde_json::from_value(json).unwrap();
    let converted = response.normalize(PROVIDER_NAME).unwrap();
    let items: Vec<completion::AssistantContent> = converted.choice.into_iter().collect();
    assert_eq!(items.len(), 2);

    assert!(items.iter().any(|item| matches!(
        item,
        completion::AssistantContent::Image(message::Image {
            data: message::DocumentSourceKind::Url(url),
            media_type: None,
            additional_params: Some(_),
            ..
        }) if url == "https://example.com/generated.png"
    )));
    assert!(
        items.iter().any(|item| matches!(
            item,
            completion::AssistantContent::Image(image)
                if is_openrouter_response_image(image)
        )),
        "generated URL images should be marked as OpenRouter response-only artifacts"
    );
}

#[test]
fn test_generated_images_do_not_break_assistant_history_conversion() {
    let generated_image = response_image_to_assistant_content(&ResponseImage {
        image_url: ImageUrl {
            url: "data:image/png;base64,abc".to_string(),
            detail: None,
        },
    });

    let content = vec![
        completion::AssistantContent::text("Here is your image."),
        generated_image,
    ];
    let messages = assistant_contents_to_messages(content).unwrap();

    assert_eq!(messages.len(), 1);
    assert!(matches!(
        &messages[0],
        Message::Assistant { content, .. }
            if content == &vec![openai::AssistantContent::Text {
                text: "Here is your image.".to_string()
            }]
    ));
}

#[test]
fn test_image_only_assistant_history_is_omitted_for_openrouter() {
    let generated_image = response_image_to_assistant_content(&ResponseImage {
        image_url: ImageUrl {
            url: "data:image/png;base64,abc".to_string(),
            detail: None,
        },
    });

    let messages = assistant_contents_to_messages(vec![generated_image]).unwrap();

    assert!(
        messages.is_empty(),
        "response-only generated image turns should not be replayed as assistant content"
    );
}

#[test]
fn test_unmarked_assistant_image_history_errors_for_openrouter() {
    let image =
        completion::AssistantContent::image_base64("abc", Some(message::ImageMediaType::PNG), None);

    let err = assistant_contents_to_messages(vec![image]).unwrap_err();

    match err {
        message::MessageError::ConversionError(message) => assert!(
            message.contains("OpenRouter does not support assistant image content"),
            "unexpected error: {message}"
        ),
    }
}

#[test]
fn test_mixed_text_and_generated_image_replays_text_only_for_openrouter() {
    let generated_image = response_image_to_assistant_content(&ResponseImage {
        image_url: ImageUrl {
            url: "https://example.com/generated.png".to_string(),
            detail: None,
        },
    });

    let messages = assistant_contents_to_messages(vec![
        completion::AssistantContent::text("Keep this text."),
        generated_image,
    ])
    .unwrap();

    let serialized = serde_json::to_value(&messages).unwrap();
    assert_eq!(
        serialized,
        json!([{
            "role": "assistant",
            "content": [{"type": "text", "text": "Keep this text."}]
        }])
    );
}

#[test]
fn test_assistant_images_not_serialized_in_request() {
    let msg = Message::Assistant {
        content: vec!["Hello".to_string().into()],
        refusal: None,
        audio: None,
        name: None,
        tool_calls: vec![],
        reasoning: None,
        reasoning_details: vec![],
        images: vec![ResponseImage {
            image_url: ImageUrl {
                url: "data:image/png;base64,abc".to_string(),
                detail: None,
            },
        }],
    };
    let serialized = serde_json::to_value(&msg).unwrap();
    assert!(
        serialized.get("images").is_none(),
        "images field must not appear in serialized assistant message"
    );
}

// -----------------------------------------------------------------------
// Refusal fallback — wire shapes the live gateway will not produce on
// demand. The recorded cells live in
// `tests/providers/openrouter/cassette/refusal_matrix.rs`.
// -----------------------------------------------------------------------

fn refusal_response(message: serde_json::Value) -> CompletionResponse {
    serde_json::from_value(json!({
        "id": "gen-refusal",
        "object": "chat.completion",
        "created": 1,
        "model": "openai/gpt-4o",
        "choices": [{ "index": 0, "message": message, "finish_reason": "stop" }],
    }))
    .unwrap()
}

#[test]
fn raw_completion_response_retains_routing_metadata() {
    let response: CompletionResponse = serde_json::from_value(json!({
        "id": "gen-routing",
        "object": "chat.completion",
        "created": 1,
        "model": "openai/gpt-4o-mini",
        "provider": "OpenAI",
        "service_tier": "default",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop"
        }]
    }))
    .expect("live OpenRouter routing metadata should deserialize");

    assert_eq!(response.provider.as_deref(), Some("OpenAI"));
    assert_eq!(response.service_tier.as_deref(), Some("default"));
}

fn text_parts(response: &completion::CompletionResponse) -> Vec<String> {
    response
        .choice
        .iter()
        .filter_map(|part| match part {
            completion::AssistantContent::Text(text) => Some(text.text.clone()),
            _ => None,
        })
        .collect()
}

/// The recorded shape: `content` held at `null` with the refusal beside
/// it. Before the fix this normalized to nothing and errored.
#[test]
fn refusal_fallback_surfaces_a_null_content_refusal() {
    let response = refusal_response(json!({
        "role": "assistant",
        "content": null,
        "refusal": "I'm very sorry, but I can't assist with that request.",
    }));

    let converted = response.normalize(PROVIDER_NAME).unwrap();
    assert_eq!(
        text_parts(&converted),
        vec!["I'm very sorry, but I can't assist with that request."]
    );
}

/// An absent `content` key reads the same as an explicit `null`.
#[test]
fn refusal_fallback_surfaces_a_missing_content_refusal() {
    let response = refusal_response(json!({
        "role": "assistant",
        "refusal": "No.",
    }));

    assert_eq!(
        text_parts(&response.normalize(PROVIDER_NAME).unwrap()),
        vec!["No."]
    );
}

/// An empty-string `content` decodes as one empty text part, which carries
/// no text — so the refusal is still the turn's only *visible* content.
///
/// This path keeps the empty part alongside it: unlike the shared OpenAI
/// normalizer, OpenRouter's does not filter empty content parts. That is
/// pre-existing behavior for any `"content": ""` turn and the fallback
/// neither causes nor changes it; the assertion records both halves rather
/// than claiming a filter this code does not have.
#[test]
fn refusal_fallback_surfaces_a_refusal_beside_empty_content() {
    let response = refusal_response(json!({
        "role": "assistant",
        "content": "",
        "refusal": "No.",
    }));

    let converted = response.normalize(PROVIDER_NAME).unwrap();
    assert_eq!(
        text_parts(&converted),
        vec!["".to_owned(), "No.".to_owned()]
    );
}

/// The whole-message rule: real content wins, and the fallback stays out
/// of the way. This is the shape the streaming path would deliver *both*
/// halves of, so pinning it records the difference rather than assuming it
/// away (see `assistant_refusal_fallback`).
#[test]
fn refusal_fallback_defers_to_non_empty_content() {
    let response = refusal_response(json!({
        "role": "assistant",
        "content": "Here is the answer.",
        "refusal": "I'm sorry.",
    }));

    assert_eq!(
        text_parts(&response.normalize(PROVIDER_NAME).unwrap()),
        vec!["Here is the answer."]
    );
}

/// An empty refusal string is not a refusal.
#[test]
fn refusal_fallback_ignores_an_empty_refusal() {
    let response = refusal_response(json!({
        "role": "assistant",
        "content": null,
        "refusal": "",
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": { "name": "ping", "arguments": "{}" }
        }],
    }));

    let converted = response.normalize(PROVIDER_NAME).unwrap();
    assert!(text_parts(&converted).is_empty(), "{:?}", converted.choice);
    assert_eq!(converted.choice.len(), 1);
}

/// A tool-calls-only turn holds `content` at `null` with no refusal: the
/// shape the fallback must leave exactly as it was.
#[test]
fn refusal_fallback_leaves_a_tool_call_turn_alone() {
    let response = refusal_response(json!({
        "role": "assistant",
        "content": null,
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": { "name": "ping", "arguments": "{}" }
        }],
    }));

    let converted = response.normalize(PROVIDER_NAME).unwrap();
    assert_eq!(converted.choice.len(), 1);
    assert!(matches!(
        converted.choice.first(),
        Some(completion::AssistantContent::ToolCall(_))
    ));
}

/// A refusal can arrive on a turn that also carries tool calls; the
/// refusal is the turn's text and the calls survive beside it.
#[test]
fn refusal_fallback_coexists_with_tool_calls() {
    let response = refusal_response(json!({
        "role": "assistant",
        "content": null,
        "refusal": "I can't help with that.",
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": { "name": "ping", "arguments": "{}" }
        }],
    }));

    let converted = response.normalize(PROVIDER_NAME).unwrap();
    assert_eq!(text_parts(&converted), vec!["I can't help with that."]);
    assert!(
        converted
            .choice
            .iter()
            .any(|part| matches!(part, completion::AssistantContent::ToolCall(_)))
    );
}

/// Reasoning blocks are not text, so a reasoning-carrying refusal turn
/// still needs the fallback for its visible content.
#[test]
fn refusal_fallback_applies_beside_reasoning_details() {
    let response = refusal_response(json!({
        "role": "assistant",
        "content": null,
        "refusal": "I can't help with that.",
        "reasoning_details": [
            { "type": "reasoning.summary", "id": "rs_1", "format": "openai-responses-v1",
              "index": 0, "summary": "considered" }
        ],
    }));

    let converted = response.normalize(PROVIDER_NAME).unwrap();
    assert_eq!(text_parts(&converted), vec!["I can't help with that."]);
    assert!(
        converted
            .choice
            .iter()
            .any(|part| matches!(part, completion::AssistantContent::Reasoning(_)))
    );
}

/// The Responses-API spelling — a `refusal` *content part* — still works;
/// the fix adds the sibling-field rule without displacing it, and the two
/// must not both fire.
#[test]
fn refusal_fallback_does_not_double_up_with_a_refusal_content_part() {
    let response = refusal_response(json!({
        "role": "assistant",
        "content": [{ "type": "refusal", "refusal": "I can't help with that." }],
        "refusal": "I can't help with that.",
    }));

    assert_eq!(
        text_parts(&response.normalize(PROVIDER_NAME).unwrap()),
        vec!["I can't help with that."]
    );
}

/// The raw text view and the normalized response must never disagree
/// about whether a refused turn said anything — the internal
/// inconsistency that made this bug visible.
#[test]
fn refusal_fallback_keeps_raw_and_normalized_text_in_step() {
    let response = refusal_response(json!({
        "role": "assistant",
        "content": null,
        "refusal": "I'm sorry, but I can't help with that.",
    }));

    let raw_text = response.text_response().unwrap();
    let normalized = response.normalize(PROVIDER_NAME).unwrap();

    assert_eq!(text_parts(&normalized), vec![raw_text]);
}
