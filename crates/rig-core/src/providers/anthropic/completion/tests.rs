use super::*;
use crate::message::EMPTY_RESPONSE_ERROR;
use serde_json::json;
use serde_path_to_error::deserialize;

#[test]
fn current_model_default_max_tokens_match_anthropic_limits() {
    assert_eq!(default_max_tokens_for_model(CLAUDE_OPUS_4_8), Some(128_000));
    assert_eq!(default_max_tokens_for_model(CLAUDE_OPUS_4_7), Some(128_000));
    assert_eq!(default_max_tokens_for_model(CLAUDE_OPUS_4_6), Some(128_000));
    assert_eq!(
        default_max_tokens_for_model(CLAUDE_SONNET_4_6),
        Some(64_000)
    );
    assert_eq!(default_max_tokens_for_model(CLAUDE_HAIKU_4_5), Some(64_000));
}

#[test]
fn unknown_model_uses_conservative_default_max_tokens_fallback() {
    assert_eq!(default_max_tokens_for_model("claude-unknown"), None);
    assert_eq!(default_max_tokens_with_fallback("claude-unknown"), 2_048);
}

#[test]
fn system_role_message_deserializes_and_round_trips() {
    let message: Message = serde_json::from_str(
        r#"
        {
            "role": "system",
            "content": "From now on, require explicit type annotations."
        }
        "#,
    )
    .unwrap();

    assert_eq!(message.role, Role::System);

    let generic: message::Message = message.try_into().unwrap();
    assert_eq!(
        generic,
        message::Message::System {
            content: "From now on, require explicit type annotations.".to_string()
        }
    );

    let provider: Message = generic.try_into().unwrap();
    assert_eq!(provider.role, Role::System);
}

#[test]
fn test_deserialize_message() {
    let assistant_message_json = r#"
        {
            "role": "assistant",
            "content": "\n\nHello there, how may I assist you today?"
        }
        "#;

    let assistant_message_json2 = r#"
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "\n\nHello there, how may I assist you today?"
                },
                {
                    "type": "tool_use",
                    "id": "toolu_01A09q90qw90lq917835lq9",
                    "name": "get_weather",
                    "input": {"location": "San Francisco, CA"}
                }
            ]
        }
        "#;

    let user_message_json = r#"
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": "/9j/4AAQSkZJRg..."
                    }
                },
                {
                    "type": "text",
                    "text": "What is in this image?"
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01A09q90qw90lq917835lq9",
                    "content": "15 degrees"
                }
            ]
        }
        "#;

    let assistant_message: Message = {
        let jd = &mut serde_json::Deserializer::from_str(assistant_message_json);
        deserialize(jd).unwrap_or_else(|err| {
            panic!("Deserialization error at {}: {}", err.path(), err);
        })
    };

    let assistant_message2: Message = {
        let jd = &mut serde_json::Deserializer::from_str(assistant_message_json2);
        deserialize(jd).unwrap_or_else(|err| {
            panic!("Deserialization error at {}: {}", err.path(), err);
        })
    };

    let user_message: Message = {
        let jd = &mut serde_json::Deserializer::from_str(user_message_json);
        deserialize(jd).unwrap_or_else(|err| {
            panic!("Deserialization error at {}: {}", err.path(), err);
        })
    };

    let Message { role, content } = assistant_message;
    assert_eq!(role, Role::Assistant);
    assert_eq!(
        content.first(),
        Some(&Content::Text {
            text: "\n\nHello there, how may I assist you today?".to_owned(),
            citations: Vec::new(),
            cache_control: None,
        })
    );

    let Message { role, content } = assistant_message2;
    {
        assert_eq!(role, Role::Assistant);
        assert_eq!(content.len(), 2);

        let mut iter = content.into_iter();

        match iter.next().unwrap() {
            Content::Text { text, .. } => {
                assert_eq!(text, "\n\nHello there, how may I assist you today?");
            }
            _ => panic!("Expected text content"),
        }

        match iter.next().unwrap() {
            Content::ToolUse { id, name, input } => {
                assert_eq!(id, "toolu_01A09q90qw90lq917835lq9");
                assert_eq!(name, "get_weather");
                assert_eq!(input, json!({"location": "San Francisco, CA"}));
            }
            _ => panic!("Expected tool use content"),
        }

        assert_eq!(iter.next(), None);
    }

    let Message { role, content } = user_message;
    {
        assert_eq!(role, Role::User);
        assert_eq!(content.len(), 3);

        let mut iter = content.into_iter();

        match iter.next().unwrap() {
            Content::Image { source, .. } => {
                assert_eq!(
                    source,
                    ImageSource::Base64 {
                        data: "/9j/4AAQSkZJRg...".to_owned(),
                        media_type: ImageFormat::JPEG,
                    }
                );
            }
            _ => panic!("Expected image content"),
        }

        match iter.next().unwrap() {
            Content::Text { text, .. } => {
                assert_eq!(text, "What is in this image?");
            }
            _ => panic!("Expected text content"),
        }

        match iter.next().unwrap() {
            Content::ToolResult {
                tool_use_id,
                content,
                is_error,
                ..
            } => {
                assert_eq!(tool_use_id, "toolu_01A09q90qw90lq917835lq9");
                assert_eq!(
                    content.first(),
                    Some(&ToolResultContent::Text {
                        text: "15 degrees".to_owned()
                    })
                );
                assert_eq!(is_error, None);
            }
            _ => panic!("Expected tool result content"),
        }

        assert_eq!(iter.next(), None);
    }
}

#[test]
fn test_message_to_message_conversion() {
    let user_message: Message = serde_json::from_str(
        r#"
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": "/9j/4AAQSkZJRg..."
                    }
                },
                {
                    "type": "text",
                    "text": "What is in this image?"
                },
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "data": "base64_encoded_pdf_data",
                        "media_type": "application/pdf"
                    }
                }
            ]
        }
        "#,
    )
    .unwrap();

    let assistant_message = Message {
        role: Role::Assistant,
        content: vec![Content::ToolUse {
            id: "toolu_01A09q90qw90lq917835lq9".to_string(),
            name: "get_weather".to_string(),
            input: json!({"location": "San Francisco, CA"}),
        }],
    };

    let tool_message = Message {
        role: Role::User,
        content: vec![Content::ToolResult {
            tool_use_id: "toolu_01A09q90qw90lq917835lq9".to_string(),
            content: vec![ToolResultContent::Text {
                text: "15 degrees".to_string(),
            }],
            is_error: None,
            cache_control: None,
        }],
    };

    let converted_user_message: message::Message = user_message.clone().try_into().unwrap();
    let converted_assistant_message: message::Message =
        assistant_message.clone().try_into().unwrap();
    let converted_tool_message: message::Message = tool_message.clone().try_into().unwrap();

    match converted_user_message.clone() {
        message::Message::User { content } => {
            assert_eq!(content.len(), 3);

            let mut iter = content.into_iter();

            match iter.next().unwrap() {
                message::UserContent::Image(message::Image {
                    data, media_type, ..
                }) => {
                    assert_eq!(data, DocumentSourceKind::base64("/9j/4AAQSkZJRg..."));
                    assert_eq!(media_type, Some(message::ImageMediaType::JPEG));
                }
                _ => panic!("Expected image content"),
            }

            match iter.next().unwrap() {
                message::UserContent::Text(message::Text { text, .. }) => {
                    assert_eq!(text, "What is in this image?");
                }
                _ => panic!("Expected text content"),
            }

            match iter.next().unwrap() {
                message::UserContent::Document(message::Document {
                    data, media_type, ..
                }) => {
                    assert_eq!(
                        data,
                        DocumentSourceKind::String("base64_encoded_pdf_data".into())
                    );
                    assert_eq!(media_type, Some(message::DocumentMediaType::PDF));
                }
                _ => panic!("Expected document content"),
            }

            assert_eq!(iter.next(), None);
        }
        _ => panic!("Expected user message"),
    }

    match converted_tool_message.clone() {
        message::Message::User { content } => {
            let Some(message::UserContent::ToolResult(message::ToolResult {
                call,
                name,
                content,
                ..
            })) = content.first()
            else {
                panic!("Expected tool result content")
            };
            assert_eq!(call, "toolu_01A09q90qw90lq917835lq9");
            // The Anthropic wire carries no tool name on `tool_result`
            // blocks, so the inbound conversion is lossy by design.
            assert_eq!(name, "");
            match content.first() {
                Some(message::ToolResultContent::Text(message::Text { text, .. })) => {
                    assert_eq!(text, "15 degrees");
                }
                _ => panic!("Expected text content"),
            }
        }
        _ => panic!("Expected tool result content"),
    }

    match converted_assistant_message.clone() {
        message::Message::Assistant { content, .. } => {
            assert_eq!(content.len(), 1);

            match content.first() {
                Some(message::AssistantContent::ToolCall(message::ToolCall {
                    id,
                    function,
                    ..
                })) => {
                    assert_eq!(id, "toolu_01A09q90qw90lq917835lq9");
                    assert_eq!(function.name, "get_weather");
                    assert_eq!(function.arguments, json!({"location": "San Francisco, CA"}));
                }
                _ => panic!("Expected tool call content"),
            }
        }
        _ => panic!("Expected assistant message"),
    }

    let original_user_message: Message = converted_user_message.try_into().unwrap();
    let original_assistant_message: Message = converted_assistant_message.try_into().unwrap();
    let original_tool_message: Message = converted_tool_message.try_into().unwrap();

    assert_eq!(user_message, original_user_message);
    assert_eq!(assistant_message, original_assistant_message);
    assert_eq!(tool_message, original_tool_message);
}

#[test]
fn test_content_format_conversion() {
    use crate::completion::message::ContentFormat;

    let source_type: SourceType = ContentFormat::Url.try_into().unwrap();
    assert_eq!(source_type, SourceType::URL);

    let content_format: ContentFormat = SourceType::URL.into();
    assert_eq!(content_format, ContentFormat::Url);

    let source_type: SourceType = ContentFormat::Base64.try_into().unwrap();
    assert_eq!(source_type, SourceType::BASE64);

    let content_format: ContentFormat = SourceType::BASE64.into();
    assert_eq!(content_format, ContentFormat::Base64);

    let source_type: SourceType = ContentFormat::String.try_into().unwrap();
    assert_eq!(source_type, SourceType::TEXT);

    let content_format: ContentFormat = SourceType::TEXT.into();
    assert_eq!(content_format, ContentFormat::String);
}

#[test]
fn test_cache_control_serialization() {
    // Test SystemContent with cache_control
    let system = SystemContent::Text {
        text: "You are a helpful assistant.".to_string(),
        cache_control: Some(CacheControl::ephemeral()),
    };
    let json = serde_json::to_string(&system).unwrap();
    assert!(json.contains(r#""cache_control":{"type":"ephemeral"}"#));
    assert!(json.contains(r#""type":"text""#));

    // Test SystemContent without cache_control (should not have cache_control field)
    let system_no_cache = SystemContent::Text {
        text: "Hello".to_string(),
        cache_control: None,
    };
    let json_no_cache = serde_json::to_string(&system_no_cache).unwrap();
    assert!(!json_no_cache.contains("cache_control"));

    // Test Content::Text with cache_control
    let content = Content::Text {
        text: "Test message".to_string(),
        citations: Vec::new(),
        cache_control: Some(CacheControl::ephemeral()),
    };
    let json_content = serde_json::to_string(&content).unwrap();
    assert!(json_content.contains(r#""cache_control":{"type":"ephemeral"}"#));

    // Manual prompt caching over a bare system prompt + conversation: the
    // system block and the tail of the last message get the marker.
    let mut system_vec = vec![SystemContent::Text {
        text: "System prompt".to_string(),
        cache_control: None,
    }];
    let mut messages = vec![
        Message {
            role: Role::User,
            content: vec![Content::Text {
                text: "First message".to_string(),
                citations: Vec::new(),
                cache_control: None,
            }],
        },
        Message {
            role: Role::Assistant,
            content: vec![Content::Text {
                text: "Response".to_string(),
                citations: Vec::new(),
                cache_control: None,
            }],
        },
    ];

    apply_prompt_cache_control(&mut system_vec, &mut messages, &mut [], true, None, None).unwrap();

    // System should have cache_control
    match &system_vec[0] {
        SystemContent::Text { cache_control, .. } => {
            assert!(cache_control.is_some());
        }
    }

    // Only the last content block of last message should have cache_control
    // First message should NOT have cache_control
    for content in messages[0].content.iter() {
        if let Content::Text { cache_control, .. } = content {
            assert!(cache_control.is_none());
        }
    }

    // Last message SHOULD have cache_control
    for content in messages[1].content.iter() {
        if let Content::Text { cache_control, .. } = content {
            assert!(cache_control.is_some());
        }
    }
}

fn generic_tool(name: &str) -> completion::ToolDefinition {
    completion::ToolDefinition {
        name: name.to_string(),
        description: format!("{name} description"),
        parameters: json!({
            "type": "object",
            "properties": {}
        }),
    }
}

fn completion_request_with_tools(
    tools: Vec<completion::ToolDefinition>,
    additional_params: Option<serde_json::Value>,
) -> CompletionRequest {
    CompletionRequest {
        model: None,
        chat_history: vec![
            message::Message::system("System prompt"),
            message::Message::from("Hello"),
        ],
        documents: Vec::new(),
        tools,
        temperature: None,
        max_tokens: Some(64),
        tool_choice: None,
        additional_params,
        output_schema: None,
        record_telemetry_content: false,
    }
}

fn completion_request_with_history(
    chat_history: Vec<message::Message>,
    preamble: Option<String>,
) -> CompletionRequest {
    CompletionRequest {
        model: None,
        chat_history: preamble
            .map(message::Message::system)
            .into_iter()
            .chain(chat_history)
            .collect(),
        documents: Vec::new(),
        tools: Vec::new(),
        temperature: None,
        max_tokens: Some(64),
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    }
}

#[test]
fn rig_tools_are_non_strict_by_default() {
    let request = completion_request_with_tools(vec![generic_tool("lookup")], None);
    let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
        model: CLAUDE_SONNET_4_6,
        request,
        prompt_caching: false,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    assert!(value["tools"][0].get("strict").is_none());
    assert!(
        value["tools"][0]["input_schema"]
            .get("additionalProperties")
            .is_none()
    );
}

#[test]
fn strict_tool_hook_is_a_noop_for_anthropic_compatible_gateways() {
    let mut additional_params = serde_json::Value::Null;
    let tools = build_tool_definitions::<crate::providers::minimax::MiniMaxAnthropic>(
        vec![generic_tool("lookup")],
        &mut additional_params,
        true,
    )
    .unwrap();

    assert!(tools[0].get("strict").is_none());
    assert!(
        tools[0]["input_schema"]
            .get("additionalProperties")
            .is_none()
    );
}

#[test]
fn strict_tools_opt_in_marks_and_sanitizes_rig_tools_only() {
    let mut tool = generic_tool("lookup");
    tool.parameters = json!({
        "type": "object",
        "additionalProperties": true,
        "properties": {
            "query": {
                "type": "string",
                "minLength": 2,
                "maxLength": 20,
                "pattern": "^[a-z]+$",
                "format": "uuid"
            },
            "kind": {
                "type": "string",
                "const": "lookup"
            },
            "legacy_filter": {
                "$ref": "#/definitions/LegacyFilter"
            },
            "options": {
                "type": "object",
                "additionalProperties": true,
                "properties": {
                    "limit": {
                        "type": ["integer", "null"],
                        "minimum": 1,
                        "maximum": 100,
                        "format": "uint32"
                    }
                }
            }
        },
        "definitions": {
            "LegacyFilter": {
                "type": "object",
                "properties": {
                    "term": { "type": "string" }
                }
            }
        },
        "required": ["query"]
    });
    let request = completion_request_with_tools(
        vec![tool],
        Some(json!({
            "tools": [{
                "type": "mcp_toolset",
                "name": "remote_tools"
            }]
        })),
    );
    let request = AnthropicCompletionRequest::try_from_params::<
        crate::providers::anthropic::client::Anthropic,
    >(
        AnthropicRequestParams {
            model: CLAUDE_SONNET_4_6,
            request,
            prompt_caching: false,
            automatic_caching: false,
            automatic_caching_ttl: None,
            static_prefix_cache_ttl: None,
        },
        true,
    )
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let rig_tool = &value["tools"][0];
    assert_eq!(rig_tool["strict"], true);
    assert_eq!(rig_tool["input_schema"]["additionalProperties"], false);
    let required = rig_tool["input_schema"]["required"]
        .as_array()
        .expect("strict object schema should list required properties");
    assert_eq!(required.len(), 1);
    assert!(required.contains(&json!("query")));
    assert_eq!(
        rig_tool["input_schema"]["properties"]["options"]["additionalProperties"],
        false
    );
    assert!(
        rig_tool["input_schema"]["properties"]["options"]
            .get("required")
            .is_none()
    );
    let query = &rig_tool["input_schema"]["properties"]["query"];
    assert_eq!(query["format"], "uuid");
    for keyword in ["minLength", "maxLength", "pattern"] {
        assert!(query.get(keyword).is_none());
    }
    let query_description = query["description"]
        .as_str()
        .expect("unsupported string constraints should become guidance");
    for guidance in ["minLength: 2", "maxLength: 20", "pattern: ^[a-z]+$"] {
        assert!(query_description.contains(guidance));
    }
    assert_eq!(
        rig_tool["input_schema"]["properties"]["kind"]["const"],
        "lookup"
    );
    assert_eq!(
        rig_tool["input_schema"]["properties"]["legacy_filter"]["$ref"],
        "#/definitions/LegacyFilter"
    );
    assert_eq!(
        rig_tool["input_schema"]["definitions"]["LegacyFilter"]["additionalProperties"],
        false
    );
    let limit = &rig_tool["input_schema"]["properties"]["options"]["properties"]["limit"];
    assert!(limit.get("format").is_none());
    assert!(
        ["minimum", "maximum"]
            .into_iter()
            .all(|keyword| limit.get(keyword).is_none())
    );
    let limit_description = limit["description"]
        .as_str()
        .expect("unsupported numeric constraints should become guidance");
    for guidance in ["minimum: 1", "maximum: 100", "format: uint32"] {
        assert!(limit_description.contains(guidance));
    }

    let provider_tool = &value["tools"][1];
    assert_eq!(provider_tool["type"], "mcp_toolset");
    assert!(provider_tool.get("strict").is_none());
}

fn system_has_cache_control(value: &serde_json::Value) -> bool {
    value["system"]
        .as_array()
        .and_then(|blocks| blocks.last())
        .and_then(|block| block.get("cache_control"))
        .is_some()
}

fn last_message_has_cache_control(value: &serde_json::Value) -> bool {
    value["messages"]
        .as_array()
        .and_then(|messages| messages.last())
        .and_then(|message| message["content"].as_array())
        .and_then(|content| content.last())
        .and_then(|content| content.get("cache_control"))
        .is_some()
}

#[test]
fn opus_4_8_preserves_mid_conversation_system_message() {
    let request = completion_request_with_history(
        vec![
            message::Message::System {
                content: "Global history instruction.".to_string(),
            },
            message::Message::from("Review this code."),
            message::Message::System {
                content: "From now on, require explicit type annotations.".to_string(),
            },
        ],
        Some("Top-level instruction.".to_string()),
    );

    let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
        model: CLAUDE_OPUS_4_8,
        request,
        prompt_caching: false,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    assert_eq!(value["system"][0]["text"], "Top-level instruction.");
    assert_eq!(value["system"][1]["text"], "Global history instruction.");

    let messages = value["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[1]["role"], "system");
    assert_eq!(
        messages[1]["content"][0]["text"],
        "From now on, require explicit type annotations."
    );
}

#[test]
fn opus_4_8_preserves_mid_conversation_system_message_before_assistant_turn() {
    let request = completion_request_with_history(
        vec![
            message::Message::user("Review this code."),
            message::Message::System {
                content: "From now on, require explicit type annotations.".to_string(),
            },
            message::Message::assistant("I will enforce explicit type annotations."),
        ],
        None,
    );

    let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
        model: CLAUDE_OPUS_4_8,
        request,
        prompt_caching: false,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let messages = value["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[1]["role"], "system");
    assert_eq!(messages[2]["role"], "assistant");
    assert!(value.get("system").is_none());
}

#[test]
fn opus_4_8_hoists_leading_system_message_when_documents_are_present() {
    let mut request = completion_request_with_history(
        vec![
            message::Message::System {
                content: "Global history instruction.".to_string(),
            },
            message::Message::assistant("Acknowledged."),
            message::Message::System {
                content: "Mid-conversation instruction.".to_string(),
            },
            message::Message::user("Answer from the document."),
        ],
        None,
    );
    request.documents = vec![completion::Document {
        id: "doc".to_string(),
        text: "Document context.".to_string(),
        additional_props: Default::default(),
    }];

    let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
        model: CLAUDE_OPUS_4_8,
        request,
        prompt_caching: false,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    assert_eq!(value["system"][0]["text"], "Global history instruction.");
    assert_eq!(value["system"][1]["text"], "Mid-conversation instruction.");

    let messages = value["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[1]["role"], "assistant");
    assert_eq!(messages[2]["role"], "user");
    assert!(
        messages[0].to_string().contains("<file id: doc>"),
        "document message should follow top-level system: {messages:?}"
    );
    assert_eq!(
        messages
            .iter()
            .filter(|message| message.to_string().contains("<file id: doc>"))
            .count(),
        1,
        "document message should appear exactly once: {messages:?}"
    );
    assert!(
        messages
            .iter()
            .all(|message| message["role"].as_str() != Some("system"))
    );
}

#[test]
fn opus_4_8_preserves_system_message_after_assistant_server_tool_result() {
    let request = completion_request_with_history(
        vec![
            message::Message::Assistant {
                id: None,
                content: vec![
                    message::AssistantContent::Text(message::Text {
                        text: String::new(),
                        additional_params: crate::message::AdditionalParams::try_from_value(
                            json!({
                                ANTHROPIC_RAW_CONTENT_KEY: {
                                    "type": "server_tool_use",
                                    "id": "srvtoolu_01",
                                    "name": "web_search",
                                    "input": {
                                        "query": "clear daytime sky color"
                                    }
                                }
                            }),
                        )
                        .expect("object params"),
                    }),
                    message::AssistantContent::Text(message::Text {
                        text: String::new(),
                        additional_params: crate::message::AdditionalParams::try_from_value(
                            json!({
                                ANTHROPIC_RAW_CONTENT_KEY: {
                                    "type": "web_search_tool_result",
                                    "tool_use_id": "srvtoolu_01",
                                    "content": {
                                        "type": "web_search_tool_result_error",
                                        "error_code": "unavailable"
                                    }
                                }
                            }),
                        )
                        .expect("object params"),
                    }),
                ],
            },
            message::Message::System {
                content: "For the rest of this conversation, answer in Spanish.".to_string(),
            },
            message::Message::assistant("Entendido."),
        ],
        None,
    );

    let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
        model: CLAUDE_OPUS_4_8,
        request,
        prompt_caching: false,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    assert!(value.get("system").is_none());

    let messages = value["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0]["role"], "assistant");
    assert_eq!(messages[0]["content"][0]["type"], "server_tool_use");
    assert_eq!(messages[0]["content"][1]["type"], "web_search_tool_result");
    assert_eq!(messages[1]["role"], "system");
    assert_eq!(
        messages[1]["content"][0]["text"],
        "For the rest of this conversation, answer in Spanish."
    );
    assert_eq!(messages[2]["role"], "assistant");
}

#[test]
fn foreign_annotated_empty_text_produces_no_anthropic_block() {
    // The Responses ingest mints empty text blocks whose params carry
    // that wire's extras; the agent deliberately keeps them in history.
    // Replayed here, they must vanish from the request — the API
    // rejects empty text blocks and foreign extras cannot reach this
    // wire — while sibling content converts unaffected.
    let foreign_annotated_empty = message::AssistantContent::Text(message::Text {
        text: String::new(),
        additional_params: message::AdditionalParams::try_from_value(json!({
            "openai_responses": {"annotations": [{"type": "url_citation"}]}
        }))
        .expect("object params"),
    });
    assert_eq!(
        anthropic_content_from_assistant_content(foreign_annotated_empty.clone())
            .expect("conversion succeeds"),
        Vec::new(),
        "a foreign-annotated empty block must produce no Anthropic content"
    );

    let message = message::Message::Assistant {
        id: None,
        content: vec![
            foreign_annotated_empty,
            message::AssistantContent::text("real answer"),
        ],
    };
    let converted = Message::try_from(message).expect("message converts");
    assert_eq!(converted.content.len(), 1, "only the real block survives");
    assert!(matches!(
        converted.content.first(),
        Some(Content::Text { text, .. }) if text == "real answer"
    ));
}

#[test]
fn opus_4_8_preserves_system_message_after_assistant_server_tool_use() {
    let request = completion_request_with_history(
        vec![
            message::Message::Assistant {
                id: None,
                content: vec![message::AssistantContent::Text(message::Text {
                    text: String::new(),
                    additional_params: crate::message::AdditionalParams::try_from_value(json!({
                        ANTHROPIC_RAW_CONTENT_KEY: {
                            "type": "server_tool_use",
                            "id": "srvtoolu_01",
                            "name": "web_search",
                            "input": {
                                "query": "clear daytime sky color"
                            }
                        }
                    }))
                    .expect("object params"),
                })],
            },
            message::Message::System {
                content: "For the rest of this conversation, answer in Spanish.".to_string(),
            },
            message::Message::assistant("Entendido."),
        ],
        None,
    );

    let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
        model: CLAUDE_OPUS_4_8,
        request,
        prompt_caching: false,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    assert!(value.get("system").is_none());

    let messages = value["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0]["role"], "assistant");
    assert_eq!(messages[0]["content"][0]["type"], "server_tool_use");
    assert_eq!(messages[1]["role"], "system");
    assert_eq!(
        messages[1]["content"][0]["text"],
        "For the rest of this conversation, answer in Spanish."
    );
    assert_eq!(messages[2]["role"], "assistant");
}

#[test]
fn opus_4_8_hoists_system_message_in_invalid_mid_conversation_position() {
    let request = completion_request_with_history(
        vec![
            message::Message::user("Review this code."),
            message::Message::System {
                content: "From now on, require explicit type annotations.".to_string(),
            },
            message::Message::user("Now review this other file."),
        ],
        None,
    );

    let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
        model: CLAUDE_OPUS_4_8,
        request,
        prompt_caching: false,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    assert_eq!(
        value["system"][0]["text"],
        "From now on, require explicit type annotations."
    );

    let messages = value["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[1]["role"], "user");
}

#[test]
fn older_anthropic_models_hoist_mid_conversation_system_message() {
    let request = completion_request_with_history(
        vec![
            message::Message::from("Review this code."),
            message::Message::System {
                content: "From now on, require explicit type annotations.".to_string(),
            },
        ],
        None,
    );

    let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
        model: CLAUDE_OPUS_4_7,
        request,
        prompt_caching: false,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    assert_eq!(
        value["system"][0]["text"],
        "From now on, require explicit type annotations."
    );

    let messages = value["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["role"], "user");
}

#[test]
fn test_tool_definition_cache_control_serialization() {
    let tool = ToolDefinition {
        name: "cached_tool".to_string(),
        description: Some("Cached tool".to_string()),
        input_schema: json!({"type": "object"}),
        strict: false,
        cache_control: Some(CacheControl::ephemeral()),
    };

    let value = serde_json::to_value(tool).unwrap();
    assert_eq!(value["cache_control"]["type"], "ephemeral");

    let tool_without_cache = ToolDefinition {
        name: "uncached_tool".to_string(),
        description: Some("Uncached tool".to_string()),
        input_schema: json!({"type": "object"}),
        strict: false,
        cache_control: None,
    };

    let value = serde_json::to_value(tool_without_cache).unwrap();
    assert!(value.get("cache_control").is_none());
}

#[test]
fn test_apply_tool_cache_control_marks_only_final_tool() {
    let mut tools = vec![
        json!({
            "name": "first_tool",
            "description": "First tool",
            "input_schema": {"type": "object"}
        }),
        json!({
            "name": "second_tool",
            "description": "Second tool",
            "input_schema": {"type": "object"}
        }),
    ];

    let mut remaining_cache_markers = 4;
    apply_tool_cache_control(
        &mut tools,
        &mut remaining_cache_markers,
        &CacheControl::ephemeral(),
    )
    .unwrap();

    assert!(tools[0].get("cache_control").is_none());
    assert_eq!(tools[1]["cache_control"]["type"], "ephemeral");
    assert_eq!(remaining_cache_markers, 3);
}

#[test]
fn test_prompt_caching_skips_final_deferred_tool_in_request() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [
                {
                    "name": "regular_tool",
                    "description": "Regular tool",
                    "input_schema": {"type": "object"}
                },
                {
                    "name": "deferred_tool",
                    "description": "Deferred tool",
                    "input_schema": {"type": "object"},
                    "defer_loading": true
                }
            ]
        })),
    );

    let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["name"], "regular_tool");
    assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
    assert_eq!(tools[1]["name"], "deferred_tool");
    assert!(tools[1].get("cache_control").is_none());
}

#[test]
fn test_prompt_caching_preserves_existing_final_tool_cache_control() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [{
                "name": "cached_tool",
                "description": "Cached tool",
                "input_schema": {"type": "object"},
                "cache_control": {"type": "ephemeral", "ttl": "1h"}
            }]
        })),
    );

    let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
    assert_eq!(tools[0]["cache_control"]["ttl"], "1h");
}

#[test]
fn test_prompt_caching_all_deferred_tools_do_not_receive_cache_control() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [
                {
                    "name": "first_deferred_tool",
                    "description": "First deferred tool",
                    "input_schema": {"type": "object"},
                    "defer_loading": true
                },
                {
                    "name": "second_deferred_tool",
                    "description": "Second deferred tool",
                    "input_schema": {"type": "object"},
                    "defer_loading": true
                }
            ]
        })),
    );

    let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert!(tools[0].get("cache_control").is_none());
    assert!(tools[1].get("cache_control").is_none());
}

#[test]
fn test_prompt_caching_preserves_earlier_tool_cache_control() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [
                {
                    "name": "earlier_tool",
                    "description": "Earlier tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral", "ttl": "1h"}
                },
                {
                    "name": "later_tool",
                    "description": "Later tool",
                    "input_schema": {"type": "object"}
                }
            ]
        })),
    );

    let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
    assert_eq!(tools[0]["cache_control"]["ttl"], "1h");
    assert_eq!(tools[1]["cache_control"]["type"], "ephemeral");
}

#[test]
fn test_prompt_caching_deferred_marker_does_not_suppress_loaded_tool_marker() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [
                {
                    "name": "regular_tool",
                    "description": "Regular tool",
                    "input_schema": {"type": "object"}
                },
                {
                    "name": "deferred_cached_tool",
                    "description": "Deferred cached tool",
                    "input_schema": {"type": "object"},
                    "defer_loading": true,
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        })),
    );

    let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
    assert_eq!(tools[1]["cache_control"]["type"], "ephemeral");
}

#[test]
fn test_prompt_caching_errors_when_tool_cache_control_ttl_order_is_invalid() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [
                {
                    "name": "first_cached_tool",
                    "description": "First cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "second_cached_tool",
                    "description": "Second cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral", "ttl": "1h"}
                }
            ]
        })),
    );

    let err = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap_err();

    assert!(err.to_string().contains("ttl `1h`"));
}

#[test]
fn test_prompt_caching_preserves_valid_mixed_ttl_tool_cache_controls() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [
                {
                    "name": "first_cached_tool",
                    "description": "First cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral", "ttl": "1h"}
                },
                {
                    "name": "second_cached_tool",
                    "description": "Second cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        })),
    );

    let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
    assert_eq!(tools[0]["cache_control"]["ttl"], "1h");
    assert_eq!(tools[1]["cache_control"]["type"], "ephemeral");
    assert!(tools[1]["cache_control"].get("ttl").is_none());
}

#[test]
fn test_prompt_caching_preserves_deferred_tool_cache_control() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [{
                "name": "deferred_cached_tool",
                "description": "Deferred cached tool",
                "input_schema": {"type": "object"},
                "defer_loading": true,
                "cache_control": {"type": "ephemeral"}
            }]
        })),
    );

    let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
}

#[test]
fn test_prompt_caching_budget_preserves_three_tool_markers_and_skips_message() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [
                {
                    "name": "first_cached_tool",
                    "description": "First cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "second_cached_tool",
                    "description": "Second cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "third_cached_tool",
                    "description": "Third cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        })),
    );

    let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
    assert_eq!(tools[1]["cache_control"]["type"], "ephemeral");
    assert_eq!(tools[2]["cache_control"]["type"], "ephemeral");
    assert!(system_has_cache_control(&value));
    assert!(!last_message_has_cache_control(&value));
}

#[test]
fn test_prompt_caching_errors_when_explicit_tool_markers_exceed_budget() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [
                {
                    "name": "first_cached_tool",
                    "description": "First cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "second_cached_tool",
                    "description": "Second cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "third_cached_tool",
                    "description": "Third cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "fourth_cached_tool",
                    "description": "Fourth cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "fifth_cached_tool",
                    "description": "Fifth cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        })),
    );

    let err = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap_err();

    assert!(err.to_string().contains("Too many Anthropic tool"));
}

#[test]
fn test_prompt_caching_errors_when_final_tool_marker_has_no_budget() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [
                {
                    "name": "first_cached_tool",
                    "description": "First cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "second_cached_tool",
                    "description": "Second cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "third_cached_tool",
                    "description": "Third cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "fourth_cached_tool",
                    "description": "Fourth cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "final_uncached_tool",
                    "description": "Final uncached tool",
                    "input_schema": {"type": "object"}
                }
            ]
        })),
    );

    let err = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap_err();

    assert!(err.to_string().contains("final non-deferred tool"));
}

#[test]
fn test_prompt_caching_replaces_null_final_tool_cache_control() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [{
                "name": "final_tool",
                "description": "Final tool",
                "input_schema": {"type": "object"},
                "cache_control": null
            }]
        })),
    );

    let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
}

#[test]
fn test_prompt_caching_ignores_null_tool_cache_control_when_budgeting() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [
                {
                    "name": "first_null_tool",
                    "description": "First null tool",
                    "input_schema": {"type": "object"},
                    "cache_control": null
                },
                {
                    "name": "second_null_tool",
                    "description": "Second null tool",
                    "input_schema": {"type": "object"},
                    "cache_control": null
                },
                {
                    "name": "third_null_tool",
                    "description": "Third null tool",
                    "input_schema": {"type": "object"},
                    "cache_control": null
                },
                {
                    "name": "fourth_null_tool",
                    "description": "Fourth null tool",
                    "input_schema": {"type": "object"},
                    "cache_control": null
                },
                {
                    "name": "final_uncached_tool",
                    "description": "Final uncached tool",
                    "input_schema": {"type": "object"}
                }
            ]
        })),
    );

    let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert!(tools[0].get("cache_control").is_none());
    assert!(tools[1].get("cache_control").is_none());
    assert!(tools[2].get("cache_control").is_none());
    assert!(tools[3].get("cache_control").is_none());
    assert_eq!(tools[4]["cache_control"]["type"], "ephemeral");
}

#[test]
fn test_prompt_caching_preserves_non_null_provider_tool_cache_control_escape_hatch() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [{
                "name": "provider_tool",
                "description": "Provider tool",
                "input_schema": {"type": "object"},
                "cache_control": {"type": "provider_specific"}
            }]
        })),
    );

    let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["type"], "provider_specific");
}

#[test]
fn test_prompt_caching_automatic_mode_uses_reduced_marker_budget() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [
                {
                    "name": "first_cached_tool",
                    "description": "First cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "second_cached_tool",
                    "description": "Second cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "third_cached_tool",
                    "description": "Third cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        })),
    );

    let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        automatic_caching: true,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
    assert_eq!(tools[1]["cache_control"]["type"], "ephemeral");
    assert_eq!(tools[2]["cache_control"]["type"], "ephemeral");
    assert_eq!(value["cache_control"]["type"], "ephemeral");
    assert!(!system_has_cache_control(&value));
    assert!(!last_message_has_cache_control(&value));
}

#[test]
fn test_prompt_caching_automatic_mode_errors_when_final_tool_marker_has_no_budget() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [
                {
                    "name": "first_cached_tool",
                    "description": "First cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "second_cached_tool",
                    "description": "Second cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "third_cached_tool",
                    "description": "Third cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "final_uncached_tool",
                    "description": "Final uncached tool",
                    "input_schema": {"type": "object"}
                }
            ]
        })),
    );

    let err = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        automatic_caching: true,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap_err();

    assert!(err.to_string().contains("final non-deferred tool"));
}

#[test]
fn test_automatic_caching_errors_when_explicit_tool_markers_exhaust_budget() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [
                {
                    "name": "first_cached_tool",
                    "description": "First cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "second_cached_tool",
                    "description": "Second cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "third_cached_tool",
                    "description": "Third cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "fourth_cached_tool",
                    "description": "Fourth cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        })),
    );

    let err = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: false,
        automatic_caching: true,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap_err();

    assert!(err.to_string().contains("Too many Anthropic tool"));
}

#[test]
fn test_automatic_caching_1h_errors_with_explicit_five_minute_tool_marker() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [{
                "name": "cached_tool",
                "description": "Cached tool",
                "input_schema": {"type": "object"},
                "cache_control": {"type": "ephemeral"}
            }]
        })),
    );

    let err = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: false,
        automatic_caching: true,
        automatic_caching_ttl: Some(CacheTtl::OneHour),
        static_prefix_cache_ttl: None,
    })
    .unwrap_err();

    assert!(err.to_string().contains("ttl `1h`"));
}

#[test]
fn test_prompt_and_automatic_caching_1h_uses_1h_generated_markers() {
    let request = completion_request_with_tools(vec![generic_tool("cached_tool")], None);

    let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        automatic_caching: true,
        automatic_caching_ttl: Some(CacheTtl::OneHour),
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
    assert_eq!(tools[0]["cache_control"]["ttl"], "1h");
    assert_eq!(
        value["system"]
            .as_array()
            .and_then(|blocks| blocks.last())
            .and_then(|block| block["cache_control"].get("ttl")),
        Some(&json!("1h"))
    );
    assert_eq!(value["cache_control"]["ttl"], "1h");
    assert!(!last_message_has_cache_control(&value));
}

#[test]
fn test_prompt_and_raw_top_level_automatic_caching_1h_uses_1h_generated_markers() {
    let request = completion_request_with_tools(
        vec![generic_tool("cached_tool")],
        Some(json!({
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
            "metadata": {"source": "test"}
        })),
    );

    let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        automatic_caching: true,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
    assert_eq!(tools[0]["cache_control"]["ttl"], "1h");
    assert_eq!(
        value["system"]
            .as_array()
            .and_then(|blocks| blocks.last())
            .and_then(|block| block["cache_control"].get("ttl")),
        Some(&json!("1h"))
    );
    assert_eq!(value["cache_control"]["ttl"], "1h");
    assert_eq!(value["metadata"]["source"], "test");
    assert!(!last_message_has_cache_control(&value));
}

#[test]
fn test_prompt_caching_uses_raw_top_level_cache_control_ttl() {
    let request = completion_request_with_tools(
        vec![generic_tool("cached_tool")],
        Some(json!({
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
            "metadata": {"source": "raw-cache-control"}
        })),
    );

    let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
    assert_eq!(tools[0]["cache_control"]["ttl"], "1h");
    assert_eq!(
        value["system"]
            .as_array()
            .and_then(|blocks| blocks.last())
            .and_then(|block| block["cache_control"].get("ttl")),
        Some(&json!("1h"))
    );
    assert_eq!(value["cache_control"]["ttl"], "1h");
    assert_eq!(value["metadata"]["source"], "raw-cache-control");
    assert!(!last_message_has_cache_control(&value));
}

#[test]
fn test_static_prefix_ttl_with_manual_caching_splits_prefix_and_tail() {
    let request = completion_request_with_tools(vec![generic_tool("cached_tool")], None);

    let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: Some(CacheTtl::OneHour),
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["ttl"], "1h");
    assert_eq!(
        value["system"]
            .as_array()
            .and_then(|blocks| blocks.last())
            .and_then(|block| block["cache_control"].get("ttl")),
        Some(&json!("1h"))
    );
    // The tail keeps the 5-minute default: a marker with no `ttl` field.
    let tail_cache_control = value["messages"]
        .as_array()
        .and_then(|messages| messages.last())
        .and_then(|message| message["content"].as_array())
        .and_then(|content| content.last())
        .map(|block| &block["cache_control"])
        .unwrap();
    assert_eq!(tail_cache_control["type"], "ephemeral");
    assert!(tail_cache_control.get("ttl").is_none());
    assert!(value.get("cache_control").is_none());
}

#[test]
fn test_static_prefix_ttl_with_automatic_caching_marks_prefix_only() {
    let request = completion_request_with_tools(vec![generic_tool("cached_tool")], None);

    let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: false,
        automatic_caching: true,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: Some(CacheTtl::OneHour),
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["ttl"], "1h");
    assert_eq!(
        value["system"]
            .as_array()
            .and_then(|blocks| blocks.last())
            .and_then(|block| block["cache_control"].get("ttl")),
        Some(&json!("1h"))
    );
    // The moving tail breakpoint is Anthropic's top-level one at the
    // 5-minute default; no explicit message marker exists.
    assert_eq!(value["cache_control"]["type"], "ephemeral");
    assert!(value["cache_control"].get("ttl").is_none());
    assert!(!last_message_has_cache_control(&value));
}

#[test]
fn test_static_prefix_ttl_alone_marks_prefix_without_tail_or_top_level() {
    let request = completion_request_with_tools(vec![generic_tool("cached_tool")], None);

    let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: false,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: Some(CacheTtl::OneHour),
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["ttl"], "1h");
    assert_eq!(
        value["system"]
            .as_array()
            .and_then(|blocks| blocks.last())
            .and_then(|block| block["cache_control"].get("ttl")),
        Some(&json!("1h"))
    );
    assert!(value.get("cache_control").is_none());
    assert!(!last_message_has_cache_control(&value));
}

#[test]
fn test_static_prefix_ttl_five_minutes_with_automatic_1h_errors_client_side() {
    let request = completion_request_with_tools(vec![generic_tool("cached_tool")], None);

    let error = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: false,
        automatic_caching: true,
        automatic_caching_ttl: Some(CacheTtl::OneHour),
        static_prefix_cache_ttl: Some(CacheTtl::FiveMinutes),
    })
    .unwrap_err();

    let message = error.to_string();
    assert!(
        message.contains("with_static_prefix_cache_ttl"),
        "error should name the knob: {message}"
    );
    assert!(
        message.contains("with_automatic_caching_1h"),
        "error should name the conflicting knob: {message}"
    );
}

#[test]
fn test_static_prefix_ttl_five_minutes_matches_automatic_default_ttl() {
    let request = completion_request_with_tools(vec![generic_tool("cached_tool")], None);

    // 5m prefix + 5m (default) top-level is uniform, not an inversion.
    let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: false,
        automatic_caching: true,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: Some(CacheTtl::FiveMinutes),
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
    // The explicit knob serializes an explicit `"5m"`, equivalent to the
    // omitted-`ttl` default.
    assert_eq!(tools[0]["cache_control"]["ttl"], "5m");
}

#[test]
fn test_static_prefix_ttl_preserves_marker_budget_arithmetic() {
    // Automatic mode reserves one marker for the top-level breakpoint; the
    // static-prefix knob spends from the same remaining budget as manual
    // caching does — two markers (final tool + system), no more.
    let request = completion_request_with_tools(vec![generic_tool("cached_tool")], None);

    let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: false,
        automatic_caching: true,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: Some(CacheTtl::OneHour),
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let marker_count = value["tools"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|tool| !tool["cache_control"].is_null())
        .count()
        + value["system"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|block| !block["cache_control"].is_null())
            .count()
        + usize::from(!value["cache_control"].is_null());
    assert_eq!(marker_count, 3);
    assert!(marker_count <= MAX_CACHE_CONTROL_MARKERS);
}

#[test]
fn test_usage_parses_per_ttl_cache_creation_breakdown() {
    let usage: Usage = serde_json::from_str(
        r#"{
                "input_tokens": 3,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 9677,
                "cache_creation": {
                    "ephemeral_5m_input_tokens": 9677,
                    "ephemeral_1h_input_tokens": 0,
                    "ephemeral_24h_input_tokens": 0
                },
                "output_tokens": 7
            }"#,
    )
    .unwrap();

    assert_eq!(usage.cache_creation_input_tokens, Some(9677));
    let cache_creation = usage.cache_creation.unwrap();
    assert_eq!(cache_creation.ephemeral_5m_input_tokens, 9677);
    assert_eq!(cache_creation.ephemeral_1h_input_tokens, 0);
}

#[test]
fn test_usage_without_cache_creation_breakdown_parses_as_none() {
    let usage: Usage = serde_json::from_str(r#"{"input_tokens": 3, "output_tokens": 7}"#).unwrap();
    assert!(usage.cache_creation.is_none());
}

#[test]
fn test_raw_top_level_automatic_caching_reduces_marker_budget() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "cache_control": {"type": "ephemeral"},
            "tools": [
                {
                    "name": "first_cached_tool",
                    "description": "First cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "second_cached_tool",
                    "description": "Second cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "third_cached_tool",
                    "description": "Third cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "fourth_cached_tool",
                    "description": "Fourth cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        })),
    );

    let err = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: false,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap_err();

    assert!(err.to_string().contains("Too many Anthropic tool"));
}

#[test]
fn test_raw_top_level_automatic_caching_1h_errors_after_explicit_five_minute_tool_marker() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
            "tools": [{
                "name": "cached_tool",
                "description": "Cached tool",
                "input_schema": {"type": "object"},
                "cache_control": {"type": "ephemeral"}
            }]
        })),
    );

    let err = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: false,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap_err();

    assert!(err.to_string().contains("ttl `1h`"));
}

#[test]
fn test_typed_automatic_caching_ttl_errors_on_conflicting_raw_top_level_ttl() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "cache_control": {"type": "ephemeral"}
        })),
    );

    let err = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: false,
        automatic_caching: true,
        automatic_caching_ttl: Some(CacheTtl::OneHour),
        static_prefix_cache_ttl: None,
    })
    .unwrap_err();

    assert!(
        err.to_string()
            .contains("conflicts with the typed automatic caching TTL")
    );
}

#[test]
fn test_prompt_caching_marks_final_tool_in_request() {
    let request = completion_request_with_tools(
        vec![generic_tool("first_tool"), generic_tool("second_tool")],
        None,
    );

    let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 2);
    assert!(tools[0].get("cache_control").is_none());
    assert_eq!(tools[1]["cache_control"]["type"], "ephemeral");
}

#[test]
fn test_prompt_caching_marks_final_additional_tool_in_request() {
    let request = completion_request_with_tools(
        vec![generic_tool("rig_tool")],
        Some(json!({
            "tools": [{
                "name": "provider_tool",
                "description": "Provider tool",
                "input_schema": {"type": "object"}
            }],
            "metadata": {"source": "test"}
        })),
    );

    let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 2);
    assert!(tools[0].get("cache_control").is_none());
    assert_eq!(tools[1]["name"], "provider_tool");
    assert_eq!(tools[1]["cache_control"]["type"], "ephemeral");
    assert_eq!(value["metadata"]["source"], "test");
}

#[test]
fn test_prompt_caching_without_tools_omits_tools() {
    let request = completion_request_with_tools(Vec::new(), None);

    let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    assert!(value.get("tools").is_none());
}

#[test]
fn test_plaintext_document_serialization() {
    let content = Content::Document {
        source: DocumentSource::Text {
            data: "Hello, world!".to_string(),
            media_type: PlainTextMediaType::Plain,
        },
        title: None,
        context: None,
        citations: None,
        cache_control: None,
    };

    let json = serde_json::to_value(&content).unwrap();
    assert_eq!(json["type"], "document");
    assert_eq!(json["source"]["type"], "text");
    assert_eq!(json["source"]["media_type"], "text/plain");
    assert_eq!(json["source"]["data"], "Hello, world!");
}

#[test]
fn test_plaintext_document_deserialization() {
    let json = r#"
        {
            "type": "document",
            "source": {
                "type": "text",
                "media_type": "text/plain",
                "data": "Hello, world!"
            }
        }
        "#;

    let content: Content = serde_json::from_str(json).unwrap();
    match content {
        Content::Document {
            source,
            cache_control,
            ..
        } => {
            assert_eq!(
                source,
                DocumentSource::Text {
                    data: "Hello, world!".to_string(),
                    media_type: PlainTextMediaType::Plain,
                }
            );
            assert_eq!(cache_control, None);
        }
        _ => panic!("Expected Document content"),
    }
}

#[test]
fn test_base64_pdf_document_serialization() {
    let content = Content::Document {
        source: DocumentSource::Base64 {
            data: "base64data".to_string(),
            media_type: DocumentFormat::PDF,
        },
        title: None,
        context: None,
        citations: None,
        cache_control: None,
    };

    let json = serde_json::to_value(&content).unwrap();
    assert_eq!(json["type"], "document");
    assert_eq!(json["source"]["type"], "base64");
    assert_eq!(json["source"]["media_type"], "application/pdf");
    assert_eq!(json["source"]["data"], "base64data");
}

#[test]
fn test_base64_pdf_document_deserialization() {
    let json = r#"
        {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": "base64data"
            }
        }
        "#;

    let content: Content = serde_json::from_str(json).unwrap();
    match content {
        Content::Document { source, .. } => {
            assert_eq!(
                source,
                DocumentSource::Base64 {
                    data: "base64data".to_string(),
                    media_type: DocumentFormat::PDF,
                }
            );
        }
        _ => panic!("Expected Document content"),
    }
}

#[test]
fn test_file_id_document_serialization() {
    let content = Content::Document {
        source: DocumentSource::File {
            file_id: "file_abc".to_string(),
        },
        title: None,
        context: None,
        citations: None,
        cache_control: None,
    };

    let json = serde_json::to_value(&content).unwrap();
    assert_eq!(json["type"], "document");
    assert_eq!(json["source"]["type"], "file");
    assert_eq!(json["source"]["file_id"], "file_abc");
}

#[test]
fn test_file_id_document_deserialization() {
    let json = r#"
        {
            "type": "document",
            "source": {
                "type": "file",
                "file_id": "file_abc"
            }
        }
        "#;

    let content: Content = serde_json::from_str(json).unwrap();
    match content {
        Content::Document { source, .. } => {
            assert_eq!(
                source,
                DocumentSource::File {
                    file_id: "file_abc".to_string(),
                }
            );
        }
        _ => panic!("Expected Document content"),
    }
}

#[test]
fn test_file_id_rig_to_anthropic_conversion() {
    use crate::completion::message as msg;

    let rig_message = msg::Message::User {
        content: vec![msg::UserContent::Document(msg::Document {
            data: DocumentSourceKind::FileId("file_abc".to_string()),
            media_type: None,
            additional_params: None,
        })],
    };

    let anthropic_message: Message = rig_message.try_into().unwrap();
    assert_eq!(anthropic_message.role, Role::User);

    let mut iter = anthropic_message.content.into_iter();
    match iter.next().unwrap() {
        Content::Document { source, .. } => {
            assert_eq!(
                source,
                DocumentSource::File {
                    file_id: "file_abc".to_string(),
                }
            );
        }
        other => panic!("Expected Document content, got: {other:?}"),
    }
}

#[test]
fn test_file_id_anthropic_to_rig_conversion() {
    use crate::completion::message as msg;

    let anthropic_message = Message {
        role: Role::User,
        content: vec![Content::Document {
            source: DocumentSource::File {
                file_id: "file_abc".to_string(),
            },
            title: None,
            context: None,
            citations: None,
            cache_control: None,
        }],
    };

    let rig_message: msg::Message = anthropic_message.try_into().unwrap();
    match rig_message {
        msg::Message::User { content } => {
            let mut iter = content.into_iter();
            match iter.next().unwrap() {
                msg::UserContent::Document(msg::Document {
                    data, media_type, ..
                }) => {
                    assert_eq!(data, DocumentSourceKind::FileId("file_abc".to_string()));
                    assert_eq!(media_type, None);
                }
                other => panic!("Expected Document content, got: {other:?}"),
            }
        }
        _ => panic!("Expected User message"),
    }
}

#[test]
fn test_plaintext_rig_to_anthropic_conversion() {
    use crate::completion::message as msg;

    let rig_message = msg::Message::User {
        content: vec![msg::UserContent::document(
            "Some plain text content".to_string(),
            Some(msg::DocumentMediaType::TXT),
        )],
    };

    let anthropic_message: Message = rig_message.try_into().unwrap();
    assert_eq!(anthropic_message.role, Role::User);

    let mut iter = anthropic_message.content.into_iter();
    match iter.next().unwrap() {
        Content::Document { source, .. } => {
            assert_eq!(
                source,
                DocumentSource::Text {
                    data: "Some plain text content".to_string(),
                    media_type: PlainTextMediaType::Plain,
                }
            );
        }
        other => panic!("Expected Document content, got: {other:?}"),
    }
}

#[test]
fn test_plaintext_anthropic_to_rig_conversion() {
    use crate::completion::message as msg;

    let anthropic_message = Message {
        role: Role::User,
        content: vec![Content::Document {
            source: DocumentSource::Text {
                data: "Some plain text content".to_string(),
                media_type: PlainTextMediaType::Plain,
            },
            title: None,
            context: None,
            citations: None,
            cache_control: None,
        }],
    };

    let rig_message: msg::Message = anthropic_message.try_into().unwrap();
    match rig_message {
        msg::Message::User { content } => {
            let mut iter = content.into_iter();
            match iter.next().unwrap() {
                msg::UserContent::Document(msg::Document {
                    data, media_type, ..
                }) => {
                    assert_eq!(
                        data,
                        DocumentSourceKind::String("Some plain text content".into())
                    );
                    assert_eq!(media_type, Some(msg::DocumentMediaType::TXT));
                }
                other => panic!("Expected Document content, got: {other:?}"),
            }
        }
        _ => panic!("Expected User message"),
    }
}

#[test]
fn test_plaintext_roundtrip_rig_to_anthropic_and_back() {
    use crate::completion::message as msg;

    let original = msg::Message::User {
        content: vec![msg::UserContent::document(
            "Round trip text".to_string(),
            Some(msg::DocumentMediaType::TXT),
        )],
    };

    let anthropic: Message = original.clone().try_into().unwrap();
    let back: msg::Message = anthropic.try_into().unwrap();

    match (&original, &back) {
        (
            msg::Message::User {
                content: orig_content,
            },
            msg::Message::User {
                content: back_content,
            },
        ) => match (orig_content.first(), back_content.first()) {
            (
                Some(msg::UserContent::Document(msg::Document {
                    media_type: orig_mt,
                    ..
                })),
                Some(msg::UserContent::Document(msg::Document {
                    media_type: back_mt,
                    ..
                })),
            ) => {
                assert_eq!(orig_mt, back_mt);
            }
            _ => panic!("Expected Document content in both"),
        },
        _ => panic!("Expected User messages"),
    }
}

#[test]
fn test_unsupported_document_type_returns_error() {
    use crate::completion::message as msg;

    let rig_message = msg::Message::User {
        content: vec![msg::UserContent::Document(msg::Document {
            data: DocumentSourceKind::String("data".into()),
            media_type: Some(msg::DocumentMediaType::HTML),
            additional_params: None,
        })],
    };

    let result: Result<Message, _> = rig_message.try_into();
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("Anthropic only supports PDF and plain text documents"),
        "Unexpected error: {err}"
    );
}

#[test]
fn test_plaintext_document_url_source_returns_error() {
    use crate::completion::message as msg;

    let rig_message = msg::Message::User {
        content: vec![msg::UserContent::Document(msg::Document {
            data: DocumentSourceKind::Url("https://example.com/doc.txt".into()),
            media_type: Some(msg::DocumentMediaType::TXT),
            additional_params: None,
        })],
    };

    let result: Result<Message, _> = rig_message.try_into();
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("Only string or base64 data is supported for plain text documents"),
        "Unexpected error: {err}"
    );
}

#[test]
fn test_plaintext_document_with_cache_control() {
    let content = Content::Document {
        source: DocumentSource::Text {
            data: "cached text".to_string(),
            media_type: PlainTextMediaType::Plain,
        },
        title: None,
        context: None,
        citations: None,
        cache_control: Some(CacheControl::ephemeral()),
    };

    let json = serde_json::to_value(&content).unwrap();
    assert_eq!(json["source"]["type"], "text");
    assert_eq!(json["source"]["media_type"], "text/plain");
    assert_eq!(json["cache_control"]["type"], "ephemeral");
}

#[test]
fn test_message_with_plaintext_document_deserialization() {
    let json = r#"
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "text",
                        "media_type": "text/plain",
                        "data": "Hello from a text file"
                    }
                },
                {
                    "type": "text",
                    "text": "Summarize this document."
                }
            ]
        }
        "#;

    let message: Message = serde_json::from_str(json).unwrap();
    assert_eq!(message.role, Role::User);
    assert_eq!(message.content.len(), 2);

    let mut iter = message.content.into_iter();

    match iter.next().unwrap() {
        Content::Document { source, .. } => {
            assert_eq!(
                source,
                DocumentSource::Text {
                    data: "Hello from a text file".to_string(),
                    media_type: PlainTextMediaType::Plain,
                }
            );
        }
        _ => panic!("Expected Document content"),
    }

    match iter.next().unwrap() {
        Content::Text { text, .. } => {
            assert_eq!(text, "Summarize this document.");
        }
        _ => panic!("Expected Text content"),
    }
}

#[test]
fn test_assistant_reasoning_multiblock_to_anthropic_content() {
    let reasoning = message::Reasoning {
        id: None,
        content: vec![
            message::ReasoningContent::Text {
                text: "step one".to_string(),
                signature: Some("sig-1".to_string()),
            },
            message::ReasoningContent::Summary("summary".to_string()),
            message::ReasoningContent::Text {
                text: "step two".to_string(),
                signature: Some("sig-2".to_string()),
            },
            message::ReasoningContent::Redacted {
                data: "redacted block".to_string(),
            },
        ],
    };

    let msg = message::Message::Assistant {
        id: None,
        content: vec![message::AssistantContent::Reasoning(reasoning)],
    };
    let converted: Message = msg.try_into().expect("convert assistant message");
    let converted_content = converted.content.clone();

    assert_eq!(converted.role, Role::Assistant);
    assert_eq!(converted_content.len(), 4);
    assert!(matches!(
        converted_content.first(),
        Some(Content::Thinking { thinking, signature: Some(signature) })
            if thinking == "step one" && signature == "sig-1"
    ));
    assert!(matches!(
        converted_content.get(1),
        Some(Content::Thinking { thinking, signature: None }) if thinking == "summary"
    ));
    assert!(matches!(
        converted_content.get(2),
        Some(Content::Thinking { thinking, signature: Some(signature) })
            if thinking == "step two" && signature == "sig-2"
    ));
    assert!(matches!(
        converted_content.get(3),
        Some(Content::RedactedThinking { data }) if data == "redacted block"
    ));
}

#[test]
fn test_redacted_thinking_content_to_assistant_reasoning() {
    let content = Content::RedactedThinking {
        data: "opaque-redacted".to_string(),
    };
    let converted: message::AssistantContent =
        content.try_into().expect("convert redacted thinking");

    assert!(matches!(
        converted,
        message::AssistantContent::Reasoning(message::Reasoning { content, .. })
            if matches!(
                content.first(),
                Some(message::ReasoningContent::Redacted { data }) if data == "opaque-redacted"
            )
    ));
}

#[test]
fn test_assistant_encrypted_reasoning_maps_to_redacted_thinking() {
    let reasoning = message::Reasoning {
        id: None,
        content: vec![message::ReasoningContent::Encrypted(
            "ciphertext".to_string(),
        )],
    };
    let msg = message::Message::Assistant {
        id: None,
        content: vec![message::AssistantContent::Reasoning(reasoning)],
    };

    let converted: Message = msg.try_into().expect("convert assistant message");
    let converted_content = converted.content;

    assert_eq!(converted_content.len(), 1);
    assert!(matches!(
        converted_content.first(),
        Some(Content::RedactedThinking { data }) if data == "ciphertext"
    ));
}

#[test]
fn empty_end_turn_response_normalizes_to_an_empty_choice() {
    let response = CompletionResponse {
        content: vec![],
        id: "msg_123".to_string(),
        model: CLAUDE_SONNET_4_6.to_string(),
        role: "assistant".to_string(),
        stop_reason: Some("end_turn".to_string()),
        stop_sequence: None,
        provider_request_id: None,
        usage: Usage {
            input_tokens: 7,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
            cache_creation: None,
            output_tokens: 2,
            output_tokens_details: None,
        },
    };

    let parsed: completion::CompletionResponse = response
        .normalize("anthropic")
        .expect("empty end_turn should not error");

    // Anthropic's documented empty `end_turn` is a turn that carried
    // nothing. It used to normalize to one fabricated empty-text part
    // because the content type could not be empty; the empty list is the
    // same turn, said honestly. Everything else about the response is
    // unchanged, which is the point of asserting it here.
    assert!(parsed.choice.is_empty());
    assert_eq!(parsed.provider, "anthropic");
    assert_eq!(parsed.message_id.as_deref(), Some("msg_123"));
    assert_eq!(parsed.model.as_deref(), Some(CLAUDE_SONNET_4_6));
    assert_eq!(parsed.finish_reason(), Some(completion::FinishReason::Stop));
}

/// Build an empty-content response with the given terminal, for exercising
/// the two legal empty cases against everything else.
fn empty_response_with(
    stop_reason: Option<&str>,
    stop_sequence: Option<&str>,
) -> CompletionResponse {
    CompletionResponse {
        content: vec![],
        id: "msg_123".to_string(),
        model: CLAUDE_SONNET_4_6.to_string(),
        role: "assistant".to_string(),
        stop_reason: stop_reason.map(str::to_string),
        stop_sequence: stop_sequence.map(str::to_string),
        provider_request_id: None,
        usage: Usage {
            input_tokens: 7,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
            cache_creation: None,
            output_tokens: 2,
            output_tokens_details: None,
        },
    }
}

#[test]
fn empty_response_outside_the_legal_terminals_still_errors() {
    for (stop_reason, stop_sequence) in [
        (Some("tool_use"), None),
        (Some("max_tokens"), None),
        (Some("refusal"), None),
        (Some("pause_turn"), None),
        (None, None),
        // Claims to have stopped on a sequence but names none: the
        // malformed shape the guard exists for, not a legal empty turn.
        (Some("stop_sequence"), None),
        // The inverse: naming a sequence does not make an illegal terminal
        // legal. The carve-out gates on the reason first, then the field.
        (Some("max_tokens"), Some("alpha")),
    ] {
        let err = empty_response_with(stop_reason, stop_sequence)
            .normalize("anthropic")
            .expect_err(&format!(
                "empty {stop_reason:?} response should remain an error"
            ));

        assert!(matches!(
            err,
            CompletionError::ResponseError(message) if message == EMPTY_RESPONSE_ERROR
        ));
    }
}

#[test]
fn empty_stop_sequence_response_naming_its_sequence_is_a_completed_turn() {
    let parsed = empty_response_with(Some("stop_sequence"), Some("alpha"))
        .normalize("anthropic")
        .expect("a completed stop-sequence turn must not normalize into an error");

    assert!(parsed.choice.is_empty());
    assert_eq!(parsed.finish_reason(), Some(completion::FinishReason::Stop));
}

#[test]
fn stop_reason_maps_onto_the_normalized_vocabulary() {
    assert_eq!(
        map_finish_reason("end_turn"),
        completion::FinishReason::Stop
    );
    assert_eq!(
        map_finish_reason("stop_sequence"),
        completion::FinishReason::Stop
    );
    assert_eq!(
        map_finish_reason("max_tokens"),
        completion::FinishReason::Length
    );
    assert_eq!(
        map_finish_reason("tool_use"),
        completion::FinishReason::ToolCalls
    );
    assert_eq!(
        map_finish_reason("refusal"),
        completion::FinishReason::ContentFilter
    );
}

#[test]
fn unknown_stop_reason_is_preserved_verbatim() {
    // Anthropic's own spelling survives, so a reason this crate does not yet
    // model never reads as a natural stop.
    assert_eq!(
        map_finish_reason("pause_turn"),
        completion::FinishReason::Other("pause_turn".to_owned())
    );
    assert_eq!(
        map_finish_reason("model_context_window_exceeded"),
        completion::FinishReason::Other("model_context_window_exceeded".to_owned())
    );
}

#[test]
fn end_turn_with_a_tool_call_is_reconciled_to_tool_calls() {
    // Anthropic reports `tool_use`, but the reconciliation the response
    // builder applies must hold for any provider that reports a plain stop
    // alongside a tool call.
    let response = CompletionResponse {
        content: vec![Content::ToolUse {
            id: "toolu_1".to_string(),
            name: "add".to_string(),
            input: json!({"x": 1}),
        }],
        id: "msg_123".to_string(),
        model: CLAUDE_SONNET_4_6.to_string(),
        role: "assistant".to_string(),
        stop_reason: Some("end_turn".to_string()),
        stop_sequence: None,
        provider_request_id: None,
        usage: Usage {
            input_tokens: 7,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
            cache_creation: None,
            output_tokens: 2,
            output_tokens_details: None,
        },
    };

    let parsed = response
        .normalize("anthropic")
        .expect("tool-use response should normalize");

    assert_eq!(
        parsed.finish_reason(),
        Some(completion::FinishReason::ToolCalls)
    );
}

#[test]
fn test_tool_result_content_in_message_roundtrip() {
    let message_json = r#"{
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01A09q90qw90lq917835lq9",
                    "content": [
                        {
                            "type": "text",
                            "text": "Here is the screenshot:"
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "iVBORw0KGgo..."
                            }
                        }
                    ]
                }
            ]
        }"#;

    let message: Message = serde_json::from_str(message_json).unwrap();
    let serialized = serde_json::to_value(&message).unwrap();

    let tool_result = &serialized["content"][0];
    assert_eq!(tool_result["type"], "tool_result");

    let image_content = &tool_result["content"][1];
    assert_eq!(image_content["type"], "image");
    assert_eq!(image_content["source"]["type"], "base64");
    assert_eq!(image_content["source"]["media_type"], "image/png");
    assert_eq!(image_content["source"]["data"], "iVBORw0KGgo...");
}

// -------------------------------------------------------------------
// Citations (#1767)
// -------------------------------------------------------------------

#[test]
fn document_serializes_citations_and_metadata() {
    let doc = Content::Document {
        source: DocumentSource::Text {
            data: "hello".into(),
            media_type: PlainTextMediaType::Plain,
        },
        title: Some("My Doc".into()),
        context: None,
        citations: Some(CitationsConfig { enabled: true }),
        cache_control: None,
    };
    let value = serde_json::to_value(&doc).unwrap();
    assert_eq!(value["citations"]["enabled"], true);
    assert_eq!(value["title"], "My Doc");
    assert!(
        value.get("context").is_none(),
        "context should be skipped when None"
    );
}

#[test]
fn text_serializes_without_citations_when_empty() {
    let content = Content::Text {
        text: "hello".into(),
        citations: Vec::new(),
        cache_control: None,
    };
    let value = serde_json::to_value(&content).unwrap();
    assert!(
        value.get("citations").is_none(),
        "empty citations vec must be skipped"
    );
}

#[test]
fn text_deserializes_char_location_citation() {
    let value = json!({
        "type": "text",
        "text": "the grass is green",
        "citations": [{
            "type": "char_location",
            "cited_text": "The grass is green.",
            "document_index": 0,
            "document_title": "Example",
            "start_char_index": 0,
            "end_char_index": 20
        }]
    });
    let parsed: Content = serde_json::from_value(value).unwrap();
    let Content::Text { citations, .. } = parsed else {
        panic!("expected Content::Text");
    };
    assert_eq!(citations.len(), 1);
    let Citation::CharLocation(citation) = &citations[0] else {
        panic!("expected CharLocation");
    };
    assert_eq!(citation.start_char_index, 0);
    assert_eq!(citation.end_char_index, 20);
}

#[test]
fn text_deserializes_search_result_location_citation() {
    let value = json!({
        "type": "text",
        "text": "API keys are required.",
        "citations": [{
            "type": "search_result_location",
            "cited_text": "All API requests must include an API key.",
            "source": "https://docs.example.com/api-reference",
            "title": "API Reference",
            "search_result_index": 0,
            "start_block_index": 0,
            "end_block_index": 1
        }]
    });

    let parsed: Content = serde_json::from_value(value).unwrap();
    let Content::Text { citations, .. } = parsed else {
        panic!("expected Content::Text");
    };

    assert!(matches!(
        &citations[0],
        Citation::SearchResultLocation(SearchResultLocationCitation {
            source,
            title: Some(title),
            search_result_index: 0,
            start_block_index: 0,
            end_block_index: 1,
            ..
        }) if source == "https://docs.example.com/api-reference" && title == "API Reference"
    ));
}

#[test]
fn text_deserializes_web_search_result_location_citation() {
    let value = json!({
        "type": "text",
        "text": "Claude Shannon worked at Bell Labs.",
        "citations": [{
            "type": "web_search_result_location",
            "cited_text": "Claude Shannon was a mathematician.",
            "url": "https://example.com/shannon",
            "title": "Claude Shannon",
            "encrypted_index": "encrypted-reference"
        }]
    });

    let parsed: Content = serde_json::from_value(value).unwrap();
    let Content::Text { citations, .. } = parsed else {
        panic!("expected Content::Text");
    };

    assert!(matches!(
        &citations[0],
        Citation::WebSearchResultLocation(WebSearchResultLocationCitation {
            url,
            title,
            encrypted_index,
            ..
        }) if url == "https://example.com/shannon"
            && title.as_deref() == Some("Claude Shannon")
            && encrypted_index == "encrypted-reference"
    ));
}

#[test]
fn text_deserializes_web_search_result_location_citation_with_null_title() {
    let value = json!({
        "type": "text",
        "text": "Claude Shannon worked at Bell Labs.",
        "citations": [{
            "type": "web_search_result_location",
            "cited_text": "Claude Shannon was a mathematician.",
            "url": "https://example.com/shannon",
            "title": null,
            "encrypted_index": "encrypted-reference"
        }]
    });

    let parsed: Content = serde_json::from_value(value).unwrap();
    let Content::Text { citations, .. } = parsed else {
        panic!("expected Content::Text");
    };

    let Citation::WebSearchResultLocation(citation) = &citations[0] else {
        panic!("expected WebSearchResultLocation");
    };
    assert_eq!(citation.title, None);

    let serialized = serde_json::to_value(&citations[0]).unwrap();
    assert!(serialized.get("title").is_some());
    assert!(serialized["title"].is_null());
}

#[test]
fn web_search_response_preserves_raw_blocks_and_citations() {
    let value = json!({
        "id": "msg_web_search",
        "model": CLAUDE_SONNET_4_6,
        "role": "assistant",
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20
        },
        "content": [
            {
                "type": "server_tool_use",
                "id": "srvtoolu_01",
                "name": "web_search",
                "input": {
                    "query": "claude shannon birth date"
                }
            },
            {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_01",
                "content": [
                    {
                        "type": "web_search_result",
                        "url": "https://example.com/shannon",
                        "title": "Claude Shannon",
                        "encrypted_content": "encrypted-content",
                        "page_age": "April 30, 2025"
                    }
                ]
            },
            {
                "type": "text",
                "text": "Claude Shannon was born on April 30, 1916.",
                "citations": [{
                    "type": "web_search_result_location",
                    "cited_text": "Claude Shannon was born on April 30, 1916.",
                    "url": "https://example.com/shannon",
                    "title": "Claude Shannon",
                    "encrypted_index": "encrypted-index"
                }]
            }
        ]
    });

    let response: CompletionResponse = serde_json::from_value(value).unwrap();
    // The wire response is consumed by the conversion, so read the
    // provider-native text off it first.
    let raw_text_response = response.text_response();
    let converted = response.normalize("anthropic").unwrap();
    assert_eq!(converted.choice.len(), 3);
    assert_eq!(
        raw_text_response.as_deref(),
        Some("Claude Shannon was born on April 30, 1916.")
    );

    let items = converted.choice.iter().collect::<Vec<_>>();
    let message::AssistantContent::Text(server_tool_use) = items[0] else {
        panic!("expected raw server_tool_use metadata");
    };
    assert_eq!(server_tool_use.text, "");
    assert_eq!(
        server_tool_use.additional_params.as_ref().unwrap()[ANTHROPIC_RAW_CONTENT_KEY]["type"],
        "server_tool_use"
    );

    let message::AssistantContent::Text(web_search_result) = items[1] else {
        panic!("expected raw web_search_tool_result metadata");
    };
    assert_eq!(
        web_search_result.additional_params.as_ref().unwrap()[ANTHROPIC_RAW_CONTENT_KEY]["content"]
            [0]["encrypted_content"],
        "encrypted-content"
    );

    let message::AssistantContent::Text(answer) = items[2] else {
        panic!("expected text answer");
    };
    let citations = anthropic_citations(answer).unwrap();
    assert!(matches!(
        citations.first(),
        Some(Citation::WebSearchResultLocation(citation))
            if citation.encrypted_index == "encrypted-index"
    ));

    let round_trip: Message = message::Message::Assistant {
        id: converted.message_id.clone(),
        content: converted.choice,
    }
    .try_into()
    .unwrap();

    let round_trip_items = round_trip.content.iter().collect::<Vec<_>>();
    assert!(matches!(
        round_trip_items.first(),
        Some(Content::ServerToolUse { id, name, input })
            if id == "srvtoolu_01"
                && name == "web_search"
                && input["query"] == "claude shannon birth date"
    ));
    assert!(matches!(
        round_trip_items.get(1),
        Some(Content::WebSearchToolResult {
            tool_use_id,
            content
        }) if tool_use_id == "srvtoolu_01"
            && content[0]["encrypted_content"] == "encrypted-content"
    ));
}

#[test]
fn web_search_tool_result_error_object_is_preserved_raw() {
    let value = json!({
        "id": "msg_web_search_error",
        "model": CLAUDE_SONNET_4_6,
        "role": "assistant",
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 2
        },
        "content": [{
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_01",
            "content": {
                "type": "web_search_tool_result_error",
                "error_code": "max_uses_exceeded"
            }
        }]
    });

    let response: CompletionResponse = serde_json::from_value(value).unwrap();
    let converted = response.normalize("anthropic").unwrap();
    let Some(message::AssistantContent::Text(web_search_result)) = converted.choice.first() else {
        panic!("expected raw web_search_tool_result metadata");
    };

    let raw_content =
        &web_search_result.additional_params.as_ref().unwrap()[ANTHROPIC_RAW_CONTENT_KEY];
    assert_eq!(raw_content["type"], "web_search_tool_result");
    assert_eq!(raw_content["content"]["error_code"], "max_uses_exceeded");
    assert_eq!(
        raw_content["content"]["type"],
        "web_search_tool_result_error"
    );

    let round_trip: Message = message::Message::Assistant {
        id: converted.message_id,
        content: converted.choice,
    }
    .try_into()
    .unwrap();

    assert!(matches!(
        round_trip.content.first(),
        Some(Content::WebSearchToolResult {
            tool_use_id,
            content
        }) if tool_use_id == "srvtoolu_01"
            && content["error_code"] == "max_uses_exceeded"
    ));
}

#[test]
fn code_execution_tool_result_variants_deserialize() {
    let normal: Content = serde_json::from_value(json!({
        "type": "code_execution_tool_result",
        "tool_use_id": "srvtoolu_normal",
        "content": {
            "type": "code_execution_result",
            "return_code": 0,
            "stdout": "42\n",
            "stderr": "",
            "content": []
        }
    }))
    .unwrap();
    assert!(matches!(
        normal,
        Content::CodeExecutionToolResult {
            ref tool_use_id,
            ref content
        } if tool_use_id == "srvtoolu_normal"
            && content["type"] == "code_execution_result"
            && content["stdout"] == "42\n"
    ));

    let encrypted: Content = serde_json::from_value(json!({
        "type": "code_execution_tool_result",
        "tool_use_id": "srvtoolu_encrypted",
        "content": {
            "type": "encrypted_code_execution_result",
            "return_code": 1,
            "stderr": "failure",
            "encrypted_stdout": "encrypted-output",
            "content": []
        }
    }))
    .unwrap();
    assert!(matches!(
        encrypted,
        Content::CodeExecutionToolResult {
            ref tool_use_id,
            ref content
        } if tool_use_id == "srvtoolu_encrypted"
            && content["type"] == "encrypted_code_execution_result"
            && content["encrypted_stdout"] == "encrypted-output"
    ));
}

#[test]
fn code_execution_tool_result_is_preserved_and_round_trips() {
    let raw_block = json!({
        "type": "code_execution_tool_result",
        "tool_use_id": "srvtoolu_01",
        "content": {
            "type": "code_execution_result",
            "return_code": 0,
            "stdout": "42\n",
            "stderr": "",
            "content": []
        }
    });
    let value = json!({
        "id": "msg_code_execution",
        "model": CLAUDE_OPUS_4_8,
        "role": "assistant",
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20
        },
        "content": [raw_block]
    });

    let response: CompletionResponse = serde_json::from_value(value).unwrap();
    let converted = response.normalize("anthropic").unwrap();
    let Some(message::AssistantContent::Text(code_execution_result)) = converted.choice.first()
    else {
        panic!("expected raw code_execution_tool_result metadata");
    };
    assert_eq!(
        code_execution_result.additional_params.as_ref().unwrap()[ANTHROPIC_RAW_CONTENT_KEY],
        raw_block
    );

    let round_trip: Message = message::Message::Assistant {
        id: converted.message_id,
        content: converted.choice,
    }
    .try_into()
    .unwrap();
    assert!(matches!(
        round_trip.content.first(),
        Some(Content::CodeExecutionToolResult {
            tool_use_id,
            content
        }) if tool_use_id == "srvtoolu_01"
            && content["type"] == "code_execution_result"
            && content["stdout"] == "42\n"
    ));
}

#[test]
fn text_deserializes_unknown_citation_without_failing() {
    let value = json!({
        "type": "text",
        "text": "future citation",
        "citations": [{
            "type": "future_location",
            "cited_text": "future text",
            "new_field": "kept"
        }]
    });

    let parsed: Content = serde_json::from_value(value).unwrap();
    let Content::Text { citations, .. } = parsed else {
        panic!("expected Content::Text");
    };

    assert!(matches!(
        &citations[0],
        Citation::Unknown(raw)
            if raw["type"] == "future_location" && raw["new_field"] == "kept"
    ));
}

#[test]
fn page_location_citation_roundtrips() {
    let citation = Citation::PageLocation(PageLocationCitation {
        cited_text: "Water is essential for life.".into(),
        document_index: 1,
        document_title: Some("PDF Doc".into()),
        start_page_number: 5,
        end_page_number: 6,
    });
    let value = serde_json::to_value(&citation).unwrap();
    assert_eq!(value["type"], "page_location");
    assert_eq!(value["start_page_number"], 5);
    let back: Citation = serde_json::from_value(value).unwrap();
    assert_eq!(back, citation);
}

#[test]
fn content_block_location_citation_roundtrips() {
    let citation = Citation::ContentBlockLocation(ContentBlockLocationCitation {
        cited_text: "These are important findings.".into(),
        document_index: 2,
        document_title: None,
        start_block_index: 0,
        end_block_index: 1,
    });
    let value = serde_json::to_value(&citation).unwrap();
    assert_eq!(value["type"], "content_block_location");
    assert!(value.get("document_title").is_none());
    let back: Citation = serde_json::from_value(value).unwrap();
    assert_eq!(back, citation);
}

#[test]
fn anthropic_citations_extracts_from_additional_params() {
    let text = message::Text {
        text: "the grass is green".into(),
        additional_params: crate::message::AdditionalParams::try_from_value(json!({
            "citations": [{
                "type": "char_location",
                "cited_text": "The grass is green.",
                "document_index": 0,
                "start_char_index": 0,
                "end_char_index": 20
            }]
        }))
        .expect("object params"),
    };
    let citations = anthropic_citations(&text).unwrap();
    assert_eq!(citations.len(), 1);
}

#[test]
fn anthropic_citations_returns_empty_when_absent() {
    let text = message::Text::new("hello".to_string());
    assert!(anthropic_citations(&text).unwrap().is_empty());
}

#[test]
fn content_text_with_citations_survives_assistant_conversion() {
    let content = Content::Text {
        text: "the grass is green".into(),
        citations: vec![Citation::CharLocation(CharLocationCitation {
            cited_text: "The grass is green.".into(),
            document_index: 0,
            document_title: None,
            start_char_index: 0,
            end_char_index: 20,
        })],
        cache_control: None,
    };
    let assistant: message::AssistantContent = content.try_into().unwrap();
    let message::AssistantContent::Text(text) = assistant else {
        panic!("expected text variant");
    };
    let recovered = anthropic_citations(&text).unwrap();
    assert_eq!(recovered.len(), 1);
}

#[test]
fn provider_text_response_concatenates_text_blocks_without_inserted_newlines() {
    let response = CompletionResponse {
        content: vec![
            Content::Text {
                text: "According to the document, ".into(),
                citations: Vec::new(),
                cache_control: None,
            },
            Content::Text {
                text: "the grass is green".into(),
                citations: Vec::new(),
                cache_control: None,
            },
            Content::Text {
                text: " and the sky is blue.".into(),
                citations: Vec::new(),
                cache_control: None,
            },
        ],
        id: "msg_1".into(),
        model: "claude-test".into(),
        role: "assistant".into(),
        stop_reason: Some("end_turn".into()),
        stop_sequence: None,
        provider_request_id: None,
        usage: Usage {
            input_tokens: 1,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
            cache_creation: None,
            output_tokens: 1,
            output_tokens_details: None,
        },
    };

    assert_eq!(
        response.text_response().as_deref(),
        Some("According to the document, the grass is green and the sky is blue.")
    );
}

#[test]
fn assistant_text_citations_survive_anthropic_request_conversion() {
    let assistant = message::Message::Assistant {
        id: None,
        content: vec![message::AssistantContent::Text(message::Text {
            text: "the grass is green".into(),
            additional_params: crate::message::AdditionalParams::try_from_value(json!({
                "citations": [{
                    "type": "char_location",
                    "cited_text": "The grass is green.",
                    "document_index": 0,
                    "start_char_index": 0,
                    "end_char_index": 20
                }]
            }))
            .expect("object params"),
        })],
    };

    let converted: Message = assistant.try_into().unwrap();
    let Some(Content::Text {
        citations, text, ..
    }) = converted.content.first()
    else {
        panic!("expected assistant text content");
    };

    assert_eq!(text, "the grass is green");
    assert_eq!(
        citations,
        &vec![Citation::CharLocation(CharLocationCitation {
            cited_text: "The grass is green.".into(),
            document_index: 0,
            document_title: None,
            start_char_index: 0,
            end_char_index: 20,
        })]
    );
}

#[test]
fn assistant_text_invalid_known_citations_are_rejected_for_anthropic_request_conversion() {
    let text = message::AssistantContent::Text(message::Text {
        text: "bad citation".into(),
        additional_params: crate::message::AdditionalParams::try_from_value(json!({
            "citations": [{
                "type": "char_location",
                "cited_text": "bad"
            }]
        }))
        .expect("object params"),
    });

    let result = anthropic_content_from_assistant_content(text);

    assert!(
        result.is_err(),
        "invalid Anthropic citation metadata should not be silently dropped"
    );
}

#[test]
fn document_additional_params_forward_to_anthropic_document() {
    let doc = message::UserContent::Document(message::Document {
        data: message::DocumentSourceKind::String("Hello world.".into()),
        media_type: Some(message::DocumentMediaType::TXT),
        additional_params: crate::message::AdditionalParams::try_from_value(json!({
            "title": "Doc1",
            "context": "ctx",
            "citations": { "enabled": true }
        }))
        .expect("object params"),
    });
    let msg = message::Message::User { content: vec![doc] };
    let converted: Message = msg.try_into().unwrap();
    let block = converted.content.first();
    let Some(Content::Document {
        title,
        context,
        citations,
        ..
    }) = block
    else {
        panic!("expected Content::Document");
    };
    assert_eq!(title.as_deref(), Some("Doc1"));
    assert_eq!(context.as_deref(), Some("ctx"));
    assert_eq!(citations, &Some(CitationsConfig { enabled: true }));
}

fn assert_reverse_document_metadata(
    source: DocumentSource,
    expected_data: DocumentSourceKind,
    expected_media_type: Option<message::DocumentMediaType>,
) -> message::Message {
    let provider_message = Message {
        role: Role::User,
        content: vec![Content::Document {
            source,
            title: Some("Doc1".into()),
            context: Some("ctx".into()),
            citations: Some(CitationsConfig { enabled: true }),
            cache_control: None,
        }],
    };

    let generic: message::Message = provider_message.try_into().unwrap();
    let message::Message::User { content } = &generic else {
        panic!("expected generic user message");
    };
    let Some(message::UserContent::Document(document)) = content.first() else {
        panic!("expected generic document");
    };

    assert_eq!(document.data, expected_data);
    assert_eq!(document.media_type, expected_media_type);
    let additional_params = document
        .additional_params
        .as_ref()
        .expect("expected Anthropic document metadata");
    assert_eq!(additional_params["title"], "Doc1");
    assert_eq!(additional_params["context"], "ctx");
    assert_eq!(additional_params["citations"]["enabled"], true);

    generic
}

#[test]
fn anthropic_document_metadata_survives_reverse_conversion_for_all_sources() {
    assert_reverse_document_metadata(
        DocumentSource::Text {
            data: "Hello world.".into(),
            media_type: PlainTextMediaType::Plain,
        },
        DocumentSourceKind::String("Hello world.".into()),
        Some(message::DocumentMediaType::TXT),
    );
    assert_reverse_document_metadata(
        DocumentSource::Base64 {
            data: "base64-pdf".into(),
            media_type: DocumentFormat::PDF,
        },
        DocumentSourceKind::String("base64-pdf".into()),
        Some(message::DocumentMediaType::PDF),
    );
    assert_reverse_document_metadata(
        DocumentSource::Url {
            url: "https://example.com/doc.pdf".into(),
        },
        DocumentSourceKind::Url("https://example.com/doc.pdf".into()),
        None,
    );
    assert_reverse_document_metadata(
        DocumentSource::File {
            file_id: "file_abc".into(),
        },
        DocumentSourceKind::FileId("file_abc".into()),
        None,
    );
}

#[test]
fn anthropic_document_metadata_survives_reverse_round_trip() {
    let provider_message = Message {
        role: Role::User,
        content: vec![Content::Document {
            source: DocumentSource::Text {
                data: "Hello world.".into(),
                media_type: PlainTextMediaType::Plain,
            },
            title: Some("Doc1".into()),
            context: Some("ctx".into()),
            citations: Some(CitationsConfig { enabled: true }),
            cache_control: None,
        }],
    };

    let generic: message::Message = provider_message.try_into().unwrap();
    let message::Message::User { content } = &generic else {
        panic!("expected generic user message");
    };
    let Some(message::UserContent::Document(document)) = content.first() else {
        panic!("expected generic document");
    };
    let additional_params = document
        .additional_params
        .as_ref()
        .expect("expected Anthropic document metadata");
    assert_eq!(additional_params["title"], "Doc1");
    assert_eq!(additional_params["context"], "ctx");
    assert_eq!(additional_params["citations"]["enabled"], true);

    let round_trip: Message = generic.try_into().unwrap();
    let Some(Content::Document {
        title,
        context,
        citations,
        ..
    }) = round_trip.content.first()
    else {
        panic!("expected Anthropic document");
    };
    assert_eq!(title.as_deref(), Some("Doc1"));
    assert_eq!(context.as_deref(), Some("ctx"));
    assert_eq!(citations, &Some(CitationsConfig { enabled: true }));
}

#[test]
fn anthropic_document_empty_metadata_stays_none_on_reverse_conversion() {
    let provider_message = Message {
        role: Role::User,
        content: vec![Content::Document {
            source: DocumentSource::Text {
                data: "Hello world.".into(),
                media_type: PlainTextMediaType::Plain,
            },
            title: None,
            context: None,
            citations: None,
            cache_control: None,
        }],
    };

    let generic: message::Message = provider_message.try_into().unwrap();
    let message::Message::User { content } = &generic else {
        panic!("expected generic user message");
    };
    let Some(message::UserContent::Document(document)) = content.first() else {
        panic!("expected generic document");
    };

    assert_eq!(document.additional_params, None);
}

#[tokio::test]
async fn completion_http_non_success_preserves_status_and_body() {
    use crate::client::CompletionClient;
    use crate::completion::CompletionModel as _;
    use crate::providers::anthropic::Client;
    use crate::test_utils::RecordingHttpClient;

    let body = r#"{"type":"error","error":{"type":"overloaded_error","message":"slow down"}}"#;
    let http_client =
        RecordingHttpClient::with_error_response(http::StatusCode::TOO_MANY_REQUESTS, body);
    let client = Client::builder()
        .api_key("test-key")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.completion_model(CLAUDE_SONNET_4_6);
    let request = model.completion_request("hello").build();

    let error = model
        .completion(request)
        .await
        .expect_err("completion should fail with non-success status");

    // rig#2314: a provider with a request-id contract preserves its
    // non-success responses as ProviderResponse, so the transport id has
    // a home on the error; this mock sent no header, so the id is None.
    assert!(matches!(error, CompletionError::ProviderResponse(_)));
    assert_eq!(error.provider_request_id(), None);
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::TOO_MANY_REQUESTS)
    );
    assert_eq!(error.provider_response_body(), Some(body));
}

#[tokio::test]
async fn completion_2xx_error_envelope_preserves_status_and_body() {
    use crate::client::CompletionClient;
    use crate::completion::CompletionModel as _;
    use crate::providers::anthropic::Client;
    use crate::test_utils::RecordingHttpClient;

    // Anthropic's `ApiResponse` is internally tagged on `type`; the `Error`
    // arm flattens `ApiErrorResponse { message }`, so a 200-OK error envelope
    // deserializes from `{"type":"error","message":"..."}` and routes through
    // `from_http_response(OK, ..)` into `ProviderResponse`.
    let body = r#"{"type":"error","message":"model overloaded"}"#;
    let http_client = RecordingHttpClient::new(body); // 200 OK
    let client = Client::builder()
        .api_key("test-key")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.completion_model(CLAUDE_SONNET_4_6);
    let request = model.completion_request("hello").build();

    let error = model
        .completion(request)
        .await
        .expect_err("completion should fail with provider error envelope");

    match &error {
        CompletionError::ProviderResponse(stored) => {
            assert_eq!(stored.body, body);
            assert_eq!(stored.status, Some(http::StatusCode::OK));
            assert_eq!(error.provider_response_body(), Some(body));
            assert_eq!(error.provider_response_status(), Some(http::StatusCode::OK));
        }
        other => panic!("expected ProviderResponse, got {other:?}"),
    }
}

#[tokio::test]
async fn completion_streaming_http_non_success_preserves_status_and_body() {
    use crate::client::CompletionClient;
    use crate::completion::CompletionModel as _;
    use crate::providers::anthropic::Client;
    use crate::test_utils::HttpErrorStreamingClient;
    use futures::StreamExt;

    let body = r#"{"type":"error","error":{"type":"overloaded_error","message":"slow down"}}"#;
    let http_client = HttpErrorStreamingClient::new(http::StatusCode::SERVICE_UNAVAILABLE, body);
    let client = Client::builder()
        .api_key("test-key")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.completion_model(CLAUDE_SONNET_4_6);
    let request = model.completion_request("hello").build();

    let mut stream = model.stream(request).await.expect("stream should start");

    // The transport failure surfaces as the first error item yielded by the stream.
    let error = loop {
        match stream.next().await {
            Some(Ok(_)) => continue,
            Some(Err(error)) => break error,
            None => panic!("stream ended without yielding the transport error"),
        }
    };

    // Streaming *connect* failures stay transport-shaped (HttpError):
    // rig#2314's ProviderResponse classification covers the unary driver
    // and in-band stream envelopes, not the SSE handshake.
    assert!(matches!(error, CompletionError::HttpError(_)));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::SERVICE_UNAVAILABLE)
    );
    assert_eq!(error.provider_response_body(), Some(body));

    // The transport failure ends the stream: nothing may follow it that
    // would read as a successfully completed turn.
    assert!(stream.next().await.is_none());
    assert!(
        stream.response.is_none(),
        "a stream cut short by a transport error must not synthesize a terminal record"
    );
}

#[test]
fn coerce_tool_input_normalizes_non_object_arguments() {
    use serde_json::json;

    // Object passes through untouched.
    assert_eq!(
        coerce_tool_input(json!({"q": "rust", "n": 3})),
        json!({"q": "rust", "n": 3})
    );

    // A JSON string that encodes an object is parsed into that object.
    assert_eq!(
        coerce_tool_input(json!("{\"q\":\"rust\"}")),
        json!({"q": "rust"})
    );

    // A non-JSON string, a JSON string that is not an object, null, arrays,
    // numbers and bools all collapse to an empty object: the only shape the
    // Anthropic API accepts for tool_use.input.
    assert_eq!(coerce_tool_input(json!("not json")), json!({}));
    assert_eq!(coerce_tool_input(json!("[1,2,3]")), json!({}));
    assert_eq!(coerce_tool_input(json!(null)), json!({}));
    assert_eq!(coerce_tool_input(json!([1, 2, 3])), json!({}));
    assert_eq!(coerce_tool_input(json!(42)), json!({}));
    assert_eq!(coerce_tool_input(json!(true)), json!({}));
}

// Regression test for issue #1429: PR #1431 added the `DocumentSource::Url`
// wire variant and response-side parsing, but the request-side
// `UserContent::Document` conversion still rejected URL-backed PDFs even
// though the Anthropic Messages API supports
// `"source": {"type": "url", ...}` for PDFs.
// The media type is optional because Anthropic's URL source is implicitly a
// PDF and does not include a media-type field on the wire.
//
// See <https://docs.anthropic.com/en/docs/build-with-claude/pdf-support>
// for URL-sourced PDF documents.
#[test]
fn url_pdf_with_or_without_media_type_converts_to_url_document_source() {
    let pdf_url = "https://example.com/resume.pdf";

    for media_type in [Some(message::DocumentMediaType::PDF), None] {
        let msg = message::Message::User {
            content: vec![message::UserContent::document_url(pdf_url, media_type)],
        };

        let converted = Message::try_from(msg).expect("URL PDF should convert");
        let json = serde_json::to_value(&converted).expect("message should serialize");

        assert_eq!(
            json.pointer("/content/0/source"),
            Some(&json!({ "type": "url", "url": pdf_url })),
            "URL PDF should map to a url document source: {json:#}"
        );
    }
}

/// Raw-capture tests: the `normalize` shape through the Anthropic model,
/// driven end to end over a mock transport that hands back a Messages body
/// *and* a `request-id` response header. Anthropic's raw type carries the
/// transport id itself (`CompletionResponse::provider_request_id`, stamped
/// by the driver), which is why the Part A contract here is a plain
/// `raw_completion` → `normalize`, with no id to reattach.
/// `with_error_response_headers` with `200 OK` is the one unary double
/// that carries response headers.
mod raw_capture {
    use super::*;
    use crate::client::CompletionClient;
    use crate::completion::CompletionModel as _;
    use crate::providers::anthropic::Client;
    use crate::test_utils::RecordingHttpClient;

    const REQUEST_ID: &str = "req_unit_anthropic_0001";

    /// A Messages body whose `stop_sequence` is set: the normalized
    /// response maps it to `FinishReason::Stop` and drops which sequence
    /// fired, so the capture provably answers more than `completion()`.
    const BODY: &str = r#"{
            "id": "msg_raw_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "stop_sequence",
            "stop_sequence": "alpha",
            "usage": {"input_tokens": 7, "output_tokens": 2}
        }"#;

    fn model() -> CompletionModel<RecordingHttpClient> {
        let mut headers = http::HeaderMap::new();
        headers.insert("request-id", http::HeaderValue::from_static(REQUEST_ID));
        let http_client =
            RecordingHttpClient::with_error_response_headers(http::StatusCode::OK, BODY, headers);
        let client = Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        client.completion_model(CLAUDE_SONNET_4_6)
    }

    /// The load-bearing capture property: `raw` is Anthropic's
    /// `CompletionResponse` as rig parsed it — it deserializes back into
    /// that type and re-serializes to the identical value, including the
    /// transport id the driver stamped onto the raw type — and
    /// re-normalizing that capture reproduces every normalized field, so
    /// `raw` and the typed route tell one story. Also reads
    /// `stop_sequence` off the capture, which the normalized response does
    /// not carry.
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
        assert_eq!(typed.stop_sequence.as_deref(), Some("alpha"));
        assert_eq!(typed.provider_request_id.as_deref(), Some(REQUEST_ID));
        assert_eq!(raw["stop_sequence"], "alpha");

        let renormalized = typed
            .normalize(<crate::providers::anthropic::client::Anthropic as AnthropicCompatibleProvider>::PROVIDER_NAME)
            .expect("re-normalize the capture");
        assert_eq!(response.identity(), renormalized.identity());
        assert_eq!(response.finish_reason(), renormalized.finish_reason());
        assert_eq!(response.model, renormalized.model);
        assert_eq!(response.usage, renormalized.usage);
        assert_eq!(response.choice, renormalized.choice);
        assert_eq!(
            response.finish_reason(),
            Some(completion::FinishReason::Stop)
        );
        assert_eq!(response.provider_request_id.as_deref(), Some(REQUEST_ID));
    }

    /// Part A contract statement for a provider whose raw type carries the
    /// transport id: `raw_completion` → `normalize` reproduces
    /// `completion()` on identity, finish reason, model and usage — the id
    /// included — with nothing to reattach.
    #[tokio::test]
    async fn raw_completion_then_normalize_reproduces_completion() {
        let model = model();

        let raw = model
            .raw_completion(model.completion_request("hello").build())
            .await
            .expect("typed route");
        assert_eq!(raw.provider_request_id.as_deref(), Some(REQUEST_ID));
        let reassembled = raw
            .normalize(<crate::providers::anthropic::client::Anthropic as AnthropicCompatibleProvider>::PROVIDER_NAME)
            .expect("normalize");

        let normalized = model
            .completion(model.completion_request("hello").build())
            .await
            .expect("normalized route");

        assert_eq!(reassembled.identity(), normalized.identity());
        assert_eq!(reassembled.finish_reason(), normalized.finish_reason());
        assert_eq!(reassembled.model, normalized.model);
        assert_eq!(reassembled.usage, normalized.usage);
        assert_eq!(reassembled.provider_request_id.as_deref(), Some(REQUEST_ID));
        assert_eq!(normalized.provider_request_id.as_deref(), Some(REQUEST_ID));
    }
}
