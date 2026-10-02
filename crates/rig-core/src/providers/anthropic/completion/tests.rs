use super::*;
use crate::error::ProviderError;
use crate::message::EMPTY_RESPONSE_ERROR;
use crate::providers::anthropic::wire::AnthropicConfig;
use crate::test_utils::json_body;
use crate::wire::WireFrame;
use serde_json::json;
use serde_path_to_error::deserialize;

/// The one-turn request every reply below is folded against.
fn hello_request() -> CompletionRequest {
    completion_request_with_history(vec![message::Message::user("hello")], None)
}

/// Fold a Messages reply body through the wire's own decoder — the one
/// mapping there is, now that the duplicate `normalize` is gone.
///
/// `POST /v1/messages` answers with a whole `message` object, which the
/// decoder treats as a frame like any other, so a cell that used to call
/// `normalize` on a typed response states the same thing by folding the
/// body the provider would have sent. `type` is injected when the cell
/// built the typed response directly, since that type does not model the
/// tag. Whatever the decoder or the fold rejects surfaces here as the
/// error the driver would report.
fn fold_reply(body: &serde_json::Value) -> Result<completion::CompletionResponse, ProviderError> {
    let mut body = body.clone();
    if let Some(map) = body.as_object_mut() {
        map.entry("type").or_insert_with(|| json!("message"));
    }
    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);
    crate::test_utils::decode_reply(
        &wire,
        &hello_request(),
        crate::wire::Mode::Unary,
        [WireFrame::Text(body.to_string())],
        body,
    )
}

#[test]
fn current_model_default_max_tokens_match_anthropic_limits() {
    assert_eq!(
        default_max_tokens_for_model(CLAUDE_FABLE_5_1),
        Some(128_000)
    );
    assert_eq!(default_max_tokens_for_model(CLAUDE_FABLE_5), Some(128_000));
    assert_eq!(default_max_tokens_for_model(CLAUDE_OPUS_5), Some(128_000));
    assert_eq!(default_max_tokens_for_model(CLAUDE_SONNET_5), Some(128_000));
    assert_eq!(default_max_tokens_for_model(CLAUDE_OPUS_4_8), Some(128_000));
    assert_eq!(default_max_tokens_for_model(CLAUDE_OPUS_4_7), Some(128_000));
    assert_eq!(default_max_tokens_for_model(CLAUDE_OPUS_4_6), Some(128_000));
    assert_eq!(
        default_max_tokens_for_model(CLAUDE_SONNET_4_6),
        Some(128_000)
    );
    assert_eq!(default_max_tokens_for_model(CLAUDE_HAIKU_4_5), Some(64_000));
}

#[test]
fn unknown_model_has_no_documented_default_max_tokens() {
    assert_eq!(default_max_tokens_for_model("claude-unknown"), None);
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
                cache_control: None,
            }],
        },
        Message {
            role: Role::Assistant,
            content: vec![Content::Text {
                text: "Response".to_string(),
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
        name: crate::message::ToolName::new(name).expect("tool name"),
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
    CompletionRequest::from(vec![
        message::Message::system("System prompt"),
        message::Message::from("Hello"),
    ])
    .max_tokens(64)
    .tools(tools)
    .additional_params(additional_params)
}

fn completion_request_with_history(
    chat_history: Vec<message::Message>,
    preamble: Option<String>,
) -> CompletionRequest {
    CompletionRequest::from(
        preamble
            .map(message::Message::system)
            .into_iter()
            .chain(chat_history)
            .collect::<Vec<_>>(),
    )
    .max_tokens(64)
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

/// Anthropic-compatible gateways do not necessarily implement Anthropic's
/// constrained tool schemas, so asking for strict tools on one leaves the
/// tool unchanged. The policy is the dialect's data, not a trait hook, so
/// this is asserted through the wire a gateway builds.
#[test]
fn strict_tool_hook_is_a_noop_for_anthropic_compatible_gateways() {
    use crate::wire::{Mode, Wire};

    let request = completion_request_with_tools(vec![generic_tool("lookup")], None);
    let encoded = crate::providers::anthropic::wire::AnthropicConfig::with_key(
        &crate::providers::anthropic::wire::ZAI,
        "k",
    )
    .completion("some-model")
    .with_strict_tools()
    .encode(request, Mode::Unary)
    .expect("the request encodes");
    let value = json_body(&encoded.request);

    assert!(value["tools"][0].get("strict").is_none());
    assert!(
        value["tools"][0]["input_schema"]
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
    let request = AnthropicCompletionRequest::try_from_params(
        AnthropicRequestParams {
            model: CLAUDE_SONNET_4_6,
            request,
            prompt_caching: false,
            automatic_caching: false,
            automatic_caching_ttl: None,
            static_prefix_cache_ttl: None,
        },
        Some(
            crate::providers::anthropic::wire::strict_tool_transform
                as fn(&mut crate::providers::anthropic::completion::ToolDefinition),
        ),
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
    assert_eq!(value["system"].as_array().map(Vec::len), Some(1));

    // The misplaced mid-conversation instruction moves after the next user
    // turn instead of into `system`.
    let messages = value["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 4);
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[1]["role"], "assistant");
    assert_eq!(messages[2]["role"], "user");
    assert_eq!(messages[3]["role"], "system");
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
    assert_eq!(
        messages
            .iter()
            .filter(|message| message["role"].as_str() == Some("system"))
            .count(),
        1
    );
}

#[test]
fn opus_4_8_preserves_system_message_after_assistant_server_tool_result() {
    let request = completion_request_with_history(
        vec![
            message::Message::Assistant(message::AssistantMessage::new(vec![
                message::AssistantContent::Opaque(message::Opaque {
                    item: json!({
                        "type": "server_tool_use",
                        "id": "srvtoolu_01",
                        "name": "web_search",
                        "input": {
                            "query": "clear daytime sky color"
                        }
                    }),
                    replay: true,
                }),
                message::AssistantContent::Opaque(message::Opaque {
                    item: json!({
                        "type": "web_search_tool_result",
                        "tool_use_id": "srvtoolu_01",
                        "content": {
                            "type": "web_search_tool_result_error",
                            "error_code": "unavailable"
                        }
                    }),
                    replay: true,
                }),
            ])),
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

/// `message` alone on the wire.
fn convert(message: message::Message) -> Result<Option<Message>, MessageError> {
    let ids = WireIds::new(std::slice::from_ref(&message));
    Message::from_message(message, &ids, 0)
}

/// Anthropic rejects a blank text block, so one never reaches the request,
/// even when it holds a provider item; sibling content converts unaffected.
#[test]
fn blank_text_produces_no_anthropic_block() {
    let blank =
        message::AssistantContent::text("  ").with_native(json!({"type": "text", "text": "  "}));
    let message = message::Message::Assistant(message::AssistantMessage::new(vec![
        blank,
        message::AssistantContent::text("real answer"),
    ]));
    let converted = convert(message)
        .expect("message converts")
        .expect("a block survives");
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
            message::Message::Assistant(message::AssistantMessage::new(vec![
                message::AssistantContent::Opaque(message::Opaque {
                    item: json!({
                        "type": "server_tool_use",
                        "id": "srvtoolu_01",
                        "name": "web_search",
                        "input": {
                            "query": "clear daytime sky color"
                        }
                    }),
                    replay: true,
                }),
            ])),
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

/// Encode a history for a model that keeps mid-conversation system messages
/// and return the wire's `system` and `messages` as JSON.
fn encode_mid_conversation_history(
    model: &str,
    history: Vec<message::Message>,
) -> serde_json::Value {
    let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
        model,
        request: completion_request_with_history(history, None),
        prompt_caching: false,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();
    serde_json::to_value(request).unwrap()
}

/// A misplaced system message moves after the next user turn instead of into
/// `system`, so the prompt prefix a cache or a bound thinking block depends on
/// stays the same. Offline because the position rule's every branch is shown
/// here; the recorded proof is the model sessions' mid-conversation phase
/// (`anthropic/models/<model>/session.yaml`), whose system message sits after
/// an assistant turn and before the user's question.
#[test]
fn opus_4_8_moves_a_misplaced_system_message_after_the_next_user_turn() {
    let value = encode_mid_conversation_history(
        CLAUDE_OPUS_4_8,
        vec![
            message::Message::user("Review this code."),
            message::Message::System {
                content: "From now on, require explicit type annotations.".to_string(),
            },
            message::Message::user("Now review this other file."),
        ],
    );

    assert!(value.get("system").is_none(), "{value}");
    let messages = value["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[1]["role"], "user");
    assert_eq!(messages[2]["role"], "system");
}

/// The agent-loop placement: a system message between an assistant's
/// `tool_use` and its `tool_result` moves after the tool results, where
/// Anthropic documents it, and a trailing one after an assistant turn (no user
/// turn follows) is still hoisted. Offline for the reason above.
#[test]
fn sonnet_5_5_defers_a_system_message_past_tool_results() {
    let lookup = || message::ToolName::new("lookup").expect("tool name");
    let tool_call = message::Message::Assistant(message::AssistantMessage::new(vec![
        message::AssistantContent::tool_call("toolu_1", lookup(), json!({})),
    ]));
    let tool_result =
        message::Message::tool_result(message::CallId::from_wire("toolu_1"), lookup(), "ok");
    let value = encode_mid_conversation_history(
        CLAUDE_SONNET_5_5,
        vec![
            message::Message::user("Look it up."),
            tool_call,
            message::Message::system("Answer in Spanish."),
            tool_result,
        ],
    );
    assert!(value.get("system").is_none(), "{value}");
    let roles: Vec<_> = value["messages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|message| message["role"].clone())
        .collect();
    assert_eq!(roles, ["user", "assistant", "user", "system"]);

    let value = encode_mid_conversation_history(
        CLAUDE_SONNET_5_5,
        vec![
            message::Message::user("Look it up."),
            message::Message::assistant("Done."),
            message::Message::system("Answer in Spanish."),
        ],
    );
    assert_eq!(value["system"][0]["text"], "Answer in Spanish.");
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

    let anthropic_message: Message = convert(rig_message).unwrap().unwrap();
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
fn test_plaintext_rig_to_anthropic_conversion() {
    use crate::completion::message as msg;

    let rig_message = msg::Message::User {
        content: vec![msg::UserContent::document_text(
            "Some plain text content".to_string(),
            Some(msg::DocumentMediaType::TXT),
        )],
    };

    let anthropic_message: Message = convert(rig_message).unwrap().unwrap();
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
fn test_unsupported_document_type_returns_error() {
    use crate::completion::message as msg;

    let rig_message = msg::Message::User {
        content: vec![msg::UserContent::Document(msg::Document {
            data: DocumentSourceKind::String("data".into()),
            media_type: Some(msg::DocumentMediaType::HTML),
            additional_params: None,
        })],
    };

    let result = convert(rig_message);
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

    let result = convert(rig_message);
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

/// An assistant turn of `blocks` on the wire.
fn assistant_wire(blocks: Vec<message::AssistantContent>) -> Vec<Content> {
    convert(message::Message::Assistant(message::AssistantMessage::new(
        blocks,
    )))
    .expect("the turn converts")
    .map(|message| message.content)
    .unwrap_or_default()
}

/// A block whose provider item is current replays the item verbatim:
/// signed thinking, redacted thinking, a call with its `caller`, cited text.
#[test]
fn current_provider_items_replay_verbatim() {
    let thinking = json!({"type": "thinking", "thinking": "step one", "signature": "sig-1"});
    let redacted = json!({"type": "redacted_thinking", "data": "redacted block"});
    let call = json!({"type": "tool_use", "id": "toolu_1", "name": "add",
        "input": {"x": 1}, "caller": {"type": "direct"}});
    let text = json!({"type": "text", "text": "Two.", "citations": [{"type": "char_location",
        "cited_text": "2", "document_index": 0, "start_char_index": 0, "end_char_index": 1}]});
    let add = message::ToolName::new("add").expect("tool name");
    let wire = assistant_wire(vec![
        message::AssistantContent::reasoning("step one").with_native(thinking.clone()),
        message::AssistantContent::Reasoning(message::Reasoning {
            redacted: true,
            ..message::Reasoning::default()
        })
        .with_native(redacted.clone()),
        message::AssistantContent::tool_call("toolu_1", add, json!({"x": 1}))
            .with_native(call.clone()),
        message::AssistantContent::text("Two.").with_native(text.clone()),
    ]);
    assert_eq!(
        wire,
        [thinking, redacted, call, text]
            .map(Content::Native)
            .to_vec()
    );
}

/// A current provider item is sent before any per-kind rule: thinking a
/// dialect (Kimi) sent unsigned goes back to it as the thinking it was,
/// not as text (#1315).
#[test]
fn a_current_unsigned_thinking_item_replays_verbatim() {
    let item = json!({"type": "thinking", "thinking": "thought", "signature": ""});
    let unsigned = message::AssistantContent::reasoning("thought").with_native(item.clone());
    assert_eq!(assistant_wire(vec![unsigned]), [Content::Native(item)]);
}

/// pi's rules for blocks with no current item: thinking goes as text,
/// redacted thinking (whose payload lived in its item) is dropped, and an
/// edited block is rebuilt from its fields.
#[test]
fn blocks_without_a_current_item_rebuild_as_pi_does() {
    let unsigned = message::AssistantContent::reasoning("thought");
    let redacted = message::AssistantContent::Reasoning(message::Reasoning {
        redacted: true,
        ..message::Reasoning::default()
    });
    let mut edited = message::AssistantContent::text("before")
        .with_native(json!({"type": "text", "text": "before", "citations": []}));
    if let message::AssistantContent::Text(text) = &mut edited {
        text.text = "after".to_owned();
    }
    let wire = assistant_wire(vec![unsigned, redacted, edited]);
    assert_eq!(
        wire,
        [
            Content::from("thought".to_owned()),
            Content::from("after".to_owned())
        ]
    );
}

/// An opaque provider item that survived history adaptation goes back as
/// it came.
#[test]
fn opaque_items_replay_verbatim() {
    let item = json!({"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search",
        "input": {"query": "rust"}});
    let wire = assistant_wire(vec![message::AssistantContent::Opaque(message::Opaque {
        item: item.clone(),
        replay: true,
    })]);
    assert_eq!(wire, [Content::Native(item)]);
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
        stop_details: None,
        container: None,
        usage: Usage {
            input_tokens: 7,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
            cache_creation: None,
            output_tokens: 2,
            output_tokens_details: None,
        },
    };

    let parsed: completion::CompletionResponse =
        fold_reply(&serde_json::to_value(&response).expect("serialize the wire type"))
            .expect("empty end_turn should not error");

    // Anthropic's documented empty `end_turn` is a turn that carried
    // nothing. It used to normalize to one fabricated empty-text part
    // because the content type could not be empty; the empty list is the
    // same turn, said honestly. Everything else about the response is
    // unchanged, which is the point of asserting it here.
    assert!(parsed.choice.is_empty());
    assert_eq!(parsed.provider(), "anthropic");
    assert_eq!(parsed.response_id(), Some("msg_123"));
    assert_eq!(parsed.model(), Some(CLAUDE_SONNET_4_6));
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
        stop_details: None,
        container: None,
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
        let err = fold_reply(
            &serde_json::to_value(empty_response_with(stop_reason, stop_sequence))
                .expect("serialize the wire type"),
        )
        .expect_err(&format!(
            "empty {stop_reason:?} response should remain an error"
        ));

        assert!(matches!(
            err,
            ProviderError::Response(message) if message == EMPTY_RESPONSE_ERROR
        ));
    }
}

#[test]
fn empty_stop_sequence_response_naming_its_sequence_is_a_completed_turn() {
    let parsed = fold_reply(
        &serde_json::to_value(empty_response_with(Some("stop_sequence"), Some("alpha")))
            .expect("serialize the wire type"),
    )
    .expect("a completed stop-sequence turn must not fold into an error");

    assert!(parsed.choice.is_empty());
    assert_eq!(parsed.finish_reason(), Some(completion::FinishReason::Stop));
}

#[test]
fn end_turn_with_a_tool_call_is_reconciled_to_tool_calls() {
    // Anthropic reports `tool_use`, but the reconciliation the response
    // builder applies must hold for any provider that reports a plain stop
    // alongside a tool call.
    let response = CompletionResponse {
        content: vec![
            serde_json::from_value(
                json!({"type": "tool_use", "id": "toolu_1", "name": "add", "input": {"x": 1}}),
            )
            .expect("a tool_use block"),
        ],
        id: "msg_123".to_string(),
        model: CLAUDE_SONNET_4_6.to_string(),
        role: "assistant".to_string(),
        stop_reason: Some("end_turn".to_string()),
        stop_sequence: None,
        stop_details: None,
        container: None,
        usage: Usage {
            input_tokens: 7,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
            cache_creation: None,
            output_tokens: 2,
            output_tokens_details: None,
        },
    };

    let parsed = fold_reply(&serde_json::to_value(&response).expect("serialize the wire type"))
        .expect("tool-use response should fold");

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

/// The citations a response text block carries, through the checked item.
fn citations_of(block: serde_json::Value) -> Vec<Citation> {
    let item: ContentItem = serde_json::from_value(block).expect("the block is well formed");
    Vec::<Citation>::deserialize(&item.0["citations"]).expect("the citations parse")
}

#[test]
fn text_deserializes_char_location_citation() {
    let citations = citations_of(json!({
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
    }));
    assert_eq!(citations.len(), 1);
    let Citation::CharLocation(citation) = &citations[0] else {
        panic!("expected CharLocation");
    };
    assert_eq!(citation.start_char_index, 0);
    assert_eq!(citation.end_char_index, 20);
}

#[test]
fn text_deserializes_search_result_location_citation() {
    let citations = citations_of(json!({
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
    }));
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
fn text_deserializes_web_search_result_location_citation_with_null_title() {
    let citations = citations_of(json!({
        "type": "text",
        "text": "Claude Shannon worked at Bell Labs.",
        "citations": [{
            "type": "web_search_result_location",
            "cited_text": "Claude Shannon was a mathematician.",
            "url": "https://example.com/shannon",
            "title": null,
            "encrypted_index": "encrypted-reference"
        }]
    }));
    let Citation::WebSearchResultLocation(citation) = &citations[0] else {
        panic!("expected WebSearchResultLocation");
    };
    assert_eq!(citation.title, None);
    assert_eq!(citation.encrypted_index, "encrypted-reference");
    let serialized = serde_json::to_value(&citations[0]).unwrap();
    assert!(serialized["title"].is_null());
}

#[test]
fn text_deserializes_unknown_citation_without_failing() {
    let citations = citations_of(json!({
        "type": "text",
        "text": "future citation",
        "citations": [{
            "type": "future_location",
            "cited_text": "future text",
            "new_field": "kept"
        }]
    }));
    assert!(matches!(
        &citations[0],
        Citation::Unknown(raw)
            if raw["type"] == "future_location" && raw["new_field"] == "kept"
    ));
}

/// A text block with a malformed known citation keeps it on its item, and
/// reading the citations reports it: the reply never fails for a field no
/// block is built from.
#[test]
fn a_malformed_known_citation_is_kept_on_its_item() {
    let value = json!({
        "id": "msg_bad", "model": CLAUDE_SONNET_4_6, "role": "assistant",
        "stop_reason": "end_turn", "stop_sequence": null,
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "content": [{"type": "text", "text": "bad",
            "citations": [{"type": "char_location", "cited_text": "bad"}]}]
    });
    let response = fold_reply(&value).expect("the reply folds");
    let [message::AssistantContent::Text(text)] = response.choice.as_slice() else {
        panic!("one text block: {:?}", response.choice);
    };
    assert!(anthropic_citations(text).is_err());
}

/// Server tools and their results, MCP blocks and whatever Anthropic adds
/// next are opaque blocks that replay; none fails the reply, and each goes
/// back to the model verbatim beside the cited answer.
#[test]
fn hosted_tool_reply_decodes_to_opaque_blocks_that_replay_verbatim() {
    let content = vec![
        json!({"type": "server_tool_use", "id": "srvtoolu_01", "name": "web_search",
            "input": {"query": "claude shannon birth date"}, "caller": {"type": "direct"}}),
        json!({"type": "web_search_tool_result", "tool_use_id": "srvtoolu_01", "content": [{
            "type": "web_search_result", "url": "https://example.com/shannon",
            "title": "Claude Shannon", "encrypted_content": "encrypted-content",
            "page_age": "April 30, 2025"}]}),
        json!({"type": "web_search_tool_result", "tool_use_id": "srvtoolu_02", "content": {
            "type": "web_search_tool_result_error", "error_code": "max_uses_exceeded"}}),
        json!({"type": "web_fetch_tool_result", "tool_use_id": "srvtoolu_03", "content": {
            "type": "web_fetch_result", "url": "https://example.com"}}),
        json!({"type": "code_execution_tool_result", "tool_use_id": "srvtoolu_04", "content": {
            "type": "encrypted_code_execution_result", "return_code": 1, "stderr": "failure",
            "encrypted_stdout": "encrypted-output", "content": []}}),
        json!({"type": "bash_code_execution_tool_result", "tool_use_id": "srvtoolu_05",
            "content": {"type": "bash_code_execution_result", "stdout": "ok", "stderr": "",
                "return_code": 0, "content": []}}),
        json!({"type": "text_editor_code_execution_tool_result", "tool_use_id": "srvtoolu_06",
            "content": {"type": "text_editor_code_execution_view_result", "content": "x"}}),
        json!({"type": "tool_search_tool_result", "tool_use_id": "srvtoolu_07",
            "content": {"type": "tool_search_tool_search_result", "tool_references": []}}),
        json!({"type": "mcp_tool_use", "id": "mcptoolu_1", "name": "fetch",
            "server_name": "docs", "input": {}}),
        json!({"type": "mcp_tool_result", "tool_use_id": "mcptoolu_1", "is_error": false,
            "content": [{"type": "text", "text": "fetched"}]}),
        json!({"type": "container_upload", "file_id": "file_1"}),
        json!({"type": "compaction", "content": "Summary so far."}),
        json!({"type": "text", "text": "Claude Shannon was born on April 30, 1916.",
            "citations": [{"type": "web_search_result_location",
                "cited_text": "Claude Shannon was born on April 30, 1916.",
                "url": "https://example.com/shannon", "title": "Claude Shannon",
                "encrypted_index": "encrypted-index"}]}),
    ];
    let value = json!({
        "id": "msg_web_search", "model": CLAUDE_SONNET_4_6, "role": "assistant",
        "stop_reason": "end_turn", "stop_sequence": null,
        "usage": {"input_tokens": 10, "output_tokens": 20},
        "content": content,
    });
    let converted = fold_reply(&value).expect("the hosted-tool reply folds");
    assert_eq!(converted.choice.len(), content.len());
    let (answer, hosted) = converted.choice.split_last().expect("blocks");
    assert!(hosted.iter().all(|block| matches!(
        block,
        message::AssistantContent::Opaque(message::Opaque { replay: true, .. })
    )));
    let message::AssistantContent::Text(answer) = answer else {
        panic!("expected the text answer last, got {answer:?}");
    };
    assert_eq!(answer.text, "Claude Shannon was born on April 30, 1916.");
    assert!(matches!(
        anthropic_citations(answer).unwrap().first(),
        Some(Citation::WebSearchResultLocation(citation))
            if citation.encrypted_index == "encrypted-index"
    ));

    let replayed = assistant_wire(converted.choice);
    assert_eq!(
        replayed,
        content.into_iter().map(Content::Native).collect::<Vec<_>>()
    );
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
fn anthropic_citations_reads_the_current_provider_item() {
    let block = message::AssistantContent::text("the grass is green").with_native(json!({
        "type": "text",
        "text": "the grass is green",
        "citations": [{
            "type": "char_location",
            "cited_text": "The grass is green.",
            "document_index": 0,
            "start_char_index": 0,
            "end_char_index": 20
        }]
    }));
    let message::AssistantContent::Text(text) = &block else {
        panic!("a text block");
    };
    assert_eq!(anthropic_citations(text).unwrap().len(), 1);
}

#[test]
fn anthropic_citations_returns_empty_when_absent() {
    let text = message::Text::new("hello".to_string());
    assert!(anthropic_citations(&text).unwrap().is_empty());
}

#[test]
fn document_additional_params_forward_to_anthropic_document() {
    let doc = message::UserContent::Document(message::Document {
        data: message::DocumentSourceKind::String("Hello world.".into()),
        media_type: Some(message::DocumentMediaType::TXT),
        additional_params: Some(json!({
            "title": "Doc1",
            "context": "ctx",
            "citations": { "enabled": true }
        })),
    });
    let msg = message::Message::User { content: vec![doc] };
    let converted = convert(msg).unwrap().unwrap();
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

#[tokio::test]
async fn completion_http_non_success_preserves_status_and_body() {
    use crate::test_utils::RecordingHttpClient;

    let body = r#"{"type":"error","error":{"type":"overloaded_error","message":"slow down"}}"#;
    let http = RecordingHttpClient::with_error_response(http::StatusCode::TOO_MANY_REQUESTS, body);
    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);

    let error = crate::driver::Model::new(wire.clone(), http.clone())
        .call(hello_request())
        .await
        .expect_err("completion should fail with non-success status");

    // rig#2314: a provider with a request-id contract preserves its
    // non-success responses as ProviderResponse, so the transport id has
    // a home on the error; this mock sent no header, so the id is None.
    assert!(matches!(error, ProviderError::ProviderResponse(_)));
    assert_eq!(error.provider_request_id(), None);
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::TOO_MANY_REQUESTS)
    );
    assert_eq!(error.provider_response_body(), Some(body));
}

#[tokio::test]
async fn completion_2xx_error_envelope_preserves_status_and_body() {
    use crate::test_utils::RecordingHttpClient;

    // Anthropic answers an in-band failure with its top-level error
    // envelope, and it can arrive under a 200: the decoder models `error`
    // as an event of the Messages wire, so the envelope routes through the
    // same `ProviderResponse` funnel as a rejected status rather than
    // reading as a corrupt frame.
    //
    // The body is asserted byte-for-byte, in the key order Anthropic's
    // recorded replies use, and carries the top-level `request_id` the
    // envelope type does not model: a preserved reply that Rig rebuilt from
    // the fields it happens to parse is not the provider's reply. The
    // status is the driver's — a success at the transport layer is still a
    // status the caller must see.
    let body = r#"{"error":{"message":"model overloaded","type":"overloaded_error"},"request_id":"req_011CXYZ","type":"error"}"#;
    let http = RecordingHttpClient::new(body); // 200 OK
    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);

    let error = crate::driver::Model::new(wire.clone(), http.clone())
        .call(hello_request())
        .await
        .expect_err("completion should fail with provider error envelope");

    assert!(matches!(error, ProviderError::ProviderResponse(_)));
    assert_eq!(error.provider_response_body(), Some(body));
    assert_eq!(error.provider_response_status(), Some(http::StatusCode::OK));
}

#[tokio::test]
async fn completion_streaming_http_non_success_preserves_status_and_body() {
    use crate::test_utils::HttpErrorStreamingClient;
    use futures::StreamExt;

    let body = r#"{"type":"error","error":{"type":"overloaded_error","message":"slow down"}}"#;
    let http = HttpErrorStreamingClient::new(http::StatusCode::SERVICE_UNAVAILABLE, body);
    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);

    let stream = crate::driver::tests::stream(&wire, &http, hello_request(), None)
        .expect("the streamed request encodes");
    let mut stream = Box::pin(stream);

    // The transport failure surfaces as the first error item yielded by the stream.
    let error = loop {
        match stream.next().await {
            Some(Ok(_)) => continue,
            Some(Err(error)) => break error,
            None => panic!("stream ended without yielding the transport error"),
        }
    };

    // A rejected SSE handshake is the provider's reply, classified like the
    // unary driver's and the in-band envelopes': one funnel.
    assert!(matches!(error, ProviderError::ProviderResponse(_)));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::SERVICE_UNAVAILABLE)
    );
    assert_eq!(error.provider_response_body(), Some(body));

    // The transport failure ends the stream: nothing may follow it that
    // would read as a successfully completed turn.
    assert!(stream.next().await.is_none());
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

        let converted = convert(msg)
            .expect("URL PDF should convert")
            .expect("a block survives");
        let json = serde_json::to_value(&converted).expect("message should serialize");

        assert_eq!(
            json.pointer("/content/0/source"),
            Some(&json!({ "type": "url", "url": pdf_url })),
            "URL PDF should map to a url document source: {json:#}"
        );
    }
}

/// Raw-capture tests: `CompletionResponse::raw` driven end to end over a
/// mock transport that hands back a Messages body *and* a `request-id`
/// response header. `raw` is the verbatim reply document — not a
/// re-serialization of the parsed type — so it answers what the normalized
/// response does not, and the transport id the driver read off the headers
/// reaches the normalized response rather than the document.
/// `with_error_response_headers` with `200 OK` is the one unary double
/// that carries response headers.
mod raw_capture {
    use super::*;
    use crate::test_utils::RecordingHttpClient;

    const REQUEST_ID: &str = "req_unit_anthropic_0001";

    /// A Messages body whose `stop_sequence` is set: the normalized
    /// response maps it to `FinishReason::Stop` and drops which sequence
    /// fired, so the capture provably answers more than the fold does.
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

    fn http() -> RecordingHttpClient {
        let mut headers = http::HeaderMap::new();
        headers.insert("request-id", http::HeaderValue::from_static(REQUEST_ID));
        RecordingHttpClient::with_error_response_headers(http::StatusCode::OK, BODY, headers)
    }

    /// The load-bearing capture property: `raw` is the reply Anthropic
    /// sent, verbatim — it still carries the `type` tag the wire type does
    /// not model, which is how you can tell it is the document and not a
    /// projection of it — it deserializes into Anthropic's own
    /// `CompletionResponse`, and it answers `stop_sequence`, which the
    /// normalized response drops.
    #[tokio::test]
    async fn completion_captures_raw_that_round_trips_into_the_wire_type() {
        let wire = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);
        let response = crate::driver::Model::new(wire.clone(), http())
            .call(hello_request())
            .await
            .expect("completion");

        let raw = &response.raw;
        assert_eq!(
            raw["type"], "message",
            "raw must be the verbatim reply, tag included"
        );
        let typed: CompletionResponse =
            serde_json::from_value(raw.clone()).expect("raw must deserialize");
        assert_eq!(typed.id, "msg_raw_1");
        assert_eq!(typed.model, "claude-sonnet-4-6");
        assert_eq!(typed.stop_reason.as_deref(), Some("stop_sequence"));
        assert_eq!(typed.stop_sequence.as_deref(), Some("alpha"));
        assert_eq!(typed.usage.input_tokens, 7);
        assert_eq!(typed.usage.output_tokens, 2);
        assert_eq!(raw["stop_sequence"], "alpha");

        // The transport id is not part of any reply document; the driver
        // read it off the `request-id` header and stamped the normalized
        // response with it.
        assert!(
            raw.get("provider_request_id").is_none(),
            "the document carries no transport id"
        );
        assert_eq!(response.provider_request_id.as_deref(), Some(REQUEST_ID));

        // The normalized response reports the reason and drops which
        // sequence fired — the whole reason `raw` is worth capturing.
        assert_eq!(
            response.finish_reason(),
            Some(completion::FinishReason::Stop)
        );
        assert_eq!(response.model(), Some("claude-sonnet-4-6"));
    }
}

/// A call rig issued an id for is spelled with one request-local alias on
/// the call and on its result.
#[test]
fn a_rig_issued_call_id_is_spelled_alike_on_call_and_result() {
    let call = message::ToolCall::new(
        message::CallId::from_wire(""),
        message::ToolFunction::new(message::ToolName::new("add").expect("tool name"), json!({})),
    );
    let request = completion_request_with_history(
        vec![
            message::Message::user("Add."),
            message::Message::Assistant(message::AssistantMessage::new(vec![
                message::AssistantContent::ToolCall(call.clone()),
            ])),
            message::Message::tool_results(vec![
                call.result(vec![message::ToolResultContent::text("3")]),
            ]),
        ],
        None,
    );
    let wire = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
        model: CLAUDE_SONNET_4_6,
        request,
        prompt_caching: false,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();
    let value = serde_json::to_value(wire).unwrap();
    assert_eq!(value["messages"][1]["content"][0]["id"], "tool-0");
    assert_eq!(value["messages"][2]["content"][0]["tool_use_id"], "tool-0");
}

/// pi's rule, which Z.AI needs: consecutive user messages of tool results
/// go as one user message.
#[test]
fn consecutive_tool_results_merge_into_one_user_message() {
    let add = || message::ToolName::new("add").expect("tool name");
    let request = completion_request_with_history(
        vec![
            message::Message::user("Add twice."),
            message::Message::Assistant(message::AssistantMessage::new(vec![
                message::AssistantContent::tool_call("toolu_1", add(), json!({})),
                message::AssistantContent::tool_call("toolu_2", add(), json!({})),
            ])),
            message::Message::tool_result(message::CallId::from_wire("toolu_1"), add(), "1"),
            message::Message::tool_result(message::CallId::from_wire("toolu_2"), add(), "2"),
        ],
        None,
    );
    let wire = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
        model: CLAUDE_SONNET_4_6,
        request,
        prompt_caching: false,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();
    let value = serde_json::to_value(wire).unwrap();
    let messages = value["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[2]["content"][0]["tool_use_id"], "toolu_1");
    assert_eq!(messages[2]["content"][1]["tool_use_id"], "toolu_2");
}

/// A `tool_use` whose `input` is not an object keeps its call, with the
/// arguments normalized, and no provider item: the item could not be sent
/// back as it is.
#[test]
fn a_tool_use_whose_input_is_not_an_object_keeps_its_call() {
    let value = json!({
        "id": "msg_bad", "model": CLAUDE_SONNET_4_6, "role": "assistant",
        "stop_reason": "tool_use", "stop_sequence": null,
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "content": [{"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": [1]}]
    });
    let response = fold_reply(&value).expect("the reply folds");
    let [message::AssistantContent::ToolCall(call)] = response.choice.as_slice() else {
        panic!("one call: {:?}", response.choice);
    };
    assert_eq!(call.function.arguments_value(), json!({}));
    assert_eq!(call.function.invalid_arguments.as_deref(), Some("[1]"));
    assert!(response.choice[0].native_item().is_none());
}

/// A failed tool result is sent with `is_error`; a successful one leaves
/// the field to its default.
#[test]
fn a_failed_tool_result_is_sent_as_an_error() {
    let name = message::ToolName::new("lookup").expect("tool name");
    let results = [false, true].map(|is_error| {
        message::UserContent::ToolResult(message::ToolResult {
            call: message::CallId::from_wire("toolu_1"),
            name: name.clone(),
            content: vec![message::ToolResultContent::text("boom")],
            is_error,
        })
    });
    let converted = convert(message::Message::User {
        content: results.to_vec(),
    })
    .expect("the results convert")
    .expect("a message");
    let value = serde_json::to_value(&converted).expect("the message serializes");
    assert!(value["content"][0].get("is_error").is_none());
    assert_eq!(value["content"][1]["is_error"], json!(true));
}

/// A hand-built assistant image, which `adapt` leaves on a same-model
/// turn, is sent as its placeholder: assistant turns take no images.
#[test]
fn an_assistant_image_is_sent_as_its_placeholder() {
    let image = message::AssistantContent::Image(message::Image {
        data: message::DocumentSourceKind::base64("aGk="),
        media_type: Some(message::ImageMediaType::PNG),
        ..message::Image::default()
    });
    assert_eq!(
        assistant_wire(vec![image]),
        [Content::from(
            crate::completion::history::ASSISTANT_IMAGE_OMITTED.to_owned()
        )]
    );
}

/// The container the last same-model turn ran in is the next request's,
/// unless the request names its own; another model's turn names none.
#[test]
fn the_last_turn_container_is_replayed_unless_the_request_names_one() {
    use crate::wire::{Operation, Wire};

    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);
    let turn = |model: &str, container: &str| {
        message::Message::Assistant(
            message::AssistantMessage {
                content: vec![message::AssistantContent::text("ran it")],
                origin: Some(message::Origin::new(
                    "anthropic.messages",
                    "anthropic",
                    model,
                )),
                stop: Some(message::StopReason::Stop),
                native: None,
            }
            .with_native(
                json!({"container": {"id": container, "expires_at": "2026-10-02T00:00:00Z"}}),
            ),
        )
    };
    let body = |history: Vec<message::Message>, params: Option<serde_json::Value>| {
        let mut request = completion_request_with_history(history, None);
        request.additional_params = params;
        let request = crate::operation::Completion::prepare(request, &wire.describe())
            .expect("the request prepares");
        json_body(
            &wire
                .encode(request, crate::wire::Mode::Unary)
                .expect("the request encodes")
                .request,
        )
    };
    let history = vec![
        message::Message::user("run it"),
        turn(CLAUDE_SONNET_4_6, "container_old"),
        message::Message::user("again"),
        turn(CLAUDE_SONNET_4_6, "container_new"),
        message::Message::user("next"),
    ];
    assert_eq!(
        body(history.clone(), None)["container"],
        json!("container_new")
    );
    assert_eq!(
        body(history, Some(json!({"container": "container_mine"})))["container"],
        json!("container_mine")
    );
    let foreign = vec![
        message::Message::user("run it"),
        turn(CLAUDE_OPUS_4_8, "container_other"),
        message::Message::user("next"),
    ];
    assert!(body(foreign, None).get("container").is_none());
}

/// A tool-result image takes the sources a user image does, a URL among
/// them, rather than being refused.
#[test]
fn a_tool_result_image_by_url_is_sent_by_url() {
    let result = message::UserContent::ToolResult(message::ToolResult {
        call: message::CallId::from_wire("toolu_1"),
        name: message::ToolName::new("screenshot").expect("tool name"),
        content: vec![message::ToolResultContent::Image(message::Image {
            data: message::DocumentSourceKind::Url("https://example.com/shot.png".to_owned()),
            ..message::Image::default()
        })],
        is_error: false,
    });
    let converted = convert(message::Message::User {
        content: vec![result],
    })
    .expect("the result converts")
    .expect("a message");
    let value = serde_json::to_value(&converted).expect("the message serializes");
    assert_eq!(
        value["content"][0]["content"][0]["source"],
        json!({"type": "url", "url": "https://example.com/shot.png"})
    );
}
