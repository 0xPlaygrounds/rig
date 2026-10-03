use super::*;
use rig_core::completion::{CompletionRequest, ToolDefinition};
use rig_core::message::{Message, Text, ToolChoice, UserContent};

// Helper to create a minimal CompletionRequest for testing
fn minimal_request() -> CompletionRequest {
    CompletionRequest {
        model: None,
        chat_history: vec![Message::User {
            content: vec![UserContent::Text(Text::new("test".to_string()))],
        }],
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

fn aws_request(request: CompletionRequest, prompt_caching: bool) -> AwsCompletionRequest {
    AwsCompletionRequest::new(request, Family::Other, prompt_caching)
}

/// Synthetic transcript checks wire correlation without a live Bedrock call.
#[test]
fn full_request_preserves_typed_tool_pairs_across_turns() {
    use rig_core::message::{AssistantContent, CallId, ToolCall, ToolFunction};
    let generated = ToolCall::new(
        CallId::from_wire(""),
        ToolFunction::new(
            rig_core::message::ToolName::new("test").expect("tool name"),
            serde_json::json!({}),
        ),
    );
    let hint = generated.id.wire().into_owned();
    let real = ToolCall::from_wire(&hint, generated.function.clone());
    let call = |call: &ToolCall| Message::from(vec![AssistantContent::ToolCall(call.clone())]);
    let result = |call: &ToolCall| Message::User {
        content: vec![UserContent::tool_result(
            call.id.clone(),
            rig_core::message::ToolName::new("test").expect("tool name"),
            vec![rig_core::message::ToolResultContent::text("done")],
        )],
    };
    let mut request = minimal_request();
    request.chat_history = vec![
        Message::system("system"),
        call(&generated),
        call(&real),
        result(&real),
        Message::assistant("intervening text"),
        result(&generated),
        call(&generated),
        result(&generated),
    ];
    let messages = aws_request(request, false).messages().unwrap();
    let mut calls = Vec::new();
    let mut results = Vec::new();
    for part in messages.iter().flat_map(|message| &message.content) {
        match part {
            aws_bedrock::ContentBlock::ToolUse(call) => calls.push(call.tool_use_id.clone()),
            aws_bedrock::ContentBlock::ToolResult(result) => {
                results.push(result.tool_use_id.clone())
            }
            _ => {}
        }
    }
    assert_eq!(calls.len(), 3);
    assert_eq!(
        results,
        [calls[1].clone(), calls[0].clone(), calls[2].clone()]
    );
    // The provider's id is sent as it is, and the rig-issued id is one alias
    // wherever its call appears, distinct from it.
    assert_eq!(calls[1], hint);
    assert_eq!(calls[0], calls[2]);
    assert_ne!(calls[0], calls[1]);
}

#[test]
fn test_tool_choice_auto_conversion() {
    // Test that rig's ToolChoice::Auto converts to AWS Auto
    let request = CompletionRequest {
        model: None,
        tool_choice: Some(ToolChoice::Auto),
        tools: vec![ToolDefinition {
            name: rig_core::message::ToolName::new("test_tool").expect("tool name"),
            description: "A test tool".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        }],
        ..minimal_request()
    };

    let aws_request = aws_request(request, false);
    let tool_config = aws_request
        .tools_config()
        .expect("Should build tool config");

    assert!(tool_config.is_some());

    let config = tool_config.unwrap();

    assert!(config.tool_choice().is_some());
    assert!(matches!(
        config.tool_choice().unwrap(),
        aws_bedrock::ToolChoice::Auto(_)
    ));
}

#[test]
fn test_tool_choice_required_conversion() {
    // Test that rig's ToolChoice::Required converts to AWS Any
    let request = CompletionRequest {
        model: None,
        tool_choice: Some(ToolChoice::Required),
        tools: vec![ToolDefinition {
            name: rig_core::message::ToolName::new("test_tool").expect("tool name"),
            description: "A test tool".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        }],
        ..minimal_request()
    };

    let aws_request = aws_request(request, false);
    let tool_config = aws_request
        .tools_config()
        .expect("Should build tool config");

    assert!(tool_config.is_some());
    let config = tool_config.unwrap();
    assert!(config.tool_choice().is_some());

    // Verify it's the Any variant
    assert!(matches!(
        config.tool_choice().unwrap(),
        aws_bedrock::ToolChoice::Any(_)
    ));
}

#[test]
fn test_tool_choice_none_conversion() {
    // Test that rig's ToolChoice::None disables Bedrock tool configuration entirely.
    let request = CompletionRequest {
        model: None,
        tool_choice: Some(ToolChoice::None),
        tools: vec![ToolDefinition {
            name: rig_core::message::ToolName::new("test_tool").expect("tool name"),
            description: "A test tool".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        }],
        ..minimal_request()
    };

    let aws_request = aws_request(request, false);
    let tool_config = aws_request
        .tools_config()
        .expect("Should build tool config");

    assert!(tool_config.is_none());
}

#[test]
fn test_tool_choice_specific_conversion() {
    // Test that rig's ToolChoice::Specific converts to AWS Tool
    let request = CompletionRequest {
        model: None,
        tool_choice: Some(ToolChoice::Specific {
            function_names: vec![
                rig_core::message::ToolName::new("specific_tool").expect("tool name"),
            ],
        }),
        tools: vec![ToolDefinition {
            name: rig_core::message::ToolName::new("specific_tool").expect("tool name"),
            description: "A specific tool".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        }],
        ..minimal_request()
    };

    let aws_request = aws_request(request, false);
    let tool_config = aws_request
        .tools_config()
        .expect("Should build tool config");

    assert!(tool_config.is_some());

    let config = tool_config.unwrap();

    assert!(config.tool_choice().is_some());
    assert!(matches!(
        config.tool_choice().unwrap(),
        aws_bedrock::ToolChoice::Tool(specific) if specific.name() == "specific_tool"
    ));
}

#[test]
fn test_no_tool_choice_when_not_specified() {
    // Test that when tool_choice is None (not set), it defaults to None in AWS
    let request = CompletionRequest {
        model: None,
        tool_choice: None, // Not set
        tools: vec![ToolDefinition {
            name: rig_core::message::ToolName::new("test_tool").expect("tool name"),
            description: "A test tool".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        }],
        ..minimal_request()
    };

    let aws_request = aws_request(request, false);
    let tool_config = aws_request
        .tools_config()
        .expect("Should build tool config");

    assert!(tool_config.is_some());
    let config = tool_config.unwrap();
    // When not specified, should be None
    assert!(config.tool_choice().is_none());
}

#[test]
fn test_tool_with_empty_parameters() {
    // Test that tools with empty parameters (like document_list) work correctly
    let request = CompletionRequest {
        model: None,
        tools: vec![ToolDefinition {
            name: rig_core::message::ToolName::new("document_list").expect("tool name"),
            description: "Lists all documents".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        }],
        ..minimal_request()
    };

    let aws_request = aws_request(request, false);
    let tool_config = aws_request
        .tools_config()
        .expect("Should build tool config");

    assert!(tool_config.is_some());
    let config = tool_config.unwrap();
    assert_eq!(config.tools().len(), 1);

    // Verify the tool was created correctly
    assert!(
        matches!(&config.tools()[0], aws_bedrock::Tool::ToolSpec(spec)
            if spec.name() == "document_list"
            && spec.description() == Some("Lists all documents")
            && spec.input_schema().is_some()
        )
    );
}

#[test]
fn test_tool_with_parameters() {
    // Test that tools with parameters work correctly
    let request = CompletionRequest {
        model: None,
        tools: vec![ToolDefinition {
            name: rig_core::message::ToolName::new("get_weather").expect("tool name"),
            description: "Get weather for a location".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }),
        }],
        ..minimal_request()
    };

    let aws_request = aws_request(request, false);
    let tool_config = aws_request
        .tools_config()
        .expect("Should build tool config");

    assert!(tool_config.is_some());

    let config = tool_config.unwrap();

    assert_eq!(config.tools().len(), 1);
    assert!(
        matches!(&config.tools()[0], aws_bedrock::Tool::ToolSpec(spec)
            if spec.name() == "get_weather"
            && spec.description() == Some("Get weather for a location")
        )
    );
}

#[test]
fn test_system_prompt_includes_system_history() {
    let request = CompletionRequest {
        model: None,
        chat_history: vec![
            Message::system("History system instruction"),
            Message::User {
                content: vec![UserContent::Text(Text::new("test".to_string()))],
            },
        ],
        ..minimal_request()
    };

    let aws_request = aws_request(request, false);
    let system_prompt = aws_request
        .system_prompt()
        .expect("system prompt should build")
        .expect("system prompt should exist");

    assert_eq!(system_prompt.len(), 1);
    assert_eq!(
        system_prompt.first(),
        Some(&aws_bedrock::SystemContentBlock::Text(
            "History system instruction".to_string()
        ))
    );
}

#[test]
fn test_system_prompt_appends_cache_point_when_prompt_caching_enabled() {
    let mut request = minimal_request();
    request
        .chat_history
        .insert(0, Message::system("System prompt"));

    let aws_request = aws_request(request, true);
    let system_prompt = aws_request
        .system_prompt()
        .expect("system prompt should build")
        .expect("system prompt should exist");

    assert_eq!(system_prompt.len(), 2);
    assert_eq!(
        system_prompt.first(),
        Some(&aws_bedrock::SystemContentBlock::Text(
            "System prompt".to_string()
        ))
    );
    assert!(matches!(
        system_prompt.last(),
        Some(aws_bedrock::SystemContentBlock::CachePoint(_))
    ));
}

#[test]
fn test_messages_exclude_system_history() {
    let request = CompletionRequest {
        model: None,
        chat_history: vec![
            Message::system("History system instruction"),
            Message::User {
                content: vec![UserContent::Text(Text::new("test".to_string()))],
            },
        ],
        ..minimal_request()
    };

    let aws_request = aws_request(request, false);
    let messages = aws_request.messages().expect("messages should convert");
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].role, aws_bedrock::ConversationRole::User);
}

#[test]
fn test_messages_append_cache_point_when_prompt_caching_enabled() {
    let aws_request = aws_request(minimal_request(), true);

    let messages = aws_request.messages().expect("messages should convert");

    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].role, aws_bedrock::ConversationRole::User);
    assert_eq!(messages[0].content.len(), 2);
    assert!(matches!(
        messages[0].content.last(),
        Some(aws_bedrock::ContentBlock::CachePoint(_))
    ));
}

#[test]
fn test_messages_skip_cache_point_when_history_contains_reasoning() {
    // Bedrock's Anthropic backend rejects "Cache point cannot be inserted
    // after reasoning block" whenever the chat history carries a prior
    // reasoning turn, even if the literal trailing block is a tool result.
    // Verify the message-level checkpoint is suppressed in that case.
    let reasoning = rig_core::message::AssistantContent::reasoning("thinking")
        .with_native(crate::types::block::reasoning_json("thinking", Some("sig")));
    let request = CompletionRequest {
        chat_history: vec![
            Message::User {
                content: vec![UserContent::Text(Text::new("user prompt".to_string()))],
            },
            Message::from(vec![reasoning]),
            Message::User {
                content: vec![UserContent::Text(Text::new("follow up".to_string()))],
            },
        ],
        ..minimal_request()
    };

    let aws_request = aws_request(request, true);

    // The system-prompt cache point path is independent and unaffected;
    // read it before `messages()` consumes the request.
    let system_only = aws_request.system_prompt().expect("system prompt builds");
    assert!(system_only.is_none() || !system_only.unwrap().is_empty());

    let messages = aws_request.messages().expect("messages should convert");

    let last_message = messages.last().expect("messages should not be empty");
    assert!(
        !last_message
            .content
            .iter()
            .any(|c| matches!(c, aws_bedrock::ContentBlock::CachePoint(_))),
        "message-level cache point should be skipped when chat history contains reasoning"
    );
}

#[test]
fn test_output_config_none_when_no_schema() {
    let request = minimal_request();
    let aws_request = aws_request(request, false);
    assert!(
        aws_request
            .output_config()
            .expect("output config builds")
            .is_none()
    );
}

#[test]
fn test_output_config_with_schema() {
    let schema: schemars::Schema = serde_json::from_value(serde_json::json!({
        "type": "object",
        "title": "WeatherResponse",
        "properties": {
            "temperature": { "type": "number" },
            "unit": { "type": "string", "enum": ["celsius", "fahrenheit"] }
        },
        "required": ["temperature", "unit"]
    }))
    .expect("valid schema");

    let request = CompletionRequest {
        output_schema: Some(schema),
        ..minimal_request()
    };

    let aws_request = aws_request(request, false);
    let output_config = aws_request.output_config().expect("output config builds");

    assert!(output_config.is_some());
    let config = output_config.unwrap();
    let text_format = config.text_format().expect("text_format should be set");
    assert_eq!(
        *text_format.r#type(),
        aws_bedrock::OutputFormatType::JsonSchema
    );

    let structure = text_format.structure().expect("structure should be set");
    let json_schema = structure
        .as_json_schema()
        .expect("should be JsonSchema variant");
    assert_eq!(json_schema.name(), Some("WeatherResponse"));

    let parsed: serde_json::Value =
        serde_json::from_str(json_schema.schema()).expect("schema should be valid JSON");
    assert_eq!(parsed["type"], "object");
    assert!(parsed["properties"]["temperature"].is_object());
}

#[test]
fn test_output_config_uses_default_name() {
    let schema: schemars::Schema = serde_json::from_value(serde_json::json!({
        "type": "object",
        "properties": {
            "result": { "type": "string" }
        }
    }))
    .expect("valid schema");

    let request = CompletionRequest {
        output_schema: Some(schema),
        ..minimal_request()
    };

    let aws_request = aws_request(request, false);
    let config = aws_request
        .output_config()
        .expect("output config builds")
        .expect("should have config");
    let text_format = config.text_format().expect("text_format should be set");
    let structure = text_format.structure().expect("structure should be set");
    let json_schema = structure
        .as_json_schema()
        .expect("should be JsonSchema variant");
    assert_eq!(json_schema.name(), Some("response_schema"));
}

fn document_request(prompt_document: bool) -> CompletionRequest {
    use rig_core::completion::Document;
    use rig_core::message::DocumentMediaType;
    let mut request = minimal_request();
    request.chat_history = vec![
        Message::system("Answer with the exact token from the document only."),
        Message::assistant("Acknowledged."),
    ];
    if prompt_document {
        request.chat_history.push(Message::User {
            content: vec![
                UserContent::document_text("A repeated attachment.", Some(DocumentMediaType::TXT)),
                UserContent::document_text("A repeated attachment.", Some(DocumentMediaType::TXT)),
                UserContent::text("According to the document, what is the ordering token?"),
            ],
        });
    } else {
        request.chat_history.push(Message::user(
            "According to the document, what is the ordering token?",
        ));
    }
    request.documents = vec![Document {
        id: "ordering-note".to_owned(),
        text: "The ordering token is violet-needle.".to_owned(),
        additional_props: Default::default(),
    }];
    request
}

fn document_names(messages: &[aws_bedrock::Message]) -> Vec<String> {
    messages
        .iter()
        .flat_map(|message| &message.content)
        .filter_map(|block| match block {
            aws_bedrock::ContentBlock::Document(document) => Some(document.name.clone()),
            _ => None,
        })
        .collect()
}

/// Static documents lead the conversation as their own user message, the
/// system instruction stays in `system`, and history follows in order.
#[test]
fn documents_are_prepended_before_history() {
    let request = aws_request(document_request(false), false);
    let system = request.system_prompt().expect("system").expect("present");
    assert!(format!("{system:?}").contains("exact token"));
    let messages = request.messages().expect("messages");
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0].role, aws_bedrock::ConversationRole::User);
    assert!(
        messages[0]
            .content
            .iter()
            .any(|block| matches!(block, aws_bedrock::ContentBlock::Document(_))),
        "{:?}",
        messages[0]
    );
    assert_eq!(messages[1].role, aws_bedrock::ConversationRole::Assistant);
    assert_eq!(messages[2].role, aws_bedrock::ConversationRole::User);
}

/// Document names are a function of the request: encoding it twice gives the
/// same bytes, and the same content sent twice in one request gets two
/// distinct names.
#[test]
fn document_names_are_deterministic_and_unique_within_a_request() {
    let first = document_names(
        &aws_request(document_request(true), false)
            .messages()
            .expect("messages"),
    );
    let second = document_names(
        &aws_request(document_request(true), false)
            .messages()
            .expect("messages"),
    );
    assert_eq!(first, second);
    assert_eq!(first.len(), 3);
    let unique: std::collections::HashSet<_> = first.iter().collect();
    assert_eq!(unique.len(), 3, "{first:?}");
    assert!(first.iter().all(|name| name.starts_with("document-")));
    assert_eq!(first[2], format!("{}-2", first[1]));
}

/// A failed, refused or synthetic result goes back with `status: error`;
/// any other has no status, which Converse reads as a success.
/// Converse documents a result's `status` only for Nova and Claude, so a
/// failed result states it to those families alone.
#[test]
fn only_nova_and_claude_get_a_result_status() {
    use crate::completion::{AMAZON_NOVA_PRO, ANTHROPIC_CLAUDE_SONNET_4_5, LLAMA_3_1_70B_INSTRUCT};
    use rig_core::message::{CallId, ToolResult, ToolResultContent};
    let result = |id: &str, is_error| {
        UserContent::ToolResult(ToolResult {
            call: CallId::from_wire(id),
            name: rig_core::message::ToolName::new("t").expect("a tool name"),
            content: vec![ToolResultContent::text("out")],
            is_error,
        })
    };
    let error = Some(aws_bedrock::ToolResultStatus::Error);
    for (model, failed) in [
        (AMAZON_NOVA_PRO, error.clone()),
        (ANTHROPIC_CLAUDE_SONNET_4_5, error),
        (LLAMA_3_1_70B_INSTRUCT, None),
    ] {
        let mut request = minimal_request();
        request.chat_history = vec![Message::User {
            content: vec![result("failed", true), result("fine", false)],
        }];
        let messages = AwsCompletionRequest::new(request, Family::of(model), false)
            .messages()
            .expect("messages");
        let statuses: Vec<_> = messages[0]
            .content
            .iter()
            .map(|block| match block {
                aws_bedrock::ContentBlock::ToolResult(result) => result.status.clone(),
                other => panic!("{other:?}"),
            })
            .collect();
        assert_eq!(statuses, [failed, None], "{model}");
    }
}

/// Only the leading system messages are system blocks. A later one stays
/// where the history put it, as user text joined to its neighbouring user
/// content, so the cached prefix before it never changes.
#[test]
fn a_later_system_message_stays_in_place() {
    let mut request = minimal_request();
    request.chat_history = vec![
        Message::system("lead"),
        Message::user("q"),
        Message::assistant("a"),
        Message::system("steer"),
        Message::user("next"),
    ];
    let request = aws_request(request, false);
    assert_eq!(
        request.system_prompt().expect("system"),
        Some(vec![aws_bedrock::SystemContentBlock::Text(
            "lead".to_owned()
        )])
    );
    let messages = request.messages().expect("messages");
    let text = |text: &str| aws_bedrock::ContentBlock::Text(text.to_owned());
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0].content, [text("q")]);
    assert_eq!(messages[1].content, [text("a")]);
    assert_eq!(messages[2].role, aws_bedrock::ConversationRole::User);
    assert_eq!(messages[2].content, [text("steer"), text("next")]);
}

/// A hosted tool's id is reserved: an id rig issued never takes it, and
/// the hosted use and result keep theirs.
#[test]
fn hosted_tool_ids_are_reserved() {
    use rig_core::message::{AssistantContent, CallId, Opaque, ToolCall, ToolFunction};
    let name = rig_core::message::ToolName::new("lookup").expect("a tool name");
    let issued = ToolCall::new(
        CallId::from_wire(""),
        ToolFunction::new(name.clone(), serde_json::json!({})),
    );
    let hosted = |item| AssistantContent::Opaque(Opaque { item, replay: true });
    let mut request = minimal_request();
    request.chat_history = vec![
        Message::user("q"),
        Message::from(vec![
            hosted(serde_json::json!({ "toolUse": {
                "toolUseId": "tool-0", "name": "nova_grounding", "input": {}, "type": "server_tool_use",
            } })),
            hosted(serde_json::json!({ "toolResult": {
                "toolUseId": "tool-0", "content": [{ "text": "found" }],
            } })),
            AssistantContent::ToolCall(issued.clone()),
        ]),
        Message::User {
            content: vec![UserContent::tool_result(
                issued.id.clone(),
                name,
                vec![rig_core::message::ToolResultContent::text("done")],
            )],
        },
    ];
    let messages = aws_request(request, false).messages().expect("messages");
    let ids: Vec<_> = messages
        .iter()
        .flat_map(|message| &message.content)
        .filter_map(|block| match block {
            aws_bedrock::ContentBlock::ToolUse(call) => Some(call.tool_use_id.clone()),
            aws_bedrock::ContentBlock::ToolResult(result) => Some(result.tool_use_id.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(ids, ["tool-0", "tool-0", "tool-1", "tool-1"]);
}
