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
    AwsCompletionRequest {
        inner: request,
        prompt_caching,
    }
}

#[test]
fn test_tool_choice_auto_conversion() {
    // Test that rig's ToolChoice::Auto converts to AWS Auto
    let request = CompletionRequest {
        model: None,
        tool_choice: Some(ToolChoice::Auto),
        tools: vec![ToolDefinition {
            name: "test_tool".to_string(),
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
            name: "test_tool".to_string(),
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
            name: "test_tool".to_string(),
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
            function_names: vec!["specific_tool".to_string()],
        }),
        tools: vec![ToolDefinition {
            name: "specific_tool".to_string(),
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
            name: "test_tool".to_string(),
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
            name: "document_list".to_string(),
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
            name: "get_weather".to_string(),
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
    let reasoning =
        rig_core::message::Reasoning::new_with_signature("thinking", Some("sig".to_string()));
    let request = CompletionRequest {
        chat_history: vec![
            Message::User {
                content: vec![UserContent::Text(Text::new("user prompt".to_string()))],
            },
            Message::Assistant {
                id: None,
                content: vec![rig_core::completion::AssistantContent::Reasoning(reasoning)],
            },
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
