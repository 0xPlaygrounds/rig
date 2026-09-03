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

/// `functionResponse.name` is the executed function's name: read from
/// the required `ToolResult::name` — never an identifier, no matter how
/// identifier-shaped the correlation handles are.
#[test]
fn tool_result_serializes_the_executed_name_not_an_identifier() {
    use rig_core::message::{
        AssistantContent, ProviderCallId, ToolCall, ToolCallId, ToolFunction, ToolResult,
        ToolResultContent,
    };

    let call = |wire_id: &str, name: &str| Message::Assistant {
        id: None,
        content: vec![AssistantContent::ToolCall(ToolCall::from_wire(
            wire_id,
            ToolFunction {
                name: name.to_owned(),
                arguments: serde_json::json!({}),
            },
        ))],
    };
    let result = |wire_id: &str, name: &str| Message::User {
        content: vec![UserContent::ToolResult(ToolResult {
            call: ToolCallId::new_or_minted(wire_id, 0),
            provider: ProviderCallId::new(wire_id),
            name: name.to_owned(),
            content: vec![ToolResultContent::text("out")],
        })],
    };
    let call_dual = |item_id: &str, call_id: &str, name: &str| Message::Assistant {
        id: None,
        content: vec![AssistantContent::ToolCall(ToolCall::from_dual_wire(
            item_id,
            call_id,
            ToolFunction {
                name: name.to_owned(),
                arguments: serde_json::json!({}),
            },
        ))],
    };
    let result_dual = |item_id: &str, call_id: &str, name: &str| Message::User {
        content: vec![UserContent::ToolResult(ToolResult {
            call: ToolCallId::new_or_minted(call_id, 0),
            provider: ProviderCallId::new(call_id).map(|provider| provider.with_item_id(item_id)),
            name: name.to_owned(),
            content: vec![ToolResultContent::text("out")],
        })],
    };

    let request = CompletionRequest {
        chat_history: vec![
            // A driver-built result carries the executed name (a repair
            // hook renamed the call: `sum` ran, not `add`) — the wire
            // name comes from the result, not the call.
            call("call_1", "add"),
            result("call_1", "sum"),
            // A cross-provider history with an OpenAI-shaped identifier:
            // `call_abc` must never reach the wire as a name.
            call("call_abc", "get_weather"),
            result("call_abc", "get_weather"),
            // A dual-identifier history (OpenAI Responses: item id
            // `fc_…` + correlator `call_…`, both carried on `provider`) —
            // `fc_1` must never reach the wire as a name.
            call_dual("fc_1", "call_9", "get_time"),
            result_dual("fc_1", "call_9", "get_time"),
        ],
        ..minimal_request()
    };

    let contents = VertexCompletionRequest(request)
        .contents()
        .expect("conversion should succeed");
    let response_names: Vec<String> = contents
        .iter()
        .flat_map(|content| content.parts.iter())
        .filter_map(|part| part.function_response().map(|fr| fr.name.clone()))
        .collect();

    assert_eq!(
        response_names,
        vec![
            "sum".to_owned(),
            "get_weather".to_owned(),
            "get_time".to_owned()
        ]
    );
}

#[test]
fn test_tool_choice_auto_conversion() {
    // Test that rig's ToolChoice::Auto converts to Vertex AI Auto mode
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

    let vertex_request = VertexCompletionRequest(request);
    let tool_config = vertex_request.tool_config();

    assert!(tool_config.is_some());
    let config = tool_config.unwrap();
    let function_calling_config = config.function_calling_config.as_ref();

    assert!(function_calling_config.is_some());
    assert_eq!(
        function_calling_config.unwrap().mode,
        vertexai::model::function_calling_config::Mode::Auto
    );
}

#[test]
fn test_tool_choice_required_conversion() {
    // Test that rig's ToolChoice::Required converts to Vertex AI Any mode
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

    let vertex_request = VertexCompletionRequest(request);
    let tool_config = vertex_request.tool_config();

    assert!(tool_config.is_some());
    let config = tool_config.unwrap();
    let function_calling_config = config.function_calling_config.as_ref();

    assert!(function_calling_config.is_some());
    assert_eq!(
        function_calling_config.unwrap().mode,
        vertexai::model::function_calling_config::Mode::Any
    );
}

#[test]
fn test_tool_choice_none_conversion() {
    // Test that rig's ToolChoice::None converts to Vertex AI None mode
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

    let vertex_request = VertexCompletionRequest(request);
    let tool_config = vertex_request.tool_config();

    assert!(tool_config.is_some());
    let config = tool_config.unwrap();
    let function_calling_config = config.function_calling_config.as_ref();

    assert!(function_calling_config.is_some());
    assert_eq!(
        function_calling_config.unwrap().mode,
        vertexai::model::function_calling_config::Mode::None
    );
}

#[test]
fn test_tool_choice_specific_conversion() {
    // Test that rig's ToolChoice::Specific converts to Vertex AI Any mode with allowed function names
    let request = CompletionRequest {
        model: None,
        tool_choice: Some(ToolChoice::Specific {
            function_names: vec!["test_tool".to_string()],
        }),
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

    let vertex_request = VertexCompletionRequest(request);
    let tool_config = vertex_request.tool_config();

    assert!(tool_config.is_some());
    let config = tool_config.unwrap();
    let function_calling_config = config.function_calling_config.as_ref();

    assert!(function_calling_config.is_some());
    let fcc = function_calling_config.unwrap();
    assert_eq!(
        fcc.mode,
        vertexai::model::function_calling_config::Mode::Any
    );
    // Verify allowed function names are set
    assert!(!fcc.allowed_function_names.is_empty());
    assert_eq!(fcc.allowed_function_names.len(), 1);
    assert_eq!(fcc.allowed_function_names[0], "test_tool");
}

#[test]
fn test_system_instruction_from_preamble() {
    // Test that preamble converts to system instruction
    let mut request = minimal_request();
    request
        .chat_history
        .insert(0, Message::system("You are a helpful assistant."));

    let vertex_request = VertexCompletionRequest(request);
    let system_instruction = vertex_request.system_instruction();

    assert!(system_instruction.is_some());
    let content = system_instruction.unwrap();
    assert_eq!(content.role.as_str(), "user");
    assert_eq!(content.parts.len(), 1);
    assert_eq!(
        content.parts[0].text(),
        Some(&"You are a helpful assistant.".to_string())
    );
}

#[test]
fn test_system_instruction_from_system_history_and_contents_skip_system() {
    let request = CompletionRequest {
        model: None,
        chat_history: vec![
            Message::system("System from history"),
            Message::User {
                content: vec![UserContent::Text(Text::new("hello".to_string()))],
            },
        ],
        ..minimal_request()
    };

    let vertex_request = VertexCompletionRequest(request);

    let system_instruction = vertex_request.system_instruction();
    assert!(system_instruction.is_some());
    let system_instruction = system_instruction.unwrap();
    assert_eq!(system_instruction.parts.len(), 1);
    assert_eq!(
        system_instruction.parts[0].text(),
        Some(&"System from history".to_string())
    );

    let contents = vertex_request.contents().expect("contents should convert");
    assert_eq!(contents.len(), 1);
    assert_eq!(contents[0].role.as_str(), "user");
}

#[test]
fn test_tools_conversion() {
    // Test that ToolDefinition converts to FunctionDeclaration
    let request = CompletionRequest {
        model: None,
        tools: vec![ToolDefinition {
            name: "add".to_string(),
            description: "Add two numbers".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "x": {"type": "number"},
                    "y": {"type": "number"}
                },
                "required": ["x", "y"]
            }),
        }],
        ..minimal_request()
    };

    let vertex_request = VertexCompletionRequest(request);
    let tools = vertex_request.tools();

    assert!(tools.is_some());
    let tool = tools.unwrap();
    // Verify function declarations exist
    assert!(!tool.function_declarations.is_empty());
    assert_eq!(tool.function_declarations.len(), 1);
    assert_eq!(tool.function_declarations[0].name.as_str(), "add");
    assert_eq!(
        tool.function_declarations[0].description.as_str(),
        "Add two numbers"
    );
}

#[test]
fn test_no_tool_choice_when_not_specified() {
    // Test that when tool_choice is None (not set), it defaults to Auto in Vertex AI
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

    let vertex_request = VertexCompletionRequest(request);
    let tool_config = vertex_request.tool_config();

    assert!(tool_config.is_some());
    let config = tool_config.unwrap();
    let function_calling_config = config.function_calling_config.as_ref();

    assert!(function_calling_config.is_some());
    // When not specified, should default to Auto
    assert_eq!(
        function_calling_config.unwrap().mode,
        vertexai::model::function_calling_config::Mode::Auto
    );
}

#[test]
fn test_tool_with_empty_parameters() {
    // Test that tools with empty parameters work correctly
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

    let vertex_request = VertexCompletionRequest(request);
    let tools = vertex_request.tools();

    assert!(tools.is_some());
    let tool = tools.unwrap();
    assert!(!tool.function_declarations.is_empty());
    assert_eq!(tool.function_declarations.len(), 1);
    assert_eq!(tool.function_declarations[0].name.as_str(), "document_list");
    assert_eq!(
        tool.function_declarations[0].description.as_str(),
        "Lists all documents"
    );
}

#[test]
fn test_tool_with_parameters() {
    // Test that tools with complex parameters work correctly
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

    let vertex_request = VertexCompletionRequest(request);
    let tools = vertex_request.tools();

    assert!(tools.is_some());
    let tool = tools.unwrap();
    assert!(!tool.function_declarations.is_empty());
    assert_eq!(tool.function_declarations.len(), 1);
    assert_eq!(tool.function_declarations[0].name.as_str(), "get_weather");
    assert_eq!(
        tool.function_declarations[0].description.as_str(),
        "Get weather for a location"
    );
}

#[test]
fn test_generation_config_with_temperature_and_max_tokens() {
    // Test that temperature and max_tokens convert to GenerationConfig
    let request = CompletionRequest {
        model: None,
        temperature: Some(0.7),
        max_tokens: Some(100),
        ..minimal_request()
    };

    let vertex_request = VertexCompletionRequest(request);
    let generation_config = vertex_request
        .generation_config()
        .expect("generation config should parse");

    assert!(generation_config.is_some());
    let config = generation_config.unwrap();
    assert_eq!(config.temperature, Some(0.7));
    assert_eq!(config.max_output_tokens, Some(100));
    assert_eq!(config.candidate_count, Some(1));
}

#[test]
fn generation_config_maps_thinking_budget_and_include_thoughts() {
    let request = CompletionRequest {
        additional_params: Some(serde_json::json!({
            "generationConfig": {
                "thinkingConfig": {
                    "thinkingBudget": 1024,
                    "includeThoughts": true
                }
            }
        })),
        ..minimal_request()
    };

    let config = VertexCompletionRequest(request)
        .generation_config()
        .expect("generation config should parse")
        .expect("generation config should exist");
    let thinking = config
        .thinking_config
        .expect("thinking config should be mapped");

    assert_eq!(thinking.thinking_budget, Some(1024));
    assert_eq!(thinking.include_thoughts, Some(true));
    assert_eq!(thinking.thinking_level, None);
}

#[test]
fn generation_config_maps_thinking_levels() {
    use vertexai::model::generation_config::thinking_config::ThinkingLevel as VertexThinkingLevel;

    for (level, expected) in [
        ("minimal", VertexThinkingLevel::Minimal),
        ("low", VertexThinkingLevel::Low),
        ("medium", VertexThinkingLevel::Medium),
        ("high", VertexThinkingLevel::High),
    ] {
        let request = CompletionRequest {
            additional_params: Some(serde_json::json!({
                "generationConfig": {
                    "thinkingConfig": { "thinkingLevel": level }
                }
            })),
            ..minimal_request()
        };

        let config = VertexCompletionRequest(request)
            .generation_config()
            .expect("generation config should parse")
            .expect("generation config should exist");
        let thinking = config
            .thinking_config
            .expect("thinking config should be mapped");

        assert_eq!(thinking.thinking_level, Some(expected));
    }
}

#[test]
fn generation_config_maps_supported_fields_and_typed_fields_take_precedence() {
    let request = CompletionRequest {
        temperature: Some(0.7),
        max_tokens: Some(100),
        additional_params: Some(serde_json::json!({
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 20,
                "candidateCount": 2,
                "stopSequences": ["END"],
                "responseMimeType": "application/json",
                "responseJsonSchema": { "type": "object" },
                "topP": 0.8,
                "topK": 40,
                "presencePenalty": 0.1,
                "frequencyPenalty": 0.2,
                "responseLogprobs": true,
                "logprobs": 3,
                "responseModalities": ["TEXT", "IMAGE"],
                "imageConfig": {
                    "aspectRatio": "16:9",
                    "imageSize": "1K"
                }
            }
        })),
        ..minimal_request()
    };

    let config = VertexCompletionRequest(request)
        .generation_config()
        .expect("generation config should parse")
        .expect("generation config should exist");

    assert_eq!(config.temperature, Some(0.7));
    assert_eq!(config.max_output_tokens, Some(100));
    assert_eq!(config.candidate_count, Some(1));
    assert_eq!(config.stop_sequences, ["END"]);
    assert_eq!(config.response_mime_type, "application/json");
    assert_eq!(
        serde_json::to_value(
            config
                .response_json_schema
                .expect("JSON schema should be mapped")
        )
        .expect("Vertex JSON schema should serialize"),
        serde_json::json!({ "type": "object" })
    );
    assert_eq!(config.top_p, Some(0.8));
    assert_eq!(config.top_k, Some(40.0));
    assert_eq!(config.presence_penalty, Some(0.1));
    assert_eq!(config.frequency_penalty, Some(0.2));
    assert_eq!(config.response_logprobs, Some(true));
    assert_eq!(config.logprobs, Some(3));
    assert_eq!(
        config.response_modalities,
        [
            vertexai::model::generation_config::Modality::Text,
            vertexai::model::generation_config::Modality::Image,
        ]
    );
    let image = config.image_config.expect("image config should be mapped");
    assert_eq!(image.aspect_ratio.as_deref(), Some("16:9"));
    assert_eq!(image.image_size.as_deref(), Some("1K"));
}

#[test]
fn generation_config_rejects_audio_response_modality() {
    let request = CompletionRequest {
        additional_params: Some(serde_json::json!({
            "generationConfig": { "responseModalities": ["AUDIO"] }
        })),
        ..minimal_request()
    };

    let error = VertexCompletionRequest(request)
        .generation_config()
        .expect_err("audio output must fail before the API call");
    assert!(matches!(error, CompletionError::RequestError(_)));
    assert!(error.to_string().contains("responseModalities AUDIO"));
}

#[test]
fn generation_config_rejects_out_of_f32_range_typed_and_provider_values() {
    let typed_request = CompletionRequest {
        temperature: Some(f64::MAX),
        ..minimal_request()
    };
    let typed_error = VertexCompletionRequest(typed_request)
        .generation_config()
        .expect_err("typed temperature beyond f32 must fail");
    assert!(matches!(typed_error, CompletionError::RequestError(_)));
    assert!(typed_error.to_string().contains("temperature"));

    let provider_error = vertex_generation_config(GeminiGenerationConfig {
        top_p: Some(f64::MAX),
        ..Default::default()
    })
    .expect_err("provider top_p beyond f32 must fail");
    assert!(matches!(provider_error, CompletionError::RequestError(_)));
    assert!(provider_error.to_string().contains("top_p"));
}

#[test]
fn generation_config_rejects_non_finite_typed_values() {
    let request = CompletionRequest {
        temperature: Some(f64::NAN),
        ..minimal_request()
    };

    let error = VertexCompletionRequest(request)
        .generation_config()
        .expect_err("non-finite typed temperature must fail");
    assert!(matches!(error, CompletionError::RequestError(_)));
    assert!(error.to_string().contains("temperature"));
}

#[test]
fn generation_config_maps_response_schema() {
    let request = CompletionRequest {
        additional_params: Some(serde_json::json!({
            "generationConfig": {
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "name": { "type": "STRING" }
                    }
                }
            }
        })),
        ..minimal_request()
    };

    let config = VertexCompletionRequest(request)
        .generation_config()
        .expect("generation config should parse")
        .expect("generation config should exist");

    let schema = config
        .response_schema
        .expect("response schema should be mapped");
    assert_eq!(schema.r#type, vertexai::model::Type::Object);
    assert_eq!(
        schema
            .properties
            .get("name")
            .map(|property| &property.r#type),
        Some(&vertexai::model::Type::String)
    );
}

#[test]
fn generation_config_typed_max_overrides_out_of_range_provider_max() {
    let request = CompletionRequest {
        max_tokens: Some(100),
        additional_params: Some(serde_json::json!({
            "generationConfig": { "maxOutputTokens": 2_147_483_648_u64 }
        })),
        ..minimal_request()
    };

    let config = VertexCompletionRequest(request)
        .generation_config()
        .expect("typed max should override an unrepresentable provider max")
        .expect("generation config should exist");
    assert_eq!(config.max_output_tokens, Some(100));
}

#[test]
fn generation_config_typed_temperature_overrides_out_of_range_provider_temperature() {
    let request = CompletionRequest {
        temperature: Some(0.7),
        additional_params: Some(serde_json::json!({
            "generationConfig": { "temperature": f64::MAX }
        })),
        ..minimal_request()
    };

    let config = VertexCompletionRequest(request)
        .generation_config()
        .expect("typed temperature should override an unrepresentable provider temperature")
        .expect("generation config should exist");
    assert_eq!(config.temperature, Some(0.7));
}

#[test]
fn generation_config_rejects_out_of_range_max_output_tokens_without_typed_override() {
    let provider_request = CompletionRequest {
        additional_params: Some(serde_json::json!({
            "generationConfig": { "maxOutputTokens": 2_147_483_648_u64 }
        })),
        ..minimal_request()
    };
    let provider_error = VertexCompletionRequest(provider_request)
        .generation_config()
        .expect_err("provider max output tokens beyond i32 must fail");
    assert!(
        provider_error
            .to_string()
            .contains("max_output_tokens exceeds Vertex AI's i32 range")
    );

    let typed_request = CompletionRequest {
        max_tokens: Some(2_147_483_648),
        ..minimal_request()
    };
    let typed_error = VertexCompletionRequest(typed_request)
        .generation_config()
        .expect_err("typed max output tokens beyond i32 must fail");
    assert!(
        typed_error
            .to_string()
            .contains("max_output_tokens exceeds Vertex AI's i32 range")
    );
}

#[test]
fn generation_config_rejects_conflicting_response_schema_forms() {
    for json_schema_key in ["responseJsonSchema", "_responseJsonSchema"] {
        let request = CompletionRequest {
            additional_params: Some(serde_json::json!({
                "generationConfig": {
                    "responseSchema": { "type": "OBJECT" },
                    (json_schema_key): { "type": "object" }
                }
            })),
            ..minimal_request()
        };

        let error = VertexCompletionRequest(request)
            .generation_config()
            .expect_err("response schema forms cannot be combined");
        assert!(matches!(error, CompletionError::RequestError(_)));
        assert!(
            error
                .to_string()
                .contains("responseSchema cannot be combined")
        );
    }
}

#[test]
fn generation_config_rejects_both_response_json_schema_aliases() {
    let request = CompletionRequest {
        additional_params: Some(serde_json::json!({
            "generationConfig": {
                "responseJsonSchema": { "type": "object" },
                "_responseJsonSchema": { "type": "object" }
            }
        })),
        ..minimal_request()
    };

    let error = VertexCompletionRequest(request)
        .generation_config()
        .expect_err("JSON schema aliases cannot be combined");
    assert!(matches!(error, CompletionError::RequestError(_)));
    assert!(
        error
            .to_string()
            .contains("responseJsonSchema cannot be combined with _responseJsonSchema")
    );
}

#[test]
fn generation_config_rejects_invalid_thinking_config() {
    let malformed_request = CompletionRequest {
        additional_params: Some(serde_json::json!({
            "generationConfig": { "thinkingConfig": { "thinkingBudget": "invalid" } }
        })),
        ..minimal_request()
    };
    assert!(
        VertexCompletionRequest(malformed_request)
            .generation_config()
            .is_err()
    );

    let conflicting_request = CompletionRequest {
        additional_params: Some(serde_json::json!({
            "generationConfig": {
                "thinkingConfig": {
                    "thinkingBudget": 1024,
                    "thinkingLevel": "high"
                }
            }
        })),
        ..minimal_request()
    };
    let error = VertexCompletionRequest(conflicting_request)
        .generation_config()
        .expect_err("mutually exclusive thinking values must fail");
    assert!(
        error
            .to_string()
            .contains("thinking_budget and thinking_level cannot both be set")
    );
}
