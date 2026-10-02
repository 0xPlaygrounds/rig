use super::*;
use crate::error::ProviderError;
use crate::message;
use serde_json::json;
use std::collections::HashMap;

/// The choice a body's `output[]` folds to, through the ONE interpreter: the
/// decoder's unary variant synthesizes the stream's events and the shared
/// fold turns them into the response — the two steps a unary reply takes.
pub(super) fn folded_choice(output: Vec<Output>) -> Vec<completion::AssistantContent> {
    let response = CompletionResponse {
        id: "resp_1".to_string(),
        object: ResponseObject::Response,
        created_at: 0,
        status: ResponseStatus::Completed,
        error: None,
        incomplete_details: None,
        instructions: None,
        max_output_tokens: None,
        model: "gpt-5-mini".to_string(),
        provider_reasoning: None,
        reasoning_metadata: None,
        reasoning_context: None,
        usage: None,
        output,
        tools: Vec::new(),
        additional_parameters: AdditionalParameters::default(),
    };
    fold_body(response).expect("the body folds").choice
}

/// A whole Responses body decoded as the OpenAI wire's unary reply.
fn fold_body(
    response: CompletionResponse,
) -> Result<completion::CompletionResponse, ProviderError> {
    let body = serde_json::to_string(&response)?;
    crate::test_utils::decode_reply(
        &openai_wire("gpt-5-mini"),
        &crate::completion::CompletionRequest::new("hello"),
        crate::wire::Mode::Unary,
        [crate::wire::WireFrame::Text(body)],
        serde_json::Value::Null,
    )
}

/// The OpenAI Responses wire, for the request-shaping assertions.
fn openai_wire(model: &str) -> wire::Responses {
    crate::providers::openai::OpenAIConfig::new("dummy-key").responses(model)
}

/// The Responses request a wire builds for a Rig request — the one
/// conversion every caller reaches, whatever opened the socket.
fn wire_request(
    wire: &wire::Responses,
    request: completion::CompletionRequest,
) -> CompletionRequest {
    wire.responses_request(request, false)
        .expect("request should convert")
}

fn test_document(id: &str, text: &str) -> crate::completion::Document {
    crate::completion::Document {
        id: id.to_string(),
        text: text.to_string(),
        additional_props: HashMap::new(),
    }
}

fn weather_tool_definition() -> completion::ToolDefinition {
    completion::ToolDefinition {
        name: crate::message::ToolName::new("get_weather").expect("tool name"),
        description: "Get the weather".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }),
    }
}

fn rig_tool_result(content: message::ToolResultContent) -> message::Message {
    message::Message::User {
        content: vec![message::UserContent::ToolResult(message::ToolResult {
            call: crate::message::CallId::from_wire("call-id"),
            name: crate::message::ToolName::new("tool".to_string()).expect("tool name"),
            content: vec![content],
        })],
    }
}

#[test]
fn mixed_user_content_preserves_order_around_tool_results() {
    let input = message::Message::User {
        content: vec![
            message::UserContent::text("before"),
            message::UserContent::tool_result(
                crate::message::CallId::from_wire("call-id"),
                crate::message::ToolName::new("tool").expect("tool name"),
                vec![message::ToolResultContent::text("tool output")],
            ),
            message::UserContent::text("after"),
        ],
    };

    let items =
        super::input_items(input, 0, &mut CustomCalls::new()).expect("input item conversion");

    assert!(matches!(
        items.as_slice(),
        [
            InputItem::Message(Message::User { content: before, .. }),
            InputItem::FunctionCallOutput(ToolResult { call_id, .. }),
            InputItem::Message(Message::User { content: after, .. }),
        ] if matches!(before.first(), Some(UserContent::InputText { text }) if text == "before")
            && call_id == "call-id"
            && matches!(after.first(), Some(UserContent::InputText { text }) if text == "after")
    ));
}

#[test]
fn tool_result_literal_text_and_structured_json_render_without_reparsing() {
    let cases = [
        (
            message::ToolResultContent::text(r#"{"status":"ok"}"#),
            r#"{"status":"ok"}"#.to_string(),
        ),
        (
            message::ToolResultContent::json(json!({ "status": "ok" })),
            r#"{"status":"ok"}"#.to_string(),
        ),
    ];

    for (content, expected) in cases {
        let input = rig_tool_result(content);

        let items =
            super::input_items(input, 0, &mut CustomCalls::new()).expect("input item conversion");
        assert!(matches!(
            items.as_slice(),
            [InputItem::FunctionCallOutput(ToolResult {
                output: ToolResultOutput::Text(output),
                ..
            })] if output == &expected
        ));
    }
}

#[test]
fn multiple_text_tool_result_blocks_preserve_order_as_rich_function_output() {
    let content = vec![
        message::ToolResultContent::text("first"),
        message::ToolResultContent::text("second"),
    ];

    let input = message::Message::User {
        content: vec![message::UserContent::ToolResult(message::ToolResult {
            call: crate::message::CallId::from_wire("call-id"),
            name: crate::message::ToolName::new("tool".to_string()).expect("tool name"),
            content,
        })],
    };

    let expected = ToolResultOutput::Content(vec![
        ToolResultOutputContent::InputText {
            text: "first".to_string(),
        },
        ToolResultOutputContent::InputText {
            text: "second".to_string(),
        },
    ]);

    let items =
        super::input_items(input, 0, &mut CustomCalls::new()).expect("input item conversion");

    match items.as_slice() {
        [InputItem::FunctionCallOutput(ToolResult { output, .. })] => {
            assert_eq!(output, &expected);
        }
        other => panic!("expected one function-call output, got {other:?}"),
    }

    let wire = serde_json::to_value(&items[0]).expect("input item should serialize");

    assert_eq!(
        wire,
        json!({
            "type": "function_call_output",
            "call_id": "call-id",
            "output": [
                {
                    "type": "input_text",
                    "text": "first"
                },
                {
                    "type": "input_text",
                    "text": "second"
                }
            ],
            "status": "completed"
        })
    );
}

#[test]
fn multiple_text_and_json_tool_result_blocks_preserve_boundaries() {
    let content = vec![
        message::ToolResultContent::text("before"),
        message::ToolResultContent::json(json!({
            "status": "ok"
        })),
        message::ToolResultContent::text("after"),
    ];

    let output =
        responses_tool_result_output(content).expect("tool-result conversion should succeed");

    assert_eq!(
        output,
        ToolResultOutput::Content(vec![
            ToolResultOutputContent::InputText {
                text: "before".to_string(),
            },
            ToolResultOutputContent::InputText {
                text: r#"{"status":"ok"}"#.to_string(),
            },
            ToolResultOutputContent::InputText {
                text: "after".to_string(),
            },
        ])
    );
}

#[test]
fn tool_result_images_and_text_preserve_order_as_rich_function_output() {
    let content = vec![
        message::ToolResultContent::text("before"),
        message::ToolResultContent::image_base64(
            "aW1hZ2U=",
            Some(message::ImageMediaType::PNG),
            None,
        ),
        message::ToolResultContent::json(json!({ "after": true })),
    ];
    let input = message::Message::User {
        content: vec![message::UserContent::ToolResult(message::ToolResult {
            call: crate::message::CallId::from_wire("call-id"),
            name: crate::message::ToolName::new("tool".to_string()).expect("tool name"),
            content,
        })],
    };

    let assert_output = |output: &ToolResultOutput| {
        assert!(matches!(
            output,
            ToolResultOutput::Content(content)
                if matches!(content.as_slice(), [
                    ToolResultOutputContent::InputText { text: before },
                    ToolResultOutputContent::InputImage { image_url, .. },
                    ToolResultOutputContent::InputText { text: after },
                ] if before == "before"
                    && image_url.as_deref() == Some("data:image/png;base64,aW1hZ2U=")
                    && after == r#"{"after":true}"#)
        ));
    };

    let items =
        super::input_items(input, 0, &mut CustomCalls::new()).expect("input item conversion");
    match items.as_slice() {
        [InputItem::FunctionCallOutput(ToolResult { output, .. })] => assert_output(output),
        other => panic!("expected one rich function output, got {other:?}"),
    }
}

#[test]
fn tool_result_file_id_image_uses_the_native_wire_field() {
    let input = rig_tool_result(message::ToolResultContent::Image(message::Image {
        data: message::DocumentSourceKind::FileId("file-image-123".to_string()),
        media_type: None,
        detail: None,
        native: None,
    }));

    let items =
        super::input_items(input, 0, &mut CustomCalls::new()).expect("input item conversion");
    let wire = serde_json::to_value(&items[0]).expect("serialize input item");
    assert_eq!(
        wire,
        json!({
            "type": "function_call_output",
            "call_id": "call-id",
            "output": [{
                "type": "input_image",
                "file_id": "file-image-123",
                "detail": "auto"
            }],
            "status": "completed"
        })
    );
}

fn weather_tool_request() -> completion::CompletionRequest {
    completion::CompletionRequest::new("what's the weather?").tools(vec![weather_tool_definition()])
}

#[test]
fn responses_tool_choice_modes_serialize_as_plain_strings() {
    for (choice, expected) in [
        (message::ToolChoice::Auto, json!("auto")),
        (message::ToolChoice::None, json!("none")),
        (message::ToolChoice::Required, json!("required")),
    ] {
        let converted = ToolChoice::try_from(choice).expect("mode should convert");
        assert_eq!(
            serde_json::to_value(&converted).expect("serialize tool choice"),
            expected
        );
    }
}

#[test]
fn responses_tool_choice_specific_single_name_serializes_as_named_function() {
    let converted = ToolChoice::try_from(message::ToolChoice::Specific {
        function_names: vec![crate::message::ToolName::new("get_weather").expect("tool name")],
    })
    .expect("single specific tool should convert");

    assert_eq!(
        serde_json::to_value(&converted).expect("serialize tool choice"),
        json!({"type": "function", "name": "get_weather"})
    );
}

#[test]
fn responses_tool_choice_specific_multiple_names_serialize_as_allowed_tools() {
    let converted = ToolChoice::try_from(message::ToolChoice::Specific {
        function_names: vec![
            crate::message::ToolName::new("add").expect("tool name"),
            crate::message::ToolName::new("subtract").expect("tool name"),
        ],
    })
    .expect("multiple specific tools should convert");

    assert_eq!(
        serde_json::to_value(&converted).expect("serialize tool choice"),
        json!({
            "type": "allowed_tools",
            "mode": "required",
            "tools": [
                {"type": "function", "name": "add"},
                {"type": "function", "name": "subtract"}
            ]
        })
    );
}

#[test]
fn responses_tool_choice_specific_empty_names_error() {
    let converted = ToolChoice::try_from(message::ToolChoice::Specific {
        function_names: vec![],
    });

    assert!(matches!(
        converted.map_err(ProviderError::from),
        Err(ProviderError::Request(error))
            if error.to_string().contains("at least one function name")
    ));
}

#[test]
fn responses_request_with_specific_tool_choice_serializes_named_function() {
    let mut request = weather_tool_request();
    request.tool_choice = Some(message::ToolChoice::Specific {
        function_names: vec![crate::message::ToolName::new("get_weather").expect("tool name")],
    });

    let request = CompletionRequest::try_from(("gpt-test".to_string(), request)).expect("convert");
    let request_json = serde_json::to_value(&request).expect("serialize request");

    assert_eq!(
        request_json.get("tool_choice"),
        Some(&json!({"type": "function", "name": "get_weather"}))
    );
}

#[test]
fn responses_function_tools_are_non_strict_by_default() {
    let tool = ResponsesToolDefinition::function(
        "get_weather",
        "Get the weather",
        weather_tool_definition().parameters,
    );

    assert!(!tool.strict);
    assert_eq!(tool.parameters["required"], json!(["location"]));
    assert!(tool.parameters.get("additionalProperties").is_none());

    // Omitted `strict` means "try strict" on the Responses API; `false` must be explicit.
    let serialized = serde_json::to_value(tool).expect("tool should serialize");
    assert_eq!(serialized.get("strict"), Some(&json!(false)));
}

#[test]
fn responses_tool_definitions_accept_nullable_strict() {
    let cases = [
        (
            json!({
                "type": "function",
                "name": "get_weather",
                "parameters": {}
            }),
            false,
        ),
        (
            json!({
                "type": "function",
                "name": "get_weather",
                "parameters": {},
                "strict": null
            }),
            false,
        ),
        (
            json!({
                "type": "function",
                "name": "get_weather",
                "parameters": {},
                "strict": false
            }),
            false,
        ),
        (
            json!({
                "type": "function",
                "name": "get_weather",
                "parameters": {},
                "strict": true
            }),
            true,
        ),
    ];

    for (value, expected) in cases {
        let tool: ResponsesToolDefinition =
            serde_json::from_value(value).expect("tool definition should deserialize");
        assert_eq!(tool.strict, expected);
    }
}

#[test]
fn responses_strict_function_tools_sanitize_schema() {
    let tool = ResponsesToolDefinition::strict_function(
        "get_weather",
        "Get the weather",
        weather_tool_definition().parameters,
    );

    assert!(tool.strict);
    assert_eq!(tool.parameters["additionalProperties"], json!(false));
    assert_eq!(tool.parameters["required"], json!(["location", "unit"]));
}

fn request_with_preamble(preamble: &str) -> completion::CompletionRequest {
    completion::CompletionRequest::from(vec![
        message::Message::system(preamble),
        message::Message::user("Hello"),
    ])
}

fn system_only_request(system_text: &str) -> completion::CompletionRequest {
    completion::CompletionRequest::new(completion::Message::system(system_text))
}

#[test]
fn responses_request_uses_top_level_instructions_for_preamble_by_default() {
    let req = CompletionRequest::try_from((
        "gpt-4o-mini".to_string(),
        request_with_preamble("You are concise."),
    ))
    .expect("request should convert");
    let serialized = serde_json::to_value(&req).expect("request should serialize");
    let input = serialized["input"]
        .as_array()
        .expect("input should be array");

    assert_eq!(serialized["instructions"], json!("You are concise."));
    assert_eq!(input.len(), 1);
    assert_eq!(input[0]["role"], "user");
}

#[test]
fn responses_request_drops_whitespace_only_preamble() {
    let req =
        CompletionRequest::try_from(("gpt-4o-mini".to_string(), request_with_preamble("  \n ")))
            .expect("request should convert");
    let serialized = serde_json::to_value(&req).expect("request should serialize");
    let input = serialized["input"]
        .as_array()
        .expect("input should be array");

    assert!(
        serialized.get("instructions").is_none(),
        "a whitespace-only preamble carries no content and is dropped"
    );
    assert_eq!(input.len(), 1);
    assert_eq!(input[0]["role"], "user");
}

#[test]
fn responses_request_lifts_system_messages_to_top_level_instructions_by_default() {
    let request = crate::completion::CompletionRequest::new("Hello")
        .preamble("System one")
        .message(completion::Message::system("System two"));

    let req = CompletionRequest::try_from(("gpt-4o-mini".to_string(), request))
        .expect("request should convert");
    let serialized = serde_json::to_value(&req).expect("request should serialize");
    let input = serialized["input"]
        .as_array()
        .expect("input should be array");

    assert_eq!(
        serialized["instructions"],
        json!("System one\n\nSystem two")
    );
    assert_eq!(input.len(), 1);
    assert_eq!(input[0]["role"], "user");
}

#[test]
fn responses_request_with_only_system_messages_keeps_them_in_input() {
    let req = CompletionRequest::try_from((
        "gpt-4o-mini".to_string(),
        system_only_request("System only"),
    ))
    .expect("request conversion should succeed");
    let serialized = serde_json::to_value(&req).expect("request should serialize");
    let input = serialized["input"]
        .as_array()
        .expect("input should be array");

    assert!(
        serialized.get("instructions").is_none(),
        "lifting a system-only history would leave input empty, so it stays in input"
    );
    assert_eq!(input.len(), 1);
    assert_eq!(input[0]["role"], "system");
    assert!(input[0].to_string().contains("System only"));
}

#[test]
fn responses_wire_can_fallback_to_system_messages_in_input() {
    let wire = openai_wire("gpt-4o-mini").with_system_instructions_as_messages();

    let req = wire_request(&wire, request_with_preamble("You are concise."));
    let serialized = serde_json::to_value(&req).expect("request should serialize");
    let input = serialized["input"]
        .as_array()
        .expect("input should be array");

    assert!(serialized.get("instructions").is_none());
    assert_eq!(input.len(), 2);
    assert_eq!(input[0]["role"], "system");
    assert!(input[0].to_string().contains("You are concise."));
    assert_eq!(input[1]["role"], "user");
}

#[test]
fn responses_wire_can_lift_all_system_messages_via_placement() {
    let wire = openai_wire("gpt-4o-mini")
        .with_system_instructions_placement(SystemInstructionsPlacement::AllInstructions);

    let request = crate::completion::CompletionRequest::new("again")
        .preamble("System one")
        .message(completion::Message::user("hi"))
        .message(completion::Message::system("Mid-conversation instruction"));

    let req = wire_request(&wire, request);
    let serialized = serde_json::to_value(&req).expect("request should serialize");
    let input = serialized["input"]
        .as_array()
        .expect("input should be array");

    assert_eq!(
        serialized["instructions"],
        json!("System one\n\nMid-conversation instruction")
    );
    assert!(
        input.iter().all(|item| item["role"] != "system"),
        "AllInstructions should leave no system items in input: {input:?}"
    );
}

#[test]
fn all_instructions_system_only_input_reports_non_system_requirement() {
    let err = CompletionRequest::try_from(ResponsesRequestParams {
        model: "gpt-4o-mini".to_string(),
        request: system_only_request("System only"),
        system_instructions_placement: SystemInstructionsPlacement::AllInstructions,
    })
    .expect_err("system-only input should fail once every item is lifted");

    assert!(
        err.to_string().contains("non-system item"),
        "error should explain that lifted system messages left input empty: {err}"
    );
}

#[test]
fn all_instructions_whitespace_only_system_input_reports_non_system_requirement() {
    let err = CompletionRequest::try_from(ResponsesRequestParams {
        model: "gpt-4o-mini".to_string(),
        request: system_only_request("   "),
        system_instructions_placement: SystemInstructionsPlacement::AllInstructions,
    })
    .expect_err("whitespace-only system input should fail once every item is lifted");

    assert!(
        err.to_string().contains("non-system item"),
        "even when lifted system text is whitespace-only (so no `instructions` field is \
             produced), the error should explain that system messages were lifted: {err}"
    );
}

#[test]
fn responses_request_conversion_keeps_tools_non_strict_by_default() {
    let req = CompletionRequest::try_from(("gpt-4o-mini".to_string(), weather_tool_request()))
        .expect("request should convert");

    let tool = &req.tools[0];
    assert!(!tool.strict);
    assert_eq!(tool.parameters["required"], json!(["location"]));
    assert!(tool.parameters.get("additionalProperties").is_none());
}

#[test]
fn responses_wire_strict_tools_opt_in_sanitizes_all_function_tools() {
    let wire =
        openai_wire("gpt-4o-mini")
            .with_strict_tools()
            .with_tool(completion::ToolDefinition {
                name: crate::message::ToolName::new("lookup").expect("tool name"),
                description: "Look something up".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {"q": {"type": "string"}}
                }),
            });

    let mut request = weather_tool_request();
    request.additional_params = Some(json!({
        "tools": [{
            "type": "function",
            "name": "extra",
            "description": "An additional_params tool",
            "parameters": {"type": "object", "properties": {"x": {"type": "string"}}}
        }]
    }));

    let req = wire_request(&wire, request);

    assert_eq!(req.tools.len(), 3);
    for tool in &req.tools {
        assert!(tool.strict, "{} should be strict", tool.name);
        assert_eq!(tool.parameters["additionalProperties"], json!(false));
    }
}

#[test]
fn responses_wire_default_preserves_all_function_tools_as_constructed() {
    let wire = openai_wire("gpt-4o-mini").with_tool(weather_tool_definition());

    let mut request = weather_tool_request();
    request.additional_params = Some(json!({
        "tools": [{
            "type": "function",
            "name": "extra",
            "description": "An additional_params tool",
            "parameters": {"type": "object", "properties": {"x": {"type": "string"}}}
        }]
    }));

    let req = wire_request(&wire, request);

    assert_eq!(req.tools.len(), 3);
    for tool in &req.tools {
        assert!(!tool.strict, "{} should not be strict", tool.name);
        assert!(tool.parameters.get("additionalProperties").is_none());
    }
}

#[test]
fn responses_explicit_strict_tool_stays_strict_on_a_default_wire() {
    let wire = openai_wire("gpt-4o-mini").with_tool(ResponsesToolDefinition::strict_function(
        "lookup",
        "Look something up",
        json!({"type": "object", "properties": {"q": {"type": "string"}}}),
    ));

    let req = wire_request(&wire, weather_tool_request());

    assert!(!req.tools[0].strict);
    assert!(req.tools[1].strict);
    assert_eq!(
        req.tools[1].parameters["additionalProperties"],
        json!(false)
    );
}

fn response_with_service_tier(service_tier: &str) -> Value {
    json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "gpt-5.4",
        "output": [],
        "service_tier": service_tier,
    })
}

#[test]
fn completion_response_deserializes_standard_service_tier() {
    let response: CompletionResponse =
        serde_json::from_value(response_with_service_tier("standard"))
            .expect("response should deserialize");

    assert!(matches!(
        response.additional_parameters.service_tier,
        Some(OpenAIServiceTier::Standard)
    ));
}

#[test]
fn completion_response_deserializes_priority_service_tier() {
    let response: CompletionResponse =
        serde_json::from_value(response_with_service_tier("priority"))
            .expect("response should deserialize");

    assert!(matches!(
        response.additional_parameters.service_tier,
        Some(OpenAIServiceTier::Priority)
    ));
}

#[test]
fn completion_response_preserves_unknown_service_tier() {
    let response: CompletionResponse =
        serde_json::from_value(response_with_service_tier("provider_experimental"))
            .expect("response should deserialize");

    let Some(OpenAIServiceTier::Other(service_tier)) = response.additional_parameters.service_tier
    else {
        panic!("expected provider-specific service tier");
    };

    assert_eq!(service_tier, "provider_experimental");
}

/// A response whose echoed `top_p` is object-shaped, as MiniMax-style
/// Responses endpoints emit it, carrying a valid tool call (rig#2483).
fn response_with_object_top_p() -> Value {
    json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "MiniMax-M2",
        "top_p": { "value": 0.95 },
        "service_tier": "default",
        "output": [{
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": "{\"location\":\"Paris\"}",
            "status": "completed"
        }],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15
        }
    })
}

/// One optional metadata field disagreeing with its Rust type must never
/// discard the response: the tool call and usage survive, `top_p` reads
/// back as `None`, and sibling metadata still decodes (rig#2483).
#[test]
fn completion_response_tolerates_object_shaped_top_p() {
    let response: CompletionResponse = serde_json::from_value(response_with_object_top_p())
        .expect("an object-shaped top_p must not fail the response");

    assert_eq!(response.additional_parameters.top_p, None);
    assert!(matches!(
        response.additional_parameters.service_tier,
        Some(OpenAIServiceTier::Default)
    ));
    assert!(
        matches!(response.output.first(), Some(Output::FunctionCall(call)) if call.name == "get_weather"),
        "the tool call must survive metadata decode failures: {:?}",
        response.output
    );
    assert_eq!(response.usage.map(|usage| usage.total_tokens), Some(15));
}

/// Numeric `top_p` still decodes — including under
/// `serde_json/arbitrary_precision`, where a `#[serde(flatten)]` into `f64`
/// used to reject it because buffered numbers arrive as an internal map
/// (rig#2493). Run this module with and without that feature.
#[test]
fn completion_response_decodes_numeric_top_p() {
    let mut body = response_with_object_top_p();
    body["top_p"] = json!(0.95);
    let response: CompletionResponse =
        serde_json::from_str(&body.to_string()).expect("numeric top_p must decode");

    assert_eq!(response.additional_parameters.top_p, Some(0.95));
    assert!(matches!(
        response.output.first(),
        Some(Output::FunctionCall(_))
    ));
}

/// The metadata projection is per key: a bad `top_p` does not take
/// `service_tier` or `store` down with it, and vice versa.
#[test]
fn completion_response_drops_only_the_mistyped_metadata_key() {
    let mut body = response_with_object_top_p();
    body["store"] = json!("not-a-bool");
    body["top_p"] = json!(0.5);
    body["prompt_cache_key"] = json!("cache-1");
    let response: CompletionResponse =
        serde_json::from_value(body).expect("a mistyped store must not fail the response");

    assert_eq!(response.additional_parameters.store, None);
    assert_eq!(response.additional_parameters.top_p, Some(0.5));
    assert_eq!(
        response.additional_parameters.prompt_cache_key.as_deref(),
        Some("cache-1")
    );
}

/// The serializer still emits the echoed metadata it decoded, so a well-formed
/// OpenAI body round-trips on every field the wire serializer writes.
#[test]
fn completion_response_round_trips_echoed_metadata() {
    let mut body = response_with_object_top_p();
    body["top_p"] = json!(1.0);
    body["store"] = json!(true);
    body["metadata"] = json!({ "k": "v" });
    let response: CompletionResponse =
        serde_json::from_value(body.clone()).expect("response should deserialize");
    let serialized = serde_json::to_value(&response).expect("response should serialize");

    for key in [
        "top_p",
        "store",
        "metadata",
        "service_tier",
        "usage",
        "output",
        "model",
    ] {
        assert_eq!(serialized[key], body[key], "field {key} must round-trip");
    }
}

#[test]
fn responses_request_keeps_documents_after_lifted_system_messages() {
    let request = crate::completion::CompletionRequest::new("Prompt")
        .message(completion::Message::system("System prompt"))
        .message(completion::Message::user("Earlier user turn"))
        .message(completion::Message::assistant("Earlier assistant turn"))
        .document(test_document("doc1", "Document text."));

    let responses_request = CompletionRequest::try_from(("gpt-4o-mini".to_string(), request))
        .expect("request conversion should succeed");

    let serialized = serde_json::to_value(&responses_request).expect("request should serialize");
    let input = serialized["input"]
        .as_array()
        .expect("input should be an array");

    assert_eq!(serialized["instructions"], json!("System prompt"));
    assert_eq!(input.len(), 4);
    assert_eq!(input[0]["role"], "user");
    assert!(
        input[0].to_string().contains("<file id: doc1>"),
        "document input should be first after system instructions are lifted: {input:?}"
    );
    assert_eq!(input[1]["role"], "user");
    assert!(
        input[1].to_string().contains("Earlier user turn"),
        "prior user history should follow document input: {input:?}"
    );
    assert_eq!(input[2]["role"], "assistant");
    assert!(
        input[2].to_string().contains("Earlier assistant turn"),
        "prior assistant history should follow prior user history: {input:?}"
    );
    assert_eq!(input[3]["role"], "user");
    assert!(
        input[3].to_string().contains("Prompt"),
        "prompt should remain last: {input:?}"
    );
}

#[test]
fn responses_direct_request_keeps_mid_conversation_system_messages_in_input() {
    let request = crate::completion::CompletionRequest::from(vec![
        completion::Message::system("System prompt"),
        completion::Message::assistant("Earlier assistant turn"),
        completion::Message::system("Mid-conversation instruction"),
        completion::Message::user("Prompt"),
    ])
    .documents(vec![test_document("doc1", "Document text.")]);

    let responses_request = CompletionRequest::try_from(("gpt-4o-mini".to_string(), request))
        .expect("request conversion should succeed");

    let serialized = serde_json::to_value(&responses_request).expect("request should serialize");
    let input = serialized["input"]
        .as_array()
        .expect("input should be an array");

    assert_eq!(
        serialized["instructions"],
        json!("System prompt"),
        "only the leading run of system messages should be lifted"
    );
    assert_eq!(input.len(), 4);
    assert_eq!(input[0]["role"], "user");
    assert!(
        input[0].to_string().contains("<file id: doc1>"),
        "document input should follow lifted system instructions: {input:?}"
    );
    assert_eq!(input[1]["role"], "assistant");
    assert_eq!(input[2]["role"], "system");
    assert!(
        input[2]
            .to_string()
            .contains("Mid-conversation instruction"),
        "mid-conversation system messages should keep their position: {input:?}"
    );
    assert_eq!(input[3]["role"], "user");
    assert_eq!(
        input
            .iter()
            .filter(|message| message.to_string().contains("<file id: doc1>"))
            .count(),
        1,
        "document input should appear exactly once: {input:?}"
    );
}

#[test]
fn service_tier_serializes_expected_strings() {
    let cases = [
        (OpenAIServiceTier::Auto, "auto"),
        (OpenAIServiceTier::Default, "default"),
        (OpenAIServiceTier::Flex, "flex"),
        (OpenAIServiceTier::Priority, "priority"),
        (OpenAIServiceTier::Standard, "standard"),
    ];

    for (service_tier, expected) in cases {
        assert_eq!(
            serde_json::to_value(service_tier).expect("service tier should serialize"),
            json!(expected)
        );
    }

    assert_eq!(
        serde_json::to_value(OpenAIServiceTier::Other(
            "provider_experimental".to_string()
        ))
        .expect("provider-specific service tier should serialize"),
        json!("provider_experimental")
    );
}

#[test]
fn responses_usage_token_usage_preserves_reasoning_tokens() {
    let usage = ResponsesUsage {
        input_tokens: 100,
        input_tokens_details: Some(InputTokensDetails {
            cached_tokens: 25,
            cache_write_tokens: None,
        }),
        output_tokens: 50,
        output_tokens_details: Some(OutputTokensDetails {
            reasoning_tokens: 15,
        }),
        total_tokens: 150,
    };

    let token_usage = crate::completion::Usage::from(&usage);

    assert_eq!(token_usage.input_tokens, Some(100));
    assert_eq!(token_usage.cached_input_tokens, Some(25));
    assert_eq!(token_usage.output_tokens, Some(50));
    assert_eq!(token_usage.reasoning_tokens, Some(15));
    assert_eq!(token_usage.total_tokens, Some(150));
}

#[test]
fn responses_usage_deserializes_without_output_token_details() {
    let usage: ResponsesUsage = serde_json::from_value(json!({
        "input_tokens": 100,
        "input_tokens_details": {
            "cached_tokens": 25
        },
        "output_tokens": 50,
        "total_tokens": 150
    }))
    .expect("usage should deserialize when output token details are omitted");

    assert!(usage.output_tokens_details.is_none());

    let token_usage = crate::completion::Usage::from(&usage);

    assert_eq!(token_usage.input_tokens, Some(100));
    assert_eq!(token_usage.cached_input_tokens, Some(25));
    assert_eq!(token_usage.output_tokens, Some(50));
    assert_eq!(token_usage.reasoning_tokens, None);
    assert_eq!(token_usage.total_tokens, Some(150));
}

#[test]
fn completion_response_accepts_top_level_reasoning_string() {
    let response: CompletionResponse = serde_json::from_value(json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "Qwen/Qwen3-4B",
        "reasoning": "thinking through the answer",
        "usage": {
            "input_tokens": 1,
            "output_tokens": 2,
            "total_tokens": 3
        },
        "output": [{
            "type": "message",
            "id": "msg_123",
            "status": "completed",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "annotations": [],
                "text": "done"
            }]
        }],
        "tools": []
    }))
    .expect("mistral.rs-style reasoning string should deserialize");

    assert_eq!(
        response.provider_reasoning.as_deref(),
        Some("thinking through the answer")
    );
    assert_eq!(response.reasoning_metadata, None);
    assert_eq!(response.reasoning_context, None);
    assert_eq!(
        serde_json::to_value(&response).expect("response should serialize")["reasoning"],
        json!("thinking through the answer")
    );

    let completion: completion::CompletionResponse =
        fold_body(response).expect("response should convert");
    let items = completion.choice.iter().collect::<Vec<_>>();
    assert!(matches!(
        items[0],
        completion::AssistantContent::Reasoning(_)
    ));
    assert!(matches!(items[1], completion::AssistantContent::Text(_)));
}

#[test]
fn completion_response_accepts_null_metadata() {
    let response: CompletionResponse = serde_json::from_value(json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "openai-compatible-model",
        "metadata": null,
        "output": [{
            "type": "message",
            "id": "msg_123",
            "status": "completed",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "annotations": [],
                "text": "done"
            }]
        }],
        "tools": []
    }))
    .expect("response with null metadata should deserialize");

    assert!(response.additional_parameters.metadata.is_empty());
}

#[test]
fn completion_response_accepts_reasoning_only_response() {
    let response: CompletionResponse = serde_json::from_value(json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "Qwen/Qwen3-4B",
        "reasoning": "thinking only",
        "usage": {
            "input_tokens": 1,
            "output_tokens": 2,
            "total_tokens": 3
        },
        "output": [],
        "tools": []
    }))
    .expect("reasoning-only response should deserialize");

    let completion: completion::CompletionResponse =
        fold_body(response).expect("reasoning-only response should convert");
    let items = completion.choice.iter().collect::<Vec<_>>();

    assert_eq!(items.len(), 1);
    assert!(matches!(
        items[0],
        completion::AssistantContent::Reasoning(_)
    ));
}

#[test]
fn truncated_incomplete_response_surfaces_length_not_an_error() {
    // A truncated `function_call` whose arguments never parsed drops its
    // item by the documented truncation policy, so the choice can be
    // rig-induced-empty. On `status: incomplete` the finish reason is the
    // diagnostic the caller needs — the emptiness guard must not eat it,
    // which is exactly how the streaming path already behaves.
    let response: CompletionResponse = serde_json::from_value(json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 0,
        "status": "incomplete",
        "incomplete_details": { "reason": "max_output_tokens" },
        "model": "gpt-test",
        "output": [],
        "tools": []
    }))
    .expect("incomplete response shape should deserialize");

    let completion =
        fold_body(response).expect("truncated incomplete response must not be an error");

    assert!(completion.choice.is_empty());
    assert_eq!(
        completion.finish_reason(),
        Some(completion::FinishReason::Length)
    );
}

fn incomplete_because(reason: &str) -> IncompleteDetailsReason {
    IncompleteDetailsReason {
        reason: reason.to_string(),
    }
}

#[test]
fn finish_reason_maps_every_documented_terminal_state() {
    assert_eq!(
        map_finish_reason(&ResponseStatus::Completed, None),
        Some(completion::FinishReason::Stop)
    );
    assert_eq!(
        map_finish_reason(
            &ResponseStatus::Incomplete,
            Some(&incomplete_because("max_output_tokens"))
        ),
        Some(completion::FinishReason::Length)
    );
    assert_eq!(
        map_finish_reason(
            &ResponseStatus::Incomplete,
            Some(&incomplete_because("content_filter"))
        ),
        Some(completion::FinishReason::ContentFilter)
    );
    // `incomplete_details` on a completed turn is not a termination reason.
    assert_eq!(
        map_finish_reason(
            &ResponseStatus::Completed,
            Some(&incomplete_because("noise"))
        ),
        Some(completion::FinishReason::Stop)
    );
    // In-flight statuses are not terminations at all.
    assert_eq!(map_finish_reason(&ResponseStatus::InProgress, None), None);
    assert_eq!(map_finish_reason(&ResponseStatus::Queued, None), None);
}

#[test]
fn finish_reason_preserves_unknown_values_verbatim() {
    // A reason OpenAI adds later must survive in OpenAI's own spelling
    // rather than being smoothed into a natural stop.
    assert_eq!(
        map_finish_reason(
            &ResponseStatus::Incomplete,
            Some(&incomplete_because("MAX_TOOL_CALLS"))
        ),
        Some(completion::FinishReason::Other(
            "MAX_TOOL_CALLS".to_string()
        ))
    );
    // So must a terminal status with no normalized counterpart, and an
    // `incomplete` that states no reason.
    assert_eq!(
        map_finish_reason(&ResponseStatus::Failed, None),
        Some(completion::FinishReason::Other("failed".to_string()))
    );
    assert_eq!(
        map_finish_reason(&ResponseStatus::Cancelled, None),
        Some(completion::FinishReason::Other("cancelled".to_string()))
    );
    let status: ResponseStatus = serde_json::from_str(r#""throttled""#)
        .expect("an unknown provider status should deserialize");
    assert_eq!(status, ResponseStatus::Other("throttled".to_string()));
    assert_eq!(
        map_finish_reason(&status, None),
        Some(completion::FinishReason::Other("throttled".to_string()))
    );
    assert_eq!(
        serde_json::to_string(&status).expect("unknown status should serialize"),
        r#""throttled""#
    );
    assert_eq!(
        map_finish_reason(&ResponseStatus::Incomplete, None),
        Some(completion::FinishReason::Other("incomplete".to_string()))
    );
    assert_eq!(
        map_finish_reason(&ResponseStatus::Incomplete, Some(&incomplete_because(""))),
        Some(completion::FinishReason::Other("incomplete".to_string()))
    );
}

#[test]
fn completion_response_completed_with_tool_call_reports_tool_calls() {
    let response: CompletionResponse = serde_json::from_value(json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "gpt-5.4",
        "output": [{
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": "{\"city\":\"London\"}",
            "status": "completed"
        }],
        "tools": []
    }))
    .expect("response should deserialize");

    let completion: completion::CompletionResponse =
        fold_body(response).expect("response should convert");

    // `completed` is reconciled up to `ToolCalls` because the turn carried
    // a function call.
    assert_eq!(
        completion.finish_reason(),
        Some(completion::FinishReason::ToolCalls)
    );
}

#[test]
fn completion_response_incomplete_reports_the_truncation_reason() {
    let response: CompletionResponse = serde_json::from_value(json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 0,
        "status": "incomplete",
        "incomplete_details": { "reason": "max_output_tokens" },
        "model": "gpt-5.4",
        "output": [{
            "type": "message",
            "id": "msg_456",
            "status": "incomplete",
            "role": "assistant",
            "content": [{ "type": "output_text", "annotations": [], "text": "half an ans" }]
        }],
        "tools": []
    }))
    .expect("response should deserialize");

    let completion: completion::CompletionResponse =
        fold_body(response).expect("response should convert");

    assert_eq!(
        completion.finish_reason(),
        Some(completion::FinishReason::Length)
    );
}

#[test]
fn completion_response_preserves_context_without_treating_config_as_text() {
    let response: CompletionResponse = serde_json::from_value(json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "Qwen/Qwen3-4B",
        "reasoning": {
            "context": "all_turns",
            "effort": "high",
            "mode": "standard",
            "summary": null
        },
        "output": [{
            "type": "message",
            "id": "msg_123",
            "status": "completed",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "annotations": [],
                "text": "done"
            }]
        }],
        "tools": []
    }))
    .expect("object-shaped reasoning should be tolerated");

    assert!(response.provider_reasoning.is_none());
    assert_eq!(response.reasoning_context.as_deref(), Some("all_turns"));
    assert_eq!(
        response.reasoning_metadata.as_ref(),
        json!({
            "context": "all_turns",
            "effort": "high",
            "mode": "standard",
            "summary": null
        })
        .as_object()
    );
    assert_eq!(
        serde_json::to_value(&response).expect("response should serialize")["reasoning"],
        json!({
            "context": "all_turns",
            "effort": "high",
            "mode": "standard",
            "summary": null
        })
    );

    let completion: completion::CompletionResponse =
        fold_body(response).expect("response should convert");
    let items = completion.choice.iter().collect::<Vec<_>>();
    assert_eq!(items.len(), 1);
    assert!(matches!(items[0], completion::AssistantContent::Text(_)));
}

#[test]
fn completion_response_preserves_unknown_reasoning_metadata_and_nulls() {
    let metadata = json!({
        "context": "future_context",
        "effort": "ultra",
        "summary": null,
        "future_control": { "depth": 3 }
    });
    let response: CompletionResponse = serde_json::from_value(json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "gpt-future",
        "reasoning": metadata,
        "output": [],
        "tools": []
    }))
    .expect("unknown reasoning metadata should deserialize");

    assert_eq!(
        response.reasoning_context.as_deref(),
        Some("future_context")
    );
    assert_eq!(response.reasoning_metadata.as_ref(), metadata.as_object());
    assert_eq!(
        serde_json::to_value(&response).expect("response should serialize")["reasoning"],
        metadata
    );
}

#[test]
fn completion_response_ignores_unsupported_reasoning_shapes() {
    for reasoning in [Value::Null, json!(["unexpected"]), json!(42), json!(true)] {
        let response: CompletionResponse = serde_json::from_value(json!({
            "id": "resp_123",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": "openai-compatible-model",
            "reasoning": reasoning,
            "output": [],
            "tools": []
        }))
        .expect("unsupported reasoning shapes should remain non-fatal");

        assert_eq!(response.provider_reasoning, None);
        assert_eq!(response.reasoning_metadata, None);
        assert_eq!(response.reasoning_context, None);
        let serialized = serde_json::to_value(&response).expect("response should serialize");
        assert!(
            !serialized
                .as_object()
                .expect("response should serialize as an object")
                .contains_key("reasoning")
        );
    }
}

#[test]
fn completion_response_reasoning_serialization_precedence_is_stable() {
    let mut response: CompletionResponse = serde_json::from_value(json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "gpt-5.6",
        "reasoning": { "context": "all_turns", "effort": "max" },
        "output": [],
        "tools": []
    }))
    .expect("reasoning metadata should deserialize");

    response.reasoning_context = Some("current_turn".to_owned());
    response.additional_parameters.reasoning =
        Some(Reasoning::new().with_effort(ReasoningEffort::Low));
    let serialized = serde_json::to_string(&response).expect("response should serialize");
    assert_eq!(serialized.matches("\"reasoning\":").count(), 1);
    assert_eq!(
        serde_json::to_value(&response).expect("response should serialize")["reasoning"],
        json!({ "context": "all_turns", "effort": "max" })
    );

    let metadata = response.reasoning_metadata.take();
    assert_eq!(
        serde_json::to_value(&response).expect("response should serialize")["reasoning"],
        json!({ "context": "current_turn" })
    );

    response.reasoning_metadata = metadata;
    response.provider_reasoning = Some("compatible-provider text".to_owned());
    assert_eq!(
        serde_json::to_value(&response).expect("response should serialize")["reasoning"],
        json!("compatible-provider text")
    );
}

fn request_with_reasoning_params(reasoning: Value) -> CompletionRequest {
    let mut request = request_with_preamble("You are concise.");
    request.additional_params = Some(json!({ "reasoning": reasoning }));

    CompletionRequest::try_from(("gpt-5.6".to_string(), request))
        .expect("request with reasoning params should convert")
}

#[test]
fn reasoning_effort_max_survives_request_conversion() {
    let request = request_with_reasoning_params(json!({ "effort": "max" }));
    let serialized = serde_json::to_value(&request).expect("request should serialize");

    assert_eq!(serialized["reasoning"], json!({ "effort": "max" }));
}

#[test]
fn reasoning_mode_pro_composes_with_independent_effort() {
    let request = request_with_reasoning_params(json!({ "effort": "high", "mode": "pro" }));
    let serialized = serde_json::to_value(&request).expect("request should serialize");

    assert_eq!(
        serialized["reasoning"],
        json!({ "effort": "high", "mode": "pro" })
    );
}

#[test]
fn reasoning_context_values_survive_request_conversion() {
    for (context, wire_value) in [
        (ReasoningContext::Auto, "auto"),
        (ReasoningContext::AllTurns, "all_turns"),
        (ReasoningContext::CurrentTurn, "current_turn"),
    ] {
        let typed = serde_json::to_value(Reasoning::new().with_context(context))
            .expect("typed reasoning should serialize");
        assert_eq!(typed, json!({ "context": wire_value }));

        let request = request_with_reasoning_params(json!({ "context": wire_value }));
        let serialized = serde_json::to_value(&request).expect("request should serialize");
        assert_eq!(serialized["reasoning"], json!({ "context": wire_value }));
    }
}

#[test]
fn reasoning_omits_unset_optional_fields() {
    let reasoning = serde_json::to_value(Reasoning::new().with_mode(ReasoningMode::Pro))
        .expect("reasoning should serialize");

    assert_eq!(reasoning, json!({ "mode": "pro" }));

    let reasoning = serde_json::to_value(
        Reasoning::new()
            .with_effort(ReasoningEffort::Max)
            .with_mode(ReasoningMode::Pro)
            .with_context(ReasoningContext::CurrentTurn)
            .with_summary_level(ReasoningSummaryLevel::Detailed),
    )
    .expect("reasoning should serialize");

    assert_eq!(
        reasoning,
        json!({
            "effort": "max",
            "mode": "pro",
            "context": "current_turn",
            "summary": "detailed"
        })
    );
}

#[test]
fn completion_response_does_not_duplicate_structured_reasoning() {
    let response: CompletionResponse = serde_json::from_value(json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "gpt-5.4",
        "reasoning": "provider top-level text",
        "output": [{
            "type": "reasoning",
            "id": "rs_123",
            "summary": [{
                "type": "summary_text",
                "text": "structured summary"
            }]
        }, {
            "type": "message",
            "id": "msg_123",
            "status": "completed",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "annotations": [],
                "text": "done"
            }]
        }],
        "tools": []
    }))
    .expect("response should deserialize");

    let completion: completion::CompletionResponse =
        fold_body(response).expect("response should convert");
    let reasoning_count = completion
        .choice
        .iter()
        .filter(|item| matches!(item, completion::AssistantContent::Reasoning(_)))
        .count();

    assert_eq!(reasoning_count, 1);
}

#[test]
fn responses_usage_add_preserves_rhs_details_when_lhs_details_are_absent() {
    let lhs = ResponsesUsage {
        input_tokens: 10,
        input_tokens_details: None,
        output_tokens: 20,
        output_tokens_details: None,
        total_tokens: 30,
    };
    let rhs = ResponsesUsage {
        input_tokens: 3,
        input_tokens_details: Some(InputTokensDetails {
            cached_tokens: 2,
            cache_write_tokens: None,
        }),
        output_tokens: 5,
        output_tokens_details: Some(OutputTokensDetails {
            reasoning_tokens: 4,
        }),
        total_tokens: 8,
    };

    let usage = lhs + rhs;
    let token_usage = crate::completion::Usage::from(&usage);

    assert_eq!(token_usage.input_tokens, Some(13));
    assert_eq!(token_usage.cached_input_tokens, Some(2));
    assert_eq!(token_usage.output_tokens, Some(25));
    assert_eq!(token_usage.reasoning_tokens, Some(4));
    assert_eq!(token_usage.total_tokens, Some(38));
}

#[test]
fn file_id_document_serializes_as_input_item_content() {
    let message = completion::Message::User {
        content: vec![message::UserContent::Document(message::Document {
            data: DocumentSourceKind::FileId("file_abc".to_string()),
            media_type: None,
            additional_params: None,
        })],
    };

    let converted =
        super::input_items(message, 0, &mut CustomCalls::new()).expect("conversion should succeed");
    let json = serde_json::to_value(&converted[0]).expect("serialize input item");

    assert_eq!(json["type"], "message");
    assert_eq!(json["role"], "user");
    assert_eq!(json["content"][0]["type"], "input_file");
    assert_eq!(json["content"][0]["file_id"], "file_abc");
    assert!(json["content"][0].get("file_data").is_none());
    assert!(json["content"][0].get("file_url").is_none());
}

#[tokio::test]
async fn responses_completion_http_non_success_preserves_status_and_body() {
    use crate::test_utils::RecordingHttpClient;

    let body = r#"{"error":{"message":"bad image","type":"invalid_request_error","code":"invalid_value"}}"#;
    let http_client = RecordingHttpClient::with_error_response(http::StatusCode::BAD_REQUEST, body);
    let model = crate::driver::Model::new(openai_wire("gpt-4o-mini"), http_client);
    let request = crate::completion::CompletionRequest::new("hello");

    let error = model
        .call(request)
        .await
        .expect_err("completion should fail with non-success status");

    // rig#2314: a provider with a request-id contract preserves its
    // non-success responses as ProviderResponse, so the transport id has
    // a home on the error; this mock sent no header, so the id is None.
    assert!(matches!(error, ProviderError::ProviderResponse(_)));
    assert_eq!(error.provider_request_id(), None);
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::BAD_REQUEST)
    );
    assert_eq!(error.provider_response_body(), Some(body));
    let json = error
        .provider_response_json()
        .expect("raw body should be valid JSON")
        .expect("parsed JSON should be present");
    assert_eq!(json["error"]["code"], "invalid_value");
}

#[test]
fn output_unknown_preserves_hosted_tool_payload() {
    let item = json!({
        "type": "web_search_call",
        "id": "ws_001",
        "status": "completed",
        "action": { "type": "search", "queries": ["rig framework"] },
    });

    let output: Output =
        serde_json::from_value(item.clone()).expect("unknown output should deserialize");

    let Output::Unknown(value) = output else {
        panic!("expected Output::Unknown for an unmodeled item type");
    };
    assert_eq!(value, item);
}

#[test]
fn output_unknown_round_trips_value_equal() {
    let item = json!({
        "type": "file_search_call",
        "id": "fs_007",
        "status": "in_progress",
        "queries": ["lifecycle"],
    });

    let output: Output =
        serde_json::from_value(item.clone()).expect("unknown output should deserialize");
    let serialized = serde_json::to_value(&output).expect("unknown output should serialize");

    assert_eq!(serialized, item);
}

#[test]
fn output_known_variant_with_bad_body_errors() {
    // A recognized `type` tag with a malformed body must still error rather
    // than silently degrading to `Output::Unknown`.
    let malformed = json!({
        "type": "function_call",
        "id": "call_1",
        // missing `arguments`, `call_id`, `name`
    });

    let result: Result<Output, _> = serde_json::from_value(malformed);
    assert!(result.is_err());
}

#[test]
fn completion_response_with_unknown_output_keeps_usage() {
    // Guards the original reason the catch-all exists: an unknown item must
    // not break decoding of the whole response or drop token usage.
    let response = json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "gpt-5.4",
        "output": [
            {
                "type": "web_search_call",
                "id": "ws_001",
                "status": "completed",
            },
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [ { "type": "output_text", "text": "hi", "annotations": [] } ],
            },
        ],
        "usage": {
            "input_tokens": 100,
            "input_tokens_details": { "cached_tokens": 25 },
            "output_tokens": 50,
            "output_tokens_details": { "reasoning_tokens": 15 },
            "total_tokens": 150,
        },
    });

    let response: CompletionResponse =
        serde_json::from_value(response).expect("response should deserialize");

    assert!(matches!(response.output.first(), Some(Output::Unknown(_))));
    let usage = response.usage.expect("usage should be present");
    assert_eq!(usage.total_tokens, 150);
}

#[test]
fn output_known_variant_round_trips_value_equal() {
    // The hand-written Serialize must reproduce the modeled wire shape, so a
    // decoded known item re-serializes value-equal to what it came from
    // (guards the `function_call` arm, including its stringified `arguments`).
    // The item ID uses the provider-native `fc_` prefix; other IDs are
    // intentionally dropped on serialization (see `OutputFunctionCall::id`).
    let item = json!({
        "type": "function_call",
        "id": "fc_1",
        "arguments": "{}",
        "call_id": "c1",
        "name": "search",
        "status": "completed",
    });

    let output: Output =
        serde_json::from_value(item.clone()).expect("known output should deserialize");
    assert!(matches!(output, Output::FunctionCall(_)));

    let serialized = serde_json::to_value(&output).expect("known output should serialize");
    assert_eq!(serialized, item);
}

#[test]
fn output_reasoning_round_trips_value_equal() {
    // Highest-value parity guard: the `Reasoning` struct variant threads its
    // fields by hand in *both* directions. Populated `encrypted_content` /
    // `status` (the `#[serde(default)]` optionals) must survive
    // serialize -> deserialize unchanged — catching a dropped field or a
    // forgotten `reasoning` dispatch arm (which would degrade to `Unknown`).
    let original = Output::Reasoning {
        id: "reasoning_1".to_string(),
        summary: vec![ReasoningSummary::SummaryText {
            text: "weighing options".to_string(),
        }],
        content: vec!["private reasoning".to_string()],
        encrypted_content: Some("ENCRYPTED".to_string()),
        signature: None,
        status: Some(ToolStatus::Completed),
    };

    let value = serde_json::to_value(&original).expect("reasoning should serialize");
    let round_tripped: Output =
        serde_json::from_value(value).expect("reasoning should deserialize");

    assert_eq!(round_tripped, original);
}

#[test]
fn output_reasoning_none_optionals_serialize_as_explicit_null() {
    // Wire-anchored complement to the round-trip test: with `None`
    // optionals, the keys must still be emitted as explicit `null` (the
    // derived behavior this hand-written serde replaced has no
    // `skip_serializing_if`). Guards against a future refactor silently
    // dropping the keys and changing the wire shape.
    let value = serde_json::to_value(Output::Reasoning {
        id: "reasoning_1".to_string(),
        summary: vec![],
        content: vec![],
        encrypted_content: None,
        signature: None,
        status: None,
    })
    .expect("reasoning should serialize");

    assert_eq!(value["type"], "reasoning");
    assert_eq!(value["encrypted_content"], Value::Null);
    assert_eq!(value["status"], Value::Null);
    assert!(value.get("encrypted_content").is_some());
    assert!(value.get("status").is_some());
}

#[test]
fn output_message_round_trips_value_equal() {
    // Wire-anchored serialize check for the `message` arm (only
    // `function_call` was anchored): a decoded message item re-serializes
    // value-equal to the input, tag included.
    let item = json!({
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [ { "type": "output_text", "text": "hello", "annotations": [] } ],
    });

    let output: Output =
        serde_json::from_value(item.clone()).expect("message item should deserialize");
    assert!(matches!(output, Output::Message(_)));

    let serialized = serde_json::to_value(&output).expect("message should serialize");
    assert_eq!(serialized, item);
}

#[test]
fn each_known_tag_decodes_to_its_modeled_variant() {
    // Guards every modeled dispatch arm: a well-formed item for each known
    // `type` must decode to its specific variant, never to `Unknown`. Adding
    // an `Output` variant without a matching deserialize arm fails here
    // instead of silently routing real items to `Unknown`.
    let message: Output = serde_json::from_value(json!({
        "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
        "content": [ { "type": "output_text", "text": "hi", "annotations": [] } ],
    }))
    .expect("message item should decode");
    assert!(matches!(message, Output::Message(_)));

    let function_call: Output = serde_json::from_value(json!({
        "type": "function_call", "id": "call_1", "arguments": "{}",
        "call_id": "c1", "name": "f", "status": "completed",
    }))
    .expect("function_call item should decode");
    assert!(matches!(function_call, Output::FunctionCall(_)));

    let reasoning: Output =
        serde_json::from_value(json!({ "type": "reasoning", "id": "r1", "summary": [] }))
            .expect("reasoning item should decode");
    assert!(matches!(reasoning, Output::Reasoning { .. }));
}

#[test]
fn output_without_usable_type_tag_decodes_to_unknown() {
    // An absent or non-string `type` is itself unmodeled, so it is captured
    // verbatim as `Unknown` rather than erroring.
    for item in [
        json!({ "id": "x", "note": "no type field" }),
        json!({ "type": 7, "id": "x" }),
    ] {
        let output: Output =
            serde_json::from_value(item.clone()).expect("should decode to Unknown");
        assert_eq!(output, Output::Unknown(item));
    }
}

// Regression tests for issue #1429: `file_url` and `filename` are mutually
// exclusive on OpenAI's Responses API (400 `mutually_exclusive_parameters`),
// so URL-backed PDFs must not carry the hardcoded `filename`. These tests
// cover the `TryFrom<crate::completion::Message> for Vec<InputItem>` path
// that `Model::call` requests actually go through.
//
// See <https://platform.openai.com/docs/guides/pdf-files> for the
// `input_file` content part and its `file_url` / `file_data` / `file_id`
// input variants.

const PDF_URL: &str = "https://example.com/resume.pdf";

fn url_pdf_message() -> message::Message {
    message::Message::User {
        content: vec![message::UserContent::document_url(
            PDF_URL,
            Some(message::DocumentMediaType::PDF),
        )],
    }
}

/// Recursively collect every JSON object with `"type": "input_file"`.
fn find_input_files(value: &serde_json::Value, out: &mut Vec<serde_json::Value>) {
    match value {
        serde_json::Value::Object(map) => {
            if map.get("type").and_then(|t| t.as_str()) == Some("input_file") {
                out.push(value.clone());
            }
            map.values().for_each(|v| find_input_files(v, out));
        }
        serde_json::Value::Array(items) => {
            items.iter().for_each(|v| find_input_files(v, out));
        }
        _ => {}
    }
}

fn sole_input_file(value: &serde_json::Value) -> serde_json::Value {
    let mut found = Vec::new();
    find_input_files(value, &mut found);
    assert_eq!(
        found.len(),
        1,
        "expected exactly one input_file item in {value:#}"
    );
    found.pop().unwrap()
}

fn assert_url_only_input_file(input_file: &serde_json::Value) {
    assert_eq!(
        input_file.get("file_url").and_then(|v| v.as_str()),
        Some(PDF_URL),
        "URL PDF should carry file_url: {input_file:#}"
    );
    assert_eq!(
        input_file.get("filename"),
        None,
        "filename must be absent for URL PDFs (issue #1429): {input_file:#}"
    );
    assert_eq!(
        input_file.get("file_data"),
        None,
        "file_data must be absent for URL PDFs: {input_file:#}"
    );
}

#[test]
fn url_pdf_via_input_item_path_omits_filename() {
    let items = super::input_items(url_pdf_message(), 0, &mut CustomCalls::new())
        .expect("URL PDF should convert to input items");
    let json = serde_json::to_value(&items).expect("input items should serialize");
    assert_url_only_input_file(&sole_input_file(&json));
}

#[test]
fn url_pdf_in_full_completion_request_omits_filename() {
    let core_request = crate::completion::CompletionRequest::new(url_pdf_message());

    let request = CompletionRequest::try_from(("gpt-4o".to_string(), core_request))
        .expect("request should convert");
    let json = serde_json::to_value(&request).expect("request should serialize");
    assert_url_only_input_file(&sole_input_file(&json));
}

#[test]
fn base64_pdf_via_input_item_path_keeps_filename() {
    let input = message::Message::User {
        content: vec![message::UserContent::Document(message::Document {
            data: DocumentSourceKind::base64("dGVzdA=="),
            media_type: Some(message::DocumentMediaType::PDF),
            additional_params: None,
        })],
    };

    let items = super::input_items(input, 0, &mut CustomCalls::new())
        .expect("base64 PDF should convert to input items");
    let json = serde_json::to_value(&items).expect("input items should serialize");
    let input_file = sole_input_file(&json);

    assert_eq!(
        input_file.get("file_data").and_then(|v| v.as_str()),
        Some("data:application/pdf;base64,dGVzdA=="),
        "base64 PDF should carry file_data: {input_file:#}"
    );
    assert_eq!(
        input_file.get("filename").and_then(|v| v.as_str()),
        Some("document.pdf"),
        "base64 PDF should keep the default filename: {input_file:#}"
    );
    assert_eq!(
        input_file.get("file_url"),
        None,
        "base64 PDF should not carry file_url: {input_file:#}"
    );
}

/// Raw-capture test: the reply document, driven end to end over a mock
/// transport that hands back a Responses body *and* an `x-request-id`
/// response header. The captured value is the body as parsed; the
/// transport id lives on the normalized response, beside the capture, not
/// inside it. `with_error_response_headers` with `200 OK` is the one unary
/// double that carries response headers.
mod raw_capture {
    use super::*;

    use crate::test_utils::RecordingHttpClient;

    const REQUEST_ID: &str = "req_unit_responses_0001";

    /// A Responses body carrying `service_tier`, which the normalized
    /// response provably lacks.
    const BODY: &str = r#"{
            "id": "resp_raw_1",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "error": null,
            "incomplete_details": null,
            "instructions": null,
            "max_output_tokens": null,
            "model": "gpt-4o-mini-2024-07-18",
            "service_tier": "default",
            "usage": {
                "input_tokens": 4,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 3,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 7
            },
            "output": [{
                "type": "message",
                "id": "msg_raw_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "hello", "annotations": []}]
            }],
            "tools": []
        }"#;

    /// The load-bearing capture property: `raw` is the Responses
    /// `CompletionResponse` as rig parsed it — it deserializes back into
    /// that type and re-serializes to the identical value — and folding
    /// that capture again (with the header id reattached, since the
    /// capture is body only) reproduces every normalized field. Also
    /// reads `service_tier` off the capture, and pins that the transport id
    /// is not part of the capture while the normalized response beside it
    /// carries the header.
    #[tokio::test]
    async fn completion_captures_raw_that_round_trips_into_the_wire_type() {
        let mut headers = http::HeaderMap::new();
        headers.insert("x-request-id", http::HeaderValue::from_static(REQUEST_ID));
        let model = crate::driver::Model::new(
            openai_wire("gpt-4o-mini"),
            RecordingHttpClient::with_error_response_headers(http::StatusCode::OK, BODY, headers),
        );

        let response = model
            .call(crate::completion::CompletionRequest::new("hello"))
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
        assert!(matches!(
            typed.additional_parameters.service_tier,
            Some(OpenAIServiceTier::Default)
        ));
        assert_eq!(raw["service_tier"], "default");
        assert!(raw.get("provider_request_id").is_none());

        let mut refolded = fold_body(typed).expect("re-fold the capture");
        refolded.provider_request_id = Some(REQUEST_ID.to_string());
        assert_eq!(response.identity(), refolded.identity());
        assert_eq!(response.finish_reason(), refolded.finish_reason());
        assert_eq!(response.model(), refolded.model());
        assert_eq!(response.usage, refolded.usage);
        assert_eq!(response.choice, refolded.choice);
        assert_eq!(response.provider_request_id.as_deref(), Some(REQUEST_ID));
    }
}

/// A tool round trip in plain values: the decoded call goes back as the
/// assistant turn, and the result names the call by its own id. Both legs
/// reach the wire under the call's `call_id` from the recorded body, not
/// its item id.
#[test]
fn a_tool_result_named_by_the_call_id_pairs_with_the_call_s_call_id() {
    let response: CompletionResponse =
        serde_json::from_value(response_with_object_top_p()).expect("response should deserialize");
    let choice = folded_choice(response.output);
    let Some(completion::AssistantContent::ToolCall(call)) = choice.first() else {
        panic!("the body's one output is a call: {choice:?}");
    };
    let result =
        completion::Message::tool_result(call.id.clone(), call.function.name.clone(), "sunny");
    let history = vec![
        completion::Message::user("Weather in Paris?"),
        completion::Message::Assistant(crate::message::AssistantMessage::new(choice.clone())),
        result,
    ];

    let request = CompletionRequest::try_from((
        "gpt-4o-mini".to_string(),
        crate::completion::CompletionRequest::from(history),
    ))
    .expect("request conversion should succeed");
    let input = serde_json::to_value(&request).expect("request should serialize")["input"].clone();

    let call_ids: Vec<(&str, &str)> = input
        .as_array()
        .expect("input should be an array")
        .iter()
        .filter_map(|item| Some((item["type"].as_str()?, item["call_id"].as_str()?)))
        .collect();
    assert_eq!(
        call_ids,
        [
            ("function_call", "call_1"),
            ("function_call_output", "call_1")
        ]
    );
}

/// The `input` the OpenAI wire for `model` sends for `history`, shaped by
/// the driver's own `prepare`.
fn input_of(model: &str, history: Vec<completion::Message>) -> Vec<Value> {
    use crate::wire::{Operation, Wire};
    let wire = openai_wire(model);
    let request = crate::operation::Completion::prepare(
        crate::completion::CompletionRequest::from(history),
        &wire.describe(),
    )
    .expect("the history is valid");
    let encoded = wire
        .encode(request, crate::wire::Mode::Unary)
        .expect("the request encodes");
    crate::test_utils::json_body(&encoded.request)["input"]
        .as_array()
        .cloned()
        .unwrap_or_default()
}

/// A turn from `model` on the OpenAI Responses wire.
fn turn_from(model: &str, content: Vec<message::AssistantContent>) -> completion::Message {
    completion::Message::Assistant(message::AssistantMessage {
        content,
        origin: Some(message::Origin::new("openai.responses", "openai", model)),
        stop: Some(message::StopReason::ToolUse),
        native: None,
    })
}

fn reasoning_item() -> Value {
    json!({
        "type": "reasoning",
        "id": "rs_1",
        "summary": [{"type": "summary_text", "text": "Plan."}],
        "encrypted_content": "cipher",
    })
}

fn message_item(id: &str, text: &str) -> Value {
    json!({
        "type": "message",
        "id": id,
        "role": "assistant",
        "status": "completed",
        "phase": "commentary",
        "content": [{"type": "output_text", "text": text, "annotations": [], "logprobs": []}],
    })
}

fn call_item() -> Value {
    json!({
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "lookup",
        "arguments": "{\"q\":\"rig\"}",
        "status": "completed",
    })
}

/// The blocks a decoder makes of [`reasoning_item`], [`message_item`] and
/// [`call_item`], each holding its item.
fn decoded_blocks() -> Vec<message::AssistantContent> {
    let call = message::ToolCall::new(
        message::CallId::from_wire("call_1"),
        message::ToolFunction::new(
            message::ToolName::new("lookup").expect("tool name"),
            json!({"q": "rig"}),
        ),
    );
    vec![
        message::AssistantContent::Reasoning(message::Reasoning::new("Plan."))
            .with_native(reasoning_item()),
        message::AssistantContent::text("Looking.").with_native(message_item("msg_1", "Looking.")),
        message::AssistantContent::ToolCall(call).with_native(call_item()),
    ]
}

fn history_with(turn: completion::Message) -> Vec<completion::Message> {
    vec![
        completion::Message::user("hello"),
        turn,
        completion::Message::tool_result(
            message::CallId::from_wire("call_1"),
            message::ToolName::new("lookup").expect("tool name"),
            "found",
        ),
    ]
}

#[test]
fn a_turn_from_the_same_model_replays_each_item_verbatim_in_order() {
    let input = input_of(
        "gpt-5.4",
        history_with(turn_from("gpt-5.4", decoded_blocks())),
    );
    assert_eq!(
        input[1..4],
        [
            reasoning_item(),
            message_item("msg_1", "Looking."),
            call_item()
        ]
    );
    assert_eq!(input[4]["type"], "function_call_output");
    assert_eq!(input[4]["call_id"], "call_1");
}

/// Another model's turn is rebuilt from its canonical fields as pi rebuilds
/// it: text as completed messages under synthetic ids, reasoning (now text)
/// in its place, the call without an item id under a sanitized call id
/// that its result follows.
#[test]
fn another_models_turn_is_rebuilt_from_its_fields() {
    let call = message::ToolCall::new(
        message::CallId::from_wire("toolu_01|odd id"),
        message::ToolFunction::new(
            message::ToolName::new("lookup").expect("tool name"),
            json!({"q": "rig"}),
        ),
    );
    let foreign = completion::Message::Assistant(message::AssistantMessage {
        content: vec![
            message::AssistantContent::Reasoning(message::Reasoning::new("Thinking."))
                .with_native(json!({"type": "thinking", "signature": "sig"})),
            message::AssistantContent::text("First."),
            message::AssistantContent::ToolCall(call.clone()),
            message::AssistantContent::text("Second."),
        ],
        origin: Some(message::Origin::new(
            "anthropic.messages",
            "anthropic",
            "claude-sonnet-4-5",
        )),
        stop: Some(message::StopReason::ToolUse),
        native: None,
    });
    let history = vec![
        completion::Message::user("hello"),
        foreign,
        completion::Message::User {
            content: vec![message::UserContent::ToolResult(
                call.result(vec![message::ToolResultContent::text("found")]),
            )],
        },
    ];
    let rebuilt = |id: &str, text: &str| {
        json!({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
            "status": "completed",
            "id": id,
        })
    };
    let input = input_of("gpt-5.4", history);
    assert_eq!(
        input[1..],
        [
            rebuilt("msg_rig_1", "Thinking."),
            rebuilt("msg_rig_1_1", "First."),
            json!({
                "type": "function_call",
                "call_id": "toolu_01_odd_id",
                "name": "lookup",
                "arguments": "{\"q\":\"rig\"}",
            }),
            rebuilt("msg_rig_1_2", "Second."),
            json!({
                "type": "function_call_output",
                "call_id": "toolu_01_odd_id",
                "output": "found",
                "status": "completed",
            }),
        ]
    );
}

/// A block edited after decoding no longer matches its item: text and
/// calls are rebuilt, and reasoning, which only its item can carry, is not
/// sent.
#[test]
fn an_edited_block_is_rebuilt_and_edited_reasoning_is_not_sent() {
    let mut blocks = decoded_blocks();
    for block in &mut blocks {
        match block {
            message::AssistantContent::Reasoning(reasoning) => reasoning.text.push('!'),
            message::AssistantContent::Text(text) => text.text.push('!'),
            message::AssistantContent::ToolCall(call) => {
                call.function.arguments = json!({"q": "edited"});
            }
            message::AssistantContent::Image(_) | message::AssistantContent::Opaque(_) => {}
        }
    }
    let input = input_of("gpt-5.4", history_with(turn_from("gpt-5.4", blocks)));
    let kinds: Vec<(&str, Option<&str>)> = input[1..4]
        .iter()
        .map(|item| {
            (
                item["type"].as_str().unwrap_or_default(),
                item["id"].as_str(),
            )
        })
        .collect();
    assert_eq!(
        kinds,
        [
            ("message", Some("msg_rig_1")),
            ("function_call", None),
            ("function_call_output", None),
        ]
    );
    assert_eq!(input[2]["arguments"], "{\"q\":\"edited\"}");
}

#[test]
fn a_custom_tool_call_is_answered_with_a_custom_tool_call_output() {
    let item = json!({
        "type": "custom_tool_call",
        "id": "ctc_1",
        "call_id": "call_1",
        "name": "lookup",
        "input": "rig",
    });
    let call = message::ToolCall::new(
        message::CallId::from_wire("call_1"),
        message::ToolFunction::new(
            message::ToolName::new("lookup").expect("tool name"),
            json!({"input": "rig"}),
        ),
    );
    let turn = turn_from(
        "gpt-5.4",
        vec![message::AssistantContent::ToolCall(call).with_native(item.clone())],
    );
    let input = input_of("gpt-5.4", history_with(turn));
    assert_eq!(input[1], item);
    assert_eq!(
        input[2],
        json!({"type": "custom_tool_call_output", "call_id": "call_1", "output": "found"})
    );
}

#[test]
fn foreign_call_ids_are_normalized_as_pi_normalizes_them() {
    use crate::completion::ReplayTarget;
    let wire = openai_wire("gpt-5.4");
    let long = format!("call_{}", "a".repeat(80));
    for (id, expected) in [
        ("call_1", "call_1".to_owned()),
        ("toolu_01|fc_abc", "toolu_01_fc_abc".to_owned()),
        ("call.with spaces..", "call_with_spaces".to_owned()),
        (long.as_str(), long[..64].to_owned()),
    ] {
        assert_eq!(
            wire.normalize_tool_call_id(id, wire.model(), None),
            expected,
            "{id}"
        );
    }
}

#[test]
fn a_stateless_request_asks_for_the_reasoning_ciphertext() {
    let include = |params: Option<Value>| {
        let mut request = crate::completion::CompletionRequest::new("hello");
        request.additional_params = params;
        serde_json::to_value(wire_request(&openai_wire("gpt-5.4"), request))
            .expect("the request serializes")
            .get("include")
            .cloned()
    };
    assert_eq!(
        include(Some(json!({"store": false}))),
        Some(json!(["reasoning.encrypted_content"]))
    );
    assert_eq!(
        include(Some(json!({"reasoning": {"effort": "low"}}))),
        Some(json!(["reasoning.encrypted_content"]))
    );
    assert_eq!(include(None), None);
}
