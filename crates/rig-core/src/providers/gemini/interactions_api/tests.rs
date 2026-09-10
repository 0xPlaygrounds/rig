use super::*;
use crate::completion::{CompletionRequest, Message};
use crate::message::{self, ToolChoice as MessageToolChoice};
use serde_json::json;

#[test]
fn test_create_request_body_simple() {
    let prompt = Message::User {
        content: vec![message::UserContent::text("Hello")],
    };

    let request = CompletionRequest {
        record_telemetry_content: false,
        model: None,
        chat_history: vec![Message::system("Be precise."), prompt],
        documents: vec![],
        tools: vec![],
        temperature: Some(0.7),
        max_tokens: Some(128),
        tool_choice: Some(MessageToolChoice::Required),
        additional_params: None,
        output_schema: None,
    };

    let result = create_request_body("gemini-2.5-flash".to_string(), request, Some(false))
        .expect("request should build");

    assert_eq!(result.model.as_deref(), Some("gemini-2.5-flash"));
    assert!(result.agent.is_none());
    assert_eq!(result.stream, Some(false));
    assert_eq!(result.system_instruction.as_deref(), Some("Be precise."));

    let config = result.generation_config.expect("generation config missing");
    assert_eq!(config.temperature, Some(0.7));
    assert_eq!(config.max_output_tokens, Some(128));
    assert!(matches!(
        config.tool_choice,
        Some(ToolChoice::Type(ToolChoiceType::Any))
    ));

    let InteractionInput::Steps(steps) = result.input else {
        panic!("expected steps input");
    };
    assert_eq!(steps.len(), 1);
    let Step::UserInput { content: contents } = &steps[0] else {
        panic!("expected user input step");
    };
    assert_eq!(contents.len(), 1);
    match &contents[0] {
        Content::Text(TextContent { text, .. }) => assert_eq!(text, "Hello"),
        other => panic!("unexpected content: {other:?}"),
    }
}

/// `functionResponse.name` is the executed function's name: read from
/// the required `ToolResult::name` — never an identifier.
#[test]
fn tool_result_serializes_the_executed_name_not_an_identifier() {
    use message::{AssistantContent, ToolCall, ToolFunction, ToolResultContent};

    let call = |item_id: Option<&str>, call_id: &str, name: &str| {
        let function = ToolFunction {
            name: name.to_owned(),
            arguments: json!({}),
        };
        let tool_call = match item_id {
            Some(item_id) => ToolCall::from_dual_wire(item_id, call_id, function),
            None => ToolCall::from_wire(call_id, function),
        };
        Message::Assistant {
            id: None,
            content: vec![AssistantContent::ToolCall(tool_call)],
        }
    };
    let result = |item_id: Option<&str>, call_id: &str, name: &str| Message::User {
        content: vec![match item_id {
            Some(item_id) => message::UserContent::tool_result_with_call_id(
                item_id,
                call_id,
                name,
                vec![ToolResultContent::text("out")],
            ),
            None => message::UserContent::tool_result_from_wire(
                call_id,
                name,
                vec![ToolResultContent::text("out")],
            ),
        }],
    };

    let request = CompletionRequest {
        record_telemetry_content: false,
        model: None,
        chat_history: vec![
            // A driver-built result carries the executed name (a repair
            // hook renamed the call: `sum` ran, not `add`).
            call(None, "call_1", "sum"),
            result(None, "call_1", "sum"),
            // An OpenAI-shaped correlator travels as the call id while
            // the required `name` field carries the executed name —
            // `call_abc` must never reach the wire as a name.
            call(None, "call_abc", "get_weather"),
            result(None, "call_abc", "get_weather"),
            // A dual-identifier result (OpenAI Responses: item id `fc_…`
            // + `call_id` `call_…`) keeps the correlator on the wire and
            // the executed name in `name` — `fc_1` must never reach the
            // wire as a name.
            call(Some("fc_1"), "call_9", "get_time"),
            result(Some("fc_1"), "call_9", "get_time"),
        ],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
    };

    let body = create_request_body("gemini-2.5-flash".to_string(), request, None)
        .expect("request should build");
    let input = serde_json::to_value(&body.input).expect("input should serialize");
    let mut names = Vec::new();
    let mut call_ids = Vec::new();
    fn collect(value: &serde_json::Value, names: &mut Vec<String>, call_ids: &mut Vec<String>) {
        match value {
            serde_json::Value::Object(map) => {
                if map.get("type").and_then(|t| t.as_str()) == Some("function_result") {
                    if let Some(name) = map.get("name").and_then(|n| n.as_str()) {
                        names.push(name.to_owned());
                    }
                    if let Some(call_id) = map.get("call_id").and_then(|c| c.as_str()) {
                        call_ids.push(call_id.to_owned());
                    }
                }
                for nested in map.values() {
                    collect(nested, names, call_ids);
                }
            }
            serde_json::Value::Array(items) => {
                for nested in items {
                    collect(nested, names, call_ids);
                }
            }
            _ => {}
        }
    }
    collect(&input, &mut names, &mut call_ids);

    assert_eq!(
        names,
        vec![
            "sum".to_owned(),
            "get_weather".to_owned(),
            "get_time".to_owned()
        ]
    );
    assert_eq!(
        call_ids,
        vec![
            "call_1".to_owned(),
            "call_abc".to_owned(),
            "call_9".to_owned()
        ]
    );
}

#[test]
fn test_tool_result_without_provider_id_sends_minted_call_id() {
    // A call id is always available now: the wire gets the
    // provider-issued id when one exists, else rig's minted handle —
    // the old "Tool results require call_id" error is unrepresentable.
    let call = message::ToolCallId::mint();
    let content = message::UserContent::ToolResult(message::ToolResult {
        call: call.clone(),
        provider: None,
        name: "get_weather".to_string(),
        content: vec![message::ToolResultContent::text("ok")],
    });

    let converted = Content::try_from(content).expect("tool result should convert");
    let Content::FunctionResult(result) = converted else {
        panic!("expected function result");
    };
    assert_eq!(result.call_id.as_deref(), Some(call.as_str()));
    assert_eq!(result.name.as_deref(), Some("get_weather"));
}

#[test]
fn test_tool_result_preserves_text_and_json_types() {
    let content = message::UserContent::ToolResult(message::ToolResult {
        call: message::ToolCallId::new_or_mint("call-123"),
        provider: message::ProviderCallId::new("call-123"),
        name: "get_weather".to_string(),
        content: vec![
            message::ToolResultContent::text(r#"{"status":"literal"}"#),
            message::ToolResultContent::json(json!({ "status": "structured" })),
        ],
    });

    let converted = Content::try_from(content).expect("tool result should convert");
    let Content::FunctionResult(result) = converted else {
        panic!("expected function result");
    };
    let expected_result = json!([
        {
            "type": "text",
            "text": "{\"status\":\"literal\"}"
        },
        {
            "type": "text",
            "text": "{\"status\":\"structured\"}"
        }
    ]);
    assert_eq!(result.result, Some(expected_result.clone()));
    assert_eq!(
        serde_json::to_value(Content::FunctionResult(result))
            .expect("function result should serialize"),
        json!({
            "type": "function_result",
            "name": "get_weather",
            "result": expected_result,
            "call_id": "call-123"
        })
    );
}

#[test]
fn test_tool_result_text_and_json_singletons_remain_scalar() {
    let cases = [
        (
            message::ToolResultContent::text(r#"{"status":"literal"}"#),
            json!("{\"status\":\"literal\"}"),
        ),
        (
            message::ToolResultContent::json(json!({ "status": "structured" })),
            json!({ "status": "structured" }),
        ),
        (
            message::ToolResultContent::json(json!("structured string")),
            json!("structured string"),
        ),
    ];

    for (tool_content, expected) in cases {
        let content = message::UserContent::ToolResult(message::ToolResult {
            call: message::ToolCallId::new_or_mint("call-123"),
            provider: message::ProviderCallId::new("call-123"),
            name: "get_weather".to_string(),
            content: vec![tool_content],
        });

        let Content::FunctionResult(result) =
            Content::try_from(content).expect("tool result should convert")
        else {
            panic!("expected function result");
        };
        assert_eq!(result.result, Some(expected));
    }
}

#[test]
fn test_tool_result_rich_singletons_use_tagged_content() {
    let cases = [
        (
            message::ToolResultContent::json(json!(["sunny", 72])),
            json!([{
                "type": "text",
                "text": "[\"sunny\",72]"
            }]),
        ),
        (
            message::ToolResultContent::image_base64(
                "image-data",
                Some(message::ImageMediaType::PNG),
                None,
            ),
            json!([{
                "type": "image",
                "data": "image-data",
                "mime_type": "image/png"
            }]),
        ),
    ];

    for (tool_content, expected) in cases {
        let content = message::UserContent::ToolResult(message::ToolResult {
            call: message::ToolCallId::new_or_mint("call-123"),
            provider: message::ProviderCallId::new("call-123"),
            name: "get_weather".to_string(),
            content: vec![tool_content],
        });

        let Content::FunctionResult(result) =
            Content::try_from(content).expect("tool result should convert")
        else {
            panic!("expected function result");
        };
        assert_eq!(result.result, Some(expected));
    }
}

#[test]
fn test_tool_result_images_and_text_serialize_as_ordered_tagged_content() {
    let tool_result = message::UserContent::ToolResult(message::ToolResult {
        call: message::ToolCallId::new_or_mint("call-image"),
        provider: message::ProviderCallId::new("call-image"),
        name: "render".to_string(),
        content: vec![
            message::ToolResultContent::image_base64(
                "first-image",
                Some(message::ImageMediaType::PNG),
                None,
            ),
            message::ToolResultContent::text("between-images"),
            message::ToolResultContent::Image(message::Image {
                data: message::DocumentSourceKind::Url(
                    "https://example.com/second.jpg".to_string(),
                ),
                media_type: Some(message::ImageMediaType::JPEG),
                detail: None,
                additional_params: None,
            }),
        ],
    });
    let request = CompletionRequest {
        record_telemetry_content: false,
        model: None,
        chat_history: vec![Message::User {
            content: vec![tool_result],
        }],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
    };

    let request = create_request_body("gemini-2.5-flash".to_string(), request, None)
        .expect("request should build");
    let serialized = serde_json::to_value(request).expect("request should serialize");

    assert_eq!(
        serialized.pointer("/input/0/content/0"),
        Some(&json!({
            "type": "function_result",
            "name": "render",
            "result": [
                {
                    "type": "image",
                    "data": "first-image",
                    "mime_type": "image/png"
                },
                {
                    "type": "text",
                    "text": "between-images"
                },
                {
                    "type": "image",
                    "uri": "https://example.com/second.jpg",
                    "mime_type": "image/jpeg"
                }
            ],
            "call_id": "call-image"
        }))
    );
}

#[test]
fn test_response_function_call_mapping() {
    let interaction = Interaction {
        id: "interaction-1".to_string(),
        steps: vec![Step::FunctionCall(FunctionCallContent {
            name: Some("get_weather".to_string()),
            arguments: Some(json!({"location": "Paris"})),
            id: Some("call-123".to_string()),
        })],
        usage: Some(InteractionUsage {
            total_input_tokens: Some(5),
            total_output_tokens: Some(7),
            total_tokens: Some(12),
            ..Default::default()
        }),
        ..Default::default()
    };

    let response: completion::CompletionResponse =
        interaction.try_into().expect("conversion should succeed");

    let choice = response.choice.first();
    match choice {
        Some(completion::AssistantContent::ToolCall(tool_call)) => {
            assert_eq!(tool_call.function.name, "get_weather");
            assert_eq!(tool_call.id, "call-123");
            assert_eq!(
                tool_call.provider.as_ref().expect("wire id").call_id,
                "call-123"
            );
        }
        other => panic!("unexpected content: {other:?}"),
    }

    assert_eq!(response.usage.input_tokens, 5);
    assert_eq!(response.usage.output_tokens, 7);
    assert_eq!(response.usage.total_tokens, 12);
}

#[test]
fn test_google_search_tool_serialization() {
    let tool = Tool::GoogleSearch;
    let value = serde_json::to_value(tool).expect("tool should serialize");
    assert_eq!(value, json!({ "type": "google_search" }));
}

#[test]
fn test_url_context_tool_serialization() {
    let tool = Tool::UrlContext;
    let value = serde_json::to_value(tool).expect("tool should serialize");
    assert_eq!(value, json!({ "type": "url_context" }));
}

#[test]
fn test_code_execution_tool_serialization() {
    let tool = Tool::CodeExecution;
    let value = serde_json::to_value(tool).expect("tool should serialize");
    assert_eq!(value, json!({ "type": "code_execution" }));
}

#[test]
fn test_google_search_helpers() {
    let interaction = Interaction {
        steps: vec![
            Step::GoogleSearchCall(GoogleSearchCallContent {
                arguments: Some(GoogleSearchCallArguments {
                    queries: Some(vec!["query-one".to_string(), "query-two".to_string()]),
                }),
                id: Some("call-1".to_string()),
            }),
            Step::GoogleSearchResult(GoogleSearchResultContent {
                result: Some(vec![GoogleSearchResult {
                    url: Some("https://example.com".to_string()),
                    title: Some("Example One".to_string()),
                    rendered_content: None,
                }]),
                signature: None,
                is_error: None,
                call_id: Some("call-1".to_string()),
            }),
            Step::GoogleSearchCall(GoogleSearchCallContent {
                arguments: Some(GoogleSearchCallArguments {
                    queries: Some(vec!["query-three".to_string()]),
                }),
                id: Some("call-2".to_string()),
            }),
            Step::GoogleSearchResult(GoogleSearchResultContent {
                result: Some(vec![GoogleSearchResult {
                    url: Some("https://example.org".to_string()),
                    title: Some("Example Two".to_string()),
                    rendered_content: None,
                }]),
                signature: None,
                is_error: None,
                call_id: Some("call-2".to_string()),
            }),
        ],
        ..Default::default()
    };

    let exchanges = interaction.google_search_exchanges();
    assert_eq!(exchanges.len(), 2);
    assert_eq!(exchanges[0].call_id.as_deref(), Some("call-1"));
    assert_eq!(
        exchanges[0].queries(),
        vec!["query-one".to_string(), "query-two".to_string()]
    );
    let exchange_results = exchanges[0].result_items();
    assert_eq!(exchange_results.len(), 1);
    assert_eq!(exchange_results[0].title.as_deref(), Some("Example One"));

    assert_eq!(exchanges[1].call_id.as_deref(), Some("call-2"));
    assert_eq!(exchanges[1].queries(), vec!["query-three".to_string()]);
    let exchange_results = exchanges[1].result_items();
    assert_eq!(exchange_results.len(), 1);
    assert_eq!(exchange_results[0].title.as_deref(), Some("Example Two"));

    let queries = interaction.google_search_queries();
    assert_eq!(queries, vec!["query-one", "query-two", "query-three"]);

    let results = interaction.google_search_results();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].title.as_deref(), Some("Example One"));
    assert_eq!(results[1].title.as_deref(), Some("Example Two"));

    let call_contents = interaction.google_search_call_contents();
    assert_eq!(call_contents.len(), 2);
    assert_eq!(call_contents[0].id.as_deref(), Some("call-1"));
    assert_eq!(call_contents[1].id.as_deref(), Some("call-2"));

    let result_contents = interaction.google_search_result_contents();
    assert_eq!(result_contents.len(), 2);
    assert_eq!(result_contents[0].call_id.as_deref(), Some("call-1"));
    assert_eq!(result_contents[1].call_id.as_deref(), Some("call-2"));
}

#[test]
fn test_google_search_helpers_without_call_id() {
    let interaction = Interaction {
        steps: vec![
            Step::GoogleSearchCall(GoogleSearchCallContent {
                arguments: Some(GoogleSearchCallArguments {
                    queries: Some(vec!["query-one".to_string()]),
                }),
                id: None,
            }),
            Step::GoogleSearchResult(GoogleSearchResultContent {
                result: Some(vec![GoogleSearchResult {
                    url: Some("https://example.com".to_string()),
                    title: Some("Example One".to_string()),
                    rendered_content: None,
                }]),
                signature: None,
                is_error: None,
                call_id: None,
            }),
            Step::GoogleSearchCall(GoogleSearchCallContent {
                arguments: Some(GoogleSearchCallArguments {
                    queries: Some(vec!["query-two".to_string()]),
                }),
                id: Some("call-2".to_string()),
            }),
            Step::GoogleSearchResult(GoogleSearchResultContent {
                result: Some(vec![GoogleSearchResult {
                    url: Some("https://example.org".to_string()),
                    title: Some("Example Two".to_string()),
                    rendered_content: None,
                }]),
                signature: None,
                is_error: None,
                call_id: None,
            }),
        ],
        ..Default::default()
    };

    let exchanges = interaction.google_search_exchanges();
    assert_eq!(exchanges.len(), 2);

    let no_id = exchanges
        .iter()
        .find(|exchange| exchange.call_id.is_none())
        .expect("expected no-id exchange");
    assert_eq!(no_id.calls.len(), 1);
    assert_eq!(no_id.results.len(), 1);

    let with_id = exchanges
        .iter()
        .find(|exchange| exchange.call_id.as_deref() == Some("call-2"))
        .expect("expected call-2 exchange");
    assert_eq!(with_id.calls.len(), 1);
    assert_eq!(with_id.results.len(), 1);
}

#[test]
fn test_url_context_helpers() {
    let interaction = Interaction {
        steps: vec![
            Step::UrlContextCall(UrlContextCallContent {
                arguments: Some(UrlContextCallArguments {
                    urls: Some(vec![
                        "https://example.com".to_string(),
                        "https://example.org".to_string(),
                    ]),
                }),
                id: Some("call-1".to_string()),
            }),
            Step::UrlContextResult(UrlContextResultContent {
                result: Some(vec![UrlContextResult {
                    url: Some("https://example.com".to_string()),
                    status: Some("success".to_string()),
                }]),
                signature: None,
                is_error: None,
                call_id: Some("call-1".to_string()),
            }),
        ],
        ..Default::default()
    };

    let exchanges = interaction.url_context_exchanges();
    assert_eq!(exchanges.len(), 1);
    assert_eq!(exchanges[0].call_id.as_deref(), Some("call-1"));
    assert_eq!(
        exchanges[0].urls(),
        vec!["https://example.com", "https://example.org"]
    );
    let results = exchanges[0].result_items();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].status.as_deref(), Some("success"));

    let urls = interaction.url_context_urls();
    assert_eq!(urls, vec!["https://example.com", "https://example.org"]);

    let results = interaction.url_context_results();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].url.as_deref(), Some("https://example.com"));

    let call_contents = interaction.url_context_call_contents();
    assert_eq!(call_contents.len(), 1);
    assert_eq!(call_contents[0].id.as_deref(), Some("call-1"));

    let result_contents = interaction.url_context_result_contents();
    assert_eq!(result_contents.len(), 1);
    assert_eq!(result_contents[0].call_id.as_deref(), Some("call-1"));
}

#[test]
fn test_url_context_helpers_without_call_id() {
    let interaction = Interaction {
        steps: vec![
            Step::UrlContextCall(UrlContextCallContent {
                arguments: Some(UrlContextCallArguments {
                    urls: Some(vec!["https://example.com".to_string()]),
                }),
                id: None,
            }),
            Step::UrlContextResult(UrlContextResultContent {
                result: Some(vec![UrlContextResult {
                    url: Some("https://example.com".to_string()),
                    status: Some("success".to_string()),
                }]),
                signature: None,
                is_error: None,
                call_id: None,
            }),
            Step::UrlContextCall(UrlContextCallContent {
                arguments: Some(UrlContextCallArguments {
                    urls: Some(vec!["https://example.org".to_string()]),
                }),
                id: Some("call-2".to_string()),
            }),
            Step::UrlContextResult(UrlContextResultContent {
                result: Some(vec![UrlContextResult {
                    url: Some("https://example.org".to_string()),
                    status: Some("success".to_string()),
                }]),
                signature: None,
                is_error: None,
                call_id: None,
            }),
        ],
        ..Default::default()
    };

    let exchanges = interaction.url_context_exchanges();
    assert_eq!(exchanges.len(), 2);

    let no_id = exchanges
        .iter()
        .find(|exchange| exchange.call_id.is_none())
        .expect("expected no-id exchange");
    assert_eq!(no_id.calls.len(), 1);
    assert_eq!(no_id.results.len(), 1);

    let with_id = exchanges
        .iter()
        .find(|exchange| exchange.call_id.as_deref() == Some("call-2"))
        .expect("expected call-2 exchange");
    assert_eq!(with_id.calls.len(), 1);
    assert_eq!(with_id.results.len(), 1);
}

#[test]
fn test_code_execution_helpers() {
    let interaction = Interaction {
        steps: vec![
            Step::CodeExecutionCall(CodeExecutionCallContent {
                arguments: Some(CodeExecutionCallArguments {
                    language: Some("python".to_string()),
                    code: Some("print(2 + 2)".to_string()),
                }),
                id: Some("call-1".to_string()),
            }),
            Step::CodeExecutionResult(CodeExecutionResultContent {
                result: Some("4\n".to_string()),
                signature: None,
                is_error: None,
                call_id: Some("call-1".to_string()),
            }),
        ],
        ..Default::default()
    };

    let exchanges = interaction.code_execution_exchanges();
    assert_eq!(exchanges.len(), 1);
    assert_eq!(exchanges[0].call_id.as_deref(), Some("call-1"));
    assert_eq!(exchanges[0].code_snippets(), vec!["print(2 + 2)"]);
    assert_eq!(exchanges[0].outputs(), vec!["4\n"]);

    let calls = interaction.code_execution_call_contents();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id.as_deref(), Some("call-1"));

    let results = interaction.code_execution_result_contents();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].call_id.as_deref(), Some("call-1"));

    let snippets = interaction.code_execution_snippets();
    assert_eq!(snippets, vec!["print(2 + 2)"]);

    let outputs = interaction.code_execution_outputs();
    assert_eq!(outputs, vec!["4\n"]);
}

#[test]
fn test_code_execution_helpers_without_call_id() {
    let interaction = Interaction {
        steps: vec![
            Step::CodeExecutionCall(CodeExecutionCallContent {
                arguments: Some(CodeExecutionCallArguments {
                    language: Some("python".to_string()),
                    code: Some("print(1 + 1)".to_string()),
                }),
                id: None,
            }),
            Step::CodeExecutionResult(CodeExecutionResultContent {
                result: Some("2\n".to_string()),
                signature: None,
                is_error: None,
                call_id: None,
            }),
            Step::CodeExecutionCall(CodeExecutionCallContent {
                arguments: Some(CodeExecutionCallArguments {
                    language: Some("python".to_string()),
                    code: Some("print(2 + 2)".to_string()),
                }),
                id: Some("call-2".to_string()),
            }),
            Step::CodeExecutionResult(CodeExecutionResultContent {
                result: Some("4\n".to_string()),
                signature: None,
                is_error: None,
                call_id: None,
            }),
        ],
        ..Default::default()
    };

    let exchanges = interaction.code_execution_exchanges();
    assert_eq!(exchanges.len(), 2);

    let no_id = exchanges
        .iter()
        .find(|exchange| exchange.call_id.is_none())
        .expect("expected no-id exchange");
    assert_eq!(no_id.calls.len(), 1);
    assert_eq!(no_id.results.len(), 1);

    let with_id = exchanges
        .iter()
        .find(|exchange| exchange.call_id.as_deref() == Some("call-2"))
        .expect("expected call-2 exchange");
    assert_eq!(with_id.calls.len(), 1);
    assert_eq!(with_id.results.len(), 1);
}

#[test]
fn test_interaction_status_helpers() {
    let mut interaction = Interaction {
        status: Some(InteractionStatus::InProgress),
        ..Default::default()
    };
    assert!(!interaction.is_terminal());
    assert!(!interaction.is_completed());

    // RequiresAction is terminal for the poll (it never advances without
    // the caller submitting tool results) but is not a completion.
    interaction.status = Some(InteractionStatus::RequiresAction);
    assert!(interaction.is_terminal());
    assert!(!interaction.is_completed());

    interaction.status = Some(InteractionStatus::Completed);
    assert!(interaction.is_terminal());
    assert!(interaction.is_completed());

    interaction.status = Some(InteractionStatus::Failed);
    assert!(interaction.is_terminal());
    assert!(!interaction.is_completed());

    interaction.status = Some(InteractionStatus::BudgetExceeded);
    assert!(interaction.is_terminal());
    assert!(!interaction.is_completed());
}

#[test]
fn test_interaction_status_maps_every_wire_variant() {
    use crate::completion::FinishReason as Normalized;

    for (status, expected) in [
        (InteractionStatus::Completed, Normalized::Stop),
        (InteractionStatus::RequiresAction, Normalized::ToolCalls),
        (InteractionStatus::BudgetExceeded, Normalized::Length),
        // Statuses rig does not model survive in the provider's own
        // spelling rather than being guessed at.
        (
            InteractionStatus::InProgress,
            Normalized::Other("in_progress".to_string()),
        ),
        (
            InteractionStatus::Incomplete,
            Normalized::Other("incomplete".to_string()),
        ),
        (
            InteractionStatus::Failed,
            Normalized::Other("failed".to_string()),
        ),
        (
            InteractionStatus::Cancelled,
            Normalized::Other("cancelled".to_string()),
        ),
    ] {
        assert_eq!(
            map_interaction_status(&status),
            expected,
            "status {status:?}"
        );
    }
}

#[test]
fn test_interaction_status_wire_spelling_matches_serde() {
    // `as_wire_str` is hand-written; keep it honest against the serde
    // representation the same enum deserializes from.
    for status in [
        InteractionStatus::InProgress,
        InteractionStatus::RequiresAction,
        InteractionStatus::Incomplete,
        InteractionStatus::BudgetExceeded,
        InteractionStatus::Completed,
        InteractionStatus::Failed,
        InteractionStatus::Cancelled,
    ] {
        let serialized = serde_json::to_value(&status).expect("status should serialize");
        assert_eq!(serialized, json!(status.as_wire_str()));
    }
}

#[test]
fn test_unknown_interaction_status_round_trips_verbatim() {
    // A status this crate does not know must land in `Unknown` with the
    // provider's spelling intact — and serialize back to the same string —
    // rather than failing the whole payload.
    let status: InteractionStatus =
        serde_json::from_value(json!("status_future")).expect("unknown status should deserialize");
    assert!(matches!(&status, InteractionStatus::Unknown(s) if s == "status_future"));
    assert_eq!(status.as_wire_str(), "status_future");
    assert_eq!(
        serde_json::to_value(&status).expect("status should serialize"),
        json!("status_future")
    );
    assert_eq!(
        map_interaction_status(&status),
        crate::completion::FinishReason::Other("status_future".to_string())
    );
}

#[test]
fn test_interaction_with_unknown_status_stays_parseable() {
    // A status Google ships tomorrow must not fail the interaction
    // payload; the unknown status is conservatively *terminal* — only the
    // known in-flight statuses keep a poll loop waiting, so a future
    // status surfaces to the caller instead of hanging it.
    let interaction: Interaction = serde_json::from_value(json!({
        "id": "int-future",
        "status": "status_future",
        "usage": {"total_tokens": 5}
    }))
    .expect("unknown status should not fail the payload");

    assert_eq!(interaction.id, "int-future");
    assert!(matches!(
        interaction.status,
        Some(InteractionStatus::Unknown(ref s)) if s == "status_future"
    ));
    assert!(interaction.is_terminal());
    assert!(!interaction.is_completed());
    assert_eq!(
        interaction.usage.as_ref().and_then(|u| u.total_tokens),
        Some(5)
    );
}

#[test]
fn test_completion_response_carries_normalized_metadata() {
    let interaction = Interaction {
        id: "interaction-meta".to_string(),
        model: Some("gemini-2.5-pro".to_string()),
        status: Some(InteractionStatus::BudgetExceeded),
        steps: vec![Step::ModelOutput {
            content: vec![Content::Text(TextContent {
                text: "partial answer".to_string(),
                annotations: None,
            })],
        }],
        ..Default::default()
    };

    let response: completion::CompletionResponse =
        interaction.try_into().expect("conversion should succeed");

    assert_eq!(response.provider, PROVIDER_NAME);
    assert_eq!(response.model.as_deref(), Some("gemini-2.5-pro"));
    assert_eq!(response.response_id.as_deref(), Some("interaction-meta"));
    assert_eq!(response.message_id, None);
    assert_eq!(
        response.finish_reason(),
        Some(crate::completion::FinishReason::Length)
    );
}

#[test]
fn test_completion_response_upgrades_completed_to_tool_calls() {
    // A `completed` interaction whose outputs are function calls is a tool
    // turn; the normalized response must say so.
    let interaction = Interaction {
        id: "interaction-tool".to_string(),
        status: Some(InteractionStatus::Completed),
        steps: vec![Step::FunctionCall(FunctionCallContent {
            name: Some("get_weather".to_string()),
            arguments: Some(json!({"location": "Paris"})),
            id: Some("call-123".to_string()),
        })],
        ..Default::default()
    };

    let response: completion::CompletionResponse =
        interaction.try_into().expect("conversion should succeed");

    assert_eq!(
        response.finish_reason(),
        Some(crate::completion::FinishReason::ToolCalls)
    );
    assert_eq!(response.model, None);
}

#[test]
fn test_budget_exceeded_status_deserializes() {
    let status: InteractionStatus = serde_json::from_value(json!("budget_exceeded"))
        .expect("budget_exceeded should deserialize");

    assert!(matches!(status, InteractionStatus::BudgetExceeded));
    assert!(status.is_terminal());
}

#[test]
fn test_budget_exceeded_status_update_deserializes() {
    let event: InteractionSseEvent = serde_json::from_value(json!({
        "event_type": "interaction.status_update",
        "interaction_id": "interaction-123",
        "status": "budget_exceeded",
        "event_id": "event-456"
    }))
    .expect("budget_exceeded status update should deserialize");

    match event {
        InteractionSseEvent::InteractionStatusUpdate {
            interaction_id,
            status,
            event_id,
        } => {
            assert_eq!(interaction_id, "interaction-123");
            assert!(matches!(status, InteractionStatus::BudgetExceeded));
            assert!(status.is_terminal());
            assert_eq!(event_id.as_deref(), Some("event-456"));
        }
        other => panic!("expected status update event, got {other:?}"),
    }
}

#[test]
fn test_build_interaction_stream_path() {
    let path = build_interaction_stream_path("interaction-123", None);
    assert_eq!(path, "/v1beta/interactions/interaction-123?stream=true");

    let path = build_interaction_stream_path("interaction-123", Some("event-456"));
    assert_eq!(
        path,
        "/v1beta/interactions/interaction-123?stream=true&last_event_id=event-456"
    );
}

#[test]
fn test_inline_citations_from_annotations() {
    let text_content = TextContent {
        text: "Hello world".to_string(),
        annotations: Some(vec![
            Annotation {
                start_index: Some(6),
                end_index: Some(11),
                source: Some("https://example.com".to_string()),
            },
            Annotation {
                start_index: Some(0),
                end_index: Some(5),
                source: Some("https://hello.example".to_string()),
            },
        ]),
    };

    let cited = text_content.with_inline_citations();
    assert_eq!(
        cited,
        "Hello[1](https://hello.example) world[2](https://example.com)"
    );

    let interaction = Interaction {
        steps: vec![Step::ModelOutput {
            content: vec![Content::Text(text_content)],
        }],
        ..Default::default()
    };

    let cited_text = interaction.text_with_inline_citations();
    assert_eq!(
        cited_text.as_deref(),
        Some("Hello[1](https://hello.example) world[2](https://example.com)")
    );
}
