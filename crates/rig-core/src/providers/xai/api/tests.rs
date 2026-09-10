use super::{Content, Message, Role, create_completion_request};
use crate::completion::{CompletionRequest, CompletionRequestBuilder, Document};
use crate::message::{
    AssistantContent, Message as RigMessage, Reasoning, ReasoningContent, ToolChoice,
    ToolResultContent, UserContent,
};
use crate::providers::openai::responses_api::ReasoningSummary;
use crate::test_utils::MockCompletionModel;

fn request_value(request: CompletionRequest) -> serde_json::Value {
    create_completion_request("grok-4-0709".to_string(), request, &[], false, false)
        .expect("request conversion should succeed")
        .1
}

#[test]
fn xai_request_includes_normalized_documents() {
    let request = CompletionRequestBuilder::new(
        MockCompletionModel::default(),
        "What does glarb-glarb mean?",
    )
    .document(Document {
        id: "doc_1".to_string(),
        text: "Definition of glarb-glarb: an ancient tool.".to_string(),
        additional_props: Default::default(),
    })
    .build();

    let serialized = request_value(request);
    let input = serialized["input"]
        .as_array()
        .expect("xAI request input should be an array");

    assert!(
        input
            .iter()
            .any(|message| message.to_string().contains("glarb-glarb")),
        "normalized documents should be forwarded into xAI input"
    );
}

#[test]
fn xai_direct_request_keeps_documents_after_system_messages() {
    let request = CompletionRequest {
        model: None,
        chat_history: vec![
            RigMessage::system("System prompt"),
            RigMessage::assistant("Earlier assistant turn"),
            RigMessage::system("Mid-conversation instruction"),
            RigMessage::user("What is glarb-glarb?"),
        ],
        documents: vec![Document {
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

    let serialized = request_value(request);
    let input = serialized["input"]
        .as_array()
        .expect("xAI request input should be an array");

    assert_eq!(input.len(), 5);
    assert_eq!(input[0]["role"], "system");
    assert_eq!(input[1]["role"], "user");
    assert!(input[1].to_string().contains("<file id: doc_1>"));
    assert_eq!(input[2]["role"], "assistant");
    assert_eq!(input[3]["role"], "system");
    assert_eq!(input[4]["role"], "user");
    assert_eq!(
        input
            .iter()
            .filter(|message| message.to_string().contains("<file id: doc_1>"))
            .count(),
        1,
        "document input should appear exactly once: {input:?}"
    );
}

#[test]
fn xai_request_uses_responses_tool_choice_for_specific_tool() {
    let request = CompletionRequestBuilder::new(MockCompletionModel::default(), "Use a tool.")
        .tool(crate::completion::ToolDefinition {
            name: "alpha".to_string(),
            description: "Alpha tool".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        })
        .tool(crate::completion::ToolDefinition {
            name: "beta".to_string(),
            description: "Beta tool".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        })
        .tool_choice(ToolChoice::Specific {
            function_names: vec!["beta".to_string()],
        })
        .build();

    let serialized = request_value(request);
    assert_eq!(
        serialized["tool_choice"],
        serde_json::json!({"type": "function", "name": "beta"})
    );
}

#[test]
fn xai_stream_request_sets_stream_without_additional_params() {
    let request = CompletionRequestBuilder::new(MockCompletionModel::default(), "hello").build();
    let (_, serialized) =
        create_completion_request("grok-4-0709".to_string(), request, &[], false, true)
            .expect("streaming request conversion should succeed");

    assert_eq!(serialized["stream"], true);
}

#[test]
fn xai_strict_mode_normalizes_function_tools_from_every_source() {
    let mut request =
        CompletionRequestBuilder::new(MockCompletionModel::default(), "Use one of the tools.")
            .tool(crate::completion::ToolDefinition {
                name: "request_tool".to_string(),
                description: "A request tool".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {"request": {"type": "string"}}
                }),
            })
            .build();
    request.additional_params = Some(serde_json::json!({
        "tools": [
            {
                "type": "function",
                "name": "additional_tool",
                "description": "An additional_params tool",
                "parameters": {
                    "type": "object",
                    "properties": {"additional": {"type": "string"}}
                }
            },
            {"type": "web_search"}
        ]
    }));
    let default_tools = [
        crate::providers::openai::responses_api::ResponsesToolDefinition::function(
            "default_tool",
            "A model-level default tool",
            serde_json::json!({
                "type": "object",
                "properties": {"default": {"type": "string"}}
            }),
        ),
    ];

    let (_, serialized) = create_completion_request(
        "grok-4-0709".to_string(),
        request,
        &default_tools,
        true,
        false,
    )
    .expect("request conversion should succeed");
    let tools = serialized["tools"]
        .as_array()
        .expect("tools should be an array");

    assert_eq!(tools.len(), 4);
    for tool in tools.iter().filter(|tool| tool["type"] == "function") {
        assert_eq!(tool["strict"], true);
        assert_eq!(tool["parameters"]["additionalProperties"], false);
        assert_eq!(
            tool["parameters"]["required"]
                .as_array()
                .expect("strict object schema should require every property")
                .len(),
            1
        );
    }
    assert_eq!(tools[2], serde_json::json!({"type": "web_search"}));
}

#[test]
fn mixed_user_content_preserves_order_without_duplicate_text() {
    let message = RigMessage::User {
        content: vec![
            UserContent::text("before"),
            UserContent::tool_result_with_call_id(
                "result-id",
                "call-id".to_string(),
                "tool",
                vec![ToolResultContent::json(serde_json::json!({ "ok": true }))],
            ),
            UserContent::text("after"),
        ],
    };

    let messages = Vec::<Message>::try_from(message).expect("mixed content should convert");
    assert_eq!(messages.len(), 3);
    assert!(matches!(
        &messages[0],
        Message::Message {
            role: Role::User,
            content: Content::Text(text),
        } if text == "before"
    ));
    assert!(matches!(
        &messages[1],
        Message::FunctionCallOutput { call_id, output }
            if call_id == "call-id" && output == r#"{"ok":true}"#
    ));
    assert!(matches!(
        &messages[2],
        Message::Message {
            role: Role::User,
            content: Content::Text(text),
        } if text == "after"
    ));
}

#[test]
fn assistant_redacted_reasoning_is_serialized_as_encrypted_content() {
    let reasoning = Reasoning {
        id: Some("rs_1".to_string()),
        content: vec![ReasoningContent::Redacted {
            data: "opaque-redacted".to_string(),
        }],
    };
    let message = RigMessage::Assistant {
        id: Some("assistant_1".to_string()),
        content: vec![AssistantContent::Reasoning(reasoning)],
    };

    let items = Vec::<Message>::try_from(message).expect("convert assistant message");
    assert_eq!(items.len(), 1);
    assert!(matches!(
        items.first(),
        Some(Message::Reasoning {
            id,
            summary,
            encrypted_content: Some(encrypted_content),
        }) if id == "rs_1" && summary.is_empty() && encrypted_content == "opaque-redacted"
    ));
}

#[test]
fn assistant_redacted_reasoning_does_not_leak_into_summary_text() {
    let reasoning = Reasoning {
        id: Some("rs_2".to_string()),
        content: vec![
            ReasoningContent::Text {
                text: "explain".to_string(),
                signature: None,
            },
            ReasoningContent::Redacted {
                data: "opaque-redacted".to_string(),
            },
        ],
    };
    let message = RigMessage::Assistant {
        id: Some("assistant_2".to_string()),
        content: vec![AssistantContent::Reasoning(reasoning)],
    };

    let items = Vec::<Message>::try_from(message).expect("convert assistant message");
    let Some(Message::Reasoning {
        summary,
        encrypted_content,
        ..
    }) = items.first()
    else {
        panic!("Expected reasoning item");
    };

    assert_eq!(
        summary,
        &vec![ReasoningSummary::SummaryText {
            text: "explain".to_string()
        }]
    );
    assert_eq!(encrypted_content.as_deref(), Some("opaque-redacted"));
}

#[test]
fn assistant_empty_reasoning_content_roundtrips_without_error() {
    let reasoning = Reasoning {
        id: Some("rs_empty".to_string()),
        content: vec![],
    };
    let message = RigMessage::Assistant {
        id: Some("assistant_2b".to_string()),
        content: vec![AssistantContent::Reasoning(reasoning)],
    };

    let items = Vec::<Message>::try_from(message).expect("convert assistant message");
    assert_eq!(items.len(), 1);
    assert!(matches!(
        items.first(),
        Some(Message::Reasoning {
            id,
            summary,
            encrypted_content,
        }) if id == "rs_empty" && summary.is_empty() && encrypted_content.is_none()
    ));
}

#[test]
fn assistant_reasoning_without_id_is_dropped_from_request_input() {
    // Only wire-genuine ids exist in durable histories; an id-less
    // reasoning item (a cross-provider replay from a wire that issues no
    // reasoning ids) drops from request input — mirroring the OpenAI
    // Responses handling — instead of failing the whole request or,
    // worse, fabricating an identifier xAI never issued (#2258 A1).
    let message = RigMessage::Assistant {
        id: Some("assistant_no_reasoning_id".to_string()),
        content: vec![AssistantContent::Reasoning(Reasoning::new("thinking"))],
    };

    let converted = Vec::<Message>::try_from(message).expect("conversion must not fail");
    assert!(
        converted
            .iter()
            .all(|item| !matches!(item, Message::Reasoning { .. })),
        "an id-less reasoning item must not reach the request: {converted:?}"
    );
}

#[test]
fn serialized_message_type_tags_are_snake_case() {
    let function_call = Message::function_call(
        "call_1".to_string(),
        "tool_name".to_string(),
        "{\"arg\":1}".to_string(),
    );
    let user_message = Message::user("hello");

    let function_call_json = serde_json::to_value(function_call).expect("serialize function_call");
    let user_message_json = serde_json::to_value(user_message).expect("serialize message");

    assert_eq!(
        function_call_json
            .get("type")
            .and_then(|value| value.as_str()),
        Some("function_call")
    );
    assert_eq!(
        user_message_json
            .get("type")
            .and_then(|value| value.as_str()),
        Some("message")
    );
}

#[test]
fn user_tool_result_without_call_id_replays_the_minted_handle() {
    // An empty wire id records no provider id and mints the correlation
    // handle; the minted handle (never an empty string) goes on the wire.
    let message = RigMessage::tool_result("", "tool_1", "result payload");

    let converted = Vec::<Message>::try_from(message).expect("id-less tool results convert");
    assert!(matches!(
        converted.as_slice(),
        [Message::FunctionCallOutput { call_id, output }]
            if !call_id.is_empty() && output == "result payload"
    ));
}

#[test]
fn assistant_tool_call_without_call_id_replays_the_minted_handle() {
    // An empty wire id records no provider id and mints the correlation
    // handle; the minted handle (never an empty string) goes on the wire.
    let message = RigMessage::Assistant {
        id: Some("assistant_3".to_string()),
        content: vec![AssistantContent::tool_call(
            "",
            "my_tool",
            serde_json::json!({"arg":"value"}),
        )],
    };

    let converted = Vec::<Message>::try_from(message).expect("id-less tool calls convert");
    assert!(matches!(
        converted.as_slice(),
        [Message::FunctionCall { call_id, name, .. }]
            if !call_id.is_empty() && name == "my_tool"
    ));
}
