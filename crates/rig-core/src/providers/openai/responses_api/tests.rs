use super::*;
use crate::completion::CompletionRequestBuilder;
use crate::message;
use crate::test_utils::MockCompletionModel;
use serde_json::json;
use std::collections::HashMap;

#[test]
fn output_text_extras_survive_generic_conversion_and_replay() {
    // Ingest capture is unconditional: the wire's sibling keys ride the
    // generic block under this wire's params key. Replay is gated: only
    // this wire's serializer reads them back, value-equal.
    let mut extras = Map::new();
    extras.insert(
        "annotations".to_string(),
        json!([{"type": "url_citation", "url": "https://example.com"}]),
    );
    let wire = OutputText {
        text: "cited".to_string(),
        extras: extras.clone(),
    };
    let generic: completion::AssistantContent = AssistantContent::OutputText(wire).into();
    let completion::AssistantContent::Text(text) = &generic else {
        panic!("expected a text block, got: {generic:?}");
    };
    assert_eq!(
        text.additional_params
            .as_ref()
            .and_then(|params| params.get(OPENAI_RESPONSES_EXTRAS_KEY)),
        Some(&Value::Object(extras.clone()))
    );

    let replayed = OutputText::from_message_text(text.text.clone(), text.additional_params.clone());
    assert_eq!(replayed.text, "cited");
    assert_eq!(replayed.extras, extras);

    // Extras ride a serde flatten, so the reserved keys the named field
    // and the tag own must never replay from history — a duplicate JSON
    // key would let persisted data shadow the block's real text or tag.
    let hostile = message::AdditionalParams::try_from_value(json!({
        OPENAI_RESPONSES_EXTRAS_KEY: {
            "text": "evil",
            "type": "evil_type",
            "annotations": ["kept"],
        }
    }))
    .expect("object params");
    let replayed = OutputText::from_message_text("real", hostile);
    assert_eq!(replayed.text, "real");
    assert!(replayed.extras.get("text").is_none());
    assert!(replayed.extras.get("type").is_none());
    assert_eq!(replayed.extras.get("annotations"), Some(&json!(["kept"])));
    let wire = serde_json::to_value(&replayed).expect("serialize");
    assert_eq!(wire.get("text"), Some(&json!("real")));

    // Replay honors only this wire's extras: an empty text block
    // annotated with a *foreign* wire's extras (the shape anthropic
    // ingest writes for raw server-tool content) produces no Responses
    // item at all — its extras cannot reach this wire, and an empty
    // assistant item the wire never sent risks a rejection — while an
    // `openai_responses`-annotated empty block still replays.
    let foreign = message::Message::Assistant {
        id: None,
        content: vec![completion::AssistantContent::Text(message::Text {
            text: String::new(),
            additional_params: message::AdditionalParams::try_from_value(json!({
                "anthropic_content": {"type": "server_tool_use", "id": "srv_1"}
            }))
            .expect("object params"),
        })],
    };
    let items: Vec<InputItem> = foreign.try_into().expect("convert");
    assert!(
        items.is_empty(),
        "foreign-annotated empty block must produce no Responses item: {items:?}"
    );

    let own_annotated_empty = |id: Option<String>| message::Message::Assistant {
        id,
        content: vec![completion::AssistantContent::Text(message::Text {
            text: String::new(),
            additional_params: message::AdditionalParams::try_from_value(json!({
                OPENAI_RESPONSES_EXTRAS_KEY: {"annotations": ["kept"]}
            }))
            .expect("object params"),
        })],
    };
    // With a message id, the Assistant form carries the extras.
    let items: Vec<InputItem> = own_annotated_empty(Some("msg_1".to_string()))
        .try_into()
        .expect("convert");
    assert_eq!(
        items.len(),
        1,
        "own-wire-annotated empty block must replay when deliverable: {items:?}"
    );
    let serialized = serde_json::to_value(&items).expect("serialize");
    assert_eq!(
        serialized[0]["content"][0]["annotations"],
        json!(["kept"]),
        "the replayed item must carry the extras: {serialized}"
    );
    // Without an id the only form is the bare-string `AssistantInput`,
    // which cannot carry extras — an empty block is skipped rather than
    // sent as a content-free item with its extras dropped.
    let items: Vec<InputItem> = own_annotated_empty(None).try_into().expect("convert");
    assert!(
        items.is_empty(),
        "undeliverable annotated empty block must be skipped: {items:?}"
    );

    // A bare block stays bare in both directions.
    let bare: completion::AssistantContent =
        AssistantContent::OutputText(OutputText::new("plain")).into();
    let completion::AssistantContent::Text(text) = &bare else {
        panic!("expected a text block, got: {bare:?}");
    };
    assert_eq!(text.additional_params, None);
    assert!(
        OutputText::from_message_text("plain", None)
            .extras
            .is_empty(),
        "no params, no extras"
    );
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
        name: "get_weather".to_string(),
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
            call: message::ToolCallId::new_or_mint("call-id"),
            provider: message::ProviderCallId::new("call-id")
                .map(|provider| provider.with_item_id("result-id")),
            name: "tool".to_string(),
            content: vec![content],
        })],
    }
}

#[test]
fn mixed_user_content_preserves_order_around_tool_results() {
    let input = message::Message::User {
        content: vec![
            message::UserContent::text("before"),
            message::UserContent::tool_result_with_call_id(
                "result-id",
                "call-id".to_string(),
                "tool",
                vec![message::ToolResultContent::text("tool output")],
            ),
            message::UserContent::text("after"),
        ],
    };

    let items = Vec::<InputItem>::try_from(input).expect("input item conversion");

    assert!(matches!(
        items.as_slice(),
        [
            InputItem {
                input: InputContent::Message(Message::User { content: before, .. }),
                ..
            },
            InputItem {
                input: InputContent::FunctionCallOutput(ToolResult { call_id, .. }),
                ..
            },
            InputItem {
                input: InputContent::Message(Message::User { content: after, .. }),
                ..
            },
        ] if matches!(before.first(), Some(UserContent::InputText { text }) if text == "before")
            && call_id == "call-id"
            && matches!(after.first(), Some(UserContent::InputText { text }) if text == "after")
    ));
}

fn reasoning_input_items(items: &[InputItem]) -> Vec<serde_json::Value> {
    items
        .iter()
        .map(|item| serde_json::to_value(item).expect("input item should serialize"))
        .filter(|value| value["type"] == "reasoning")
        .collect()
}

/// F7 leak route (a): reasoning replayed cross-provider — another
/// provider's stream aggregated under a boundary-minted id and swapped
/// onto a Responses model — must not serialize the fabricated id
/// upstream; the item is dropped like main dropped id-less reasoning.
/// A wire-plausible id keeps round-tripping.
#[tokio::test]
async fn cross_provider_minted_reasoning_ids_are_not_serialized_upstream() {
    use crate::completion::CompletionModel as _;
    use crate::test_utils::MockStreamEvent;
    use futures::StreamExt as _;

    // The constant-id shape gemini/ollama/chat-compat streams leave in
    // history, via the mock model's streaming pipeline.
    let model = MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::reasoning_delta("thinking hard"),
        MockStreamEvent::text("answer"),
        MockStreamEvent::final_response_with_default_usage(),
    ]]);
    let request = CompletionRequestBuilder::new(model.clone(), "hi").build();
    let mut stream = model.stream(request).await.expect("mock stream");
    while stream.next().await.is_some() {}
    let choice = stream.choice.clone();
    // The provenance funnel: a minted stream identity never becomes the
    // durable `Reasoning::id`, so the replayed history carries no id at
    // all — there is nothing for a serializer gate to filter, and no
    // gate exists.
    assert!(
        choice.iter().any(
            |content| matches!(content, message::AssistantContent::Reasoning(reasoning)
                if reasoning.id.is_none())
        ),
        "a minted stream identity must aggregate as an id-less reasoning part"
    );

    let items = Vec::<InputItem>::try_from(crate::completion::Message::Assistant {
        id: None,
        content: choice,
    })
    .expect("history should convert");
    assert!(
        reasoning_input_items(&items).is_empty(),
        "an id-less reasoning part must not reach the request input"
    );

    // A wire-plausible id is provider-issued and must round-trip.
    let items = Vec::<InputItem>::try_from(crate::completion::Message::Assistant {
        id: None,
        content: vec![message::AssistantContent::Reasoning(message::Reasoning {
            id: Some("rs_0123".to_string()),
            content: vec![message::ReasoningContent::Text {
                text: "real item".to_string(),
                signature: None,
            }],
        })],
    })
    .expect("history should convert");
    let reasoning = reasoning_input_items(&items);
    assert_eq!(reasoning.len(), 1);
    assert_eq!(reasoning[0]["id"], "rs_0123");
}

/// F7 leak route (b), closed structurally: a same-provider delta-only
/// Responses stream whose reasoning deltas lack `item_id` keys
/// accumulation by a minted identity that never becomes a durable id, so
/// the next request carries no fabricated `output-{index}` item.
#[tokio::test]
async fn delta_only_stream_minted_output_ids_are_not_serialized_upstream() {
    use crate::test_utils::streaming_conformance::{fixtures, ok_chunks};
    use bytes::Bytes;

    let sse = |frame: &serde_json::Value| Bytes::from(format!("data: {frame}\n\n"));
    let frames = vec![
        // No `item_id`: the streaming adapter mints `output-0`.
        sse(&json!({
            "type": "response.reasoning_text.delta",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 1,
            "delta": "unattributed thought",
        })),
        sse(&json!({
            "type": "response.completed",
            "sequence_number": 2,
            "response": {
                "id": "resp_1",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "gpt-5.4",
                "output": [],
                "tools": [],
                "usage": null,
            },
        })),
    ];
    let drained = fixtures::openai_responses::driver()
        .drive(ok_chunks(frames))
        .await
        .expect("stream should complete");
    // The minted `output_index` identity keys accumulation only; the
    // aggregated part carries no durable id, so nothing can go upstream.
    assert!(
        drained.choice.iter().any(
            |content| matches!(content, message::AssistantContent::Reasoning(reasoning)
                if reasoning.id.is_none())
        ),
        "an id-less delta stream must aggregate as an id-less reasoning part"
    );

    let items = Vec::<InputItem>::try_from(crate::completion::Message::Assistant {
        id: None,
        content: drained.choice,
    })
    .expect("history should convert");
    assert!(
        reasoning_input_items(&items).is_empty(),
        "an id-less reasoning part must not reach the request input"
    );
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

        let items: Vec<InputItem> = input.try_into().expect("input item conversion");
        assert!(matches!(
            items.as_slice(),
            [InputItem {
                input: InputContent::FunctionCallOutput(ToolResult {
                    output: ToolResultOutput::Text(output),
                    ..
                }),
                ..
            }] if output == &expected
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
            call: message::ToolCallId::new_or_mint("call-id"),
            provider: message::ProviderCallId::new("call-id")
                .map(|provider| provider.with_item_id("result-id")),
            name: "tool".to_string(),
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

    let items: Vec<InputItem> = input.try_into().expect("input item conversion");

    match items.as_slice() {
        [
            InputItem {
                input: InputContent::FunctionCallOutput(ToolResult { output, .. }),
                ..
            },
        ] => {
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
            call: message::ToolCallId::new_or_mint("call-id"),
            provider: message::ProviderCallId::new("call-id")
                .map(|provider| provider.with_item_id("result-id")),
            name: "tool".to_string(),
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

    let items: Vec<InputItem> = input.try_into().expect("input item conversion");
    match items.as_slice() {
        [
            InputItem {
                input: InputContent::FunctionCallOutput(ToolResult { output, .. }),
                ..
            },
        ] => assert_output(output),
        other => panic!("expected one rich function output, got {other:?}"),
    }
}

#[test]
fn tool_result_file_id_image_uses_the_native_wire_field() {
    let input = rig_tool_result(message::ToolResultContent::Image(message::Image {
        data: message::DocumentSourceKind::FileId("file-image-123".to_string()),
        media_type: None,
        detail: None,
        additional_params: None,
    }));

    let items: Vec<InputItem> = input.try_into().expect("input item conversion");
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
    completion::CompletionRequest {
        model: None,
        chat_history: vec![message::Message::user("what's the weather?")],
        documents: Vec::new(),
        tools: vec![weather_tool_definition()],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    }
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
        function_names: vec!["get_weather".to_string()],
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
        function_names: vec!["add".to_string(), "subtract".to_string()],
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
        converted,
        Err(CompletionError::RequestError(error))
            if error.to_string().contains("at least one function name")
    ));
}

#[test]
fn responses_request_with_specific_tool_choice_serializes_named_function() {
    let mut request = weather_tool_request();
    request.tool_choice = Some(message::ToolChoice::Specific {
        function_names: vec!["get_weather".to_string()],
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

    let serialized = serde_json::to_value(tool).expect("tool should serialize");
    assert!(serialized.get("strict").is_none());
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
    completion::CompletionRequest {
        model: None,
        chat_history: vec![
            message::Message::system(preamble),
            message::Message::user("Hello"),
        ],
        documents: Vec::new(),
        tools: Vec::new(),
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    }
}

fn system_only_request(system_text: &str) -> completion::CompletionRequest {
    completion::CompletionRequest {
        model: None,
        chat_history: vec![completion::Message::system(system_text)],
        documents: Vec::new(),
        tools: Vec::new(),
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    }
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
    let request = CompletionRequestBuilder::new(MockCompletionModel::default(), "Hello")
        .preamble("System one".to_string())
        .message(completion::Message::system("System two"))
        .build();

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
fn responses_model_can_fallback_to_system_messages_in_input() {
    let client = crate::providers::openai::Client::new_with(
        "dummy-key",
        crate::test_utils::RecordingHttpClient::new(""),
    )
    .expect("client");
    let model =
        ResponsesCompletionModel::new(client, "gpt-4o-mini").with_system_instructions_as_messages();

    let req = model
        .create_completion_request(request_with_preamble("You are concise."))
        .expect("request should convert");
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
fn responses_client_can_fallback_to_system_messages_in_input() {
    use crate::prelude::CompletionClient;

    let client = crate::providers::openai::Client::new_with(
        "dummy-key",
        crate::test_utils::RecordingHttpClient::new(""),
    )
    .expect("client")
    .with_system_instructions_as_messages();
    let model = client.completion_model("gpt-4o-mini");

    let req = model
        .create_completion_request(request_with_preamble("You are concise."))
        .expect("request should convert");
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
fn responses_model_can_lift_all_system_messages_via_placement() {
    let client = crate::providers::openai::Client::new_with(
        "dummy-key",
        crate::test_utils::RecordingHttpClient::new(""),
    )
    .expect("client");
    let model = ResponsesCompletionModel::new(client, "gpt-4o-mini")
        .with_system_instructions_placement(SystemInstructionsPlacement::AllInstructions);

    let request = CompletionRequestBuilder::new(MockCompletionModel::default(), "again")
        .preamble("System one".to_string())
        .message(completion::Message::user("hi"))
        .message(completion::Message::system("Mid-conversation instruction"))
        .build();

    let req = model
        .create_completion_request(request)
        .expect("request should convert");
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
fn responses_client_placement_survives_completions_api_round_trip() {
    use crate::prelude::CompletionClient;

    let client = crate::providers::openai::Client::new_with(
        "dummy-key",
        crate::test_utils::RecordingHttpClient::new(""),
    )
    .expect("client")
    .with_system_instructions_placement(SystemInstructionsPlacement::InputSystemMessages)
    .completions_api()
    .responses_api();
    let model = client.completion_model("gpt-4o-mini");

    let req = model
        .create_completion_request(request_with_preamble("You are concise."))
        .expect("request should convert");
    let serialized = serde_json::to_value(&req).expect("request should serialize");

    assert!(
        serialized.get("instructions").is_none(),
        "placement configured before completions_api() should survive responses_api()"
    );
    assert_eq!(serialized["input"][0]["role"], "system");
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
fn responses_model_strict_tools_opt_in_sanitizes_all_function_tools() {
    let client = crate::providers::openai::Client::new_with(
        "dummy-key",
        crate::test_utils::RecordingHttpClient::new(""),
    )
    .expect("client");
    let model = ResponsesCompletionModel::new(client, "gpt-4o-mini")
        .with_strict_tools()
        .with_tool(completion::ToolDefinition {
            name: "lookup".to_string(),
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

    let req = model
        .create_completion_request(request)
        .expect("request should convert");

    assert_eq!(req.tools.len(), 3);
    for tool in &req.tools {
        assert!(tool.strict, "{} should be strict", tool.name);
        assert_eq!(tool.parameters["additionalProperties"], json!(false));
    }
}

#[test]
fn responses_model_default_preserves_all_function_tools_as_constructed() {
    let client = crate::providers::openai::Client::new_with(
        "dummy-key",
        crate::test_utils::RecordingHttpClient::new(""),
    )
    .expect("client");
    let model =
        ResponsesCompletionModel::new(client, "gpt-4o-mini").with_tool(weather_tool_definition());

    let mut request = weather_tool_request();
    request.additional_params = Some(json!({
        "tools": [{
            "type": "function",
            "name": "extra",
            "description": "An additional_params tool",
            "parameters": {"type": "object", "properties": {"x": {"type": "string"}}}
        }]
    }));

    let req = model
        .create_completion_request(request)
        .expect("request should convert");

    assert_eq!(req.tools.len(), 3);
    for tool in &req.tools {
        assert!(!tool.strict, "{} should not be strict", tool.name);
        assert!(tool.parameters.get("additionalProperties").is_none());
    }
}

#[test]
fn responses_explicit_strict_tool_stays_strict_on_default_model() {
    let client = crate::providers::openai::Client::new_with(
        "dummy-key",
        crate::test_utils::RecordingHttpClient::new(""),
    )
    .expect("client");
    let model = ResponsesCompletionModel::new(client, "gpt-4o-mini").with_tool(
        ResponsesToolDefinition::strict_function(
            "lookup",
            "Look something up",
            json!({"type": "object", "properties": {"q": {"type": "string"}}}),
        ),
    );

    let req = model
        .create_completion_request(weather_tool_request())
        .expect("request should convert");

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

#[test]
fn responses_request_keeps_documents_after_lifted_system_messages() {
    let request = CompletionRequestBuilder::new(MockCompletionModel::default(), "Prompt")
        .message(completion::Message::system("System prompt"))
        .message(completion::Message::user("Earlier user turn"))
        .message(completion::Message::assistant("Earlier assistant turn"))
        .document(test_document("doc1", "Document text."))
        .build();

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
    let request = crate::completion::CompletionRequest {
        model: None,
        chat_history: vec![
            completion::Message::system("System prompt"),
            completion::Message::assistant("Earlier assistant turn"),
            completion::Message::system("Mid-conversation instruction"),
            completion::Message::user("Prompt"),
        ],
        documents: vec![test_document("doc1", "Document text.")],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    };

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
        input_tokens_details: Some(InputTokensDetails { cached_tokens: 25 }),
        output_tokens: 50,
        output_tokens_details: Some(OutputTokensDetails {
            reasoning_tokens: 15,
        }),
        total_tokens: 150,
    };

    let token_usage = crate::completion::Usage::from(&usage);

    assert_eq!(token_usage.input_tokens, 100);
    assert_eq!(token_usage.cached_input_tokens, 25);
    assert_eq!(token_usage.output_tokens, 50);
    assert_eq!(token_usage.reasoning_tokens, 15);
    assert_eq!(token_usage.total_tokens, 150);
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

    assert_eq!(token_usage.input_tokens, 100);
    assert_eq!(token_usage.cached_input_tokens, 25);
    assert_eq!(token_usage.output_tokens, 50);
    assert_eq!(token_usage.reasoning_tokens, 0);
    assert_eq!(token_usage.total_tokens, 150);
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

    let completion: completion::CompletionResponse = response
        .normalize("openai")
        .expect("response should convert");
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

    let completion: completion::CompletionResponse = response
        .normalize("openai")
        .expect("reasoning-only response should convert");
    let items = completion.choice.iter().collect::<Vec<_>>();

    assert_eq!(items.len(), 1);
    assert!(matches!(
        items[0],
        completion::AssistantContent::Reasoning(_)
    ));
}

#[test]
fn completion_response_rejects_empty_response_without_reasoning() {
    let response: CompletionResponse = serde_json::from_value(json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "Qwen/Qwen3-4B",
        "output": [],
        "tools": []
    }))
    .expect("empty response shape should deserialize");

    let err = response
        .normalize("openai")
        .expect_err("empty response without reasoning should be rejected");

    assert!(
        err.to_string()
            .contains(crate::message::EMPTY_RESPONSE_ERROR)
    );
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

    let completion = response
        .normalize("openai")
        .expect("truncated incomplete response must not be an error");

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
fn completion_response_carries_the_message_id_not_the_response_id() {
    let response: CompletionResponse = serde_json::from_value(json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "gpt-5.4",
        "output": [{
            "type": "message",
            "id": "msg_456",
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

    let completion: completion::CompletionResponse = response
        .normalize("openai")
        .expect("response should convert");

    // The two IDs are distinct in this API: `resp_...` names the response,
    // `msg_...` names the assistant message.
    assert_eq!(completion.message_id.as_deref(), Some("msg_456"));
    assert_eq!(completion.provider, "openai");
    assert_eq!(completion.model.as_deref(), Some("gpt-5.4"));
    assert_eq!(
        completion.finish_reason(),
        Some(completion::FinishReason::Stop)
    );
}

#[test]
fn completion_response_provider_name_is_an_input() {
    let response: CompletionResponse = serde_json::from_value(json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "gpt-5.3-codex",
        "output": [{
            "type": "message",
            "id": "msg_456",
            "status": "completed",
            "role": "assistant",
            "content": [{ "type": "output_text", "annotations": [], "text": "done" }]
        }],
        "tools": []
    }))
    .expect("response should deserialize");

    let completion: completion::CompletionResponse = response
        .normalize("chatgpt")
        .expect("response should convert");

    assert_eq!(completion.provider, "chatgpt");
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

    let completion: completion::CompletionResponse = response
        .normalize("openai")
        .expect("response should convert");

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

    let completion: completion::CompletionResponse = response
        .normalize("openai")
        .expect("response should convert");

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

    let completion: completion::CompletionResponse = response
        .normalize("openai")
        .expect("response should convert");
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

    let completion: completion::CompletionResponse = response
        .normalize("openai")
        .expect("response should convert");
    let reasoning_count = completion
        .choice
        .iter()
        .filter(|item| matches!(item, completion::AssistantContent::Reasoning(_)))
        .count();

    assert_eq!(reasoning_count, 1);
}

#[test]
fn idless_reasoning_only_is_skipped_without_empty_input_item() {
    let assistant = completion::Message::Assistant {
        id: None,
        content: vec![message::AssistantContent::Reasoning(
            message::Reasoning::new("provider reasoning"),
        )],
    };

    let converted =
        Vec::<InputItem>::try_from(assistant).expect("idless reasoning should degrade gracefully");

    assert!(converted.is_empty());
}

#[test]
fn completion_history_idless_reasoning_plus_text_preserves_text_input_item() {
    let assistant = completion::Message::Assistant {
        id: Some("msg_123".to_string()),
        content: vec![
            message::AssistantContent::Reasoning(message::Reasoning::new("provider reasoning")),
            message::AssistantContent::Text(Text::new("final answer")),
        ],
    };

    let converted =
        Vec::<InputItem>::try_from(assistant).expect("assistant history should convert");

    assert_eq!(converted.len(), 1);
    assert!(matches!(converted[0].role, Some(Role::Assistant)));
    let InputContent::Message(Message::Assistant { content, .. }) = &converted[0].input else {
        panic!("expected assistant message input item");
    };
    assert!(matches!(
        content.first(),
        Some(AssistantContentType::Text(AssistantContent::OutputText(OutputText { text, .. }))) if text == "final answer"
    ));
}

#[test]
fn assistant_text_without_idless_reasoning_replays_as_output_text() {
    let assistant = completion::Message::Assistant {
        id: Some("msg_123".to_string()),
        content: vec![message::AssistantContent::Text(Text::new("final answer"))],
    };

    let converted =
        Vec::<InputItem>::try_from(assistant).expect("assistant history should convert");

    assert_eq!(converted.len(), 1);
    let InputContent::Message(Message::Assistant { content, .. }) = &converted[0].input else {
        panic!("expected assistant message input item");
    };
    assert!(matches!(
        content.first(),
        Some(AssistantContentType::Text(AssistantContent::OutputText(OutputText { text, .. }))) if text == "final answer"
    ));
}

#[test]
fn idless_completion_assistant_text_replays_as_easy_input_message() {
    let assistant = completion::Message::Assistant {
        id: None,
        content: vec![message::AssistantContent::Text(Text::new("final answer"))],
    };

    let converted =
        Vec::<InputItem>::try_from(assistant).expect("assistant history should convert");

    assert_eq!(converted.len(), 1);
    assert!(matches!(converted[0].role, Some(Role::Assistant)));
    let InputContent::Message(Message::AssistantInput { content, .. }) = &converted[0].input else {
        panic!("expected assistant input message item");
    };
    assert_eq!(content, "final answer");

    let serialized =
        serde_json::to_value(&converted[0]).expect("input item should serialize to JSON");
    assert_eq!(serialized["type"], json!("message"));
    assert_eq!(serialized["role"], json!("assistant"));
    assert_eq!(serialized["content"], json!("final answer"));
    assert!(serialized.get("id").is_none());
    assert!(serialized.get("status").is_none());
}

#[test]
fn structured_reasoning_with_id_still_converts_to_input_item() {
    let assistant = completion::Message::Assistant {
        id: Some("msg_123".to_string()),
        content: vec![message::AssistantContent::Reasoning(message::Reasoning {
            id: Some("rs_123".to_string()),
            content: vec![message::ReasoningContent::Summary(
                "structured summary".to_string(),
            )],
        })],
    };

    let converted =
        Vec::<InputItem>::try_from(assistant).expect("structured reasoning should convert");

    assert_eq!(converted.len(), 1);
    assert!(converted[0].role.is_none());
    assert!(matches!(
        &converted[0].input,
        InputContent::Reasoning(OpenAIReasoning { id, .. }) if id == "rs_123"
    ));
}

#[test]
fn assistant_reasoning_text_tool_call_convert_in_responses_replay_order() {
    let assistant = completion::Message::Assistant {
        id: Some("msg_123".to_string()),
        content: vec![
            message::AssistantContent::Reasoning(message::Reasoning {
                id: Some("rs_123".to_string()),
                content: vec![message::ReasoningContent::Summary(
                    "structured summary".to_string(),
                )],
            }),
            message::AssistantContent::Text(Text::new("final answer")),
            message::AssistantContent::tool_call_with_call_id(
                "fc_123",
                "call_123".to_string(),
                "lookup",
                json!({"query": "rig"}),
            ),
        ],
    };

    let converted =
        Vec::<InputItem>::try_from(assistant).expect("assistant history should convert");

    assert_eq!(converted.len(), 3);
    assert!(converted[0].role.is_none());
    assert!(matches!(
        &converted[0].input,
        InputContent::Reasoning(OpenAIReasoning { id, .. }) if id == "rs_123"
    ));

    assert!(matches!(converted[1].role, Some(Role::Assistant)));
    let InputContent::Message(Message::Assistant { content, id, .. }) = &converted[1].input else {
        panic!("expected assistant output message");
    };
    assert_eq!(id, "msg_123");
    assert!(matches!(
        content.first(),
        Some(AssistantContentType::Text(AssistantContent::OutputText(OutputText { text, .. })))
            if text == "final answer"
    ));

    assert!(converted[2].role.is_none());
    let InputContent::FunctionCall(OutputFunctionCall {
        id, call_id, name, ..
    }) = &converted[2].input
    else {
        panic!("expected function call input item");
    };
    assert_eq!(id, "fc_123");
    assert_eq!(call_id, "call_123");
    assert_eq!(name, "lookup");
}

#[test]
fn mocked_second_turn_request_omits_unreplayable_reasoning() {
    let request = crate::completion::CompletionRequest {
        model: None,
        chat_history: vec![
            completion::Message::system("You are concise."),
            completion::Message::User {
                content: vec![message::UserContent::Text(Text::new(
                    "Think briefly, then answer.",
                ))],
            },
            completion::Message::Assistant {
                id: Some("msg_123".to_string()),
                content: vec![
                    message::AssistantContent::Reasoning(message::Reasoning::new(
                        "provider reasoning",
                    )),
                    message::AssistantContent::Text(Text::new("final answer")),
                ],
            },
            completion::Message::Assistant {
                id: None,
                content: vec![
                    message::AssistantContent::Reasoning(message::Reasoning::new(
                        "provider reasoning only",
                    )),
                    message::AssistantContent::Text(Text::new("")),
                ],
            },
            completion::Message::User {
                content: vec![message::UserContent::Text(Text::new(
                    "/no_think Reply with exactly: OK",
                ))],
            },
        ],
        documents: Vec::new(),
        tools: Vec::new(),
        temperature: None,
        max_tokens: Some(64),
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    };

    let request = CompletionRequest::try_from(("Qwen/Qwen3-4B".to_string(), request))
        .expect("request should convert");
    let value = serde_json::to_value(&request).expect("request should serialize");
    let input = value["input"]
        .as_array()
        .expect("mocked multi-turn request should serialize input as an array");

    assert!(
        !input.iter().any(|item| {
            item.get("type") == Some(&json!("reasoning")) && item.get("id").is_none()
        })
    );
    assert!(!input.iter().any(|item| {
        item.get("role") == Some(&json!("assistant"))
            && item
                .get("content")
                .and_then(Value::as_array)
                .is_some_and(Vec::is_empty)
    }));

    let assistant_items = input
        .iter()
        .filter(|item| item.get("role") == Some(&json!("assistant")))
        .collect::<Vec<_>>();

    assert_eq!(assistant_items.len(), 1);
    assert_eq!(assistant_items[0]["content"][0]["type"], "output_text");
    assert_eq!(assistant_items[0]["content"][0]["text"], "final answer");
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
        input_tokens_details: Some(InputTokensDetails { cached_tokens: 2 }),
        output_tokens: 5,
        output_tokens_details: Some(OutputTokensDetails {
            reasoning_tokens: 4,
        }),
        total_tokens: 8,
    };

    let usage = lhs + rhs;
    let token_usage = crate::completion::Usage::from(&usage);

    assert_eq!(token_usage.input_tokens, 13);
    assert_eq!(token_usage.cached_input_tokens, 2);
    assert_eq!(token_usage.output_tokens, 25);
    assert_eq!(token_usage.reasoning_tokens, 4);
    assert_eq!(token_usage.total_tokens, 38);
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

    let converted: Vec<InputItem> = message.try_into().expect("conversion should succeed");
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
    use crate::client::CompletionClient;
    use crate::completion::CompletionModel;
    use crate::providers::openai::Client;
    use crate::test_utils::RecordingHttpClient;

    let body = r#"{"error":{"message":"bad image","type":"invalid_request_error","code":"invalid_value"}}"#;
    let http_client = RecordingHttpClient::with_error_response(http::StatusCode::BAD_REQUEST, body);
    let client = Client::builder()
        .api_key("test-key")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.completion_model("gpt-4o-mini");
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
        status: Some(ToolStatus::Completed),
    };

    let value = serde_json::to_value(&original).expect("reasoning should serialize");
    let round_tripped: Output =
        serde_json::from_value(value).expect("reasoning should deserialize");

    assert_eq!(round_tripped, original);
}

#[test]
fn output_reasoning_conversion_omits_empty_encrypted_content() {
    let output = Output::Reasoning {
        id: "reasoning_1".to_string(),
        summary: vec![],
        content: vec!["visible reasoning".to_string()],
        encrypted_content: Some(String::new()),
        status: Some(ToolStatus::Completed),
    };

    let converted = Vec::<completion::AssistantContent>::from(output);

    assert_eq!(converted.len(), 1);
    let completion::AssistantContent::Reasoning(reasoning) = &converted[0] else {
        panic!("expected reasoning output");
    };
    assert_eq!(reasoning.id.as_deref(), Some("reasoning_1"));
    assert_eq!(reasoning.content.len(), 1);
    assert!(matches!(
        reasoning.content.first(),
        Some(message::ReasoningContent::Text { text, .. })
            if text == "visible reasoning"
    ));
}

#[test]
fn output_reasoning_conversion_preserves_non_empty_encrypted_content() {
    let output = Output::Reasoning {
        id: "reasoning_1".to_string(),
        summary: vec![],
        content: vec![],
        encrypted_content: Some("ciphertext".to_string()),
        status: Some(ToolStatus::Completed),
    };

    let converted = Vec::<completion::AssistantContent>::from(output);

    assert_eq!(converted.len(), 1);
    let completion::AssistantContent::Reasoning(reasoning) = &converted[0] else {
        panic!("expected reasoning output");
    };
    assert_eq!(
        reasoning.content,
        vec![message::ReasoningContent::Encrypted(
            "ciphertext".to_string()
        )]
    );
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
// that `CompletionModel::completion()` requests actually go through.
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
    let items = Vec::<InputItem>::try_from(url_pdf_message())
        .expect("URL PDF should convert to input items");
    let json = serde_json::to_value(&items).expect("input items should serialize");
    assert_url_only_input_file(&sole_input_file(&json));
}

#[test]
fn url_pdf_in_full_completion_request_omits_filename() {
    let core_request = crate::completion::CompletionRequest {
        model: None,
        chat_history: vec![url_pdf_message()],
        documents: Vec::new(),
        tools: Vec::new(),
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    };

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

    let items =
        Vec::<InputItem>::try_from(input).expect("base64 PDF should convert to input items");
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

/// Raw-capture tests: the `normalize` shape through the Responses model,
/// driven end to end over a mock transport that hands back a Responses
/// body *and* an `x-request-id` response header. The Responses raw type
/// carries the transport id (`CompletionResponse::provider_request_id`,
/// stamped by the driver), which is why the Part A contract here is a
/// plain `raw_completion` → `normalize`. Its manual `Serialize` mirrors
/// the wire body and deliberately never emits that id, so the captured
/// value is the body as parsed — the transport id lives on the normalized
/// response, beside the capture, not inside it. `with_error_response_headers`
/// with `200 OK` is the one unary double that carries response headers.
mod raw_capture {
    use super::*;
    use crate::client::CompletionClient;
    use crate::completion::CompletionModel as _;
    use crate::providers::openai::Client;
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

    fn model() -> ResponsesCompletionModel<RecordingHttpClient> {
        let mut headers = http::HeaderMap::new();
        headers.insert("x-request-id", http::HeaderValue::from_static(REQUEST_ID));
        let http_client =
            RecordingHttpClient::with_error_response_headers(http::StatusCode::OK, BODY, headers);
        let client = Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        client.completion_model("gpt-4o-mini")
    }

    /// The load-bearing capture property: `raw` is the Responses
    /// `CompletionResponse` as rig parsed it — it deserializes back into
    /// that type and re-serializes to the identical value — and
    /// re-normalizing that capture (with the header id reattached, since
    /// the capture is body only) reproduces every normalized field. Also
    /// reads `service_tier` off the capture,
    /// and pins that the capture mirrors the wire body: the transport id
    /// the driver stamped onto the raw type is not part of it (the manual
    /// `Serialize` never emits it), so a value deserialized from `raw`
    /// reports `None` there while the normalized response beside it still
    /// carries the header.
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
        assert!(matches!(
            typed.additional_parameters.service_tier,
            Some(OpenAIServiceTier::Default)
        ));
        assert_eq!(raw["service_tier"], "default");
        assert!(raw.get("provider_request_id").is_none());
        assert_eq!(typed.provider_request_id, None);

        let renormalized = typed
            .normalize(
                <crate::providers::openai::OpenAIResponses as ResponsesProviderExt>::PROVIDER_NAME,
            )
            .expect("re-normalize the capture")
            .with_optional_provider_request_id(Some(REQUEST_ID.to_string()));
        assert_eq!(response.identity(), renormalized.identity());
        assert_eq!(response.finish_reason(), renormalized.finish_reason());
        assert_eq!(response.model, renormalized.model);
        assert_eq!(response.usage, renormalized.usage);
        assert_eq!(response.choice, renormalized.choice);
        assert_eq!(response.provider_request_id.as_deref(), Some(REQUEST_ID));
        assert_eq!(response.identity().message_id.as_deref(), Some("msg_raw_1"));
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
            .normalize(
                <crate::providers::openai::OpenAIResponses as ResponsesProviderExt>::PROVIDER_NAME,
            )
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
