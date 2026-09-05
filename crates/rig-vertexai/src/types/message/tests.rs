use super::*;
use crate::types::completion_response::VertexGenerateContentOutput;
use google_cloud_aiplatform_v1 as vertexai;
use rig_core::completion::CompletionResponse;
use rig_core::message::{Message, Text, ToolCallId, ToolResult, ToolResultContent};

#[test]
fn test_user_text_message_conversion() {
    let message = Message::User {
        content: vec![rig_core::message::UserContent::Text(Text::new(
            "Hello".to_string(),
        ))],
    };

    let rig_message = RigMessage(message);
    let vertex_content: Result<vertexai::model::Content, _> = rig_message.try_into();

    assert!(vertex_content.is_ok());
    let content = vertex_content.unwrap();
    assert_eq!(content.role.as_str(), "user");
    assert_eq!(content.parts.len(), 1);
    assert_eq!(content.parts[0].text(), Some(&"Hello".to_string()));
}

#[test]
fn test_assistant_text_message_conversion() {
    let message = Message::Assistant {
        id: None,
        content: vec![AssistantContent::Text(Text::new("Hi there".to_string()))],
    };

    let rig_message = RigMessage(message);
    let vertex_content: Result<vertexai::model::Content, _> = rig_message.try_into();

    assert!(vertex_content.is_ok());
    let content = vertex_content.unwrap();
    assert_eq!(content.role.as_str(), "model");
    assert_eq!(content.parts.len(), 1);
    assert_eq!(content.parts[0].text(), Some(&"Hi there".to_string()));
}

#[test]
fn test_assistant_image_response_round_trips_through_history_in_order() {
    let raw_image = vec![0, 1, 2, 255];
    let response = vertexai::model::GenerateContentResponse::new().set_candidates([
        vertexai::model::Candidate::new().set_content(
            vertexai::model::Content::new()
                .set_role("model")
                .set_parts([
                    vertexai::model::Part::new().set_text("before"),
                    vertexai::model::Part::new().set_inline_data(
                        vertexai::model::Blob::new()
                            .set_mime_type("image/png")
                            .set_data(raw_image.clone()),
                    ),
                    vertexai::model::Part::new().set_text("after"),
                ]),
        ),
    ]);
    let response: CompletionResponse = VertexGenerateContentOutput(response)
        .try_into()
        .expect("image response should convert");

    let content: vertexai::model::Content = RigMessage(Message::Assistant {
        id: None,
        content: response.choice,
    })
    .try_into()
    .expect("assistant history image should convert");

    assert_eq!(content.parts.len(), 3);
    assert_eq!(content.parts[0].text().map(String::as_str), Some("before"));
    let image = content.parts[1]
        .inline_data()
        .expect("middle part should be an inline image");
    assert_eq!(image.mime_type, "image/png");
    assert_eq!(image.data.as_ref(), raw_image.as_slice());
    assert_eq!(content.parts[2].text().map(String::as_str), Some("after"));
}

#[test]
fn test_assistant_image_history_rejects_invalid_or_unsupported_input() {
    let cases = [
        (
            AssistantContent::image_base64(BASE64.encode([1]), None, None),
            "Media type",
        ),
        (
            AssistantContent::image_base64(BASE64.encode([1]), Some(ImageMediaType::GIF), None),
            "Unsupported",
        ),
        (
            AssistantContent::image_base64("not valid base64", Some(ImageMediaType::PNG), None),
            "Invalid base64",
        ),
    ];

    for (image, expected_message) in cases {
        let result: Result<vertexai::model::Content, CompletionError> =
            RigMessage(Message::Assistant {
                id: None,
                content: vec![image],
            })
            .try_into();
        let Err(error) = result else {
            panic!("invalid assistant image must fail")
        };
        assert!(matches!(error, CompletionError::RequestError(_)));
        assert!(error.to_string().contains(expected_message));
    }
}

#[test]
fn test_assistant_tool_call_message_conversion() {
    use rig_core::message::{ToolCall, ToolFunction};
    // Vertex issues no call ids, so decoded calls carry a minted id and
    // no provider id; neither reaches the outbound wire.
    let tool_call = ToolCall::new(
        ToolCallId::minted(0),
        ToolFunction::new(
            "add".to_string(),
            serde_json::json!({
                "x": 5,
                "y": 3
            }),
        ),
    );

    let message = Message::Assistant {
        id: None,
        content: vec![AssistantContent::ToolCall(tool_call)],
    };

    let rig_message = RigMessage(message);
    let vertex_content: Result<vertexai::model::Content, _> = rig_message.try_into();

    assert!(vertex_content.is_ok());
    let content = vertex_content.unwrap();
    assert_eq!(content.role.as_str(), "model");
    assert_eq!(content.parts.len(), 1);

    let function_call = content.parts[0].function_call();
    assert!(function_call.is_some());
    let function_call = function_call.unwrap();
    assert_eq!(function_call.name.as_str(), "add");
}

#[test]
fn test_assistant_tool_call_echoes_thought_signature() {
    use rig_core::message::{ToolCall, ToolFunction};
    let raw = b"\x00\x01\x02thinking-sig\xff";
    let tool_call = ToolCall::new(
        ToolCallId::minted(0),
        ToolFunction::new("add".to_string(), serde_json::json!({"x": 5})),
    )
    .with_signature(Some(BASE64.encode(raw)));
    let content: vertexai::model::Content = RigMessage(Message::Assistant {
        id: None,
        content: vec![AssistantContent::ToolCall(tool_call)],
    })
    .try_into()
    .unwrap();
    assert_eq!(content.parts[0].thought_signature.as_ref(), raw.as_slice());
}

#[test]
fn test_assistant_tool_call_malformed_signature_is_dropped_not_fatal() {
    // A malformed signature must not abort the whole turn — it is dropped with a warning.
    use rig_core::message::{ToolCall, ToolFunction};
    let tool_call = ToolCall::new(
        ToolCallId::minted(0),
        ToolFunction::new("add".to_string(), serde_json::json!({"x": 5})),
    )
    .with_signature(Some("!!! not base64 !!!".to_string()));
    let content: vertexai::model::Content = RigMessage(Message::Assistant {
        id: None,
        content: vec![AssistantContent::ToolCall(tool_call)],
    })
    .try_into()
    .expect("malformed signature should not fail the conversion");
    assert_eq!(content.parts.len(), 1);
    assert!(content.parts[0].thought_signature.is_empty());
    assert!(content.parts[0].function_call().is_some());
}

#[test]
fn test_assistant_reasoning_echoes_thought_signature() {
    let raw = b"\x00\x01\x02thinking-text-sig\xff";
    let reasoning =
        rig_core::message::Reasoning::new_with_signature("thinking text", Some(BASE64.encode(raw)));

    let content: vertexai::model::Content = RigMessage(Message::Assistant {
        id: None,
        content: vec![AssistantContent::Reasoning(reasoning)],
    })
    .try_into()
    .unwrap();

    assert_eq!(content.parts.len(), 1);
    assert_eq!(
        content.parts[0].text().map(String::as_str),
        Some("thinking text")
    );
    assert!(content.parts[0].thought);
    assert_eq!(content.parts[0].thought_signature.as_ref(), raw.as_slice());
}

#[test]
fn test_assistant_reasoning_malformed_signature_is_dropped_not_fatal() {
    let reasoning = rig_core::message::Reasoning::new_with_signature(
        "thinking text",
        Some("!!! not base64 !!!".to_string()),
    );

    let content: vertexai::model::Content = RigMessage(Message::Assistant {
        id: None,
        content: vec![AssistantContent::Reasoning(reasoning)],
    })
    .try_into()
    .expect("malformed signature should not fail the conversion");

    assert_eq!(content.parts.len(), 1);
    assert!(content.parts[0].thought);
    assert!(content.parts[0].thought_signature.is_empty());
}

#[test]
fn test_user_tool_result_message_conversion() {
    // Vertex results are id-less on the wire: a minted correlation
    // handle, no provider id, and the required executed-tool name that
    // becomes `functionResponse.name`.
    let tool_result = ToolResult {
        call: ToolCallId::minted(0),
        provider: None,
        name: "add".to_string(),
        content: vec![ToolResultContent::Text(Text::new("8".to_string()))],
    };

    let message = Message::User {
        content: vec![rig_core::message::UserContent::ToolResult(tool_result)],
    };

    let rig_message = RigMessage(message);
    let vertex_content: Result<vertexai::model::Content, _> = rig_message.try_into();

    assert!(vertex_content.is_ok());
    let content = vertex_content.unwrap();
    assert_eq!(content.role.as_str(), "user");
    assert_eq!(content.parts.len(), 1);

    let function_response = content.parts[0].function_response();
    assert!(function_response.is_some());
    let function_response = function_response.unwrap();
    assert_eq!(function_response.name.as_str(), "add");
    assert_eq!(
        function_response
            .response
            .as_ref()
            .and_then(|response| response.get("output")),
        Some(&serde_json::Value::String("8".to_string()))
    );
}

#[test]
fn structured_tool_result_stays_structured_at_the_vertex_boundary() {
    let value = serde_json::json!({ "answer": 8 });
    let message = Message::User {
        content: vec![rig_core::message::UserContent::ToolResult(ToolResult {
            call: ToolCallId::minted(0),
            provider: None,
            name: "lookup".to_string(),
            content: vec![ToolResultContent::json(value.clone())],
        })],
    };

    let content: vertexai::model::Content = RigMessage(message)
        .try_into()
        .expect("tool result should convert");
    let response = content.parts[0]
        .function_response()
        .expect("function response");
    assert_eq!(
        response
            .response
            .as_ref()
            .and_then(|response| response.get("output")),
        Some(&value)
    );
}

#[test]
fn image_tool_result_maps_to_native_function_response_part() {
    let raw = vec![0, 1, 2, 255];
    let message = Message::User {
        content: vec![rig_core::message::UserContent::ToolResult(ToolResult {
            call: ToolCallId::minted(0),
            provider: None,
            name: "inspect".to_string(),
            content: vec![ToolResultContent::image_base64(
                BASE64.encode(&raw),
                Some(ImageMediaType::PNG),
                None,
            )],
        })],
    };

    let content: vertexai::model::Content = RigMessage(message)
        .try_into()
        .expect("image tool result should convert");
    let response = content.parts[0]
        .function_response()
        .expect("function response");
    assert_eq!(response.parts.len(), 1);
    let blob = response.parts[0].inline_data().expect("inline image");
    assert_eq!(blob.mime_type, "image/png");
    assert_eq!(blob.data.as_ref(), raw.as_slice());
    assert_eq!(blob.display_name, "rig_tool_result_image_0");
    assert_eq!(
        response
            .response
            .as_ref()
            .and_then(|response| response.get("output")),
        Some(&serde_json::json!({ "$ref": "rig_tool_result_image_0" }))
    );
}

#[test]
fn mixed_tool_result_preserves_structured_and_media_order() {
    let content = vec![
        ToolResultContent::text("before"),
        ToolResultContent::image_raw(vec![1, 2, 3], Some(ImageMediaType::JPEG), None),
        ToolResultContent::json(serde_json::json!({ "after": true })),
        ToolResultContent::image_url("gs://bucket/result.png", Some(ImageMediaType::PNG), None),
    ];
    let message = Message::User {
        content: vec![rig_core::message::UserContent::ToolResult(ToolResult {
            call: ToolCallId::minted(0),
            provider: None,
            name: "inspect".to_string(),
            content,
        })],
    };

    let content: vertexai::model::Content = RigMessage(message)
        .try_into()
        .expect("mixed tool result should convert");
    let response = content.parts[0]
        .function_response()
        .expect("function response");
    assert_eq!(
        response
            .response
            .as_ref()
            .and_then(|response| response.get("output")),
        Some(&serde_json::json!([
            "before",
            { "$ref": "rig_tool_result_image_0" },
            { "after": true },
            { "$ref": "rig_tool_result_image_1" }
        ]))
    );
    assert_eq!(response.parts.len(), 2);
    assert_eq!(
        response.parts[0]
            .inline_data()
            .expect("first media part")
            .data
            .as_ref(),
        &[1, 2, 3]
    );
    assert_eq!(
        response.parts[0]
            .inline_data()
            .expect("first media part")
            .display_name,
        "rig_tool_result_image_0"
    );
    assert_eq!(
        response.parts[1]
            .file_data()
            .expect("second media part")
            .file_uri,
        "gs://bucket/result.png"
    );
    assert_eq!(
        response.parts[1]
            .file_data()
            .expect("second media part")
            .display_name,
        "rig_tool_result_image_1"
    );
}

#[test]
fn tool_result_image_refs_avoid_names_reserved_by_structured_json() {
    let content = vec![
        ToolResultContent::json(serde_json::json!({
            "literal": { "$ref": "rig_tool_result_image_0" }
        })),
        ToolResultContent::image_raw(vec![1, 2, 3], Some(ImageMediaType::PNG), None),
    ];
    let message = Message::User {
        content: vec![rig_core::message::UserContent::ToolResult(ToolResult {
            call: ToolCallId::minted(0),
            provider: None,
            name: "inspect".to_string(),
            content,
        })],
    };

    let content: vertexai::model::Content = RigMessage(message)
        .try_into()
        .expect("colliding ref should be avoided");
    let response = content.parts[0]
        .function_response()
        .expect("function response");
    assert_eq!(
        response
            .response
            .as_ref()
            .and_then(|response| response.get("output")),
        Some(&serde_json::json!([
            { "literal": { "$ref": "rig_tool_result_image_0" } },
            { "$ref": "rig_tool_result_image_1" }
        ]))
    );
    assert_eq!(
        response.parts[0]
            .inline_data()
            .expect("image part")
            .display_name,
        "rig_tool_result_image_1"
    );
}

#[test]
fn unsupported_tool_result_image_media_type_is_rejected_locally() {
    let message = Message::User {
        content: vec![rig_core::message::UserContent::ToolResult(ToolResult {
            call: ToolCallId::minted(0),
            provider: None,
            name: "inspect".to_string(),
            content: vec![ToolResultContent::image_raw(
                vec![1, 2, 3],
                Some(ImageMediaType::GIF),
                None,
            )],
        })],
    };

    let error = vertexai::model::Content::try_from(RigMessage(message))
        .expect_err("unsupported media must fail before the provider request");
    assert!(
        error.to_string().contains("expected JPEG, PNG, or WEBP"),
        "unexpected conversion error: {error}"
    );
}
