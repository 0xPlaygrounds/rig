use aws_sdk_bedrockruntime::types as aws_bedrock;
use base64::{Engine, prelude::BASE64_STANDARD};
use rig_core::message::{DocumentSourceKind, Image, ImageMediaType, Text, ToolResultContent};

use crate::types::{json, tool};

#[test]
fn rig_tool_text_to_aws_tool() {
    let content = ToolResultContent::Text(Text::new("42"));
    let aws_tool = tool::to_aws(content).unwrap().unwrap();
    assert_eq!(aws_tool.as_text().unwrap(), "42");
}

/// Converse rejects blank text, so a blank tool-result text sends nothing.
#[test]
fn blank_tool_text_sends_nothing() {
    assert!(
        tool::to_aws(ToolResultContent::text(" \n"))
            .unwrap()
            .is_none()
    );
}

#[test]
fn rig_tool_image_to_aws_tool() {
    let encoded_str = BASE64_STANDARD.encode("img_data");
    let image = Image {
        data: DocumentSourceKind::Base64(encoded_str),
        media_type: Some(ImageMediaType::JPEG),
        detail: None,
        native: None,
    };
    let content = ToolResultContent::Image(image);
    assert!(tool::to_aws(content).unwrap().unwrap().is_image());
}

#[test]
fn rig_tool_json_maps_to_native_aws_json() {
    let expected = serde_json::json!({ "answer": -3, "exact": true });
    let content = ToolResultContent::Json {
        value: expected.clone(),
    };

    let aws_tool: aws_bedrock::ToolResultContentBlock = tool::to_aws(content)
        .expect("JSON should render at the AWS boundary")
        .expect("JSON is never blank");
    let document = match aws_tool {
        aws_bedrock::ToolResultContentBlock::Json(document) => document,
        other => panic!("expected Bedrock JSON tool result, got {other:?}"),
    };
    let actual = json::to_value(document);
    assert_eq!(actual, expected);
}

#[test]
fn rig_tool_non_object_json_is_wrapped_for_bedrock() {
    for value in [
        serde_json::Value::Null,
        serde_json::json!(true),
        serde_json::json!(-3),
        serde_json::json!("literal text"),
        serde_json::json!([1, 2, 3]),
    ] {
        let content = ToolResultContent::Json {
            value: value.clone(),
        };

        let aws_tool: aws_bedrock::ToolResultContentBlock = tool::to_aws(content)
            .expect("JSON should render at the AWS boundary")
            .expect("JSON is never blank");
        let document = match aws_tool {
            aws_bedrock::ToolResultContentBlock::Json(document) => document,
            other => panic!("expected Bedrock JSON tool result, got {other:?}"),
        };
        let actual = json::to_value(document);
        assert_eq!(actual, serde_json::json!({ "result": value }));
    }
}
