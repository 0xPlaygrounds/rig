use crate::message::{DocumentSourceKind, ImageMediaType};

use super::*;

#[test]
fn an_empty_content_list_cannot_become_a_tool_output() {
    // A zero-block tool result cannot be sent — the request boundary
    // rejects it — so the failure surfaces at construction as an ordinary
    // tool error instead of aborting the run one request later. Every
    // route is closed: the rich-content tool return, the explicit
    // constructor, and the fallible conversion. One empty text block, by
    // contrast, is a legitimate empty result and passes.
    let error = Vec::<ToolResultContent>::new()
        .into_tool_output()
        .expect_err("an empty rich-content list must not become a ToolOutput");
    assert!(error.to_string().contains("no content blocks"));

    assert!(ToolOutput::content(Vec::new()).is_err());
    assert!(ToolOutput::try_from(Vec::<ToolResultContent>::new()).is_err());

    let output = vec![ToolResultContent::text("")]
        .into_tool_output()
        .unwrap();
    assert_eq!(output, ToolOutput::text(""));
}

#[test]
fn json_shaped_strings_remain_literal_text() {
    let text = r#"{"type":"image","data":"not-an-envelope"}"#.to_string();
    let output = text.clone().into_tool_output().unwrap();

    assert_eq!(output, ToolOutput::text(text.clone()));
    let content = output.into_content();
    assert!(matches!(content.first(), Some(ToolResultContent::Text(value)) if value.text == text));
}

#[test]
fn structured_values_remain_json_until_terminal_rendering() {
    let value = serde_json::json!({"status": "ok", "count": 2});
    let output = value.clone().into_tool_output().unwrap();

    assert_eq!(output, ToolOutput::json(value.clone()));
    assert_eq!(output.render(), value.to_string());
    let content = output.into_content();
    assert!(matches!(
        content.first(),
        Some(ToolResultContent::Json { value: content_value }) if *content_value == value
    ));
}

#[test]
fn explicit_json_string_is_distinct_from_literal_text() {
    let explicit = serde_json::Value::String("hello".to_string());

    let json_output = explicit.clone().into_tool_output().unwrap();
    let text_output = "hello".to_string().into_tool_output().unwrap();

    assert_eq!(json_output, ToolOutput::json(explicit.clone()));
    assert_eq!(json_output.as_json(), Some(&explicit));
    assert_eq!(json_output.as_text(), None);
    assert_eq!(text_output, ToolOutput::text("hello"));
    assert_eq!(text_output.as_text(), Some("hello"));
}

#[test]
fn explicit_image_content_preserves_its_type() {
    let image = ToolResultContent::image_base64("base64data==", Some(ImageMediaType::JPEG), None);
    let output = image.into_tool_output().unwrap();

    let content = output.into_content();
    assert!(matches!(
        content.first(),
        Some(ToolResultContent::Image(image))
            if image.media_type == Some(ImageMediaType::JPEG)
                && matches!(&image.data, DocumentSourceKind::Base64(data) if data == "base64data==")
    ));
}

#[test]
fn direct_ordered_content_is_not_serialized_as_json() {
    let content = vec![
        ToolResultContent::text("before"),
        ToolResultContent::image_base64("base64data==", Some(ImageMediaType::PNG), None),
        ToolResultContent::json(serde_json::json!({"after": true})),
    ];

    let output = content.clone().into_tool_output().unwrap();

    assert_eq!(output.as_content(), &content);
}

#[test]
fn singleton_plain_content_has_one_canonical_representation() {
    assert_eq!(
        ToolOutput::text("hello"),
        ToolOutput::one(ToolResultContent::text("hello"))
    );
    assert_eq!(
        ToolOutput::json(serde_json::json!({"ok": true})),
        ToolOutput::one(ToolResultContent::json(serde_json::json!({"ok": true})))
    );
}
