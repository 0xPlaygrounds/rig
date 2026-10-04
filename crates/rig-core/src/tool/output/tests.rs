use crate::message::ImageMediaType;

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
fn direct_ordered_content_is_not_serialized_as_json() {
    let content = vec![
        ToolResultContent::text("before"),
        ToolResultContent::image_base64("base64data==", Some(ImageMediaType::PNG), None),
        ToolResultContent::json(serde_json::json!({"after": true})),
    ];

    let output = content.clone().into_tool_output().unwrap();

    assert_eq!(output.as_content(), &content);
}
