use crate::message::ImageMediaType;

use super::*;

#[test]
fn debug_reports_shape_without_tool_content() {
    let output = ToolOutput::content(vec![
        ToolResultContent::text("Bearer secret-tool-output"),
        ToolResultContent::json(serde_json::json!({
            "credential": "secret-json-output"
        })),
        ToolResultContent::image_base64("secret-image-output", Some(ImageMediaType::PNG), None),
    ])
    .expect("fixture content is non-empty");

    let debug = format!("{output:?}");
    assert!(debug.contains("content_count: 3"));
    assert!(debug.contains("text"));
    assert!(debug.contains("json"));
    assert!(debug.contains("image"));
    for secret in [
        "secret-tool-output",
        "secret-json-output",
        "secret-image-output",
    ] {
        assert!(!debug.contains(secret));
    }
}
