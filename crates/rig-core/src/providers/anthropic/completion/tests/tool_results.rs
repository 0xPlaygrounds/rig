use super::*;
use message::{
    CallId, Image, ImageMediaType, ToolName, ToolResult, ToolResultContent, UserContent,
};

fn result(id: &str, is_error: bool, content: Vec<ToolResultContent>) -> UserContent {
    let Ok(name) = ToolName::new("shot") else {
        panic!("valid tool name");
    };
    UserContent::ToolResult(ToolResult {
        call: CallId::from_wire(id),
        name,
        content,
        is_error,
    })
}

fn blocks(content: Vec<UserContent>) -> Vec<Value> {
    let Ok(Some(value)) = convert(message::Message::User { content }) else {
        panic!("user message must encode");
    };
    let Some(content) = value.get("content").and_then(Value::as_array) else {
        panic!("user content must be an array");
    };
    content.clone()
}

fn screenshot() -> ToolResultContent {
    ToolResultContent::Image(Image {
        data: message::DocumentSourceKind::base64("aGk="),
        media_type: Some(ImageMediaType::PNG),
        ..Image::default()
    })
}

#[test]
fn empty_tool_results_use_strings_and_errors_have_a_fallback() {
    for content in [
        vec![],
        vec![ToolResultContent::text("")],
        vec![ToolResultContent::text(" \n\t")],
    ] {
        for is_error in [false, true] {
            let sent = blocks(vec![result("toolu_empty", is_error, content.clone())]);
            assert_eq!(sent.len(), 1);
            assert_eq!(
                sent.first().and_then(|block| block.get("content")),
                Some(&json!(if is_error {
                    "Tool failed with no output."
                } else {
                    ""
                }))
            );
            assert_eq!(
                sent.first().and_then(|block| block.get("is_error")),
                is_error.then_some(&Value::Bool(true))
            );
        }
    }
}

#[test]
fn error_images_follow_all_results_and_success_images_stay_nested() {
    let sent = blocks(vec![
        result(
            "toolu_error",
            true,
            vec![ToolResultContent::text("assertion failed"), screenshot()],
        ),
        result(
            "toolu_ok",
            false,
            vec![ToolResultContent::text("screenshot"), screenshot()],
        ),
    ]);
    assert_eq!(
        sent,
        vec![
            json!({"type": "tool_result", "tool_use_id": "toolu_error", "is_error": true,
            "content": [{"type": "text", "text": "assertion failed"}]}),
            json!({"type": "tool_result", "tool_use_id": "toolu_ok",
            "content": [{"type": "text", "text": "screenshot"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGk="}}]}),
            json!({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGk="}}),
        ]
    );
}

#[test]
fn image_only_errors_keep_a_nonempty_result_and_the_image() {
    let sent = blocks(vec![result("toolu_image", true, vec![screenshot()])]);
    assert_eq!(
        sent,
        vec![
            json!({"type": "tool_result", "tool_use_id": "toolu_image", "is_error": true,
            "content": "Tool failed with no output."}),
            json!({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGk="}}),
        ]
    );
}
