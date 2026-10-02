//! Images in `role:"tool"` messages. Official OpenAI Chat Completions
//! answers 400 on `gpt-4o` and drops the image on `gpt-5`; llama.cpp reads
//! it. The adapter moves an image out of a result for a dialect that reads
//! none ([`Quirks::supports_image_tool_results`]), so the encoder only
//! carries one where it is read.
//!
//! [`Quirks::supports_image_tool_results`]: crate::providers::openai::wire::Quirks::supports_image_tool_results

use super::*;
use crate::message;

fn params(content: Vec<message::ToolResultContent>) -> OpenAIRequestParams {
    OpenAIRequestParams {
        model: "test-model".to_string(),
        request: crate::completion::CompletionRequest::new(message::Message::User {
            content: vec![message::UserContent::ToolResult(message::ToolResult {
                is_error: false,
                call: crate::message::CallId::from_wire("call_1"),
                name: crate::message::ToolName::new("view_file".to_string()).expect("tool name"),
                content,
            })],
        }),
        strict_tools: false,
        tool_result_array_content: false,
        supports_tools: true,
        supports_response_format: true,
        response_format_with_tools: false,
    }
}

fn image() -> message::ToolResultContent {
    message::ToolResultContent::image_base64(
        "iVBORw0KGgo=",
        Some(message::ImageMediaType::PNG),
        None,
    )
}

/// An image in a result reaches the wire.
#[test]
fn an_image_tool_result_is_sent() {
    let request = CompletionRequest::try_from(params(vec![image()])).expect("the image is sent");
    let wire = serde_json::to_value(&request.messages).expect("serialize");
    let content = &wire[0]["content"];
    assert_eq!(content[0]["type"], "image_url", "{wire}");
    assert!(
        content[0]["image_url"]["url"]
            .as_str()
            .is_some_and(|u| u.starts_with("data:image/png;base64,")),
        "{wire}"
    );
}

/// An image forces array form even when the provider flattens text results,
/// because a string has nowhere to put it.
#[test]
fn an_image_forces_array_content_even_when_flattening_is_configured() {
    let request = CompletionRequest::try_from(params(vec![image()])).expect("build");
    let wire = serde_json::to_value(&request.messages).expect("serialize");
    assert!(
        wire[0]["content"].is_array(),
        "image results must stay an array: {wire}"
    );
}

/// A text-only result flattens to a string.
#[test]
fn a_text_tool_result_is_a_string() {
    let request = CompletionRequest::try_from(params(vec![message::ToolResultContent::text("ok")]))
        .expect("text results build");
    let wire = serde_json::to_value(&request.messages).expect("serialize");
    assert_eq!(wire[0]["content"], "ok", "{wire}");
}

/// And the image shape round-trips.
#[test]
fn an_image_part_round_trips_through_serde() {
    let parsed: Message = serde_json::from_str(
        r#"{"role":"tool","tool_call_id":"c1","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}"#,
    )
    .expect("an image part should deserialize");
    let Message::ToolResult { content, .. } = parsed else {
        panic!("expected a tool result");
    };
    assert!(content.has_image());
    let wire = serde_json::to_value(&content).expect("serialize");
    assert_eq!(wire[0]["type"], "image_url");
}
