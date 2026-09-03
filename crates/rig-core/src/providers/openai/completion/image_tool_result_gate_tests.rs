//! The per-provider gate on images in `role:"tool"` messages.
//!
//! Measured, not assumed. Official OpenAI Chat Completions answers 400 on
//! `gpt-4o`/`gpt-4o-mini` (*"Image URLs are only allowed for messages with
//! role 'user'"*) and, on `gpt-5` … `gpt-5.5`, answers 200 with the image
//! discarded and the model describing what it never received. llama.cpp
//! (`b1-6d05498`, Qwen3-VL-2B) delivers it: a magenta/green/yellow square
//! handed back through a tool is named correctly 3/3, matching a control
//! that sends the same bytes in a `user` message.
//!
//! These are unit tests rather than cassettes because the behaviour under
//! test is a *local* refusal — no request is made, so there is nothing to
//! record. The provider-side facts they encode are pinned by recorded cells
//! in the llamacpp and openai suites.

use super::*;
use crate::message;

fn params(
    supports_image_tool_results: bool,
    content: Vec<message::ToolResultContent>,
) -> OpenAIRequestParams {
    OpenAIRequestParams {
        model: "test-model".to_string(),
        request: crate::completion::CompletionRequest {
            model: None,
            chat_history: vec![message::Message::User {
                content: vec![message::UserContent::ToolResult(message::ToolResult {
                    call: message::ToolCallId::new_or_minted("call_1", 0),
                    provider: message::ProviderCallId::new("call_1"),
                    name: "view_file".to_string(),
                    content,
                })],
            }],
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        },
        strict_tools: false,
        tool_result_array_content: false,
        supports_image_tool_results,
        supports_tools: true,
        supports_response_format: true,
    }
}

fn image() -> message::ToolResultContent {
    message::ToolResultContent::image_base64(
        "iVBORw0KGgo=",
        Some(message::ImageMediaType::PNG),
        None,
    )
}

/// A provider that cannot carry the image refuses locally rather than
/// flattening it away or letting the provider answer.
#[test]
fn an_image_tool_result_is_refused_when_the_provider_cannot_carry_it() {
    let error = CompletionRequest::try_from(params(false, vec![image()]))
        .expect_err("a provider without the capability must refuse");
    let message = error.to_string();
    assert!(
        message.contains("does not accept an image in a tool result"),
        "{message}"
    );
    assert!(
        message.contains("Responses API"),
        "the refusal should name the surface that works: {message}"
    );
}

/// The same request is built when the provider does carry it.
#[test]
fn an_image_tool_result_is_sent_when_the_provider_supports_it() {
    let request = CompletionRequest::try_from(params(true, vec![image()]))
        .expect("a capable provider should accept the image");
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
    let request = CompletionRequest::try_from(params(true, vec![image()])).expect("build");
    let wire = serde_json::to_value(&request.messages).expect("serialize");
    assert!(
        wire[0]["content"].is_array(),
        "image results must stay an array: {wire}"
    );
}

/// Text-only results are untouched by the gate, on both settings.
#[test]
fn a_text_tool_result_is_unaffected_by_the_gate() {
    for supports in [false, true] {
        let request = CompletionRequest::try_from(params(
            supports,
            vec![message::ToolResultContent::text("ok")],
        ))
        .unwrap_or_else(|e| panic!("text results must always build (supports={supports}): {e}"));
        let wire = serde_json::to_value(&request.messages).expect("serialize");
        assert_eq!(wire[0]["content"], "ok", "supports={supports}: {wire}");
    }
}

/// A mixed result is refused as a whole rather than silently losing its
/// image half.
#[test]
fn a_mixed_text_and_image_result_is_refused_rather_than_partly_sent() {
    let error = CompletionRequest::try_from(params(
        false,
        vec![
            message::ToolResultContent::text("here is the file"),
            image(),
        ],
    ))
    .expect_err("the image half cannot be dropped silently");
    assert!(
        error.to_string().contains("does not accept an image"),
        "{error}"
    );
}

/// A wire tool result carrying an image converts back into rig's types with
/// the image intact.
///
/// The inbound counterpart of the gate. Flattening with `as_text()` turned
/// this into `Text("")` — a silent drop, and one that used to be impossible:
/// before the image variant existed such a body failed to deserialize, so
/// the loss was at least loud.
#[test]
fn an_inbound_tool_result_image_is_not_flattened_away() {
    let wire: Message = serde_json::from_str(
        r#"{"role":"tool","tool_call_id":"c1","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgo="}}]}"#,
    )
    .expect("deserialize");

    let converted = message::Message::try_from(wire).expect("convert back");
    let message::Message::User { content } = converted else {
        panic!("a tool result converts to a user message");
    };
    let message::UserContent::ToolResult(result) = &content[0] else {
        panic!("expected a tool result");
    };
    assert!(
        matches!(
            result.content.first(),
            Some(message::ToolResultContent::Image(_))
        ),
        "the image must survive the round trip, got {:?}",
        result.content
    );
}

/// A mixed result keeps both halves, in order.
#[test]
fn an_inbound_mixed_tool_result_keeps_text_and_image() {
    let wire: Message = serde_json::from_str(
        r#"{"role":"tool","tool_call_id":"c1","content":[{"type":"text","text":"here"},{"type":"image_url","image_url":{"url":"https://example.com/x.png"}}]}"#,
    )
    .expect("deserialize");

    let converted = message::Message::try_from(wire).expect("convert back");
    let message::Message::User { content } = converted else {
        panic!("user message")
    };
    let message::UserContent::ToolResult(result) = &content[0] else {
        panic!("tool result")
    };
    assert_eq!(result.content.len(), 2, "{:?}", result.content);
    assert!(matches!(
        result.content[0],
        message::ToolResultContent::Text(_)
    ));
    assert!(matches!(
        result.content[1],
        message::ToolResultContent::Image(_)
    ));
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
