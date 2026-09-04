//! The strings and the fold against the goldens that pin them. Each test
//! names its CONTRACT row; the goldens are read from `rig-verify`'s
//! fixtures, never restated.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use rig_core::{
    completion::message::{Message, ToolChoice, UserContent},
    effect::{EffectKind, FamilyDescriptor, HandlerDescriptor, HandlerKey},
};

use super::*;
use crate::agent::OutputKind;

fn golden(name: &str) -> serde_json::Value {
    let path = format!(
        "{}/../rig-verify/fixtures/{name}.effects.json",
        env!("CARGO_MANIFEST_DIR")
    );
    serde_json::from_str(&std::fs::read_to_string(path).expect("the golden is committed"))
        .expect("the golden loads")
}

fn request(name: &str, record: usize) -> serde_json::Value {
    golden(name)["records"][record]["kind"]["request"].clone()
}

fn system_content(request: &serde_json::Value) -> Option<String> {
    request["chat_history"]
        .as_array()?
        .first()
        .filter(|message| message["role"] == "system")
        .and_then(|message| message["content"].as_str())
        .map(str::to_owned)
}

/// CONTRACT §strings: the output tool's name and description
/// (`anthropic_output_tool_unary` `/records/0/kind/request/tools/0`).
#[test]
fn the_output_tool_is_the_goldens() {
    let tool = request("anthropic_output_tool_unary", 0)["tools"][0].clone();
    assert_eq!(tool["name"], text::OUTPUT_TOOL_NAME);
    assert_eq!(tool["description"], text::OUTPUT_TOOL_DESCRIPTION);
}

/// CONTRACT §strings: the tool-mode augmentation after a blank line
/// (`anthropic_output_tool_unary` `/records/0/kind/request/chat_history/0`).
#[test]
fn the_tool_augmentation_is_the_goldens() {
    let system =
        system_content(&request("anthropic_output_tool_unary", 0)).expect("a system message");
    let expected = format!(
        "You are a concise assistant. Answer directly.{}{}",
        text::AUGMENTATION_SEPARATOR,
        text::output_tool_augmentation("final_result")
    );
    assert_eq!(system, expected);
}

/// CONTRACT §strings: the prompted augmentation with the canonical schema
/// (`anthropic_output_prompted_unary` `/records/0/kind/request/chat_history/0`).
#[test]
fn the_prompted_augmentation_is_the_goldens() {
    let system =
        system_content(&request("anthropic_output_prompted_unary", 0)).expect("a system message");
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "category": {"type": "string"},
            "summary": {"type": "string"}
        },
        "required": ["title", "category", "summary"]
    });
    let expected = format!(
        "You are a concise assistant. Answer directly.{}{}",
        text::AUGMENTATION_SEPARATOR,
        text::prompted_augmentation(&to_canonical_string(&schema))
    );
    assert_eq!(system, expected);
}

/// CONTRACT §reprompts: the text-answer reprompt
/// (`mock_output_tool_text_reprompt` `/records/1/kind/request/chat_history/3`).
#[test]
fn the_text_reprompt_is_the_goldens() {
    let last = request("mock_output_tool_text_reprompt", 1)["chat_history"][3].clone();
    assert_eq!(last["role"], "user");
    assert_eq!(
        last["content"][0]["text"],
        text::reprompt_text_answer("final_result")
    );
}

/// CONTRACT §reprompts: the missing-field reprompt as a tool result
/// (`mock_output_tool_missing_field_reprompt` `/records/1/kind/request/chat_history/3`).
#[test]
fn the_missing_field_reprompt_is_the_goldens() {
    let last = request("mock_output_tool_missing_field_reprompt", 1)["chat_history"][3].clone();
    assert_eq!(last["content"][0]["type"], "toolresult");
    assert_eq!(last["content"][0]["name"], "final_result");
    assert_eq!(
        last["content"][0]["content"][0]["text"],
        reprompt_missing_fields("final_result", &["summary".to_owned()])
    );
}

/// CONTRACT §output: `Auto` with a schema is `Native` when the provider
/// composes native output with tools (`anthropic_request_shape_output_schema_unary`),
/// `Tool` is `Native` under `tool_choice: none`
/// (`anthropic_output_tool_under_none_degrades`), no schema is `Native`.
#[test]
fn output_resolution_follows_the_goldens() {
    assert_eq!(
        resolve_output(OutputKind::Auto, true, 0, true, true),
        OutputKind::Native
    );
    assert_eq!(
        resolve_output(OutputKind::Auto, true, 1, true, false),
        OutputKind::Tool
    );
    assert_eq!(
        resolve_output(OutputKind::Tool, true, 0, false, true),
        OutputKind::Native
    );
    assert_eq!(
        resolve_output(OutputKind::Tool, true, 0, true, true),
        OutputKind::Tool
    );
    assert_eq!(
        resolve_output(OutputKind::Prompted, false, 0, true, true),
        OutputKind::Native
    );
    assert!(!output_tool_callable(
        Some(&ToolChoice::None),
        "final_result"
    ));
    assert!(output_tool_callable(
        Some(&ToolChoice::Specific {
            function_names: vec!["final_result".to_owned()]
        }),
        "final_result"
    ));
}

/// CONTRACT §output: the output tool's name is numbered from 1 on a
/// collision (`rig_agent::run::prepare` docs; no golden collides).
#[test]
fn the_output_tool_name_numbers_from_one() {
    assert_eq!(output_tool_name(&["add"]), "final_result");
    assert_eq!(output_tool_name(&["final_result"]), "final_result_1");
    assert_eq!(
        output_tool_name(&["final_result", "final_result_1"]),
        "final_result_2"
    );
}

/// CONTRACT §history: an empty turn is no history
/// (`anthropic_request_shape_tool_choice_none` `/records/0/outcome`).
#[test]
fn an_empty_turn_is_not_history() {
    assert!(is_empty_assistant_turn(&[]));
    assert!(is_empty_assistant_turn(&[AssistantContent::text("")]));
    assert!(!is_empty_assistant_turn(&[AssistantContent::text("x")]));
}

/// CONTRACT §derivation: the fold over the smoke golden's graph is the
/// smoke golden's request, field for field
/// (`anthropic_completion_smoke` `/records/0/kind/request`).
#[test]
fn the_fold_reproduces_the_smoke_request() {
    let prompt = MessageParts::User {
        content: vec![UserContent::text(
            "In one or two sentences, explain what Rust programming language is and why memory safety matters.",
        )],
    };
    let graph = RequestGraph {
        preamble: Some("You are a concise assistant. Answer directly."),
        utterances: vec![&prompt],
        documents: Vec::new(),
        tools: Vec::new(),
        temperature: None,
        max_tokens: None,
        additional_params: None,
        tool_choice: None,
        output: OutputKind::Native,
        schema: None,
        output_tool: None,
    };
    let folded = serde_json::to_value(fold_request(&graph)).expect("serde");
    assert_eq!(folded, request("anthropic_completion_smoke", 0));
}

/// CONTRACT §derivation: static context is the attachments in order
/// (`anthropic_request_shape_static_context` `/records/0/kind/request/documents`),
/// and a granted tool is its descriptor
/// (`anthropic_request_shape_tool_choice_none` `/records/0/kind/request/tools`).
#[test]
fn documents_and_tools_fold_from_the_graph() {
    let golden = request("anthropic_request_shape_static_context", 0);
    let documents: Vec<Document> =
        serde_json::from_value(golden["documents"].clone()).expect("serde");
    let prompt = MessageParts::User {
        content: vec![UserContent::text("What does \"glarb-glarb\" mean?")],
    };
    let graph = RequestGraph {
        preamble: Some("You are a concise assistant. Answer directly."),
        utterances: vec![&prompt],
        documents,
        tools: Vec::new(),
        temperature: Some(0.0),
        max_tokens: None,
        additional_params: None,
        tool_choice: None,
        output: OutputKind::Native,
        schema: None,
        output_tool: None,
    };
    assert_eq!(
        serde_json::to_value(fold_request(&graph)).expect("serde"),
        golden
    );

    let golden = request("anthropic_request_shape_tool_choice_none", 0);
    let tool = &golden["tools"][0];
    let descriptor = HandlerDescriptor {
        key: HandlerKey::from("golden/tool:add#0"),
        family: FamilyDescriptor::Tool {
            name: tool["name"].as_str().expect("name").to_owned(),
            description: tool["description"]
                .as_str()
                .expect("description")
                .to_owned(),
            parameters: tool["parameters"].clone(),
            embedding: None,
        },
        layers: Vec::new(),
    };
    let prompt = MessageParts::User {
        content: vec![UserContent::text(
            "What is 17 + 25? Reply with just the number.",
        )],
    };
    let graph = RequestGraph {
        preamble: Some("You are a concise assistant. Answer directly."),
        utterances: vec![&prompt],
        documents: Vec::new(),
        tools: vec![&descriptor],
        temperature: Some(0.0),
        max_tokens: None,
        additional_params: None,
        tool_choice: Some(&ToolChoice::None),
        output: OutputKind::Native,
        schema: None,
        output_tool: None,
    };
    assert_eq!(
        serde_json::to_value(fold_request(&graph)).expect("serde"),
        golden
    );
}

/// A system message is never an utterance; user and assistant parts round
/// trip through `MessageParts`.
#[test]
fn message_parts_round_trip_but_never_a_system_message() {
    assert!(
        MessageParts::from_message(&Message::System {
            content: "x".to_owned()
        })
        .is_none()
    );
    let user = user_text("hi");
    let parts = MessageParts::from_message(&user).expect("a user message");
    assert_eq!(
        serde_json::to_value(parts.to_message()).expect("serde"),
        serde_json::to_value(&user).expect("serde")
    );
    let _ = EffectKind::Custom {
        kind: std::sync::Arc::from("unused"),
        payload: serde_json::Value::Null,
    };
}
