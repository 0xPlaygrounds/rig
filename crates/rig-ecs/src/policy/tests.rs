//! The strings and the fold against the goldens that pin them. Each test
//! names its CONTRACT row; the goldens are read from `rig-cassette`'s
//! fixtures, never restated.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::indexing_slicing)]

use rig_core::{
    completion::message::{Message, ToolChoice, UserContent},
    effect::{EffectKind, FamilyDescriptor, HandlerDescriptor, HandlerKey},
};

use rig_core::structured_output::{
    OUTPUT_TOOL_NAME, output_tool_callable, reprompt_missing_fields, reprompt_text_answer,
};

use super::*;
use crate::agent::OutputKind;

fn golden(name: &str) -> serde_json::Value {
    let path = format!(
        "{}/../rig-cassette/fixtures/effects/{name}.effects.json",
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
    assert_eq!(tool["name"], OUTPUT_TOOL_NAME);
    assert_eq!(tool["description"], OUTPUT_TOOL_DESCRIPTION);
}

/// CONTRACT §strings: the tool-mode augmentation after a blank line
/// (`anthropic_output_tool_unary` `/records/0/kind/request/chat_history/0`).
#[test]
fn the_tool_augmentation_is_the_goldens() {
    let system =
        system_content(&request("anthropic_output_tool_unary", 0)).expect("a system message");
    let expected = format!(
        "You are a concise assistant. Answer directly.{}{}",
        AUGMENTATION_SEPARATOR,
        output_tool_augmentation("final_result")
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
        AUGMENTATION_SEPARATOR,
        prompted_augmentation(&to_canonical_string(&schema))
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
        reprompt_text_answer("final_result")
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
            function_names: vec![
                rig_core::message::ToolName::new("final_result").expect("tool name")
            ]
        }),
        "final_result"
    ));
}

/// A graph with no preamble and no utterance has no conversation to send.
#[test]
fn a_graph_without_a_conversation_does_not_fold() {
    let graph = RequestGraph {
        preamble: None,
        utterances: Vec::new(),
        documents: Vec::new(),
        tools: Vec::new(),
        temperature: None,
        max_tokens: None,
        additional_params: None,
        tool_choice: None,
        output: OutputKind::Native,
        schema: None,
        output_tool: None,
        output_tool_config: None,
    };
    assert!(matches!(
        fold_request(&graph),
        Err(crate::agent::content::parts::ContentError::Missing)
    ));
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
        output_tool_config: None,
    };
    assert_eq!(
        serde_json::to_value(fold_request(&graph).expect("the graph folds")).expect("serde"),
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
        output_tool_config: None,
    };
    assert_eq!(
        serde_json::to_value(fold_request(&graph).expect("the graph folds")).expect("serde"),
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
    let user = Message::user("hi");
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

#[test]
fn the_tool_result_cut_keeps_at_most_the_limit_on_character_boundaries() {
    use crate::agent::content::parts::ToolResultLimit;
    let limit = |max_bytes| ToolResultLimit {
        max_bytes,
        marker: "<{omitted}>".to_owned(),
    };
    assert_eq!(limit_tool_result_text("abcdef", &limit(6)), None);
    assert_eq!(
        limit_tool_result_text("abcdefg", &limit(6)).as_deref(),
        Some("abc<1>efg")
    );
    assert_eq!(
        limit_tool_result_text("abcdefg", &limit(5)).as_deref(),
        Some("ab<2>efg"),
        "an odd limit gives the tail the extra byte"
    );
    assert_eq!(
        limit_tool_result_text("abcdefg", &limit(0)).as_deref(),
        Some("<7>")
    );
    // "é" is two bytes: 3 floors the head to one character, the tail of
    // 4 bytes is two characters.
    assert_eq!(
        limit_tool_result_text("éééééé", &limit(7)).as_deref(),
        Some("é<6>éé")
    );
    // A tail start inside a character rounds forward.
    assert_eq!(
        limit_tool_result_text("aéééé", &limit(3)).as_deref(),
        Some("a<6>é")
    );
    let cut = limit_tool_result_text("日本語のテキスト", &limit(5)).unwrap();
    assert_eq!(cut, "<21>ト");
}

#[test]
fn the_tool_result_cut_takes_a_zero_budget_and_a_marker_wider_than_the_budget() {
    use crate::agent::content::parts::ToolResultLimit;
    let limit = |max_bytes, marker: &str| ToolResultLimit {
        max_bytes,
        marker: marker.to_owned(),
    };
    // Nothing kept: the marker alone, the whole text counted.
    assert_eq!(
        limit_tool_result_text("abcdefg", &limit(0, "<{omitted}>")).as_deref(),
        Some("<7>")
    );
    assert_eq!(
        limit_tool_result_text("é", &limit(0, "…")).as_deref(),
        Some("…")
    );
    // A budget smaller than the marker: the marker is not budgeted, so the
    // cut is wider than the limit, and the kept bytes are still at most it.
    let marker = "[... {omitted} bytes omitted ...]";
    assert_eq!(
        limit_tool_result_text("abcdefg", &limit(1, marker)).as_deref(),
        Some("[... 6 bytes omitted ...]g")
    );
    assert_eq!(
        limit_tool_result_text("abcdefg", &limit(2, marker)).as_deref(),
        Some("a[... 5 bytes omitted ...]g")
    );
    // A one-byte budget over multibyte text keeps nothing at the head and
    // rounds the tail forward to a whole character.
    assert_eq!(
        limit_tool_result_text("éé", &limit(1, marker)).as_deref(),
        Some("[... 4 bytes omitted ...]")
    );
    // A zero-length text under a zero budget is within the limit.
    assert_eq!(limit_tool_result_text("", &limit(0, marker)), None);
}
