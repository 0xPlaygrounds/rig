//! All 2^3 combinations of the fields a cached content owns.
//!
//! Gemini rejects `cachedContent` alongside `system_instruction`, `tools` or
//! `tool_config` with a single 400 that does not say which one you set. Rig
//! checks all three before the request leaves the process, so the matrix is
//! exhaustive and free — no socket, no fixture.

use super::gemini_api_types::GenerateContentRequest;
use crate::completion::{CompletionRequest, ToolDefinition};
use crate::message::{Message, ToolChoice, UserContent};

const HANDLE: &str = "cachedContents/matrix";

fn build(system: bool, tools: bool, tool_choice: bool) -> GenerateContentRequest {
    super::create_request_body(CompletionRequest {
        chat_history: system
            .then(|| Message::system("you are terse"))
            .into_iter()
            .chain([Message::User {
                content: vec![UserContent::text("hi")],
            }])
            .collect(),
        documents: vec![],
        tools: if tools {
            vec![ToolDefinition {
                name: "probe".to_owned(),
                description: "probe".to_owned(),
                parameters: serde_json::json!({"type": "object", "properties": {}}),
            }]
        } else {
            vec![]
        },
        temperature: None,
        max_tokens: None,
        tool_choice: tool_choice.then_some(ToolChoice::Auto),
        additional_params: None,
        model: None,
        output_schema: None,
        record_telemetry_content: false,
    })
    .expect("request should build")
}

#[test]
fn every_combination_of_owned_fields_is_classified() {
    let mut checked = 0usize;

    for system in [false, true] {
        for tools in [false, true] {
            for tool_choice in [false, true] {
                let mut request = build(system, tools, tool_choice);
                let outcome = request.with_cached_content(HANDLE);
                let label = format!("system={system} tools={tools} tool_choice={tool_choice}");

                if !system && !tools && !tool_choice {
                    outcome.unwrap_or_else(|error| {
                        panic!("{label}: a clean request must accept a handle: {error}")
                    });
                    let body = serde_json::to_value(&request).expect("serialize");
                    assert_eq!(
                        body.get("cachedContent").and_then(|value| value.as_str()),
                        Some(HANDLE),
                        "{label}: the handle should reach the wire"
                    );
                } else {
                    let error = outcome.expect_err(&format!(
                        "{label}: the cache owns these fields, so this must be refused"
                    ));
                    let message = error.to_string();
                    // Assert against the conflict clause, not the whole
                    // message: the sentence that follows it names all three
                    // fields unconditionally ("already owns the system
                    // instruction, tools and tool choice"), so a naive
                    // `message.contains("tools")` passed for every cell in
                    // the matrix, including the ones that set no tools. The
                    // clause is the only part that varies per cell, which is
                    // the whole advantage over the provider's own 400.
                    let clause = message
                        .split_once(". The cached content")
                        .map(|(head, _)| head)
                        .unwrap_or(message.as_str());
                    if system {
                        assert!(clause.contains("system instruction"), "{label}: {message}");
                    }
                    assert_eq!(
                        clause.contains("tools"),
                        tools,
                        "{label}: the clause must name the tool set only when it conflicted: \
                             {message}"
                    );
                    if tool_choice {
                        assert!(clause.contains("tool choice"), "{label}: {message}");
                    }
                    assert!(message.contains(HANDLE), "{label}: {message}");
                }
                checked += 1;
            }
        }
    }

    assert_eq!(checked, 8, "the matrix should be exhaustive");
}

/// Handle syntax, exhaustively over the shapes a caller might pass.
#[test]
fn handle_syntax_is_validated() {
    for (handle, accepted) in [
        ("cachedContents/abc123", true),
        ("cachedContents/", true),
        ("abc123", false),
        ("", false),
        ("cachedcontents/abc", false),
        ("/cachedContents/abc", false),
        ("models/abc", false),
        (" cachedContents/abc", false),
    ] {
        let mut request = build(false, false, false);
        let outcome = request.with_cached_content(handle);
        assert_eq!(
            outcome.is_ok(),
            accepted,
            "handle {handle:?} should {} be accepted",
            if accepted { "" } else { "not" }
        );
    }
}
