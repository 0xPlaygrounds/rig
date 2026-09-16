//! Edge matrix for the `strict` flag on OpenAI Responses function tools
//! (rig#2477).
//!
//! **Bug.** `ResponsesToolDefinition::strict` carried
//! `skip_serializing_if = is_false`, so a non-strict tool reached the wire with
//! no `strict` key at all. On the Responses API an omitted `strict` is not
//! `false`: the server attempts to normalize the schema into strict mode and,
//! when it can, constrains decoding so every declared property must be
//! present. The visible symptom is a model that cannot omit an optional
//! argument — `{"fact": "...", "source": ""}` on every call — while rig's own
//! docs promised "disabled by default".
//!
//! **Fix.** `strict` is always serialized. `false` is now an explicit wire
//! value, `true` still arrives only through `with_strict` /
//! `with_strict_tools`.
//!
//! **How these cells fail on `origin/main`.** The cassette harness matches
//! the recorded *request body*, so every cell that advertises a non-strict
//! tool is a mock miss on `main` (which sends no `strict` key) and passes with
//! the fix. Each cell also reads its own fixture back and asserts the literal
//! `strict` value, so the claim is visible rather than implied by the match.
//!
//! | # | cell | transport | level | strict | observed |
//! |---|------|-----------|-------|--------|----------|
//! | 1 | `non_strict_tool_omits_optional_argument_blocking` | blocking | raw model | `false` | optional arg omitted |
//! | 2 | `non_strict_tool_omits_optional_argument_streaming` | streaming | raw model | `false` | optional arg omitted |
//! | 3 | `strict_tools_opt_in_sends_strict_true` | blocking | raw model | `true` | every property present |
//! | 4 | `agent_tool_turn_sends_strict_false` | blocking | agent + tool | `false` | both turns non-strict |
//!
//! Unit cells for the serializer itself live beside the type in
//! `crates/rig-core/src/providers/openai/responses_api/tests.rs`
//! (`responses_function_tools_are_non_strict_by_default`).

use rig::completion::{CompletionModel, ToolDefinition};
use rig::message::AssistantContent;
use rig::prelude::*;
use rig::providers::openai;
use serde_json::{Value, json};

use super::super::support::with_openai_cassette;
use crate::support::{Adder, collect_raw_stream_observation};

const RECORD_FACT: &str = "record_fact";
const PREAMBLE: &str = "You are a note-taking assistant. Record facts with the record_fact tool.";
const OMIT_SOURCE_PROMPT: &str = "Call record_fact exactly once with fact = \
    \"Water boils at 100 degrees Celsius at sea level\". \
    There is no source for this fact, so do not pass a source argument.";

/// A tool with one required and one optional property. Whether the model may
/// leave `source` out is exactly what `strict` decides.
fn record_fact_tool() -> ToolDefinition {
    ToolDefinition {
        name: RECORD_FACT.to_string(),
        description: "Record a fact, optionally with the source it came from.".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "fact": { "type": "string", "description": "The fact to record." },
                "source": { "type": "string", "description": "Where the fact came from, if known." }
            },
            "required": ["fact"]
        }),
    }
}

fn record_fact_arguments(choice: &[AssistantContent]) -> Value {
    choice
        .iter()
        .find_map(|content| match content {
            AssistantContent::ToolCall(call) if call.function.name == RECORD_FACT => {
                Some(call.function.arguments.clone())
            }
            _ => None,
        })
        .expect("response should contain the record_fact tool call")
}

fn assert_source_omitted(arguments: &Value) {
    assert_eq!(
        arguments.get("fact").and_then(Value::as_str),
        Some("Water boils at 100 degrees Celsius at sea level"),
        "the required argument should be present verbatim, got {arguments}"
    );
    assert!(
        arguments.get("source").is_none(),
        "a non-strict tool must let the model omit an optional property; got {arguments}"
    );
}

/// The literal `strict` value of every function tool in every recorded
/// request of `scenario`, in wire order.
fn recorded_strict_flags(scenario: &str) -> Vec<Option<Value>> {
    crate::cassettes::recorded_interaction_bodies("openai", scenario)
        .into_iter()
        .flat_map(|(request, _)| {
            let body: Value = serde_json::from_str(&request).expect("request body is JSON");
            body.get("tools")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default()
        })
        .filter(|tool| tool.get("type").and_then(Value::as_str) == Some("function"))
        .map(|tool| tool.get("strict").cloned())
        .collect()
}

fn assert_recorded_strict(scenario: &str, expected: bool) {
    let flags = recorded_strict_flags(scenario);
    assert!(
        !flags.is_empty(),
        "{scenario}: fixture should record at least one function tool"
    );
    for flag in &flags {
        assert_eq!(
            flag.as_ref(),
            Some(&json!(expected)),
            "{scenario}: every function tool must carry an explicit `strict: {expected}`, got {flags:?}"
        );
    }
}

#[tokio::test]
async fn non_strict_tool_omits_optional_argument_blocking() {
    with_openai_cassette(
        "strict_tool_matrix/non_strict_tool_omits_optional_argument_blocking",
        |client| async move {
            let model = client.openai.completion(openai::GPT_4O_MINI);
            let request = model
                .completion_request(OMIT_SOURCE_PROMPT)
                .preamble(PREAMBLE.to_string())
                .tool(record_fact_tool())
                .build();

            let response = model
                .completion(request)
                .await
                .expect("non-strict tool request should succeed");

            assert_source_omitted(&record_fact_arguments(&response.choice));
        },
    )
    .await;

    assert_recorded_strict(
        "strict_tool_matrix/non_strict_tool_omits_optional_argument_blocking",
        false,
    );
}

#[tokio::test]
async fn non_strict_tool_omits_optional_argument_streaming() {
    with_openai_cassette(
        "strict_tool_matrix/non_strict_tool_omits_optional_argument_streaming",
        |client| async move {
            let model = client.openai.completion(openai::GPT_4O_MINI);
            let request = model
                .completion_request(OMIT_SOURCE_PROMPT)
                .preamble(PREAMBLE.to_string())
                .tool(record_fact_tool())
                .build();

            let stream = model
                .stream(request)
                .await
                .expect("non-strict streaming tool request should start");
            let observation = collect_raw_stream_observation(stream).await;

            assert!(
                observation.errors.is_empty(),
                "stream should not error: {:?}",
                observation.errors
            );
            let call = observation
                .tool_calls
                .iter()
                .find(|call| call.function.name == RECORD_FACT)
                .expect("stream should complete the record_fact tool call");
            assert_source_omitted(&call.function.arguments);
        },
    )
    .await;

    assert_recorded_strict(
        "strict_tool_matrix/non_strict_tool_omits_optional_argument_streaming",
        false,
    );
}

#[tokio::test]
async fn strict_tools_opt_in_sends_strict_true() {
    with_openai_cassette(
        "strict_tool_matrix/strict_tools_opt_in_sends_strict_true",
        |client| async move {
            let model = client
                .openai
                .completion(openai::GPT_4O_MINI)
                .map_wire(|wire| wire.with_strict_tools());
            let request = model
                .completion_request(OMIT_SOURCE_PROMPT)
                .preamble(PREAMBLE.to_string())
                .tool(record_fact_tool())
                .build();

            let response = model
                .completion(request)
                .await
                .expect("strict tool request should succeed");

            let arguments = record_fact_arguments(&response.choice);
            assert!(
                arguments
                    .get("fact")
                    .and_then(Value::as_str)
                    .is_some_and(|fact| fact.contains("100 degrees Celsius")),
                "the required argument should be present, got {arguments}"
            );
            // Strict mode is the control: constrained decoding forces every
            // declared property into the object, so the optional one arrives
            // (as `null`) even though the prompt asked to leave it out.
            assert!(
                arguments.get("source").is_some(),
                "a strict tool cannot omit a declared property; got {arguments}"
            );
        },
    )
    .await;

    assert_recorded_strict(
        "strict_tool_matrix/strict_tools_opt_in_sends_strict_true",
        true,
    );
}

#[tokio::test]
async fn agent_tool_turn_sends_strict_false() {
    with_openai_cassette(
        "strict_tool_matrix/agent_tool_turn_sends_strict_false",
        |client| async move {
            let agent = client
                .openai
                .agent(openai::GPT_4O_MINI)
                .preamble("You are a calculator. Use the add tool for arithmetic, then answer.")
                .tool(Adder)
                .default_max_turns(4)
                .build();

            let answer = agent
                .prompt("What is 17 + 25? Use the add tool.")
                .await
                .expect("agent tool turn should succeed");

            assert!(
                answer.output.contains("42"),
                "the agent should report the tool's result, got {answer:?}"
            );
        },
    )
    .await;

    // Both the tool-call turn and the follow-up turn advertise the tool, and
    // neither may drift back to an omitted `strict`.
    let flags = recorded_strict_flags("strict_tool_matrix/agent_tool_turn_sends_strict_false");
    assert_eq!(
        flags.len(),
        2,
        "the agent run should record the tool on two requests, got {flags:?}"
    );
    assert_recorded_strict(
        "strict_tool_matrix/agent_tool_turn_sends_strict_false",
        false,
    );
}
