//! Cross-family tool calling with llama.cpp.
//!
//! `--jinja` makes `llama-server` render the model's own chat template, so
//! the tool-call wire is a property of the model family, not of llama.cpp.
//! Coverage is targeted: one blocking tool cell per non-Qwen family
//! ([`llama_family_calls_a_tool`]), and the rest of the matrix stays on Qwen.
//!
//! `EMITS_COMPLETE_SINGLE_CHUNK_TOOL_CALLS` is `false` for llama.cpp: on
//! every family that can call a tool, arguments stream one token at a time,
//! so the first chunk carries the name beside a lone `{`. Gemma's template
//! reports no tool support in `GET /props`, so it has no tool cell; a
//! capability the loaded model lacks is the model choice, not a rig defect.

use rig::message::AssistantContent;
use serde_json::Value;

use crate::cassettes::recorded_statuses_and_bodies;
use crate::support::Subtract;

use super::super::cassette_support::*;
use rig::completion::CompletionRequest;

const TOOL_PROMPT: &str = "Calculate 2 - 5 using the tool.";

/// The `(name, arguments)` pairs a recorded blocking response asked for.
fn recorded_tool_calls(scenario: &str) -> Vec<(String, String)> {
    let recorded = recorded_statuses_and_bodies("llamacpp", scenario);
    let (status, body) = recorded.last().expect("an interaction");
    assert_eq!(*status, 200, "{scenario}: {body}");
    let response: Value = serde_json::from_str(body).expect("response should be JSON");
    response["choices"][0]["message"]["tool_calls"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .iter()
        .map(|call| {
            (
                call["function"]["name"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string(),
                call["function"]["arguments"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string(),
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Llama 3.2
// ---------------------------------------------------------------------------

#[tokio::test]
async fn llama_family_calls_a_tool() {
    with_llamacpp_llama_family_cassette(
        "model_family_matrix/llama_blocking_tool_call",
        |client| async move {
            let model = client.completion(CASSETTE_LLAMA_MODEL);
            let response = model
                .call(
                    CompletionRequest::new(TOOL_PROMPT)
                        .tool(rig::tool::tool_definition(&Subtract))
                        .max_tokens(512),
                )
                .await
                .expect("Llama 3.2's template supports tool calls");

            let calls = response
                .choice
                .iter()
                .filter_map(|item| match item {
                    AssistantContent::ToolCall(call) => Some(call.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(calls.len(), 1, "{:?}", response.choice);
            assert_eq!(calls[0].function.name, "subtract");
            assert!(
                calls[0].function.invalid_arguments.is_none(),
                "a different template must still produce object arguments: {:?}",
                calls[0].function.arguments
            );
        },
    )
    .await;

    let calls = recorded_tool_calls("model_family_matrix/llama_blocking_tool_call");
    assert_eq!(calls.len(), 1, "{calls:?}");
    assert_eq!(calls[0].0, "subtract");
    serde_json::from_str::<Value>(&calls[0].1)
        .expect("the wire's stringified arguments must be JSON");
}
