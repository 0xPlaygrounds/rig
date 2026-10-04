//! Cross-family tool calling: the dimension that tests
//! `EMITS_COMPLETE_SINGLE_CHUNK_TOOL_CALLS`.
//!
//! `--jinja` makes `llama-server` render the **model's own** chat template, so
//! the tool-call wire is a property of the *model family*, not of llama.cpp.
//! Qwen, Llama and Mistral each emit and parse tool calls through different
//! delimiters, and a claim about "what llama.cpp does with tool calls" that
//! was only ever checked against Qwen is a claim about Qwen.
//!
//! Coverage here is **targeted, not broad**, exactly as the corpus rules
//! require: one blocking tool cell per non-Qwen family. The rest of the matrix
//! stays on Qwen.
//!
//! | Family | Model | Template tool support | Blocking | Streaming |
//! | --- | --- | --- | --- | --- |
//! | Qwen (smoke) | `unsloth/Qwen3-1.7B-GGUF` Q4_K_M | yes | `tools.rs`, `tool_matrix.rs` | `streaming_tools.rs` |
//! | Qwen (competent) | `unsloth/Qwen3-8B-GGUF` Q4_K_M | yes, parallel | `tool_matrix.rs` | — |
//! | Llama | `bartowski/Llama-3.2-3B-Instruct-GGUF` Q4_K_M | yes, no parallel | [`llama_family_calls_a_tool`] | none |
//!
//! # The finding: llama.cpp does not emit single-chunk tool calls
//!
//! `EMITS_COMPLETE_SINGLE_CHUNK_TOOL_CALLS` asks whether the backend can put a
//! whole tool call — id, name and **complete** arguments — into one streaming
//! chunk. The provider inherited `true` for it. Measured against
//! `llama-server` b10964-b29c606e2 it is **false on every family that can call a
//! tool at all**: arguments stream one token at a time, so the first chunk
//! carries the name beside a lone `{` and the closing `}` arrives ten chunks
//! later. Even a *zero-argument* call streams as `{` then `}` rather than as
//! `{}`.
//!
//! The const is now `false`. Flipping it changes no output and that was
//! checked rather than argued: the shared accumulator's immediate-emit is a
//! probe that finalizes a call only when its
//! accumulated arguments parse, so a lone `{` was already being declined, and
//! the whole recorded streaming corpus replays byte-identically either way.
//! What changes is that the const stops asserting something untrue — and that
//! a future build whose partial arguments happened to parse could not finalize
//! a truncated call.
//!
//! # Gemma has no tools, and that is a note rather than a finding
//!
//! `GET /props` on the Gemma server reports
//! `chat_template_caps.supports_tools: false` and
//! `supports_tool_calls: false`. Its template has no tool section, so
//! llama.cpp has nothing to render tool definitions into and nothing to parse
//! a call back out of; the model answers a tool prompt with a ```` ```tool_code ````
//! block as ordinary prose. A capability the loaded model lacks is not a rig
//! defect — it is the model choice — so the cell records the shape and the
//! streaming twin is dropped with this as its reason.

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

// ---------------------------------------------------------------------------
// Mistral Small 3.2
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Gemma 3 — no tool support in the template
// ---------------------------------------------------------------------------
