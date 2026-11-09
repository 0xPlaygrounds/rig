//! Cassette-backed DeepSeek long-session and tool-contract regression tests.
//!
//! These scenarios stress Rig's DeepSeek OpenAI-compatible chat-completions path
//! with multi-turn tool loops, streamed tool-call deltas, complex JSON tool
//! arguments, explicit tool choice, reasoning metadata, structured JSON output,
//! and caller-owned long chat history.

use anyhow::Result;
use rig::completion::{GenerationOptions, Reasoning};
use rig::message::{AssistantContent, ToolChoice};
use rig::providers::deepseek;
use rig::tool::Tool;
use serde_json::json;

use crate::matrix::agent_tool_sessions::{
    ComplexSession, InvocationLog, chat_parallel_signals, stream_complex_session,
};
use crate::support::{
    AlphaSignal, BetaSignal, TWO_TOOL_STREAM_PREAMBLE, assert_contains_all_case_insensitive,
    assert_nonempty_response, assistant_text_response, collect_raw_stream_observation,
};

use super::support::with_deepseek_cassette_result;
use rig::completion::CompletionRequest;

pub(super) const SESSION_MODEL: &str = deepseek::DEEPSEEK_V4_FLASH;

/// Thinking off: `{"thinking":{"type":"disabled"}}` on DeepSeek.
pub(super) fn non_thinking() -> GenerationOptions {
    GenerationOptions::default().reasoning(Reasoning::Off)
}

fn thinking_params() -> serde_json::Value {
    json!({
        "thinking": { "type": "enabled" }
    })
}

const COMPLEX_SESSION: ComplexSession = ComplexSession {
    preamble: "\
You are a deterministic DeepSeek tool orchestration test harness. Use the tools instead of inventing values. \
For the production-readiness scenario, call exactly one tool at a time in this order: \
1. ping_empty with an empty JSON object. \
2. inspect_manifest with project rig-deepseek, flags critical=true and retries=2, steps plan weight=1 and verify weight=2, and the exact note from the user. \
3. join_labels with labels [north, beta gamma, quote:\"delta\", slash\\path] and separator |. \
4. escape_echo with the exact escaped text from the user. \
After all tool results are available, answer in one short sentence that includes EMPTY-OK, MANIFEST-OK, LABELS-OK, and ESCAPE-OK.",
    prompt: "\
Run the production-readiness scenario. The manifest note is `line one; line two says \"hello\" and path C:\\rig\\deepseek`. \
The escaped text is `Line 1\nLine \"2\" with backslash \\ and unicode snowman ☃`.",
    project: "rig-deepseek",
    note: "line one; line two says \"hello\" and path C:\\rig\\deepseek",
    escaped_text: "Line 1\nLine \"2\" with backslash \\ and unicode snowman ☃",
    optional_probe: None,
};

/// The provider-only facts of a completed turn, read off the captured `raw`.
///
/// The normalized response carries one finish reason for the turn and none of
/// the provider's own identity spellings per choice, so the reply's verbatim
/// `raw` body is the only place a per-choice finish reason, the provider id
/// and the provider's model string can be checked — and it rides on the very
/// response asserted beside it, so both views cost one cassette interaction.
fn assert_response_metadata(response: &rig::completion::CompletionResponse) {
    let raw = &response.raw;
    assert_nonempty_response(
        raw["id"]
            .as_str()
            .expect("raw DeepSeek response should preserve id"),
    );
    assert_nonempty_response(
        raw["model"]
            .as_str()
            .expect("raw DeepSeek response should preserve model"),
    );
    let choices = raw["choices"]
        .as_array()
        .expect("raw DeepSeek response should preserve choices");
    assert!(
        !choices.is_empty()
            && choices.iter().all(|choice| {
                choice["finish_reason"]
                    .as_str()
                    .is_some_and(|reason| !reason.is_empty())
            }),
        "raw DeepSeek choices should preserve finish reasons"
    );
    assert!(
        response.usage.input_tokens.is_some_and(|n| n > 0)
            && response.usage.output_tokens.is_some_and(|n| n > 0),
        "usage should be populated: {:?}",
        response.usage
    );
}

#[tokio::test]
async fn sequential_complex_tool_calls_streaming() -> Result<()> {
    with_deepseek_cassette_result(
        "agent_tool_sessions/sequential_complex_tool_calls_streaming",
        |client| async move {
            let log = InvocationLog::default();
            let agent = COMPLEX_SESSION
                .equip(
                    rig::AgentBuilder::new(client.completion(SESSION_MODEL)),
                    &log,
                )
                .options(non_thinking())
                .additional_params(json!({"parallel_tool_calls": false}))
                .build();
            stream_complex_session(&agent, &COMPLEX_SESSION, &log).await
        },
    )
    .await
}

#[tokio::test]
async fn parallel_tool_calls_single_turn_nonstreaming() -> Result<()> {
    with_deepseek_cassette_result(
        "agent_tool_sessions/parallel_tool_calls_single_turn_nonstreaming",
        |client| async move {
            let agent = rig::AgentBuilder::new(client.completion(SESSION_MODEL))
                .preamble(TWO_TOOL_STREAM_PREAMBLE)
                .tool(AlphaSignal)
                .tool(BetaSignal)
                .options(non_thinking())
                .additional_params(json!({"parallel_tool_calls": true}))
                .default_max_turns(5)
                .build();
            chat_parallel_signals(&agent).await
        },
    )
    .await
}

#[tokio::test]
async fn tool_choice_required_specific_and_none() -> Result<()> {
    with_deepseek_cassette_result(
        "agent_tool_sessions/tool_choice_required_specific_and_none",
        |client| async move {
            let model = client.completion(SESSION_MODEL);

            let required = model
                .call(CompletionRequest::new(
                            "Call lookup_harbor_label exactly once with an empty object and do not answer in prose.",
                        )
                        .tool(rig::tool::tool_definition(&AlphaSignal))
                        .tool_choice(ToolChoice::Required)
                        .options(non_thinking()))
                .await?;
            anyhow::ensure!(
                required.choice.iter().any(|content| matches!(
                    content,
                    AssistantContent::ToolCall(tool_call)
                        if tool_call.function.name == AlphaSignal::NAME
                            && tool_call.function.arguments_value() == json!({})
                )),
                "required tool choice should force lookup_harbor_label"
            );

            let specific = model
                .call(CompletionRequest::new(
                            "Call the orchard-label tool exactly once with an empty object and do not call any other tool.",
                        )
                        .tool(rig::tool::tool_definition(&AlphaSignal))
                        .tool(rig::tool::tool_definition(&BetaSignal))
                        .tool_choice(ToolChoice::Specific {
                            function_names: vec![rig_core::message::ToolName::new(BetaSignal::NAME).expect("tool name")],
                        })
                        .options(non_thinking()))
                .await?;
            let specific_calls = specific
                .choice
                .iter()
                .filter_map(|content| match content {
                    AssistantContent::ToolCall(tool_call) => Some(tool_call.function.name.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>();
            anyhow::ensure!(
                specific_calls == vec![BetaSignal::NAME],
                "specific tool choice should force only lookup_orchard_label, saw {specific_calls:?}"
            );

            let none = model
                .call(CompletionRequest::new(
                            "Do not call tools. Reply with exactly this phrase: no-tool-answer",
                        )
                        .tool(rig::tool::tool_definition(&AlphaSignal))
                        .tool_choice(ToolChoice::None)
                        .options(non_thinking()))
                .await?;
            let none_text = assistant_text_response(&none.choice)
                .ok_or_else(|| anyhow::anyhow!("ToolChoice::None response should contain text"))?;
            assert_contains_all_case_insensitive(&none_text, &["no-tool-answer"]);
            anyhow::ensure!(
                none.choice
                    .iter()
                    .all(|content| !matches!(content, AssistantContent::ToolCall(_))),
                "ToolChoice::None should not surface tool calls"
            );

            Ok(())
        },
    )
    .await
}

#[tokio::test]
async fn reasoning_enabled_preserves_reasoning_content_deltas_and_usage() -> Result<()> {
    with_deepseek_cassette_result(
        "agent_tool_sessions/reasoning_enabled_preserves_reasoning_content_deltas_and_usage",
        |client| async move {
            let model = client.completion(SESSION_MODEL);
            let request = CompletionRequest::new(
                    "Use concise reasoning to solve: if three probes each verify two cassettes, how many cassette verifications occur? Answer with the number.",
                )
                .preamble("You are a concise reliability engineer.")
                .additional_params(thinking_params());

            let response = model.call(request).await?;

            anyhow::ensure!(
                response
                    .choice
                    .iter()
                    .any(|content| matches!(content, AssistantContent::Reasoning(_))),
                "DeepSeek reasoning response should preserve reasoning_content separately"
            );
            anyhow::ensure!(
                response.usage.reasoning_tokens.is_some_and(|n| n > 0),
                "core usage should preserve DeepSeek reasoning tokens: {:?}",
                response.usage
            );
            let raw_reasoning_tokens = response.raw["usage"]["completion_tokens_details"]
                ["reasoning_tokens"]
                .as_u64();
            anyhow::ensure!(
                response.usage.reasoning_tokens == raw_reasoning_tokens
                    && raw_reasoning_tokens.is_some_and(|n| n > 0),
                "usage reasoning tokens should match raw provider details"
            );
            assert_response_metadata(&response);

            let stream_request = CompletionRequest::new("Briefly solve 2 + 2, then answer with the number.")
                .additional_params(thinking_params());
            let observation = collect_raw_stream_observation(model.stream(stream_request)?).await;
            anyhow::ensure!(
                observation.events.contains(&"reasoning_delta"),
                "streaming DeepSeek reasoning should emit reasoning deltas, saw {:?}",
                observation.events
            );
            let first_reasoning = observation
                .events
                .iter()
                .position(|event| *event == "reasoning_delta")
                .ok_or_else(|| anyhow::anyhow!("expected reasoning delta"))?;
            let first_text = observation
                .events
                .iter()
                .position(|event| *event == "text")
                .ok_or_else(|| anyhow::anyhow!("expected text after reasoning"))?;
            anyhow::ensure!(
                first_reasoning < first_text,
                "reasoning deltas should precede final answer text: {:?}",
                observation.events
            );

            Ok(())
        },
    )
    .await
}

#[tokio::test]
async fn json_object_response_format_roundtrip() -> Result<()> {
    with_deepseek_cassette_result(
        "agent_tool_sessions/json_object_response_format_roundtrip",
        |client| async move {
            let model = client.completion(SESSION_MODEL);
            let request = CompletionRequest::new(
                    "Return a JSON object with release lane canary, risk low, and checks compile=true and replay=true.",
                )
                .preamble("Return only valid JSON. No markdown.")
                .options(non_thinking())
                .additional_params(json!({
                    "response_format": { "type": "json_object" }
                }));

            let response = model.call(request).await?;
            let text = assistant_text_response(&response.choice)
                .ok_or_else(|| anyhow::anyhow!("JSON response should contain text"))?;
            let plan: serde_json::Value = serde_json::from_str(&text)?;

            let serialized = plan.to_string();
            assert_contains_all_case_insensitive(&serialized, &["canary", "low", "compile", "replay"]);
            assert_response_metadata(&response);

            Ok(())
        },
    )
    .await
}
