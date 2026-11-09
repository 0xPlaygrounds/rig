//! Cassette-backed xAI/Grok long-session and tool-contract regression tests.
//!
//! These scenarios stress Rig's xAI Responses API integration with multi-turn
//! tool loops, streamed tool calls, complex JSON arguments, reasoning metadata,
//! explicit tool choice, structured JSON output, multimodal input, and
//! caller-owned long chat history.

use anyhow::Result;
use rig::completion::Message;
use rig::message::{AssistantContent, ToolChoice};
use rig::providers::xai;
use rig::tool::Tool;
use serde_json::json;

use crate::matrix::agent_tool_sessions::{
    ComplexSession, InspectManifest, InvocationLog, stream_complex_session,
};
use crate::support::{
    ALPHA_SIGNAL_OUTPUT, AlphaSignal, BetaSignal, TWO_TOOL_STREAM_PREAMBLE,
    assert_contains_all_case_insensitive, assert_nonempty_response,
    assert_raw_stream_tool_call_arguments_are_objects, assistant_text_response,
    collect_raw_stream_observation,
};

use super::support::with_xai_cassette_result;
use rig::completion::{CompletionRequest, Effort};

pub(super) const SESSION_MODEL: &str = "grok-4.3";
const SESSION_MAX_TOKENS: Option<u64> = None;
const REASONING_MODEL: &str = xai::GROK_3_MINI;

const COMPLEX_SESSION: ComplexSession = ComplexSession {
    preamble: "\
You are a deterministic xAI tool orchestration test harness. Use the tools instead of inventing values. \
For the production-readiness scenario, call exactly one tool at a time in this order: \
1. ping_empty with an empty JSON object. \
2. inspect_manifest with project rig-xai, flags critical=true and retries=2, steps plan weight=1 and verify weight=2, and the exact note from the user. \
3. join_labels with labels [north, beta gamma, quote:\"delta\", slash\\path] and separator |. \
4. escape_echo with the exact escaped text from the user. \
After all tool results are available, answer in one short sentence that includes EMPTY-OK, MANIFEST-OK, LABELS-OK, and ESCAPE-OK.",
    prompt: "\
Run the production-readiness scenario. The manifest note is `line one; line two says \"hello\" and path C:\\rig\\xai`. \
The escaped text is `Line 1\nLine \"2\" with backslash \\ and unicode snowman ☃`.",
    project: "rig-xai",
    note: "line one; line two says \"hello\" and path C:\\rig\\xai",
    escaped_text: "Line 1\nLine \"2\" with backslash \\ and unicode snowman ☃",
    optional_probe: None,
};

/// Assert the provider-native metadata xAI reports on its own wire response.
///
/// The response id (`resp_...`), status, and full usage envelope are read
/// from the provider reply captured in
/// [`rig::completion::CompletionResponse::raw`], which is the same reply the
/// normalized response was built from, so a cassette still records exactly
/// one interaction.
fn assert_raw_response_metadata(raw: &serde_json::Value) {
    assert_nonempty_response(raw["id"].as_str().unwrap_or_default());
    assert_nonempty_response(raw["model"].as_str().unwrap_or_default());
    assert_eq!(raw["status"], "completed");
    assert!(
        raw["usage"].is_object(),
        "raw xAI response should preserve usage metadata"
    );
}

fn assert_response_metadata(response: &rig::completion::CompletionResponse) {
    assert_nonempty_response(
        response
            .model()
            .expect("normalized xAI response should report the provider model"),
    );
    assert_eq!(
        response.finish_reason(),
        Some(rig::completion::FinishReason::Stop),
        "xAI `status: completed` should normalize to a stop finish reason"
    );
    assert!(
        crate::raw_capture::responses::message_item_id(response)
            .is_some_and(|id| id.starts_with("msg_")),
        "xAI Responses message id should be preserved, got {:?}",
        crate::raw_capture::responses::message_item_id(response)
    );
}

crate::matrix::case_matrix! {
    wrapper: with_xai_cassette_result, family: agent_tool_sessions_case;
    # [tokio :: test]
    # [ignore = "stale cassette: the shared session cell sends no `store: false`, so xAI stores its responses and the recorder refuses the re-record"]
    sequential_complex_tool_calls_nonstreaming: ("agent_tool_sessions/sequential_complex_tool_calls_nonstreaming", sequential_complex_tool_calls_nonstreaming);
    # [tokio :: test]
    # [ignore = "stale cassette: the shared session cell sends no `store: false`, so xAI stores its responses and the recorder refuses the re-record"]
    parallel_tool_calls_single_turn_nonstreaming: ("agent_tool_sessions/parallel_tool_calls_single_turn_nonstreaming", parallel_tool_calls_single_turn_nonstreaming);
}

#[tokio::test]
async fn sequential_complex_tool_calls_streaming() -> Result<()> {
    with_xai_cassette_result(
        "agent_tool_sessions/sequential_complex_tool_calls_streaming",
        |client| async move {
            let log = InvocationLog::default();
            let agent = COMPLEX_SESSION
                .equip(
                    rig::AgentBuilder::new(client.completion(SESSION_MODEL)),
                    &log,
                )
                .parallel_tool_calls(false)
                .additional_params(json!({"store": false}))
                .build();
            stream_complex_session(&agent, &COMPLEX_SESSION, &log).await
        },
    )
    .await
}

#[tokio::test]
async fn raw_stream_complex_tool_call_deltas_have_object_arguments() -> Result<()> {
    with_xai_cassette_result(
        "agent_tool_sessions/raw_stream_complex_tool_call_deltas_have_object_arguments",
        |client| async move {
            let model = client.completion(SESSION_MODEL);
            let tool = InspectManifest::default();
            let request = CompletionRequest::new(
                    "Call inspect_manifest exactly once for project rig-xai with critical=true, retries=2, \
                     steps [{name: plan, weight: 1}, {name: verify, weight: 2}], and note `streamed nested JSON`. \
                     Do not write normal text before the tool call.",
                )
                .preamble("Use the requested tool call and no prose before it.")
                .tool(rig::tool::tool_definition(&tool))
                .tool_choice(ToolChoice::Required)
                .additional_params(json!({ "store": false }));

            let observation = collect_raw_stream_observation(model.stream(request)?).await;

            assert_raw_stream_tool_call_arguments_are_objects(
                &observation,
                &[InspectManifest::NAME],
            );
            let tool_call = observation
                .tool_calls
                .iter()
                .find(|tool_call| tool_call.function.name == InspectManifest::NAME)
                .ok_or_else(|| anyhow::anyhow!("raw stream should emit inspect_manifest"))?;
            anyhow::ensure!(tool_call.function.arguments["project"] == "rig-xai");
            anyhow::ensure!(tool_call.function.arguments["flags"]["critical"] == true);
            anyhow::ensure!(
                tool_call.function.arguments["steps"].as_array().map(Vec::len) == Some(2)
            );

            Ok(())
        },
    )
    .await
}

#[tokio::test]
async fn long_history_replay_with_tool_result_continuation() -> Result<()> {
    with_xai_cassette_result(
        "agent_tool_sessions/long_history_replay_with_tool_result_continuation",
        |client| async move {
            let model = client.completion(SESSION_MODEL);
            let request = CompletionRequest::new(
                    "Answer in one short sentence: what is my favorite color, which label came from the tool, \
                     and which release lane did I choose? Do not call any tools.",
                )
                .preamble("You are concise and should rely on the provided chat history.")
                .message(Message::user("My favorite color is teal. Please remember it."))
                .message(Message::assistant("Noted: your favorite color is teal."))
                .message(Message::user("For this release, use the canary lane."))
                .message(Message::assistant("Understood: the release lane is canary."))
                .message(Message::user("Look up the harbor label with the tool."))
                .message(Message::Assistant(rig_core::message::AssistantMessage::new(vec![AssistantContent::tool_call(
                        "call_REDACTED_1",
                        rig_core::message::ToolName::new(AlphaSignal::NAME).expect("tool name"),
                        json!({}),
                    )])))
                .message(Message::tool_result(rig_core::message::CallId::from_wire("call_REDACTED_1"), rig_core::message::ToolName::new(AlphaSignal::NAME).expect("tool name"), ALPHA_SIGNAL_OUTPUT))
                .message(Message::assistant("The harbor label is crimson-harbor."))
                .tool(rig::tool::tool_definition(&AlphaSignal))
                .tool_choice(ToolChoice::None)
                .additional_params(json!({ "store": false }));

            let response = model.call(request).await?;
            let raw = &response.raw;
            assert_raw_response_metadata(raw);
            let text = assistant_text_response(&response.choice)
                .ok_or_else(|| anyhow::anyhow!("response should include assistant text"))?;

            assert_contains_all_case_insensitive(&text, &["teal", ALPHA_SIGNAL_OUTPUT, "canary"]);
            anyhow::ensure!(
                response.usage.input_tokens.is_some_and(|n| n > 0) && response.usage.output_tokens.is_some_and(|n| n > 0),
                "usage should be populated on long-history replay: {:?}",
                response.usage
            );
            assert_response_metadata(&response);

            Ok(())
        },
    )
    .await
}

#[tokio::test]
async fn tool_choice_required_specific_and_none() -> Result<()> {
    with_xai_cassette_result(
        "agent_tool_sessions/tool_choice_required_specific_and_none",
        |client| async move {
            let model = client.completion(SESSION_MODEL);

            let required = model
                .call(CompletionRequest::new(
                            "Call lookup_harbor_label exactly once with an empty object and do not answer in prose.",
                        )
                        .tool(rig::tool::tool_definition(&AlphaSignal))
                        .tool_choice(ToolChoice::Required)
                        .additional_params(json!({ "store": false })))
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
                        .additional_params(json!({ "store": false })))
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
                        .additional_params(json!({ "store": false })))
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
async fn reasoning_effort_preserves_reasoning_content_and_usage() -> Result<()> {
    with_xai_cassette_result(
        "agent_tool_sessions/reasoning_effort_preserves_reasoning_content_and_usage",
        |client| async move {
            let model = client.completion(REASONING_MODEL);
            let request = CompletionRequest::new(
                    "Use concise reasoning to solve: if three probes each verify two cassettes, how many cassette verifications occur? Answer with the number.",
                )
                .preamble("You are a concise reliability engineer.")
                .reasoning(Effort::Low)
                .additional_params(json!({ "reasoning": { "summary": "detailed" }, "store": false }));

            let response = model.call(request).await?;
            let raw = &response.raw;

            anyhow::ensure!(
                raw["output"]
                    .as_array()
                    .is_some_and(|items| items.iter().any(|item| item["type"] == "reasoning")),
                "raw xAI output should preserve provider reasoning item"
            );
            let raw_reasoning_tokens =
                raw["usage"]["output_tokens_details"]["reasoning_tokens"].as_u64();
            assert_raw_response_metadata(raw);

            anyhow::ensure!(
                response
                    .choice
                    .iter()
                    .any(|content| matches!(content, AssistantContent::Reasoning(_))),
                "xAI reasoning response should preserve a reasoning content block"
            );
            anyhow::ensure!(
                response.usage.reasoning_tokens.is_some_and(|n| n > 0),
                "core usage should preserve xAI reasoning tokens: {:?}",
                response.usage
            );
            anyhow::ensure!(
                response.usage.reasoning_tokens == raw_reasoning_tokens
                    && raw_reasoning_tokens.is_some_and(|n| n > 0),
                "usage reasoning tokens should match raw provider details"
            );
            assert_response_metadata(&response);

            Ok(())
        },
    )
    .await
}

#[tokio::test]
async fn nested_json_schema_response_format_roundtrip() -> Result<()> {
    with_xai_cassette_result(
        "agent_tool_sessions/nested_json_schema_response_format_roundtrip",
        |client| async move {
            let model = client.completion(SESSION_MODEL);
            let request = CompletionRequest::new(
                    "Return the xAI cassette release validation plan with lane canary, risk low, and checks compile=true and replay=true.",
                )
                .preamble("Return only JSON matching the supplied schema.")
                .additional_params(json!({
                    "store": false,
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "xai_release_plan",
                            "strict": true,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "release": {
                                        "type": "object",
                                        "properties": {
                                            "lane": { "type": "string" },
                                            "risk": { "type": "string" }
                                        },
                                        "required": ["lane", "risk"],
                                        "additionalProperties": false
                                    },
                                    "checks": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": { "type": "string" },
                                                "required": { "type": "boolean" }
                                            },
                                            "required": ["name", "required"],
                                            "additionalProperties": false
                                        }
                                    }
                                },
                                "required": ["release", "checks"],
                                "additionalProperties": false
                            }
                        }
                    }
                }));

            let response = model.call(request).await?;
            let raw = &response.raw;
            assert_raw_response_metadata(raw);
            let text = assistant_text_response(&response.choice)
                .ok_or_else(|| anyhow::anyhow!("schema response should contain text"))?;
            let plan: serde_json::Value = serde_json::from_str(&text)?;

            anyhow::ensure!(plan["release"]["lane"] == "canary");
            anyhow::ensure!(plan["release"]["risk"] == "low");
            let checks = plan["checks"]
                .as_array()
                .ok_or_else(|| anyhow::anyhow!("checks should be an array"))?;
            anyhow::ensure!(
                checks
                    .iter()
                    .any(|check| check["name"] == "compile" && check["required"] == true),
                "structured output should include compile=true"
            );
            anyhow::ensure!(
                checks
                    .iter()
                    .any(|check| check["name"] == "replay" && check["required"] == true),
                "structured output should include replay=true"
            );
            assert_response_metadata(&response);

            Ok(())
        },
    )
    .await
}
