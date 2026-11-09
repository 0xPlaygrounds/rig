//! Cassette-backed Mistral long-session and tool-contract regression tests.
//!
//! These scenarios exercise Rig's Mistral chat-completions path with
//! multi-turn tool loops, real SSE streaming, complex JSON tool arguments,
//! caller-owned history, native/JSON-object structured output, tool choice, and
//! provider usage/metadata preservation.

use anyhow::Result;
use rig::completion::Message;
use rig::message::{AssistantContent, ToolChoice};
use rig::providers::mistral;
use rig::tool::Tool;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use crate::matrix::agent_tool_sessions::{ComplexSession, InspectManifest, ProbeNote};
use crate::support::{
    ALPHA_SIGNAL_OUTPUT, AlphaSignal, BETA_SIGNAL_OUTPUT, BetaSignal, TWO_TOOL_STREAM_PREAMBLE,
    TWO_TOOL_STREAM_PROMPT, assert_contains_all_case_insensitive, assert_nonempty_response,
    assert_raw_stream_tool_call_arguments_are_objects, assert_two_tool_roundtrip_contract,
    assistant_text_response, collect_raw_stream_observation, collect_stream_observation,
};

use super::support::with_mistral_cassette_result;
use rig::completion::CompletionRequest;

const SESSION_MODEL: &str = mistral::MISTRAL_SMALL;
const SESSION_MAX_TOKENS: Option<u64> = None;
const STRUCTURED_MODEL: &str = mistral::MISTRAL_SMALL;

const COMPLEX_SESSION: ComplexSession = ComplexSession {
    preamble: "\
You are a deterministic Mistral tool orchestration test harness. Use the tools instead of inventing values. \
For the production-readiness scenario, call exactly one tool at a time in this order: \
1. ping_empty with an empty JSON object. \
2. inspect_manifest with project rig-mistral, flags critical=true and retries=2, steps plan weight=1 and verify weight=2, and the exact note from the user. \
3. join_labels with labels [north, beta gamma, quote:\"delta\", slash\\path] and separator |. \
4. optional_nullable_probe with required name sentinel, optional note omitted if possible, and nullable_code null. \
5. escape_echo with the exact escaped text from the user. \
After all tool results are available, answer in one short sentence that includes EMPTY-OK, MANIFEST-OK, LABELS-OK, OPTIONAL-OK, and ESCAPE-OK.",
    prompt: "\
Run the production-readiness scenario. The manifest note is `line one; line two says \"hello\" and path C:\\rig\\mistral`. \
The escaped text is `Line 1\nLine \"2\" with backslash \\ and unicode snowman ☃`.",
    project: "rig-mistral",
    note: "line one; line two says \"hello\" and path C:\\rig\\mistral",
    escaped_text: "Line 1\nLine \"2\" with backslash \\ and unicode snowman ☃",
    optional_probe: Some(ProbeNote::String),
};

/// Run one completion and return both Mistral's own reply document, typed, and
/// the normalized response the completion path derived from it.
///
/// [`rig::completion::CompletionResponse::raw`] is the escape hatch for the
/// provider-specific fields the normalized response does not carry, and
/// Mistral's own response type reads it back. Doing that here keeps the
/// provider-native-versus-normalized checks below on a single cassette
/// interaction.
async fn raw_and_normalized_completion<
    W: rig_core::wire::Wire<Op = rig_core::operation::Completion>,
    T: rig_core::driver::Transport<W>,
>(
    model: &rig_core::driver::Model<W, T>,
    request: rig::completion::CompletionRequest,
) -> Result<(serde_json::Value, rig::completion::CompletionResponse)> {
    let normalized = model.call(request).await?;
    Ok((normalized.raw.clone(), normalized))
}

fn assert_response_metadata(
    response: &rig::completion::CompletionResponse,
    raw: &serde_json::Value,
) {
    assert_nonempty_response(raw["id"].as_str().unwrap_or_default());
    assert_eq!(response.response_id(), raw["id"].as_str());
    assert_nonempty_response(raw["model"].as_str().unwrap_or_default());
    assert!(
        raw["choices"]
            .as_array()
            .into_iter()
            .flatten()
            .all(|choice| choice["finish_reason"]
                .as_str()
                .is_some_and(|reason| !reason.is_empty())),
        "raw Mistral choices should preserve finish reasons"
    );
    let raw_usage = &raw["usage"];
    assert!(
        response.usage.input_tokens.is_some_and(|n| n > 0),
        "usage should include input tokens"
    );
    assert_eq!(
        response.usage.output_tokens,
        raw_usage["completion_tokens"].as_u64()
    );
    assert_eq!(
        response.usage.total_tokens,
        raw_usage["total_tokens"].as_u64()
    );
}

crate::matrix::case_matrix! {
    wrapper: with_mistral_cassette_result, family: agent_tool_sessions_case;
    # [tokio :: test]
    # [ignore = "stale cassette: its request predates item-shaped history, and Mistral rate-limited the re-record"]
    sequential_complex_tool_calls_nonstreaming: ("agent_tool_sessions/sequential_complex_tool_calls_nonstreaming", sequential_complex_tool_calls_nonstreaming);
    # [tokio :: test]
    # [ignore = "stale cassette: its request predates item-shaped history, and Mistral rate-limited the re-record"]
    sequential_complex_tool_calls_streaming: ("agent_tool_sessions/sequential_complex_tool_calls_streaming", sequential_complex_tool_calls_streaming);
    # [tokio :: test]
    # [ignore = "stale cassette: its request predates item-shaped history, and Mistral rate-limited the re-record"]
    parallel_tool_calls_single_turn_nonstreaming: ("agent_tool_sessions/parallel_tool_calls_single_turn_nonstreaming", parallel_tool_calls_single_turn_nonstreaming);
    # [tokio :: test]
    # [ignore = "stale cassette: its request predates item-shaped history, and Mistral rate-limited the re-record"]
    parallel_tool_calls_single_turn_streaming: ("agent_tool_sessions/parallel_tool_calls_single_turn_streaming", parallel_tool_calls_single_turn_streaming);
    # [tokio :: test]
    # [ignore = "stale cassette: its request predates item-shaped history, and Mistral rate-limited the re-record"]
    long_history_replay_with_tool_result_continuation: ("agent_tool_sessions/long_history_replay_with_tool_result_continuation", long_history_replay_with_tool_result_continuation);
}

#[tokio::test]
async fn raw_stream_complex_tool_call_deltas_have_object_arguments() -> Result<()> {
    with_mistral_cassette_result(
        "agent_tool_sessions/raw_stream_complex_tool_call_deltas_have_object_arguments",
        |client| async move {
            let model = client.completion(SESSION_MODEL);
            let tool = InspectManifest::default();
            let request = CompletionRequest::new(
                    "Call inspect_manifest exactly once for project rig-mistral with critical=true, retries=2, \
                     steps [{name: plan, weight: 1}, {name: verify, weight: 2}], and note `streamed nested JSON`. \
                     Do not write normal text before the tool call.",
                )
                .preamble("Use the requested tool call and no prose before it.")
                .tool(rig::tool::tool_definition(&tool))
                .tool_choice(ToolChoice::Required);

            let observation = collect_raw_stream_observation(model.stream(request)?).await;

            assert_raw_stream_tool_call_arguments_are_objects(&observation, &[InspectManifest::NAME]);
            let tool_call = observation
                .tool_calls
                .iter()
                .find(|tool_call| tool_call.function.name == InspectManifest::NAME)
                .ok_or_else(|| anyhow::anyhow!("raw stream should emit inspect_manifest"))?;
            anyhow::ensure!(tool_call.function.arguments["project"] == "rig-mistral");
            anyhow::ensure!(tool_call.function.arguments["flags"]["critical"] == true);
            anyhow::ensure!(tool_call.function.arguments["steps"].as_array().map(Vec::len) == Some(2));

            Ok(())
        },
    )
    .await
}

#[tokio::test]
async fn tool_choice_auto_any_specific_and_none() -> Result<()> {
    with_mistral_cassette_result(
        "agent_tool_sessions/tool_choice_auto_any_specific_and_none",
        |client| async move {
            let model = client.completion(SESSION_MODEL);

            let auto = model
                .call(CompletionRequest::new("Call lookup_harbor_label exactly once with an empty object.")
                        .tool(rig::tool::tool_definition(&AlphaSignal))
                        .tool_choice(ToolChoice::Auto))
                .await?;
            anyhow::ensure!(
                auto.choice.iter().any(|content| matches!(
                    content,
                    AssistantContent::ToolCall(tool_call)
                        if tool_call.function.name == AlphaSignal::NAME
                            && tool_call.function.arguments_value() == json!({})
                )),
                "auto tool choice should allow lookup_harbor_label"
            );

            let any = model
                .call(CompletionRequest::new("Call lookup_harbor_label exactly once with an empty object and do not answer in prose.")
                        .tool(rig::tool::tool_definition(&AlphaSignal))
                        .tool_choice(ToolChoice::Required))
                .await?;
            anyhow::ensure!(
                any.choice.iter().any(|content| matches!(
                    content,
                    AssistantContent::ToolCall(tool_call)
                        if tool_call.function.name == AlphaSignal::NAME
                            && tool_call.function.arguments_value() == json!({})
                )),
                "required tool choice should serialize to Mistral `any` and force lookup_harbor_label"
            );

            let specific = model
                .call(CompletionRequest::new("Call the orchard-label tool exactly once with an empty object and do not call any other tool.")
                        .tool(rig::tool::tool_definition(&AlphaSignal))
                        .tool(rig::tool::tool_definition(&BetaSignal))
                        .tool_choice(ToolChoice::Specific {
                            function_names: vec![rig_core::message::ToolName::new(BetaSignal::NAME).expect("tool name")],
                        }))
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
                .call(CompletionRequest::new("Do not call tools. Reply with exactly this phrase: no-tool-answer")
                        .tool(rig::tool::tool_definition(&AlphaSignal))
                        .tool_choice(ToolChoice::None))
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

#[derive(Debug, Deserialize, JsonSchema)]
struct StructuredReleasePlan {
    lane: String,
    checks: StructuredChecks,
    risk: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct StructuredChecks {
    compile: bool,
    replay: bool,
}

#[tokio::test]
async fn json_schema_structured_output_roundtrip() -> Result<()> {
    with_mistral_cassette_result(
        "agent_tool_sessions/json_schema_structured_output_roundtrip",
        |client| async move {
            let model = client.completion(STRUCTURED_MODEL);
            let request = CompletionRequest::new(
                "Return lane=canary, risk=low, checks.compile=true, and checks.replay=true.",
            )
            .preamble("Return only the requested structured object.")
            .output_schema(schemars::schema_for!(StructuredReleasePlan));

            let (raw, response) = raw_and_normalized_completion(&model, request).await?;
            let text = assistant_text_response(&response.choice)
                .ok_or_else(|| anyhow::anyhow!("structured response should contain text"))?;
            let plan: StructuredReleasePlan = serde_json::from_str(&text)?;

            anyhow::ensure!(plan.lane.eq_ignore_ascii_case("canary"));
            anyhow::ensure!(plan.risk.eq_ignore_ascii_case("low"));
            anyhow::ensure!(plan.checks.compile);
            anyhow::ensure!(plan.checks.replay);
            assert_response_metadata(&response, &raw);

            Ok(())
        },
    )
    .await
}
