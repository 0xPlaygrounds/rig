//! Cassette-backed Groq long-session and OpenAI-compatible chat regression tests.
//!
//! These scenarios exercise Rig's Groq chat-completions path with multi-turn
//! tool loops, low-latency SSE streaming, complex JSON tool arguments, caller-
//! owned history, JSON response formats, explicit tool choice, usage accounting,
//! and provider metadata preservation.

use rig::streaming::Item;

use anyhow::Result;
use futures::StreamExt;
use rig::completion::Message;
use rig::message::{AssistantContent, ToolChoice};
use rig::providers::openai;
use rig::streaming::StreamEvent;
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

use super::support::with_groq_cassette_result;
use rig::completion::CompletionRequest;

const SESSION_MODEL: &str = "qwen/qwen3.8-27b";
/// Groq rejects `qwen/qwen3.8-27b` requests whose output cap exceeds its
/// 1,000-token-per-minute limit, and an unset cap counts as 2,048.
const SESSION_MAX_TOKENS: Option<u64> = Some(512);
const JSON_OBJECT_MODEL: &str = "qwen/qwen3.8-27b";
const JSON_SCHEMA_MODEL: &str = "openai/gpt-oss-20b";
/// `qwen/qwen3.8-27b` refuses the verbatim-phrase prompt about one time in
/// five, while `gpt-oss` fills empty tool arguments with `{"": {}}`, so the
/// no-tool step alone runs on `gpt-oss`.
const TOOL_CHOICE_NONE_MODEL: &str = "openai/gpt-oss-120b";

const COMPLEX_SESSION: ComplexSession = ComplexSession {
    preamble: "\
You are a deterministic Groq tool orchestration test harness. Use the tools instead of inventing values. \
For the production-readiness scenario, call exactly one tool at a time in this order: \
1. ping_empty with an empty JSON object. \
2. inspect_manifest with project rig-groq, flags critical=true and retries=2, steps plan weight=1 and verify weight=2, and the exact note from the user. \
3. join_labels with labels [north, beta gamma, quote:\"delta\", slash\\path] and separator |. \
4. optional_nullable_probe with required name sentinel, optional note omitted if possible, and nullable_code null. \
5. escape_echo with the exact escaped text from the user. \
After all tool results are available, answer in one short sentence that includes EMPTY-OK, MANIFEST-OK, LABELS-OK, OPTIONAL-OK, and ESCAPE-OK.",
    prompt: "\
Run the production-readiness scenario. The manifest note is `line one; line two says \"hello\" and path C:/rig/groq`. \
The escaped text is `Line 1\nLine \"2\" with colon: ok and unicode snowman ☃`.",
    project: "rig-groq",
    note: "line one; line two says \"hello\" and path C:/rig/groq",
    escaped_text: "Line 1\nLine \"2\" with colon: ok and unicode snowman ☃",
    optional_probe: Some(ProbeNote::Nullable),
};

/// The Groq wire metadata the normalized response deliberately does not carry.
///
/// The fields the parity assertions need are read out of the reply document.
struct RawResponseMetadata {
    id: String,
    model: String,
    system_fingerprint: Option<String>,
    finish_reasons: Vec<String>,
    usage: Option<openai::Usage>,
}

impl RawResponseMetadata {
    fn capture(raw: &serde_json::Value) -> Self {
        Self {
            id: raw["id"].as_str().unwrap_or_default().to_owned(),
            model: raw["model"].as_str().unwrap_or_default().to_owned(),
            system_fingerprint: raw["system_fingerprint"].as_str().map(str::to_owned),
            finish_reasons: raw["choices"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|choice| choice["finish_reason"].as_str().map(str::to_owned))
                .collect(),
            usage: serde_json::from_value(raw["usage"].clone()).ok(),
        }
    }
}

/// Run one completion and return both Groq's own wire metadata — read back
/// out of the reply document the driver captured — and the normalized
/// response the same reply folded into.
///
/// [`rig::completion::CompletionResponse::raw`] is the escape hatch for the
/// provider-specific fields the normalized response does not carry
/// (`system_fingerprint`, Groq's queue and total timings): it holds the
/// provider's reply verbatim, so reading it back as the shared OpenAI type
/// keeps both views on a single cassette interaction.
async fn raw_and_normalized_completion<
    W: rig_core::wire::Wire<Op = rig_core::operation::Completion>,
    T: rig_core::driver::Transport<W>,
>(
    model: &rig_core::driver::Model<W, T>,
    request: rig::completion::CompletionRequest,
) -> Result<(RawResponseMetadata, rig::completion::CompletionResponse)> {
    let normalized = model.call(request).await?;
    let metadata = RawResponseMetadata::capture(&normalized.raw);
    Ok((metadata, normalized))
}

fn assert_response_metadata(
    response: &rig::completion::CompletionResponse,
    raw: &RawResponseMetadata,
) {
    assert_nonempty_response(&raw.id);
    assert_nonempty_response(&raw.model);
    if let Some(system_fingerprint) = &raw.system_fingerprint {
        assert_nonempty_response(system_fingerprint);
    }
    assert!(
        raw.finish_reasons
            .iter()
            .all(|finish_reason| !finish_reason.is_empty()),
        "raw Groq choices should preserve finish reasons"
    );
    let raw_usage = raw
        .usage
        .as_ref()
        .expect("raw response should preserve usage");
    assert!(
        response.usage.input_tokens.is_some_and(|n| n > 0),
        "usage should include input tokens"
    );
    if let Some(completion_tokens) = raw_usage.completion_tokens {
        assert_eq!(response.usage.output_tokens, Some(completion_tokens as u64));
    }
    assert_eq!(
        response.usage.total_tokens,
        Some(raw_usage.total_tokens as u64)
    );
    if let Some(queue_time) = raw_usage.queue_time {
        assert!(queue_time >= 0.0);
    }
    if let Some(total_time) = raw_usage.total_time {
        assert!(total_time >= 0.0);
    }
}

crate::matrix::case_matrix! {
    wrapper: with_groq_cassette_result, family: agent_tool_sessions_case;
    # [tokio :: test]
    sequential_complex_tool_calls_nonstreaming: ("agent_tool_sessions/sequential_complex_tool_calls_nonstreaming", sequential_complex_tool_calls_nonstreaming);
    # [tokio :: test]
    sequential_complex_tool_calls_streaming: ("agent_tool_sessions/sequential_complex_tool_calls_streaming", sequential_complex_tool_calls_streaming);
    # [tokio :: test]
    parallel_tool_calls_single_turn_nonstreaming: ("agent_tool_sessions/parallel_tool_calls_single_turn_nonstreaming", parallel_tool_calls_single_turn_nonstreaming);
    # [tokio :: test]
    parallel_tool_calls_single_turn_streaming: ("agent_tool_sessions/parallel_tool_calls_single_turn_streaming", parallel_tool_calls_single_turn_streaming);
    # [tokio :: test]
    long_history_replay_with_tool_result_continuation: ("agent_tool_sessions/long_history_replay_with_tool_result_continuation", long_history_replay_with_tool_result_continuation);
}

#[tokio::test]
async fn raw_stream_complex_tool_call_deltas_have_object_arguments() -> Result<()> {
    with_groq_cassette_result(
        "agent_tool_sessions/raw_stream_complex_tool_call_deltas_have_object_arguments",
        |client| async move {
            let model = client.completion(SESSION_MODEL);
            let tool = InspectManifest::default();
            let request = CompletionRequest::new(
                    "Call inspect_manifest exactly once for project rig-groq with critical=true, retries=2, \
                     steps [{name: plan, weight: 1}, {name: verify, weight: 2}], and note `streamed nested JSON`. \
                     Do not write normal text before the tool call.",
                )
                .preamble("Use the requested tool call and no prose before it.")
                .tool(rig::tool::tool_definition(&tool))
                .tool_choice(ToolChoice::Required)
                .max_tokens(SESSION_MAX_TOKENS);

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
            anyhow::ensure!(tool_call.function.arguments["project"] == "rig-groq");
            anyhow::ensure!(tool_call.function.arguments["flags"]["critical"] == true);
            anyhow::ensure!(
                tool_call.function.arguments["steps"].as_array().map(Vec::len) == Some(2)
            );
            anyhow::ensure!(
                observation.events.contains(&"tool_call_delta"),
                "raw stream should surface tool-call deltas before final tool calls"
            );

            Ok(())
        },
    )
    .await
}

#[tokio::test]
async fn tool_choice_auto_required_specific_and_none() -> Result<()> {
    with_groq_cassette_result(
        "agent_tool_sessions/tool_choice_auto_required_specific_and_none",
        |client| async move {
            let model = client.completion(SESSION_MODEL);

            let auto = model
                .call(CompletionRequest::new("Call lookup_harbor_label exactly once with an empty object.")
                        .tool(rig::tool::tool_definition(&AlphaSignal))
                        .tool_choice(ToolChoice::Auto)
                        .max_tokens(SESSION_MAX_TOKENS))
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

            let required = model
                .call(CompletionRequest::new("Call lookup_harbor_label exactly once with an empty object and do not answer in prose.")
                        .tool(rig::tool::tool_definition(&AlphaSignal))
                        .tool_choice(ToolChoice::Required)
                        .max_tokens(SESSION_MAX_TOKENS))
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
                .call(CompletionRequest::new("Call the orchard-label tool exactly once with an empty object and do not call any other tool.")
                        .tool(rig::tool::tool_definition(&AlphaSignal))
                        .tool(rig::tool::tool_definition(&BetaSignal))
                        .tool_choice(ToolChoice::Specific {
                            function_names: vec![rig_core::message::ToolName::new(BetaSignal::NAME).expect("tool name")],
                        })
                        .max_tokens(SESSION_MAX_TOKENS))
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

            let none_model = client.completion(TOOL_CHOICE_NONE_MODEL);
            let none = none_model
                .call(CompletionRequest::new("Do not call tools. Reply with exactly this phrase: no-tool-answer")
                        .tool(rig::tool::tool_definition(&AlphaSignal))
                        .tool_choice(ToolChoice::None)
                        .max_tokens(SESSION_MAX_TOKENS))
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
async fn json_object_response_format_roundtrip() -> Result<()> {
    with_groq_cassette_result(
        "agent_tool_sessions/json_object_response_format_roundtrip",
        |client| async move {
            let model = client.completion(JSON_OBJECT_MODEL);
            let request = CompletionRequest::new(
                    "Return a JSON object with release lane canary, risk low, and checks compile=true and replay=true.",
                )
                .preamble("Return only valid JSON. No markdown.")
                .additional_params(json!({"response_format": { "type": "json_object" }}))
                .max_tokens(128);

            let (raw, response) = raw_and_normalized_completion(&model, request).await?;
            let text = assistant_text_response(&response.choice)
                .ok_or_else(|| anyhow::anyhow!("JSON response should contain text"))?;
            let plan: serde_json::Value = serde_json::from_str(&text)?;

            let serialized = plan.to_string();
            assert_contains_all_case_insensitive(
                &serialized,
                &["canary", "low", "compile", "replay"],
            );
            assert_response_metadata(&response, &raw);

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
    with_groq_cassette_result(
        "agent_tool_sessions/json_schema_structured_output_roundtrip",
        |client| async move {
            let model = client.completion(JSON_SCHEMA_MODEL);
            let request = CompletionRequest::new(
                "Return lane=canary, risk=low, checks.compile=true, and checks.replay=true.",
            )
            .preamble("Return only the requested structured object.")
            .output_schema(schemars::schema_for!(StructuredReleasePlan))
            .max_tokens(128);

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

#[tokio::test]
async fn low_latency_streaming_text_surfaces_final_usage() -> Result<()> {
    with_groq_cassette_result(
        "agent_tool_sessions/low_latency_streaming_text_surfaces_final_usage",
        |client| async move {
            let model = client.completion(SESSION_MODEL);
            let mut stream = model
                .stream(CompletionRequest::new(
                            "Reply with exactly this comma-separated sequence and no extra words: alpha,beta,gamma,delta,epsilon,zeta,eta,theta",
                        )
                        .preamble("Stream the requested short sequence exactly.")
                        .max_tokens(64))
                ?;

            let mut text_chunks = 0usize;
            while let Some(item) = stream.next().await {
                match item? {
                    Item::Event(StreamEvent::Text { text, .. }) if !text.is_empty() => {
                        text_chunks += 1;
                    }
                    _ => {}
                }
            }
            let usage = stream.finish().await?.usage;

            anyhow::ensure!(text_chunks > 0, "stream should emit text deltas");
            let input_tokens = usage.input_tokens.unwrap_or(0);
            let output_tokens = usage.output_tokens.unwrap_or(0);
            anyhow::ensure!(input_tokens > 0, "stream usage should include input tokens");
            anyhow::ensure!(output_tokens > 0, "stream usage should include output tokens");
            anyhow::ensure!(
                usage.total_tokens.unwrap_or(0) >= input_tokens + output_tokens,
                "stream usage totals should cover input + output tokens: {usage:?}"
            );

            Ok(())
        },
    )
    .await
}
