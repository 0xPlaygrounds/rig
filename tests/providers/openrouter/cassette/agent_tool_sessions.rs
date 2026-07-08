//! Cassette-backed OpenRouter long-session and tool-contract regression tests.
//!
//! These scenarios stress OpenRouter's OpenAI-compatible chat-completions path
//! with multi-turn tool loops, streamed tool-call deltas, complex JSON tool
//! arguments, explicit tool choice, long caller-owned chat history, and usage
//! surfaced from routed upstream providers.

use std::sync::{Arc, Mutex};

use anyhow::Result;
use rig::OneOrMany;
use rig::client::CompletionClient;
use rig::completion::{Chat, CompletionModel, Message, TypedPrompt};
use rig::message::{AssistantContent, ToolChoice, UserContent};
use rig::streaming::{StreamingChat, StreamingPrompt};
use rig::tool::Tool;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::support::{
    ALPHA_SIGNAL_OUTPUT, AlphaSignal, BETA_SIGNAL_OUTPUT, BetaSignal, TWO_TOOL_STREAM_PREAMBLE,
    TWO_TOOL_STREAM_PROMPT, assert_contains_all_case_insensitive, assert_nonempty_response,
    assert_raw_stream_tool_call_arguments_are_objects, assert_two_tool_roundtrip_contract,
    collect_raw_stream_observation, collect_stream_observation,
};

use super::super::{TOOL_MODEL, support::with_openrouter_cassette_result};

const SESSION_MODEL: &str = TOOL_MODEL;
const STRUCTURED_MODEL: &str = "google/gemini-2.5-flash";

const COMPLEX_SESSION_PREAMBLE: &str = "\
You are a deterministic OpenRouter tool orchestration test harness. Use the tools instead of inventing values. \
For the production-readiness scenario, call exactly one tool at a time in this order: \
1. ping_empty with an empty JSON object. \
2. inspect_manifest with project rig-openrouter, flags critical=true and retries=2, steps plan weight=1 and verify weight=2, and the exact note from the user. \
3. join_labels with labels [north, beta gamma, quote:\"delta\", slash\\path] and separator |. \
4. escape_echo with the exact escaped text from the user. \
After all tool results are available, answer in one short sentence that includes EMPTY-OK, MANIFEST-OK, LABELS-OK, and ESCAPE-OK.";

const COMPLEX_SESSION_PROMPT: &str = "\
Run the production-readiness scenario. The manifest note is `line one; line two says \"hello\" and path C:\\rig\\openrouter`. \
The escaped text is `Line 1\nLine \"2\" with backslash \\ and unicode snowman ☃`.";

#[derive(Clone, Debug, PartialEq)]
struct ToolInvocation {
    name: &'static str,
    args: serde_json::Value,
}

type InvocationLog = Arc<Mutex<Vec<ToolInvocation>>>;

fn push_invocation<T: Serialize>(log: &InvocationLog, name: &'static str, args: &T) {
    log.lock()
        .expect("tool invocation log lock should not be poisoned")
        .push(ToolInvocation {
            name,
            args: serde_json::to_value(args).expect("tool args should serialize"),
        });
}

#[derive(Clone)]
struct PingEmpty {
    log: InvocationLog,
}

#[derive(Clone)]
struct InspectManifest {
    log: InvocationLog,
}

#[derive(Clone)]
struct JoinLabels {
    log: InvocationLog,
}

#[derive(Clone)]
struct EscapeEcho {
    log: InvocationLog,
}

#[derive(Debug, Deserialize, Serialize)]
struct EmptyArgs {}

#[derive(Debug, Deserialize, Serialize)]
struct ManifestArgs {
    project: String,
    flags: ManifestFlags,
    steps: Vec<ManifestStep>,
    note: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct ManifestFlags {
    critical: bool,
    retries: u8,
}

#[derive(Debug, Deserialize, Serialize)]
struct ManifestStep {
    name: String,
    weight: i32,
}

#[derive(Debug, Deserialize, Serialize)]
struct JoinArgs {
    labels: Vec<String>,
    separator: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct EchoArgs {
    text: String,
}

#[derive(Debug, thiserror::Error)]
#[error("session tool error")]
struct SessionToolError;

impl Tool for PingEmpty {
    const NAME: &'static str = "ping_empty";
    type Error = SessionToolError;
    type Args = EmptyArgs;
    type Output = String;

    fn description(&self) -> String {
        "Return EMPTY-OK. This tool takes no arguments.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {},
            "required": []
        })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        push_invocation(&self.log, Self::NAME, &args);
        Ok("EMPTY-OK".to_string())
    }
}

impl Tool for InspectManifest {
    const NAME: &'static str = "inspect_manifest";
    type Error = SessionToolError;
    type Args = ManifestArgs;
    type Output = String;

    fn description(&self) -> String {
        "Validate a nested deployment manifest.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "project": { "type": "string" },
                "flags": {
                    "type": "object",
                    "properties": {
                        "critical": { "type": "boolean" },
                        "retries": { "type": "integer" }
                    },
                    "required": ["critical", "retries"]
                },
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": { "type": "string" },
                            "weight": { "type": "integer" }
                        },
                        "required": ["name", "weight"]
                    }
                },
                "note": { "type": "string" }
            },
            "required": ["project", "flags", "steps", "note"]
        })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        push_invocation(&self.log, Self::NAME, &args);
        Ok(format!(
            "MANIFEST-OK project={} steps={} retries={}",
            args.project,
            args.steps.len(),
            args.flags.retries
        ))
    }
}

impl Tool for JoinLabels {
    const NAME: &'static str = "join_labels";
    type Error = SessionToolError;
    type Args = JoinArgs;
    type Output = String;

    fn description(&self) -> String {
        "Join label strings with the requested separator.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "labels": {
                    "type": "array",
                    "items": { "type": "string" }
                },
                "separator": { "type": "string" }
            },
            "required": ["labels", "separator"]
        })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        push_invocation(&self.log, Self::NAME, &args);
        Ok(format!("LABELS-OK {}", args.labels.join(&args.separator)))
    }
}

impl Tool for EscapeEcho {
    const NAME: &'static str = "escape_echo";
    type Error = SessionToolError;
    type Args = EchoArgs;
    type Output = String;

    fn description(&self) -> String {
        "Echo a string containing escaping-sensitive characters.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "text": { "type": "string" }
            },
            "required": ["text"]
        })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        push_invocation(&self.log, Self::NAME, &args);
        Ok(format!("ESCAPE-OK {}", args.text))
    }
}

fn complex_tools(log: &InvocationLog) -> (PingEmpty, InspectManifest, JoinLabels, EscapeEcho) {
    (
        PingEmpty { log: log.clone() },
        InspectManifest { log: log.clone() },
        JoinLabels { log: log.clone() },
        EscapeEcho { log: log.clone() },
    )
}

fn assert_complex_invocations(log: &InvocationLog) {
    let invocations = log
        .lock()
        .expect("tool invocation log lock should not be poisoned")
        .clone();
    let names = invocations.iter().map(|call| call.name).collect::<Vec<_>>();
    assert_eq!(
        names,
        vec![
            PingEmpty::NAME,
            InspectManifest::NAME,
            JoinLabels::NAME,
            EscapeEcho::NAME,
        ],
        "expected one complex tool call of each shape in order"
    );

    assert_eq!(invocations[0].args, json!({}));
    assert_eq!(invocations[1].args["project"], "rig-openrouter");
    assert_eq!(
        invocations[1].args["flags"],
        json!({"critical": true, "retries": 2})
    );
    assert_eq!(
        invocations[1].args["steps"].as_array().map(Vec::len),
        Some(2)
    );
    assert_eq!(
        invocations[1].args["note"],
        "line one; line two says \"hello\" and path C:\\rig\\openrouter"
    );
    assert_eq!(
        invocations[2].args,
        json!({
            "labels": ["north", "beta gamma", "quote:\"delta\"", "slash\\path"],
            "separator": "|"
        })
    );
    assert_eq!(
        invocations[3].args["text"],
        "Line 1\nLine \"2\" with backslash \\ and unicode snowman ☃"
    );
}

struct ToolEvent {
    message_index: usize,
    name_or_id: String,
}

fn history_tool_calls(history: &[Message]) -> Vec<ToolEvent> {
    let mut calls = Vec::new();
    for (message_index, message) in history.iter().enumerate() {
        if let Message::Assistant { content, .. } = message {
            for item in content.iter() {
                if let AssistantContent::ToolCall(tool_call) = item {
                    calls.push(ToolEvent {
                        message_index,
                        name_or_id: tool_call.function.name.clone(),
                    });
                }
            }
        }
    }
    calls
}

fn history_tool_results(history: &[Message]) -> Vec<ToolEvent> {
    let mut results = Vec::new();
    for (message_index, message) in history.iter().enumerate() {
        if let Message::User { content } = message {
            for item in content.iter() {
                if let UserContent::ToolResult(tool_result) = item {
                    results.push(ToolEvent {
                        message_index,
                        name_or_id: tool_result.id.clone(),
                    });
                }
            }
        }
    }
    results
}

fn assert_history_records_sequential_tool_roundtrips(history: &[Message], expected_tools: &[&str]) {
    let calls = history_tool_calls(history);
    let results = history_tool_results(history);

    assert_eq!(
        calls
            .iter()
            .map(|call| call.name_or_id.as_str())
            .collect::<Vec<_>>(),
        expected_tools,
        "caller-owned chat history should preserve tool call order"
    );
    assert_eq!(
        results.len(),
        expected_tools.len(),
        "caller-owned chat history should contain one tool result per call"
    );

    for (index, call) in calls.iter().enumerate() {
        let result = &results[index];
        assert!(
            call.message_index < result.message_index,
            "tool result should follow its assistant tool call"
        );
        if let Some(next_call) = calls.get(index + 1) {
            assert!(
                result.message_index < next_call.message_index,
                "next tool call should occur after the previous tool result"
            );
        }
    }
}

#[tokio::test]
async fn sequential_complex_tool_calls_nonstreaming() -> Result<()> {
    with_openrouter_cassette_result(
        "agent_tool_sessions/sequential_complex_tool_calls_nonstreaming",
        |client| async move {
            let log = Arc::new(Mutex::new(Vec::new()));
            let (ping, manifest, labels, echo) = complex_tools(&log);
            let agent = client
                .agent(SESSION_MODEL)
                .preamble(COMPLEX_SESSION_PREAMBLE)
                .tool(ping)
                .tool(manifest)
                .tool(labels)
                .tool(echo)
                .additional_params(json!({"parallel_tool_calls": false}))
                .default_max_turns(10)
                .build();
            let mut history = Vec::<Message>::new();

            let response = agent.chat(COMPLEX_SESSION_PROMPT, &mut history).await?;

            assert_contains_all_case_insensitive(
                &response,
                &["EMPTY-OK", "MANIFEST-OK", "LABELS-OK", "ESCAPE-OK"],
            );
            assert_complex_invocations(&log);
            assert_history_records_sequential_tool_roundtrips(
                &history,
                &[
                    PingEmpty::NAME,
                    InspectManifest::NAME,
                    JoinLabels::NAME,
                    EscapeEcho::NAME,
                ],
            );

            Ok(())
        },
    )
    .await
}

#[tokio::test]
async fn sequential_complex_tool_calls_streaming() -> Result<()> {
    with_openrouter_cassette_result(
        "agent_tool_sessions/sequential_complex_tool_calls_streaming",
        |client| async move {
            let log = Arc::new(Mutex::new(Vec::new()));
            let (ping, manifest, labels, echo) = complex_tools(&log);
            let agent = client
                .agent(SESSION_MODEL)
                .preamble(COMPLEX_SESSION_PREAMBLE)
                .tool(ping)
                .tool(manifest)
                .tool(labels)
                .tool(echo)
                .additional_params(json!({"parallel_tool_calls": false}))
                .build();

            let mut stream = agent
                .stream_chat(COMPLEX_SESSION_PROMPT, Vec::<Message>::new())
                .multi_turn(10)
                .await;
            let observation = collect_stream_observation(&mut stream).await;

            anyhow::ensure!(
                observation.errors.is_empty(),
                "stream should not emit errors: {:?}",
                observation.errors
            );
            let expected_tool_calls = vec![
                PingEmpty::NAME.to_string(),
                InspectManifest::NAME.to_string(),
                JoinLabels::NAME.to_string(),
                EscapeEcho::NAME.to_string(),
            ];
            anyhow::ensure!(
                observation.tool_calls == expected_tool_calls,
                "stream should expose the same ordered tool calls as non-streaming; saw {:?}",
                observation.tool_calls
            );
            anyhow::ensure!(
                observation.tool_results == 4,
                "expected 4 streamed tool results, saw {}",
                observation.tool_results
            );
            anyhow::ensure!(
                observation.got_final_response,
                "stream should emit a final response"
            );
            let response = observation
                .final_response_text
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("stream should produce final response text"))?;
            assert_contains_all_case_insensitive(
                response,
                &["EMPTY-OK", "MANIFEST-OK", "LABELS-OK", "ESCAPE-OK"],
            );
            assert_complex_invocations(&log);

            Ok(())
        },
    )
    .await
}

#[tokio::test]
async fn parallel_tool_calls_single_turn_nonstreaming() -> Result<()> {
    with_openrouter_cassette_result(
        "agent_tool_sessions/parallel_tool_calls_single_turn_nonstreaming",
        |client| async move {
            let agent = client
                .agent(SESSION_MODEL)
                .preamble(TWO_TOOL_STREAM_PREAMBLE)
                .tool(AlphaSignal)
                .tool(BetaSignal)
                .default_max_turns(5)
                .build();
            let mut history = Vec::<Message>::new();

            let response = agent.chat(TWO_TOOL_STREAM_PROMPT, &mut history).await?;

            assert_contains_all_case_insensitive(
                &response,
                &[ALPHA_SIGNAL_OUTPUT, BETA_SIGNAL_OUTPUT],
            );
            let calls = history_tool_calls(&history);
            let call_names = calls
                .iter()
                .map(|call| call.name_or_id.as_str())
                .collect::<Vec<_>>();
            anyhow::ensure!(
                calls.len() == 2
                    && call_names.contains(&AlphaSignal::NAME)
                    && call_names.contains(&BetaSignal::NAME),
                "expected both zero-argument tools in one model turn, saw {:?}",
                call_names
            );
            anyhow::ensure!(
                calls[0].message_index == calls[1].message_index,
                "parallel tool calls should be recorded on one assistant message"
            );
            let result_count = history_tool_results(&history).len();
            anyhow::ensure!(
                result_count == 2,
                "expected two tool results, saw {result_count}"
            );

            Ok(())
        },
    )
    .await
}

#[tokio::test]
async fn parallel_tool_calls_single_turn_streaming() -> Result<()> {
    with_openrouter_cassette_result(
        "agent_tool_sessions/parallel_tool_calls_single_turn_streaming",
        |client| async move {
            let agent = client
                .agent(SESSION_MODEL)
                .preamble(TWO_TOOL_STREAM_PREAMBLE)
                .tool(AlphaSignal)
                .tool(BetaSignal)
                .build();

            let mut stream = agent
                .stream_prompt(TWO_TOOL_STREAM_PROMPT)
                .multi_turn(5)
                .await;
            let observation = collect_stream_observation(&mut stream).await;

            assert_two_tool_roundtrip_contract(
                &observation,
                &[AlphaSignal::NAME, BetaSignal::NAME],
                &[ALPHA_SIGNAL_OUTPUT, BETA_SIGNAL_OUTPUT],
            );

            Ok(())
        },
    )
    .await
}

#[tokio::test]
async fn raw_stream_complex_tool_call_deltas_have_object_arguments() -> Result<()> {
    with_openrouter_cassette_result(
        "agent_tool_sessions/raw_stream_complex_tool_call_deltas_have_object_arguments",
        |client| async move {
            let log = Arc::new(Mutex::new(Vec::new()));
            let model = client.completion_model(SESSION_MODEL);
            let tool = InspectManifest { log };
            let request = model
                .completion_request(
                    "Call inspect_manifest exactly once for project rig-openrouter with critical=true, retries=2, \
                     steps [{name: plan, weight: 1}, {name: verify, weight: 2}], and note `streamed nested JSON`. \
                     Do not write normal text before the tool call.",
                )
                .preamble("Use the requested tool call and no prose before it.".to_string())
                .tool(rig::tool::tool_definition(&tool))
                .tool_choice(ToolChoice::Required)
                .build();

            let observation = collect_raw_stream_observation(model.stream(request).await?).await;

            assert_raw_stream_tool_call_arguments_are_objects(
                &observation,
                &[InspectManifest::NAME],
            );
            let tool_call = observation
                .tool_calls
                .iter()
                .find(|tool_call| tool_call.function.name == InspectManifest::NAME)
                .ok_or_else(|| anyhow::anyhow!("raw stream should emit inspect_manifest"))?;
            anyhow::ensure!(tool_call.function.arguments["project"] == "rig-openrouter");
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
    with_openrouter_cassette_result(
        "agent_tool_sessions/long_history_replay_with_tool_result_continuation",
        |client| async move {
            let model = client.completion_model(SESSION_MODEL);
            let request = model
                .completion_request(
                    "Answer in one short sentence: what is my favorite color, which label came from the tool, \
                     and which release lane did I choose? Do not call any tools.",
                )
                .preamble("You are concise and should rely on the provided chat history.".to_string())
                .message(Message::user("My favorite color is teal. Please remember it."))
                .message(Message::assistant("Noted: your favorite color is teal."))
                .message(Message::user("For this release, use the canary lane."))
                .message(Message::assistant("Understood: the release lane is canary."))
                .message(Message::user("Look up the harbor label with the tool."))
                .message(Message::Assistant {
                    id: None,
                    content: OneOrMany::one(AssistantContent::tool_call(
                        "call_REDACTED_1",
                        AlphaSignal::NAME,
                        json!({}),
                    )),
                })
                .message(Message::tool_result("call_REDACTED_1", ALPHA_SIGNAL_OUTPUT))
                .message(Message::assistant("The harbor label is crimson-harbor."))
                .tool(rig::tool::tool_definition(&AlphaSignal))
                .tool_choice(ToolChoice::None)
                .build();

            let response = model.completion(request).await?;
            let text = response
                .choice
                .iter()
                .filter_map(|content| match content {
                    AssistantContent::Text(text) => Some(text.text.as_str()),
                    _ => None,
                })
                .collect::<String>();

            assert_contains_all_case_insensitive(&text, &["teal", ALPHA_SIGNAL_OUTPUT, "canary"]);
            anyhow::ensure!(
                response.usage.input_tokens > 0 && response.usage.output_tokens > 0,
                "usage should be populated on long-history replay: {:?}",
                response.usage
            );
            anyhow::ensure!(
                response
                    .raw_response
                    .choices
                    .iter()
                    .all(|choice| choice.finish_reason.is_some()),
                "raw response should preserve finish reasons"
            );
            assert_nonempty_response(&response.raw_response.model);

            Ok(())
        },
    )
    .await
}

#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct NestedPlan {
    release: ReleaseInfo,
    checks: Vec<PlanCheck>,
}

#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct ReleaseInfo {
    lane: String,
    risk: String,
}

#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct PlanCheck {
    name: String,
    required: bool,
}

#[tokio::test]
async fn nested_structured_output_schema_roundtrip() -> Result<()> {
    with_openrouter_cassette_result(
        "agent_tool_sessions/nested_structured_output_schema_roundtrip",
        |client| async move {
            let agent = client
                .agent(STRUCTURED_MODEL)
                .preamble(
                    "Return only data that satisfies the requested schema. Use lane canary, risk low, \
                     and checks compile=true and replay=true.",
                )
                .additional_params(json!({
                    "provider": {
                        "require_parameters": true,
                        "order": ["Google AI Studio", "Google Vertex"]
                    }
                }))
                .build();

            let plan: NestedPlan = agent
                .prompt_typed("Create the OpenRouter cassette release validation plan.")
                .await?;

            anyhow::ensure!(plan.release.lane.eq_ignore_ascii_case("canary"));
            anyhow::ensure!(plan.release.risk.eq_ignore_ascii_case("low"));
            anyhow::ensure!(
                plan.checks
                    .iter()
                    .any(|check| check.name.eq_ignore_ascii_case("compile") && check.required),
                "structured output should include the compile check"
            );
            anyhow::ensure!(
                plan.checks
                    .iter()
                    .any(|check| check.name.eq_ignore_ascii_case("replay") && check.required),
                "structured output should include the replay check"
            );

            Ok(())
        },
    )
    .await
}
