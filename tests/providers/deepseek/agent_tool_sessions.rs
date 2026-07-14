//! Cassette-backed DeepSeek long-session and tool-contract regression tests.
//!
//! These scenarios stress Rig's DeepSeek OpenAI-compatible chat-completions path
//! with multi-turn tool loops, streamed tool-call deltas, complex JSON tool
//! arguments, explicit tool choice, reasoning metadata, structured JSON output,
//! and caller-owned long chat history.

use std::sync::{Arc, Mutex};

use anyhow::Result;
use rig::OneOrMany;
use rig::client::CompletionClient;
use rig::completion::{Chat, CompletionModel, Message};
use rig::message::{AssistantContent, ToolChoice, UserContent};
use rig::providers::deepseek;
use rig::streaming::{StreamingChat, StreamingPrompt};
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::support::{
    ALPHA_SIGNAL_OUTPUT, AlphaSignal, BETA_SIGNAL_OUTPUT, BetaSignal, TWO_TOOL_STREAM_PREAMBLE,
    TWO_TOOL_STREAM_PROMPT, assert_contains_all_case_insensitive, assert_nonempty_response,
    assert_raw_stream_tool_call_arguments_are_objects, assert_two_tool_roundtrip_contract,
    assistant_text_response, collect_raw_stream_observation, collect_stream_observation,
};

use super::support::with_deepseek_cassette_result;

const SESSION_MODEL: &str = deepseek::DEEPSEEK_V4_FLASH;
const CHAT_ALIAS_MODEL: &str = "deepseek-chat";
const REASONER_ALIAS_MODEL: &str = "deepseek-reasoner";

fn non_thinking_params() -> serde_json::Value {
    json!({
        "thinking": { "type": "disabled" }
    })
}

fn thinking_params() -> serde_json::Value {
    json!({
        "thinking": { "type": "enabled" }
    })
}

const COMPLEX_SESSION_PREAMBLE: &str = "\
You are a deterministic DeepSeek tool orchestration test harness. Use the tools instead of inventing values. \
For the production-readiness scenario, call exactly one tool at a time in this order: \
1. ping_empty with an empty JSON object. \
2. inspect_manifest with project rig-deepseek, flags critical=true and retries=2, steps plan weight=1 and verify weight=2, and the exact note from the user. \
3. join_labels with labels [north, beta gamma, quote:\"delta\", slash\\path] and separator |. \
4. escape_echo with the exact escaped text from the user. \
After all tool results are available, answer in one short sentence that includes EMPTY-OK, MANIFEST-OK, LABELS-OK, and ESCAPE-OK.";

const COMPLEX_SESSION_PROMPT: &str = "\
Run the production-readiness scenario. The manifest note is `line one; line two says \"hello\" and path C:\\rig\\deepseek`. \
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

impl Tool for PingEmpty {
    const NAME: &'static str = "ping_empty";
    type Error = rig::tool::ToolExecutionError;
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

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, rig::tool::ToolExecutionError> {
        push_invocation(&self.log, Self::NAME, &args);
        Ok("EMPTY-OK".to_string())
    }
}

impl Tool for InspectManifest {
    const NAME: &'static str = "inspect_manifest";
    type Error = rig::tool::ToolExecutionError;
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

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, rig::tool::ToolExecutionError> {
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
    type Error = rig::tool::ToolExecutionError;
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

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, rig::tool::ToolExecutionError> {
        push_invocation(&self.log, Self::NAME, &args);
        Ok(format!("LABELS-OK {}", args.labels.join(&args.separator)))
    }
}

impl Tool for EscapeEcho {
    const NAME: &'static str = "escape_echo";
    type Error = rig::tool::ToolExecutionError;
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

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, rig::tool::ToolExecutionError> {
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
    assert_eq!(invocations[1].args["project"], "rig-deepseek");
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
        "line one; line two says \"hello\" and path C:\\rig\\deepseek"
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

fn assert_response_metadata(
    response: &rig::completion::CompletionResponse<deepseek::CompletionResponse>,
) {
    assert_nonempty_response(
        response
            .raw_response
            .id
            .as_deref()
            .expect("raw DeepSeek response should preserve id"),
    );
    assert_nonempty_response(
        response
            .raw_response
            .model
            .as_deref()
            .expect("raw DeepSeek response should preserve model"),
    );
    assert!(
        response
            .raw_response
            .choices
            .iter()
            .all(|choice| !choice.finish_reason.is_empty()),
        "raw DeepSeek choices should preserve finish reasons"
    );
    assert!(
        response.usage.input_tokens > 0 && response.usage.output_tokens > 0,
        "usage should be populated: {:?}",
        response.usage
    );
}

#[tokio::test]
async fn sequential_complex_tool_calls_nonstreaming() -> Result<()> {
    with_deepseek_cassette_result(
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
                .additional_params(json_utils_merge(
                    non_thinking_params(),
                    json!({"parallel_tool_calls": false}),
                ))
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
    with_deepseek_cassette_result(
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
                .additional_params(json_utils_merge(
                    non_thinking_params(),
                    json!({"parallel_tool_calls": false}),
                ))
                .build();

            let mut stream = agent
                .stream_chat(COMPLEX_SESSION_PROMPT, Vec::<Message>::new())
                .max_turns(10)
                .await;
            let observation = collect_stream_observation(&mut stream).await;

            anyhow::ensure!(
                observation.errors.is_empty(),
                "stream should not emit errors: {:?}",
                observation.errors
            );
            anyhow::ensure!(
                observation.tool_calls
                    == vec![
                        PingEmpty::NAME.to_string(),
                        InspectManifest::NAME.to_string(),
                        JoinLabels::NAME.to_string(),
                        EscapeEcho::NAME.to_string(),
                    ],
                "stream should expose ordered tool calls, saw {:?}",
                observation.tool_calls
            );
            anyhow::ensure!(
                observation.tool_results == 4,
                "expected 4 streamed tool results, saw {}",
                observation.tool_results
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
    with_deepseek_cassette_result(
        "agent_tool_sessions/parallel_tool_calls_single_turn_nonstreaming",
        |client| async move {
            let agent = client
                .agent(SESSION_MODEL)
                .preamble(TWO_TOOL_STREAM_PREAMBLE)
                .tool(AlphaSignal)
                .tool(BetaSignal)
                .additional_params(json_utils_merge(
                    non_thinking_params(),
                    json!({"parallel_tool_calls": true}),
                ))
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
                "expected both zero-argument tools, saw {:?}",
                call_names
            );
            anyhow::ensure!(
                calls[0].message_index == calls[1].message_index,
                "parallel tool calls should be recorded on one assistant message"
            );
            anyhow::ensure!(
                history_tool_results(&history).len() == 2,
                "expected two tool results"
            );

            Ok(())
        },
    )
    .await
}

#[tokio::test]
async fn parallel_tool_calls_single_turn_streaming() -> Result<()> {
    with_deepseek_cassette_result(
        "agent_tool_sessions/parallel_tool_calls_single_turn_streaming",
        |client| async move {
            let agent = client
                .agent(SESSION_MODEL)
                .preamble(TWO_TOOL_STREAM_PREAMBLE)
                .tool(AlphaSignal)
                .tool(BetaSignal)
                .additional_params(json_utils_merge(
                    non_thinking_params(),
                    json!({"parallel_tool_calls": true}),
                ))
                .build();

            let mut stream = agent
                .stream_prompt(TWO_TOOL_STREAM_PROMPT)
                .max_turns(5)
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
    with_deepseek_cassette_result(
        "agent_tool_sessions/raw_stream_complex_tool_call_deltas_have_object_arguments",
        |client| async move {
            let log = Arc::new(Mutex::new(Vec::new()));
            let model = client.completion_model(SESSION_MODEL);
            let tool = InspectManifest { log };
            let request = model
                .completion_request(
                    "Call inspect_manifest exactly once for project rig-deepseek with critical=true, retries=2, \
                     steps [{name: plan, weight: 1}, {name: verify, weight: 2}], and note `streamed nested JSON`. \
                     Do not write normal text before the tool call.",
                )
                .preamble("Use the requested tool call and no prose before it.".to_string())
                .tool(rig::tool::tool_definition(&tool))
                .tool_choice(ToolChoice::Required)
                .additional_params(non_thinking_params())
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
            anyhow::ensure!(tool_call.function.arguments["project"] == "rig-deepseek");
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
    with_deepseek_cassette_result(
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
                .additional_params(non_thinking_params())
                .build();

            let response = model.completion(request).await?;
            let text = assistant_text_response(&response.choice)
                .ok_or_else(|| anyhow::anyhow!("response should include assistant text"))?;

            assert_contains_all_case_insensitive(&text, &["teal", ALPHA_SIGNAL_OUTPUT, "canary"]);
            assert_response_metadata(&response);

            Ok(())
        },
    )
    .await
}

#[tokio::test]
async fn tool_choice_required_specific_and_none() -> Result<()> {
    with_deepseek_cassette_result(
        "agent_tool_sessions/tool_choice_required_specific_and_none",
        |client| async move {
            let model = client.completion_model(SESSION_MODEL);

            let required = model
                .completion(
                    model
                        .completion_request(
                            "Call lookup_harbor_label exactly once with an empty object and do not answer in prose.",
                        )
                        .tool(rig::tool::tool_definition(&AlphaSignal))
                        .tool_choice(ToolChoice::Required)
                        .additional_params(non_thinking_params())
                        .build(),
                )
                .await?;
            anyhow::ensure!(
                required.choice.iter().any(|content| matches!(
                    content,
                    AssistantContent::ToolCall(tool_call)
                        if tool_call.function.name == AlphaSignal::NAME
                            && tool_call.function.arguments == json!({})
                )),
                "required tool choice should force lookup_harbor_label"
            );

            let specific = model
                .completion(
                    model
                        .completion_request(
                            "Call the orchard-label tool exactly once with an empty object and do not call any other tool.",
                        )
                        .tool(rig::tool::tool_definition(&AlphaSignal))
                        .tool(rig::tool::tool_definition(&BetaSignal))
                        .tool_choice(ToolChoice::Specific {
                            function_names: vec![BetaSignal::NAME.to_string()],
                        })
                        .additional_params(non_thinking_params())
                        .build(),
                )
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
                "specific tool choice should force only lookup_orchard_label, saw {:?}",
                specific_calls
            );

            let none = model
                .completion(
                    model
                        .completion_request(
                            "Do not call tools. Reply with exactly this phrase: no-tool-answer",
                        )
                        .tool(rig::tool::tool_definition(&AlphaSignal))
                        .tool_choice(ToolChoice::None)
                        .additional_params(non_thinking_params())
                        .build(),
                )
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
            let model = client.completion_model(SESSION_MODEL);
            let request = model
                .completion_request(
                    "Use concise reasoning to solve: if three probes each verify two cassettes, how many cassette verifications occur? Answer with the number.",
                )
                .preamble("You are a concise reliability engineer.".to_string())
                .additional_params(thinking_params())
                .build();

            let response = model.completion(request).await?;

            anyhow::ensure!(
                response
                    .choice
                    .iter()
                    .any(|content| matches!(content, AssistantContent::Reasoning(_))),
                "DeepSeek reasoning response should preserve reasoning_content separately"
            );
            anyhow::ensure!(
                response.usage.reasoning_tokens > 0,
                "core usage should preserve DeepSeek reasoning tokens: {:?}",
                response.usage
            );
            let raw_reasoning_tokens = response
                .raw_response
                .usage
                .completion_tokens_details
                .as_ref()
                .and_then(|details| details.reasoning_tokens)
                .unwrap_or_default() as u64;
            anyhow::ensure!(
                response.usage.reasoning_tokens == raw_reasoning_tokens && raw_reasoning_tokens > 0,
                "usage reasoning tokens should match raw provider details"
            );
            assert_response_metadata(&response);

            let stream_request = model
                .completion_request("Briefly solve 2 + 2, then answer with the number.")
                .additional_params(thinking_params())
                .build();
            let observation = collect_raw_stream_observation(model.stream(stream_request).await?).await;
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
async fn chat_alias_vs_reasoner_alias_behavior() -> Result<()> {
    with_deepseek_cassette_result(
        "agent_tool_sessions/chat_alias_vs_reasoner_alias_behavior",
        |client| async move {
            let chat_model = client.completion_model(CHAT_ALIAS_MODEL);
            let chat = chat_model
                .completion(
                    chat_model
                        .completion_request("Reply with exactly: chat-mode-ok")
                        .build(),
                )
                .await?;
            let chat_text = assistant_text_response(&chat.choice)
                .ok_or_else(|| anyhow::anyhow!("deepseek-chat should return text"))?;
            assert_contains_all_case_insensitive(&chat_text, &["chat-mode-ok"]);
            anyhow::ensure!(
                chat.choice
                    .iter()
                    .all(|content| !matches!(content, AssistantContent::Reasoning(_))),
                "deepseek-chat alias should not emit reasoning content"
            );

            let reasoner_model = client.completion_model(REASONER_ALIAS_MODEL);
            let reasoner = reasoner_model
                .completion(
                    reasoner_model
                        .completion_request("Reply with exactly: reasoner-mode-ok")
                        .build(),
                )
                .await?;
            let reasoner_text = assistant_text_response(&reasoner.choice)
                .ok_or_else(|| anyhow::anyhow!("deepseek-reasoner should return text"))?;
            assert_contains_all_case_insensitive(&reasoner_text, &["reasoner-mode-ok"]);
            anyhow::ensure!(
                reasoner
                    .choice
                    .iter()
                    .any(|content| matches!(content, AssistantContent::Reasoning(_))),
                "deepseek-reasoner alias should emit reasoning content"
            );
            anyhow::ensure!(
                reasoner.usage.reasoning_tokens > 0,
                "deepseek-reasoner usage should surface reasoning tokens"
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
            let model = client.completion_model(SESSION_MODEL);
            let request = model
                .completion_request(
                    "Return a JSON object with release lane canary, risk low, and checks compile=true and replay=true.",
                )
                .preamble("Return only valid JSON. No markdown.".to_string())
                .additional_params(json_utils_merge(non_thinking_params(), json!({
                    "response_format": { "type": "json_object" }
                })))
                .build();

            let response = model.completion(request).await?;
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

fn json_utils_merge(left: serde_json::Value, right: serde_json::Value) -> serde_json::Value {
    let mut left = left;
    let Some(left_obj) = left.as_object_mut() else {
        return right;
    };
    let Some(right_obj) = right.as_object() else {
        return left;
    };

    for (key, value) in right_obj {
        left_obj.insert(key.clone(), value.clone());
    }

    left
}
