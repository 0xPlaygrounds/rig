//! Cassette-backed Groq long-session and OpenAI-compatible chat regression tests.
//!
//! These scenarios exercise Rig's Groq chat-completions path with multi-turn
//! tool loops, low-latency SSE streaming, complex JSON tool arguments, caller-
//! owned history, JSON response formats, explicit tool choice, usage accounting,
//! and provider metadata preservation.

use std::sync::{Arc, Mutex};

use anyhow::Result;
use futures::StreamExt;
use rig::OneOrMany;
use rig::client::{AgentClientExt, CompletionClient};
use rig::completion::{Chat, CompletionModel, GetTokenUsage, Message};
use rig::message::{AssistantContent, ToolChoice};
use rig::providers::openai;
use rig::streaming::{StreamedAssistantContent, StreamingChat, StreamingPrompt};
use rig::tool::Tool;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::support::{
    ALPHA_SIGNAL_OUTPUT, AlphaSignal, BETA_SIGNAL_OUTPUT, BetaSignal, TWO_TOOL_STREAM_PREAMBLE,
    TWO_TOOL_STREAM_PROMPT, assert_contains_all_case_insensitive, assert_nonempty_response,
    assert_raw_stream_tool_call_arguments_are_objects, assert_two_tool_roundtrip_contract,
    assistant_text_response, collect_raw_stream_observation, collect_stream_observation,
};

use super::support::with_groq_cassette_result;

const SESSION_MODEL: &str = "llama-3.3-70b-versatile";
const JSON_OBJECT_MODEL: &str = "llama-3.3-70b-versatile";
const JSON_SCHEMA_MODEL: &str = "openai/gpt-oss-20b";

const COMPLEX_SESSION_PREAMBLE: &str = "\
You are a deterministic Groq tool orchestration test harness. Use the tools instead of inventing values. \
For the production-readiness scenario, call exactly one tool at a time in this order: \
1. ping_empty with an empty JSON object. \
2. inspect_manifest with project rig-groq, flags critical=true and retries=2, steps plan weight=1 and verify weight=2, and the exact note from the user. \
3. join_labels with labels [north, beta gamma, quote:\"delta\", slash\\path] and separator |. \
4. optional_nullable_probe with required name sentinel, optional note omitted if possible, and nullable_code null. \
5. escape_echo with the exact escaped text from the user. \
After all tool results are available, answer in one short sentence that includes EMPTY-OK, MANIFEST-OK, LABELS-OK, OPTIONAL-OK, and ESCAPE-OK.";

const COMPLEX_SESSION_PROMPT: &str = "\
Run the production-readiness scenario. The manifest note is `line one; line two says \"hello\" and path C:/rig/groq`. \
The escaped text is `Line 1\nLine \"2\" with colon: ok and unicode snowman ☃`.";

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
struct OptionalNullableProbe {
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
struct OptionalNullableArgs {
    name: String,
    #[serde(default)]
    note: Option<String>,
    #[serde(default)]
    nullable_code: Option<String>,
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

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
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

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
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
                "labels": { "type": "array", "items": { "type": "string" } },
                "separator": { "type": "string" }
            },
            "required": ["labels", "separator"]
        })
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        push_invocation(&self.log, Self::NAME, &args);
        Ok(format!("LABELS-OK {}", args.labels.join(&args.separator)))
    }
}

impl Tool for OptionalNullableProbe {
    const NAME: &'static str = "optional_nullable_probe";
    type Error = SessionToolError;
    type Args = OptionalNullableArgs;
    type Output = String;

    fn description(&self) -> String {
        "Validate optional and nullable argument serialization.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "note": { "type": ["string", "null"] },
                "nullable_code": { "type": ["string", "null"] }
            },
            "required": ["name", "nullable_code"]
        })
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        push_invocation(&self.log, Self::NAME, &args);
        Ok(format!(
            "OPTIONAL-OK name={} note={} nullable={}",
            args.name,
            args.note.unwrap_or_else(|| "missing".to_string()),
            args.nullable_code.unwrap_or_else(|| "null".to_string())
        ))
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
            "properties": { "text": { "type": "string" } },
            "required": ["text"]
        })
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        push_invocation(&self.log, Self::NAME, &args);
        Ok(format!("ESCAPE-OK {}", args.text))
    }
}

fn complex_tools(
    log: &InvocationLog,
) -> (
    PingEmpty,
    InspectManifest,
    JoinLabels,
    OptionalNullableProbe,
    EscapeEcho,
) {
    (
        PingEmpty { log: log.clone() },
        InspectManifest { log: log.clone() },
        JoinLabels { log: log.clone() },
        OptionalNullableProbe { log: log.clone() },
        EscapeEcho { log: log.clone() },
    )
}

fn assert_complex_invocations(log: &InvocationLog) {
    let invocations = log
        .lock()
        .expect("tool invocation log lock should not be poisoned")
        .clone();
    let names = invocations
        .iter()
        .map(|invocation| invocation.name)
        .collect::<Vec<_>>();
    assert_eq!(
        names,
        vec![
            PingEmpty::NAME,
            InspectManifest::NAME,
            JoinLabels::NAME,
            OptionalNullableProbe::NAME,
            EscapeEcho::NAME,
        ]
    );
    assert_eq!(invocations[0].args, json!({}));
    assert_eq!(invocations[1].args["project"], "rig-groq");
    assert_eq!(invocations[1].args["flags"]["critical"], true);
    assert_eq!(invocations[1].args["flags"]["retries"], 2);
    assert_eq!(
        invocations[1].args["steps"].as_array().map(Vec::len),
        Some(2)
    );
    assert_eq!(
        invocations[2].args["labels"],
        json!(["north", "beta gamma", "quote:\"delta\"", "slash\\path"])
    );
    assert_eq!(invocations[2].args["separator"], "|");
    assert_eq!(invocations[3].args["name"], "sentinel");
    assert!(
        invocations[3].args.get("note").is_none()
            || invocations[3].args["note"].is_null()
            || invocations[3].args["note"].as_str().is_some(),
        "optional note should be omitted, null, or a string when supplied"
    );
    assert!(invocations[3].args["nullable_code"].is_null());
    assert_eq!(
        invocations[4].args["text"],
        "Line 1\nLine \"2\" with colon: ok and unicode snowman ☃"
    );
}

struct HistoryToolCall {
    message_index: usize,
    name: String,
}

struct HistoryToolResult {
    message_index: usize,
}

fn history_tool_calls(history: &[Message]) -> Vec<HistoryToolCall> {
    history
        .iter()
        .enumerate()
        .flat_map(|(message_index, message)| match message {
            Message::Assistant { content, .. } => content
                .iter()
                .filter_map(move |content| match content {
                    AssistantContent::ToolCall(tool_call) => Some(HistoryToolCall {
                        message_index,
                        name: tool_call.function.name.clone(),
                    }),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        })
        .collect()
}

fn history_tool_results(history: &[Message]) -> Vec<HistoryToolResult> {
    history
        .iter()
        .enumerate()
        .flat_map(|(message_index, message)| match message {
            Message::User { content } => content
                .iter()
                .filter_map(move |content| match content {
                    rig::message::UserContent::ToolResult(_) => {
                        Some(HistoryToolResult { message_index })
                    }
                    _ => None,
                })
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        })
        .collect()
}

fn assert_history_records_sequential_tool_roundtrips(history: &[Message], expected_tools: &[&str]) {
    let calls = history_tool_calls(history);
    let results = history_tool_results(history);
    assert_eq!(
        calls
            .iter()
            .map(|call| call.name.as_str())
            .collect::<Vec<_>>(),
        expected_tools,
        "caller-owned chat history should preserve tool call order"
    );
    assert_eq!(results.len(), expected_tools.len());

    for (index, call) in calls.iter().enumerate() {
        let result = &results[index];
        assert!(call.message_index < result.message_index);
        if let Some(next_call) = calls.get(index + 1) {
            assert!(result.message_index < next_call.message_index);
        }
    }
}

fn assert_response_metadata(
    response: &rig::completion::CompletionResponse<openai::CompletionResponse>,
) {
    assert_nonempty_response(&response.raw_response.id);
    assert_nonempty_response(&response.raw_response.model);
    if let Some(system_fingerprint) = &response.raw_response.system_fingerprint {
        assert_nonempty_response(system_fingerprint);
    }
    assert!(
        response
            .raw_response
            .choices
            .iter()
            .all(|choice| !choice.finish_reason.is_empty()),
        "raw Groq choices should preserve finish reasons"
    );
    let raw_usage = response
        .raw_response
        .usage
        .as_ref()
        .expect("raw response should preserve usage");
    assert!(
        response.usage.input_tokens > 0,
        "usage should include input tokens"
    );
    if let Some(completion_tokens) = raw_usage.completion_tokens {
        assert_eq!(response.usage.output_tokens, completion_tokens as u64);
    }
    assert_eq!(response.usage.total_tokens, raw_usage.total_tokens as u64);
    if let Some(queue_time) = raw_usage.queue_time {
        assert!(queue_time >= 0.0);
    }
    if let Some(total_time) = raw_usage.total_time {
        assert!(total_time >= 0.0);
    }
}

#[tokio::test]
async fn sequential_complex_tool_calls_nonstreaming() -> Result<()> {
    with_groq_cassette_result(
        "agent_tool_sessions/sequential_complex_tool_calls_nonstreaming",
        |client| async move {
            let log = Arc::new(Mutex::new(Vec::new()));
            let (ping, manifest, labels, optional, echo) = complex_tools(&log);
            let agent = client
                .agent(SESSION_MODEL)
                .preamble(COMPLEX_SESSION_PREAMBLE)
                .tool(ping)
                .tool(manifest)
                .tool(labels)
                .tool(optional)
                .tool(echo)
                .additional_params(json!({"parallel_tool_calls": false}))
                .default_max_turns(10)
                .build();
            let mut history = Vec::<Message>::new();

            let response = agent.chat(COMPLEX_SESSION_PROMPT, &mut history).await?;

            assert_contains_all_case_insensitive(
                &response,
                &[
                    "EMPTY-OK",
                    "MANIFEST-OK",
                    "LABELS-OK",
                    "OPTIONAL-OK",
                    "ESCAPE-OK",
                ],
            );
            assert_complex_invocations(&log);
            assert_history_records_sequential_tool_roundtrips(
                &history,
                &[
                    PingEmpty::NAME,
                    InspectManifest::NAME,
                    JoinLabels::NAME,
                    OptionalNullableProbe::NAME,
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
    with_groq_cassette_result(
        "agent_tool_sessions/sequential_complex_tool_calls_streaming",
        |client| async move {
            let log = Arc::new(Mutex::new(Vec::new()));
            let (ping, manifest, labels, optional, echo) = complex_tools(&log);
            let agent = client
                .agent(SESSION_MODEL)
                .preamble(COMPLEX_SESSION_PREAMBLE)
                .tool(ping)
                .tool(manifest)
                .tool(labels)
                .tool(optional)
                .tool(echo)
                .additional_params(json!({"parallel_tool_calls": false}))
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
                        OptionalNullableProbe::NAME.to_string(),
                        EscapeEcho::NAME.to_string(),
                    ],
                "stream should expose ordered tool calls, saw {:?}",
                observation.tool_calls
            );
            anyhow::ensure!(
                observation.tool_results == 5,
                "expected 5 streamed tool results"
            );
            let response = observation
                .final_response_text
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("stream should produce final response text"))?;
            assert_contains_all_case_insensitive(
                response,
                &[
                    "EMPTY-OK",
                    "MANIFEST-OK",
                    "LABELS-OK",
                    "OPTIONAL-OK",
                    "ESCAPE-OK",
                ],
            );
            assert_complex_invocations(&log);

            Ok(())
        },
    )
    .await
}

#[tokio::test]
async fn parallel_tool_calls_single_turn_nonstreaming() -> Result<()> {
    with_groq_cassette_result(
        "agent_tool_sessions/parallel_tool_calls_single_turn_nonstreaming",
        |client| async move {
            let agent = client
                .agent(SESSION_MODEL)
                .preamble(TWO_TOOL_STREAM_PREAMBLE)
                .tool(AlphaSignal)
                .tool(BetaSignal)
                .additional_params(json!({"parallel_tool_calls": true}))
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
                .map(|call| call.name.as_str())
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
    with_groq_cassette_result(
        "agent_tool_sessions/parallel_tool_calls_single_turn_streaming",
        |client| async move {
            let agent = client
                .agent(SESSION_MODEL)
                .preamble(TWO_TOOL_STREAM_PREAMBLE)
                .tool(AlphaSignal)
                .tool(BetaSignal)
                .additional_params(json!({"parallel_tool_calls": true}))
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
    with_groq_cassette_result(
        "agent_tool_sessions/raw_stream_complex_tool_call_deltas_have_object_arguments",
        |client| async move {
            let log = Arc::new(Mutex::new(Vec::new()));
            let model = client.completion_model(SESSION_MODEL);
            let tool = InspectManifest { log };
            let request = model
                .completion_request(
                    "Call inspect_manifest exactly once for project rig-groq with critical=true, retries=2, \
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
async fn long_history_replay_with_tool_result_continuation() -> Result<()> {
    with_groq_cassette_result(
        "agent_tool_sessions/long_history_replay_with_tool_result_continuation",
        |client| async move {
            let model = client.completion_model(SESSION_MODEL);
            let tool_call_id = "call_REDACTED_1";
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
                        tool_call_id,
                        AlphaSignal::NAME,
                        json!({}),
                    )),
                })
                .message(Message::tool_result_with_call_id(
                    tool_call_id,
                    Some(tool_call_id.to_string()),
                    ALPHA_SIGNAL_OUTPUT,
                ))
                .message(Message::assistant("The harbor label is crimson-harbor."))
                .tool(rig::tool::tool_definition(&AlphaSignal))
                .tool_choice(ToolChoice::None)
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
async fn tool_choice_auto_required_specific_and_none() -> Result<()> {
    with_groq_cassette_result(
        "agent_tool_sessions/tool_choice_auto_required_specific_and_none",
        |client| async move {
            let model = client.completion_model(SESSION_MODEL);

            let auto = model
                .completion(
                    model
                        .completion_request("Call lookup_harbor_label exactly once with an empty object.")
                        .tool(rig::tool::tool_definition(&AlphaSignal))
                        .tool_choice(ToolChoice::Auto)
                        .build(),
                )
                .await?;
            anyhow::ensure!(
                auto.choice.iter().any(|content| matches!(
                    content,
                    AssistantContent::ToolCall(tool_call)
                        if tool_call.function.name == AlphaSignal::NAME
                            && tool_call.function.arguments == json!({})
                )),
                "auto tool choice should allow lookup_harbor_label"
            );

            let required = model
                .completion(
                    model
                        .completion_request("Call lookup_harbor_label exactly once with an empty object and do not answer in prose.")
                        .tool(rig::tool::tool_definition(&AlphaSignal))
                        .tool_choice(ToolChoice::Required)
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
                        .completion_request("Call the orchard-label tool exactly once with an empty object and do not call any other tool.")
                        .tool(rig::tool::tool_definition(&AlphaSignal))
                        .tool(rig::tool::tool_definition(&BetaSignal))
                        .tool_choice(ToolChoice::Specific {
                            function_names: vec![BetaSignal::NAME.to_string()],
                        })
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
                        .completion_request("Do not call tools. Reply with exactly this phrase: no-tool-answer")
                        .tool(rig::tool::tool_definition(&AlphaSignal))
                        .tool_choice(ToolChoice::None)
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
async fn json_object_response_format_roundtrip() -> Result<()> {
    with_groq_cassette_result(
        "agent_tool_sessions/json_object_response_format_roundtrip",
        |client| async move {
            let model = client.completion_model(JSON_OBJECT_MODEL);
            let request = model
                .completion_request(
                    "Return a JSON object with release lane canary, risk low, and checks compile=true and replay=true.",
                )
                .preamble("Return only valid JSON. No markdown.".to_string())
                .additional_params(json!({"response_format": { "type": "json_object" }}))
                .max_tokens(128)
                .build();

            let response = model.completion(request).await?;
            let text = assistant_text_response(&response.choice)
                .ok_or_else(|| anyhow::anyhow!("JSON response should contain text"))?;
            let plan: serde_json::Value = serde_json::from_str(&text)?;

            let serialized = plan.to_string();
            assert_contains_all_case_insensitive(
                &serialized,
                &["canary", "low", "compile", "replay"],
            );
            assert_response_metadata(&response);

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
            let model = client.completion_model(JSON_SCHEMA_MODEL);
            let request = model
                .completion_request(
                    "Return lane=canary, risk=low, checks.compile=true, and checks.replay=true.",
                )
                .preamble("Return only the requested structured object.".to_string())
                .output_schema(schemars::schema_for!(StructuredReleasePlan))
                .max_tokens(128)
                .build();

            let response = model.completion(request).await?;
            let text = assistant_text_response(&response.choice)
                .ok_or_else(|| anyhow::anyhow!("structured response should contain text"))?;
            let plan: StructuredReleasePlan = serde_json::from_str(&text)?;

            anyhow::ensure!(plan.lane.eq_ignore_ascii_case("canary"));
            anyhow::ensure!(plan.risk.eq_ignore_ascii_case("low"));
            anyhow::ensure!(plan.checks.compile);
            anyhow::ensure!(plan.checks.replay);
            assert_response_metadata(&response);

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
            let model = client.completion_model(SESSION_MODEL);
            let mut stream = model
                .stream(
                    model
                        .completion_request(
                            "Reply with exactly this comma-separated sequence and no extra words: alpha,beta,gamma,delta,epsilon,zeta,eta,theta",
                        )
                        .preamble("Stream the requested short sequence exactly.".to_string())
                        .max_tokens(64)
                        .build(),
                )
                .await?;

            let mut text_chunks = 0usize;
            let mut final_usage = None;
            while let Some(item) = stream.next().await {
                match item? {
                    StreamedAssistantContent::Text(text) => {
                        if !text.text.is_empty() {
                            text_chunks += 1;
                        }
                    }
                    StreamedAssistantContent::Final(response) => {
                        final_usage = Some(response.token_usage());
                    }
                    _ => {}
                }
            }

            anyhow::ensure!(text_chunks > 0, "stream should emit text deltas");
            let usage = final_usage.ok_or_else(|| anyhow::anyhow!("stream should emit final usage"))?;
            anyhow::ensure!(usage.input_tokens > 0, "stream usage should include input tokens");
            anyhow::ensure!(usage.output_tokens > 0, "stream usage should include output tokens");
            anyhow::ensure!(
                usage.total_tokens >= usage.input_tokens + usage.output_tokens,
                "stream usage totals should cover input + output tokens: {:?}",
                usage
            );

            Ok(())
        },
    )
    .await
}
