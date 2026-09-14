//! Original neutral conformance tools/assertions; execution is native ECS.
// Keep the original conformance error type and unchanged predicate signatures.
#![allow(clippy::result_large_err)]
use super::runtime::{NativeResponse, configured, execute};
use rig::{
    completion::CompletionModel,
    message::{AssistantContent, Message, UserContent},
    tool::{Tool, ToolContext},
};
use rig_agent::test_utils::{ScenarioError, ScenarioReport, validate_protocol_hygiene};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Instant,
};
fn contract(scenario: &'static str, details: impl Into<String>) -> ScenarioError {
    ScenarioError::Contract {
        scenario,
        details: details.into(),
    }
}
/// Error used by the deterministic conformance tools.
#[derive(Debug, thiserror::Error)]
#[error("model-conformance tool failed")]
pub struct ConformanceToolError;
const FORCE_TOOLS_PREAMBLE: &str = "You are a calculator assistant. You MUST use the provided tools for every arithmetic operation instead of computing results yourself. Once you have all the tool results you need, reply with the final numeric answer in plain text.";
const PARALLEL_PROMPT: &str = "Compute 3 + 4 and 10 - 2. You MUST call the add tool and the subtract tool together in your first response, as two parallel function calls, then report both results.";
const PING_OUTPUT: &str = "pong-crimson-7423";
const MOTTO_OUTPUT: &str = "steady hands\ncalm waters";
fn tool_result_values(message: &Message) -> Vec<serde_json::Value> {
    let Message::User { content } = message else {
        return Vec::new();
    };
    content
        .iter()
        .filter_map(|item| match item {
            UserContent::ToolResult(result) => Some(result),
            _ => None,
        })
        .flat_map(|result| result.content.iter())
        .filter_map(|content| match content {
            rig_core::message::ToolResultContent::Text(text) => {
                Some(serde_json::Value::String(text.text.clone()))
            }
            rig_core::message::ToolResultContent::Json { value } => Some(value.clone()),
            rig_core::message::ToolResultContent::Image(_) => None,
        })
        .collect()
}
fn validate_tool_correlation(
    scenario: &'static str,
    messages: &[Message],
) -> Result<(), ScenarioError> {
    let mut calls = Vec::new();
    let mut results = Vec::new();
    let mut turn = 0usize;
    for message in messages {
        match message {
            Message::Assistant { content, .. } => {
                turn += 1;
                calls.extend(content.iter().filter_map(|item| {
                    match item {
                        AssistantContent::ToolCall(call) => Some((
                            turn,
                            &call.id,
                            call.provider
                                .as_ref()
                                .map(|provider| provider.call_id.as_str()),
                        )),
                        _ => None,
                    }
                }));
            }
            Message::User { content } => {
                results.extend(content.iter().filter_map(|item| {
                    match item {
                        UserContent::ToolResult(result) => Some((
                            turn,
                            &result.call,
                            result
                                .provider
                                .as_ref()
                                .map(|provider| provider.call_id.as_str()),
                        )),
                        _ => None,
                    }
                }));
            }
            Message::System { .. } => {}
        }
    }
    if calls.is_empty() {
        return Err(contract(
            scenario,
            format!("history has no assistant tool calls: {messages:?}"),
        ));
    }
    for (turn, id, call_id) in &calls {
        let matches = results
            .iter()
            .filter(|(result_turn, result_id, result_call_id)| {
                result_turn == turn && result_id == id && call_id == result_call_id
            })
            .count();
        if matches != 1 {
            return Err(contract(
                scenario,
                format!(
                    "tool call id={id:?} call_id={call_id:?} has {matches} correlated results; calls={calls:?}, results={results:?}"
                ),
            ));
        }
    }
    if results.len() != calls.len() {
        return Err(contract(
            scenario,
            format!(
                "history contains dangling calls or results: calls={calls:?}, results={results:?}"
            ),
        ));
    }
    Ok(())
}
#[derive(Debug, Deserialize, JsonSchema)]
struct OperationArgs {
    x: i64,
    y: i64,
}
#[derive(Clone)]
struct CountingAdd(Arc<AtomicUsize>);
impl Tool for CountingAdd {
    const NAME: &'static str = "add";
    type Error = ConformanceToolError;
    type Args = OperationArgs;
    type Output = i64;
    fn description(&self) -> String {
        "Add x and y together".to_string()
    }
    fn parameters(&self) -> serde_json::Value {
        serde_json::json!(
            { "type" : "object", "properties" : { "x" : { "type" : "number",
            "description" : "The first operand" }, "y" : { "type" : "number",
            "description" : "The second operand" } }, "required" : ["x", "y"] }
        )
    }
    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(args.x + args.y)
    }
}
#[derive(Clone)]
struct CountingSubtract(Arc<AtomicUsize>);
impl Tool for CountingSubtract {
    const NAME: &'static str = "subtract";
    type Error = ConformanceToolError;
    type Args = OperationArgs;
    type Output = i64;
    fn description(&self) -> String {
        "Subtract y from x (i.e. x - y)".to_string()
    }
    fn parameters(&self) -> serde_json::Value {
        serde_json::json!(
            { "type" : "object", "properties" : { "x" : { "type" : "number",
            "description" : "The first operand" }, "y" : { "type" : "number",
            "description" : "The second operand" } }, "required" : ["x", "y"] }
        )
    }
    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(args.x - args.y)
    }
}
#[derive(Debug, Deserialize, JsonSchema)]
struct EmptyArgs {}
#[derive(Clone)]
struct PingTool(Arc<AtomicUsize>);
impl Tool for PingTool {
    const NAME: &'static str = "ping";
    type Error = ConformanceToolError;
    type Args = EmptyArgs;
    type Output = String;
    fn description(&self) -> String {
        "Return the current ping marker. Takes no arguments.".to_string()
    }
    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({ "type" : "object", "properties" : {}, "required" : [] })
    }
    async fn call(
        &self,
        _context: &mut ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(PING_OUTPUT.to_string())
    }
}
#[derive(Clone)]
struct MottoTool(Arc<AtomicUsize>);
impl Tool for MottoTool {
    const NAME: &'static str = "fetch_motto";
    type Error = ConformanceToolError;
    type Args = EmptyArgs;
    type Output = String;
    fn description(&self) -> String {
        "Fetch the two-line workshop motto.".to_string()
    }
    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({ "type" : "object", "properties" : {}, "required" : [] })
    }
    async fn call(
        &self,
        _context: &mut ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(MOTTO_OUTPUT.to_string())
    }
}
#[derive(Debug, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
struct ConfigOutput {
    service: String,
    max_retries: u64,
}
#[derive(Clone)]
struct ConfigTool(Arc<AtomicUsize>);
impl Tool for ConfigTool {
    const NAME: &'static str = "fetch_config";
    type Error = ConformanceToolError;
    type Args = EmptyArgs;
    type Output = ConfigOutput;
    fn description(&self) -> String {
        "Fetch the service configuration object.".to_string()
    }
    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({ "type" : "object", "properties" : {}, "required" : [] })
    }
    async fn call(
        &self,
        _context: &mut ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(ConfigOutput {
            service: "cassette-lab".to_string(),
            max_retries: 3,
        })
    }
}
fn report_from_response(
    name: &'static str,
    started: Instant,
    tool_calls: usize,
    response: NativeResponse,
) -> Result<ScenarioReport, ScenarioError> {
    if let Some(messages) = response.messages.as_deref() {
        validate_protocol_hygiene(
            name,
            &response.output,
            messages,
            &[
                "<tool_call>",
                "</tool_call>",
                "<tool_response>",
                "</tool_response>",
                "<|im_start|>",
                "<|im_end|>",
                "<think>",
                "</think>",
            ],
        )?;
    }
    Ok(ScenarioReport {
        name,
        tool_calls,
        prompt_tokens: response.usage.input_tokens,
        generated_tokens: response.usage.output_tokens,
        history_messages: response.messages.as_ref().map_or(0, Vec::len),
        duration: started.elapsed(),
        response: response.output,
    })
}
/// Require the extended run's accumulated message history and validate
/// canonical tool call/result correlation over it.
fn correlated_messages<'a>(
    scenario: &'static str,
    response: &'a NativeResponse,
) -> Result<&'a [Message], ScenarioError> {
    let messages = response
        .messages
        .as_deref()
        .ok_or_else(|| contract(scenario, "extended run omitted accumulated message history"))?;
    validate_tool_correlation(scenario, messages)?;
    Ok(messages)
}
/// [`correlated_messages`], flattened into every tool-result value in history.
fn correlated_result_values(
    scenario: &'static str,
    response: &NativeResponse,
) -> Result<Vec<serde_json::Value>, ScenarioError> {
    Ok(correlated_messages(scenario, response)?
        .iter()
        .flat_map(tool_result_values)
        .collect())
}
fn value_matches_integer(value: &serde_json::Value, expected: i64) -> bool {
    value.as_i64() == Some(expected)
        || value
            .as_str()
            .and_then(|text| text.trim().parse::<i64>().ok())
            == Some(expected)
}
pub(super) async fn parallel_tools(
    model: impl CompletionModel + 'static,
    tool_concurrency: Option<usize>,
) -> Result<ScenarioReport, ScenarioError> {
    let add_calls = Arc::new(AtomicUsize::new(0));
    let subtract_calls = Arc::new(AtomicUsize::new(0));
    let started = Instant::now();
    let mut ecs = configured(model, FORCE_TOOLS_PREAMBLE, Some(3));
    ecs.tool(CountingAdd(add_calls.clone()));
    ecs.tool(CountingSubtract(subtract_calls.clone()));
    let response = execute(&mut ecs, PARALLEL_PROMPT, false, Some(3), tool_concurrency).await;
    let scenario = if tool_concurrency == Some(1) {
        "parallel_tools_serial_execution"
    } else {
        "parallel_tools"
    };
    let messages = correlated_messages(scenario, &response)?;
    let Some((call_index, calls)) = messages.iter().enumerate().find_map(|(index, message)| {
        let Message::Assistant { content, .. } = message else {
            return None;
        };
        let calls = content
            .iter()
            .filter_map(|item| match item {
                AssistantContent::ToolCall(call) => Some(call),
                _ => None,
            })
            .collect::<Vec<_>>();
        (calls.len() == 2).then_some((index, calls))
    }) else {
        return Err(contract(
            scenario,
            format!("no assistant turn contained exactly two tool calls: {messages:?}"),
        ));
    };
    let mut names = calls
        .iter()
        .map(|call| call.function.name.as_str())
        .collect::<Vec<_>>();
    names.sort_unstable();
    if names != ["add", "subtract"] {
        return Err(contract(
            scenario,
            format!("parallel turn called {names:?}, expected add and subtract"),
        ));
    }
    let results_message = messages.get(call_index + 1).ok_or_else(|| {
        contract(
            scenario,
            "parallel call turn has no following result message",
        )
    })?;
    let values = tool_result_values(results_message);
    if !values.iter().any(|value| value_matches_integer(value, 7))
        || !values.iter().any(|value| value_matches_integer(value, 8))
        || values.len() != 2
    {
        return Err(contract(
            scenario,
            format!("parallel result message did not contain exactly 7 and 8: {values:?}"),
        ));
    }
    let add = add_calls.load(Ordering::SeqCst);
    let subtract = subtract_calls.load(Ordering::SeqCst);
    if add != 1 || subtract != 1 {
        return Err(contract(
            scenario,
            format!("execution counts were add={add}, subtract={subtract}, expected one each"),
        ));
    }
    report_from_response(scenario, started, add + subtract, response)
}
pub(super) async fn zero_argument_tool(
    model: impl CompletionModel + 'static,
) -> Result<ScenarioReport, ScenarioError> {
    const SCENARIO: &str = "zero_argument_tool";
    let calls = Arc::new(AtomicUsize::new(0));
    let started = Instant::now();
    let mut ecs = configured(
        model,
        "You must use the provided tools. Report tool outputs exactly as returned.",
        Some(2),
    );
    ecs.tool(PingTool(calls.clone()));
    let response = execute(
        &mut ecs,
        "Call the ping tool, then report the exact marker it returns.",
        false,
        Some(2),
        None,
    )
    .await;
    let values = correlated_result_values(SCENARIO, &response)?;
    if calls.load(Ordering::SeqCst) != 1
        || !values
            .iter()
            .any(|value| value.as_str() == Some(PING_OUTPUT))
        || !response.output.contains(PING_OUTPUT)
    {
        return Err(contract(
            SCENARIO,
            format!(
                "calls={}, results={values:?}, response={:?}",
                calls.load(Ordering::SeqCst),
                response.output
            ),
        ));
    }
    report_from_response(SCENARIO, started, 1, response)
}
pub(super) async fn tool_output_serialization(
    model: impl CompletionModel + 'static,
) -> Result<ScenarioReport, ScenarioError> {
    const SCENARIO: &str = "tool_output_serialization";
    let started = Instant::now();
    let motto_calls = Arc::new(AtomicUsize::new(0));
    let config_calls = Arc::new(AtomicUsize::new(0));
    let mut ecs = configured(
        model,
        "You must use the provided tools before answering.",
        Some(3),
    );
    ecs.tool(MottoTool(motto_calls.clone()));
    ecs.tool(ConfigTool(config_calls.clone()));
    let response = execute(
        &mut ecs,
        "Call fetch_motto and fetch_config, then summarize both outputs in one sentence.",
        false,
        Some(3),
        None,
    )
    .await;
    let values = correlated_result_values(SCENARIO, &response)?;
    let expected_config = serde_json::to_value(ConfigOutput {
        service: "cassette-lab".to_string(),
        max_retries: 3,
    })?;
    let motto_ok = values
        .iter()
        .any(|value| value.as_str() == Some(MOTTO_OUTPUT));
    let config_ok = values.iter().any(|value| {
        value == &expected_config
            || value
                .as_str()
                .and_then(|text| serde_json::from_str::<serde_json::Value>(text).ok())
                .as_ref()
                == Some(&expected_config)
    });
    let motto_count = motto_calls.load(Ordering::SeqCst);
    let config_count = config_calls.load(Ordering::SeqCst);
    if !motto_ok || !config_ok || motto_count != 1 || config_count != 1 {
        return Err(contract(
            SCENARIO,
            format!(
                "expected one verbatim motto and one semantic config JSON; motto_calls={motto_count}, config_calls={config_count}, values={values:?}"
            ),
        ));
    }
    report_from_response(SCENARIO, started, motto_count + config_count, response)
}
