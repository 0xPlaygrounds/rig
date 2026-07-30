//! Provider-neutral behavioral scenarios for completion-model conformance.
//!
//! These helpers test the model/agent contract only. Provider wire formats,
//! authentication, HTTP streaming, and cassette matching remain provider-suite
//! responsibilities.

use std::{
    collections::BTreeSet,
    sync::{
        Arc, Mutex, MutexGuard,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use futures::StreamExt;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    agent::{
        AgentBuilder, CompletionCallAction, InvalidToolCallAction, MultiTurnStreamItem,
        ObservationAction, OutputMode, RequestPatch, StreamingError, ToolCallAction,
        ToolResultAction,
        run::{AgentRun, AgentRunStep, ModelTurn, ModelTurnOutcome},
    },
    completion::{
        AssistantContent, CompletionError, CompletionRequest, Message, PromptError, ToolDefinition,
    },
    provider::{ProviderConfig, Runtime},
    tool::PortableTool,
};
use rig_core::message::{ToolChoice, UserContent};

/// Typed failure from a portable model-conformance scenario.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ScenarioError {
    /// A buffered agent run failed.
    #[error(transparent)]
    Prompt(#[from] PromptError),
    /// A direct model completion failed.
    #[error(transparent)]
    Completion(#[from] CompletionError),
    /// A streaming agent run failed.
    #[error(transparent)]
    Streaming(#[from] StreamingError),
    /// Structured content could not be decoded.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// Structured extraction failed.
    #[error(transparent)]
    Extraction(#[from] crate::extract::ExtractError),
    /// The model or agent violated the portable behavioral contract.
    #[error("{scenario} conformance failed: {details}")]
    Contract {
        /// Stable scenario name.
        scenario: &'static str,
        /// Actionable observation explaining the failure.
        details: String,
    },
}

impl ScenarioError {
    fn contract(scenario: &'static str, details: impl Into<String>) -> Self {
        Self::Contract {
            scenario,
            details: details.into(),
        }
    }
}

/// Validate the portable diagnostics carried by an unknown or disallowed tool
/// call failure.
pub fn validate_unknown_tool_failure(
    error: &PromptError,
    expected_tool: &str,
    expected_allowed_tools: &[&str],
) -> Result<(), ScenarioError> {
    const SCENARIO: &str = "unknown_tool_failure";
    let PromptError::UnknownToolCall {
        tool_name,
        allowed_tools,
        chat_history,
        ..
    } = error
    else {
        return Err(ScenarioError::contract(
            SCENARIO,
            format!("expected UnknownToolCall, observed {error:?}"),
        ));
    };
    let expected_allowed = expected_allowed_tools
        .iter()
        .map(|name| (*name).to_string())
        .collect::<Vec<_>>();
    // A rejected repair target is never written back into history; diagnostics
    // retain the model's original call while `tool_name` names the rejected
    // target. Requiring a call-bearing assistant turn covers both paths.
    let history_has_call = chat_history.iter().any(|message| {
        matches!(
            message,
            Message::Assistant { content, .. }
                if content.iter().any(|item| matches!(item, AssistantContent::ToolCall(_)))
        )
    });
    if tool_name != expected_tool || allowed_tools != &expected_allowed || !history_has_call {
        return Err(ScenarioError::contract(
            SCENARIO,
            format!(
                "tool={tool_name:?}, allowed={allowed_tools:?}, expected_tool={expected_tool:?}, expected_allowed={expected_allowed:?}, history_has_call={history_has_call}, history={chat_history:?}"
            ),
        ));
    }
    Ok(())
}

/// Validate cancellation diagnostics, including the exact reason and retained
/// assistant tool-call history.
pub fn validate_cancelled_failure(
    error: &PromptError,
    expected_reason: &str,
    expected_tool: &str,
) -> Result<(), ScenarioError> {
    const SCENARIO: &str = "cancelled_failure";
    let PromptError::PromptCancelled {
        chat_history,
        reason,
    } = error
    else {
        return Err(ScenarioError::contract(
            SCENARIO,
            format!("expected PromptCancelled, observed {error:?}"),
        ));
    };
    let history_has_call = chat_history.iter().any(|message| {
        matches!(
            message,
            Message::Assistant { content, .. }
                if content.iter().any(|item| matches!(
                    item,
                    AssistantContent::ToolCall(call) if call.function.name == expected_tool
                ))
        )
    });
    if reason != expected_reason || !history_has_call {
        return Err(ScenarioError::contract(
            SCENARIO,
            format!(
                "reason={reason:?}, expected={expected_reason:?}, history_has_call={history_has_call}, history={chat_history:?}"
            ),
        ));
    }
    Ok(())
}

/// Validate max-turn diagnostics, including the exact configured budget and a
/// retained pending prompt.
pub fn validate_max_turns_failure(
    error: &PromptError,
    expected_max_turns: usize,
) -> Result<(), ScenarioError> {
    const SCENARIO: &str = "max_turns_failure";
    let PromptError::MaxTurnsError {
        max_turns,
        chat_history,
        prompt,
    } = error
    else {
        return Err(ScenarioError::contract(
            SCENARIO,
            format!("expected MaxTurnsError, observed {error:?}"),
        ));
    };
    let pending_prompt_retained = matches!(
        prompt.as_ref(),
        Message::User { content } if content.iter().next().is_some()
    );
    if *max_turns != expected_max_turns || chat_history.is_empty() || !pending_prompt_retained {
        return Err(ScenarioError::contract(
            SCENARIO,
            format!(
                "max_turns={max_turns}, expected={expected_max_turns}, history={chat_history:?}, pending_prompt_retained={pending_prompt_retained}, pending_prompt={prompt:?}"
            ),
        ));
    }
    Ok(())
}

/// Decode a structured-output response with a typed conformance failure that
/// retains the scenario name and raw response.
pub fn decode_structured_output<T>(
    scenario: &'static str,
    response: &str,
) -> Result<T, ScenarioError>
where
    T: serde::de::DeserializeOwned,
{
    serde_json::from_str(response).map_err(|error| {
        ScenarioError::contract(
            scenario,
            format!("structured output did not decode: {error}; response={response:?}"),
        )
    })
}

/// Validate that model-family control markers are absent from user-visible
/// output and persisted history.
pub fn validate_protocol_hygiene(
    scenario: &'static str,
    visible_output: &str,
    messages: &[Message],
    forbidden_markers: &[&str],
) -> Result<(), ScenarioError> {
    let serialized = serde_json::to_string(messages)?;
    let leaked = forbidden_markers
        .iter()
        .filter(|marker| visible_output.contains(**marker) || serialized.contains(**marker))
        .copied()
        .collect::<Vec<_>>();
    if !leaked.is_empty() {
        return Err(ScenarioError::contract(
            scenario,
            format!(
                "protocol markers leaked: {leaked:?}; output={visible_output:?}, history={messages:?}"
            ),
        ));
    }
    Ok(())
}

/// Validate that every observed tool invocation contains the expected rewritten
/// argument fields while allowing unrelated original fields to remain.
pub fn validate_rewritten_arguments(
    scenario: &'static str,
    observations: &[serde_json::Value],
    expected_fields: &serde_json::Value,
) -> Result<(), ScenarioError> {
    let Some(expected) = expected_fields.as_object() else {
        return Err(ScenarioError::contract(
            scenario,
            "expected rewritten fields must be a JSON object",
        ));
    };
    if observations.is_empty() {
        return Err(ScenarioError::contract(
            scenario,
            "the rewritten tool was never invoked",
        ));
    }
    for observation in observations {
        let Some(actual) = observation.as_object() else {
            return Err(ScenarioError::contract(
                scenario,
                format!("observed rewritten arguments were not an object: {observation:?}"),
            ));
        };
        for (key, expected_value) in expected {
            if actual.get(key) != Some(expected_value) {
                return Err(ScenarioError::contract(
                    scenario,
                    format!(
                        "rewritten field {key:?} expected {expected_value:?}, observed {observation:?}"
                    ),
                ));
            }
        }
    }
    Ok(())
}

/// Validate that a sensitive raw tool result was produced but did not reach the
/// model's user-visible response after result hooks ran.
pub fn validate_result_redaction(
    scenario: &'static str,
    tool_produced_secret: bool,
    visible_output: &str,
    secret: &str,
) -> Result<(), ScenarioError> {
    if !tool_produced_secret || visible_output.is_empty() || visible_output.contains(secret) {
        return Err(ScenarioError::contract(
            scenario,
            format!(
                "produced_secret={tool_produced_secret}, secret_visible={}, output={visible_output:?}",
                visible_output.contains(secret)
            ),
        ));
    }
    Ok(())
}

/// Validate a provider-neutral person extraction and its usage accounting.
pub fn validate_extraction_fields(
    scenario: &'static str,
    first_name: Option<&str>,
    last_name: Option<&str>,
    job: Option<&str>,
    usage: crate::completion::Usage,
) -> Result<(), ScenarioError> {
    let fields_match = first_name.is_some_and(|value| value.eq_ignore_ascii_case("Ada"))
        && last_name.is_some_and(|value| value.eq_ignore_ascii_case("Lovelace"))
        && job.is_some_and(|value| value.to_ascii_lowercase().contains("mathematician"));
    if !fields_match || !usage.has_values() {
        return Err(ScenarioError::contract(
            scenario,
            format!(
                "first_name={first_name:?}, last_name={last_name:?}, job={job:?}, usage={usage:?}"
            ),
        ));
    }
    Ok(())
}

/// Summary emitted by a portable model-conformance scenario.
#[derive(Debug, Clone)]
pub struct ScenarioReport {
    /// Stable scenario name.
    pub name: &'static str,
    /// Number of model-invoked tool calls observed by the scenario.
    pub tool_calls: usize,
    /// Aggregated prompt tokens reported by the model across all turns.
    pub prompt_tokens: u64,
    /// Aggregated generated tokens reported by the model across all turns.
    pub generated_tokens: u64,
    /// Number of messages retained in the completed run history.
    pub history_messages: usize,
    /// End-to-end scenario duration.
    pub duration: Duration,
    /// The model's final user-visible response.
    pub response: String,
}

/// Error used by the deterministic conformance tools.
#[derive(Debug, thiserror::Error)]
#[error("model-conformance tool failed")]
pub struct ConformanceToolError;

const FORCE_TOOLS_PREAMBLE: &str = "You are a calculator assistant. You MUST use the provided tools for every arithmetic operation instead of computing results yourself. Once you have all the tool results you need, reply with the final numeric answer in plain text.";
const PARALLEL_PROMPT: &str = "Compute 3 + 4 and 10 - 2. You MUST call the add tool and the subtract tool together in your first response, as two parallel function calls, then report both results.";
const PING_OUTPUT: &str = "pong-crimson-7423";
const MOTTO_OUTPUT: &str = "steady hands\ncalm waters";

fn lock_recover<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

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
    for message in messages {
        match message {
            Message::Assistant { content, .. } => {
                calls.extend(content.iter().filter_map(|item| match item {
                    AssistantContent::ToolCall(call) => {
                        Some((call.id.as_str(), call.call_id.as_deref()))
                    }
                    _ => None,
                }));
            }
            Message::User { content } => {
                results.extend(content.iter().filter_map(|item| match item {
                    UserContent::ToolResult(result) => {
                        Some((result.id.as_str(), result.call_id.as_deref()))
                    }
                    _ => None,
                }));
            }
            Message::System { .. } => {}
        }
    }
    if calls.is_empty() {
        return Err(ScenarioError::contract(
            scenario,
            format!("history has no assistant tool calls: {messages:?}"),
        ));
    }
    for (id, call_id) in &calls {
        let matches = results
            .iter()
            .filter(|(result_id, result_call_id)| result_id == id && call_id == result_call_id)
            .count();
        if matches != 1 {
            return Err(ScenarioError::contract(
                scenario,
                format!(
                    "tool call id={id:?} call_id={call_id:?} has {matches} correlated results; calls={calls:?}, results={results:?}"
                ),
            ));
        }
    }
    if results.len() != calls.len() {
        return Err(ScenarioError::contract(
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

impl PortableTool for CountingAdd {
    const NAME: &'static str = "add";
    type Error = ConformanceToolError;
    type Args = OperationArgs;
    type Output = i64;

    fn description(&self) -> String {
        "Add x and y together".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "x": { "type": "number", "description": "The first operand" },
                "y": { "type": "number", "description": "The second operand" }
            },
            "required": ["x", "y"]
        })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(args.x + args.y)
    }
}

#[derive(Clone)]
struct CountingSum(Arc<AtomicUsize>);

impl PortableTool for CountingSum {
    const NAME: &'static str = "sum";
    type Error = ConformanceToolError;
    type Args = OperationArgs;
    type Output = i64;

    fn description(&self) -> String {
        "Add x and y together (alias of add)".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        CountingAdd(Arc::new(AtomicUsize::new(0))).parameters()
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(args.x + args.y)
    }
}

#[derive(Clone)]
struct CountingSubtract(Arc<AtomicUsize>);

impl PortableTool for CountingSubtract {
    const NAME: &'static str = "subtract";
    type Error = ConformanceToolError;
    type Args = OperationArgs;
    type Output = i64;

    fn description(&self) -> String {
        "Subtract y from x (i.e. x - y)".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "x": { "type": "number", "description": "The first operand" },
                "y": { "type": "number", "description": "The second operand" }
            },
            "required": ["x", "y"]
        })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(args.x - args.y)
    }
}

/// Add/replace one argument key on every `add` call, chaining into later
/// entries (the classic `RewriteArgument` hook, as a record).
fn rewrite_argument(key: &'static str, value: serde_json::Value) -> HookEntry {
    hook_entry("rewrite-argument", move |event| {
        let HookEvent::ToolCall { call, .. } = event else {
            return HookDecision::Continue;
        };
        if call.function.name != CountingAdd::NAME {
            return HookDecision::ToolCall(ToolCallAction::run());
        }
        let mut arguments = call.function.arguments;
        let Some(object) = arguments.as_object_mut() else {
            return HookDecision::ToolCall(ToolCallAction::run());
        };
        object.insert(key.to_string(), value.clone());
        HookDecision::ToolCall(ToolCallAction::rewrite(arguments))
    })
}

/// Record the effective arguments each `ToolCall` event carried (the classic
/// `ObserveArguments` hook, as a record).
fn observe_arguments(seen: Arc<Mutex<Vec<serde_json::Value>>>) -> HookEntry {
    hook_entry("observe-arguments", move |event| {
        if let HookEvent::ToolCall { call, .. } = event {
            lock_recover(&seen).push(call.function.arguments);
        }
        HookDecision::ToolCall(ToolCallAction::run())
    })
}

/// Replace every `add` result presentation (the classic `ReplaceResult` hook).
fn replace_result(replacement: &'static str) -> HookEntry {
    hook_entry("replace-result", move |event| {
        let HookEvent::ToolResult { call, .. } = event else {
            return HookDecision::Continue;
        };
        if call.function.name == CountingAdd::NAME {
            HookDecision::ToolResult(ToolResultAction::rewrite(replacement))
        } else {
            HookDecision::ToolResult(ToolResultAction::keep())
        }
    })
}

/// Wrap the *current* presentation, proving rewrites chain (the classic
/// `WrapResult` hook).
fn wrap_result() -> HookEntry {
    hook_entry("wrap-result", |event| {
        let HookEvent::ToolResult {
            call, presentation, ..
        } = event
        else {
            return HookDecision::Continue;
        };
        if call.function.name == CountingAdd::NAME {
            HookDecision::ToolResult(ToolResultAction::rewrite(format!(
                "[{}]",
                presentation.render()
            )))
        } else {
            HookDecision::ToolResult(ToolResultAction::keep())
        }
    })
}

/// Patch only the first turn's request (the classic `FirstTurnPatch` hook —
/// the turn index now arrives on the event instead of the context).
fn first_turn_patch(patch: RequestPatch) -> HookEntry {
    hook_entry("first-turn-patch", move |event| {
        let HookEvent::BeforeModelCall { turn, .. } = event else {
            return HookDecision::Continue;
        };
        if turn == 1 {
            HookDecision::CompletionCall(CompletionCallAction::patch(patch.clone()))
        } else {
            HookDecision::CompletionCall(CompletionCallAction::continue_run())
        }
    })
}

/// Stop the run from the tool-result event (the classic `StopAfterResult`).
fn stop_after_result(reason: &'static str) -> HookEntry {
    hook_entry("stop-after-result", move |event| {
        let HookEvent::ToolResult { call, .. } = event else {
            return HookDecision::Continue;
        };
        if call.function.name == CountingAdd::NAME {
            HookDecision::ToolResult(ToolResultAction::stop(reason))
        } else {
            HookDecision::ToolResult(ToolResultAction::keep())
        }
    })
}

#[derive(Debug, Deserialize, JsonSchema)]
struct EmptyArgs {}

#[derive(Clone)]
struct PingTool(Arc<AtomicUsize>);

impl PortableTool for PingTool {
    const NAME: &'static str = "ping";
    type Error = ConformanceToolError;
    type Args = EmptyArgs;
    type Output = String;

    fn description(&self) -> String {
        "Return the current ping marker. Takes no arguments.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object", "properties": {}, "required": [] })
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(PING_OUTPUT.to_string())
    }
}

#[derive(Clone)]
struct MottoTool(Arc<AtomicUsize>);

impl PortableTool for MottoTool {
    const NAME: &'static str = "fetch_motto";
    type Error = ConformanceToolError;
    type Args = EmptyArgs;
    type Output = String;

    fn description(&self) -> String {
        "Fetch the two-line workshop motto.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object", "properties": {}, "required": [] })
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(MOTTO_OUTPUT.to_string())
    }
}

#[derive(Debug, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
struct ConfigOutput {
    service: String,
    max_retries: u64,
}

impl crate::tool::IntoToolOutput for ConfigOutput {
    fn into_tool_output(self) -> Result<crate::tool::ToolOutput, crate::tool::ToolExecutionError> {
        crate::tool::serialize_to_tool_output(&self)
    }
}

#[derive(Clone)]
struct ConfigTool(Arc<AtomicUsize>);

impl PortableTool for ConfigTool {
    const NAME: &'static str = "fetch_config";
    type Error = ConformanceToolError;
    type Args = EmptyArgs;
    type Output = ConfigOutput;

    fn description(&self) -> String {
        "Fetch the service configuration object.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object", "properties": {}, "required": [] })
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(ConfigOutput {
            service: "cassette-lab".to_string(),
            max_retries: 3,
        })
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ComplexMode {
    Careful,
    Fast,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
struct ComplexProfile {
    name: String,
    tags: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
struct ComplexArgs {
    profile: ComplexProfile,
    mode: ComplexMode,
    note: Option<String>,
    quote: String,
}

impl crate::tool::IntoToolOutput for ComplexArgs {
    fn into_tool_output(self) -> Result<crate::tool::ToolOutput, crate::tool::ToolExecutionError> {
        crate::tool::serialize_to_tool_output(&self)
    }
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct ExtractedPerson {
    #[schemars(required)]
    first_name: Option<String>,
    #[schemars(required)]
    last_name: Option<String>,
    #[schemars(required)]
    job: Option<String>,
}

#[derive(Clone)]
struct CaptureComplexTool {
    calls: Arc<AtomicUsize>,
    captured: Arc<Mutex<Option<ComplexArgs>>>,
}

impl PortableTool for CaptureComplexTool {
    const NAME: &'static str = "store_profile";
    type Error = ConformanceToolError;
    type Args = ComplexArgs;
    type Output = ComplexArgs;

    fn description(&self) -> String {
        "Store one profile with its nested tags, mode, optional note, and exact quoted text."
            .to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(ComplexArgs)).unwrap_or_default()
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        *lock_recover(&self.captured) = Some(args.clone());
        Ok(args)
    }
}

fn has_tool_roundtrip(messages: Option<&[Message]>) -> bool {
    let saw_call = messages.is_some_and(|messages| {
        messages.iter().any(|message| {
            matches!(
                message,
                Message::Assistant { content, .. }
                    if content.iter().any(|item| matches!(item, AssistantContent::ToolCall(_)))
            )
        })
    });
    let saw_result = messages.is_some_and(|messages| {
        messages.iter().any(|message| {
            matches!(
                message,
                Message::User { content }
                    if content.iter().any(|item| matches!(item, UserContent::ToolResult(_)))
            )
        })
    });
    saw_call && saw_result
}

#[derive(Debug, Deserialize, JsonSchema)]
struct RepeatArgs {
    /// The text to repeat.
    text: String,
    /// Number of repetitions; defaults to 2 when omitted.
    times: Option<u32>,
}

#[derive(Clone)]
struct RepeatTool {
    calls: Arc<AtomicUsize>,
}

impl PortableTool for RepeatTool {
    const NAME: &'static str = "repeat_text";
    type Error = ConformanceToolError;
    type Args = RepeatArgs;
    type Output = String;

    fn description(&self) -> String {
        "Repeat `text`. `times` is optional and defaults to 2.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(RepeatArgs)).unwrap_or_default()
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(vec![args.text.as_str(); args.times.unwrap_or(2) as usize].join(" "))
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct BinOpArgs {
    a: i64,
    b: i64,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ArithmeticResult {
    answer: i64,
    explanation: Option<String>,
}

#[derive(Clone)]
struct AddTool(Arc<AtomicUsize>);

impl PortableTool for AddTool {
    const NAME: &'static str = "add";
    type Error = ConformanceToolError;
    type Args = BinOpArgs;
    type Output = i64;

    fn description(&self) -> String {
        "Add two integers a and b.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(BinOpArgs)).unwrap_or_default()
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(args.a + args.b)
    }
}

#[derive(Clone)]
struct MultiplyTool(Arc<AtomicUsize>);

impl PortableTool for MultiplyTool {
    const NAME: &'static str = "multiply";
    type Error = ConformanceToolError;
    type Args = BinOpArgs;
    type Output = i64;

    fn description(&self) -> String {
        "Multiply two integers a and b.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(BinOpArgs)).unwrap_or_default()
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(args.a * args.b)
    }
}

fn report_from_response(
    name: &'static str,
    started: Instant,
    tool_calls: usize,
    response: crate::agent::PromptResponse,
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

/// Build a bare completion request for direct provider calls, bypassing the
/// agent runtime (the moral successor of the removed `CompletionRequestBuilder`).
fn direct_request(
    prompt: &str,
    tools: Vec<ToolDefinition>,
    tool_choice: Option<ToolChoice>,
    max_tokens: u64,
) -> CompletionRequest {
    CompletionRequest {
        model: None,
        preamble: None,
        chat_history: rig_core::OneOrMany::one(Message::user(prompt)),
        documents: Vec::new(),
        tools,
        temperature: Some(0.0),
        max_tokens: Some(max_tokens),
        tool_choice,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    }
}

fn value_matches_integer(value: &serde_json::Value, expected: i64) -> bool {
    value.as_i64() == Some(expected)
        || value
            .as_str()
            .and_then(|text| text.trim().parse::<i64>().ok())
            == Some(expected)
}

/// Runs two independent calls in one assistant turn and validates canonical
/// call/result history correlation.
///
/// Set `tool_concurrency` to `Some(1)` to prove that serial host execution does
/// not split or drop the model's parallel call batch.
pub async fn parallel_tools<F>(
    provider: ProviderConfig,
    configure: F,
    tool_concurrency: Option<usize>,
) -> Result<ScenarioReport, ScenarioError>
where
    F: FnOnce(AgentBuilder) -> AgentBuilder,
{
    let add_calls = Arc::new(AtomicUsize::new(0));
    let subtract_calls = Arc::new(AtomicUsize::new(0));
    let started = Instant::now();
    let agent = configure(AgentBuilder::new(provider))
        .preamble(FORCE_TOOLS_PREAMBLE)
        .temperature(0.0)
        .tool(CountingAdd(add_calls.clone()))
        .tool(CountingSubtract(subtract_calls.clone()))
        .default_max_turns(3)
        .build();
    let request = agent.runner(PARALLEL_PROMPT).max_turns(3);
    let response = match tool_concurrency {
        Some(concurrency) => request.tool_concurrency(concurrency).run().await?,
        None => request.run().await?,
    };
    let scenario = if tool_concurrency == Some(1) {
        "parallel_tools_serial_execution"
    } else {
        "parallel_tools"
    };
    let messages = response.messages.as_deref().ok_or_else(|| {
        ScenarioError::contract(scenario, "extended run omitted accumulated message history")
    })?;
    validate_tool_correlation(scenario, messages)?;

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
        return Err(ScenarioError::contract(
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
        return Err(ScenarioError::contract(
            scenario,
            format!("parallel turn called {names:?}, expected add and subtract"),
        ));
    }
    let results_message = messages.get(call_index + 1).ok_or_else(|| {
        ScenarioError::contract(
            scenario,
            "parallel call turn has no following result message",
        )
    })?;
    let values = tool_result_values(results_message);
    if !values.iter().any(|value| value_matches_integer(value, 7))
        || !values.iter().any(|value| value_matches_integer(value, 8))
        || values.len() != 2
    {
        return Err(ScenarioError::contract(
            scenario,
            format!("parallel result message did not contain exactly 7 and 8: {values:?}"),
        ));
    }
    let add = add_calls.load(Ordering::SeqCst);
    let subtract = subtract_calls.load(Ordering::SeqCst);
    if add != 1 || subtract != 1 {
        return Err(ScenarioError::contract(
            scenario,
            format!("execution counts were add={add}, subtract={subtract}, expected one each"),
        ));
    }
    report_from_response(scenario, started, add + subtract, response)
}

/// Runs a zero-argument tool and validates verbatim string-result handling.
pub async fn zero_argument_tool<F>(
    provider: ProviderConfig,
    configure: F,
) -> Result<ScenarioReport, ScenarioError>
where
    F: FnOnce(AgentBuilder) -> AgentBuilder,
{
    const SCENARIO: &str = "zero_argument_tool";
    let calls = Arc::new(AtomicUsize::new(0));
    let started = Instant::now();
    let agent = configure(AgentBuilder::new(provider))
        .preamble("You must use the provided tools. Report tool outputs exactly as returned.")
        .temperature(0.0)
        .tool(PingTool(calls.clone()))
        .default_max_turns(2)
        .build();
    let response = agent
        .runner("Call the ping tool, then report the exact marker it returns.")
        .max_turns(2)
        .run()
        .await?;
    let messages = response.messages.as_deref().ok_or_else(|| {
        ScenarioError::contract(SCENARIO, "extended run omitted accumulated message history")
    })?;
    validate_tool_correlation(SCENARIO, messages)?;
    let values = messages
        .iter()
        .flat_map(tool_result_values)
        .collect::<Vec<_>>();
    if calls.load(Ordering::SeqCst) != 1
        || !values
            .iter()
            .any(|value| value.as_str() == Some(PING_OUTPUT))
        || !response.output.contains(PING_OUTPUT)
    {
        return Err(ScenarioError::contract(
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

/// Runs string- and JSON-returning tools and validates that neither output is
/// double encoded.
pub async fn tool_output_serialization<F>(
    provider: ProviderConfig,
    configure: F,
) -> Result<ScenarioReport, ScenarioError>
where
    F: FnOnce(AgentBuilder) -> AgentBuilder,
{
    const SCENARIO: &str = "tool_output_serialization";
    let started = Instant::now();
    let motto_calls = Arc::new(AtomicUsize::new(0));
    let config_calls = Arc::new(AtomicUsize::new(0));
    let agent = configure(AgentBuilder::new(provider))
        .preamble("You must use the provided tools before answering.")
        .temperature(0.0)
        .tool(MottoTool(motto_calls.clone()))
        .tool(ConfigTool(config_calls.clone()))
        .default_max_turns(3)
        .build();
    let response = agent
        .runner("Call fetch_motto and fetch_config, then summarize both outputs in one sentence.")
        .max_turns(3)
        .run()
        .await?;
    let messages = response.messages.as_deref().ok_or_else(|| {
        ScenarioError::contract(SCENARIO, "extended run omitted accumulated message history")
    })?;
    validate_tool_correlation(SCENARIO, messages)?;
    let values = messages
        .iter()
        .flat_map(tool_result_values)
        .collect::<Vec<_>>();
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
        return Err(ScenarioError::contract(
            SCENARIO,
            format!(
                "expected one verbatim motto and one semantic config JSON; motto_calls={motto_count}, config_calls={config_count}, values={values:?}"
            ),
        ));
    }
    report_from_response(SCENARIO, started, motto_count + config_count, response)
}

/// Runs a nested, escaped, Unicode-bearing argument payload through typed tool
/// deserialization and validates the exact semantic value received by the tool.
pub async fn complex_tool_arguments<F>(
    provider: ProviderConfig,
    configure: F,
) -> Result<ScenarioReport, ScenarioError>
where
    F: FnOnce(AgentBuilder) -> AgentBuilder,
{
    const SCENARIO: &str = "complex_tool_arguments";
    let expected = ComplexArgs {
        profile: ComplexProfile {
            name: "Zoë \"Z\"".to_string(),
            tags: vec!["rust".to_string(), "東京".to_string()],
        },
        mode: ComplexMode::Careful,
        note: Some("line one\nline two".to_string()),
        quote: "path C:\\tmp and \"quoted\"".to_string(),
    };
    let calls = Arc::new(AtomicUsize::new(0));
    let captured = Arc::new(Mutex::new(None));
    let started = Instant::now();
    let agent = configure(AgentBuilder::new(provider))
        .preamble("Use store_profile exactly once with every value supplied by the user.")
        .temperature(0.0)
        .tool(CaptureComplexTool {
            calls: calls.clone(),
            captured: captured.clone(),
        })
        .default_max_turns(3)
        .build();
    let response = agent
        .runner(
            "Call store_profile with profile.name exactly `Zoë \\\"Z\\\"`, profile.tags exactly [`rust`, `東京`], mode `careful`, note containing the two lines `line one` and `line two` separated by a newline, and quote exactly `path C:\\\\tmp and \\\"quoted\\\"`. Then confirm it was stored.",
        )
        .max_turns(3)
        .run()
        .await?;
    let observed = lock_recover(&captured).clone();
    if calls.load(Ordering::SeqCst) != 1 || observed.as_ref() != Some(&expected) {
        return Err(ScenarioError::contract(
            SCENARIO,
            format!(
                "calls={}, expected={expected:?}, observed={observed:?}, response={:?}",
                calls.load(Ordering::SeqCst),
                response.output
            ),
        ));
    }
    let messages = response.messages.as_deref().ok_or_else(|| {
        ScenarioError::contract(SCENARIO, "extended run omitted accumulated message history")
    })?;
    validate_tool_correlation(SCENARIO, messages)?;
    report_from_response(SCENARIO, started, 1, response)
}

/// Runs the same deterministic text request through buffered and raw streaming
/// completion surfaces and validates equivalent visible content and usage.
pub async fn buffered_streaming_text_parity(
    provider: ProviderConfig,
    rt: Arc<Runtime>,
) -> Result<ScenarioReport, ScenarioError> {
    const SCENARIO: &str = "buffered_streaming_text_parity";
    const PROMPT: &str = "Answer with exactly the single word Paris.";
    let started = Instant::now();
    let request = || direct_request(PROMPT, Vec::new(), None, 32);
    let buffered = crate::provider::complete(&provider, &rt, request()).await?;
    let buffered_text = buffered
        .choice
        .iter()
        .filter_map(|item| match item {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<String>();

    let mut stream = crate::provider::open_stream(&provider, &rt, request()).await?;
    let mut streamed_text = String::new();
    let mut streamed_usage = None;
    while let Some(item) = stream.next().await {
        match item? {
            crate::streaming::StreamedAssistantContent::Text(text) => {
                streamed_text.push_str(&text.text);
            }
            crate::streaming::StreamedAssistantContent::Final(response) => {
                streamed_usage = Some(response.usage);
            }
            crate::streaming::StreamedAssistantContent::ToolCall { .. }
            | crate::streaming::StreamedAssistantContent::ToolCallDelta { .. }
            | crate::streaming::StreamedAssistantContent::Reasoning(_)
            | crate::streaming::StreamedAssistantContent::ReasoningDelta { .. }
            | crate::streaming::StreamedAssistantContent::Unknown(_) => {}
        }
    }
    let usage = streamed_usage.ok_or_else(|| {
        ScenarioError::contract(SCENARIO, "raw stream omitted its final response metadata")
    })?;
    let normalize = |text: &str| {
        text.trim()
            .trim_matches(|character: char| !character.is_alphanumeric())
            .to_string()
    };
    let buffered_answer = normalize(&buffered_text);
    let streamed_answer = normalize(&streamed_text);
    if !buffered_answer.eq_ignore_ascii_case("Paris")
        || !streamed_answer.eq_ignore_ascii_case("Paris")
        || !buffered.usage.has_values()
        || !usage.has_values()
    {
        return Err(ScenarioError::contract(
            SCENARIO,
            format!(
                "buffered={buffered_text:?}, streamed={streamed_text:?}, buffered_usage={:?}, streamed_usage={usage:?}",
                buffered.usage
            ),
        ));
    }
    Ok(ScenarioReport {
        name: SCENARIO,
        tool_calls: 0,
        prompt_tokens: usage.input_tokens,
        generated_tokens: usage.output_tokens,
        history_messages: 0,
        duration: started.elapsed(),
        response: streamed_text,
    })
}

/// Runs Rig's structured extractor and validates both the extracted semantic
/// fields and accumulated usage.
pub async fn structured_extraction(
    provider: ProviderConfig,
) -> Result<ScenarioReport, ScenarioError> {
    const SCENARIO: &str = "structured_extraction";
    const INPUT: &str = "Hello, my name is Ada Lovelace and I work as a mathematician.";
    let started = Instant::now();
    // The classic `ExtractorBuilder` preset keeps this scenario's recorded
    // exchange byte-identical: output tool `submit`, the extraction preamble,
    // `ToolChoice::Required`, no preamble augmentation, one model call.
    let response = crate::extract::extract_with_options::<ExtractedPerson>(
        AgentConfig::new().with_max_tokens(384),
        provider,
        Arc::new(Runtime::new()),
        INPUT,
        crate::extract::ExtractOptions::classic_extractor(),
    )
    .await?;
    validate_extraction_fields(
        SCENARIO,
        response.value.first_name.as_deref(),
        response.value.last_name.as_deref(),
        response.value.job.as_deref(),
        response.usage,
    )?;
    Ok(ScenarioReport {
        name: SCENARIO,
        tool_calls: 1,
        prompt_tokens: response.usage.input_tokens,
        generated_tokens: response.usage.output_tokens,
        history_messages: 0,
        duration: started.elapsed(),
        response: format!(
            "{} {} — {}",
            response.value.first_name.as_deref().unwrap_or_default(),
            response.value.last_name.as_deref().unwrap_or_default(),
            response.value.job.as_deref().unwrap_or_default()
        ),
    })
}

fn restricted_recovery_run(
    prompt: &str,
    turn: ModelTurn,
    retries: usize,
) -> Result<AgentRun, ScenarioError> {
    const SCENARIO: &str = "invalid_tool_recovery";
    let mut run = AgentRun::new(prompt)
        .max_turns(2)
        .max_invalid_tool_call_retries(retries);
    if !matches!(run.next_step()?, AgentRunStep::CallModel { .. }) {
        return Err(ScenarioError::contract(
            SCENARIO,
            "fresh AgentRun did not request a model turn",
        ));
    }
    let outcome = run.model_response(turn)?;
    let ModelTurnOutcome::NeedsResolution(context) = outcome else {
        return Err(ScenarioError::contract(
            SCENARIO,
            format!("disallowed tool call did not require resolution: {outcome:?}"),
        ));
    };
    if context.tool_name != CountingAdd::NAME {
        return Err(ScenarioError::contract(
            SCENARIO,
            format!("expected rejected add call, observed {context:?}"),
        ));
    }
    Ok(run)
}

/// Uses one real model turn to exercise fail-fast, retry exhaustion, repair,
/// rejected repair, and skip handling without executing a disallowed call.
pub async fn invalid_tool_recovery<F>(
    provider: ProviderConfig,
    configure: F,
) -> Result<ScenarioReport, ScenarioError>
where
    F: FnOnce(AgentBuilder) -> AgentBuilder,
{
    const SCENARIO: &str = "invalid_tool_recovery";
    const PROMPT: &str = "Call the add tool exactly once with x=2 and y=3. Do not call sum.";
    let started = Instant::now();
    let add_calls = Arc::new(AtomicUsize::new(0));
    let sum_calls = Arc::new(AtomicUsize::new(0));
    let agent = configure(AgentBuilder::new(provider))
        .preamble(FORCE_TOOLS_PREAMBLE)
        .temperature(0.0)
        .tool(CountingAdd(add_calls.clone()))
        .tool(CountingSum(sum_calls.clone()))
        .tool_choice(ToolChoice::Required)
        .build();
    let captured: Arc<Mutex<Option<ModelTurn>>> = Arc::new(Mutex::new(None));
    let capture_slot = captured.clone();
    let capture_turn = hook_entry("capture-turn", move |event| {
        let HookEvent::CompletionResponse { response, .. } = event else {
            return HookDecision::Continue;
        };
        *lock_recover(&capture_slot) = Some(ModelTurn::new(
            response.message_id.clone(),
            response.choice.clone(),
            response.usage,
            BTreeSet::new(),
            BTreeSet::new(),
        ));
        HookDecision::Observation(ObservationAction::stop("captured conformance model turn"))
    });
    let stopped = agent.runner(PROMPT).add_hook(capture_turn).run().await;
    if !matches!(stopped, Err(PromptError::PromptCancelled { .. })) {
        return Err(ScenarioError::contract(
            SCENARIO,
            format!("capture hook did not stop after the model response: {stopped:?}"),
        ));
    }
    let response = lock_recover(&captured).take().ok_or_else(|| {
        ScenarioError::contract(SCENARIO, "capture hook observed no model response")
    })?;
    let emitted = response
        .choice
        .iter()
        .filter(|item| {
            matches!(item, AssistantContent::ToolCall(call) if call.function.name == CountingAdd::NAME)
        })
        .count();
    if emitted != 1 {
        return Err(ScenarioError::contract(
            SCENARIO,
            format!(
                "model emitted {emitted} add calls, response={:?}",
                response.choice
            ),
        ));
    }
    let executable = BTreeSet::from([CountingAdd::NAME.to_string(), CountingSum::NAME.to_string()]);
    let allowed = BTreeSet::from([CountingSum::NAME.to_string()]);
    let turn = ModelTurn::new(
        response.message_id,
        response.choice,
        response.usage,
        executable,
        allowed,
    );

    let mut fail = restricted_recovery_run(PROMPT, turn.clone(), 0)?;
    let error = match fail.resolve_invalid_tool_call(InvalidToolCallAction::fail()) {
        Err(error) => error,
        Ok(outcome) => {
            return Err(ScenarioError::contract(
                SCENARIO,
                format!("fail action unexpectedly returned {outcome:?}"),
            ));
        }
    };
    validate_unknown_tool_failure(&error, CountingAdd::NAME, &[CountingSum::NAME])?;

    let mut retry = restricted_recovery_run(PROMPT, turn.clone(), 0)?;
    let error = match retry
        .resolve_invalid_tool_call(InvalidToolCallAction::retry("choose an allowed tool"))
    {
        Err(error) => error,
        Ok(outcome) => {
            return Err(ScenarioError::contract(
                SCENARIO,
                format!("exhausted retry unexpectedly returned {outcome:?}"),
            ));
        }
    };
    validate_unknown_tool_failure(&error, CountingAdd::NAME, &[CountingSum::NAME])?;

    let mut rejected_repair = restricted_recovery_run(PROMPT, turn.clone(), 0)?;
    let error =
        match rejected_repair.resolve_invalid_tool_call(InvalidToolCallAction::repair("missing")) {
            Err(error) => error,
            Ok(outcome) => {
                return Err(ScenarioError::contract(
                    SCENARIO,
                    format!("disallowed repair unexpectedly returned {outcome:?}"),
                ));
            }
        };
    validate_unknown_tool_failure(&error, "missing", &[CountingSum::NAME])?;

    let mut repaired = restricted_recovery_run(PROMPT, turn.clone(), 0)?;
    if !matches!(
        repaired.resolve_invalid_tool_call(InvalidToolCallAction::repair(CountingSum::NAME))?,
        ModelTurnOutcome::Continue { .. }
    ) {
        return Err(ScenarioError::contract(
            SCENARIO,
            "valid repair did not continue",
        ));
    }
    let AgentRunStep::CallTools { calls } = repaired.next_step()? else {
        return Err(ScenarioError::contract(
            SCENARIO,
            "valid repair did not produce pending tool execution",
        ));
    };
    let repaired_call = calls.first();
    if calls.len() != 1
        || !repaired_call.is_some_and(|call| {
            call.tool_call.function.name == CountingSum::NAME && call.preresolved_result.is_none()
        })
    {
        return Err(ScenarioError::contract(
            SCENARIO,
            format!("repaired pending calls were incorrect: {calls:?}"),
        ));
    }

    let mut skipped = restricted_recovery_run(PROMPT, turn.clone(), 0)?;
    if !matches!(
        skipped.resolve_invalid_tool_call(InvalidToolCallAction::skip("disabled for this turn"))?,
        ModelTurnOutcome::Continue { .. }
    ) {
        return Err(ScenarioError::contract(SCENARIO, "skip did not continue"));
    }
    let AgentRunStep::CallTools { calls } = skipped.next_step()? else {
        return Err(ScenarioError::contract(
            SCENARIO,
            "skip did not produce a pre-resolved pending call",
        ));
    };
    let skipped_is_preresolved = match calls.first() {
        Some(call) => call.preresolved_result.is_some(),
        None => false,
    };
    if calls.len() != 1 || !skipped_is_preresolved {
        return Err(ScenarioError::contract(
            SCENARIO,
            format!("skipped pending calls were incorrect: {calls:?}"),
        ));
    }
    if add_calls.load(Ordering::SeqCst) != 0 || sum_calls.load(Ordering::SeqCst) != 0 {
        return Err(ScenarioError::contract(
            SCENARIO,
            "recovery scenario executed a tool body",
        ));
    }

    Ok(ScenarioReport {
        name: SCENARIO,
        tool_calls: emitted,
        prompt_tokens: turn.usage.input_tokens,
        generated_tokens: turn.usage.output_tokens,
        history_messages: 2,
        duration: started.elapsed(),
        response: "fail, retry, repair, rejected repair, and skip passed".to_string(),
    })
}

/// Exercises chained argument/result rewrites and a first-turn-only request
/// patch. Completion proves `tool_choice=Required` did not leak to turn two.
pub async fn hook_rewrites_and_request_patch<F>(
    provider: ProviderConfig,
    configure: F,
) -> Result<ScenarioReport, ScenarioError>
where
    F: FnOnce(AgentBuilder) -> AgentBuilder,
{
    const SCENARIO: &str = "hook_rewrites_and_request_patch";
    let started = Instant::now();
    let calls = Arc::new(AtomicUsize::new(0));
    let observed: Arc<Mutex<Vec<serde_json::Value>>> = Arc::new(Mutex::new(Vec::new()));
    let observed_probe = observed.clone();
    let agent = configure(AgentBuilder::new(provider))
        .preamble("Use add for arithmetic and report only the tool result.")
        .temperature(0.0)
        .tool(CountingAdd(calls.clone()))
        .default_max_turns(3)
        .build();
    let response = agent
        .runner("Use add once for x=1 and y=1, then report what the tool returns.")
        .max_turns(3)
        .add_hook(first_turn_patch(
            RequestPatch::new()
                .active_tools([CountingAdd::NAME])
                .tool_choice(ToolChoice::Required),
        ))
        .add_hook(rewrite_argument("x", serde_json::json!(7)))
        .add_hook(rewrite_argument("y", serde_json::json!(8)))
        .add_hook(observe_arguments(observed))
        .add_hook(replace_result("portable-redacted"))
        .add_hook(wrap_result())
        .run()
        .await?;
    let observations = lock_recover(&observed_probe).clone();
    validate_rewritten_arguments(
        SCENARIO,
        &observations,
        &serde_json::json!({ "x": 7, "y": 8 }),
    )?;
    let messages = response.messages.as_deref().ok_or_else(|| {
        ScenarioError::contract(SCENARIO, "extended hook run omitted message history")
    })?;
    let results = messages
        .iter()
        .flat_map(tool_result_values)
        .collect::<Vec<_>>();
    if calls.load(Ordering::SeqCst) != 1
        || !results
            .iter()
            .any(|value| value == &serde_json::json!("[portable-redacted]"))
        || response.completion_calls.len() != 2
    {
        return Err(ScenarioError::contract(
            SCENARIO,
            format!(
                "calls={}, completion_calls={}, results={results:?}, output={:?}",
                calls.load(Ordering::SeqCst),
                response.completion_calls.len(),
                response.output
            ),
        ));
    }
    report_from_response(SCENARIO, started, 1, response)
}

/// Exercises post-execution cancellation and max-turn diagnostics through the
/// public agent driver using real model-emitted calls.
pub async fn cancellation_and_max_turns<F>(
    provider: ProviderConfig,
    configure: F,
) -> Result<ScenarioReport, ScenarioError>
where
    F: Fn(AgentBuilder) -> AgentBuilder,
{
    const SCENARIO: &str = "cancellation_and_max_turns";
    const REASON: &str = "portable result veto";
    let started = Instant::now();
    let cancelled_calls = Arc::new(AtomicUsize::new(0));
    let cancelled_agent = configure(AgentBuilder::new(provider.clone()))
        .preamble("Use add for arithmetic; never calculate by hand.")
        .temperature(0.0)
        .tool(CountingAdd(cancelled_calls.clone()))
        .build();
    let cancelled = match cancelled_agent
        .runner("Use add once to compute x=20 plus y=22.")
        .max_turns(2)
        .add_hook(stop_after_result(REASON))
        .run()
        .await
    {
        Err(error) => error,
        Ok(output) => {
            return Err(ScenarioError::contract(
                SCENARIO,
                format!("result cancellation unexpectedly completed: {output:?}"),
            ));
        }
    };
    validate_cancelled_failure(&cancelled, REASON, CountingAdd::NAME)?;

    let max_turn_calls = Arc::new(AtomicUsize::new(0));
    let max_turn_agent = configure(AgentBuilder::new(provider))
        .preamble("Use add for arithmetic; never calculate by hand.")
        .temperature(0.0)
        .tool(CountingAdd(max_turn_calls.clone()))
        .build();
    let max_turn = match max_turn_agent
        .runner("Use add once to compute x=20 plus y=22, then report the result.")
        .max_turns(1)
        .run()
        .await
    {
        Err(error) => error,
        Ok(output) => {
            return Err(ScenarioError::contract(
                SCENARIO,
                format!("one-turn budget unexpectedly completed: {output:?}"),
            ));
        }
    };
    validate_max_turns_failure(&max_turn, 1)?;
    let cancelled_count = cancelled_calls.load(Ordering::SeqCst);
    let max_turn_count = max_turn_calls.load(Ordering::SeqCst);
    if cancelled_count != 1 || max_turn_count != 1 {
        return Err(ScenarioError::contract(
            SCENARIO,
            format!("cancelled executions={cancelled_count}, max-turn executions={max_turn_count}"),
        ));
    }
    Ok(ScenarioReport {
        name: SCENARIO,
        tool_calls: cancelled_count + max_turn_count,
        prompt_tokens: 0,
        generated_tokens: 0,
        history_messages: 4,
        duration: started.elapsed(),
        response: "post-result cancellation and max-turn diagnostics passed".to_string(),
    })
}

/// Runs the portable optional-argument tool scenario.
///
/// `configure` is deliberately outside the scenario so a provider suite can
/// attach transport-only settings without putting them into the shared model
/// contract.
pub async fn optional_argument<F>(
    provider: ProviderConfig,
    configure: F,
) -> Result<ScenarioReport, ScenarioError>
where
    F: FnOnce(AgentBuilder) -> AgentBuilder,
{
    let calls = Arc::new(AtomicUsize::new(0));
    let started = Instant::now();
    let agent = configure(AgentBuilder::new(provider))
        .preamble("Use the repeat_text tool whenever asked to repeat text.")
        .tool(RepeatTool {
            calls: calls.clone(),
        })
        .default_max_turns(4)
        .build();
    let result = agent
        .runner(
            "Use the repeat_text tool to repeat the word \"banana\" 3 times, then show me the exact result.",
        )
        .run()
        .await?;
    let response = result.output.clone();
    let tool_calls = calls.load(Ordering::SeqCst);
    if tool_calls == 0
        || response.matches("banana").count() < 1
        || !has_tool_roundtrip(result.messages.as_deref())
    {
        return Err(ScenarioError::contract(
            "optional_argument",
            format!("calls={tool_calls}, response={response:?}"),
        ));
    }
    report_from_response("optional_argument", started, tool_calls, result)
}

/// Runs a portable two-tool sequential arithmetic scenario.
pub async fn sequential_tools<F>(
    provider: ProviderConfig,
    configure: F,
) -> Result<ScenarioReport, ScenarioError>
where
    F: FnOnce(AgentBuilder) -> AgentBuilder,
{
    let add_calls = Arc::new(AtomicUsize::new(0));
    let multiply_calls = Arc::new(AtomicUsize::new(0));
    let started = Instant::now();
    let agent = configure(AgentBuilder::new(provider))
        .preamble(
            "You are a calculator. Use the add and multiply tools for arithmetic; never compute by hand.",
        )
        .tool(AddTool(add_calls.clone()))
        .tool(MultiplyTool(multiply_calls.clone()))
        .default_max_turns(6)
        .build();
    let result = agent
        .runner(
            "Compute (4 + 6) * 2. First call the add tool, then call the multiply tool on the result. Tell me the final number.",
        )
        .run()
        .await?;
    let response = result.output.clone();
    let add = add_calls.load(Ordering::SeqCst);
    let multiply = multiply_calls.load(Ordering::SeqCst);
    if add == 0
        || multiply == 0
        || !response.contains("20")
        || !has_tool_roundtrip(result.messages.as_deref())
    {
        return Err(ScenarioError::contract(
            "sequential_tools",
            format!("add={add}, multiply={multiply}, response={response:?}"),
        ));
    }
    report_from_response("sequential_tools", started, add + multiply, result)
}

/// Runs a tool through Rig's multi-turn streaming agent driver.
pub async fn streaming_tool<F>(
    provider: ProviderConfig,
    configure: F,
) -> Result<ScenarioReport, ScenarioError>
where
    F: FnOnce(AgentBuilder) -> AgentBuilder,
{
    let calls = Arc::new(AtomicUsize::new(0));
    let started = Instant::now();
    let agent = configure(AgentBuilder::new(provider))
        .preamble("Use the add tool for arithmetic; do not calculate by hand.")
        .tool(AddTool(calls.clone()))
        .default_max_turns(4)
        .build();
    let mut stream = agent
        .stream_prompt("Use add to calculate 17 + 25, then state the final number.")
        .max_turns(4)
        .await;
    let mut final_response = None;
    let mut final_count = 0_usize;
    let mut completion_usage = crate::completion::Usage::new();
    let mut streamed_call_ids = Vec::new();
    let mut streamed_result_ids = Vec::new();
    while let Some(item) = stream.next().await {
        match item? {
            MultiTurnStreamItem::StreamAssistantItem(
                crate::streaming::StreamedAssistantContent::ToolCall {
                    internal_call_id, ..
                },
            ) => streamed_call_ids.push(internal_call_id),
            MultiTurnStreamItem::StreamUserItem(
                crate::streaming::StreamedUserContent::ToolResult {
                    internal_call_id, ..
                },
            ) => streamed_result_ids.push(internal_call_id),
            MultiTurnStreamItem::CompletionCall(call) => completion_usage += call.usage,
            MultiTurnStreamItem::FinalResponse(response) => {
                final_count += 1;
                final_response = Some(response);
            }
            MultiTurnStreamItem::StreamAssistantItem(_)
            | MultiTurnStreamItem::ToolExecutionCommitted { .. }
            | MultiTurnStreamItem::ModelTurnRetried { .. } => {}
        }
    }
    let result = final_response.ok_or_else(|| {
        ScenarioError::contract("streaming_tool", "stream produced no final response")
    })?;
    let response = result.output.clone();
    let history_messages = result.messages.as_ref().map_or(0, Vec::len);
    let tool_calls = calls.load(Ordering::SeqCst);
    streamed_call_ids.sort();
    streamed_result_ids.sort();
    let correlated_stream =
        !streamed_call_ids.is_empty() && streamed_call_ids == streamed_result_ids;
    let correlated_history = result
        .messages
        .as_deref()
        .is_some_and(|messages| validate_tool_correlation("streaming_tool", messages).is_ok());
    if tool_calls == 0
        || !response.contains("42")
        || history_messages < 4
        || final_count != 1
        || !correlated_stream
        || !correlated_history
        || completion_usage != result.usage
        || result.completion_calls.is_empty()
    {
        return Err(ScenarioError::contract(
            "streaming_tool",
            format!(
                "calls={tool_calls}, final_count={final_count}, streamed_call_ids={streamed_call_ids:?}, streamed_result_ids={streamed_result_ids:?}, completion_usage={completion_usage:?}, final_usage={:?}, history_messages={history_messages}, response={response:?}",
                result.usage
            ),
        ));
    }
    if let Some(messages) = result.messages.as_deref() {
        validate_protocol_hygiene(
            "streaming_tool",
            &response,
            messages,
            &["<tool_call>", "</tool_call>", "<think>", "</think>"],
        )?;
    }
    Ok(ScenarioReport {
        name: "streaming_tool",
        tool_calls,
        prompt_tokens: result.usage.input_tokens,
        generated_tokens: result.usage.output_tokens,
        history_messages,
        duration: started.elapsed(),
        response,
    })
}

/// Runs a normal tool followed by Rig's synthetic structured-output tool.
pub async fn structured_after_tool<F>(
    provider: ProviderConfig,
    configure: F,
) -> Result<ScenarioReport, ScenarioError>
where
    F: FnOnce(AgentBuilder) -> AgentBuilder,
{
    let calls = Arc::new(AtomicUsize::new(0));
    let started = Instant::now();
    let agent = configure(AgentBuilder::new(provider))
        .preamble(
            "Use add for arithmetic, then finish by calling the structured output tool exactly once.",
        )
        .output_schema::<ArithmeticResult>()
        .output_mode(OutputMode::Tool)
        .tool(AddTool(calls.clone()))
        .default_max_turns(5)
        .build();
    let result = agent
        .runner("Use add to calculate 19 + 23. Return answer=42 and a short optional explanation.")
        .run()
        .await?;
    let response = result.output.clone();
    let parsed: ArithmeticResult = serde_json::from_str(&response)?;
    let tool_calls = calls.load(Ordering::SeqCst);
    if tool_calls == 0 || parsed.answer != 42 || !has_tool_roundtrip(result.messages.as_deref()) {
        return Err(ScenarioError::contract(
            "structured_after_tool",
            format!("calls={tool_calls}, response={response:?}"),
        ));
    }
    let _ = parsed.explanation;
    report_from_response("structured_after_tool", started, tool_calls + 1, result)
}

/// Runs all portable tool-choice modes directly against a completion model.
pub async fn tool_choice_modes(
    provider: ProviderConfig,
    rt: Arc<Runtime>,
) -> Result<ScenarioReport, ScenarioError> {
    let definition = |name: &str| ToolDefinition {
        name: name.to_string(),
        description: format!("Return the supplied integer using {name}."),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"]
        }),
    };
    let tools = vec![definition("alpha"), definition("beta")];
    let started = Instant::now();
    let none = crate::provider::complete(
        &provider,
        &rt,
        direct_request(
            "Answer with only the number 4. Do not call a function.",
            tools.clone(),
            Some(ToolChoice::None),
            64,
        ),
    )
    .await?;
    if none
        .choice
        .iter()
        .any(|item| matches!(item, AssistantContent::ToolCall(_)))
    {
        return Err(ScenarioError::contract(
            "tool_choice_modes",
            "tool_choice none emitted a tool call",
        ));
    }

    let required = crate::provider::complete(
        &provider,
        &rt,
        direct_request(
            "Call alpha with value 7.",
            tools.clone(),
            Some(ToolChoice::Required),
            96,
        ),
    )
    .await?;
    let required_calls = required
        .choice
        .iter()
        .filter(|item| matches!(item, AssistantContent::ToolCall(_)))
        .count();
    if required_calls == 0 {
        return Err(ScenarioError::contract(
            "tool_choice_modes",
            "tool_choice required emitted no tool call",
        ));
    }

    let specific = crate::provider::complete(
        &provider,
        &rt,
        direct_request(
            "Call beta with value 9.",
            tools,
            Some(ToolChoice::Specific {
                function_names: vec!["beta".to_string()],
            }),
            96,
        ),
    )
    .await?;
    let specific_calls = specific
        .choice
        .iter()
        .filter_map(|item| match item {
            AssistantContent::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect::<Vec<_>>();
    if specific_calls.is_empty()
        || specific_calls
            .iter()
            .any(|call| call.function.name != "beta")
    {
        return Err(ScenarioError::contract(
            "tool_choice_modes",
            "specific tool choice did not select only beta",
        ));
    }

    Ok(ScenarioReport {
        name: "tool_choice_modes",
        tool_calls: required_calls + specific_calls.len(),
        prompt_tokens: none.usage.input_tokens
            + required.usage.input_tokens
            + specific.usage.input_tokens,
        generated_tokens: none.usage.output_tokens
            + required.usage.output_tokens
            + specific.usage.output_tokens,
        history_messages: 0,
        duration: started.elapsed(),
        response: "none, required, and specific modes passed".to_string(),
    })
}

/// Runs a streamed real-tool turn followed by Rig's synthetic output tool.
pub async fn streaming_structured_after_tool<F>(
    provider: ProviderConfig,
    configure: F,
) -> Result<ScenarioReport, ScenarioError>
where
    F: FnOnce(AgentBuilder) -> AgentBuilder,
{
    let calls = Arc::new(AtomicUsize::new(0));
    let started = Instant::now();
    let agent = configure(AgentBuilder::new(provider))
        .preamble(
            "Use add for arithmetic, then finish by calling the structured output tool exactly once.",
        )
        .output_schema::<ArithmeticResult>()
        .output_mode(OutputMode::Tool)
        .tool(AddTool(calls.clone()))
        .default_max_turns(5)
        .build();
    let mut stream = agent
        .stream_prompt(
            "Use add to calculate 19 + 23. Return answer=42 and a short optional explanation.",
        )
        .max_turns(5)
        .await;
    let mut final_response = None;
    let mut final_count = 0_usize;
    while let Some(item) = stream.next().await {
        if let MultiTurnStreamItem::FinalResponse(response) = item? {
            final_count += 1;
            final_response = Some(response);
        }
    }
    let result = final_response.ok_or_else(|| {
        ScenarioError::contract(
            "streaming_structured_after_tool",
            "stream produced no final response",
        )
    })?;
    let parsed: ArithmeticResult = serde_json::from_str(&result.output)?;
    let calls = calls.load(Ordering::SeqCst);
    if calls == 0
        || final_count != 1
        || parsed.answer != 42
        || !has_tool_roundtrip(result.messages.as_deref())
    {
        return Err(ScenarioError::contract(
            "streaming_structured_after_tool",
            format!(
                "calls={calls}, final_count={final_count}, response={:?}",
                result.output
            ),
        ));
    }
    report_from_response(
        "streaming_structured_after_tool",
        started,
        calls + 1,
        result,
    )
}

// =========================================================================
// Session-driver twins
//
// The scenarios below are data-oriented twins of the classic-agent scenarios
// above, driven by `SessionAgent` / `AgentStream::next_item_with_tools` /
// `extract_with_usage` instead of `AgentBuilder`. Each twin preserves its
// classic sibling's request shape (same preamble, temperature, tools, and
// prompts — proven byte-identical against the recorded provider cassettes)
// and its assertions exactly. The classic scenarios stay untouched until the
// classic runtime is deleted.
// =========================================================================

use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};

use crate::agent::AgentConfig;
use crate::agent_api::SessionAgent;
use crate::executor::ToolExecutor;
use crate::hooks::{HookDecision, HookEntry, HookEvent, Hooks};
use crate::stream::{AgentStream, AgentStreamItem};
use crate::tool::PortableDynamicTool;

/// Plain-data configuration overrides a provider suite may apply to a
/// session-driven conformance scenario.
///
/// This is the data-oriented successor of the classic scenarios'
/// `Fn(AgentBuilder) -> AgentBuilder` closures: every field a consuming
/// suite actually configures through those closures is representable here
/// (today that is `additional_params`; the remaining fields cover the other
/// model-shape settings the classic builder exposed). `None` keeps the
/// scenario's built-in value.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct ScenarioOverrides {
    /// Replace the scenario's system prompt.
    pub preamble: Option<String>,
    /// Replace the scenario's sampling temperature.
    pub temperature: Option<f64>,
    /// Replace the scenario's maximum output-token count.
    pub max_tokens: Option<u64>,
    /// Provider-specific request parameters (schemaless passthrough), e.g.
    /// Ollama's `{"think": false}`.
    pub additional_params: Option<serde_json::Value>,
    /// Replace the scenario's model-call budget.
    pub max_turns: Option<usize>,
}

impl ScenarioOverrides {
    /// No overrides: run the scenario exactly as written.
    pub fn new() -> Self {
        Self::default()
    }

    /// Replace the scenario's system prompt.
    pub fn preamble(mut self, preamble: impl Into<String>) -> Self {
        self.preamble = Some(preamble.into());
        self
    }

    /// Replace the scenario's sampling temperature.
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Replace the scenario's maximum output-token count.
    pub fn max_tokens(mut self, max_tokens: u64) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set provider-specific request parameters.
    pub fn additional_params(mut self, params: serde_json::Value) -> Self {
        self.additional_params = Some(params);
        self
    }

    /// Replace the scenario's model-call budget.
    pub fn max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = Some(max_turns);
        self
    }

    /// Apply the set overrides onto a scenario-built [`AgentConfig`].
    fn apply(&self, mut config: AgentConfig) -> AgentConfig {
        if let Some(preamble) = &self.preamble {
            config.preamble = Some(preamble.clone());
        }
        if let Some(temperature) = self.temperature {
            config.temperature = Some(temperature);
        }
        if let Some(max_tokens) = self.max_tokens {
            config.max_tokens = Some(max_tokens);
        }
        if let Some(params) = &self.additional_params {
            config.additional_params = Some(params.clone());
        }
        if let Some(max_turns) = self.max_turns {
            config.max_turns = Some(max_turns);
        }
        config
    }
}

/// Erase a typed conformance [`PortableTool`] into a [`PortableDynamicTool`]
/// record with an identical provider-facing definition.
fn portable_tool<T>(tool: T) -> PortableDynamicTool
where
    T: PortableTool + 'static,
{
    PortableDynamicTool::from_portable(tool)
}

/// Named hook entry over a synchronous decision function.
fn hook_entry(
    name: &str,
    decide: impl Fn(HookEvent) -> HookDecision + WasmCompatSend + WasmCompatSync + 'static,
) -> HookEntry {
    HookEntry::new(name, move |event| {
        let decision = decide(event);
        Box::pin(async move { decision })
    })
}

/// Build the scenario's base [`AgentConfig`] and apply the overrides.
fn scenario_config(
    overrides: &ScenarioOverrides,
    preamble: &str,
    temperature: Option<f64>,
    max_turns: usize,
) -> AgentConfig {
    let mut config = AgentConfig::new().with_preamble(preamble);
    config.temperature = temperature;
    config.max_turns = Some(max_turns);
    overrides.apply(config)
}

/// Session twin of [`parallel_tools`]: two independent calls in one
/// assistant turn, canonical call/result correlation, executed through
/// [`ToolExecutor::execute_batch`].
pub async fn parallel_tools_session(
    provider: ProviderConfig,
    overrides: ScenarioOverrides,
    tool_concurrency: Option<usize>,
) -> Result<ScenarioReport, ScenarioError> {
    let add_calls = Arc::new(AtomicUsize::new(0));
    let subtract_calls = Arc::new(AtomicUsize::new(0));
    let started = Instant::now();
    let config = scenario_config(&overrides, FORCE_TOOLS_PREAMBLE, Some(0.0), 3);
    let executor = ToolExecutor::new()
        .register(portable_tool(CountingAdd(add_calls.clone())))
        .register(portable_tool(CountingSubtract(subtract_calls.clone())))
        .tool_concurrency(tool_concurrency.unwrap_or(1));
    let agent = SessionAgent::new(config, provider).with_executor(executor);
    let response = agent.run(PARALLEL_PROMPT).await?;
    let scenario = if tool_concurrency == Some(1) {
        "parallel_tools_serial_execution_session"
    } else {
        "parallel_tools_session"
    };
    let messages = response.messages.as_deref().ok_or_else(|| {
        ScenarioError::contract(scenario, "session run omitted accumulated message history")
    })?;
    validate_tool_correlation(scenario, messages)?;

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
        return Err(ScenarioError::contract(
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
        return Err(ScenarioError::contract(
            scenario,
            format!("parallel turn called {names:?}, expected add and subtract"),
        ));
    }
    let results_message = messages.get(call_index + 1).ok_or_else(|| {
        ScenarioError::contract(
            scenario,
            "parallel call turn has no following result message",
        )
    })?;
    let values = tool_result_values(results_message);
    if !values.iter().any(|value| value_matches_integer(value, 7))
        || !values.iter().any(|value| value_matches_integer(value, 8))
        || values.len() != 2
    {
        return Err(ScenarioError::contract(
            scenario,
            format!("parallel result message did not contain exactly 7 and 8: {values:?}"),
        ));
    }
    let add = add_calls.load(Ordering::SeqCst);
    let subtract = subtract_calls.load(Ordering::SeqCst);
    if add != 1 || subtract != 1 {
        return Err(ScenarioError::contract(
            scenario,
            format!("execution counts were add={add}, subtract={subtract}, expected one each"),
        ));
    }
    report_from_response(scenario, started, add + subtract, response)
}

/// Session twin of [`zero_argument_tool`]: a zero-argument tool with verbatim
/// string-result handling.
pub async fn zero_argument_tool_session(
    provider: ProviderConfig,
    overrides: ScenarioOverrides,
) -> Result<ScenarioReport, ScenarioError> {
    const SCENARIO: &str = "zero_argument_tool_session";
    let calls = Arc::new(AtomicUsize::new(0));
    let started = Instant::now();
    let config = scenario_config(
        &overrides,
        "You must use the provided tools. Report tool outputs exactly as returned.",
        Some(0.0),
        2,
    );
    let executor = ToolExecutor::new().register(portable_tool(PingTool(calls.clone())));
    let agent = SessionAgent::new(config, provider).with_executor(executor);
    let response = agent
        .run("Call the ping tool, then report the exact marker it returns.")
        .await?;
    let messages = response.messages.as_deref().ok_or_else(|| {
        ScenarioError::contract(SCENARIO, "session run omitted accumulated message history")
    })?;
    validate_tool_correlation(SCENARIO, messages)?;
    let values = messages
        .iter()
        .flat_map(tool_result_values)
        .collect::<Vec<_>>();
    if calls.load(Ordering::SeqCst) != 1
        || !values
            .iter()
            .any(|value| value.as_str() == Some(PING_OUTPUT))
        || !response.output.contains(PING_OUTPUT)
    {
        return Err(ScenarioError::contract(
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

/// Session twin of [`tool_output_serialization`]: string- and JSON-returning
/// tools, neither output double encoded.
pub async fn tool_output_serialization_session(
    provider: ProviderConfig,
    overrides: ScenarioOverrides,
) -> Result<ScenarioReport, ScenarioError> {
    const SCENARIO: &str = "tool_output_serialization_session";
    let started = Instant::now();
    let motto_calls = Arc::new(AtomicUsize::new(0));
    let config_calls = Arc::new(AtomicUsize::new(0));
    let config = scenario_config(
        &overrides,
        "You must use the provided tools before answering.",
        Some(0.0),
        3,
    );
    let executor = ToolExecutor::new()
        .register(portable_tool(MottoTool(motto_calls.clone())))
        .register(portable_tool(ConfigTool(config_calls.clone())));
    let agent = SessionAgent::new(config, provider).with_executor(executor);
    let response = agent
        .run("Call fetch_motto and fetch_config, then summarize both outputs in one sentence.")
        .await?;
    let messages = response.messages.as_deref().ok_or_else(|| {
        ScenarioError::contract(SCENARIO, "session run omitted accumulated message history")
    })?;
    validate_tool_correlation(SCENARIO, messages)?;
    let values = messages
        .iter()
        .flat_map(tool_result_values)
        .collect::<Vec<_>>();
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
        return Err(ScenarioError::contract(
            SCENARIO,
            format!(
                "expected one verbatim motto and one semantic config JSON; motto_calls={motto_count}, config_calls={config_count}, values={values:?}"
            ),
        ));
    }
    report_from_response(SCENARIO, started, motto_count + config_count, response)
}

/// Session twin of [`complex_tool_arguments`]: nested, escaped,
/// Unicode-bearing arguments through typed deserialization.
pub async fn complex_tool_arguments_session(
    provider: ProviderConfig,
    overrides: ScenarioOverrides,
) -> Result<ScenarioReport, ScenarioError> {
    const SCENARIO: &str = "complex_tool_arguments_session";
    let expected = ComplexArgs {
        profile: ComplexProfile {
            name: "Zoë \"Z\"".to_string(),
            tags: vec!["rust".to_string(), "東京".to_string()],
        },
        mode: ComplexMode::Careful,
        note: Some("line one\nline two".to_string()),
        quote: "path C:\\tmp and \"quoted\"".to_string(),
    };
    let calls = Arc::new(AtomicUsize::new(0));
    let captured = Arc::new(Mutex::new(None));
    let started = Instant::now();
    let config = scenario_config(
        &overrides,
        "Use store_profile exactly once with every value supplied by the user.",
        Some(0.0),
        3,
    );
    let executor = ToolExecutor::new().register(portable_tool(CaptureComplexTool {
        calls: calls.clone(),
        captured: captured.clone(),
    }));
    let agent = SessionAgent::new(config, provider).with_executor(executor);
    let response = agent
        .run(
            "Call store_profile with profile.name exactly `Zoë \\\"Z\\\"`, profile.tags exactly [`rust`, `東京`], mode `careful`, note containing the two lines `line one` and `line two` separated by a newline, and quote exactly `path C:\\\\tmp and \\\"quoted\\\"`. Then confirm it was stored.",
        )
        .await?;
    let observed = lock_recover(&captured).clone();
    if calls.load(Ordering::SeqCst) != 1 || observed.as_ref() != Some(&expected) {
        return Err(ScenarioError::contract(
            SCENARIO,
            format!(
                "calls={}, expected={expected:?}, observed={observed:?}, response={:?}",
                calls.load(Ordering::SeqCst),
                response.output
            ),
        ));
    }
    let messages = response.messages.as_deref().ok_or_else(|| {
        ScenarioError::contract(SCENARIO, "session run omitted accumulated message history")
    })?;
    validate_tool_correlation(SCENARIO, messages)?;
    report_from_response(SCENARIO, started, 1, response)
}

/// Session twin of [`optional_argument`]: an optional-argument tool through
/// the session driver.
pub async fn optional_argument_session(
    provider: ProviderConfig,
    overrides: ScenarioOverrides,
) -> Result<ScenarioReport, ScenarioError> {
    const SCENARIO: &str = "optional_argument_session";
    let calls = Arc::new(AtomicUsize::new(0));
    let started = Instant::now();
    let config = scenario_config(
        &overrides,
        "Use the repeat_text tool whenever asked to repeat text.",
        None,
        4,
    );
    let executor = ToolExecutor::new().register(portable_tool(RepeatTool {
        calls: calls.clone(),
    }));
    let agent = SessionAgent::new(config, provider).with_executor(executor);
    let result = agent
        .run(
            "Use the repeat_text tool to repeat the word \"banana\" 3 times, then show me the exact result.",
        )
        .await?;
    let response = result.output.clone();
    let tool_calls = calls.load(Ordering::SeqCst);
    if tool_calls == 0
        || response.matches("banana").count() < 1
        || !has_tool_roundtrip(result.messages.as_deref())
    {
        return Err(ScenarioError::contract(
            SCENARIO,
            format!("calls={tool_calls}, response={response:?}"),
        ));
    }
    report_from_response(SCENARIO, started, tool_calls, result)
}

/// Session twin of [`sequential_tools`]: two-tool sequential arithmetic.
pub async fn sequential_tools_session(
    provider: ProviderConfig,
    overrides: ScenarioOverrides,
) -> Result<ScenarioReport, ScenarioError> {
    const SCENARIO: &str = "sequential_tools_session";
    let add_calls = Arc::new(AtomicUsize::new(0));
    let multiply_calls = Arc::new(AtomicUsize::new(0));
    let started = Instant::now();
    let config = scenario_config(
        &overrides,
        "You are a calculator. Use the add and multiply tools for arithmetic; never compute by hand.",
        None,
        6,
    );
    let executor = ToolExecutor::new()
        .register(portable_tool(AddTool(add_calls.clone())))
        .register(portable_tool(MultiplyTool(multiply_calls.clone())));
    let agent = SessionAgent::new(config, provider).with_executor(executor);
    let result = agent
        .run(
            "Compute (4 + 6) * 2. First call the add tool, then call the multiply tool on the result. Tell me the final number.",
        )
        .await?;
    let response = result.output.clone();
    let add = add_calls.load(Ordering::SeqCst);
    let multiply = multiply_calls.load(Ordering::SeqCst);
    if add == 0
        || multiply == 0
        || !response.contains("20")
        || !has_tool_roundtrip(result.messages.as_deref())
    {
        return Err(ScenarioError::contract(
            SCENARIO,
            format!("add={add}, multiply={multiply}, response={response:?}"),
        ));
    }
    report_from_response(SCENARIO, started, add + multiply, result)
}

/// Session twin of [`streaming_tool`]: a tool round trip through
/// [`AgentStream::next_item_with_tools`], validating streamed call/result
/// correlation, per-call usage accounting, and final history.
pub async fn streaming_tool_session(
    provider: ProviderConfig,
    overrides: ScenarioOverrides,
) -> Result<ScenarioReport, ScenarioError> {
    const SCENARIO: &str = "streaming_tool_session";
    let calls = Arc::new(AtomicUsize::new(0));
    let started = Instant::now();
    let config = scenario_config(
        &overrides,
        "Use the add tool for arithmetic; do not calculate by hand.",
        None,
        4,
    );
    let executor = ToolExecutor::new().register(portable_tool(AddTool(calls.clone())));
    let mut stream = AgentStream::new(
        config,
        provider,
        Arc::new(Runtime::new()),
        "Use add to calculate 17 + 25, then state the final number.",
    )
    .with_tools(executor.catalog());
    let mut final_response = None;
    let mut final_count = 0_usize;
    let mut completion_usage = crate::completion::Usage::new();
    let mut streamed_call_ids = Vec::new();
    let mut streamed_result_ids = Vec::new();
    while let Some(item) = stream.next_item_with_tools(&executor).await {
        match item? {
            AgentStreamItem::Assistant(crate::streaming::StreamedAssistantContent::ToolCall {
                internal_call_id,
                ..
            }) => streamed_call_ids.push(internal_call_id),
            AgentStreamItem::User(crate::streaming::StreamedUserContent::ToolResult {
                internal_call_id,
                ..
            }) => streamed_result_ids.push(internal_call_id),
            AgentStreamItem::CompletionCall(call) => completion_usage += call.usage,
            AgentStreamItem::Final(response) => {
                final_count += 1;
                final_response = Some(response);
            }
            _ => {}
        }
    }
    let result = final_response
        .ok_or_else(|| ScenarioError::contract(SCENARIO, "stream produced no final response"))?;
    let response = result.output.clone();
    let history_messages = result.messages.as_ref().map_or(0, Vec::len);
    let tool_calls = calls.load(Ordering::SeqCst);
    streamed_call_ids.sort();
    streamed_result_ids.sort();
    let correlated_stream =
        !streamed_call_ids.is_empty() && streamed_call_ids == streamed_result_ids;
    let correlated_history = result
        .messages
        .as_deref()
        .is_some_and(|messages| validate_tool_correlation(SCENARIO, messages).is_ok());
    if tool_calls == 0
        || !response.contains("42")
        || history_messages < 4
        || final_count != 1
        || !correlated_stream
        || !correlated_history
        || completion_usage != result.usage
        || result.completion_calls.is_empty()
    {
        return Err(ScenarioError::contract(
            SCENARIO,
            format!(
                "calls={tool_calls}, final_count={final_count}, streamed_call_ids={streamed_call_ids:?}, streamed_result_ids={streamed_result_ids:?}, completion_usage={completion_usage:?}, final_usage={:?}, history_messages={history_messages}, response={response:?}",
                result.usage
            ),
        ));
    }
    if let Some(messages) = result.messages.as_deref() {
        validate_protocol_hygiene(
            SCENARIO,
            &response,
            messages,
            &["<tool_call>", "</tool_call>", "<think>", "</think>"],
        )?;
    }
    Ok(ScenarioReport {
        name: SCENARIO,
        tool_calls,
        prompt_tokens: result.usage.input_tokens,
        generated_tokens: result.usage.output_tokens,
        history_messages,
        duration: started.elapsed(),
        response,
    })
}

/// Session twin of [`structured_after_tool`]: a real tool followed by the
/// synthetic structured-output tool.
pub async fn structured_after_tool_session(
    provider: ProviderConfig,
    overrides: ScenarioOverrides,
) -> Result<ScenarioReport, ScenarioError> {
    const SCENARIO: &str = "structured_after_tool_session";
    let calls = Arc::new(AtomicUsize::new(0));
    let started = Instant::now();
    let mut config = scenario_config(
        &overrides,
        "Use add for arithmetic, then finish by calling the structured output tool exactly once.",
        None,
        5,
    );
    config.output_schema = Some(schemars::schema_for!(ArithmeticResult));
    config.output_mode = OutputMode::Tool;
    let executor = ToolExecutor::new().register(portable_tool(AddTool(calls.clone())));
    let agent = SessionAgent::new(config, provider).with_executor(executor);
    let result = agent
        .run("Use add to calculate 19 + 23. Return answer=42 and a short optional explanation.")
        .await?;
    let response = result.output.clone();
    let parsed: ArithmeticResult = serde_json::from_str(&response)?;
    let tool_calls = calls.load(Ordering::SeqCst);
    if tool_calls == 0 || parsed.answer != 42 || !has_tool_roundtrip(result.messages.as_deref()) {
        return Err(ScenarioError::contract(
            SCENARIO,
            format!("calls={tool_calls}, response={response:?}"),
        ));
    }
    let _ = parsed.explanation;
    report_from_response(SCENARIO, started, tool_calls + 1, result)
}

/// Session twin of [`streaming_structured_after_tool`]: a streamed real-tool
/// turn followed by the synthetic output tool.
pub async fn streaming_structured_after_tool_session(
    provider: ProviderConfig,
    overrides: ScenarioOverrides,
) -> Result<ScenarioReport, ScenarioError> {
    const SCENARIO: &str = "streaming_structured_after_tool_session";
    let calls = Arc::new(AtomicUsize::new(0));
    let started = Instant::now();
    let mut config = scenario_config(
        &overrides,
        "Use add for arithmetic, then finish by calling the structured output tool exactly once.",
        None,
        5,
    );
    config.output_schema = Some(schemars::schema_for!(ArithmeticResult));
    config.output_mode = OutputMode::Tool;
    let executor = ToolExecutor::new().register(portable_tool(AddTool(calls.clone())));
    let mut stream = AgentStream::new(
        config,
        provider,
        Arc::new(Runtime::new()),
        "Use add to calculate 19 + 23. Return answer=42 and a short optional explanation.",
    )
    .with_tools(executor.catalog());
    let mut final_response = None;
    let mut final_count = 0_usize;
    while let Some(item) = stream.next_item_with_tools(&executor).await {
        if let AgentStreamItem::Final(response) = item? {
            final_count += 1;
            final_response = Some(response);
        }
    }
    let result = final_response
        .ok_or_else(|| ScenarioError::contract(SCENARIO, "stream produced no final response"))?;
    let parsed: ArithmeticResult = serde_json::from_str(&result.output)?;
    let calls = calls.load(Ordering::SeqCst);
    if calls == 0
        || final_count != 1
        || parsed.answer != 42
        || !has_tool_roundtrip(result.messages.as_deref())
    {
        return Err(ScenarioError::contract(
            SCENARIO,
            format!(
                "calls={calls}, final_count={final_count}, response={:?}",
                result.output
            ),
        ));
    }
    report_from_response(SCENARIO, started, calls + 1, result)
}

/// Session twin of [`hook_rewrites_and_request_patch`]: chained
/// argument/result rewrites and a first-turn-only request patch through
/// [`Hooks`]. Completion proves `tool_choice=Required` did not leak to turn
/// two.
pub async fn hook_rewrites_and_request_patch_session(
    provider: ProviderConfig,
    overrides: ScenarioOverrides,
) -> Result<ScenarioReport, ScenarioError> {
    const SCENARIO: &str = "hook_rewrites_and_request_patch_session";
    let started = Instant::now();
    let calls = Arc::new(AtomicUsize::new(0));
    let observed: Arc<Mutex<Vec<serde_json::Value>>> = Arc::new(Mutex::new(Vec::new()));
    let config = scenario_config(
        &overrides,
        "Use add for arithmetic and report only the tool result.",
        Some(0.0),
        3,
    );
    let executor = ToolExecutor::new().register(portable_tool(CountingAdd(calls.clone())));
    let rewrite = |key: &'static str, value: serde_json::Value| {
        move |event: HookEvent| {
            let HookEvent::ToolCall { call, .. } = event else {
                return HookDecision::Continue;
            };
            if call.function.name != CountingAdd::NAME {
                return HookDecision::ToolCall(ToolCallAction::run());
            }
            let mut arguments = call.function.arguments;
            let Some(object) = arguments.as_object_mut() else {
                return HookDecision::ToolCall(ToolCallAction::run());
            };
            object.insert(key.to_string(), value.clone());
            HookDecision::ToolCall(ToolCallAction::rewrite(arguments))
        }
    };
    let observed_sink = observed.clone();
    let hooks = Hooks::new()
        .with(hook_entry("first_turn_patch", |event| {
            let HookEvent::BeforeModelCall { turn, .. } = event else {
                return HookDecision::Continue;
            };
            if turn == 1 {
                HookDecision::CompletionCall(CompletionCallAction::patch(
                    RequestPatch::new()
                        .active_tools([CountingAdd::NAME])
                        .tool_choice(ToolChoice::Required),
                ))
            } else {
                HookDecision::Continue
            }
        }))
        .with(hook_entry("rewrite_x", rewrite("x", serde_json::json!(7))))
        .with(hook_entry("rewrite_y", rewrite("y", serde_json::json!(8))))
        .with(hook_entry("observe", move |event| {
            if let HookEvent::ToolCall { call, .. } = &event {
                lock_recover(&observed_sink).push(call.function.arguments.clone());
            }
            HookDecision::Continue
        }))
        .with(hook_entry("replace_result", |event| {
            let HookEvent::ToolResult { call, .. } = &event else {
                return HookDecision::Continue;
            };
            if call.function.name == CountingAdd::NAME {
                HookDecision::ToolResult(ToolResultAction::rewrite("portable-redacted"))
            } else {
                HookDecision::ToolResult(ToolResultAction::keep())
            }
        }))
        .with(hook_entry("wrap_result", |event| {
            let HookEvent::ToolResult {
                call, presentation, ..
            } = &event
            else {
                return HookDecision::Continue;
            };
            if call.function.name == CountingAdd::NAME {
                HookDecision::ToolResult(ToolResultAction::rewrite(format!(
                    "[{}]",
                    presentation.render()
                )))
            } else {
                HookDecision::ToolResult(ToolResultAction::keep())
            }
        }));
    let agent = SessionAgent::new(config, provider)
        .with_executor(executor)
        .with_hooks(hooks);
    let response = agent
        .run("Use add once for x=1 and y=1, then report what the tool returns.")
        .await?;
    let observations = lock_recover(&observed).clone();
    validate_rewritten_arguments(
        SCENARIO,
        &observations,
        &serde_json::json!({ "x": 7, "y": 8 }),
    )?;
    let messages = response.messages.as_deref().ok_or_else(|| {
        ScenarioError::contract(SCENARIO, "session hook run omitted message history")
    })?;
    let results = messages
        .iter()
        .flat_map(tool_result_values)
        .collect::<Vec<_>>();
    if calls.load(Ordering::SeqCst) != 1
        || !results
            .iter()
            .any(|value| value == &serde_json::json!("[portable-redacted]"))
        || response.completion_calls.len() != 2
    {
        return Err(ScenarioError::contract(
            SCENARIO,
            format!(
                "calls={}, completion_calls={}, results={results:?}, output={:?}",
                calls.load(Ordering::SeqCst),
                response.completion_calls.len(),
                response.output
            ),
        ));
    }
    report_from_response(SCENARIO, started, 1, response)
}

/// Session twin of [`cancellation_and_max_turns`]: post-execution
/// cancellation via a [`Hooks`] tool-result stop, and max-turn diagnostics.
pub async fn cancellation_and_max_turns_session(
    provider: ProviderConfig,
    overrides: ScenarioOverrides,
) -> Result<ScenarioReport, ScenarioError> {
    const SCENARIO: &str = "cancellation_and_max_turns_session";
    const REASON: &str = "portable result veto";
    let started = Instant::now();
    let cancelled_calls = Arc::new(AtomicUsize::new(0));
    let cancelled_config = scenario_config(
        &overrides,
        "Use add for arithmetic; never calculate by hand.",
        Some(0.0),
        2,
    );
    let cancelled_executor =
        ToolExecutor::new().register(portable_tool(CountingAdd(cancelled_calls.clone())));
    let cancelled_agent = SessionAgent::new(cancelled_config, provider.clone())
        .with_executor(cancelled_executor)
        .with_hooks(Hooks::new().with(hook_entry("stop_after_result", |event| {
            let HookEvent::ToolResult { call, .. } = &event else {
                return HookDecision::Continue;
            };
            if call.function.name == CountingAdd::NAME {
                HookDecision::ToolResult(ToolResultAction::stop(REASON))
            } else {
                HookDecision::ToolResult(ToolResultAction::keep())
            }
        })));
    let cancelled = match cancelled_agent
        .run("Use add once to compute x=20 plus y=22.")
        .await
    {
        Err(error) => error,
        Ok(output) => {
            return Err(ScenarioError::contract(
                SCENARIO,
                format!("result cancellation unexpectedly completed: {output:?}"),
            ));
        }
    };
    validate_cancelled_failure(&cancelled, REASON, CountingAdd::NAME)?;

    let max_turn_calls = Arc::new(AtomicUsize::new(0));
    let mut max_turn_config = scenario_config(
        &overrides,
        "Use add for arithmetic; never calculate by hand.",
        Some(0.0),
        1,
    );
    // A caller override of max_turns must not defeat the budget assertion.
    max_turn_config.max_turns = Some(1);
    let max_turn_executor =
        ToolExecutor::new().register(portable_tool(CountingAdd(max_turn_calls.clone())));
    let max_turn_agent =
        SessionAgent::new(max_turn_config, provider).with_executor(max_turn_executor);
    let max_turn = match max_turn_agent
        .run("Use add once to compute x=20 plus y=22, then report the result.")
        .await
    {
        Err(error) => error,
        Ok(output) => {
            return Err(ScenarioError::contract(
                SCENARIO,
                format!("one-turn budget unexpectedly completed: {output:?}"),
            ));
        }
    };
    validate_max_turns_failure(&max_turn, 1)?;
    let cancelled_count = cancelled_calls.load(Ordering::SeqCst);
    let max_turn_count = max_turn_calls.load(Ordering::SeqCst);
    if cancelled_count != 1 || max_turn_count != 1 {
        return Err(ScenarioError::contract(
            SCENARIO,
            format!("cancelled executions={cancelled_count}, max-turn executions={max_turn_count}"),
        ));
    }
    Ok(ScenarioReport {
        name: SCENARIO,
        tool_calls: cancelled_count + max_turn_count,
        prompt_tokens: 0,
        generated_tokens: 0,
        history_messages: 4,
        duration: started.elapsed(),
        response: "post-result cancellation and max-turn diagnostics passed".to_string(),
    })
}

/// Session twin of [`invalid_tool_recovery`]: one real model turn captured
/// through [`Hooks`], then fail-fast, retry exhaustion, repair, rejected
/// repair, and skip handling on a pure [`AgentRun`] — no tool body executes.
pub async fn invalid_tool_recovery_session(
    provider: ProviderConfig,
    overrides: ScenarioOverrides,
) -> Result<ScenarioReport, ScenarioError> {
    const SCENARIO: &str = "invalid_tool_recovery_session";
    const PROMPT: &str = "Call the add tool exactly once with x=2 and y=3. Do not call sum.";
    let started = Instant::now();
    let add_calls = Arc::new(AtomicUsize::new(0));
    let sum_calls = Arc::new(AtomicUsize::new(0));
    let mut config = scenario_config(&overrides, FORCE_TOOLS_PREAMBLE, Some(0.0), 1);
    config.tool_choice = Some(ToolChoice::Required);
    let executor = ToolExecutor::new()
        .register(portable_tool(CountingAdd(add_calls.clone())))
        .register(portable_tool(CountingSum(sum_calls.clone())));
    let captured: Arc<Mutex<Option<rig_core::completion::CompletionResponse>>> =
        Arc::new(Mutex::new(None));
    let capture_sink = captured.clone();
    let agent = SessionAgent::new(config, provider)
        .with_executor(executor)
        .with_hooks(Hooks::new().with(hook_entry("capture_turn", move |event| {
            if let HookEvent::CompletionResponse { response, .. } = event {
                *lock_recover(&capture_sink) = Some(response);
            }
            HookDecision::Observation(ObservationAction::stop("captured conformance model turn"))
        })));
    let stopped = agent.run(PROMPT).await;
    if !matches!(stopped, Err(PromptError::PromptCancelled { .. })) {
        return Err(ScenarioError::contract(
            SCENARIO,
            format!("capture hook did not stop after the model response: {stopped:?}"),
        ));
    }
    let response = lock_recover(&captured).take().ok_or_else(|| {
        ScenarioError::contract(SCENARIO, "capture hook observed no model response")
    })?;
    let emitted = response
        .choice
        .iter()
        .filter(|item| {
            matches!(item, AssistantContent::ToolCall(call) if call.function.name == CountingAdd::NAME)
        })
        .count();
    if emitted != 1 {
        return Err(ScenarioError::contract(
            SCENARIO,
            format!(
                "model emitted {emitted} add calls, response={:?}",
                response.choice
            ),
        ));
    }
    let executable = BTreeSet::from([CountingAdd::NAME.to_string(), CountingSum::NAME.to_string()]);
    let allowed = BTreeSet::from([CountingSum::NAME.to_string()]);
    let turn = ModelTurn::new(
        response.message_id,
        response.choice,
        response.usage,
        executable,
        allowed,
    );

    let mut fail = restricted_recovery_run(PROMPT, turn.clone(), 0)?;
    let error = match fail.resolve_invalid_tool_call(InvalidToolCallAction::fail()) {
        Err(error) => error,
        Ok(outcome) => {
            return Err(ScenarioError::contract(
                SCENARIO,
                format!("fail action unexpectedly returned {outcome:?}"),
            ));
        }
    };
    validate_unknown_tool_failure(&error, CountingAdd::NAME, &[CountingSum::NAME])?;

    let mut retry = restricted_recovery_run(PROMPT, turn.clone(), 0)?;
    let error = match retry
        .resolve_invalid_tool_call(InvalidToolCallAction::retry("choose an allowed tool"))
    {
        Err(error) => error,
        Ok(outcome) => {
            return Err(ScenarioError::contract(
                SCENARIO,
                format!("exhausted retry unexpectedly returned {outcome:?}"),
            ));
        }
    };
    validate_unknown_tool_failure(&error, CountingAdd::NAME, &[CountingSum::NAME])?;

    let mut rejected_repair = restricted_recovery_run(PROMPT, turn.clone(), 0)?;
    let error =
        match rejected_repair.resolve_invalid_tool_call(InvalidToolCallAction::repair("missing")) {
            Err(error) => error,
            Ok(outcome) => {
                return Err(ScenarioError::contract(
                    SCENARIO,
                    format!("disallowed repair unexpectedly returned {outcome:?}"),
                ));
            }
        };
    validate_unknown_tool_failure(&error, "missing", &[CountingSum::NAME])?;

    let mut repaired = restricted_recovery_run(PROMPT, turn.clone(), 0)?;
    if !matches!(
        repaired.resolve_invalid_tool_call(InvalidToolCallAction::repair(CountingSum::NAME))?,
        ModelTurnOutcome::Continue { .. }
    ) {
        return Err(ScenarioError::contract(
            SCENARIO,
            "valid repair did not continue",
        ));
    }
    let AgentRunStep::CallTools { calls } = repaired.next_step()? else {
        return Err(ScenarioError::contract(
            SCENARIO,
            "valid repair did not produce pending tool execution",
        ));
    };
    let repaired_call = calls.first();
    if calls.len() != 1
        || !repaired_call.is_some_and(|call| {
            call.tool_call.function.name == CountingSum::NAME && call.preresolved_result.is_none()
        })
    {
        return Err(ScenarioError::contract(
            SCENARIO,
            format!("repaired pending calls were incorrect: {calls:?}"),
        ));
    }

    let mut skipped = restricted_recovery_run(PROMPT, turn.clone(), 0)?;
    if !matches!(
        skipped.resolve_invalid_tool_call(InvalidToolCallAction::skip("disabled for this turn"))?,
        ModelTurnOutcome::Continue { .. }
    ) {
        return Err(ScenarioError::contract(SCENARIO, "skip did not continue"));
    }
    let AgentRunStep::CallTools { calls } = skipped.next_step()? else {
        return Err(ScenarioError::contract(
            SCENARIO,
            "skip did not produce a pre-resolved pending call",
        ));
    };
    let skipped_is_preresolved = match calls.first() {
        Some(call) => call.preresolved_result.is_some(),
        None => false,
    };
    if calls.len() != 1 || !skipped_is_preresolved {
        return Err(ScenarioError::contract(
            SCENARIO,
            format!("skipped pending calls were incorrect: {calls:?}"),
        ));
    }
    if add_calls.load(Ordering::SeqCst) != 0 || sum_calls.load(Ordering::SeqCst) != 0 {
        return Err(ScenarioError::contract(
            SCENARIO,
            "recovery scenario executed a tool body",
        ));
    }

    Ok(ScenarioReport {
        name: SCENARIO,
        tool_calls: emitted,
        prompt_tokens: turn.usage.input_tokens,
        generated_tokens: turn.usage.output_tokens,
        history_messages: 2,
        duration: started.elapsed(),
        response: "fail, retry, repair, rejected repair, and skip passed".to_string(),
    })
}

/// Session twin of [`structured_extraction`]: [`crate::extract::extract_with_usage`]
/// over the session runtime, validating extracted fields and usage.
///
/// Note: the session extraction path advertises the default synthetic output
/// tool (`final_result`) without the classic extractor's `submit` tool name,
/// bespoke preamble, or forced tool choice, so its wire shape intentionally
/// differs from the classic extractor's recorded cassettes.
pub async fn structured_extraction_session(
    provider: ProviderConfig,
    overrides: ScenarioOverrides,
) -> Result<ScenarioReport, ScenarioError> {
    const SCENARIO: &str = "structured_extraction_session";
    const INPUT: &str = "Hello, my name is Ada Lovelace and I work as a mathematician.";
    let started = Instant::now();
    let mut config = AgentConfig::new();
    config.max_tokens = Some(384);
    let config = overrides.apply(config);
    let outcome = crate::extract::extract_with_usage::<ExtractedPerson>(
        config,
        provider,
        Arc::new(Runtime::new()),
        INPUT,
        0,
    )
    .await
    .map_err(|error| ScenarioError::contract(SCENARIO, format!("extraction failed: {error}")))?;
    validate_extraction_fields(
        SCENARIO,
        outcome.value.first_name.as_deref(),
        outcome.value.last_name.as_deref(),
        outcome.value.job.as_deref(),
        outcome.usage,
    )?;
    Ok(ScenarioReport {
        name: SCENARIO,
        tool_calls: 1,
        prompt_tokens: outcome.usage.input_tokens,
        generated_tokens: outcome.usage.output_tokens,
        history_messages: 0,
        duration: started.elapsed(),
        response: format!(
            "{} {} — {}",
            outcome.value.first_name.as_deref().unwrap_or_default(),
            outcome.value.last_name.as_deref().unwrap_or_default(),
            outcome.value.job.as_deref().unwrap_or_default()
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::completion::Usage;
    use crate::provider::MockScript;
    use rig_core::{
        OneOrMany,
        completion::CompletionResponse as ModelResponse,
        message::{Text, ToolCall, ToolFunction},
        streaming::{StreamFinal, StreamedAssistantContent},
    };

    fn tool_call(id: &str, name: &str, arguments: serde_json::Value) -> AssistantContent {
        AssistantContent::ToolCall(ToolCall::new(
            id.to_string(),
            ToolFunction::new(name.to_string(), arguments),
        ))
    }

    fn usage(input: u64, output: u64) -> Usage {
        Usage {
            input_tokens: input,
            output_tokens: output,
            total_tokens: input + output,
            ..Usage::new()
        }
    }

    fn response_from(contents: Vec<AssistantContent>, usage: Usage) -> ModelResponse {
        let choice = OneOrMany::many(contents).expect("at least one content item");
        ModelResponse::new(choice, usage, "mock")
    }

    fn text_response(text: &str) -> ModelResponse {
        response_from(vec![AssistantContent::text(text)], Usage::new())
    }

    fn tool_call_response(id: &str, name: &str, arguments: serde_json::Value) -> ModelResponse {
        response_from(vec![tool_call(id, name, arguments)], Usage::new())
    }

    fn mock(responses: Vec<ModelResponse>) -> ProviderConfig {
        ProviderConfig::Mock(MockScript::from_responses(responses))
    }

    fn fixture_contract(condition: bool, details: &str) -> Result<(), ScenarioError> {
        if condition {
            Ok(())
        } else {
            Err(ScenarioError::contract("test_fixture", details))
        }
    }

    #[tokio::test]
    async fn parallel_contract_validates_batch_and_correlation() -> Result<(), ScenarioError> {
        let first = response_from(
            vec![
                tool_call("call_add", "add", serde_json::json!({"x": 3, "y": 4})),
                tool_call(
                    "call_subtract",
                    "subtract",
                    serde_json::json!({"x": 10, "y": 2}),
                ),
            ],
            Usage::new(),
        );
        let report = parallel_tools(
            mock(vec![first, text_response("7 and 8")]),
            |builder| builder,
            Some(1),
        )
        .await?;
        fixture_contract(report.tool_calls == 2, "parallel tool-call count")?;
        fixture_contract(report.history_messages >= 4, "parallel history length")?;
        Ok(())
    }

    #[tokio::test]
    async fn zero_argument_and_output_serialization_contracts_pass() -> Result<(), ScenarioError> {
        let zero = zero_argument_tool(
            mock(vec![
                tool_call_response("ping_call", "ping", serde_json::json!({})),
                text_response(PING_OUTPUT),
            ]),
            |builder| builder,
        )
        .await?;
        fixture_contract(zero.tool_calls == 1, "zero-argument call count")?;

        let first = response_from(
            vec![
                tool_call("motto_call", "fetch_motto", serde_json::json!({})),
                tool_call("config_call", "fetch_config", serde_json::json!({})),
            ],
            Usage::new(),
        );
        let serialized =
            tool_output_serialization(mock(vec![first, text_response("summary")]), |builder| {
                builder
            })
            .await?;
        fixture_contract(serialized.tool_calls == 2, "serialized-output call count")?;
        Ok(())
    }

    #[tokio::test]
    async fn complex_arguments_preserve_nested_unicode_and_escapes() -> Result<(), ScenarioError> {
        let arguments = serde_json::json!({
            "profile": {"name": "Zoë \"Z\"", "tags": ["rust", "東京"]},
            "mode": "careful",
            "note": "line one\nline two",
            "quote": "path C:\\tmp and \"quoted\""
        });
        let report = complex_tool_arguments(
            mock(vec![
                tool_call_response("profile_call", "store_profile", arguments),
                text_response("stored"),
            ]),
            |builder| builder,
        )
        .await?;
        fixture_contract(report.tool_calls == 1, "complex-argument call count")?;
        Ok(())
    }

    #[tokio::test]
    async fn extraction_contract_requires_fields_and_usage() -> Result<(), ScenarioError> {
        let report = structured_extraction(mock(vec![response_from(
            vec![tool_call(
                "submit_call",
                "submit",
                serde_json::json!({
                    "first_name": "Ada",
                    "last_name": "Lovelace",
                    "job": "mathematician"
                }),
            )],
            usage(20, 5),
        )]))
        .await?;
        fixture_contract(report.prompt_tokens == 20, "extraction input usage")?;
        fixture_contract(report.generated_tokens == 5, "extraction output usage")?;
        Ok(())
    }

    #[tokio::test]
    async fn streaming_contract_checks_events_history_and_usage() -> Result<(), ScenarioError> {
        let script = MockScript::from_responses(Vec::new()).with_streams(vec![
            vec![
                StreamedAssistantContent::ToolCall {
                    tool_call: ToolCall::new(
                        "add_call".to_string(),
                        ToolFunction::new("add".to_string(), serde_json::json!({"a": 17, "b": 25})),
                    ),
                    internal_call_id: "mock-internal-1".to_string(),
                },
                StreamedAssistantContent::Final(StreamFinal::new("mock", usage(10, 2))),
            ],
            vec![
                StreamedAssistantContent::Text(Text::new("42")),
                StreamedAssistantContent::Final(StreamFinal::new("mock", usage(14, 1))),
            ],
        ]);
        let report = streaming_tool(ProviderConfig::Mock(script), |builder| builder).await?;
        fixture_contract(report.prompt_tokens == 24, "streaming input usage")?;
        fixture_contract(report.generated_tokens == 3, "streaming output usage")?;
        Ok(())
    }

    #[tokio::test]
    async fn invalid_recovery_paths_do_not_execute_tools() -> Result<(), ScenarioError> {
        let report = invalid_tool_recovery(
            mock(vec![tool_call_response(
                "invalid-add",
                "add",
                serde_json::json!({ "x": 2, "y": 3 }),
            )]),
            |builder| builder,
        )
        .await?;
        fixture_contract(report.tool_calls == 1, "recovery source call count")?;
        Ok(())
    }

    #[tokio::test]
    async fn hook_rewrites_chain_and_request_patch_is_turn_local() -> Result<(), ScenarioError> {
        let report = hook_rewrites_and_request_patch(
            mock(vec![
                tool_call_response("hook-add", "add", serde_json::json!({ "x": 1, "y": 1 })),
                text_response("[portable-redacted]"),
            ]),
            |builder| builder,
        )
        .await?;
        fixture_contract(report.tool_calls == 1, "hook execution count")?;
        Ok(())
    }

    #[tokio::test]
    async fn cancellation_and_max_turn_controls_retain_diagnostics() -> Result<(), ScenarioError> {
        let report = cancellation_and_max_turns(
            mock(vec![
                tool_call_response("cancel-add", "add", serde_json::json!({ "x": 20, "y": 22 })),
                tool_call_response("budget-add", "add", serde_json::json!({ "x": 20, "y": 22 })),
            ]),
            |builder| builder,
        )
        .await?;
        fixture_contract(report.tool_calls == 2, "run-control execution count")?;
        Ok(())
    }

    #[test]
    fn typed_validators_reject_bad_structured_output_and_protocol_leaks() {
        let invalid = decode_structured_output::<ConfigOutput>("invalid_json", "not json");
        assert!(matches!(invalid, Err(ScenarioError::Contract { .. })));

        let messages = vec![Message::Assistant {
            id: None,
            content: OneOrMany::one(AssistantContent::text("visible <tool_call>")),
        }];
        let hygiene = validate_protocol_hygiene(
            "protocol_hygiene",
            "visible <tool_call>",
            &messages,
            &["<tool_call>"],
        );
        assert!(matches!(hygiene, Err(ScenarioError::Contract { .. })));
    }

    #[tokio::test]
    async fn session_parallel_contract_validates_batch_and_correlation() -> Result<(), ScenarioError>
    {
        let first = response_from(
            vec![
                tool_call("call_add", "add", serde_json::json!({"x": 3, "y": 4})),
                tool_call(
                    "call_subtract",
                    "subtract",
                    serde_json::json!({"x": 10, "y": 2}),
                ),
            ],
            Usage::new(),
        );
        let report = parallel_tools_session(
            mock(vec![first, text_response("7 and 8")]),
            ScenarioOverrides::new(),
            Some(1),
        )
        .await?;
        fixture_contract(report.tool_calls == 2, "session parallel tool-call count")?;
        fixture_contract(report.history_messages >= 4, "session parallel history")?;
        Ok(())
    }

    #[tokio::test]
    async fn session_zero_argument_and_output_serialization_contracts_pass()
    -> Result<(), ScenarioError> {
        let zero = zero_argument_tool_session(
            mock(vec![
                tool_call_response("ping_call", "ping", serde_json::json!({})),
                text_response(PING_OUTPUT),
            ]),
            ScenarioOverrides::new(),
        )
        .await?;
        fixture_contract(zero.tool_calls == 1, "session zero-argument call count")?;

        let first = response_from(
            vec![
                tool_call("motto_call", "fetch_motto", serde_json::json!({})),
                tool_call("config_call", "fetch_config", serde_json::json!({})),
            ],
            Usage::new(),
        );
        let serialized = tool_output_serialization_session(
            mock(vec![first, text_response("summary")]),
            ScenarioOverrides::new(),
        )
        .await?;
        fixture_contract(
            serialized.tool_calls == 2,
            "session serialized-output call count",
        )?;
        Ok(())
    }

    #[tokio::test]
    async fn session_complex_arguments_preserve_nested_unicode_and_escapes()
    -> Result<(), ScenarioError> {
        let arguments = serde_json::json!({
            "profile": {"name": "Zoë \"Z\"", "tags": ["rust", "東京"]},
            "mode": "careful",
            "note": "line one\nline two",
            "quote": "path C:\\tmp and \"quoted\""
        });
        let report = complex_tool_arguments_session(
            mock(vec![
                tool_call_response("profile_call", "store_profile", arguments),
                text_response("stored"),
            ]),
            ScenarioOverrides::new(),
        )
        .await?;
        fixture_contract(
            report.tool_calls == 1,
            "session complex-argument call count",
        )?;
        Ok(())
    }

    #[tokio::test]
    async fn session_extraction_contract_requires_fields_and_usage() -> Result<(), ScenarioError> {
        let report = structured_extraction_session(
            mock(vec![response_from(
                vec![tool_call(
                    "submit_call",
                    "final_result",
                    serde_json::json!({
                        "first_name": "Ada",
                        "last_name": "Lovelace",
                        "job": "mathematician"
                    }),
                )],
                usage(20, 5),
            )]),
            ScenarioOverrides::new(),
        )
        .await?;
        fixture_contract(report.prompt_tokens == 20, "session extraction input usage")?;
        fixture_contract(
            report.generated_tokens == 5,
            "session extraction output usage",
        )?;
        Ok(())
    }

    #[tokio::test]
    async fn session_streaming_contract_checks_events_history_and_usage()
    -> Result<(), ScenarioError> {
        let script = MockScript::from_responses(Vec::new()).with_streams(vec![
            vec![
                StreamedAssistantContent::ToolCall {
                    tool_call: ToolCall::new(
                        "add_call".to_string(),
                        ToolFunction::new("add".to_string(), serde_json::json!({"a": 17, "b": 25})),
                    ),
                    internal_call_id: "mock-internal-1".to_string(),
                },
                StreamedAssistantContent::Final(StreamFinal::new("mock", usage(10, 2))),
            ],
            vec![
                StreamedAssistantContent::Text(Text::new("42")),
                StreamedAssistantContent::Final(StreamFinal::new("mock", usage(14, 1))),
            ],
        ]);
        let report =
            streaming_tool_session(ProviderConfig::Mock(script), ScenarioOverrides::new()).await?;
        fixture_contract(report.prompt_tokens == 24, "session streaming input usage")?;
        fixture_contract(
            report.generated_tokens == 3,
            "session streaming output usage",
        )?;
        Ok(())
    }

    #[tokio::test]
    async fn session_invalid_recovery_paths_do_not_execute_tools() -> Result<(), ScenarioError> {
        let report = invalid_tool_recovery_session(
            mock(vec![tool_call_response(
                "invalid-add",
                "add",
                serde_json::json!({ "x": 2, "y": 3 }),
            )]),
            ScenarioOverrides::new(),
        )
        .await?;
        fixture_contract(report.tool_calls == 1, "session recovery source call count")?;
        Ok(())
    }

    #[tokio::test]
    async fn session_hook_rewrites_chain_and_request_patch_is_turn_local()
    -> Result<(), ScenarioError> {
        let report = hook_rewrites_and_request_patch_session(
            mock(vec![
                tool_call_response("hook-add", "add", serde_json::json!({ "x": 1, "y": 1 })),
                text_response("[portable-redacted]"),
            ]),
            ScenarioOverrides::new(),
        )
        .await?;
        fixture_contract(report.tool_calls == 1, "session hook execution count")?;
        Ok(())
    }

    #[tokio::test]
    async fn session_cancellation_and_max_turn_controls_retain_diagnostics()
    -> Result<(), ScenarioError> {
        let report = cancellation_and_max_turns_session(
            mock(vec![
                tool_call_response("cancel-add", "add", serde_json::json!({ "x": 20, "y": 22 })),
                tool_call_response("budget-add", "add", serde_json::json!({ "x": 20, "y": 22 })),
            ]),
            ScenarioOverrides::new(),
        )
        .await?;
        fixture_contract(
            report.tool_calls == 2,
            "session run-control execution count",
        )?;
        Ok(())
    }

    #[tokio::test]
    async fn scenario_overrides_apply_over_scenario_defaults() {
        let config = ScenarioOverrides::new()
            .preamble("override preamble")
            .temperature(0.5)
            .max_tokens(99)
            .additional_params(serde_json::json!({"think": false}))
            .max_turns(7)
            .apply(scenario_config(
                &ScenarioOverrides::new(),
                "scenario preamble",
                Some(0.0),
                2,
            ));
        assert_eq!(config.preamble.as_deref(), Some("override preamble"));
        assert_eq!(config.temperature, Some(0.5));
        assert_eq!(config.max_tokens, Some(99));
        assert_eq!(
            config.additional_params,
            Some(serde_json::json!({"think": false}))
        );
        assert_eq!(config.max_turns, Some(7));

        // Unset overrides keep the scenario's built-in values.
        let untouched = scenario_config(&ScenarioOverrides::new(), "scenario preamble", None, 4);
        assert_eq!(untouched.preamble.as_deref(), Some("scenario preamble"));
        assert_eq!(untouched.temperature, None);
        assert_eq!(untouched.max_turns, Some(4));
    }

    #[test]
    fn invalid_tool_diagnostics_require_rejected_call_history() {
        let history = vec![Message::Assistant {
            id: None,
            content: OneOrMany::one(tool_call(
                "bad_call",
                "missing",
                serde_json::json!({"value": 1}),
            )),
        }];
        let error = PromptError::UnknownToolCall {
            tool_name: "missing".to_string(),
            available_tools: vec!["add".to_string()],
            allowed_tools: Vec::new(),
            chat_history: Box::new(history),
        };
        assert!(validate_unknown_tool_failure(&error, "missing", &[]).is_ok());
        assert!(validate_unknown_tool_failure(&error, "other", &[]).is_err());
    }
}
