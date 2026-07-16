//! Shared fixtures for the `agent_run` cassette suites: the arithmetic tools
//! advertised to Gemini and helpers for hand-driving the sans-IO
//! [`AgentRun`](rig::agent::run::AgentRun) state machine.
#![allow(dead_code)]

use std::collections::BTreeSet;

use rig::agent::CompletionCall;
use rig::agent::run::{ModelTurn, PendingToolCall};
use rig::completion::{CompletionModel, CompletionRequestBuilder, ToolDefinition, Usage};
use rig::message::{AssistantContent, Message, ToolChoice, ToolResultContent, UserContent};
use rig::providers::gemini;
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::json;

pub(crate) struct GeminiAgent {
    model: gemini::completion::CompletionModel,
    preamble: String,
    tools: Vec<ToolDefinition>,
    tool_choice: Option<ToolChoice>,
}

impl GeminiAgent {
    pub(crate) fn new(
        model: gemini::completion::CompletionModel,
        preamble: impl Into<String>,
        tool_names: &[&str],
        tool_choice: Option<ToolChoice>,
    ) -> Self {
        Self {
            model,
            preamble: preamble.into(),
            tools: tool_names
                .iter()
                .map(|name| match *name {
                    "add" => operation_definition("add", "Add x and y together"),
                    "sum" => operation_definition("sum", "Add x and y together (alias of add)"),
                    "subtract" => {
                        operation_definition("subtract", "Subtract y from x (i.e. x - y)")
                    }
                    other => panic!("unsupported raw harness tool `{other}`"),
                })
                .collect(),
            tool_choice,
        }
    }

    pub(crate) fn request(
        &self,
        prompt: Message,
        history: Vec<Message>,
    ) -> CompletionRequestBuilder<gemini::completion::CompletionModel> {
        let mut request = self
            .model
            .completion_request(prompt)
            .messages(history)
            .preamble(self.preamble.clone())
            .tools(self.tools.clone());
        if let Some(tool_choice) = &self.tool_choice {
            request = request.tool_choice(tool_choice.clone());
        }
        request
    }
}

pub(crate) const FORCE_TOOLS_PREAMBLE: &str = "You are a calculator assistant. You MUST use the provided tools for every arithmetic operation instead of computing results yourself. Once you have all the tool results you need, reply with the final numeric answer in plain text.";

#[derive(Deserialize)]
pub(crate) struct OperationArgs {
    x: i64,
    y: i64,
}

#[derive(Debug, thiserror::Error)]
#[error("math error")]
pub(crate) struct MathError;

fn operation_definition(name: &str, description: &str) -> ToolDefinition {
    ToolDefinition {
        name: name.to_string(),
        description: description.to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "x": { "type": "number", "description": "The first operand" },
                "y": { "type": "number", "description": "The second operand" }
            },
            "required": ["x", "y"]
        }),
    }
}

pub(crate) struct Add;

impl Tool for Add {
    const NAME: &'static str = "add";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i64;

    fn description(&self) -> String {
        "Add x and y together".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        operation_definition(Self::NAME, "Add x and y together").parameters
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Ok(args.x + args.y)
    }
}

pub(crate) struct Subtract;

impl Tool for Subtract {
    const NAME: &'static str = "subtract";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i64;

    fn description(&self) -> String {
        "Subtract y from x (i.e. x - y)".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        operation_definition(Self::NAME, "Subtract y from x (i.e. x - y)").parameters
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Ok(args.x - args.y)
    }
}

/// Alias of `add` registered alongside it so invalid tool-call repairs can
/// rename `add` to `sum` while the recorded wire history stays coherent.
pub(crate) struct Sum;

impl Tool for Sum {
    const NAME: &'static str = "sum";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i64;

    fn description(&self) -> String {
        "Add x and y together (alias of add)".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        operation_definition(Self::NAME, "Add x and y together (alias of add)").parameters
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Ok(args.x + args.y)
    }
}

pub(crate) fn tool_names(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|name| (*name).to_string()).collect()
}

/// Execute one arithmetic tool call by name, the way a driver would.
pub(crate) fn execute_arithmetic(name: &str, arguments: &serde_json::Value) -> i64 {
    let operand = |key: &str| {
        arguments
            .get(key)
            .and_then(serde_json::Value::as_f64)
            .unwrap_or_else(|| panic!("tool args should carry `{key}`: {arguments}")) as i64
    };
    let (x, y) = (operand("x"), operand("y"));
    match name {
        "add" | "sum" => x + y,
        "subtract" => x - y,
        other => panic!("unexpected tool `{other}`"),
    }
}

/// Answer every pending call: preresolved results pass through unexecuted,
/// the rest run the arithmetic tools.
pub(crate) fn execute_pending_calls(calls: &[PendingToolCall]) -> Vec<UserContent> {
    calls
        .iter()
        .map(|call| {
            if let Some(result) = call.preresolved_result.clone() {
                return result;
            }
            let output = execute_arithmetic(
                &call.tool_call.function.name,
                &call.tool_call.function.arguments,
            );
            let content = rig::OneOrMany::one(ToolResultContent::json(serde_json::json!(output)));
            match call.tool_call.call_id.clone() {
                Some(call_id) => UserContent::tool_result_with_call_id(
                    call.tool_call.id.clone(),
                    call_id,
                    content,
                ),
                None => UserContent::tool_result(call.tool_call.id.clone(), content),
            }
        })
        .collect()
}

/// One hand-driven, non-streamed model call through an explicit raw model
/// harness. This exercises the sans-IO `AgentRun` protocol without pretending
/// to execute a configured `Agent`, whose only execution path is `AgentRunner`.
pub(crate) async fn call_model(
    agent: &GeminiAgent,
    prompt: Message,
    history: Vec<Message>,
    executable: &BTreeSet<String>,
    allowed: &BTreeSet<String>,
) -> ModelTurn {
    let response = agent
        .request(prompt, history)
        .send()
        .await
        .expect("gemini completion should succeed");
    ModelTurn::new(
        response.message_id.clone(),
        response.choice.clone(),
        response.usage,
        executable.clone(),
        allowed.clone(),
    )
}

pub(crate) fn assistant_tool_call_names(message: &Message) -> Vec<String> {
    match message {
        Message::Assistant { content, .. } => content
            .iter()
            .filter_map(|item| match item {
                AssistantContent::ToolCall(tool_call) => Some(tool_call.function.name.clone()),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

pub(crate) fn history_has_assistant_tool_call(history: &[Message], tool_name: &str) -> bool {
    history.iter().any(|message| {
        assistant_tool_call_names(message)
            .iter()
            .any(|n| n == tool_name)
    })
}

/// Texts of every tool result carried by a user message.
pub(crate) fn tool_result_texts(message: &Message) -> Vec<String> {
    let Message::User { content } = message else {
        return Vec::new();
    };
    content
        .iter()
        .flat_map(user_content_tool_result_texts)
        .collect()
}

pub(crate) fn user_content_tool_result_texts(content: &UserContent) -> Vec<String> {
    let UserContent::ToolResult(tool_result) = content else {
        return Vec::new();
    };
    tool_result
        .content
        .iter()
        .filter_map(|item| match item {
            ToolResultContent::Text(text) => Some(text.text.clone()),
            ToolResultContent::Json { value } => Some(value.to_string()),
            ToolResultContent::Image(_) => None,
        })
        .collect()
}

pub(crate) fn is_tool_result_user_message(message: &Message) -> bool {
    matches!(
        message,
        Message::User { content }
            if content.iter().any(|item| matches!(item, UserContent::ToolResult(_)))
    )
}

pub(crate) fn sum_completion_call_usage(calls: &[CompletionCall]) -> Usage {
    let mut total = Usage::new();
    for call in calls {
        total += call.usage;
    }
    total
}

/// Assert each assistant message in `messages` records content in canonical
/// replay order: reasoning blocks, then text, then tool calls.
pub(crate) fn assert_canonical_assistant_order(messages: &[Message]) {
    for message in messages {
        let Message::Assistant { content, .. } = message else {
            continue;
        };
        let kind_rank = |item: &AssistantContent| match item {
            AssistantContent::Reasoning(_) => 0_u8,
            AssistantContent::ToolCall(_) => 2,
            _ => 1,
        };
        let ranks: Vec<u8> = content.iter().map(kind_rank).collect();
        let mut sorted = ranks.clone();
        sorted.sort_unstable();
        assert_eq!(
            ranks, sorted,
            "assistant content should be in canonical reasoning → text → tool-call order: {message:?}"
        );
    }
}
