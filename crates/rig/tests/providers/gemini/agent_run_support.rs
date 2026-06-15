//! Shared fixtures for the `agent_run` cassette suites: the arithmetic tools
//! advertised to Gemini and helpers for hand-driving the sans-IO
//! [`AgentRun`](rig::agent::run::AgentRun) state machine.
#![allow(dead_code)]

use std::collections::BTreeSet;

use rig::agent::CompletionCall;
use rig::agent::run::{ModelTurn, PendingToolCall};
use rig::completion::{Completion, ToolDefinition, Usage};
use rig::message::{AssistantContent, Message, ToolResultContent, UserContent};
use rig::providers::gemini;
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::json;

pub(crate) type GeminiAgent = rig::agent::Agent<gemini::completion::CompletionModel>;

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

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        operation_definition(Self::NAME, "Add x and y together")
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(args.x + args.y)
    }
}

pub(crate) struct Subtract;

impl Tool for Subtract {
    const NAME: &'static str = "subtract";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i64;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        operation_definition(Self::NAME, "Subtract y from x (i.e. x - y)")
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
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

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        operation_definition(Self::NAME, "Add x and y together (alias of add)")
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(args.x + args.y)
    }
}

pub(crate) fn tool_names(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|name| (*name).to_string()).collect()
}

/// Execute one arithmetic tool call by name, the way a driver would.
pub(crate) fn execute_arithmetic(name: &str, arguments: &serde_json::Value) -> String {
    let operand = |key: &str| {
        arguments
            .get(key)
            .and_then(serde_json::Value::as_f64)
            .unwrap_or_else(|| panic!("tool args should carry `{key}`: {arguments}")) as i64
    };
    let (x, y) = (operand("x"), operand("y"));
    let result = match name {
        "add" | "sum" => x + y,
        "subtract" => x - y,
        other => panic!("unexpected tool `{other}`"),
    };
    result.to_string()
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
            let content = ToolResultContent::from_tool_output(output);
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

/// One hand-driven, non-streamed model call: send the step's prompt and
/// history through the agent's completion builder and shape the response into
/// a [`ModelTurn`] with the given advertised tool names.
pub(crate) async fn call_model(
    agent: &GeminiAgent,
    prompt: Message,
    history: Vec<Message>,
    executable: &BTreeSet<String>,
    allowed: &BTreeSet<String>,
) -> ModelTurn {
    let response = agent
        .completion(prompt, history)
        .await
        .expect("completion request should build")
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
            _ => None,
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
