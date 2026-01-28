use anyhow::Result;
use rig::agent::{CancelSignal, PromptHook, ToolCallHookAction};
use rig::completion::{CompletionModel, Prompt, ToolDefinition};
use rig::message::ToolChoice;
use rig::prelude::*;
use rig::{providers, tool::Tool};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Clone)]
struct AssertToolCallIdHook;

impl<M> PromptHook<M> for AssertToolCallIdHook
where
    M: CompletionModel,
{
    async fn on_tool_call(
        &self,
        _tool_name: &str,
        tool_call_id: Option<String>,
        _args: &str,
        _cancel_sig: CancelSignal,
    ) -> ToolCallHookAction {
        assert!(
            tool_call_id.as_deref().is_some_and(|id| !id.is_empty()),
            "expected tool_call_id to be Some(non-empty)"
        );
        ToolCallHookAction::Continue
    }

    async fn on_tool_result(
        &self,
        _tool_name: &str,
        tool_call_id: Option<String>,
        _args: &str,
        _result: &str,
        _cancel_sig: CancelSignal,
    ) {
        assert!(
            tool_call_id.as_deref().is_some_and(|id| !id.is_empty()),
            "expected tool_call_id to be Some(non-empty)"
        );
    }
}

#[derive(Deserialize)]
struct OperationArgs {
    x: i32,
    y: i32,
}

#[derive(Debug, thiserror::Error)]
#[error("Math error")]
struct MathError;

#[derive(Deserialize, Serialize)]
struct Adder;

impl Tool for Adder {
    const NAME: &'static str = "add";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "add".to_string(),
            description: "Add x and y together".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "x": {
                        "type": "number",
                        "description": "The first number to add"
                    },
                    "y": {
                        "type": "number",
                        "description": "The second number to add"
                    }
                },
                "required": ["x", "y"],
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(args.x + args.y)
    }
}

#[tokio::test]
#[ignore = "This test requires OPENAI_API_KEY"]
async fn openai_prompt_tool_call_id() -> Result<(), anyhow::Error> {
    let _ = tracing_subscriber::fmt::try_init();

    let agent = providers::openai::Client::from_env()
        .agent(providers::openai::GPT_4O)
        .preamble("You are a calculator. Always call the `add` tool exactly once before replying.")
        .tool_choice(ToolChoice::Auto)
        .tool(Adder)
        .build();

    let _res = agent
        .prompt("What is 2 + 5? Use the tool.")
        .multi_turn(1)
        .with_hook(AssertToolCallIdHook)
        .await?;

    Ok(())
}

#[tokio::test]
#[ignore = "This test requires ANTHROPIC_API_KEY"]
async fn anthropic_prompt_tool_call_id() -> Result<(), anyhow::Error> {
    let _ = tracing_subscriber::fmt::try_init();

    let agent = providers::anthropic::Client::from_env()
        .agent("claude-haiku-4-5-20251001")
        .preamble("You are a calculator. Always call the `add` tool exactly once before replying.")
        .max_tokens(1024)
        .tool_choice(ToolChoice::Auto)
        .tool(Adder)
        .build();

    let _res = agent
        .prompt("What is 2 + 5? Use the tool.")
        .multi_turn(1)
        .with_hook(AssertToolCallIdHook)
        .await?;

    Ok(())
}

#[tokio::test]
#[ignore = "This test requires GEMINI_API_KEY"]
async fn gemini_prompt_tool_call_id() -> Result<(), anyhow::Error> {
    let _ = tracing_subscriber::fmt::try_init();

    let agent = providers::gemini::Client::from_env()
        .agent(providers::gemini::completion::GEMINI_2_5_FLASH)
        .preamble("You are a calculator. Always call the `add` tool exactly once before replying.")
        .max_tokens(1024)
        .tool_choice(ToolChoice::Auto)
        .tool(Adder)
        .build();

    let _res = agent
        .prompt("What is 2 + 5? Use the tool.")
        .multi_turn(1)
        .with_hook(AssertToolCallIdHook)
        .await?;

    Ok(())
}
