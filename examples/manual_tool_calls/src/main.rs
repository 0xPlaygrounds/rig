//! Demonstrates observing tool calls while `AgentRunner` retains ownership of
//! request preparation, hook dispatch, tool execution, and lifecycle state.
//! Requires `OPENAI_API_KEY`.

use anyhow::Result;
use rig::agent::{AgentHook, HookContext, ToolCall as ToolCallEvent, ToolCallAction};
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::CompletionModel;
use rig::providers::openai;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Deserialize)]
struct OperationArgs {
    x: i32,
    y: i32,
}

#[derive(Debug, thiserror::Error)]
#[error("math error")]
struct MathError;

#[derive(Deserialize, Serialize)]
struct Add;

impl Tool for Add {
    const NAME: &'static str = "add";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    fn description(&self) -> String {
        "Add x and y together".to_string()
    }
    fn parameters(&self) -> serde_json::Value {
        json!({"type":"object","properties":{"x":{"type":"number"},"y":{"type":"number"}},"required":["x","y"]})
    }
    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Ok(args.x + args.y)
    }
}

#[derive(Deserialize, Serialize)]
struct Subtract;

impl Tool for Subtract {
    const NAME: &'static str = "subtract";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    fn description(&self) -> String {
        "Subtract y from x".to_string()
    }
    fn parameters(&self) -> serde_json::Value {
        json!({"type":"object","properties":{"x":{"type":"number"},"y":{"type":"number"}},"required":["x","y"]})
    }
    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Ok(args.x - args.y)
    }
}

struct ToolLogger;

impl<M: CompletionModel> AgentHook<M> for ToolLogger {
    async fn on_tool_call(&self, _ctx: &HookContext, event: ToolCallEvent<'_>) -> ToolCallAction {
        println!("tool call: {}({})", event.tool_name, event.args);
        ToolCallAction::run()
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let agent = openai::Client::from_env()?
        .agent(openai::GPT_4O_MINI)
        .preamble(
            "Use the provided tools for every arithmetic operation, then give a short answer.",
        )
        .tool(Add)
        .tool(Subtract)
        .build();

    let response = agent
        .runner("Calculate (20 - 5) + (8 - 3).")
        .max_turns(8)
        .add_hook(ToolLogger)
        .run()
        .await?;
    println!("{}", response.output);
    Ok(())
}
