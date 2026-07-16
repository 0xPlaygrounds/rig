//! Demonstrates the lowest-level supported agent execution surface:
//! [`rig::agent::AgentRunner`]. The runner owns request preparation, hooks,
//! tool execution, and lifecycle state.
//! Requires `OPENAI_API_KEY`.

use anyhow::Result;
use rig::agent::{
    AgentHook, AgentRun, AgentRunnerOutcome, HookContext, ToolCall as ToolCallEvent, ToolCallAction,
};
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::CompletionModel;
use rig::providers::openai;
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize)]
struct OperationArgs {
    x: i32,
    y: i32,
}

#[derive(Debug, thiserror::Error)]
#[error("math error")]
struct MathError;

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
        .agent(openai::GPT_4O)
        .preamble("Always use the provided tools to compute results.")
        .tool(Add)
        .build();
    let prompt = "What is 2 + 5?";
    let mut checkpoint: Option<AgentRun> = None;
    loop {
        let mut runner = agent.runner(prompt).max_turns(2).add_hook(ToolLogger);
        if let Some(run) = checkpoint.take() {
            runner = runner.resume(run);
        }

        match runner.run_until_interruption().await? {
            AgentRunnerOutcome::Interrupted(run) => {
                let json = serde_json::to_string(&run)?;
                println!("paused with pending tools: {json}");
                checkpoint = Some(serde_json::from_str(&json)?);
            }
            AgentRunnerOutcome::Completed(response) => {
                println!("{}", response.output);
                break;
            }
            _ => anyhow::bail!("unsupported runner outcome"),
        }
    }
    Ok(())
}
