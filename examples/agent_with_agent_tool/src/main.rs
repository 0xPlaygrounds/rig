//! An agent that uses another agent as a tool.
//!
//! `Agent` is `Clone`, so a sub-agent can be exposed as a tool with a
//! [`PortableDynamicTool`] whose callback closes over the inner agent and
//! prompts it. The outer agent simply registers that record with
//! `.dynamic_tool(...)`.
//!
//! Both agents are built from the same plain-data provider config
//! (`openai::functions::Config`, which names the model) wrapped in
//! `ProviderConfig` — there is no client to share.
//!
//! Requires `OPENAI_API_KEY`.

use anyhow::Result;
use rig::prelude::*;
use rig::tool::{PortableDynamicTool, ToolExecutionError, ToolOutput};
use rig::{providers, tool::Tool};
use serde::{Deserialize, Serialize};
use serde_json::json;

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

    fn description(&self) -> String {
        "Add x and y together".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
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
        })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        println!("[tool-call] Adding {} and {}", args.x, args.y);
        let result = args.x + args.y;
        Ok(result)
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
        "Subtract y from x (i.e.: x - y)".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "x": {
                    "type": "number",
                    "description": "The number to subtract from"
                },
                "y": {
                    "type": "number",
                    "description": "The number to subtract"
                }
            },
            "required": ["x", "y"],
        })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        println!("[tool-call] Subtracting {} from {}", args.y, args.x);
        let result = args.x - args.y;
        Ok(result)
    }
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(false)
        .init();

    // The provider is plain data: one config, cloned into both agents.
    let cfg = providers::openai::functions::Config::from_env(providers::openai::GPT_4O)?;

    // Create agent with a single context prompt and two tools
    let calculator_agent = AgentBuilder::new(ProviderConfig::OpenAi(cfg.clone()))
        .preamble("You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.")
        .max_tokens(1024)
        .default_max_turns(2)
        .tool(Adder)
        .tool(Subtract)
        .build();

    // Expose the calculator agent as a tool: the dynamic tool's callback
    // closes over a clone of the inner agent and forwards the prompt to it.
    let inner = calculator_agent.clone();
    let calculator_tool = PortableDynamicTool::new(
        "calculator",
        "Delegate arithmetic questions to the calculator agent.",
        json!({
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The arithmetic question for the calculator agent"
                }
            },
            "required": ["prompt"]
        }),
        move |args| {
            let inner = inner.clone();
            Box::pin(async move {
                let prompt = args
                    .get("prompt")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                let reply = inner
                    .prompt(prompt)
                    .await
                    .map_err(|e| ToolExecutionError::other(e.to_string()))?;
                Ok(ToolOutput::text(reply))
            })
        },
    );

    // Create agent which has the calculator agent as a tool
    let agent_using_agent = AgentBuilder::new(ProviderConfig::OpenAi(cfg))
        .preamble("You are a helpful assistant that can solve problems. Use the tool provided to answer the user's question.")
        .max_tokens(1024)
        .default_max_turns(2)
        .dynamic_tool(calculator_tool)
        .build();

    // Prompt the agent and print the response
    println!("Calculate 2 - 5");

    println!(
        "OpenAI Agent-Using Agent: {}",
        agent_using_agent.prompt("Calculate 2 - 5").await?
    );

    Ok(())
}
