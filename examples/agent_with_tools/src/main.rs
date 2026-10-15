//! Demonstrates registering runtime-defined tools on an agent.
//! Requires `OPENAI_API_KEY`.
//! Run it to see the model use arithmetic tools instead of answering from scratch.

use anyhow::Result;
use rig::completion::ToolDefinition;
use rig::message::{EmptyToolName, ToolName};
use rig::prelude::*;
use rig::providers::openai::{self, OpenAI};
use rig::tool::{DynamicTool, Schema, ToolOutput};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize, JsonSchema)]
struct OperationArgs {
    x: i32,
    y: i32,
}

fn runtime_tools() -> Result<Vec<DynamicTool>, EmptyToolName> {
    let parameters = Schema::arguments::<OperationArgs>();
    Ok(vec![
        DynamicTool::new(
            ToolDefinition::new(ToolName::new("add")?, "Add x and y", parameters.clone()),
            |args| {
                Box::pin(async move {
                    let args: OperationArgs = serde_json::from_value(args).map_err(|error| {
                        rig::tool::ToolExecutionError::invalid_args(error.to_string())
                            .with_source(error)
                    })?;
                    Ok(ToolOutput::json(json!(args.x + args.y)))
                })
            },
        ),
        DynamicTool::new(
            ToolDefinition::new(ToolName::new("subtract")?, "Subtract y from x", parameters),
            |args| {
                Box::pin(async move {
                    let args: OperationArgs = serde_json::from_value(args).map_err(|error| {
                        rig::tool::ToolExecutionError::invalid_args(error.to_string())
                            .with_source(error)
                    })?;
                    Ok(ToolOutput::json(json!(args.x - args.y)))
                })
            },
        ),
    ])
}

#[tokio::main]
async fn main() -> Result<()> {
    let agent = AgentBuilder::new(OpenAI::from_env()?.completion(openai::GPT_4O))
        .preamble(
            "You are a calculator here to help the user perform arithmetic operations. \
             You must use the provided tools before answering.",
        )
        .dynamic_tools(runtime_tools()?)
        .max_tokens(1024)
        .default_max_turns(2)
        .build();

    let response = agent.prompt("Calculate 2 - 5.").await?.output();
    println!("{response}");

    Ok(())
}
