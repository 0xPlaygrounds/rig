//! Demonstrates registering runtime-defined tools on an agent.
//! Requires `OPENAI_API_KEY`.
//! Run it to see the model use arithmetic tools instead of answering from scratch.
use rig::prelude::AgentClientExt;

use anyhow::Result;
use rig::client::ProviderClient;
use rig::completion::Prompt;
use rig::providers::openai;
use rig::tool::{DynamicTool, ToolOutput};
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize)]
struct OperationArgs {
    x: i32,
    y: i32,
}

fn runtime_tools() -> Vec<DynamicTool> {
    let parameters = json!({
        "type": "object",
        "properties": {
            "x": { "type": "integer" },
            "y": { "type": "integer" }
        },
        "required": ["x", "y"]
    });
    vec![
        DynamicTool::new(
            "add",
            "Add x and y",
            parameters.clone(),
            |_context, args| {
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
            "subtract",
            "Subtract y from x",
            parameters,
            |_context, args| {
                Box::pin(async move {
                    let args: OperationArgs = serde_json::from_value(args).map_err(|error| {
                        rig::tool::ToolExecutionError::invalid_args(error.to_string())
                            .with_source(error)
                    })?;
                    Ok(ToolOutput::json(json!(args.x - args.y)))
                })
            },
        ),
    ]
}

#[tokio::main]
async fn main() -> Result<()> {
    let agent = openai::Client::from_env()?
        .agent(openai::GPT_4O)
        .preamble(
            "You are a calculator here to help the user perform arithmetic operations. \
             You must use the provided tools before answering.",
        )
        .dynamic_tools(runtime_tools())
        .max_tokens(1024)
        .default_max_turns(2)
        .build();

    let response = agent.prompt("Calculate 2 - 5.").await?;
    println!("{response}");

    Ok(())
}
