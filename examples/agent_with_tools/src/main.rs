//! Demonstrates registering runtime-defined tools and a provider-hosted web
//! search on one agent, then reading what kind of tool each catalog entry is.
//! Requires `OPENAI_API_KEY`.
//! Run it to see the model search the web, then use the arithmetic tools
//! instead of answering from scratch.

use anyhow::Result;
use rig::agent::tool::CatalogEntry;
use rig::completion::ProviderToolDefinition;
use rig::message::{AssistantContent, EmptyToolName, Message, ToolName};
use rig::prelude::*;
use rig::providers::openai::{self, OpenAI};
use rig::tool::{DynamicTool, ToolOutput};
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize)]
struct OperationArgs {
    x: i32,
    y: i32,
}

fn runtime_tools() -> Result<Vec<DynamicTool>, EmptyToolName> {
    let parameters = json!({
        "type": "object",
        "properties": {
            "x": { "type": "integer" },
            "y": { "type": "integer" }
        },
        "required": ["x", "y"]
    });
    Ok(vec![
        DynamicTool::new(
            ToolName::new("add")?,
            "Add x and y",
            parameters.clone(),
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
            ToolName::new("subtract")?,
            "Subtract y from x",
            parameters,
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
    let agent = AgentBuilder::new(OpenAI::from_env()?.completion(openai::GPT_5_4_MINI))
        .preamble(
            "You are a calculator here to help the user perform arithmetic operations. \
             Search the web for facts you do not know, and use the arithmetic tools for \
             every calculation.",
        )
        .dynamic_tools(runtime_tools()?)
        .provider_tool(ProviderToolDefinition::new("web_search"))
        .max_tokens(2048)
        .default_max_turns(4)
        .build();

    for entry in agent.tool_server_handle().snapshot().entries() {
        match entry {
            CatalogEntry::Registered(tool) => {
                println!("{}: {:?}", tool.definition().name, tool.origin());
            }
            CatalogEntry::ProviderHosted(tool) => println!("{}: run by the provider", tool.kind),
        }
    }

    let response = agent
        .prompt("In which year was Rust 1.0 released? Subtract that year from 2030.")
        .await?;
    for message in &response.messages {
        let Message::Assistant(turn) = message else {
            continue;
        };
        for block in &turn.content {
            match block {
                AssistantContent::ToolCall(call) => println!("rig ran {}", call.function.name),
                AssistantContent::Opaque(item) => {
                    println!("the provider ran {}", item.kind().unwrap_or("an item"));
                }
                _ => {}
            }
        }
    }
    println!("{}", response.output());

    Ok(())
}
