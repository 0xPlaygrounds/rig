//! Demonstrates registering boxed tools on an agent.
//! Requires `OPENAI_API_KEY`.
//! Run it to see the model use arithmetic tools instead of answering from scratch.

use anyhow::Result;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{Prompt, ToolDefinition};
use rig::providers::openai;
use rig::tool::{Tool, ToolDyn};
use serde::Deserialize;
use serde::Serialize;
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

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "add".to_string(),
            description: "Add x and y together".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "x": { "type": "number", "description": "The first number to add" },
                    "y": { "type": "number", "description": "The second number to add" }
                },
                "required": ["x", "y"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
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

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "subtract".to_string(),
            description: "Subtract y from x (i.e.: x - y)".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "x": { "type": "number", "description": "The number to subtract from" },
                    "y": { "type": "number", "description": "The number to subtract" }
                },
                "required": ["x", "y"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(args.x - args.y)
    }
}

fn boxed_tools() -> Vec<Box<dyn ToolDyn>> {
    vec![Box::new(Add), Box::new(Subtract)]
}

#[tokio::main]
async fn main() -> Result<()> {
    let agent = openai::Client::from_env()
        .agent(openai::GPT_4O)
        .preamble(
            "You are a calculator here to help the user perform arithmetic operations. \
             You must use the provided tools before answering.",
        )
        .tools(boxed_tools())
        .max_tokens(1024)
        .build();

    let response = agent.prompt("Calculate 2 - 5.").await?;
    println!("{response}");

    Ok(())
}
