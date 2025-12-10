use rig::prelude::*;
use rig::providers::anthropic::completion::CLAUDE_3_5_SONNET;
use rig::providers::openai::GPT_4O;
use rig::{
    completion::{Prompt, ToolDefinition},
    providers,
    tool::Tool,
};

use serde::{Deserialize, Serialize};

use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(false)
        .init();

    // Initialize the Mira client using environment variables
    let client = providers::mira::Client::from_env();

    // Test API connection first by listing models
    println!("\nTesting API connection by listing models...");
    match client.list_models().await {
        Ok(models) => {
            println!("Successfully connected to Mira API!");
            println!("Available models:");
            for model in models {
                println!("- {model}");
            }
            println!("\nProceeding with chat completion...\n");
        }
        Err(e) => {
            return Err(anyhow::anyhow!(
                "Failed to connect to Mira API: {}. Please verify your API key and network connection.",
                e
            ));
        }
    }

    // Create a basic agent for general conversation
    let agent = client
        .agent(GPT_4O)
        .preamble("You are a helpful AI assistant.")
        .temperature(0.7)
        .build();

    // Send a message and get response
    let response = agent.prompt("What are the 7 wonders of the world?").await?;
    println!("Basic Agent Response: {response}");

    // Create a calculator agent with tools
    let calculator_agent = client
        .agent(CLAUDE_3_5_SONNET)
        .preamble("You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.")
        .max_tokens(1024)
        .tool(Adder)
        .tool(Subtract)
        .build();

    // Test the calculator agent
    println!("\nTesting Calculator Agent:");
    println!(
        "Mira Calculator Agent: {}",
        calculator_agent.prompt("Calculate 15 - 7").await?
    );
    Ok(())
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
                }
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
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

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        serde_json::from_value(json!({
            "name": "subtract",
            "description": "Subtract y from x (i.e.: x - y)",
            "parameters": {
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
                }
            }
        }))
        .expect("Tool Definition")
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let result = args.x - args.y;
        Ok(result)
    }
}
