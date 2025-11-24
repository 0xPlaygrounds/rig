use anyhow::Result;
use rig::prelude::*;
use rig::{
    completion::{Prompt, ToolDefinition},
    tool::Tool,
};
use rig_vertexai::{Client, completion::GEMINI_2_5_FLASH_LITE};
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};

#[derive(Deserialize, JsonSchema)]
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
            parameters: serde_json::to_value(schema_for!(OperationArgs))
                .expect("converting JSON schema to JSON value should never fail"),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        println!("[tool-call] Adding {} and {}", args.x, args.y);
        let result = args.x + args.y;
        Ok(result)
    }
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt().with_target(false).init();

    // Create Vertex AI client using implicit credentials
    let client = Client::from_env();

    // Create agent with a calculator tool
    let calculator_agent = client
        .agent(GEMINI_2_5_FLASH_LITE)
        .tool(Adder)
        .max_tokens(1024)
        .build();

    // Prompt the agent and print the response
    println!("Calculate 15 + 27");
    let answer = calculator_agent.prompt("Calculate 15 + 27").await?;
    println!("Vertex AI Calculator Agent: {answer}");

    Ok(())
}
