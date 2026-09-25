use anyhow::Result;
use rig_agent::prelude::*;
use rig_agent::tool::ToolContext;
use rig_vertexai::{
    VertexAi,
    completion::{GEMINI_2_5_FLASH_LITE, GenerateContent},
};
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use serde_json::json;

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

    fn description(&self) -> String {
        "Add x and y together".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!(schema_for!(OperationArgs))
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        println!("[tool-call] Adding {} and {}", args.x, args.y);
        let result = args.x + args.y;
        Ok(result)
    }
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt().with_target(false).init();

    // Create the Vertex AI model using implicit credentials
    let model = GenerateContent::new(GEMINI_2_5_FLASH_LITE).on(VertexAi::from_env()?);

    // Create agent with a calculator tool
    let calculator_agent = AgentBuilder::new(model)
        .tool(Adder)
        .max_tokens(1024)
        .build();

    // Prompt the agent and print the response
    println!("Calculate 15 + 27");
    let answer = calculator_agent.prompt("Calculate 15 + 27").await?.output;
    println!("Vertex AI Calculator Agent: {answer}");

    Ok(())
}
