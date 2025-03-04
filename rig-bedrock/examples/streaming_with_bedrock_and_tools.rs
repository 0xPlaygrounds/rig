use std::{
    error::Error,
    fmt::{Display, Formatter},
};

use rig::{completion::ToolDefinition, tool::Tool};
use rig_bedrock::{client::ClientBuilder, completion::AMAZON_NOVA_LITE_V1};
use serde::{Deserialize, Serialize};
use serde_json::json;

use rig::streaming::{stream_to_stdout, StreamingPrompt};

#[derive(Deserialize)]
struct OperationArgs {
    x: i32,
    y: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathError {
    Reason(&'static str),
}

impl Display for MathError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use MathError::*;
        match self {
            Reason(err) => write!(f, "Math error: {}", err),
        }
    }
}

impl Error for MathError {}

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

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt().init();
    // Create agent with a single context prompt and two tools
    let agent = ClientBuilder::new()
        .build()
        .await
        .agent(AMAZON_NOVA_LITE_V1)
        .preamble(
            "You are a calculator here to help the user perform arithmetic
            operations. Use the tools provided to answer the user's question.
            make your answer long, so we can test the streaming functionality,
            like 20 words",
        )
        .max_tokens(1024)
        .tool(Adder)
        .build();

    println!("Calculate 2 + 5");
    let mut stream = agent.stream_prompt("Calculate 2 + 5").await?;
    stream_to_stdout(agent, &mut stream).await?;
    Ok(())
}
