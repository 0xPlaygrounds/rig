//! Migrated from `examples/multi_turn_streaming.rs`.

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::ToolDefinition;
use rig::providers::anthropic;
use rig::streaming::StreamingPrompt;
use rig::tool::Tool;
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};

use crate::support::{assert_mentions_expected_number, collect_stream_final_response};

#[derive(Deserialize, JsonSchema)]
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
            parameters: serde_json::to_value(schema_for!(OperationArgs))
                .expect("schema should serialize"),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(args.x + args.y)
    }
}

#[derive(Deserialize, Serialize)]
struct Divide;

impl Tool for Divide {
    const NAME: &'static str = "divide";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "divide".to_string(),
            description: "Compute the quotient of x and y.".to_string(),
            parameters: serde_json::to_value(schema_for!(OperationArgs))
                .expect("schema should serialize"),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(args.x / args.y)
    }
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn multi_turn_streaming_tools() {
    let client = anthropic::Client::from_env();
    let agent = client
        .agent(anthropic::completion::CLAUDE_3_5_SONNET)
        .preamble("You must use tools for arithmetic.")
        .tool(Add)
        .tool(Divide)
        .build();

    let mut stream = agent
        .stream_prompt("Calculate (2 + 2) / 2 and describe the result.")
        .multi_turn(10)
        .await;
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("multi-turn stream should succeed");

    assert_mentions_expected_number(&response, 2);
}
