//! Migrated from `examples/agent_with_default_max_turns.rs`.

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{Prompt, ToolDefinition};
use rig::providers::anthropic;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::support::assert_mentions_expected_number;

#[derive(Deserialize)]
struct OperationArgs {
    x: i32,
    y: i32,
}

#[derive(Debug, thiserror::Error)]
#[error("Math error")]
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
                "properties": { "x": { "type": "number" }, "y": { "type": "number" } },
                "required": ["x", "y"]
            }),
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
            parameters: json!({
                "type": "object",
                "properties": { "x": { "type": "number" }, "y": { "type": "number" } },
                "required": ["x", "y"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(args.x / args.y)
    }
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn default_max_turns_allows_multi_step_tool_use() {
    let client = anthropic::Client::from_env();
    let agent = client
        .agent(anthropic::completion::CLAUDE_3_5_SONNET)
        .preamble(
            "You are an assistant that must use the available tools for arithmetic. \
             Never compute the result yourself.",
        )
        .tool(Add)
        .tool(Divide)
        .default_max_turns(10)
        .build();

    let response = agent
        .prompt("Calculate (3 + 5) / 4 and describe the result.")
        .await
        .expect("prompt should succeed");

    assert_mentions_expected_number(&response, 2);
}
