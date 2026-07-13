//! Preserves the live default-max-turns example as provider-local regression coverage.

use anyhow::Result;
use rig::client::CompletionClient;
use rig::completion::Prompt;
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
enum MathError {
    #[error("division by zero")]
    DivisionByZero,
}

#[derive(Deserialize, Serialize)]
struct Add;

impl Tool for Add {
    const NAME: &'static str = "add";
    type Args = OperationArgs;
    type Output = i32;

    fn description(&self) -> String {
        "Add x and y together".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": { "x": { "type": "number" }, "y": { "type": "number" } },
            "required": ["x", "y"]
        })
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, rig::tool::ToolExecutionError> {
        Ok(args.x + args.y)
    }
}

#[derive(Deserialize, Serialize)]
struct Divide;

impl Tool for Divide {
    const NAME: &'static str = "divide";
    type Args = OperationArgs;
    type Output = i32;

    fn description(&self) -> String {
        "Compute the quotient of x and y.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": { "x": { "type": "number" }, "y": { "type": "number" } },
            "required": ["x", "y"]
        })
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, rig::tool::ToolExecutionError> {
        if args.y == 0 {
            return Err(rig::tool::ToolExecutionError::from_error(
                MathError::DivisionByZero,
            ));
        }

        Ok(args.x / args.y)
    }
}

#[tokio::test]
async fn default_max_turns_allows_multi_step_tool_use() -> Result<()> {
    super::super::support::with_anthropic_cassette_result(
        "default_max_turns/default_max_turns_allows_multi_step_tool_use",
        |client| async move {
            let agent = client
                .agent(anthropic::completion::CLAUDE_SONNET_4_6)
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
                .await?;

            assert_mentions_expected_number(&response, 2);

            Ok(())
        },
    )
    .await
}
