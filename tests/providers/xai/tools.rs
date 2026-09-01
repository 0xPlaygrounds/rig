//! xAI tools smoke test.

use rig::completion::Prompt;
use rig::prelude::*;
use rig::providers::xai;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};

use super::support::with_xai_cassette;
use crate::support::{TOOLS_PREAMBLE, TOOLS_PROMPT, assert_mentions_expected_number};

#[derive(Deserialize)]
struct OperationArgs {
    x: f64,
    y: f64,
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
    type Output = f64;

    fn description(&self) -> String {
        "Add x and y together".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "x": { "type": "number", "description": "The first number to add" },
                "y": { "type": "number", "description": "The second number to add" }
            },
            "required": ["x", "y"]
        })
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Ok(args.x + args.y)
    }
}

#[derive(Deserialize, Serialize)]
struct Subtract;

impl Tool for Subtract {
    const NAME: &'static str = "subtract";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = f64;

    fn description(&self) -> String {
        "Subtract y from x (that is, x - y)".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "x": { "type": "number", "description": "The number to subtract from" },
                "y": { "type": "number", "description": "The number to subtract" }
            },
            "required": ["x", "y"]
        })
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Ok(args.x - args.y)
    }
}

#[tokio::test]
async fn tools_smoke() {
    with_xai_cassette("tools/tools_smoke", |client| async move {
        let agent = client
            .agent(xai::completion::GROK_3_MINI)
            .preamble(TOOLS_PREAMBLE)
            .tool(Adder)
            .tool(Subtract)
            .default_max_turns(2)
            .build();

        let response = agent
            .prompt(TOOLS_PROMPT)
            .await
            .expect("tool prompt should succeed");

        assert_mentions_expected_number(&response, -3);
    })
    .await;
}
