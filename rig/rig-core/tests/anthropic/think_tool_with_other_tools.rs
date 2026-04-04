//! Migrated from `examples/anthropic_think_tool_with_other_tools.rs`.

use rig::client::CompletionClient;
use rig::completion::{Prompt, ToolDefinition};
use rig::providers::anthropic;
use rig::tool::Tool;
use rig::tools::ThinkTool;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::support::{assert_contains_any_case_insensitive, assert_mentions_expected_number};

#[derive(Deserialize)]
struct CalculatorArgs {
    x: f64,
    y: f64,
}

#[derive(Debug, thiserror::Error)]
#[error("calculator error")]
struct CalculatorError;

#[derive(Deserialize, Serialize)]
struct Calculator;

impl Tool for Calculator {
    const NAME: &'static str = "calculator";
    type Error = CalculatorError;
    type Args = CalculatorArgs;
    type Output = f64;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "calculator".to_string(),
            description: "Add x and y together.".to_string(),
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

#[derive(Deserialize)]
struct DatabaseLookupArgs {
    query: String,
}

#[derive(Debug, thiserror::Error)]
#[error("database lookup error")]
struct DatabaseLookupError;

#[derive(Deserialize, Serialize)]
struct DatabaseLookup;

impl Tool for DatabaseLookup {
    const NAME: &'static str = "database_lookup";
    type Error = DatabaseLookupError;
    type Args = DatabaseLookupArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "database_lookup".to_string(),
            description: "Look up shipping rates, policies, or inventory.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": { "query": { "type": "string" } },
                "required": ["query"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let value = match args.query.as_str() {
            "shipping_rates" => {
                "Standard shipping: $5.99, Express shipping: $15.99, Next-day shipping: $29.99"
            }
            "product_inventory" => {
                "Product A: 15 units, Product B: 8 units, Product C: Out of stock"
            }
            _ => "Customers can return items within 30 days with a receipt for a full refund.",
        };
        Ok(value.to_string())
    }
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn think_tool_with_other_tools() {
    let api_key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set");
    let client = anthropic::Client::builder()
        .api_key(&api_key)
        .anthropic_beta("token-efficient-tools-2025-02-19")
        .build()
        .expect("client should build");

    let agent = client
        .agent(anthropic::completion::CLAUDE_3_7_SONNET)
        .name("Customer Service Agent")
        .preamble(
            "You are a customer service agent. Use the think tool to analyze the situation, \
             then use the database lookup and calculator tools when needed.",
        )
        .tool(ThinkTool)
        .tool(Calculator)
        .tool(DatabaseLookup)
        .build();

    let response = agent
        .prompt(
            "I ordered 3 units of Product A at $25 each and 2 units of Product B at $40 each. \
             I want to return 1 unit of Product A and exchange the 2 units of Product B for Product C. \
             How much will I get refunded, and is Product C in stock? Also include express shipping.",
        )
        .max_turns(10)
        .await
        .expect("prompt should succeed");

    assert_mentions_expected_number(&response, 25);
    assert_contains_any_case_insensitive(&response, &["out of stock", "express shipping"]);
}
