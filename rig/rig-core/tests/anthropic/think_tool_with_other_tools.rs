//! Migrated from `examples/anthropic_think_tool_with_other_tools.rs`.

use std::iter::Peekable;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use rig::client::CompletionClient;
use rig::completion::{Prompt, ToolDefinition};
use rig::message::{AssistantContent, Message};
use rig::providers::anthropic;
use rig::tool::Tool;
use rig::tools::ThinkTool;
use serde::Deserialize;
use serde_json::json;

use crate::support::{assert_contains_any_case_insensitive, assert_mentions_expected_number};

#[derive(Deserialize)]
struct CalculatorArgs {
    expression: String,
}

#[derive(Debug, thiserror::Error)]
#[error("calculator error: {0}")]
struct CalculatorError(String);

struct Calculator {
    call_count: Arc<AtomicUsize>,
}

impl Calculator {
    fn new(call_count: Arc<AtomicUsize>) -> Self {
        Self { call_count }
    }
}

impl Tool for Calculator {
    const NAME: &'static str = "calculator";
    type Error = CalculatorError;
    type Args = CalculatorArgs;
    type Output = f64;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "calculator".to_string(),
            description: "Evaluate arithmetic expressions with +, -, *, /, and parentheses."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "An arithmetic expression such as '25 + (2 * 40)'"
                    }
                },
                "required": ["expression"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        evaluate_expression(&args.expression).map_err(CalculatorError)
    }
}

fn evaluate_expression(expression: &str) -> Result<f64, String> {
    let tokens = tokenize(expression)?;
    let mut iter = tokens.into_iter().peekable();
    let result = parse_expression(&mut iter)?;

    if let Some(token) = iter.next() {
        Err(format!("Unexpected token: {token}"))
    } else {
        Ok(result)
    }
}

fn tokenize(expression: &str) -> Result<Vec<String>, String> {
    let mut tokens = Vec::new();
    let mut current = String::new();

    for c in expression.chars() {
        if c.is_whitespace() {
            continue;
        }

        if c.is_ascii_digit() || c == '.' {
            current.push(c);
            continue;
        }

        if !current.is_empty() {
            tokens.push(current.clone());
            current.clear();
        }

        if "+-*/()".contains(c) {
            tokens.push(c.to_string());
        } else {
            return Err(format!("Invalid character: {c}"));
        }
    }

    if !current.is_empty() {
        tokens.push(current);
    }

    Ok(tokens)
}

fn parse_expression<I>(tokens: &mut Peekable<I>) -> Result<f64, String>
where
    I: Iterator<Item = String>,
{
    let mut result = parse_term(tokens)?;

    while let Some(operator) = tokens.peek() {
        if operator == "+" || operator == "-" {
            let operator = tokens.next().expect("peeked token should exist");
            let rhs = parse_term(tokens)?;
            if operator == "+" {
                result += rhs;
            } else {
                result -= rhs;
            }
        } else {
            break;
        }
    }

    Ok(result)
}

fn parse_term<I>(tokens: &mut Peekable<I>) -> Result<f64, String>
where
    I: Iterator<Item = String>,
{
    let mut result = parse_factor(tokens)?;

    while let Some(operator) = tokens.peek() {
        if operator == "*" || operator == "/" {
            let operator = tokens.next().expect("peeked token should exist");
            let rhs = parse_factor(tokens)?;
            if operator == "*" {
                result *= rhs;
            } else if rhs == 0.0 {
                return Err("Division by zero".to_string());
            } else {
                result /= rhs;
            }
        } else {
            break;
        }
    }

    Ok(result)
}

fn parse_factor<I>(tokens: &mut Peekable<I>) -> Result<f64, String>
where
    I: Iterator<Item = String>,
{
    match tokens.next() {
        Some(token) if token == "(" => {
            let result = parse_expression(tokens)?;
            match tokens.next().as_deref() {
                Some(")") => Ok(result),
                _ => Err("Mismatched parentheses".to_string()),
            }
        }
        Some(token) => token
            .parse::<f64>()
            .map_err(|_| format!("Unexpected token: {token}")),
        None => Err("Unexpected end of expression".to_string()),
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Query {
    CustomerPolicy,
    ShippingRates,
    ProductInventory,
}

#[derive(Deserialize)]
struct DatabaseLookupArgs {
    query: Query,
}

#[derive(Debug, thiserror::Error)]
#[error("database lookup error")]
struct DatabaseLookupError;

struct DatabaseLookup {
    call_count: Arc<AtomicUsize>,
}

impl DatabaseLookup {
    fn new(call_count: Arc<AtomicUsize>) -> Self {
        Self { call_count }
    }
}

impl Tool for DatabaseLookup {
    const NAME: &'static str = "database_lookup";
    type Error = DatabaseLookupError;
    type Args = DatabaseLookupArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "database_lookup".to_string(),
            description: "Look up customer_policy, shipping_rates, or product_inventory."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "One of customer_policy, shipping_rates, or product_inventory"
                    }
                },
                "required": ["query"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.call_count.fetch_add(1, Ordering::SeqCst);

        let value = match args.query {
            Query::CustomerPolicy => {
                "Customers can return items within 30 days with a receipt for a full refund."
            }
            Query::ShippingRates => {
                "Standard shipping: $5.99, Express shipping: $15.99, Next-day shipping: $29.99"
            }
            Query::ProductInventory => {
                "Product A: 15 units, Product B: 8 units, Product C: Out of stock"
            }
        };

        Ok(value.to_string())
    }
}

fn collect_assistant_tool_calls(messages: &[Message]) -> Vec<(String, serde_json::Value)> {
    let mut tool_calls = Vec::new();

    for message in messages {
        if let Message::Assistant { content, .. } = message {
            for item in content.iter() {
                if let AssistantContent::ToolCall(tool_call) = item {
                    tool_calls.push((
                        tool_call.function.name.clone(),
                        tool_call.function.arguments.clone(),
                    ));
                }
            }
        }
    }

    tool_calls
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn think_tool_with_other_tools() -> Result<()> {
    let calculator_calls = Arc::new(AtomicUsize::new(0));
    let database_lookup_calls = Arc::new(AtomicUsize::new(0));

    let api_key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set");
    let client = anthropic::Client::builder()
        .api_key(&api_key)
        .build()
        .expect("client should build");

    let agent = client
        .agent(anthropic::completion::CLAUDE_SONNET_4_6)
        .name("Customer Service Agent")
        .preamble(
            "You are a customer service agent for an online store.
            You have access to several tools:

            1. The 'think' tool allows you to reason through complex problems step by step.
               Use it when you need to analyze information or plan your response.

            2. The 'calculator' tool can perform arithmetic using expressions.

            3. The 'database_lookup' tool can retrieve information about store policies,
               shipping rates, and product inventory.

            When handling customer inquiries, use the 'think' tool to analyze the situation
            before responding or using other tools.",
        )
        .tool(ThinkTool)
        .tool(Calculator::new(calculator_calls.clone()))
        .tool(DatabaseLookup::new(database_lookup_calls.clone()))
        .build();

    let response = agent
        .prompt(
            "I ordered 3 units of Product A at $25 each and 2 units of Product B at $40 each. \
             I want to return 1 unit of Product A and exchange the 2 units of Product B for Product C. \
             How much will I get refunded, and is Product C in stock? \
             Also, how much would it cost to ship the exchanged items with express shipping? \
             Lastly, how much would it cost to buy Product A + 2 Product B with slow (standard) shipping?",
        )
        .max_turns(10)
        .extended_details()
        .await
        .expect("prompt should succeed");

    assert_mentions_expected_number(&response.output, 25);
    assert_contains_any_case_insensitive(
        &response.output,
        &["out of stock", "express shipping", "110.99", "$110.99"],
    );

    assert!(
        calculator_calls.load(Ordering::SeqCst) >= 1,
        "calculator should be invoked at least once"
    );
    assert!(
        database_lookup_calls.load(Ordering::SeqCst) >= 2,
        "database lookup should be invoked for both shipping and inventory"
    );

    let messages = response
        .messages
        .expect("extended details should include messages");
    let tool_calls = collect_assistant_tool_calls(&messages);

    for tool_name in ["think", "calculator", "database_lookup"] {
        assert!(
            tool_calls.iter().any(|(name, _)| name == tool_name),
            "expected a {tool_name} tool call, saw {:?}",
            tool_calls
                .iter()
                .map(|(name, _)| name.as_str())
                .collect::<Vec<_>>()
        );
    }

    let queries = tool_calls
        .iter()
        .filter(|(name, _)| name == "database_lookup")
        .filter_map(|(_, args)| args.get("query").and_then(|value| value.as_str()))
        .collect::<Vec<_>>();
    assert!(
        queries.contains(&"product_inventory"),
        "expected product_inventory lookup, saw {queries:?}"
    );
    assert!(
        queries.contains(&"shipping_rates"),
        "expected shipping_rates lookup, saw {queries:?}"
    );

    Ok(())
}
