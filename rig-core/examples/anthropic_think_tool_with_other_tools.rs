use anyhow::Result;
use rig::completion::Prompt;
use rig::message::Message;
use rig::prelude::*;
use rig::think_tool::ThinkTool;
use rig::{completion::ToolDefinition, providers, tool::Tool};
use serde::{Deserialize, Serialize};
use serde_json::json;

// Define a simple calculator tool for demonstration
#[derive(Deserialize)]
struct CalculatorArgs {
    expression: String,
}

#[derive(Debug, thiserror::Error)]
#[error("Calculator error: {0}")]
struct CalculatorError(String);

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
            description: "Evaluate mathematical expressions with basic operators (+, -, *, /) and parentheses. \
                          Examples of valid expressions: '2 + 2', '5 * (10 - 3)', '25 + (2 * 40)'. \
                          Does not support advanced functions like sin, cos, or logarithms.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate (e.g., '2 + 2', '5 * (10 - 3)', etc.)"
                    }
                },
                "required": ["expression"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let expr = args.expression;

        // Implement a simple parser and evaluator for mathematical expressions
        fn evaluate(expression: &str) -> Result<f64, String> {
            let tokens = tokenize(expression)?;
            let mut iter = tokens.into_iter().peekable();
            parse_expression(&mut iter)
        }

        fn tokenize(expression: &str) -> Result<Vec<String>, String> {
            let mut tokens = Vec::new();
            let mut num = String::new();

            for c in expression.chars() {
                if c.is_whitespace() {
                    continue;
                } else if c.is_ascii_digit() || c == '.' {
                    num.push(c);
                } else {
                    if !num.is_empty() {
                        tokens.push(num.clone());
                        num.clear();
                    }
                    if "+-*/()".contains(c) {
                        tokens.push(c.to_string());
                    } else {
                        return Err(format!("Invalid character: {}", c));
                    }
                }
            }

            if !num.is_empty() {
                tokens.push(num);
            }

            Ok(tokens)
        }

        fn parse_expression<I>(tokens: &mut std::iter::Peekable<I>) -> Result<f64, String>
        where
            I: Iterator<Item = String>,
        {
            let mut result = parse_term(tokens)?;

            while let Some(op) = tokens.peek() {
                if op == "+" || op == "-" {
                    let op = tokens.next().unwrap();
                    let rhs = parse_term(tokens)?;
                    if op == "+" {
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

        fn parse_term<I>(tokens: &mut std::iter::Peekable<I>) -> Result<f64, String>
        where
            I: Iterator<Item = String>,
        {
            let mut result = parse_factor(tokens)?;

            while let Some(op) = tokens.peek() {
                if op == "*" || op == "/" {
                    let op = tokens.next().unwrap();
                    let rhs = parse_factor(tokens)?;
                    if op == "*" {
                        result *= rhs;
                    } else {
                        if rhs == 0.0 {
                            return Err("Division by zero".to_string());
                        }
                        result /= rhs;
                    }
                } else {
                    break;
                }
            }

            Ok(result)
        }

        fn parse_factor<I>(tokens: &mut std::iter::Peekable<I>) -> Result<f64, String>
        where
            I: Iterator<Item = String>,
        {
            if let Some(token) = tokens.next() {
                if token == "(" {
                    let result = parse_expression(tokens)?;
                    if tokens.next() != Some(")".to_string()) {
                        return Err("Mismatched parentheses".to_string());
                    }
                    Ok(result)
                } else if let Ok(num) = token.parse::<f64>() {
                    Ok(num)
                } else {
                    Err(format!("Unexpected token: {}", token))
                }
            } else {
                Err("Unexpected end of expression".to_string())
            }
        }

        match evaluate(&expr) {
            Ok(result) => Ok(result),
            Err(err) => Err(CalculatorError(err)),
        }
    }
}

// Define a database lookup tool for demonstration
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
#[error("Database lookup error: {0}")]
struct DatabaseLookupError(String);

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
            description: "Look up information in a database. Only can use `customer_policy`,
            `shipping_rates` and `product_inventory` as valid queries."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to look up in the database"
                    }
                },
                "required": ["query"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        // This is a mock database that returns predefined responses for specific queries
        match args.query {
            Query::CustomerPolicy => Ok(
                "Customers can return items within 30 days with a receipt for a full refund."
                    .to_string(),
            ),
            Query::ShippingRates => Ok(
                "Standard shipping: $5.99, Express shipping: $15.99, Next-day shipping: $29.99"
                    .to_string(),
            ),
            Query::ProductInventory => {
                Ok("Product A: 15 units, Product B: 8 units, Product C: Out of stock".to_string())
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt().init();

    let api_key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY not set");
    let client = providers::anthropic::ClientBuilder::new(&api_key)
        .anthropic_beta("token-efficient-tools-2025-02-19")
        .build()?;

    // Create agent with the Think tool and other tools
    let agent = client
        .agent(providers::anthropic::CLAUDE_3_7_SONNET)
        .name("Customer Service Agent")
        .preamble(
            "You are a customer service agent for an online store.
            You have access to several tools:

            1. The 'think' tool allows you to reason through complex problems step by step.
               Use it when you need to analyze information or plan your response.

            2. The 'calculator' tool can perform basic math operations.

            3. The 'database_lookup' tool can retrieve information about store policies,
               shipping rates, and product inventory.

            When handling customer inquiries, use the 'think' tool to analyze the situation
            before responding or using other tools. This will help you provide accurate
            and helpful responses.

            IMPORTANT: Remember you have `parallel_tool_calling` enabled which means you can call
             multiple tools at once.",
        )
        .tool(ThinkTool)
        .tool(Calculator)
        .tool(DatabaseLookup)
        .build();

    println!("Customer service agent with Think tool");

    // Example prompt that would benefit from the Think tool in combination with other tools
    let prompt = "I ordered 3 units of Product A at $25 each and 2 units of Product B at $40 each. \
                 I want to return 1 unit of Product A and exchange the 2 units of Product B for Product C. \
                 How much will I get refunded, and is Product C in stock? \
                 Also, how much would it cost to ship the exchanged items with express shipping?.
                 Lastly, how much would it cost to buy product A + 2 product B with slow shipping?
                 ";

    let mut chat_history: Vec<Message> = Vec::new();
    let resp = agent
        .prompt(prompt)
        .with_history(&mut chat_history)
        .multi_turn(10)
        .await?;

    println!("Chat history:");
    for entry in &chat_history {
        println!("{}\n", serde_json::to_string_pretty(entry).unwrap());
    }

    println!("\n\nResponse: {}", resp);

    Ok(())
}
