use rig::prelude::*;
use rig::{
    agent::Agent,
    completion::{CompletionError, CompletionModel, Prompt, PromptError, ToolDefinition},
    extractor::Extractor,
    message::Message,
    providers::anthropic,
    tool::Tool,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

const CHAIN_OF_THOUGHT_PROMPT: &str = "
You are an assistant that extracts reasoning steps from a given prompt.
Do not return text, only return a tool call.
";

#[derive(Deserialize, Serialize, Debug, Clone, JsonSchema)]
struct ChainOfThoughtSteps {
    steps: Vec<String>,
}

struct ReasoningAgent<M: CompletionModel> {
    chain_of_thought_extractor: Extractor<M, ChainOfThoughtSteps>,
    executor: Agent<M>,
}

impl<M: CompletionModel> Prompt for ReasoningAgent<M> {
    #[allow(refining_impl_trait)]
    async fn prompt(&self, prompt: impl Into<Message> + Send) -> Result<String, PromptError> {
        let prompt: Message = prompt.into();
        let mut chat_history = vec![prompt.clone()];
        let extracted = self
            .chain_of_thought_extractor
            .extract(prompt)
            .await
            .map_err(|e| {
                tracing::error!("Extraction error: {:?}", e);
                CompletionError::ProviderError("".into())
            })?;
        if extracted.steps.is_empty() {
            return Ok("No reasoning steps provided.".into());
        }
        let mut reasoning_prompt = String::new();
        for (i, step) in extracted.steps.iter().enumerate() {
            reasoning_prompt.push_str(&format!("Step {}: {}\n", i + 1, step));
        }
        let response = self
            .executor
            .prompt(reasoning_prompt.as_str())
            .with_history(&mut chat_history)
            .multi_turn(20)
            .await?;
        tracing::info!(
            "full chat history generated: {}",
            serde_json::to_string_pretty(&chat_history).unwrap()
        );
        Ok(response)
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(false)
        .init();

    // Create Anthropic client
    let anthropic_client = anthropic::Client::from_env();
    let agent = ReasoningAgent {
        chain_of_thought_extractor: anthropic_client
            .extractor(anthropic::completion::CLAUDE_3_5_SONNET)
            .preamble(CHAIN_OF_THOUGHT_PROMPT)
            .build(),

        executor: anthropic_client
            .agent(anthropic::completion::CLAUDE_3_5_SONNET)
            .preamble(
                "You are an assistant here to help the user select which tool is most appropriate to perform arithmetic operations.
                Follow these instructions closely.
                1. Consider the user's request carefully and identify the core elements of the request.
                2. Select which tool among those made available to you is appropriate given the context.
                3. This is very important: never perform the operation yourself.
                4. When you think you've finished calling tools for the operation, present the final result from the series of tool calls you made.
                "
            )
            .tool(Add)
            .tool(Subtract)
            .tool(Multiply)
            .tool(Divide)
            .build(),
    };

    // Prompt the agent and print the response
    let result = agent
        .prompt("Calculate ((15 + 25) * (100 - 50)) / (200 / (10 + 10))")
        .await?;

    println!("\n\nReasoning Agent Chat History: {result}");

    Ok(())
}

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
        serde_json::from_value(json!({
            "name": "add",
            "description": "Add x and y together",
            "parameters": {
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
            }
        }))
        .expect("Tool Definition")
    }
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let result = args.x + args.y;
        Ok(result)
    }
}
#[derive(Deserialize, Serialize)]
struct Subtract;
impl Tool for Subtract {
    const NAME: &'static str = "subtract";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;
    async fn definition(&self, _prompt: String) -> ToolDefinition {
        serde_json::from_value(json!({
            "name": "subtract",
            "description": "Subtract y from x (i.e.: x - y)",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "number",
                        "description": "The number to subtract from"
                    },
                    "y": {
                        "type": "number",
                        "description": "The number to subtract"
                    }
                }
            }
        }))
        .expect("Tool Definition")
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let result = args.x - args.y;
        Ok(result)
    }
}

struct Multiply;

impl Tool for Multiply {
    const NAME: &'static str = "multiply";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        serde_json::from_value(json!({
            "name": "multiply",
            "description": "Compute the product of x and y (i.e.: x * y)",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "number",
                        "description": "The first factor in the product"
                    },
                    "y": {
                        "type": "number",
                        "description": "The second factor in the product"
                    }
                }
            }
        }))
        .expect("Tool Definition")
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let result = args.x * args.y;
        Ok(result)
    }
}

struct Divide;

impl Tool for Divide {
    const NAME: &'static str = "divide";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        serde_json::from_value(json!({
            "name": "divide",
            "description": "Compute the Quotient of x and y (i.e.: x / y). Useful for ratios.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "number",
                        "description": "The Dividend of the division. The number being divided"
                    },
                    "y": {
                        "type": "number",
                        "description": "The Divisor of the division. The number by which the dividend is being divided"
                    }
                }
            }
        }))
        .expect("Tool Definition")
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let result = args.x / args.y;
        Ok(result)
    }
}
