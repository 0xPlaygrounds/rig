//! Demonstrates manual tool-call handling with `Agent::completion()`.
//! Requires `OPENAI_API_KEY`.
//!
//! Unlike `agent.prompt(...)`, this example never lets Rig execute tools automatically.
//! It:
//! 1. sends a low-level completion request,
//! 2. collects one or more `ToolCall`s from the model output,
//! 3. executes them locally with a `ToolSet`,
//! 4. feeds the tool results back to the model, and
//! 5. repeats until the model returns a final text answer.

use anyhow::{Result, bail};
use rig::OneOrMany;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{Completion, ToolDefinition};
use rig::message::{AssistantContent, Message, ToolCall, ToolChoice};
use rig::providers::openai;
use rig::tool::{Tool, ToolSet};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Deserialize)]
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
            parameters: json!({
                "type": "object",
                "properties": {
                    "x": { "type": "number", "description": "The first number to add" },
                    "y": { "type": "number", "description": "The second number to add" }
                },
                "required": ["x", "y"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(args.x + args.y)
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
        ToolDefinition {
            name: "subtract".to_string(),
            description: "Subtract y from x (x - y)".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "x": { "type": "number", "description": "The number to subtract from" },
                    "y": { "type": "number", "description": "The number to subtract" }
                },
                "required": ["x", "y"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(args.x - args.y)
    }
}

fn collect_tool_calls(choice: &OneOrMany<AssistantContent>) -> Vec<ToolCall> {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::ToolCall(tool_call) => Some(tool_call.clone()),
            _ => None,
        })
        .collect()
}

fn extract_text(choice: &OneOrMany<AssistantContent>) -> String {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn tool_result_message(tool_call: &ToolCall, output: String) -> Message {
    Message::tool_result_with_call_id(tool_call.id.clone(), tool_call.call_id.clone(), output)
}

#[tokio::main]
async fn main() -> Result<()> {
    const MAX_ROUNDS: usize = 8;

    let agent = openai::Client::from_env()?
        .agent(openai::GPT_4O_MINI)
        .preamble(
            "You are a calculator. Never do arithmetic from memory. \
             Use the provided tools for every intermediate step. \
             You may emit one or multiple tool calls in a single turn. \
             Once all tool results are available, give a short final answer.",
        )
        .tool(Add)
        .tool(Subtract)
        .build();

    let local_tools = ToolSet::builder()
        .static_tool(Add)
        .static_tool(Subtract)
        .build();

    let mut history = Vec::new();
    let mut current_prompt = Message::user(
        "Calculate (20 - 5) + (8 - 3). Use tools for each intermediate step before answering.",
    );

    for round in 1..=MAX_ROUNDS {
        let mut request = agent
            .completion(current_prompt.clone(), history.clone())
            .await?;
        if round == 1 {
            // Force the first turn through the tool path so the example always demonstrates it.
            request = request.tool_choice(ToolChoice::Required);
        }

        let response = request.send().await?;
        let tool_calls = collect_tool_calls(&response.choice);

        history.push(current_prompt.clone());
        history.push(Message::Assistant {
            id: response.message_id.clone(),
            content: response.choice.clone(),
        });

        if tool_calls.is_empty() {
            let final_text = extract_text(&response.choice);
            println!("\nFinal answer:\n{final_text}");
            return Ok(());
        }

        println!(
            "\nRound {round}: model requested {} tool call(s)",
            tool_calls.len()
        );

        for tool_call in &tool_calls {
            let args = serde_json::to_string(&tool_call.function.arguments)?;
            let output = local_tools
                .call(&tool_call.function.name, args.clone())
                .await?;
            println!("  {}({args}) -> {}", tool_call.function.name, output);
            history.push(tool_result_message(tool_call, output));
        }

        current_prompt = match history.pop() {
            Some(prompt) => prompt,
            None => bail!("tool loop history unexpectedly empty"),
        };
    }

    bail!("manual tool loop exceeded {MAX_ROUNDS} rounds")
}
