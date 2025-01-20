use anyhow::Result;
use rig::{
    completion::{Chat, Message, Prompt, ToolDefinition},
    providers,
    tool::Tool,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Deserialize)]
struct OperationArgs {
    x: i32,
    y: i32,
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
        tracing::info!("Adding {} and {}", args.x, args.y);
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
                        "description": "The number to substract from"
                    },
                    "y": {
                        "type": "number",
                        "description": "The number to substract"
                    }
                }
            }
        }))
        .expect("Tool Definition")
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        tracing::info!("Subtracting {} from {}", args.y, args.x);
        let result = args.x - args.y;
        Ok(result)
    }
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create local client
    let local = providers::openai::Client::from_url("", "http://192.168.0.10:11434/v1");

    let span = info_span!("calculator_agent");

    // Create agent with a single context prompt and two tools
    let calculator_agent = local
        .agent("c4ai-command-r7b-12-2024-abliterated")
        .preamble("You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.")
        .tool(Adder)
        .tool(Subtract)
        .max_tokens(1024)
        .build();

    // Initialize chat history
    let mut chat_history = Vec::new();
    println!("Calculator Agent: Ready to help with calculations! (Type 'quit' to exit)");

    loop {
        print!("\nYou: ");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.to_lowercase() == "quit" {
            break;
        }

        // Add user message to history
        chat_history.push(Message {
            role: "user".into(),
            content: input.into(),
        });

        // Get response from agent
        let response = calculator_agent
            .chat(input, chat_history.clone())
            .instrument(span.clone())
            .await?;

        // Add assistant's response to history
        chat_history.push(Message {
            role: "assistant".into(),
            content: response.clone(),
        });

        println!("Calculator Agent: {}", response);
    }

    println!("\nGoodbye!");
    Ok(())
}
