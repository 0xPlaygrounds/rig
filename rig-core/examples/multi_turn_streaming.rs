use futures::{Stream, StreamExt};
use rig::{
    OneOrMany,
    agent::Agent,
    client::{CompletionClient, ProviderClient},
    completion::{self, CompletionError, CompletionModel, PromptError, ToolDefinition},
    message::{AssistantContent, Message, Text, ToolResultContent, UserContent},
    providers::anthropic,
    streaming::StreamingCompletion,
    tool::{Tool, ToolSetError},
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::pin::Pin;
use thiserror::Error;

#[derive(Debug, Error)]
enum StreamingError {
    #[error("CompletionError: {0}")]
    Completion(#[from] CompletionError),
    #[error("PromptError: {0}")]
    Prompt(#[from] PromptError),
    #[error("ToolSetError: {0}")]
    Tool(#[from] ToolSetError),
}

type StreamingResult = Pin<Box<dyn Stream<Item = Result<Text, StreamingError>> + Send>>;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // tracing_subscriber::registry()
    //     .with(
    //         tracing_subscriber::EnvFilter::try_from_default_env()
    //             .unwrap_or_else(|_| "stdout=info".into()),
    //     )
    //     .with(tracing_subscriber::fmt::layer())
    //     .init();

    // Create Anthropic client
    let anthropic_client = anthropic::Client::from_env();

    // Create agent with a single context prompt and a calculator tools
    let calculator_agent = anthropic_client
        .agent(anthropic::CLAUDE_3_5_SONNET)
        .preamble(
            "You are an assistant here to help the user select which tool is most appropriate to perform arithmetic operations.
            Follow these instructions closely.
            1. Consider the user's request carefully and identify the core elements of the request.
            2. Select which tool among those made available to you is appropriate given the context.
            3. This is very important: never perform the operation yourself.
            "
        )
        .tool(Add)
        .tool(Subtract)
        .tool(Multiply)
        .tool(Divide)
        .build();

    // Prompt the agent and get the stream
    let mut stream = multi_turn_prompt(
        calculator_agent,
        "Calculate 2 * (3 + 5) / 9  = ?. Describe the result to me.",
        Vec::new(),
    )
    .await;

    custom_stream_to_stdout(&mut stream).await?;

    Ok(())
}

async fn multi_turn_prompt<M>(
    agent: Agent<M>,
    prompt: impl Into<Message> + Send,
    mut chat_history: Vec<completion::Message>,
) -> StreamingResult
where
    M: CompletionModel + 'static,
    <M as CompletionModel>::StreamingResponse: std::marker::Send,
{
    let prompt: Message = prompt.into();

    (Box::pin(async_stream::stream! {
        let mut current_prompt = prompt;
        let mut did_call_tool = false;

        'outer: loop {
            let mut stream = agent
                .stream_completion(current_prompt.clone(), chat_history.clone())
                .await?
                .stream()
                .await?;

            chat_history.push(current_prompt.clone());

            let mut tool_calls = vec![];
            let mut tool_results = vec![];

            while let Some(content) = stream.next().await {
                match content {
                    Ok(AssistantContent::Text(text)) => {
                        yield Ok(Text { text: text.text });
                        did_call_tool = false;
                    },
                    Ok(AssistantContent::ToolCall(tool_call)) => {
                        let tool_result =
                            agent.tools.call(&tool_call.function.name, tool_call.function.arguments.to_string()).await?;

                        let tool_call_msg = AssistantContent::ToolCall(tool_call.clone());

                        tool_calls.push(tool_call_msg);
                        tool_results.push((tool_call.id, tool_result));

                        did_call_tool = true;
                        // break;
                    },
                    Err(e) => {
                        yield Err(e.into());
                        break 'outer;
                    }
                }
            }

            // Add (parallel) tool calls to chat history
            if !tool_calls.is_empty() {
                chat_history.push(Message::Assistant {
                    id: None,
                    content: OneOrMany::many(tool_calls).expect("Impossible EmptyListError"),
                });
            }

            // Add tool results to chat history
            for (id, tool_result) in tool_results {
                chat_history.push(Message::User {
                    content: OneOrMany::one(UserContent::tool_result(
                        id,
                        OneOrMany::one(ToolResultContent::text(tool_result)),
                    )),
                });
            }

            // Set the current prompt to the last message in the chat history
            current_prompt = match chat_history.pop() {
                Some(prompt) => prompt,
                None => unreachable!("Chat history should never be empty at this point"),
            };

            if !did_call_tool {
                break;
            }
        }

    })) as _
}

/// helper function to stream a completion request to stdout
async fn custom_stream_to_stdout(stream: &mut StreamingResult) -> Result<(), std::io::Error> {
    print!("Response: ");
    while let Some(content) = stream.next().await {
        match content {
            Ok(Text { text }) => {
                print!("{text}");
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            Err(err) => {
                eprintln!("Error: {err}");
            }
        }
    }
    println!(); // New line after streaming completes

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
