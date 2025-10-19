use futures::{Stream, StreamExt};
use rig::providers::gemini;
use rig::tool::ToolError;
use rig::{
    OneOrMany,
    agent::Agent,
    client::{CompletionClient, ProviderClient},
    completion::{self, CompletionError, CompletionModel, PromptError, ToolDefinition},
    message::{AssistantContent, Message, Text, ToolResultContent, UserContent},
    streaming::{StreamedAssistantContent, StreamingCompletion},
    tool::{Tool, ToolSetError},
};
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};

use std::pin::Pin;
use thiserror::Error;

#[derive(Debug, Error)]
enum StreamingError {
    #[error("CompletionError: {0}")]
    Completion(#[from] CompletionError),
    #[error("PromptError: {0}")]
    Prompt(#[from] Box<PromptError>),
    #[error("ToolSetError: {0}")]
    Tool(#[from] ToolSetError),
}

type StreamingResult = Pin<Box<dyn Stream<Item = Result<Text, StreamingError>> + Send>>;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // tracing_subscriber::registry()
    //     .with(
    //         tracing_subscriber::EnvFilter::try_from_default_env()
    //             .unwrap_or_else(|_| "info".into()),
    //     )
    //     .with(tracing_subscriber::fmt::layer())
    //     .init();

    // Create gemini client
    let llm_client = gemini::Client::from_env();

    // Create agent with a single context prompt and a calculator tools
    let calculator_agent = llm_client
        .agent("gemini-2.5-flash")
        .preamble("You are an calculator. You must use tools to get the user result")
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
                    Ok(StreamedAssistantContent::Text(text)) => {
                        yield Ok(Text { text: text.text });
                        did_call_tool = false;
                    },
                    Ok(StreamedAssistantContent::ToolCall(tool_call)) => {
                        let tool_result =
                            agent.tool_server_handle.call_tool(&tool_call.function.name, &tool_call.function.arguments.to_string()).await
                            .map_err(|x| StreamingError::Tool(ToolSetError::ToolCallError(ToolError::ToolCallError(x.into()))))?;

                        let tool_call_msg = AssistantContent::ToolCall(tool_call.clone());

                        tool_calls.push(tool_call_msg);
                        tool_results.push((tool_call.id, tool_call.call_id, tool_result));

                        did_call_tool = true;
                        // break;
                    },
                    Ok(StreamedAssistantContent::Reasoning(rig::message::Reasoning { reasoning, .. })) => {
                        if !reasoning.is_empty() {
                            yield Ok(Text { text: reasoning.first().unwrap().to_owned() });
                        }
                        did_call_tool = false;
                    },
                    Ok(_) => {
                        // do nothing here as we don't need to accumulate token usage
                    }
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
            for (id, call_id, tool_result) in tool_results {
                if let Some(call_id) = call_id {
                    chat_history.push(Message::User {
                        content: OneOrMany::one(UserContent::tool_result_with_call_id(
                            id,
                            call_id,
                            OneOrMany::one(ToolResultContent::text(tool_result)),
                        )),
                    });
                } else {
                    chat_history.push(Message::User {
                        content: OneOrMany::one(UserContent::tool_result(
                            id,
                            OneOrMany::one(ToolResultContent::text(tool_result)),
                        )),
                    });

                }

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
                eprintln!("Error: {err:#?}");
            }
        }
    }
    println!(); // New line after streaming completes

    Ok(())
}

#[derive(Deserialize, JsonSchema)]
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
            parameters: serde_json::to_value(schema_for!(OperationArgs))
                .expect("converting JSON schema to JSON value should never fail"),
        }
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
        ToolDefinition {
            name: "subtract".to_string(),
            description: "Subtract y from x (i.e.: x - y)".to_string(),
            parameters: serde_json::to_value(schema_for!(OperationArgs))
                .expect("converting JSON schema to JSON value should never fail"),
        }
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
        ToolDefinition {
            name: "multiply".to_string(),
            description: "Compute the product of x and y (i.e.: x * y)".to_string(),
            parameters: serde_json::to_value(schema_for!(OperationArgs))
                .expect("converting JSON schema to JSON value should never fail"),
        }
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
        ToolDefinition {
            name: "divide".to_string(),
            description: "Compute the Quotient of x and y (i.e.: x / y). Useful for ratios."
                .to_string(),
            parameters: serde_json::to_value(schema_for!(OperationArgs))
                .expect("converting JSON schema to JSON value should never fail"),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let result = args.x / args.y;
        Ok(result)
    }
}
