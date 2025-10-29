use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::providers;
use rig::tool::Tool;
use rig_derive::rig_tool;
use std::time::Duration;

// Example demonstrating async tool usage
#[rig_tool(
    description = "A tool that simulates an async operation",
    params(
        input = "Input value to process",
        delay_ms = "Delay in milliseconds before returning result"
    ),
    required(input, delay_ms)
)]
async fn async_operation(input: String, delay_ms: u64) -> Result<String, rig::tool::ToolError> {
    tokio::time::sleep(Duration::from_millis(delay_ms)).await;

    Ok(format!(
        "Processed after {}ms: {}",
        delay_ms,
        input.to_uppercase()
    ))
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().pretty().init();

    let async_agent = providers::openai::Client::from_env()
        .agent(providers::openai::GPT_4O)
        .preamble("You are an agent with tools access, always use the tools")
        .max_tokens(1024)
        .tool(AsyncOperation)
        .build();

    println!("Tool definition:");
    println!(
        "ASYNCOPERATION: {}",
        serde_json::to_string_pretty(&AsyncOperation.definition(String::default()).await).unwrap()
    );

    for prompt in [
        "What tools do you have?",
        "Process the text 'hello world' with a delay of 1000ms",
        "Process the text 'async operation' with a delay of 500ms",
        "Process the text 'concurrent calls' with a delay of 200ms",
        "Process the text 'error handling' with a delay of 'not a number'",
    ] {
        println!("User: {prompt}");
        println!("Agent: {}", async_agent.prompt(prompt).await.unwrap());
    }
}
