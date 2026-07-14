use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::completion::Prompt;
use rig_core::providers;
use rig_derive::rig_tool;

/// A tool that performs string operations
#[rig_tool]
fn string_processor(
    /// The input text to process
    text: String,
    /// The operation to perform (uppercase, lowercase, reverse)
    operation: String,
) -> Result<String, rig_core::tool::ToolExecutionError> {
    let result = match operation.as_str() {
        "uppercase" => text.to_uppercase(),
        "lowercase" => text.to_lowercase(),
        "reverse" => text.chars().rev().collect(),
        _ => {
            return Err(rig_core::tool::ToolExecutionError::other(format!(
                "Unknown operation: {operation}"
            )));
        }
    };

    Ok(result)
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt().pretty().init();

    let string_agent = providers::openai::Client::from_env()?
        .agent(providers::openai::GPT_4O)
        .preamble("You are an agent with tools access, always use the tools")
        .max_tokens(1024)
        .tool(StringProcessor)
        .build();

    println!("Tool definition:");
    println!(
        "STRINGPROCESSOR: {}",
        serde_json::to_string_pretty(&rig_core::tool::tool_definition(&StringProcessor))?
    );

    for prompt in [
        "What tools do you have?",
        "Convert 'hello world' to uppercase",
        "Convert 'HELLO WORLD' to lowercase",
        "Reverse the string 'hello world'",
        "Convert 'hello world' to uppercase and repeat it 3 times",
        "Perform an invalid operation on 'hello world'",
    ] {
        println!("User: {prompt}");
        println!("Agent: {}", string_agent.prompt(prompt).await?);
    }

    Ok(())
}
