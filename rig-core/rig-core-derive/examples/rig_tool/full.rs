use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::providers;
use rig::tool::Tool;
use rig_derive::rig_tool;

// Example with full attributes including parameter descriptions
#[rig_tool(
    description = "A tool that performs string operations",
    params(
        text = "The input text to process",
        operation = "The operation to perform (uppercase, lowercase, reverse)",
    ),
    required(text, operation)
)]
fn string_processor(text: String, operation: String) -> Result<String, rig::tool::ToolError> {
    let result = match operation.as_str() {
        "uppercase" => text.to_uppercase(),
        "lowercase" => text.to_lowercase(),
        "reverse" => text.chars().rev().collect(),
        _ => {
            return Err(rig::tool::ToolError::ToolCallError(
                format!("Unknown operation: {operation}").into(),
            ));
        }
    };

    Ok(result)
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().pretty().init();

    let string_agent = providers::openai::Client::from_env()
        .agent(providers::openai::GPT_4O)
        .preamble("You are an agent with tools access, always use the tools")
        .max_tokens(1024)
        .tool(StringProcessor)
        .build();

    println!("Tool definition:");
    println!(
        "STRINGPROCESSOR: {}",
        serde_json::to_string_pretty(&StringProcessor.definition(String::default()).await).unwrap()
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
        println!("Agent: {}", string_agent.prompt(prompt).await.unwrap());
    }
}
