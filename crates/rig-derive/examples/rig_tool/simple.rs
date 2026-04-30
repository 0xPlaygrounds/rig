use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::completion::Prompt;
use rig_core::providers;
use rig_derive::rig_tool;

// Simple example with no attributes (`required` is still needed for OpenAI's strict function tool calling)
#[rig_tool(required(a, b))]
fn add(a: i32, b: i32) -> Result<i32, rig_core::tool::ToolError> {
    Ok(a + b)
}

#[rig_tool(required(a, b))]
fn subtract(a: i32, b: i32) -> Result<i32, rig_core::tool::ToolError> {
    Ok(a - b)
}

#[rig_tool(required(a, b))]
fn multiply(a: i32, b: i32) -> Result<i32, rig_core::tool::ToolError> {
    Ok(a * b)
}

#[rig_tool(required(a, b))]
fn divide(a: i32, b: i32) -> Result<i32, rig_core::tool::ToolError> {
    if b == 0 {
        Err(rig_core::tool::ToolError::ToolCallError(
            "Division by zero".into(),
        ))
    } else {
        Ok(a / b)
    }
}

#[rig_tool]
fn answer_secret_question() -> Result<(bool, bool, bool, bool, bool), rig_core::tool::ToolError> {
    Ok((false, false, true, false, false))
}

#[rig_tool]
fn how_many_rs(s: String) -> Result<usize, rig_core::tool::ToolError> {
    Ok(s.chars()
        .filter(|c| *c == 'r' || *c == 'R')
        .collect::<Vec<_>>()
        .len())
}

#[rig_tool]
fn sum_numbers(numbers: Vec<i64>) -> Result<i64, rig_core::tool::ToolError> {
    Ok(numbers.iter().sum())
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt().pretty().init();

    let calculator_agent = providers::openai::Client::from_env()?
        .agent(providers::openai::GPT_4O)
        .preamble("You are an agent with tools access, always use the tools")
        .max_tokens(1024)
        .tool(Add)
        .tool(Subtract)
        .tool(Multiply)
        .tool(Divide)
        .tool(AnswerSecretQuestion)
        .tool(HowManyRs)
        .tool(SumNumbers)
        .build();

    for prompt in [
        "What tools do you have?",
        "Calculate 5 + 3",
        "What is 10 + 20?",
        "Add 100 and 200",
    ] {
        println!("User: {prompt}");
        println!("Agent: {}", calculator_agent.prompt(prompt).await?);
    }

    Ok(())
}
