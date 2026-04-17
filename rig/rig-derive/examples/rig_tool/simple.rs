use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers;
use rig_derive::rig_tool;

/// Add two numbers
#[rig_tool]
fn add(
    /// First number
    a: i32,
    /// Second number
    b: i32,
) -> Result<i32, rig::tool::ToolError> {
    Ok(a + b)
}

/// Subtract two numbers
#[rig_tool]
fn subtract(
    /// First number
    a: i32,
    /// Second number
    b: i32,
) -> Result<i32, rig::tool::ToolError> {
    Ok(a - b)
}

/// Multiply two numbers
#[rig_tool]
fn multiply(
    /// First number
    a: i32,
    /// Second number
    b: i32,
) -> Result<i32, rig::tool::ToolError> {
    Ok(a * b)
}

/// Divide two numbers
#[rig_tool]
fn divide(
    /// Dividend
    a: i32,
    /// Divisor
    b: i32,
) -> Result<i32, rig::tool::ToolError> {
    if b == 0 {
        Err(rig::tool::ToolError::ToolCallError(
            "Division by zero".into(),
        ))
    } else {
        Ok(a / b)
    }
}

/// Answer the secret question
#[rig_tool]
fn answer_secret_question() -> Result<(bool, bool, bool, bool, bool), rig::tool::ToolError> {
    Ok((false, false, true, false, false))
}

/// Count the number of R characters in a string
#[rig_tool]
fn how_many_rs(
    /// The string to search
    s: String,
) -> Result<usize, rig::tool::ToolError> {
    Ok(s.chars()
        .filter(|c| *c == 'r' || *c == 'R')
        .collect::<Vec<_>>()
        .len())
}

/// Sum a list of numbers
#[rig_tool]
fn sum_numbers(
    /// Numbers to sum
    numbers: Vec<i64>,
) -> Result<i64, rig::tool::ToolError> {
    Ok(numbers.iter().sum())
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().pretty().init();

    let calculator_agent = providers::openai::Client::from_env()
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
        println!("Agent: {}", calculator_agent.prompt(prompt).await.unwrap());
    }
}
