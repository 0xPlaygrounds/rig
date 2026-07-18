use rig_agent::client_ext::AgentClientExt;
use rig_agent::completion::Prompt;
use rig_core::client::ProviderClient;
use rig_core::providers;
use rig_derive::rig_tool;

/// Add two numbers
#[rig_tool]
fn add(
    /// First number
    a: i32,
    /// Second number
    b: i32,
) -> Result<i32, rig_core::tool::ToolExecutionError> {
    Ok(a + b)
}

/// Subtract two numbers
#[rig_tool]
fn subtract(
    /// First number
    a: i32,
    /// Second number
    b: i32,
) -> Result<i32, rig_core::tool::ToolExecutionError> {
    Ok(a - b)
}

/// Multiply two numbers
#[rig_tool]
fn multiply(
    /// First number
    a: i32,
    /// Second number
    b: i32,
) -> Result<i32, rig_core::tool::ToolExecutionError> {
    Ok(a * b)
}

/// Divide two numbers
#[rig_tool]
fn divide(
    /// Dividend
    a: i32,
    /// Divisor
    b: i32,
) -> Result<i32, rig_core::tool::ToolExecutionError> {
    if b == 0 {
        Err(rig_core::tool::ToolExecutionError::other(
            "Division by zero",
        ))
    } else {
        Ok(a / b)
    }
}

/// Answer the secret question
#[rig_tool]
fn answer_secret_question()
-> Result<(bool, bool, bool, bool, bool), rig_core::tool::ToolExecutionError> {
    Ok((false, false, true, false, false))
}

/// Count the number of R characters in a string
#[rig_tool]
fn how_many_rs(
    /// The string to search
    s: String,
) -> Result<usize, rig_core::tool::ToolExecutionError> {
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
) -> Result<i64, rig_core::tool::ToolExecutionError> {
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
