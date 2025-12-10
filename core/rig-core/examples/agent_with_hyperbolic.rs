use rig::completion::Prompt;
use rig::prelude::*;
use rig::providers::hyperbolic;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create OpenAI client
    let client = hyperbolic::Client::from_env();

    // Create agent with a single context prompt
    let comedian_agent = client
        .agent(hyperbolic::DEEPSEEK_R1)
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .build();

    // Prompt the agent and print the response
    let response = comedian_agent.prompt("Entertain me!").await?;
    println!("{response}");
    Ok(())
}
