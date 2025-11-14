use rig::prelude::*;
use rig::providers::groq;
use rig::{completion::Prompt, providers::groq::DEEPSEEK_R1_DISTILL_LLAMA_70B};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create OpenAI client
    let client = groq::Client::from_env();

    // Create agent with a single context prompt
    let comedian_agent = client
        .agent(DEEPSEEK_R1_DISTILL_LLAMA_70B)
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .build();

    // Prompt the agent and print the response
    let response = comedian_agent.prompt("Entertain me!").await?;
    println!("{response}");
    Ok(())
}
