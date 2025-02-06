use std::env;

use rig::{
    completion::Prompt,
    providers::{self, groq::DEEPSEEK_R1_DISTILL_LLAMA_70B},
};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create OpenAI client
    let client =
        providers::groq::Client::new(&env::var("GROQ_API_KEY").expect("GROQ_API_KEY not set"));

    // Create agent with a single context prompt
    let comedian_agent = client
        .agent(DEEPSEEK_R1_DISTILL_LLAMA_70B)
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .build();

    // Prompt the agent and print the response
    let response = comedian_agent.prompt("Entertain me!").await?;
    println!("{}", response);

    Ok(())
}
