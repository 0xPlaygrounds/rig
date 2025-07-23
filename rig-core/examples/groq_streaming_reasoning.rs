use rig::prelude::*;
use rig::streaming::{StreamingPrompt, stream_to_stdout};
use rig::{
    completion::Prompt,
    providers::{self, groq::DEEPSEEK_R1_DISTILL_LLAMA_70B},
};
use std::env;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create OpenAI client
    let client =
        providers::groq::Client::new(&env::var("GROQ_API_KEY").expect("GROQ_API_KEY not set"));

    let json = serde_json::json!({
        "reasoning_format": "parsed"
    });
    // Create agent with a single context prompt
    let comedian_agent = client
        .agent(DEEPSEEK_R1_DISTILL_LLAMA_70B)
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .additional_params(json)
        .build();

    // Prompt the agent and print the response
    let mut stream = comedian_agent.stream_prompt("Entertain me!").await?;
    stream_to_stdout(&comedian_agent, &mut stream).await?;
    // println!("{response}");
    Ok(())
}
