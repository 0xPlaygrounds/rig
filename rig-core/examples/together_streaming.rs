use rig::prelude::*;
use rig::{
    agent::stream_to_stdout,
    providers::together::{self},
    streaming::StreamingPrompt,
};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create streaming agent with a single context prompt
    let agent = together::Client::from_env()
        .agent(together::LLAMA_3_8B_CHAT_HF)
        .preamble("Be precise and concise.")
        .temperature(0.5)
        .build();

    // Stream the response and print chunks as they arrive
    let mut stream = agent
        .stream_prompt("When and where and what type is the next solar eclipse?")
        .await;

    let _ = stream_to_stdout(&mut stream).await?;

    Ok(())
}
