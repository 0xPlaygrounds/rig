use rig::prelude::*;
use rig::{
    providers::gemini::{self, completion::GEMINI_1_5_FLASH},
    streaming::{StreamingPrompt, stream_to_stdout},
};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create streaming agent with a single context prompt
    let agent = gemini::Client::from_env()
        .agent(GEMINI_1_5_FLASH)
        .preamble("Be precise and concise.")
        .temperature(0.5)
        .build();

    // Stream the response and print chunks as they arrive
    let mut stream = agent
        .stream_prompt("When and where and what type is the next solar eclipse?")
        .await?;

    stream_to_stdout(&agent, &mut stream).await?;

    if let Some(response) = stream.response {
        println!(
            "Usage: {:?} tokens",
            response.usage_metadata.total_token_count
        );
    };

    println!("Message: {:?}", stream.choice);

    Ok(())
}
