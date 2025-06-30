use rig::prelude::*;
use rig::providers::xai;
use rig::streaming::{StreamingPrompt, stream_to_stdout};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create streaming agent with a single context prompt
    let agent = xai::Client::from_env()
        .agent(xai::GROK_3_MINI)
        .preamble("Be precise and concise.")
        .temperature(0.5)
        .build();

    // Stream the response and print chunks as they arrive
    let mut stream = agent
        .stream_prompt("When and where and what type is the next solar eclipse?")
        .await?;

    stream_to_stdout(&agent, &mut stream).await?;

    Ok(())
}
