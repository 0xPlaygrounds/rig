use rig::{
    providers::anthropic::{self, CLAUDE_3_5_SONNET},
    streaming::{stream_to_stdout, StreamingPrompt},
};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create streaming agent with a single context prompt
    let agent = anthropic::Client::from_env()
        .agent(CLAUDE_3_5_SONNET)
        .preamble("Be precise and concise.")
        .temperature(0.5)
        .build();

    // Stream the response and print chunks as they arrive
    let mut stream = agent
        .stream_prompt("When and where and what type is the next solar eclipse?")
        .await?;

    stream_to_stdout(agent, &mut stream).await?;

    Ok(())
}
