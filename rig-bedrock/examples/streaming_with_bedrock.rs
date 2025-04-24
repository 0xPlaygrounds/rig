use rig::streaming::{stream_to_stdout, StreamingPrompt};
use rig_bedrock::{client::ClientBuilder, completion::AMAZON_NOVA_LITE};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create streaming agent with a single context prompt
    let agent = ClientBuilder::new()
        .build()
        .await
        .agent(AMAZON_NOVA_LITE)
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
