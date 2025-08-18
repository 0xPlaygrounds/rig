use rig::agent::stream_to_stdout;
use rig::client::{CompletionClient, ProviderClient};
use rig::streaming::StreamingPrompt;
use rig_bedrock::{client::Client, completion::AMAZON_NOVA_LITE};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create streaming agent with a single context prompt
    let agent = Client::from_env()
        .agent(AMAZON_NOVA_LITE)
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
