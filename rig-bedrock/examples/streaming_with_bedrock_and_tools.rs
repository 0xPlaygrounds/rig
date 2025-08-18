use rig::agent::stream_to_stdout;
use rig::client::{CompletionClient, ProviderClient};
use rig::streaming::StreamingPrompt;
use rig_bedrock::{client::Client, completion::AMAZON_NOVA_LITE};
mod common;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt().init();
    // Create agent with a single context prompt and two tools
    let agent = Client::from_env()
        .agent(AMAZON_NOVA_LITE)
        .preamble(
            "You are a calculator here to help the user perform arithmetic
            operations. Use the tools provided to answer the user's question.
            make your answer long, so we can test the streaming functionality,
            like 20 words",
        )
        .max_tokens(1024)
        .tool(common::Adder)
        .build();

    println!("Calculate 2 + 5");
    let mut stream = agent.stream_prompt("Calculate 2 + 5").await;
    let _ = stream_to_stdout(&mut stream).await?;
    Ok(())
}
